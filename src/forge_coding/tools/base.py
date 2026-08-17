"""Shared Forge/LangChain tool adapter and direct-execution seam."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Mapping
from typing import TYPE_CHECKING, Annotated, Any, cast

from langchain.tools import ToolRuntime
from langchain_core.tools import InjectedToolArg, StructuredTool, ToolException
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, create_model

from forge_agent.context import ForgeRuntimeContext
from forge_agent.tools import AgentToolResult, ToolCancellationToken, ToolExecutor
from forge_agent.types import JSONValue

if TYPE_CHECKING:
    from forge_coding.tools.definition import ToolDefinition

_JSON_TYPE_TO_PYTHON: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict[str, Any],
    "array": list[Any],
}


class ToolInputError(ValueError):
    """Raised when a tool receives invalid structured arguments."""


class ForgeStructuredTool(StructuredTool):
    """A native StructuredTool with a Forge direct-execution seam.

    ``StructuredTool`` owns the provider-visible name, description, schema and
    coroutine.  Forge only adds the direct executor needed by slash commands
    and tests; product prompt metadata lives in :class:`ToolDefinition`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    _executor: ToolExecutor | None = PrivateAttr(default=None)
    # Compatibility-only prompt projections.  The product catalog remains
    # authoritative; these properties let old callers migrate incrementally.
    _prompt_snippet: str | None = PrivateAttr(default=None)
    _prompt_guidelines: tuple[str, ...] = PrivateAttr(default=())

    def __init__(self, *args: Any, executor: ToolExecutor | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._executor = executor

    @property
    def executor(self) -> ToolExecutor:
        """Return the direct executor owned by this native tool."""

        if self._executor is None:
            raise RuntimeError(f"Tool {self.name} has no Forge direct executor")
        return self._executor

    @property
    def prompt_snippet(self) -> str | None:
        """Deprecated compatibility view; use ``ToolDefinition`` instead."""

        return self._prompt_snippet

    @property
    def prompt_guidelines(self) -> tuple[str, ...]:
        """Deprecated compatibility view; use ``ToolDefinition`` instead."""

        return self._prompt_guidelines

    def _set_prompt_metadata(
        self,
        snippet: str | None,
        guidelines: tuple[str, ...],
    ) -> None:
        """Attach a compatibility projection for legacy direct callers."""

        self._prompt_snippet = snippet
        self._prompt_guidelines = guidelines

    async def execute(
        self,
        arguments: Mapping[str, JSONValue],
        signal: ToolCancellationToken | None = None,
    ) -> AgentToolResult:
        """Forge direct-execution seam for slash commands and tests.

        The production agent loop executes this tool through
        ``ainvoke``/``ToolRuntime``; this method runs the underlying
        ``ToolExecutor`` directly without LangChain's injected runtime.
        """

        result = await self.executor(arguments, signal=signal)
        if result.tool_call_id:
            return result
        return result.model_copy(update={"tool_call_id": ""})


def _args_schema_from_json_schema(
    name: str,
    input_schema: Mapping[str, JSONValue],
) -> type[BaseModel]:
    """Build a compatibility Pydantic model from a legacy JSON schema.

    Built-in tools pass explicit Pydantic models.  This narrow adapter remains
    only for the old ``ToolDefinition(name=..., input_schema=...)`` factory
    surface so existing callers can migrate without a flag day.
    """

    return create_model(
        f"{name.title()}ToolInput",
        __config__=ConfigDict(arbitrary_types_allowed=True),
        **_args_fields_from_json_schema(input_schema),
    )


def _args_schema_for_tool(definition: ToolDefinition) -> type[BaseModel]:
    """Compatibility helper for callers of the old definition adapter."""

    return _args_schema_from_json_schema(definition.name, definition.input_schema)


def _runtime_args_schema(args_schema: type[BaseModel]) -> type[BaseModel]:
    """Add LangChain's hidden runtime injection field to an input model.

    ``ToolRuntime`` is delivered by the graph, never by the model.  The
    ``InjectedToolArg`` marker keeps it out of ``tool_call_schema`` while the
    internal field lets Pydantic validate trusted graph-injected arguments
    without weakening the model-facing input model's ``extra`` policy.
    """

    if "runtime" in args_schema.model_fields:
        return args_schema
    return create_model(
        f"{args_schema.__name__}WithRuntime",
        __base__=args_schema,
        runtime=(
            Annotated[Any, InjectedToolArg()],
            Field(default=None, exclude=True),
        ),
    )


def _create_native_tool(
    *,
    name: str,
    description: str,
    args_schema: type[BaseModel],
    executor: ToolExecutor,
) -> ForgeStructuredTool:
    """Create a native Forge tool from an explicit input model and executor."""

    runtime_args_schema = _runtime_args_schema(args_schema)

    async def invoke(
        *,
        runtime: ToolRuntime | None = None,
        **arguments: Any,
    ) -> tuple[str, dict[str, JSONValue]]:
        context = (
            runtime.context
            if runtime is not None and isinstance(runtime.context, ForgeRuntimeContext)
            else None
        )
        try:
            result = await _call_executor(executor, arguments, context=context)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - tool boundary is model-facing
            # A raised executor failure is a real execution error, not a
            # completed tool run.  Raising ToolException (with
            # handle_tool_error=True) lets LangChain record a ToolMessage with
            # status=error and keeps Forge's failure view consistent.
            result = AgentToolResult(
                tool_call_id="",
                name=name,
                ok=False,
                content=str(exc),
                error=str(exc),
            )
            runtime_tool_call_id = getattr(runtime, "tool_call_id", None)
            if not result.tool_call_id and isinstance(runtime_tool_call_id, str):
                result = result.model_copy(update={"tool_call_id": runtime_tool_call_id})
            raise ToolException(_tool_result_text(result)) from exc
        runtime_tool_call_id = getattr(runtime, "tool_call_id", None)
        if not result.tool_call_id and isinstance(runtime_tool_call_id, str):
            result = result.model_copy(update={"tool_call_id": runtime_tool_call_id})
        artifact = result.model_dump(mode="json")
        return _tool_result_text(result), cast(dict[str, JSONValue], artifact)

    # ``from __future__ import annotations`` leaves the annotation as a
    # string, while StructuredTool's injected-argument cache inspects the raw
    # signature. Restore the runtime class explicitly.
    invoke.__annotations__["runtime"] = ToolRuntime
    return cast(
        ForgeStructuredTool,
        ForgeStructuredTool.from_function(
            coroutine=invoke,
            name=name,
            description=description,
            args_schema=runtime_args_schema,
            response_format="content_and_artifact",
            handle_tool_error=True,
            executor=executor,
        ),
    )


def _args_fields_from_json_schema(input_schema: Mapping[str, JSONValue]) -> dict[str, Any]:
    """Translate a JSON ``input_schema`` into Pydantic field definitions.

    Optional properties get a ``None`` default so providers may omit them; the
    required list is honored for required arguments.  Types outside the JSON
    primitives are treated as ``Any``.
    """

    fields: dict[str, Any] = {}
    properties = input_schema.get("properties", {})
    raw_required = input_schema.get("required")
    required = set(raw_required) if isinstance(raw_required, list) else set()
    if not isinstance(properties, Mapping):
        return fields
    for name, prop in properties.items():
        if not isinstance(name, str) or not isinstance(prop, Mapping):
            continue
        field_type: Any = Any
        if isinstance(prop.get("type"), str):
            field_type = _JSON_TYPE_TO_PYTHON.get(str(prop.get("type")), Any)
        description = prop.get("description")
        if name in required:
            fields[name] = (
                field_type,
                Field(description=description if isinstance(description, str) else None),
            )
        else:
            fields[name] = (
                field_type | None,
                Field(
                    default=None,
                    description=description if isinstance(description, str) else None,
                ),
            )
    return fields


def _call_executor(
    executor: ToolExecutor,
    arguments: Mapping[str, JSONValue],
    *,
    context: ForgeRuntimeContext | None,
) -> Awaitable[AgentToolResult]:
    """Invoke a tool executor with the injected runtime context.

    Every Forge ``ToolExecutor`` receives the session-owned
    ``ForgeRuntimeContext`` as ``context``; executors that do not need it
    declare the parameter and ignore it.
    """

    return executor(arguments, signal=None, context=context)


def _tool_result_text(result: AgentToolResult) -> str:
    content = result.content
    if result.data is not None and not content:
        content = json.dumps(result.data, ensure_ascii=False)
    if not result.ok and result.error and result.error not in content:
        content = f"{content}\n\nError: {result.error}"
    return content
