"""Shared Forge/LangChain tool adapter and direct-execution seam."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Mapping
from typing import Annotated, Any, cast

from langchain.tools import ToolRuntime
from langchain_core.tools import InjectedToolArg, StructuredTool, ToolException
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, create_model

from forge_agent.context import ForgeRuntimeContext
from forge_agent.tools import AgentToolResult, ToolCancellationToken, ToolExecutor
from forge_agent.types import JSONValue


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

    def __init__(self, *args: Any, executor: ToolExecutor | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._executor = executor

    @property
    def executor(self) -> ToolExecutor:
        """Return the direct executor owned by this native tool."""

        if self._executor is None:
            raise RuntimeError(f"Tool {self.name} has no Forge direct executor")
        return self._executor

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
