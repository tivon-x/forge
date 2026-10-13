"""Request identity and native tool replay on the existing session journal."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any, NotRequired, cast, get_origin
from uuid import uuid4

from jsonschema.validators import validator_for  # type: ignore[import-untyped]
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain.tools import InjectedState, InjectedStore, ToolRuntime
from langchain_core.messages import AIMessage, RemoveMessage, ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command
from referencing import Registry

from forge_agent.session import CustomEntry, LeafEntry
from forge_agent.tool_execution import (
    NESTED_CALLS_METADATA_KEY,
    ToolCallHandler,
    _error_message,
    _execute_tool_handler,
    get_nested_tool_executor,
    tool_exposure,
)
from forge_coding.features.documents import _allocate_id, _find_document
from forge_coding.sessions.checkpoints import JournalCheckpointSaver

if TYPE_CHECKING:
    from forge_coding.sessions.session import CodingSession

EXECUTION_KIND = "forge.execution.v1"


def _declaration(tool: BaseTool | dict[str, Any]) -> dict[str, Any]:
    if not isinstance(tool, BaseTool):
        return tool
    schema = tool.tool_call_schema
    return {
        "kind": "forge.tool",
        "name": tool.name,
        "description": tool.description,
        "schema": schema
        if isinstance(schema, dict)
        else (
            cast(Any, schema).model_json_schema()
            if hasattr(schema, "model_json_schema")
            else schema.schema()
        ),
    }


class ExecutionState(AgentState[Any]):
    forge_request_index: NotRequired[int]


class GenerationMiddleware(AgentMiddleware[ExecutionState, Any, Any]):
    state_schema = ExecutionState
    durable_generation = True

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[..., Awaitable[ModelResponse[Any]]],
    ) -> Any:
        execution = request.runtime.context.execution
        return await execution.model(request, handler)

    async def aafter_model(self, state: ExecutionState, runtime: Any) -> dict[str, Any]:
        del runtime
        return {"forge_request_index": state.get("forge_request_index", 0) + 1}


class DurableExecution:
    """One live run identity; native state and journal outcomes are the facts."""

    def __init__(self, session: CodingSession, saver: JournalCheckpointSaver, run_id: int) -> None:
        self.session, self.saver, self.run_id = session, saver, run_id
        self.thread_id = f"forge-run-{run_id}"
        self.host = uuid4().hex
        self.previous_live = copy.deepcopy(session._live)

    def request_key(self, request: ModelRequest[Any]) -> str:
        return f"{self.thread_id}:request:{request.state.get('forge_request_index', 0)}"

    async def read(self, key: str) -> dict[str, Any] | None:
        document = await self.saver.documents.get(EXECUTION_KIND, key)
        return document["value"] if document else None

    def compaction_writes(self, records: dict[str, Any], entry_id: str) -> list[dict[str, Any]]:
        pending = [
            d
            for d in records["documents"]
            if d["record"]["kind"] == EXECUTION_KIND
            and d["value"].get("run") == self.run_id
            and "outcome" not in d["value"]
            and d["value"].get("phase") in {"request", "retry"}
        ]
        if not pending:
            return []
        document = max(pending, key=lambda d: d["value"]["index"])
        return [
            {
                "type": "document.change",
                "id": document["record"]["id"],
                "content": {
                    "kind": "base",
                    "version": 1,
                    "value": {**document["value"], "phase": "prepare", "compacted": entry_id},
                },
            }
        ]

    async def change(
        self,
        key: str,
        callback: Callable[[dict[str, Any] | None], dict[str, Any]],
        *,
        live: Callable[[dict[str, Any]], None] | None = None,
        aborted: dict[str, Any] | None = None,
        response_messages: list[Any] | None = None,
    ) -> dict[str, Any]:
        session = self.session
        async with session._feature_lock:
            session._assert_session_usable()
            storage = self.saver.documents.storage
            records = await storage.read_records()
            current = _find_document(records, EXECUTION_KIND, key)
            value = callback(copy.deepcopy(current["value"]) if current else None)
            content = {"kind": "base", "version": 1, "value": value}
            write = (
                {"type": "document.change", "id": current["record"]["id"], "content": content}
                if current
                else {
                    "type": "document.create",
                    "record": {
                        "id": _allocate_id(records),
                        "kind": EXECUTION_KIND,
                        "key": key,
                        "scope": {"kind": "conversation", "conversationId": 0},
                        "history": "latest",
                    },
                    "content": content,
                }
            )
            writes = [write]
            visible = copy.deepcopy(session._live)
            if live is not None:
                live(visible)
                writes.append(session._live_write(dict(records), visible))
            try:
                entries = list(session._pending_initial_entries)
                next_parent = session._last_parent_id
                accepted_runs: dict[str, str | None] = {}
                if response_messages:
                    staged, next_parent, accepted_runs = session._message_entries(response_messages)
                    entries.extend(staged)
                audit = (
                    CustomEntry(
                        parent_id=next_parent,
                        namespace="forge.aborted_attempt.v1",
                        data=aborted,
                    )
                    if aborted is not None
                    else None
                )
                if audit is not None:
                    entries.append(audit)
                    next_parent = audit.id
                if audit is not None or response_messages:
                    entries.append(LeafEntry(parent_id=next_parent, entry_id=next_parent))
                await storage.commit_records(writes, entries)
                session._pending_initial_entries = ()
                if audit is not None or response_messages:
                    session._last_parent_id = next_parent
                    await session._refresh_persisted_state(leaf_id=next_parent)
                    session._accept_message_runs(accepted_runs)
                if live is not None:
                    session._live = visible
                await session._refresh_committed_view()
            except BaseException as error:
                session._invalidate_storage_session(error)
                raise
            return value

    async def model(self, request: ModelRequest[Any], handler: Callable[..., Any]) -> Any:
        key = self.request_key(request)
        saved = await self.read(key)
        if saved is not None and "outcome" in saved:
            response = ModelResponse(
                result=self.saver.decode(saved["outcome"]),
                structured_response=self.saver.decode(saved["structured"]),
            )
            command = self.saver.decode(saved["command"]) if "command" in saved else None
            return ExtendedModelResponse(response, command) if command is not None else response
        if saved is None or saved.get("phase") == "prepare":
            prepared = saved or {}
            if prepared.get("compacted"):
                request = request.override(messages=list(self.session._state.messages))
            saved = await self.change(
                key,
                lambda _: {
                    "phase": "request",
                    "conversation": 0,
                    "run": self.run_id,
                    "index": request.state.get("forge_request_index", 0),
                    "attempt": prepared.get("attempt", 0),
                    "compacted": prepared.get("compacted"),
                    "model": self.session.model,
                    "provider": self.session.provider_name,
                    "thinking": self.session.thinking_level,
                    "settings": self.saver.encode(request.model_settings),
                    "messages": self.saver.encode(request.messages),
                    "system": self.saver.encode(request.system_message),
                    "tool_choice": self.saver.encode(request.tool_choice),
                    "tools": [_declaration(tool) for tool in request.tools],
                    "host": self.host,
                },
            )
        elif saved.get("host") != self.host:
            partial = self.previous_live.get("assistant")
            thinking = self.previous_live.get("thinking")

            def resumed(value: dict[str, Any] | None) -> dict[str, Any]:
                assert value is not None
                if partial or thinking:
                    value.setdefault("aborted", []).append(
                        {"attempt": value["attempt"], "content": partial, "thinking": thinking}
                    )
                value["host"] = self.host
                return value

            saved = await self.change(
                key,
                resumed,
                live=lambda v: v.update(assistant="", thinking="", aborted=False),
                aborted={
                    "request": key,
                    "attempt": saved["attempt"],
                    "content": partial,
                    "thinking": thinking,
                    "status": "aborted",
                }
                if partial or thinking
                else None,
            )
        provider = request.model
        if (
            saved["model"] != self.session.model
            or saved["provider"] != self.session.provider_name
            or saved["thinking"] != self.session.thinking_level
        ):
            from forge_coding.providers.runtime import create_model_provider

            settings = self.session._provider_settings
            config = (
                settings.get_provider(saved["provider"])
                if settings is not None
                else self.session._runtime_provider_config
            )
            if config is None or config.name != saved["provider"]:
                raise RuntimeError("Prepared provider is unavailable; restore its configuration")
            provider = create_model_provider(
                config,
                credential_store=self.session._credential_store,
                model=saved["model"],
                thinking_level=saved["thinking"],
            )
            self.session._owned_providers.append(provider)
        tools_by_name = {tool.name: tool for tool in request.tools if isinstance(tool, BaseTool)}
        bound_tools: list[BaseTool | dict[str, Any]] = []
        copied: list[tuple[BaseTool, BaseTool]] = []
        for declaration in saved["tools"]:
            if isinstance(declaration, str):
                if declaration not in tools_by_name:
                    raise RuntimeError("Prepared tool declaration is unavailable")
                bound_tools.append(tools_by_name[declaration])
            elif declaration.get("kind") == "forge.tool":
                current = tools_by_name.get(declaration["name"])
                if current is None:
                    raise RuntimeError("Prepared tool declaration is unavailable")
                declared = (
                    current
                    if _declaration(current) == declaration
                    else current.model_copy(
                        update={
                            "description": declaration["description"],
                            "args_schema": declaration["schema"],
                        }
                    )
                )
                bound_tools.append(declared)
                if current is not declared:
                    copied.append((current, declared))
            else:
                bound_tools.append(declaration)
        bound = request.override(
            model=provider,
            messages=self.saver.decode(saved["messages"]),
            system_message=self.saver.decode(saved["system"]),
            model_settings=self.saver.decode(saved["settings"]),
            tools=bound_tools,
            tool_choice=self.saver.decode(saved["tool_choice"])
            if "tool_choice" in saved
            else request.tool_choice,
        )
        try:
            response = await handler(bound)
        finally:
            for current, declared in copied:
                current.metadata = {
                    **(current.metadata or {}),
                    "forge.declared": (declared.metadata or {}).get("forge.declared"),
                }
        command = response.command if isinstance(response, ExtendedModelResponse) else None
        response = (
            response.model_response if isinstance(response, ExtendedModelResponse) else response
        )
        if isinstance(response, AIMessage):
            response = ModelResponse(result=[response])
        if saved.get("compacted"):
            update = command.update if command is not None else {}
            if update is not None and not isinstance(update, dict):
                raise RuntimeError("Compaction requires a mapping native model state update")
            command = (
                replace(
                    command,
                    update={
                        **(update or {}),
                        "messages": [
                            RemoveMessage(id=REMOVE_ALL_MESSAGES),
                            *bound.messages,
                            *((update or {}).get("messages", [])),
                        ],
                    },
                )
                if command is not None
                else Command(
                    update={"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *bound.messages]}
                )
            )
        for message in response.result:
            if not message.id:
                message.id = f"forge-response-{uuid4().hex}"
        outcome, structured = (
            self.saver.encode(response.result),
            self.saver.encode(response.structured_response),
        )
        if self.session._progress is not None:
            await self.session._progress.flush()
        committed_messages = [
            message
            for message in self.session._unpersisted_messages(bound.messages)
            if isinstance(message, (AIMessage, ToolMessage))
        ]
        await self.change(
            key,
            lambda value: {
                **cast(dict[str, Any], value),
                "phase": "tools"
                if any(isinstance(m, AIMessage) and m.tool_calls for m in response.result)
                else "done",
                "outcome": outcome,
                "structured": structured,
                "command": self.saver.encode(command),
            },
            live=lambda v: v.update(assistant="", thinking="", generating=False),
            response_messages=[*committed_messages, *response.result],
        )
        return ExtendedModelResponse(response, command) if command is not None else response

    async def retry_state(self, request: ModelRequest[Any]) -> tuple[int, float]:
        value = await self.read(self.request_key(request))
        return (value.get("attempt", 0), value.get("until", 0)) if value else (0, 0)

    async def record_retry(self, request: ModelRequest[Any], attempt: int, delay: float) -> None:
        if self.session._progress is not None:
            await self.session._progress.flush()
        partial = self.session._live.get("assistant")
        thinking = self.session._live.get("thinking")
        await self.change(
            self.request_key(request),
            lambda value: {
                **cast(dict[str, Any], value),
                "phase": "retry",
                "attempt": attempt,
                "until": time.time() + delay,
            },
            live=lambda v: v.update(assistant="", thinking=""),
            aborted={
                "request": self.request_key(request),
                "attempt": attempt - 1,
                "content": partial,
                "thinking": thinking,
                "status": "aborted",
            }
            if partial or thinking
            else None,
        )

    async def tool(
        self, request: ToolCallRequest, handler: ToolCallHandler, *, nested: bool
    ) -> tuple[Any, bool]:
        # Scope provider call IDs by their committed request, and children by native namespace.
        messages = request.state.get("messages", [])
        assistant = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        settings = request.runtime.config.get("configurable", {})
        namespace = settings.get("checkpoint_ns", "")
        assistant_id = assistant.id if assistant else ""
        request_index = request.state.get("forge_request_index", 1) - 1
        identity = (
            f"{self.thread_id}:{namespace}:{request_index}:{assistant_id}:{request.tool_call['id']}"
        )
        key = "tool:" + hashlib.sha256(identity.encode()).hexdigest()
        saved = await self.read(key)
        if saved is not None and "outcome" in saved:
            return self.saver.decode(saved["outcome"]), saved["valid"]
        tool = request.tool
        allowed = (
            isinstance(tool, BaseTool) and (tool.metadata or {}).get("forge.available") is not False
        )
        if isinstance(tool, BaseTool) and allowed:
            try:
                exposure = tool_exposure(tool)
            except ValueError:
                result, valid = await _execute_tool_handler(request, handler, nested=nested)
                return await self.outcome(key, result, valid, request)
            allowed = exposure not in (
                {"hidden", "model-only"} if nested else {"hidden", "codemode"}
            )
            if nested:
                allowed = allowed and not tool.return_direct
            else:
                metadata = tool.metadata or {}
                allowed = (
                    allowed
                    and metadata.get("forge.codemode_only") is not True
                    and (exposure != "deferred" or metadata.get("forge.declared") is True)
                )
        replay = (
            (tool.metadata or {}).get("forge.replay", "unsafe")
            if isinstance(tool, BaseTool)
            else "unsafe"
        )
        if saved is not None:
            request = request.override(tool_call={**request.tool_call, "args": saved["arguments"]})
            if not allowed or saved["replay"] != "safe" or replay != "safe":
                interrupted = _error_message(
                    request, "Tool interrupted; it may have partially run."
                )
                interrupted.artifact = {
                    "forge.partial": self.previous_live.get("tools", {}).get(
                        request.tool_call["id"]
                    )
                }
                return await self.outcome(key, interrupted, True, request)
        if not allowed:
            result, valid = await _execute_tool_handler(request, handler, nested=nested)
            return await self.outcome(key, result, valid, request)
        schema = (
            (tool.args_schema if isinstance(tool.args_schema, dict) else tool.get_input_schema())
            if isinstance(tool, BaseTool)
            else None
        )
        if isinstance(schema, dict):
            try:
                validator_for(schema)(schema, registry=Registry()).validate(
                    request.tool_call["args"]
                )
            except Exception:
                invalid = _error_message(request, "Invalid tool arguments.")
                return await self.outcome(key, invalid, True, request)
        elif schema is not None:
            try:
                arguments = dict(request.tool_call["args"])
                injected: set[str] = set()
                model_fields = (
                    cast(Any, schema).model_fields
                    if hasattr(schema, "model_fields")
                    else cast(Any, schema).__fields__
                )
                for name, field in model_fields.items():
                    if (
                        field.annotation is ToolRuntime
                        or get_origin(field.annotation) is ToolRuntime
                    ):
                        arguments[name] = request.runtime
                        injected.add(name)
                    for annotation in getattr(field, "metadata", ()):
                        if isinstance(annotation, InjectedState) or annotation is InjectedState:
                            selected = getattr(annotation, "field", None)
                            arguments[name] = request.state[selected] if selected else request.state
                        elif isinstance(annotation, InjectedStore) or annotation is InjectedStore:
                            arguments[name] = request.runtime.store
                        elif annotation is InjectedToolCallId or isinstance(
                            annotation, InjectedToolCallId
                        ):
                            arguments[name] = request.tool_call["id"]
                        else:
                            continue
                        injected.add(name)
                validated = (
                    cast(Any, schema).model_validate(arguments)
                    if hasattr(schema, "model_validate")
                    else cast(Any, schema).parse_obj(arguments)
                )
                final_arguments = (
                    validated.model_dump(mode="json", exclude=injected)
                    if hasattr(validated, "model_dump")
                    else json.loads(validated.json(exclude=injected))
                )
                request = request.override(tool_call={**request.tool_call, "args": final_arguments})
            except ValueError:
                invalid = _error_message(request, "Invalid tool arguments.")
                return await self.outcome(key, invalid, True, request)
        if saved is None:
            await self.change(
                key,
                lambda _: {
                    "phase": "execute",
                    "call": request.tool_call["id"],
                    "assistant": assistant.id if assistant else None,
                    "arguments": request.tool_call["args"],
                    "replay": replay,
                    "identity": identity,
                    "request_index": request_index,
                    "namespace": namespace,
                },
                live=lambda v: v["tools"].setdefault(
                    request.tool_call["id"], {"start": None, "updates": []}
                ),
            )
        if saved is not None:
            await self.change(
                key,
                lambda v: cast(dict[str, Any], v),
                live=lambda v: v["tools"].pop(request.tool_call["id"], None),
            )
        result, valid = await _execute_tool_handler(request, handler, nested=nested)
        if not nested and isinstance(result, ToolMessage):
            try:
                trace = get_nested_tool_executor().artifact()
            except RuntimeError:
                trace = None
            if trace and trace.get("calls"):
                artifact = dict(result.artifact) if isinstance(result.artifact, dict) else {}
                artifact[NESTED_CALLS_METADATA_KEY] = trace
                result = result.model_copy(update={"artifact": artifact})
        return await self.outcome(key, result, valid, request)

    async def outcome(
        self, key: str, result: Any, valid: bool, request: ToolCallRequest
    ) -> tuple[Any, bool]:
        if isinstance(result, ToolMessage) and not result.id:
            result.id = f"forge-tool-{uuid4().hex}"
        try:
            encoded = self.saver.encode(result)
        except (TypeError, ValueError):
            result = _error_message(request, "Tool result cannot be persisted.")
            result.id = f"forge-tool-{uuid4().hex}"
            encoded = self.saver.encode(result)
        assistant = next(
            (m for m in reversed(request.state.get("messages", [])) if isinstance(m, AIMessage)),
            None,
        )
        identity = {
            "call": request.tool_call["id"],
            "assistant": assistant.id if assistant else None,
            "namespace": request.runtime.config.get("configurable", {}).get("checkpoint_ns", ""),
            "request_index": request.state.get("forge_request_index", 1) - 1,
        }
        if self.session._progress is not None:
            await self.session._progress.flush()
        await self.change(
            key,
            lambda value: {
                **(value or identity),
                "phase": "done",
                "outcome": encoded,
                "valid": valid,
            },
            live=lambda v: v["tools"].pop(request.tool_call["id"], None),
        )
        return result, valid

    async def abort_messages(self) -> None:
        records = await self.saver.documents.storage.read_records()
        partial = self.previous_live.get("assistant")
        thinking = self.previous_live.get("thinking")
        if partial or thinking:
            interrupted = next(
                (
                    d
                    for d in records["documents"]
                    if d["record"]["kind"] == EXECUTION_KIND
                    and d["value"].get("run") == self.run_id
                    and "outcome" not in d["value"]
                ),
                None,
            )
            if interrupted is not None:
                value = interrupted["value"]
                await self.change(
                    interrupted["record"]["key"],
                    lambda v: {**cast(dict[str, Any], v), "phase": "aborted"},
                    aborted={
                        "request": interrupted["record"]["key"],
                        "attempt": value["attempt"],
                        "content": partial,
                        "thinking": thinking,
                        "status": "aborted",
                    },
                    live=lambda v: v.update(aborted=True),
                )
        values = [d["value"] for d in records["documents"] if d["record"]["kind"] == EXECUTION_KIND]
        requests = sorted(
            (v for v in values if v.get("run") == self.run_id and "outcome" in v),
            key=lambda v: v["index"],
        )
        messages = list(self.session._harness.messages)
        known = {m.id for m in messages if m.id}
        for request in requests:
            for response in self.saver.decode(request["outcome"]):
                if response.id not in known:
                    messages.append(response)
                    known.add(response.id)
                if not isinstance(response, AIMessage):
                    continue
                position = next(i for i, m in enumerate(messages) if m.id == response.id)
                paired = {
                    m.tool_call_id for m in messages[position + 1 :] if isinstance(m, ToolMessage)
                }
                for call in response.tool_calls:
                    if call["id"] in paired:
                        continue
                    saved = next(
                        (
                            v
                            for v in values
                            if v.get("assistant") == response.id and v.get("call") == call["id"]
                        ),
                        None,
                    )
                    result = (
                        self.saver.decode(saved["outcome"])
                        if saved and "outcome" in saved
                        else ToolMessage(
                            content="Tool call aborted",
                            name=call["name"],
                            tool_call_id=call["id"],
                            status="error",
                        )
                    )
                    if isinstance(result, Command) and isinstance(result.update, dict):
                        result = next(
                            (
                                m
                                for m in result.update.get("messages", [])
                                if isinstance(m, ToolMessage) and m.tool_call_id == call["id"]
                            ),
                            ToolMessage(
                                content="Tool call aborted",
                                name=call["name"],
                                tool_call_id=call["id"],
                                status="error",
                            ),
                        )
                    if isinstance(result, ToolMessage):
                        messages.append(result)
        self.session._harness.replace_messages(messages)
