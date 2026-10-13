"""Native LangGraph checkpoints on the session's single journal commit line."""

from __future__ import annotations

import asyncio
import base64
import builtins
import hashlib
from collections.abc import AsyncIterator, Coroutine, Iterator, Sequence
from dataclasses import fields, is_dataclass
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel

from forge_coding.features.documents import DocumentStore, _allocate_id, _find_document

CHECKPOINT_KIND = "forge.graph.v1"


class JournalCheckpointSaver(BaseCheckpointSaver[int]):
    """Async saver; retain complete native ancestry, including delta seeds."""

    def __init__(self, documents: DocumentStore) -> None:
        super().__init__(
            serde=JsonPlusSerializer(
                pickle_fallback=False,
                allowed_json_modules=None,
                allowed_msgpack_modules=None,
            )
        )
        self.documents = documents
        self.loop = asyncio.get_running_loop()

    def sync[T](self, operation: Coroutine[Any, Any, T]) -> T:
        try:
            current = asyncio.get_running_loop()
        except RuntimeError:
            current = None
        if current is self.loop or not self.loop.is_running():
            operation.close()
            raise RuntimeError("Use the async checkpoint API on the session event loop")
        return asyncio.run_coroutine_threadsafe(operation, self.loop).result()

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return self.sync(self.aget_tuple(config))

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        async def collect() -> builtins.list[CheckpointTuple]:
            return [
                item async for item in self.alist(config, filter=filter, before=before, limit=limit)
            ]

        return iter(self.sync(collect()))

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return self.sync(self.aput(config, checkpoint, metadata, new_versions))

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self.sync(self.aput_writes(config, writes, task_id, task_path))

    def delete_thread(self, thread_id: str) -> None:
        self.sync(self.adelete_thread(thread_id))

    def encode(self, value: Any) -> builtins.list[str]:
        kind, payload = self.serde.dumps_typed(value)
        if kind == "pickle":
            raise ValueError("Pickle checkpoints are forbidden")
        # Validate the reconstruction allowlist before admitting any write.
        restored = self.serde.loads_typed((kind, payload))
        self._validate_roundtrip(value, restored)
        return [kind, base64.b64encode(payload).decode("ascii")]

    def _validate_roundtrip(self, original: Any, restored: Any) -> None:
        if (
            isinstance(original, BaseModel)
            or is_dataclass(original)
            and not isinstance(original, type)
        ):
            if not isinstance(restored, type(original)):
                raise TypeError("Checkpoint object cannot be reconstructed")
            names = (
                type(original).model_fields
                if isinstance(original, BaseModel)
                else (field.name for field in fields(original))
            )
            for name in names:
                self._validate_roundtrip(getattr(original, name), getattr(restored, name))
        elif isinstance(original, dict):
            if not isinstance(restored, dict) or original.keys() != restored.keys():
                raise TypeError("Checkpoint mapping cannot be reconstructed")
            for key in original:
                self._validate_roundtrip(original[key], restored[key])
        elif isinstance(original, (list, tuple)):
            if not isinstance(restored, (list, tuple)) or len(original) != len(restored):
                raise TypeError("Checkpoint sequence cannot be reconstructed")
            for item, recovered in zip(original, restored, strict=True):
                self._validate_roundtrip(item, recovered)
        elif original is not None and restored is None:
            raise TypeError("Unsupported checkpoint object")

    def decode(self, value: Sequence[str]) -> Any:
        if value[0] == "pickle":
            raise ValueError("Pickle checkpoints are forbidden")
        return self.serde.loads_typed((value[0], base64.b64decode(value[1], validate=True)))

    @staticmethod
    def key(thread: str, namespace: str, checkpoint: str) -> str:
        return hashlib.sha256(f"{thread}\0{namespace}\0{checkpoint}".encode()).hexdigest()

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread = config["configurable"]["thread_id"]
        namespace = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)
        if checkpoint_id is None:
            async for item in self.alist(
                {"configurable": {"thread_id": thread, "checkpoint_ns": namespace}}, limit=1
            ):
                return item
            return None
        document = await self.documents.get(
            CHECKPOINT_KIND, self.key(thread, namespace, checkpoint_id)
        )
        if document is None:
            return None
        return await self.checkpoint_tuple(document["value"])

    async def checkpoint_tuple(self, value: dict[str, Any]) -> CheckpointTuple:
        checkpoint = self.decode(value["checkpoint"])
        channels: dict[str, Any] = {}
        unresolved = dict(checkpoint["channel_versions"])
        cursor: dict[str, Any] | None = value
        while cursor is not None and unresolved:
            for channel, blob in cursor["blobs"].items():
                if channel in unresolved and blob["version"] == unresolved[channel]:
                    del unresolved[channel]
                    if blob["value"] is not None:
                        channels[channel] = self.decode(blob["value"])
            parent = cursor["parent"]
            if not parent:
                break
            settings = parent["configurable"]
            document = await self.documents.get(
                CHECKPOINT_KIND,
                self.key(
                    settings["thread_id"],
                    settings.get("checkpoint_ns", ""),
                    settings["checkpoint_id"],
                ),
            )
            if document is None:
                raise ValueError("Checkpoint ancestor is missing")
            cursor = document["value"]
        checkpoint["channel_values"] = channels
        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": value["thread"],
                    "checkpoint_ns": value["namespace"],
                    "checkpoint_id": value["id"],
                }
            },
            checkpoint=checkpoint,
            metadata=self.decode(value["metadata"]),
            parent_config=value["parent"],
            pending_writes=[
                (w["task"], w["channel"], self.decode(w["value"])) for w in value["writes"]
            ],
        )

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        records = await self.documents.storage.read_records()
        values = [
            d["value"] for d in records["documents"] if d["record"]["kind"] == CHECKPOINT_KIND
        ]
        values.sort(key=lambda v: v["id"], reverse=True)
        count = 0
        for value in values:
            if config is not None:
                settings = config["configurable"]
                if value["thread"] != settings["thread_id"]:
                    continue
                if "checkpoint_ns" in settings and value["namespace"] != settings["checkpoint_ns"]:
                    continue
                if get_checkpoint_id(config) and value["id"] != get_checkpoint_id(config):
                    continue
            if before is not None and value["id"] >= (get_checkpoint_id(before) or ""):
                continue
            item = await self.checkpoint_tuple(value)
            if filter and any(item.metadata.get(k) != v for k, v in filter.items()):
                continue
            if limit is not None and count >= limit:
                break
            count += 1
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        settings = config["configurable"]
        thread, namespace = settings["thread_id"], settings.get("checkpoint_ns", "")
        parent_id = get_checkpoint_id(config)
        parent = (
            {
                "configurable": {
                    "thread_id": thread,
                    "checkpoint_ns": namespace,
                    "checkpoint_id": parent_id,
                }
            }
            if parent_id
            else None
        )
        value = {
            "thread": thread,
            "namespace": namespace,
            "id": checkpoint["id"],
            "parent": parent,
            "checkpoint": self.encode({**checkpoint, "channel_values": {}}),
            "blobs": {
                channel: {
                    "version": version,
                    "value": self.encode(checkpoint["channel_values"][channel])
                    if channel in checkpoint["channel_values"]
                    else None,
                }
                for channel, version in new_versions.items()
            },
            "metadata": self.encode(get_checkpoint_metadata(config, metadata)),
            "writes": [],
        }
        key = self.key(thread, namespace, checkpoint["id"])
        async with self.documents.line:
            self.documents._before_change()
            records = await self.documents.storage.read_records()
            current = _find_document(records, CHECKPOINT_KIND, key)
            if current is not None:
                value["writes"] = current["value"]["writes"]
                value["blobs"] = {**current["value"]["blobs"], **value["blobs"]}
            write = (
                {
                    "type": "document.change",
                    "id": current["record"]["id"],
                    "content": {"kind": "base", "version": 1, "value": value},
                }
                if current
                else {
                    "type": "document.create",
                    "record": {
                        "id": _allocate_id(records),
                        "kind": CHECKPOINT_KIND,
                        "key": key,
                        "scope": {"kind": "conversation", "conversationId": 0},
                        "history": "latest",
                    },
                    "content": {"kind": "base", "version": 1, "value": value},
                }
            )
            await self.documents._commit(write)
        return {
            "configurable": {
                "thread_id": thread,
                "checkpoint_ns": namespace,
                "checkpoint_id": checkpoint["id"],
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        settings = config["configurable"]
        key = self.key(
            settings["thread_id"], settings.get("checkpoint_ns", ""), settings["checkpoint_id"]
        )
        prepared: builtins.list[dict[str, Any]] = [
            {
                "task": task_id,
                "index": WRITES_IDX_MAP.get(channel, index),
                "channel": channel,
                "value": self.encode(value),
                "path": task_path,
            }
            for index, (channel, value) in enumerate(writes)
        ]
        async with self.documents.line:
            records = await self.documents.storage.read_records()
            document = _find_document(records, CHECKPOINT_KIND, key)
            if document is None:
                raise ValueError("Pending writes require a committed checkpoint")
            value = dict(document["value"])
            pending = {(w["task"], w["index"]): w for w in value["writes"]}
            for write in prepared:
                identity = (write["task"], write["index"])
                if write["index"] < 0 or identity not in pending:
                    pending[identity] = write
            value["writes"] = list(pending.values())
            self.documents._before_change()
            await self.documents._commit(
                {
                    "type": "document.change",
                    "id": document["record"]["id"],
                    "content": {"kind": "base", "version": 1, "value": value},
                }
            )

    async def adelete_thread(self, thread_id: str) -> None:
        async with self.documents.line:
            records = await self.documents.storage.read_records()
            writes = [
                {"type": "document.retire", "id": d["record"]["id"]}
                for d in records["documents"]
                if d["record"]["kind"] == CHECKPOINT_KIND and d["value"]["thread"] == thread_id
            ]
            if writes:
                self.documents._before_change()
                try:
                    await self.documents.storage.commit_records(writes)
                    await self.documents._after_commit()
                except BaseException as error:
                    await self.documents._report_error(error)
                    raise
