"""Offline regression coverage for the R2 usage/cost ledger."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from fake_models import ScriptedChatModel
from forge_agent.retry import RetryPolicy
from forge_agent.session import CustomEntry, JsonlSessionStorage, LeafEntry
from forge_agent.session.entries import BranchSummaryEntry, MessageEntry
from forge_coding import CodingSession, CodingSessionConfig
from forge_coding.commands.default_registry import _usage_cost_text
from forge_coding.providers.catalog import ModelCostTier
from forge_coding.providers.config import OpenAICompatibleProviderConfig, ProviderModelMetadata
from forge_coding.sessions.branch_summary import summarize_branch_messages_with_model
from forge_coding.sessions.compaction import CompactionPlan
from forge_coding.sessions.session import _stream_native_model_result
from forge_coding.sessions.usage import (
    USAGE_NAMESPACE,
    UsageRecord,
    UsageTotals,
    aggregate_usage_entries,
    cache_miss_tokens,
    merge_stream_metadata,
    usage_record_from_message,
)


def _provider(*, cache_subset: bool = False, cost: dict[str, float] | None = None):
    compat = {"usage_cache_details": "subset"} if cache_subset else {}
    return OpenAICompatibleProviderConfig(
        name="test-provider",
        models=("test-model",),
        default_model="test-model",
        compat=compat,
        model_metadata={
            "test-model": ProviderModelMetadata(
                cost=cost if cost is not None else {"input": 2, "output": 4},
            )
        },
    )


def test_usage_preserves_total_input_and_marks_cache_partial() -> None:
    message = AIMessage(
        content="ok",
        usage_metadata={
            "input_tokens": 1_000,
            "output_tokens": 500,
            "total_tokens": 1_500,
            "input_token_details": {"cache_read": 100, "cache_creation": 0},
        },
        response_metadata={"model_name": "test-model"},
    )
    record = usage_record_from_message(
        message,
        purpose="agent",
        provider="test-provider",
        requested_model="test-model",
        provider_config=_provider(),
    )

    assert record.input_tokens == 1_000
    assert record.cache_read_tokens == 100
    assert record.cache_write_tokens == 0
    assert record.normalization == "partial"
    assert record.cost is None
    assert record.pricing_source is not None


def test_usage_prices_exact_calls_and_free_rates() -> None:
    message = AIMessage(
        content="ok",
        usage_metadata={"input_tokens": 1_000, "output_tokens": 500, "total_tokens": 1_500},
        response_metadata={"model_name": "test-model"},
    )
    record = usage_record_from_message(
        message,
        purpose="compaction",
        provider="test-provider",
        requested_model="test-model",
        provider_config=_provider(),
    )
    assert record.normalization == "exact"
    assert record.cost == 0.004
    assert record.pricing_source is not None

    free = usage_record_from_message(
        message,
        purpose="auto_name",
        provider="test-provider",
        requested_model="test-model",
        provider_config=_provider(cost={"input": 0, "output": 0}),
    )
    assert free.cost == 0
    assert free.pricing_source is not None


def test_missing_billable_cache_details_are_partial_and_unpriced() -> None:
    record = usage_record_from_message(
        AIMessage(
            content="ok",
            usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
            response_metadata={"model_name": "test-model"},
        ),
        purpose="agent",
        provider="test-provider",
        requested_model="test-model",
        provider_config=_provider(
            cost={"input": 1, "output": 2, "cacheRead": 0.1, "cacheWrite": 1.25}
        ),
    )

    assert record.normalization == "partial"
    assert record.cost is None


@pytest.mark.parametrize(
    "purpose", ["agent", "compaction", "branch_summary", "auto_name", "subagent"]
)
def test_all_usage_purposes_round_trip(purpose: str) -> None:
    message = AIMessage(
        content="ok",
        usage_metadata={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
        response_metadata={"model_name": "test-model"},
    )
    record = usage_record_from_message(
        message,
        purpose=purpose,  # type: ignore[arg-type]
        provider="test-provider",
        requested_model="test-model",
        provider_config=_provider(),
    )
    entry = CustomEntry(namespace=USAGE_NAMESPACE, data=record.to_data())
    restored = UsageRecord.from_data(entry.data)
    assert restored == record
    assert aggregate_usage_entries([entry]).by_purpose[purpose].calls == 1


@pytest.mark.parametrize("input_tokens,expected_input_rate", [(100, 1.0), (101, 3.0)])
def test_pricing_tier_threshold_is_strictly_above(
    input_tokens: int,
    expected_input_rate: float,
) -> None:
    provider = OpenAICompatibleProviderConfig(
        name="test-provider",
        models=("test-model",),
        default_model="test-model",
        model_metadata={
            "test-model": ProviderModelMetadata(
                cost={"input": 1, "output": 2},
                cost_tiers=(ModelCostTier(input_tokens_above=100, input=3, output=4),),
            )
        },
    )
    record = usage_record_from_message(
        AIMessage(
            content="ok",
            usage_metadata={
                "input_tokens": input_tokens,
                "output_tokens": 1,
                "total_tokens": input_tokens + 1,
            },
            response_metadata={"model_name": "test-model"},
        ),
        purpose="agent",
        provider="test-provider",
        requested_model="test-model",
        provider_config=provider,
    )
    assert record.rates["input"] == expected_input_rate


@pytest.mark.parametrize(
    "details,missing_field",
    [({"cache_read": 10}, "cache_write_tokens"), ({"cache_creation": 10}, "cache_read_tokens")],
)
def test_partial_cache_details_keep_missing_field_unknown_and_unpriced(
    details: dict[str, int],
    missing_field: str,
) -> None:
    record = usage_record_from_message(
        AIMessage(
            content="ok",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 10,
                "total_tokens": 110,
                "input_token_details": details,
            },
            response_metadata={"model_name": "test-model"},
        ),
        purpose="agent",
        provider="test-provider",
        requested_model="test-model",
        provider_config=_provider(cache_subset=True, cost={"input": 1, "output": 2}),
    )
    assert getattr(record, missing_field) is None
    assert record.cost is None


def test_usage_aggregate_preserves_unknown_tokens_and_sums_known_costs() -> None:
    def record(*, input_tokens: int | None, cost: float | None) -> UsageRecord:
        return UsageRecord(
            purpose="agent",
            provider="p",
            requested_model="m",
            response_model="m",
            input_tokens=input_tokens,
            output_tokens=2,
            cache_read_tokens=0,
            cache_write_tokens=0,
            total_tokens=None if input_tokens is None else input_tokens + 2,
            normalization="partial",
            rates={
                "input": None,
                "output": None,
                "cache_read": None,
                "cache_write": None,
                "tier": None,
            },
            cost=cost,
            pricing_source="frozen" if cost is not None else None,
        )

    first = record(input_tokens=10, cost=0.25)
    unknown = record(input_tokens=None, cost=None)
    totals = aggregate_usage_entries(
        [
            CustomEntry(namespace=USAGE_NAMESPACE, data=first.to_data()),
            CustomEntry(namespace=USAGE_NAMESPACE, data=unknown.to_data()),
        ]
    )
    assert totals.input_tokens is None
    assert totals.output_tokens == 4
    assert totals.cost == 0.25
    assert totals.known_cost_calls == 1


def test_usage_cost_projection_marks_mixed_pricing_and_unknown_as_na() -> None:
    assert _usage_cost_text(UsageTotals(calls=2, cost=0.25, known_cost_calls=1)) == (
        "$0.250000 (1/2 priced)"
    )
    assert _usage_cost_text(UsageTotals(calls=1)) == "n/a"


def test_inconsistent_total_is_partial_and_nonfinite_cost_is_rejected() -> None:
    record = usage_record_from_message(
        AIMessage(
            content="ok",
            usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 99},
            response_metadata={"model_name": "test-model"},
        ),
        purpose="agent",
        provider="test-provider",
        requested_model="test-model",
        provider_config=_provider(cost={"input": float("inf"), "output": 1}),
    )
    assert record.normalization == "partial"
    assert record.cost is None
    assert record.pricing_source is None
    assert UsageRecord.from_data(record.to_data()) == record


def test_cache_miss_uses_total_input_and_structural_resets() -> None:
    def entry(record: UsageRecord) -> CustomEntry:
        return CustomEntry(namespace=USAGE_NAMESPACE, data=record.to_data())

    def fact(input_tokens: int, cache_read: int, purpose: str = "agent") -> UsageRecord:
        return UsageRecord(
            purpose=purpose,  # type: ignore[arg-type]
            provider="p",
            requested_model="m",
            response_model="m",
            input_tokens=input_tokens,
            output_tokens=1,
            cache_read_tokens=cache_read,
            cache_write_tokens=0,
            total_tokens=input_tokens + 1,
            normalization="partial",
            rates={
                "input": None,
                "output": None,
                "cache_read": None,
                "cache_write": None,
                "tier": None,
            },
            cost=None,
            pricing_source=None,
        )

    first = fact(1_500, 1_000)
    internal = fact(1, 0, "auto_name")
    second = fact(3_000, 0)
    structural = BranchSummaryEntry(parent_id="root", summary="summary")
    third = fact(5_000, 0)
    rows = [entry(first), entry(internal), entry(second), structural, entry(third)]
    # auto_name is ignored, and the actual BranchSummaryEntry resets the chain.
    assert cache_miss_tokens(rows) == 1_500
    assert cache_miss_tokens([entry(first), entry(second)]) == 1_500


def test_cache_miss_noise_floor_and_model_switch_match_pi() -> None:
    def fact(input_tokens: int, cache_read: int, *, model: str) -> UsageRecord:
        return UsageRecord(
            purpose="agent",
            provider="p",
            requested_model=model,
            response_model=model,
            input_tokens=input_tokens,
            output_tokens=1,
            cache_read_tokens=cache_read,
            cache_write_tokens=0,
            total_tokens=input_tokens + 1,
            normalization="exact",
            rates={
                "input": None,
                "output": None,
                "cache_read": None,
                "cache_write": None,
                "tier": None,
            },
            cost=None,
            pricing_source=None,
        )

    # Exactly 1024 is noise; changing models does not reset the comparison.
    assert cache_miss_tokens((fact(1_024, 0, model="a"), fact(2_000, 0, model="b"))) is None
    assert cache_miss_tokens((fact(1_025, 0, model="a"), fact(2_000, 0, model="b"))) == 1_025


def test_stream_metadata_merges_nested_details() -> None:
    assert merge_stream_metadata(
        {"input_tokens": 10, "input_token_details": {"cache_read": 2}},
        {"output_tokens": 3, "input_token_details": {"cache_creation": 1}},
    ) == {
        "input_tokens": 10,
        "output_tokens": 3,
        "input_token_details": {"cache_read": 2, "cache_creation": 1},
    }


@pytest.mark.anyio
async def test_helper_stream_metadata_is_merged_across_chunks() -> None:
    class ChunkModel:
        async def astream(self, _messages, **_kwargs):
            yield SimpleNamespace(
                content="a",
                usage_metadata={"input_tokens": 100, "input_token_details": {"cache_read": 20}},
                response_metadata={"model_name": "test-model"},
            )
            yield SimpleNamespace(
                content="b",
                usage_metadata={
                    "output_tokens": 3,
                    "total_tokens": 103,
                    "input_token_details": {"cache_creation": 2},
                },
                response_metadata={"finish_reason": "stop"},
            )

    result = await _stream_native_model_result(
        ChunkModel(),  # type: ignore[arg-type]
        system="system",
        messages=[],
    )
    assert result.message.usage_metadata == {
        "input_tokens": 100,
        "output_tokens": 3,
        "total_tokens": 103,
        "input_token_details": {"cache_read": 20, "cache_creation": 2},
    }
    assert result.message.response_metadata == {
        "model_name": "test-model",
        "finish_reason": "stop",
    }


@pytest.mark.anyio
async def test_empty_branch_summary_still_saves_complete_usage() -> None:
    class EmptySummaryModel:
        async def astream(self, _messages, **_kwargs):
            yield SimpleNamespace(
                content="",
                usage_metadata={"input_tokens": 8, "output_tokens": 0, "total_tokens": 8},
                response_metadata={"model_name": "test-model"},
            )

    usage: list[object] = []
    summary = await summarize_branch_messages_with_model(
        provider=EmptySummaryModel(),  # type: ignore[arg-type]
        model="test-model",
        messages=(MessageEntry(message=AIMessage(content="old")).message,),
        usage_sink=usage,
    )
    assert summary is None
    assert len(usage) == 1
    assert usage[0].usage_metadata["total_tokens"] == 8


@pytest.mark.anyio
async def test_empty_branch_summary_stream_saves_partial_usage() -> None:
    class EmptyStreamModel:
        async def astream(self, _messages, **_kwargs):
            if False:
                yield None

    usage: list[object] = []
    summary = await summarize_branch_messages_with_model(
        provider=EmptyStreamModel(),  # type: ignore[arg-type]
        model="test-model",
        messages=(HumanMessage(content="old"),),
        usage_sink=usage,
    )

    assert summary is None
    assert len(usage) == 1
    record = usage_record_from_message(
        usage[0],
        purpose="branch_summary",
        provider="test-provider",
        requested_model="test-model",
    )
    assert record.normalization == "partial"
    assert record.total_tokens is None


@pytest.mark.anyio
async def test_branch_summary_retries_transient_failures_and_records_only_success_usage() -> None:
    class RetryBranchSummaryModel:
        def __init__(self) -> None:
            self.calls = 0

        async def astream(self, _messages, **_kwargs):
            self.calls += 1
            if self.calls <= 3:
                raise RuntimeError("service unavailable")
            yield SimpleNamespace(
                content="retried summary",
                usage_metadata={"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
                response_metadata={"model_name": "test-model"},
            )

    provider = RetryBranchSummaryModel()
    usage: list[object] = []
    summary = await summarize_branch_messages_with_model(
        provider=provider,  # type: ignore[arg-type]
        model="test-model",
        messages=(HumanMessage(content="old"),),
        policy=RetryPolicy(initial_delay=0, max_delay=0),
        usage_sink=usage,
    )

    assert summary is not None
    assert "retried summary" in summary
    assert provider.calls == 4
    assert len(usage) == 1
    assert usage[0].usage_metadata["total_tokens"] == 3


@pytest.mark.anyio
async def test_branch_summary_does_not_retry_deterministic_failure() -> None:
    class InvalidBranchSummaryModel:
        def __init__(self) -> None:
            self.calls = 0

        async def astream(self, _messages, **_kwargs):
            self.calls += 1
            raise RuntimeError("invalid request")
            yield None

    provider = InvalidBranchSummaryModel()
    with pytest.raises(RuntimeError, match="invalid request"):
        await summarize_branch_messages_with_model(
            provider=provider,  # type: ignore[arg-type]
            model="test-model",
            messages=(HumanMessage(content="old"),),
            policy=RetryPolicy(initial_delay=0, max_delay=0),
        )

    assert provider.calls == 1


def test_usage_missing_rates_are_unknown_not_zero() -> None:
    message = AIMessage(
        content="ok",
        usage_metadata={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
    )
    record = usage_record_from_message(
        message,
        purpose="branch_summary",
        provider="test-provider",
        requested_model="test-model",
        provider_config=None,
    )
    assert record.cost is None
    assert record.pricing_source is None
    assert record.rates == {
        "input": None,
        "output": None,
        "cache_read": None,
        "cache_write": None,
        "tier": None,
    }

    incomplete = usage_record_from_message(
        message,
        purpose="agent",
        provider="test-provider",
        requested_model="test-model",
        provider_config=_provider(cost={"input": 1}),
    )
    assert incomplete.cost is None
    assert incomplete.pricing_source is None


def test_usage_aggregate_and_cache_miss_reset() -> None:
    def entry(record: UsageRecord) -> CustomEntry:
        return CustomEntry(namespace=USAGE_NAMESPACE, data=record.to_data())

    def fact(input_tokens: int, cache_read: int, purpose: str = "agent") -> UsageRecord:
        return UsageRecord(
            purpose=purpose,  # type: ignore[arg-type]
            provider="p",
            requested_model="m",
            response_model="m",
            input_tokens=input_tokens,
            output_tokens=1,
            cache_read_tokens=cache_read,
            cache_write_tokens=0,
            total_tokens=input_tokens + 1,
            normalization="partial",
            rates={
                "input": None,
                "output": None,
                "cache_read": None,
                "cache_write": None,
                "tier": None,
            },
            cost=None,
            pricing_source=None,
        )

    first = fact(5_000, 2_000)
    second = fact(5_000, 0)
    summary = fact(1, 0, "compaction")
    third = fact(5_000, 0)
    totals = aggregate_usage_entries([entry(first), entry(second), entry(summary), entry(third)])

    assert totals.calls == 4
    assert totals.total_tokens == 15_005
    assert totals.cost is None
    assert cache_miss_tokens((first, second, summary, third)) == 5_000


@pytest.mark.anyio
async def test_root_usage_entry_follows_ai_message_and_updates_active_totals(
    tmp_path: Path,
) -> None:
    response = AIMessage(
        content="done",
        usage_metadata={"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
        response_metadata={"model_name": "test-model"},
    )
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel([response]),
            model="test-model",
            provider_name="test-provider",
            system="Forge",
            storage=storage,
            cwd=tmp_path,
            enable_subagents=False,
        )
    )
    _ = [event async for event in session.prompt("hello")]

    entries = await storage.read_all()
    ai_entry = next(
        entry for entry in entries if entry.type == "message" and entry.message.type == "ai"
    )
    usage_entry = next(
        entry for entry in entries if entry.type == "custom" and entry.namespace == USAGE_NAMESPACE
    )
    leaf = entries[-1]
    assert usage_entry.parent_id == ai_entry.id
    assert leaf.type == "leaf"
    assert leaf.entry_id == usage_entry.id
    assert session.usage_totals.calls == 1
    assert session.usage_totals.total_tokens == 13


@pytest.mark.anyio
async def test_branching_from_ai_keeps_adjacent_usage_node_active(tmp_path: Path) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    root = MessageEntry(id="root", message=AIMessage(content="root"))
    answer = MessageEntry(id="answer", parent_id="root", message=AIMessage(content="answer"))
    usage = usage_record_from_message(
        AIMessage(
            content="answer",
            usage_metadata={"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
        ),
        purpose="agent",
        provider="test-provider",
        requested_model="test-model",
    )
    usage_entry = CustomEntry(
        id="answer-usage",
        parent_id="answer",
        namespace=USAGE_NAMESPACE,
        data=usage.to_data(),
    )
    await storage.append(root)
    await storage.append(answer)
    await storage.append(usage_entry)
    # The leaf points at the usage child, not at the storage-only pointer.
    await storage.append(LeafEntry(parent_id="answer-usage", entry_id="answer-usage"))
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel(),
            model="test-model",
            provider_name="test-provider",
            system="Forge",
            storage=storage,
            cwd=tmp_path,
            enable_subagents=False,
        )
    )

    await session.branch_to_entry("answer")

    entries = await storage.read_all()
    leaf = entries[-1]
    assert leaf.type == "leaf"
    assert leaf.entry_id == "answer-usage"
    assert session.usage_totals.calls == 1
    assert session.usage_totals.total_tokens == 6


@pytest.mark.anyio
async def test_failed_compaction_persists_completed_helper_usage_without_reset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    await storage.append(MessageEntry(id="root", message=AIMessage(content="root")))
    await storage.append(
        MessageEntry(id="answer", parent_id="root", message=AIMessage(content="answer"))
    )
    await storage.append(LeafEntry(parent_id="answer", entry_id="answer"))
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=ScriptedChatModel(),
            model="test-model",
            provider_name="test-provider",
            system="Forge",
            storage=storage,
            cwd=tmp_path,
            enable_subagents=False,
        )
    )
    monkeypatch.setattr(
        session,
        "_recent_preserving_compaction_plan",
        lambda: CompactionPlan(
            replace_entry_ids=("root",),
            messages_to_summarize=(HumanMessage(content="root"),),
        ),
    )

    async def fail_after_one_helper(*_args, usage_sink=None, **_kwargs):
        usage_sink.append(
            AIMessage(
                content="helper complete",
                usage_metadata={"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
            )
        )
        raise RuntimeError("second helper failed")

    monkeypatch.setattr(session, "_generate_compaction_summary", fail_after_one_helper)
    with pytest.raises(RuntimeError, match="second helper failed"):
        await session.compact()

    entries = await storage.read_all()
    assert not any(entry.type == "compaction" for entry in entries)
    usage_entries = [
        entry for entry in entries if entry.type == "custom" and entry.namespace == USAGE_NAMESPACE
    ]
    assert len(usage_entries) == 1
    assert usage_entries[0].data["purpose"] == "compaction"
