"""Durable model-call usage and cost accounting.

The ledger is deliberately a small projection of a LangChain response.  The
native message remains the runtime truth; this module only copies allowlisted
usage/model fields into a JSONL ``CustomEntry``.  In particular, child
prompts, completions, tool arguments, and provider metadata are never stored.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from forge_agent.session.entries import (
    BranchSummaryEntry,
    CompactionEntry,
    CustomEntry,
    SessionEntry,
)
from forge_agent.types import JSONValue

USAGE_NAMESPACE = "forge.usage.v1"
USAGE_SCHEMA_VERSION = 1
CACHE_MISS_NOISE_FLOOR = 1024

UsagePurpose = Literal[
    "agent",
    "compaction",
    "branch_summary",
    "auto_name",
    "subagent",
]
UsageNormalization = Literal["exact", "inferred", "partial"]


def _token(value: object) -> int | None:
    """Accept only provider-reported non-negative integer token counts."""

    return value if type(value) is int and value >= 0 else None


def _number(value: object) -> float | None:
    """Accept finite numeric rates while preserving explicit zero."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if result >= 0 and math.isfinite(result) else None


def _has_nonzero_cache_rate(rates: Mapping[str, object]) -> bool:
    return any(
        value is not None and value > 0
        for value in (_number(rates.get("cache_read")), _number(rates.get("cache_write")))
    )


def _detail_value(details: Mapping[str, Any], *keys: str) -> tuple[int | None, bool]:
    """Return a token detail and whether the provider supplied that field."""

    for key in keys:
        if key in details:
            return _token(details.get(key)), True
    return None, False


def _response_model(message: object) -> str | None:
    """Read a provider response model from allowlisted response metadata."""

    metadata = getattr(message, "response_metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    for key in ("response_model", "model_name", "model", "model_id"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _usage_metadata(message: object) -> Mapping[str, Any] | None:
    value = getattr(message, "usage_metadata", None)
    return value if isinstance(value, Mapping) else None


def merge_stream_metadata(
    previous: Mapping[str, Any] | None,
    incoming: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Merge streamed metadata without dropping fields from earlier chunks."""

    if previous is None and incoming is None:
        return None
    merged: dict[str, Any] = dict(previous or {})
    for key, value in (incoming or {}).items():
        old = merged.get(key)
        if isinstance(old, Mapping) and isinstance(value, Mapping):
            nested = dict(old)
            nested.update(value)
            merged[key] = nested
        elif value is not None:
            merged[key] = value
    return merged


def _model_metadata(provider_config: object | None, model: str) -> object | None:
    if provider_config is None:
        return None
    metadata = getattr(provider_config, "model_metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    return metadata.get(model)


def _cache_details_are_subset(provider_config: object | None) -> bool:
    """Return an explicit adapter guarantee, never an inferred provider guess.

    LangChain's ``input_tokens`` is the provider's total input count.  Cache
    detail fields are not assumed to be additive or a decomposition unless an
    adapter opts in with this explicit flag (or the durable compat field).
    """

    if provider_config is None:
        return False
    direct = getattr(provider_config, "usage_cache_details_are_subset", None)
    if direct is True:
        return True
    compat = getattr(provider_config, "compat", None)
    return isinstance(compat, Mapping) and compat.get("usage_cache_details") == "subset"


def _pricing_payload(provider: str, model: str, metadata: object) -> dict[str, JSONValue] | None:
    cost = getattr(metadata, "cost", None)
    tiers = getattr(metadata, "cost_tiers", ())
    if not isinstance(cost, Mapping) or not cost:
        return None
    normalized_cost: dict[str, JSONValue] = {}
    for key in ("input", "output", "cacheRead", "cacheWrite", "cache_read", "cache_write"):
        value = _number(cost.get(key))
        if value is not None:
            normalized_cost[key] = value
    # Input/output rates are the minimum frozen catalog payload.  A cache-only
    # or otherwise incomplete price is unknown, not a priced zero.
    if "input" not in normalized_cost or "output" not in normalized_cost:
        return None
    normalized_tiers: list[JSONValue] = []
    if isinstance(tiers, Sequence) and not isinstance(tiers, (str, bytes, bytearray)):
        for tier in tiers:
            threshold = getattr(tier, "input_tokens_above", None)
            if type(threshold) is not int or threshold < 0:
                continue
            normalized_tiers.append(
                {
                    "input_tokens_above": threshold,
                    "input": _number(getattr(tier, "input", None)),
                    "output": _number(getattr(tier, "output", None)),
                    "cache_read": _number(getattr(tier, "cache_read", None)),
                    "cache_write": _number(getattr(tier, "cache_write", None)),
                }
            )
    return {
        "schema": USAGE_SCHEMA_VERSION,
        "provider": provider,
        "model": model,
        "cost": normalized_cost,
        "cost_tiers": normalized_tiers,
    }


def _pricing_source(payload: Mapping[str, JSONValue] | None) -> str | None:
    if payload is None:
        return None
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return f"catalog:v{USAGE_SCHEMA_VERSION}:sha256:{hashlib.sha256(encoded).hexdigest()}"


def _rates_and_cost(
    *,
    provider: str,
    model: str,
    metadata: object | None,
    input_tokens: int | None,
    output_tokens: int | None,
    cache_read_tokens: int | None,
    cache_write_tokens: int | None,
    cache_details_present: bool,
    cache_read_present: bool,
    cache_write_present: bool,
    cache_subset_guaranteed: bool,
) -> tuple[dict[str, JSONValue], float | None, str | None]:
    payload = _pricing_payload(provider, model, metadata) if metadata is not None else None
    source = _pricing_source(payload)
    if payload is None:
        return (
            {
                "input": None,
                "output": None,
                "cache_read": None,
                "cache_write": None,
                "tier": None,
            },
            None,
            None,
        )

    raw_cost = payload.get("cost")
    cost_map = raw_cost if isinstance(raw_cost, Mapping) else {}
    input_rate = _number(cost_map.get("input"))
    output_rate = _number(cost_map.get("output"))
    cache_read_rate = _number(cost_map.get("cacheRead"))
    if cache_read_rate is None:
        cache_read_rate = _number(cost_map.get("cache_read"))
    cache_write_rate = _number(cost_map.get("cacheWrite"))
    if cache_write_rate is None:
        cache_write_rate = _number(cost_map.get("cache_write"))

    tier_value: int | None = None
    selected = {
        "input": input_rate,
        "output": output_rate,
        "cache_read": cache_read_rate,
        "cache_write": cache_write_rate,
    }
    # A tier is request-wide and needs the cache decomposition.  Do not guess
    # it when LangChain only guarantees total input_tokens.
    if input_tokens is not None and (not cache_details_present or cache_subset_guaranteed):
        # LangChain input_tokens is already total input.  Only an explicit
        # adapter guarantee lets us decompose it into uncached and cache
        # portions; never add cache details to the total a second time.
        prompt_tokens = input_tokens
        tiers = payload.get("cost_tiers")
        if isinstance(tiers, Sequence) and not isinstance(tiers, (str, bytes, bytearray)):
            matching = [
                tier
                for tier in tiers
                if isinstance(tier, Mapping)
                and type(tier.get("input_tokens_above")) is int
                and prompt_tokens > cast(int, tier["input_tokens_above"])
            ]
            if matching:
                chosen = max(matching, key=lambda item: cast(int, item["input_tokens_above"]))
                tier_value = cast(int, chosen["input_tokens_above"])
                for key in selected:
                    value = _number(chosen.get(key))
                    if value is not None:
                        selected[key] = value

    rates: dict[str, JSONValue] = {**selected, "tier": tier_value}
    # A catalog with discounted cache rates tells us cache usage can change
    # the bill.  If the provider omitted the cache fields entirely, pricing
    # the whole prompt at the ordinary input rate would silently treat an
    # unknown cache amount as zero.
    if not cache_details_present and _has_nonzero_cache_rate(selected):
        return rates, None, source
    # Explicit cache detail values are retained but cannot be priced unless the
    # provider adapter guarantees they are a subset of total input_tokens.
    if cache_details_present and not cache_subset_guaranteed:
        return rates, None, source
    # A missing cache component is unknown even when the adapter guarantees
    # cache details are a subset of total input tokens.  Never price it as zero.
    if (
        cache_details_present
        and cache_subset_guaranteed
        and (
            not cache_read_present
            or not cache_write_present
            or cache_read_tokens is None
            or cache_write_tokens is None
        )
    ):
        return rates, None, source
    if input_tokens is None or output_tokens is None:
        return rates, None, source
    if selected["input"] is None or selected["output"] is None:
        return rates, None, source
    if cache_read_tokens is not None and cache_read_tokens > 0 and selected["cache_read"] is None:
        return rates, None, source
    if (
        cache_write_tokens is not None
        and cache_write_tokens > 0
        and selected["cache_write"] is None
    ):
        return rates, None, source
    uncached_input = input_tokens
    if cache_details_present and cache_subset_guaranteed:
        cached = (cache_read_tokens or 0) + (cache_write_tokens or 0)
        if cached > input_tokens:
            return rates, None, source
        uncached_input = input_tokens - cached
    input_rate = selected["input"]
    output_rate = selected["output"]
    if not isinstance(input_rate, float) or not isinstance(output_rate, float):
        return rates, None, source
    cost = uncached_input * input_rate + output_tokens * output_rate
    if cache_read_tokens is not None:
        cache_read_rate = selected["cache_read"]
        if cache_read_tokens > 0 and not isinstance(cache_read_rate, float):
            return rates, None, source
        if isinstance(cache_read_rate, float):
            cost += cache_read_tokens * cache_read_rate
    if cache_write_tokens is not None:
        cache_write_rate = selected["cache_write"]
        if cache_write_tokens > 0 and not isinstance(cache_write_rate, float):
            return rates, None, source
        if isinstance(cache_write_rate, float):
            cost += cache_write_tokens * cache_write_rate
    normalized_cost = cost / 1_000_000
    return rates, normalized_cost if math.isfinite(normalized_cost) else None, source


@dataclass(frozen=True, slots=True)
class UsageRecord:
    """One immutable, allowlisted billable model-call fact."""

    purpose: UsagePurpose
    provider: str
    requested_model: str
    response_model: str
    input_tokens: int | None
    output_tokens: int | None
    cache_read_tokens: int | None
    cache_write_tokens: int | None
    total_tokens: int | None
    normalization: UsageNormalization
    rates: dict[str, JSONValue]
    cost: float | None
    pricing_source: str | None

    def to_data(self) -> dict[str, JSONValue]:
        return {
            "purpose": self.purpose,
            "provider": self.provider,
            "requested_model": self.requested_model,
            "response_model": self.response_model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "total_tokens": self.total_tokens,
            "normalization": self.normalization,
            "rates": dict(self.rates),
            "cost": self.cost,
            "pricing_source": self.pricing_source,
        }

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> UsageRecord | None:
        if value.get("purpose") not in {
            "agent",
            "compaction",
            "branch_summary",
            "auto_name",
            "subagent",
        }:
            return None
        tokens = {
            key: (_token(value.get(key)) if value.get(key) is not None else None)
            for key in (
                "input_tokens",
                "output_tokens",
                "cache_read_tokens",
                "cache_write_tokens",
                "total_tokens",
            )
        }
        if any(value.get(key) is not None and tokens[key] is None for key in tokens):
            return None
        provider = value.get("provider")
        requested = value.get("requested_model")
        response = value.get("response_model")
        normalization = value.get("normalization")
        rates = value.get("rates")
        cost = value.get("cost")
        source = value.get("pricing_source")
        if not all(isinstance(item, str) for item in (provider, requested, response)):
            return None
        if normalization not in {"exact", "inferred", "partial"} or not isinstance(rates, Mapping):
            return None
        if cost is not None and (
            isinstance(cost, bool)
            or not isinstance(cost, (int, float))
            or not math.isfinite(float(cost))
            or float(cost) < 0
        ):
            return None
        if source is not None and not isinstance(source, str):
            return None
        normalized_rates: dict[str, JSONValue] = {}
        for key in ("input", "output", "cache_read", "cache_write"):
            raw = rates.get(key)
            if raw is not None and _number(raw) is None:
                return None
            normalized_rates[key] = None if raw is None else float(cast(float, raw))
        tier = rates.get("tier")
        if tier is not None and (type(tier) is not int or tier < 0):
            return None
        normalized_rates["tier"] = tier
        return cls(
            purpose=cast(UsagePurpose, value["purpose"]),
            provider=cast(str, provider),
            requested_model=cast(str, requested),
            response_model=cast(str, response),
            input_tokens=tokens["input_tokens"],
            output_tokens=tokens["output_tokens"],
            cache_read_tokens=tokens["cache_read_tokens"],
            cache_write_tokens=tokens["cache_write_tokens"],
            total_tokens=tokens["total_tokens"],
            normalization=cast(UsageNormalization, normalization),
            rates=normalized_rates,
            cost=None if cost is None else float(cast(float, cost)),
            pricing_source=source,
        )


def usage_record_from_message(
    message: object,
    *,
    purpose: UsagePurpose,
    provider: str,
    requested_model: str,
    provider_config: object | None = None,
) -> UsageRecord:
    """Normalize one successful native AI response into a durable fact."""

    metadata = _usage_metadata(message)
    input_tokens = output_tokens = total_tokens = None
    cache_read_tokens = cache_write_tokens = None
    cache_details_present = False
    cache_read_present = False
    cache_write_present = False
    total_tokens_invalid = False
    if metadata is not None:
        input_tokens = _token(metadata.get("input_tokens"))
        output_tokens = _token(metadata.get("output_tokens"))
        total_tokens = _token(metadata.get("total_tokens"))
        details = metadata.get("input_token_details")
        if isinstance(details, Mapping):
            cache_read_tokens, read_present = _detail_value(details, "cache_read", "cacheRead")
            cache_write_tokens, write_present = _detail_value(
                details, "cache_creation", "cache_write", "cacheWrite"
            )
            cache_read_present = read_present
            cache_write_present = write_present
            cache_details_present = read_present or write_present
        total_tokens_invalid = (
            "total_tokens" in metadata
            and metadata.get("total_tokens") is not None
            and total_tokens is None
        )
        if (
            ("total_tokens" not in metadata or metadata.get("total_tokens") is None)
            and input_tokens is not None
            and output_tokens is not None
        ):
            total_tokens = input_tokens + output_tokens

    response_model = _response_model(message)
    model_inferred = response_model is None
    response_model = response_model or requested_model
    subset_guaranteed = _cache_details_are_subset(provider_config)
    rates, cost, source = _rates_and_cost(
        provider=provider,
        model=response_model if not model_inferred else requested_model,
        metadata=_model_metadata(provider_config, response_model)
        or _model_metadata(provider_config, requested_model),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        cache_details_present=cache_details_present,
        cache_read_present=cache_read_present,
        cache_write_present=cache_write_present,
        cache_subset_guaranteed=subset_guaranteed,
    )
    missing_billable_cache_details = not cache_details_present and _has_nonzero_cache_rate(rates)
    if metadata is None or input_tokens is None or output_tokens is None or total_tokens is None:
        normalization: UsageNormalization = "partial"
    elif (
        total_tokens_invalid
        or missing_billable_cache_details
        or (
            cache_details_present
            and (
                not subset_guaranteed
                or not cache_read_present
                or not cache_write_present
                or cache_read_tokens is None
                or cache_write_tokens is None
            )
        )
        or total_tokens != input_tokens + output_tokens
    ):
        normalization = "partial"
    elif model_inferred or total_tokens == input_tokens + output_tokens:
        normalization = (
            "inferred" if model_inferred or metadata.get("total_tokens") is None else "exact"
        )
    else:
        normalization = "exact"
    if model_inferred and normalization == "exact":
        normalization = "inferred"
    return UsageRecord(
        purpose=purpose,
        provider=provider,
        requested_model=requested_model,
        response_model=response_model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        total_tokens=total_tokens,
        normalization=normalization,
        rates=rates,
        cost=cost,
        pricing_source=source if cost is not None or source is not None else None,
    )


@dataclass(frozen=True, slots=True)
class UsageTotals:
    """Incrementally cacheable aggregate for the active branch."""

    calls: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    total_tokens: int | None = None
    cost: float | None = None
    known_cost_calls: int = 0
    cache_miss_tokens: int | None = None
    by_purpose: dict[str, UsageTotals] = field(default_factory=dict)
    by_provider_model: dict[str, UsageTotals] = field(default_factory=dict)


def _sum_known(records: Sequence[UsageRecord], attr: str) -> int | None:
    values = [getattr(record, attr) for record in records]
    if not values or any(value is None for value in values):
        return None
    return sum(cast(Iterable[int], values))


def _totals_for_records(
    records: Sequence[UsageRecord],
    *,
    cache_miss_tokens: int | None = None,
    breakdown: bool = True,
) -> UsageTotals:
    known_costs = [record.cost for record in records if record.cost is not None]
    purposes: dict[str, UsageTotals] = {}
    provider_models: dict[str, UsageTotals] = {}
    if breakdown:
        for purpose in sorted({record.purpose for record in records}):
            purposes[purpose] = _totals_for_records(
                tuple(record for record in records if record.purpose == purpose), breakdown=False
            )
        for key in sorted({f"{record.provider}:{record.response_model}" for record in records}):
            provider, _, model = key.partition(":")
            provider_models[key] = _totals_for_records(
                tuple(
                    record
                    for record in records
                    if record.provider == provider and record.response_model == model
                ),
                breakdown=False,
            )
    return UsageTotals(
        calls=len(records),
        input_tokens=_sum_known(records, "input_tokens"),
        output_tokens=_sum_known(records, "output_tokens"),
        cache_read_tokens=_sum_known(records, "cache_read_tokens"),
        cache_write_tokens=_sum_known(records, "cache_write_tokens"),
        total_tokens=_sum_known(records, "total_tokens"),
        # Sum priced calls while retaining the count of unknown calls.  Unknown
        # pricing is never treated as a zero-cost call.
        cost=sum(known_costs) if known_costs else None,
        known_cost_calls=len(known_costs),
        cache_miss_tokens=cache_miss_tokens,
        by_purpose=purposes,
        by_provider_model=provider_models,
    )


def usage_records_from_entries(entries: Iterable[SessionEntry]) -> tuple[UsageRecord, ...]:
    records: list[UsageRecord] = []
    for entry in entries:
        if not isinstance(entry, CustomEntry):
            continue
        if entry.namespace != USAGE_NAMESPACE:
            continue
        record = UsageRecord.from_data(entry.data)
        if record is not None:
            records.append(record)
    return tuple(records)


def cache_miss_tokens(entries: Sequence[SessionEntry] | Sequence[UsageRecord]) -> int | None:
    """Compute Pi-shaped cache misses for consecutive explicit cache facts.

    The durable structural entries, rather than a usage purpose, reset the
    chain.  Only agent calls participate; internal helper calls are ignored.
    LangChain's total ``input_tokens`` is already the prompt total and is not
    augmented with cache detail fields.
    """

    items = tuple(entries)
    legacy_records = bool(items) and isinstance(items[0], UsageRecord)
    previous: UsageRecord | None = None
    misses = 0
    observed = False
    for item in items:
        record: UsageRecord | None
        if legacy_records:
            record = cast(UsageRecord, item)
            if record.purpose in {"compaction", "branch_summary"}:
                previous = None
        else:
            if isinstance(item, (CompactionEntry, BranchSummaryEntry)):
                previous = None
                continue
            if not isinstance(item, CustomEntry) or item.namespace != USAGE_NAMESPACE:
                continue
            record = UsageRecord.from_data(item.data)
            if record is None:
                continue
        if record is None or record.purpose != "agent":
            continue
        # A cache write-only fact cannot tell us how much of the previous
        # prompt was reused.  Keep the field unknown and skip that call.
        if record.input_tokens is None or record.cache_read_tokens is None:
            previous = None
            continue
        if previous is not None:
            previous_prompt = previous.input_tokens
            current_prompt = record.input_tokens
            current_read = record.cache_read_tokens
            if previous_prompt is not None and current_read is not None:
                miss = max(0, min(previous_prompt, current_prompt) - current_read)
                if miss > CACHE_MISS_NOISE_FLOOR:
                    misses += miss
                    observed = True
        previous = record
    return misses if observed else None


def aggregate_usage_entries(entries: Iterable[SessionEntry]) -> UsageTotals:
    items = tuple(entries)
    records = usage_records_from_entries(items)
    return _totals_for_records(records, cache_miss_tokens=cache_miss_tokens(items))


__all__ = [
    "CACHE_MISS_NOISE_FLOOR",
    "USAGE_NAMESPACE",
    "UsageNormalization",
    "UsagePurpose",
    "UsageRecord",
    "UsageTotals",
    "aggregate_usage_entries",
    "cache_miss_tokens",
    "merge_stream_metadata",
    "usage_record_from_message",
    "usage_records_from_entries",
]
