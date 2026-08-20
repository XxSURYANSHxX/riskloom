from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

from riskloom.simulation.config import GeneratorConfig
from riskloom.simulation.generation import GeneratedRecord
from riskloom.simulation.label_schema import ScenarioType, SplitName


def fixed_decimal_ratio(numerator: int, denominator: int, places: int = 6) -> str:
    if type(places) is not int or not 0 <= places <= 12:
        raise ValueError("fixed_decimal_places_out_of_range")
    if denominator <= 0:
        return "0." + "0" * places
    scale = 10**places
    scaled = (numerator * scale + denominator // 2) // denominator
    return f"{scaled // scale}.{scaled % scale:0{places}d}"


def _nearest_rank(values: list[int], percentile: int) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    rank = max(1, (percentile * len(ordered) + 99) // 100)
    return ordered[rank - 1]


def _integer_distribution(values: Iterable[int]) -> dict[str, int]:
    collected = list(values)
    if not collected:
        return {"count": 0, "max": 0, "min": 0, "p50": 0, "p95": 0}
    return {
        "count": len(collected),
        "max": max(collected),
        "min": min(collected),
        "p50": _nearest_rank(collected, 50),
        "p95": _nearest_rank(collected, 95),
    }


def _sorted_counts(values: Iterable[str]) -> dict[str, int]:
    counts = Counter(values)
    return {key: counts[key] for key in sorted(counts)}


def _reuse_summary(records: list[GeneratedRecord], field: str) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for record in records:
        value = getattr(record.event, field)
        if value is not None:
            counts[str(value)] += 1
    return {
        "events_per_token": _integer_distribution(counts.values()),
        "unique_tokens": len(counts),
    }


def _attack_token_summary(records: list[GeneratedRecord], field: str) -> dict[str, Any]:
    values = [getattr(record.event, field) for record in records]
    non_missing = [str(value) for value in values if value is not None]
    unique_count = len(set(non_missing))
    return {
        "attack_events_per_unique_token": fixed_decimal_ratio(len(records), unique_count),
        "non_missing_events": len(non_missing),
        "unique_tokens": unique_count,
        "unique_tokens_per_attack_event": fixed_decimal_ratio(unique_count, len(records)),
    }


def build_report(
    records: list[GeneratedRecord],
    dataset_id: str,
    config: GeneratorConfig,
) -> dict[str, Any]:
    split_records: dict[SplitName, list[GeneratedRecord]] = defaultdict(list)
    for record in records:
        split_records[record.label.split].append(record)

    splits: dict[str, Any] = {}
    for split in SplitName:
        rows = split_records[split]
        attack_count = sum(record.label.is_attack for record in rows)
        splits[split.value] = {
            "attack": {
                "denominator": len(rows),
                "numerator": attack_count,
                "prevalence": fixed_decimal_ratio(attack_count, len(rows)),
            },
            "event_count": len(rows),
            "failure_categories": _sorted_counts(
                record.event.failure_category.value
                for record in rows
                if record.event.failure_category is not None
            ),
            "outcomes": _sorted_counts(record.event.outcome.value for record in rows),
            "scenarios": _sorted_counts(record.label.scenario_type.value for record in rows),
        }

    amounts = [record.event.amount_subunits for record in records]
    campaign_sizes = Counter(
        record.label.campaign_id for record in records if record.label.campaign_id is not None
    )
    merchant_counts = Counter(record.event.merchant_id for record in records)

    retry_groups: dict[str, list[GeneratedRecord]] = defaultdict(list)
    for record in records:
        if record.label.scenario_type is ScenarioType.LEGITIMATE_RETRY:
            instance_id = record.label.generator_metadata.scenario_instance_id
            if instance_id is not None:
                retry_groups[instance_id].append(record)
    successful_retries = sum(
        any(record.event.outcome.value == "authorized" for record in group)
        for group in retry_groups.values()
    )

    attack_entity_reuse: dict[str, Any] = {}
    for split in SplitName:
        attacks = [record for record in split_records[split] if record.label.is_attack]
        attack_entity_reuse[split.value] = {
            "attack_events": len(attacks),
            "entities": {
                entity: _attack_token_summary(attacks, f"{entity}_token")
                for entity in sorted(("device", "network", "session"))
            },
        }

    total_attacks = sum(record.label.is_attack for record in records)
    return {
        "amount_subunits": {
            "count": len(amounts),
            "max": max(amounts, default=0),
            "min": min(amounts, default=0),
            "p50": _nearest_rank(amounts, 50),
            "p95": _nearest_rank(amounts, 95),
            "p99": _nearest_rank(amounts, 99),
            "sum": sum(amounts),
        },
        "attack": {
            "denominator": len(records),
            "numerator": total_attacks,
            "prevalence": fixed_decimal_ratio(total_attacks, len(records)),
        },
        "attack_entity_reuse": {
            key: attack_entity_reuse[key] for key in sorted(attack_entity_reuse)
        },
        "campaign_sizes": _integer_distribution(campaign_sizes.values()),
        "channels": _sorted_counts(record.event.channel.value for record in records),
        "controlled_test_shift_policy": config.controlled_test_shift.model_dump(mode="json"),
        "dataset_id": dataset_id,
        "entity_reuse": {
            field: _reuse_summary(records, field)
            for field in sorted(
                (
                    "checkout_id",
                    "customer_token",
                    "device_token",
                    "network_token",
                    "payment_instrument_token",
                    "session_token",
                )
            )
        },
        "event_count": len(records),
        "failure_categories": _sorted_counts(
            record.event.failure_category.value
            for record in records
            if record.event.failure_category is not None
        ),
        "merchant_event_counts": _integer_distribution(merchant_counts.values()),
        "outcomes": _sorted_counts(record.event.outcome.value for record in records),
        "retry_chains": {
            "chain_count": len(retry_groups),
            "chain_lengths": _integer_distribution(len(group) for group in retry_groups.values()),
            "eventual_success": {
                "denominator": len(retry_groups),
                "numerator": successful_retries,
                "rate": fixed_decimal_ratio(successful_retries, len(retry_groups)),
            },
        },
        "scenarios": _sorted_counts(record.label.scenario_type.value for record in records),
        "splits": {key: splits[key] for key in sorted(splits)},
    }
