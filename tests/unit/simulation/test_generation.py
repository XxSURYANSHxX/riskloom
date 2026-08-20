import random
from collections import Counter

import pytest
from pydantic import ValidationError

from riskloom.simulation.config import GeneratorConfig
from riskloom.simulation.event_schema import CheckoutAttemptEvent
from riskloom.simulation.generation import GeneratedRecord, generate_records
from riskloom.simulation.identifiers import deterministic_identifier, random_stream
from riskloom.simulation.label_schema import ScenarioType, SplitName


def test_exact_quota_order_join_and_event_uniqueness(
    tiny_config: GeneratorConfig,
    tiny_records: list[GeneratedRecord],
) -> None:
    assert len(tiny_records) == 300
    assert len({record.event.event_id for record in tiny_records}) == 300
    assert [record.event.event_id for record in tiny_records] == [
        record.label.event_id for record in tiny_records
    ]
    ordering = [(record.event.occurred_at, record.event.event_id) for record in tiny_records]
    assert ordering == sorted(ordering)

    expected = {
        ScenarioType.NORMAL: 70,
        ScenarioType.LEGITIMATE_RETRY: 8,
        ScenarioType.FLASH_SALE: 12,
        ScenarioType.SHARED_INFRASTRUCTURE: 5,
        ScenarioType.LEGITIMATE_FAILURE: 3,
        ScenarioType.CARD_TESTING_CAMPAIGN: 2,
    }
    for split in SplitName:
        actual = Counter(
            record.label.scenario_type for record in tiny_records if record.label.split is split
        )
        assert actual == Counter(expected)
        assert sum(actual.values()) == 100

    assert tiny_config.total_events == 300


def test_entity_identifiers_are_formatted_and_intentionally_reused(
    tiny_records: list[GeneratedRecord],
) -> None:
    reusable_fields = (
        "merchant_id",
        "checkout_id",
        "customer_token",
        "device_token",
        "network_token",
        "session_token",
        "payment_instrument_token",
    )
    for field in reusable_fields:
        non_null = [
            getattr(record.event, field) for record in tiny_records if getattr(record.event, field)
        ]
        assert len(set(non_null)) < len(non_null), field
        assert all(len(value.rsplit("_", 1)[1]) == 32 for value in non_null)
        assert all(value.rsplit("_", 1)[1] == value.rsplit("_", 1)[1].lower() for value in non_null)


def test_retry_chains_consume_exact_quota_and_reuse_identity(
    tiny_records: list[GeneratedRecord],
) -> None:
    retries = [
        record
        for record in tiny_records
        if record.label.scenario_type is ScenarioType.LEGITIMATE_RETRY
    ]
    assert len(retries) == 24
    instances: dict[str, list[GeneratedRecord]] = {}
    for record in retries:
        instance_id = record.label.generator_metadata.scenario_instance_id
        assert instance_id is not None
        instances.setdefault(instance_id, []).append(record)
    assert len({group[0].event.checkout_id for group in instances.values()}) == len(instances)
    for group in instances.values():
        assert 2 <= len(group) <= 4
        assert len({record.event.checkout_id for record in group}) == 1
        assert len({record.event.session_token for record in group}) == 1
        assert (
            min(group, key=lambda record: record.event.occurred_at).event.outcome.value == "failed"
        )


def test_hard_negatives_and_shift_exist_without_model_visible_marker(
    tiny_config: GeneratorConfig,
    tiny_records: list[GeneratedRecord],
) -> None:
    hard_negative_types = {
        ScenarioType.LEGITIMATE_RETRY,
        ScenarioType.FLASH_SALE,
        ScenarioType.SHARED_INFRASTRUCTURE,
        ScenarioType.LEGITIMATE_FAILURE,
    }
    assert hard_negative_types.issubset({record.label.scenario_type for record in tiny_records})
    shifted = [
        record
        for record in tiny_records
        if record.label.is_attack and record.label.split is SplitName.TEST
    ]
    multiplier = tiny_config.controlled_test_shift.minimum_unique_entity_ratio_multiplier
    for field in ("device_token", "session_token"):
        shifted_unique = len(
            {
                getattr(record.event, field)
                for record in shifted
                if getattr(record.event, field) is not None
            }
        )
        for split in (SplitName.TRAIN, SplitName.CALIBRATION):
            baseline = [
                record
                for record in tiny_records
                if record.label.is_attack and record.label.split is split
            ]
            baseline_unique = len(
                {
                    getattr(record.event, field)
                    for record in baseline
                    if getattr(record.event, field) is not None
                }
            )
            assert shifted_unique * len(baseline) >= (multiplier * baseline_unique * len(shifted))
            assert len(shifted) * baseline_unique <= len(baseline) * shifted_unique
    for split in SplitName:
        attacks = [
            record
            for record in tiny_records
            if record.label.is_attack and record.label.split is split
        ]
        networks = [record.event.network_token for record in attacks if record.event.network_token]
        assert len(set(networks)) * 10_000 <= (
            tiny_config.controlled_test_shift.maximum_unique_network_ratio_basis_points
            * len(attacks)
        )
        assert len(networks) * 10_000 >= (
            tiny_config.controlled_test_shift.minimum_network_presence_basis_points * len(attacks)
        )
    for record in tiny_records:
        assert not {
            "campaign_id",
            "generator_metadata",
            "is_attack",
            "scenario_type",
            "split",
        }.intersection(record.event.model_dump())


def test_campaigns_consume_quota_and_span_merchants(
    tiny_config: GeneratorConfig,
    tiny_records: list[GeneratedRecord],
) -> None:
    for split in tiny_config.splits:
        attacks = [
            record
            for record in tiny_records
            if record.label.is_attack and record.label.split is split.name
        ]
        campaigns: dict[str, list[GeneratedRecord]] = {}
        for record in attacks:
            assert record.label.campaign_id is not None
            campaigns.setdefault(record.label.campaign_id, []).append(record)
        assert len(campaigns) == split.campaign_count
        assert sum(map(len, campaigns.values())) == tiny_config.scenario_counts(split)["attack"]
        assert all(group for group in campaigns.values())
        assert all(
            len(group) < 2 or len({record.event.merchant_id for record in group}) >= 2
            for group in campaigns.values()
        )


def test_outcome_failure_category_invariant_is_bidirectional(
    tiny_records: list[GeneratedRecord],
) -> None:
    authorized = next(
        record.event for record in tiny_records if record.event.outcome.value == "authorized"
    )
    failed = next(record.event for record in tiny_records if record.event.outcome.value == "failed")
    authorized_data = authorized.model_dump(mode="json")
    authorized_data["failure_category"] = "unknown"
    with pytest.raises(ValidationError):
        CheckoutAttemptEvent.model_validate(authorized_data)
    failed_data = failed.model_dump(mode="json")
    failed_data["failure_category"] = None
    with pytest.raises(ValidationError):
        CheckoutAttemptEvent.model_validate(failed_data)


def test_event_and_label_schemas_reject_unknown_free_form_fields(
    tiny_records: list[GeneratedRecord],
) -> None:
    event_data = tiny_records[0].event.model_dump(mode="json")
    event_data["notes"] = "untrusted free-form value"
    with pytest.raises(ValidationError) as event_error:
        CheckoutAttemptEvent.model_validate(event_data)
    assert "untrusted free-form value" not in str(event_error.value)

    label_data = tiny_records[0].label.model_dump(mode="json")
    label_data["reviewer_comment"] = "untrusted free-form value"
    with pytest.raises(ValidationError) as label_error:
        type(tiny_records[0].label).model_validate(label_data)
    assert "untrusted free-form value" not in str(label_error.value)

    campaign = next(record.label for record in tiny_records if record.label.is_attack)
    campaign_data = campaign.model_dump(mode="json")
    campaign_data["is_attack"] = False
    campaign_data["campaign_id"] = None
    campaign_data["generator_metadata"]["campaign_profile"] = None
    with pytest.raises(ValidationError, match="must be attack labels"):
        type(campaign).model_validate(campaign_data)


def test_different_seeds_change_valid_events(tiny_config: GeneratorConfig) -> None:
    first = generate_records(tiny_config, 1)
    second = generate_records(tiny_config, 2)
    assert [record.event.event_id for record in first] != [
        record.event.event_id for record in second
    ]


def test_identifiers_and_prng_streams_ignore_global_random_state() -> None:
    random.seed(1)
    first = random_stream(7, "train", "normal").getrandbits(256)
    random.seed(2)
    second = random_stream(7, "train", "normal").getrandbits(256)
    assert first == second
    assert first != random_stream(7, "train", "campaign").getrandbits(256)
    assert first != random_stream(7, "test", "normal").getrandbits(256)

    identifier = deterministic_identifier("evt", 7, "event", "train", 1)
    assert identifier.startswith("evt_")
    assert len(identifier.removeprefix("evt_")) == 32
    assert identifier.removeprefix("evt_").isalnum()
    assert identifier == identifier.lower()
