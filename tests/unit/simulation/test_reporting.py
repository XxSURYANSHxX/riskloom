import json
from pathlib import Path

import pytest

from riskloom.simulation.config import GeneratorConfig
from riskloom.simulation.generation import GeneratedRecord
from riskloom.simulation.reporting import build_report, fixed_decimal_ratio
from riskloom.simulation.validation import read_records


def test_report_recomputes_exactly_and_uses_fixed_rates(
    tiny_output: tuple[Path, object],
) -> None:
    output, _ = tiny_output
    records, report, manifest = read_records(output)
    config = GeneratorConfig.model_validate(manifest["effective_configuration"])
    assert report == build_report(records, manifest["dataset_id"], config)
    assert report["attack"] == {
        "denominator": 300,
        "numerator": 6,
        "prevalence": "0.020000",
    }
    assert not _contains_float(report)
    assert list(report["splits"]) == sorted(report["splits"])
    parsed = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert parsed == report


def test_fixed_decimal_ratio_uses_integer_arithmetic() -> None:
    assert fixed_decimal_ratio(1, 3) == "0.333333"
    assert fixed_decimal_ratio(2, 3) == "0.666667"
    assert fixed_decimal_ratio(0, 0) == "0.000000"
    with pytest.raises(ValueError, match="places_out_of_range"):
        fixed_decimal_ratio(1, 2, -1)


def test_entity_reuse_report_has_no_raw_identifiers(
    tiny_records: list[GeneratedRecord],
    tiny_config: GeneratorConfig,
) -> None:
    report_text = json.dumps(build_report(tiny_records, "a" * 64, tiny_config), sort_keys=True)
    assert "evt_" not in report_text
    assert "mrc_" not in report_text
    assert "cmp_" not in report_text


def test_controlled_shift_report_uses_coherent_directional_metrics(
    tiny_records: list[GeneratedRecord],
    tiny_config: GeneratorConfig,
) -> None:
    report = build_report(tiny_records, "a" * 64, tiny_config)
    assert report["controlled_test_shift_policy"] == {
        "maximum_unique_network_ratio_basis_points": 5_000,
        "minimum_network_presence_basis_points": 9_000,
        "minimum_unique_entity_ratio_multiplier": 2,
    }
    test = report["attack_entity_reuse"]["test"]["entities"]
    calibration = report["attack_entity_reuse"]["calibration"]["entities"]
    for entity in ("device", "session"):
        test_unique = test[entity]["unique_tokens"]
        test_events = report["attack_entity_reuse"]["test"]["attack_events"]
        calibration_unique = calibration[entity]["unique_tokens"]
        calibration_events = report["attack_entity_reuse"]["calibration"]["attack_events"]
        assert test_unique * calibration_events >= 2 * calibration_unique * test_events
        assert test_events * calibration_unique <= calibration_events * test_unique
        assert test[entity]["unique_tokens_per_attack_event"] == fixed_decimal_ratio(
            test_unique, test_events
        )


def _contains_float(value: object) -> bool:
    if isinstance(value, float):
        return True
    if isinstance(value, dict):
        return any(_contains_float(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_float(item) for item in value)
    return False
