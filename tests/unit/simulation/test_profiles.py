from pathlib import Path

from riskloom.simulation.config import load_generator_config


def test_builtin_profiles_have_exact_split_and_scenario_counts() -> None:
    repository_root = Path(__file__).parents[3]
    expectations = {
        "smoke.json": ([1_400, 300, 300], [4, 1, 1], 2_000),
        "development.json": ([66_000, 17_000, 17_000], [20, 5, 5], 100_000),
    }
    expected_percentages = {
        "attack": 2,
        "flash_sale": 12,
        "legitimate_failure": 3,
        "legitimate_retry": 8,
        "normal": 70,
        "shared_infrastructure": 5,
    }

    for filename, (event_counts, durations, total) in expectations.items():
        config = load_generator_config(repository_root / "configs/simulation" / filename)
        assert [split.event_count for split in config.splits] == event_counts
        assert [split.duration_days for split in config.splits] == durations
        assert config.total_events == total
        assert config.controlled_test_shift.model_dump() == {
            "maximum_unique_network_ratio_basis_points": 5_000,
            "minimum_network_presence_basis_points": 9_000,
            "minimum_unique_entity_ratio_multiplier": 2,
        }
        for split in config.splits:
            assert config.scenario_counts(split) == {
                scenario: split.event_count * percentage // 100
                for scenario, percentage in expected_percentages.items()
            }
