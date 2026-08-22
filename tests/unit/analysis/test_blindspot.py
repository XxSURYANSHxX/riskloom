"""The live-serving blind-spot measurement path, and proof the offline path is untouched."""

from pathlib import Path

import pytest

from riskloom.analysis.blindspot import (
    ASSUMED_AUTHORIZED,
    TRUE_OUTCOMES,
    _assume_authorized,
    extract,
)
from riskloom.features.artifacts import canonical_json_bytes, file_metadata
from riskloom.features.config import FeatureConfig, load_feature_config
from riskloom.features.schema import FEATURE_NAMES
from riskloom.simulation.event_schema import (
    Channel,
    CheckoutAttemptEvent,
    FailureCategory,
    Outcome,
)

POLICY_VALIDATION = Path("artifacts/simulations/policy-validation")
DEVELOPMENT_FEATURES = Path("artifacts/features/development-v1.1.0-config-bound/features.jsonl")
LOCKED_FEATURES_SHA256 = "1e019b88017869b088d92500c8d795f69bac48732335e8465e62adf3d96ed1b7"

OUTCOME_DERIVED = tuple(name for name in FEATURE_NAMES if "failure" in name)


@pytest.fixture(scope="module")
def feature_config() -> FeatureConfig:
    return load_feature_config(Path("configs/features/default.json"))


def _event(index: int, *, failed: bool) -> CheckoutAttemptEvent:
    from datetime import UTC, datetime, timedelta  # noqa: PLC0415

    return CheckoutAttemptEvent(
        event_id=f"evt_{index:032x}",
        merchant_id="mrc_" + "1" * 32,
        occurred_at=datetime(2026, 3, 1, tzinfo=UTC) + timedelta(seconds=index),
        checkout_id=f"chk_{index:032x}",
        customer_token=None,
        device_token="dev_" + "2" * 32,
        network_token="net_" + "3" * 32,
        session_token="ses_" + "4" * 32,
        payment_instrument_token="pmt_" + "5" * 32,
        amount_subunits=25_000,
        currency="INR",
        outcome=Outcome.FAILED if failed else Outcome.AUTHORIZED,
        failure_category=FailureCategory.INSTRUMENT_DECLINED if failed else None,
        channel=Channel.WEB,
    )


def test_assume_authorized_changes_only_the_outcome_pair() -> None:
    failed = _event(1, failed=True)
    assumed = _assume_authorized(failed)
    original = failed.model_dump(mode="json")
    adapted = assumed.model_dump(mode="json")
    assert {key for key in original if original[key] != adapted[key]} == {
        "outcome",
        "failure_category",
    }
    assert assumed.outcome is Outcome.AUTHORIZED
    assert assumed.failure_category is None


def test_assume_authorized_leaves_an_already_authorized_event_untouched() -> None:
    authorized = _event(2, failed=False)
    assert _assume_authorized(authorized) is authorized


def test_unsupported_mode_is_refused(tmp_path: Path, feature_config: FeatureConfig) -> None:
    events = tmp_path / "events.jsonl"
    labels = tmp_path / "labels.jsonl"
    events.write_bytes(b"")
    labels.write_bytes(b"")
    with pytest.raises(ValueError, match="unsupported"):
        extract(events, labels, feature_config, mode="something_else")


def _write_pair(tmp_path: Path, failures: tuple[int, ...]) -> tuple[Path, Path]:
    events = tmp_path / "events.jsonl"
    labels = tmp_path / "labels.jsonl"
    event_lines = []
    label_lines = []
    for index in range(12):
        event = _event(index + 1, failed=index in failures)
        event_lines.append(canonical_json_bytes(event.model_dump(mode="json")))
        label_lines.append(
            canonical_json_bytes({"event_id": event.event_id, "is_attack": index % 5 == 0})
        )
    events.write_bytes(b"".join(event_lines))
    labels.write_bytes(b"".join(label_lines))
    return events, labels


def test_the_two_modes_differ_only_in_the_failure_derived_features(
    tmp_path: Path, feature_config: FeatureConfig
) -> None:
    events, labels = _write_pair(tmp_path, failures=(1, 4, 7, 10))
    true_pair = extract(events, labels, feature_config, mode=TRUE_OUTCOMES)
    assumed_pair = extract(events, labels, feature_config, mode=ASSUMED_AUTHORIZED)

    assert true_pair.targets.tolist() == assumed_pair.targets.tolist()
    assert true_pair.feature_matrix.shape == assumed_pair.feature_matrix.shape

    diverged: set[str] = set()
    for row_index in range(true_pair.feature_matrix.shape[0]):
        for column, name in enumerate(FEATURE_NAMES):
            if (
                true_pair.feature_matrix[row_index][column]
                != (assumed_pair.feature_matrix[row_index][column])
            ):
                diverged.add(name)
    assert diverged
    assert diverged.issubset(set(OUTCOME_DERIVED))


def test_modes_agree_exactly_when_no_event_failed(
    tmp_path: Path, feature_config: FeatureConfig
) -> None:
    events, labels = _write_pair(tmp_path, failures=())
    true_pair = extract(events, labels, feature_config, mode=TRUE_OUTCOMES)
    assumed_pair = extract(events, labels, feature_config, mode=ASSUMED_AUTHORIZED)
    assert true_pair.feature_matrix.tolist() == assumed_pair.feature_matrix.tolist()


def test_misaligned_event_and_label_streams_are_refused(
    tmp_path: Path, feature_config: FeatureConfig
) -> None:
    events, labels = _write_pair(tmp_path, failures=())
    labels.write_bytes(
        canonical_json_bytes({"event_id": "evt_" + "f" * 32, "is_attack": False}) * 12
    )
    with pytest.raises(ValueError, match="not aligned"):
        extract(events, labels, feature_config, mode=TRUE_OUTCOMES)


@pytest.mark.skipif(
    not DEVELOPMENT_FEATURES.exists(),
    reason="locked feature artifacts are Git-ignored and absent in this checkout",
)
def test_locked_development_feature_dataset_hash_is_unchanged() -> None:
    """Day 6 must not have perturbed the offline extraction path in any way.

    If anything about feature computation, ordering or serialisation had shifted, this hash would
    move. It is the same value pinned in the modeling source contract.
    """

    assert file_metadata(DEVELOPMENT_FEATURES).sha256 == LOCKED_FEATURES_SHA256


@pytest.mark.skipif(
    not (POLICY_VALIDATION / "events.jsonl").exists(),
    reason="policy-validation batch is Git-ignored and absent in this checkout",
)
def test_true_outcome_mode_reproduces_the_committed_offline_extraction(
    feature_config: FeatureConfig,
) -> None:
    """Cross-check: the analysis path is not a re-derivation with its own quirks.

    Running the true-outcome mode over the policy-validation batch must reproduce the feature rows
    already published for it by the ordinary offline extraction.
    """

    import json  # noqa: PLC0415

    pair = extract(
        POLICY_VALIDATION / "events.jsonl",
        POLICY_VALIDATION / "labels.jsonl",
        feature_config,
        mode=TRUE_OUTCOMES,
    )
    published = Path("artifacts/features/policy-validation/features.jsonl")
    rows = [json.loads(line) for line in published.read_bytes().splitlines() if line.strip()]
    assert len(rows) == pair.feature_matrix.shape[0]
    for row_index, row in enumerate(rows[:200]):
        values = row["features"]
        for column, name in enumerate(FEATURE_NAMES):
            assert pair.feature_matrix[row_index][column] == values[name], (row_index, name)
