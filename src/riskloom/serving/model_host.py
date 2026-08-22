"""Startup-validated binding between the running service and the locked Day 4 artifacts.

This module deliberately imports only ``riskloom.modeling.artifacts``, ``.config`` and ``.model``.
It must never import ``riskloom.modeling.training``, which reaches ``riskloom.policy.bands`` and
would break the guarantee that no Gate C1 policy band is reachable from a live decision.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from riskloom.features.config import (
    FEATURE_ENGINE_VERSION,
    FEATURE_SCHEMA_VERSION,
    FeatureConfig,
    load_feature_config,
)
from riskloom.features.schema import FEATURE_COUNT, FEATURE_NAMES, FeatureRecord
from riskloom.modeling.artifacts import load_locked_model
from riskloom.modeling.canonical import file_metadata
from riskloom.modeling.config import load_modeling_config
from riskloom.modeling.model import LockedModel, portable_probabilities


class ServingBundleError(RuntimeError):
    """The service cannot bind safely to the locked artifacts, so it must not start."""


@dataclass(frozen=True, slots=True)
class ServingBundle:
    """Everything a live decision needs, proven consistent at startup."""

    model: LockedModel
    feature_config: FeatureConfig
    feature_dataset_id: str

    @property
    def decision_threshold(self) -> float:
        """The full float64 threshold as loaded from model.json.

        This is the only value the decision comparison may use. It is never read back from the
        Numeric(20, 18) audit column, which exists for storage and audit alone.
        """

        return self.model.decision_threshold

    def probability(self, record: FeatureRecord) -> float:
        """Score one feature record through unmodified portable JSON inference."""

        values = record.features.model_dump()
        matrix = np.asarray([[values[name] for name in FEATURE_NAMES]], dtype=np.float64)
        scored = portable_probabilities(self.model, matrix)
        probability = float(scored[0])
        if not np.isfinite(probability):
            raise ServingBundleError("serving_probability_not_finite")
        return probability


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ServingBundleError("serving_feature_manifest_unreadable") from None
    if not isinstance(value, dict):
        raise ServingBundleError("serving_feature_manifest_invalid")
    return value


def load_serving_bundle(
    *,
    feature_config_path: Path,
    modeling_config_path: Path,
    model_directory: Path,
    feature_manifest_path: Path,
) -> ServingBundle:
    """Bind to the locked artifacts, or refuse to start.

    The chain proven here is unbroken: the running engine's feature configuration equals the
    effective configuration recorded in the feature manifest; that manifest's own hash equals the
    one pinned in the modeling configuration the model was trained under; and the locked model's
    feature order equals the live feature schema. Nominal agreement is not accepted anywhere.
    """

    try:
        feature_config = load_feature_config(feature_config_path)
        modeling_config = load_modeling_config(modeling_config_path)
    except ValueError:
        raise ServingBundleError("serving_configuration_invalid") from None

    contract = modeling_config.source_contract
    manifest = _read_manifest(feature_manifest_path)

    observed_manifest_sha256 = str(file_metadata(feature_manifest_path)["sha256"])
    if observed_manifest_sha256 != contract.feature_manifest_sha256:
        raise ServingBundleError("serving_feature_manifest_hash_mismatch")

    if feature_config.model_dump(mode="json") != manifest.get("effective_configuration"):
        raise ServingBundleError("serving_feature_configuration_mismatch")

    version_checks = (
        (manifest.get("feature_engine_version") == FEATURE_ENGINE_VERSION, "engine_version"),
        (manifest.get("feature_schema_version") == FEATURE_SCHEMA_VERSION, "schema_version"),
        (manifest.get("feature_count") == FEATURE_COUNT, "feature_count"),
        (
            manifest.get("feature_schema_version") == contract.feature_schema_version,
            "contract_schema_version",
        ),
        (
            manifest.get("feature_engine_version") == contract.feature_engine_version,
            "contract_engine_version",
        ),
        (
            manifest.get("feature_dataset_id") == contract.feature_dataset_id,
            "feature_dataset_id",
        ),
    )
    for valid, name in version_checks:
        if not valid:
            raise ServingBundleError(f"serving_feature_{name}_mismatch")

    try:
        model, _, _ = load_locked_model(model_directory, modeling_config)
    except ValueError:
        raise ServingBundleError("serving_locked_model_invalid") from None

    # LockedModel already enforces this, but the live path restates it: a model whose feature
    # order differed from the running schema would silently score permuted inputs.
    if model.feature_order != list(FEATURE_NAMES):
        raise ServingBundleError("serving_feature_order_mismatch")

    return ServingBundle(
        model=model,
        feature_config=feature_config,
        feature_dataset_id=contract.feature_dataset_id,
    )
