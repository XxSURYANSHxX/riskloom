import json
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

MODEL_CONFIG_SCHEMA_VERSION = "1.0.0"
MODEL_SCHEMA_VERSION = "1.0.0"
TRAINING_REPORT_SCHEMA_VERSION = "1.0.0"
EVALUATION_SCHEMA_VERSION = "1.0.0"


class ModelingConfigurationError(ValueError):
    """A safe, non-data-bearing configuration error."""


class SourceContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True, strict=True)

    dataset_profile: Literal["development"] = "development"
    simulation_generator_version: Literal["1.1.0"] = "1.1.0"
    simulation_config_schema_version: Literal["1.1.0"] = "1.1.0"
    simulation_configuration_sha256: Literal[
        "140ecd643528fadc5583957a07a4ee50452b0cac5c3a92a20aa41431320ca42c"
    ]
    simulation_dataset_id: Literal[
        "0634addd42f766029e0ee6477f8e6d07f043f6c6a5820a5785e009d7176954bc"
    ]
    simulation_manifest_sha256: Literal[
        "2254263b2af89751aba11961a481554098df007476215919ebb8d4629d8fa495"
    ]
    events_sha256: Literal["d0cc608d7b5384f5dcc2faa5027e2265fb360f03c3e329d8ab45db685a0ff1c8"]
    labels_sha256: Literal["1d72bc06059f640e6734ecfba43d39d43b4eca6de7065037fbc0a9249e368cd1"]
    simulation_report_sha256: Literal[
        "39f7a3a564ded9d139cb14e780855d877bd586d6012101e42536840c3d286302"
    ]
    feature_dataset_id: Literal["4b1ec1e0f03bc8b691d7abf5351546953a6a721ff4ed405882820998f30cecb5"]
    feature_manifest_sha256: Literal[
        "15337d0e9220f7ca96b4ded8157bc3ad29f38a6c5db9d357dea00b09371f28ba"
    ]
    features_sha256: Literal["1e019b88017869b088d92500c8d795f69bac48732335e8465e62adf3d96ed1b7"]
    feature_report_sha256: Literal[
        "e9e84c04365e1fbf98b0e542c90e16ba82e40cc623ee4e003a7a4214c24def72"
    ]
    feature_engine_version: Literal["1.0.0"] = "1.0.0"
    feature_schema_version: Literal["1.0.0"] = "1.0.0"
    feature_count: Literal[75] = 75


class LogisticConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True, strict=True)

    C: float = Field(gt=0, allow_inf_nan=False)
    l1_ratio: float = Field(default=0.0, ge=0.0, le=0.0, allow_inf_nan=False)
    dual: Literal[False] = False
    tolerance: float = Field(gt=0, allow_inf_nan=False)
    fit_intercept: Literal[True] = True
    intercept_scaling: float = Field(default=1.0, ge=1.0, le=1.0, allow_inf_nan=False)
    solver: Literal["lbfgs"] = "lbfgs"
    maximum_iterations: int = Field(ge=1)
    random_state: int


class GradientBoostingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True, strict=True)

    loss: Literal["log_loss"] = "log_loss"
    learning_rate: float = Field(gt=0, allow_inf_nan=False)
    estimator_count: int = Field(ge=1)
    subsample: float = Field(default=1.0, ge=1.0, le=1.0, allow_inf_nan=False)
    minimum_samples_split: int = Field(ge=2)
    minimum_samples_leaf: int = Field(ge=1)
    maximum_depth: int = Field(ge=1)
    minimum_impurity_decrease: float = Field(default=0.0, ge=0.0, le=0.0, allow_inf_nan=False)
    maximum_features: Literal[None] = None
    maximum_leaf_nodes: Literal[None] = None
    tolerance: float = Field(gt=0, allow_inf_nan=False)
    random_state: int


class PlattConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True, strict=True)

    C: float = Field(gt=0, allow_inf_nan=False)
    tolerance: float = Field(gt=0, allow_inf_nan=False)
    maximum_iterations: int = Field(ge=1)
    solver: Literal["lbfgs"] = "lbfgs"
    random_state: int


class ModelingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True, strict=True)

    config_schema_version: Literal["1.0.0"] = "1.0.0"
    calibration_boundary_basis_points: Literal[6_000] = 6_000
    false_positive_cost_units: int = Field(ge=0)
    false_negative_cost_units: int = Field(gt=0)
    reliability_bin_count: int = Field(ge=2, le=100)
    probability_clip_epsilon: float = Field(gt=0, lt=0.01, allow_inf_nan=False)
    parity_absolute_tolerance: float = Field(gt=0, allow_inf_nan=False)
    parity_sample_rows: int = Field(ge=3, le=10_000)
    source_contract: SourceContract
    logistic_regression: LogisticConfig
    gradient_boosting: GradientBoostingConfig
    platt_calibration: PlattConfig

    @model_validator(mode="after")
    def validate_locked_policy(self) -> Self:
        if self.false_positive_cost_units != 1 or self.false_negative_cost_units != 25:
            raise ValueError("modeling cost policy must remain 1:25")
        if self.reliability_bin_count != 10:
            raise ValueError("modeling reliability bin count must remain 10")
        return self


def load_modeling_config(path: Path) -> ModelingConfig:
    try:
        raw = path.read_bytes()
        if b"\r" in raw or not raw.endswith(b"\n"):
            raise ModelingConfigurationError("modeling_configuration_not_canonical")
        config = ModelingConfig.model_validate_json(raw, strict=True)
        expected = (
            json.dumps(
                config.model_dump(mode="json"),
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
        if raw != expected:
            raise ModelingConfigurationError("modeling_configuration_not_canonical")
        return config
    except ModelingConfigurationError:
        raise
    except (OSError, UnicodeError, ValueError):
        raise ModelingConfigurationError("modeling_configuration_invalid") from None
