import json
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

FEATURE_CONFIG_SCHEMA_VERSION = "1.0.0"
FEATURE_ENGINE_VERSION = "1.0.0"
FEATURE_SCHEMA_VERSION = "1.0.0"


class FeatureConfigurationError(ValueError):
    """Safe feature-configuration error without configuration contents."""


class FeatureConfig(BaseModel):
    """The locked Day 3 feature configuration."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True, frozen=True)

    config_schema_version: Literal["1.0.0"]
    feature_schema_version: Literal["1.0.0"]
    rolling_windows_seconds: tuple[int, int, int]
    checkout_history_window_seconds: int
    failure_rate_window_seconds: int

    @model_validator(mode="after")
    def validate_locked_configuration(self) -> Self:
        if self.rolling_windows_seconds != (60, 300, 3_600):
            raise ValueError("rolling windows must be exactly 60, 300, and 3600 seconds")
        if self.checkout_history_window_seconds != 3_600:
            raise ValueError("checkout history window must be exactly 3600 seconds")
        if self.failure_rate_window_seconds != 300:
            raise ValueError("failure-rate window must be exactly 300 seconds")
        return self


def load_feature_config(path: Path) -> FeatureConfig:
    try:
        if path.stat().st_size > 65_536:
            raise FeatureConfigurationError("feature_configuration_oversized")
        raw = json.loads(path.read_text(encoding="utf-8"))
        return FeatureConfig.model_validate(raw)
    except FeatureConfigurationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        raise FeatureConfigurationError("feature_configuration_invalid") from None
