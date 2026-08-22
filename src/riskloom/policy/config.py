import json
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

POLICY_CONFIG_SCHEMA_VERSION = "1.0.0"
POLICY_BAND_SCHEMA_VERSION = "1.0.0"
POLICY_COMPARISON_SCHEMA_VERSION = "1.0.0"


class PolicyConfigurationError(ValueError):
    """A safe, non-data-bearing policy configuration error."""


class PolicyConfig(BaseModel):
    """Abstract, transparent policy weights and approval gates.

    Every cost below is an abstract unit, exactly as framed in the Day 4 modeling configuration.
    None of them is a rupee-denominated claim, an estimate of merchant loss, or a statement about
    real chargeback economics. They exist only to make the trade-off between missing an attack,
    wrongly denying a legitimate customer, and sending an event to manual review explicit and
    auditable.

    ``false_negative_cost_units`` and ``false_positive_cost_units`` are inherited unchanged from
    the locked Day 4 configuration so that the old single-threshold policy and the new banded
    policy are scored on exactly the same function.

    ``review_cost_units`` is the new Day 5 weight. It is an operational cost incurred for every
    reviewed event regardless of whether that event turns out to be fraud, which is why it is a
    separate term rather than something folded into the false-positive cost. The default of 3 says
    a review costs meaningfully more than taking no action at all and meaningfully less than
    wrongly denying a legitimate customer outright, which at the inherited 1:25 ratio sits nearer
    the false-positive end of the scale. It is a policy choice, not a measurement.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True, strict=True)

    config_schema_version: Literal["1.0.0"] = "1.0.0"
    false_negative_cost_units: int = Field(ge=1)
    false_positive_cost_units: int = Field(ge=0)
    review_cost_units: int = Field(ge=0)
    # Merchant-set ceiling on the denied-legitimate rate under the new policy, in basis points.
    # Configurable per merchant risk appetite; 100 bp (1.00%) is the documented default and is
    # deliberately above the 60.02 bp the locked single-threshold policy produced in Gate B2 so
    # that the ceiling constrains regressions rather than encoding the current model's result.
    maximum_false_positive_rate_basis_points: int = Field(ge=1, le=10_000)
    # Minimum evidence required before an old-versus-new comparison is allowed to be trusted.
    minimum_validation_rows: int = Field(ge=1)
    minimum_validation_attacks: int = Field(ge=1)
    # Upper bound on distinct sweep candidates per threshold. The sweep is over observed
    # probabilities, so this bounds the pair count, not the resolution near zero.
    threshold_grid_size: int = Field(ge=8, le=4_096)

    @model_validator(mode="after")
    def validate_inherited_day_four_cost_ratio(self) -> Self:
        """Preserve the locked Day 4 false-positive to false-negative ratio of 1:25.

        The ratio is the invariant, not the absolute magnitudes. Both policies in a comparison are
        always scored with the same weights, so a uniform rescaling changes no ranking between
        them. Rescaling is meaningful only because ``review_cost_units`` is an integer: on the
        original 1:25 scale a false positive costs 1 unit, so no integer review cost can sit
        strictly between doing nothing and denying, and the review tier is unreachable. Scaling the
        pair up leaves room for a review cost that is genuinely between the two.
        """

        if self.false_positive_cost_units < 1:
            raise ValueError("policy false positive cost must be at least 1 unit")
        if self.false_negative_cost_units != 25 * self.false_positive_cost_units:
            raise ValueError("policy costs must inherit the locked 1:25 Day 4 cost policy")
        return self


def load_policy_config(path: Path) -> PolicyConfig:
    try:
        raw = path.read_bytes()
        if b"\r" in raw or not raw.endswith(b"\n"):
            raise PolicyConfigurationError("policy_configuration_not_canonical")
        config = PolicyConfig.model_validate_json(raw, strict=True)
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
            raise PolicyConfigurationError("policy_configuration_not_canonical")
        return config
    except PolicyConfigurationError:
        raise
    except (OSError, UnicodeError, ValueError):
        raise PolicyConfigurationError("policy_configuration_invalid") from None
