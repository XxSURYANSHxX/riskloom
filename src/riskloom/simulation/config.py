import json
from datetime import datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from riskloom.simulation.identifiers import SIMULATION_ALGORITHM_VERSION
from riskloom.simulation.label_schema import CampaignProfile, SplitName

GENERATOR_VERSION = SIMULATION_ALGORITHM_VERSION


class ScenarioWeights(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    normal: int = Field(default=7_000, ge=0, le=10_000)
    legitimate_retry: int = Field(default=800, ge=0, le=10_000)
    flash_sale: int = Field(default=1_200, ge=0, le=10_000)
    shared_infrastructure: int = Field(default=500, ge=0, le=10_000)
    legitimate_failure: int = Field(default=300, ge=0, le=10_000)
    attack: int = Field(default=200, ge=0, le=10_000)

    @model_validator(mode="after")
    def require_complete_distribution(self) -> Self:
        expected = {
            "normal": 7_000,
            "legitimate_retry": 800,
            "flash_sale": 1_200,
            "shared_infrastructure": 500,
            "legitimate_failure": 300,
            "attack": 200,
        }
        if self.model_dump() != expected:
            raise ValueError("scenario weights must match the approved distribution")
        return self


class ChannelWeights(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    web: int = Field(default=5_000, ge=0, le=10_000)
    mobile_web: int = Field(default=3_000, ge=0, le=10_000)
    mobile_app: int = Field(default=2_000, ge=0, le=10_000)

    @model_validator(mode="after")
    def require_complete_distribution(self) -> Self:
        if sum(self.model_dump().values()) != 10_000:
            raise ValueError("channel weights must sum to 10000 basis points")
        return self


class FailureWeights(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    authentication: int = Field(default=3_000, ge=0, le=10_000)
    insufficient_funds: int = Field(default=2_500, ge=0, le=10_000)
    instrument_declined: int = Field(default=2_500, ge=0, le=10_000)
    temporary_processing: int = Field(default=1_500, ge=0, le=10_000)
    unknown: int = Field(default=500, ge=0, le=10_000)

    @model_validator(mode="after")
    def require_complete_distribution(self) -> Self:
        if sum(self.model_dump().values()) != 10_000:
            raise ValueError("failure weights must sum to 10000 basis points")
        return self


class MissingnessRates(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    customer: int = Field(default=500, ge=0, le=10_000)
    device: int = Field(default=300, ge=0, le=10_000)
    network: int = Field(default=200, ge=0, le=10_000)


class OutcomeRates(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    normal_failure: int = Field(default=800, ge=0, le=10_000)
    flash_sale_failure: int = Field(default=1_200, ge=0, le=10_000)
    shared_infrastructure_failure: int = Field(default=1_500, ge=0, le=10_000)
    attack_failure: int = Field(default=7_500, ge=0, le=10_000)
    retry_eventual_success: int = Field(default=7_500, ge=0, le=10_000)


class RetryBounds(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    minimum_attempts: int = Field(default=2, ge=2, le=4)
    maximum_attempts: int = Field(default=4, ge=2, le=4)
    minimum_gap_seconds: int = Field(default=20, ge=1, le=600)
    maximum_gap_seconds: int = Field(default=180, ge=1, le=600)

    @model_validator(mode="after")
    def validate_ranges(self) -> Self:
        if self.minimum_attempts > self.maximum_attempts:
            raise ValueError("minimum retry attempts must not exceed maximum")
        if self.minimum_gap_seconds > self.maximum_gap_seconds:
            raise ValueError("minimum retry gap must not exceed maximum")
        return self


class EntityPools(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    customers: int = Field(ge=1, le=1_000_000)
    devices: int = Field(ge=1, le=1_000_000)
    networks: int = Field(ge=1, le=1_000_000)
    instruments: int = Field(ge=1, le=1_000_000)


class ControlledTestShift(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    minimum_unique_entity_ratio_multiplier: int = Field(default=2, ge=2, le=10)
    maximum_unique_network_ratio_basis_points: int = Field(default=5_000, ge=1, le=5_000)
    minimum_network_presence_basis_points: int = Field(default=9_000, ge=5_000, le=10_000)


class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    name: SplitName
    duration_days: int = Field(ge=1, le=365)
    event_count: int = Field(ge=100, le=10_000_000, multiple_of=100)
    campaign_count: int = Field(ge=1, le=10_000)
    campaign_profile: CampaignProfile


class GeneratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    config_schema_version: Literal["1.0.0"]
    dataset_profile: Literal["smoke", "development"]
    start_at: datetime
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    merchant_count: int = Field(ge=2, le=10_000)
    entity_pools: EntityPools
    controlled_test_shift: ControlledTestShift = Field(default_factory=ControlledTestShift)
    splits: list[SplitConfig] = Field(min_length=3, max_length=3)
    scenario_weights: ScenarioWeights = Field(default_factory=ScenarioWeights)
    channel_weights: ChannelWeights = Field(default_factory=ChannelWeights)
    failure_weights: FailureWeights = Field(default_factory=FailureWeights)
    missingness_rates: MissingnessRates = Field(default_factory=MissingnessRates)
    outcome_rates: OutcomeRates = Field(default_factory=OutcomeRates)
    retry_bounds: RetryBounds = Field(default_factory=RetryBounds)
    amount_minimum_subunits: int = Field(default=5_000, ge=1, le=100_000_000)
    amount_maximum_subunits: int = Field(default=2_000_000, ge=1, le=100_000_000)
    merchant_catalog_points: int = Field(default=12, ge=2, le=100)

    @model_validator(mode="after")
    def validate_configuration(self) -> Self:
        utc_offset = self.start_at.utcoffset()
        if utc_offset is None or utc_offset.total_seconds() != 0:
            raise ValueError("start_at must be timezone-aware UTC")
        if self.start_at.microsecond:
            raise ValueError("start_at must use whole-second precision")
        if self.amount_minimum_subunits >= self.amount_maximum_subunits:
            raise ValueError("amount minimum must be below maximum")
        expected_splits = [SplitName.TRAIN, SplitName.CALIBRATION, SplitName.TEST]
        if [split.name for split in self.splits] != expected_splits:
            raise ValueError("splits must be ordered train, calibration, test")
        weights = self.scenario_weights.model_dump()
        for split in self.splits:
            for weight in weights.values():
                if split.event_count * weight % 10_000:
                    raise ValueError("split event counts must allocate every scenario exactly")
            attack_count = split.event_count * self.scenario_weights.attack // 10_000
            if split.campaign_count > attack_count:
                raise ValueError("campaign count must not exceed attack event count")
            if attack_count < 2 * split.campaign_count:
                raise ValueError("every campaign must receive at least two events")
            retry_count = split.event_count * self.scenario_weights.legitimate_retry // 10_000
            if 0 < retry_count < self.retry_bounds.minimum_attempts:
                raise ValueError("retry quota cannot form a complete retry chain")
            expected_profile = (
                CampaignProfile.ENTITY_REUSE_SHIFT
                if split.name is SplitName.TEST
                else CampaignProfile.BASELINE_REUSE
            )
            if split.campaign_profile is not expected_profile:
                raise ValueError("campaign profile does not match split isolation policy")
        if self.missingness_rates.network > (
            10_000 - self.controlled_test_shift.minimum_network_presence_basis_points
        ):
            raise ValueError("network missingness conflicts with controlled test-shift policy")
        return self

    @property
    def total_events(self) -> int:
        return sum(split.event_count for split in self.splits)

    def scenario_counts(self, split: SplitConfig) -> dict[str, int]:
        return {
            name: split.event_count * weight // 10_000
            for name, weight in sorted(self.scenario_weights.model_dump().items())
        }


def load_generator_config(path: Path) -> GeneratorConfig:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("simulation_configuration_unreadable") from None
    return GeneratorConfig.model_validate(raw)
