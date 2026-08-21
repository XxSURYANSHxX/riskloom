import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from riskloom.simulation.identifiers import (
    SIMULATION_ALGORITHM_VERSION,
    effective_configuration_fingerprint,
)
from riskloom.simulation.label_schema import CampaignProfile, SplitName

GENERATOR_VERSION = SIMULATION_ALGORITHM_VERSION

SMOKE_MAX_TOTAL_EVENTS = 3_000
SMOKE_MAX_EVENTS_PER_SPLIT = 1_400
SMOKE_MAX_TOTAL_ATTACK_EVENTS = 60
SMOKE_MAX_ATTACK_EVENTS_PER_SPLIT = 28
SMOKE_MAX_TOTAL_DURATION_DAYS = 6
SMOKE_MAX_DURATION_DAYS_PER_SPLIT = 4
SMOKE_MAX_CAMPAIGNS_PER_SPLIT = 10
SMOKE_MAX_PLACEMENT_CAMPAIGNS_PER_SIDE = 5
SMOKE_MAX_PLACEMENT_SAMPLING_ATTEMPTS = 4_096


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


class CampaignPlacement(BaseModel):
    """Locked irregular campaign-placement constraints for a protected time boundary."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True, frozen=True)

    protected_boundary_basis_points: int = Field(ge=1, le=9_999)
    minimum_campaigns_before_boundary: int = Field(ge=2, le=10_000)
    minimum_campaigns_after_boundary: int = Field(ge=2, le=10_000)
    minimum_gap_seconds: int = Field(ge=300, le=3_600)
    maximum_sampling_attempts_per_campaign: int = Field(ge=1, le=100_000)


class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    name: SplitName
    duration_days: int = Field(ge=1, le=365)
    event_count: int = Field(ge=100, le=10_000_000, multiple_of=100)
    campaign_count: int = Field(ge=1, le=10_000)
    campaign_profile: CampaignProfile
    campaign_placement: CampaignPlacement | None = None


class GeneratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    config_schema_version: Literal["1.0.0", "1.1.0"]
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
        validate_profile_contract(self)
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
            placement = split.campaign_placement
            if placement is not None:
                assigned_campaigns = (
                    placement.minimum_campaigns_before_boundary
                    + placement.minimum_campaigns_after_boundary
                )
                if assigned_campaigns != split.campaign_count:
                    raise ValueError("placement sides must allocate every campaign exactly")
                if attack_count % split.campaign_count:
                    raise ValueError("placed campaigns must receive equal event quotas")
                split_duration_ms = split.duration_days * 86_400_000
                boundary_offset_ms = (
                    split_duration_ms * placement.protected_boundary_basis_points // 10_000
                )
                maximum_campaign_duration_ms = 90 * 60 * 1_000
                minimum_gap_ms = placement.minimum_gap_seconds * 1_000
                before_required_ms = (
                    placement.minimum_campaigns_before_boundary * maximum_campaign_duration_ms
                    + (placement.minimum_campaigns_before_boundary - 1) * minimum_gap_ms
                )
                after_required_ms = (
                    placement.minimum_campaigns_after_boundary * maximum_campaign_duration_ms
                    + (placement.minimum_campaigns_after_boundary - 1) * minimum_gap_ms
                )
                if before_required_ms > boundary_offset_ms or after_required_ms > (
                    split_duration_ms - boundary_offset_ms
                ):
                    raise ValueError("campaign placement cannot fit inside protected intervals")
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

    def _validate_locked_development_contract(self) -> None:
        train, calibration, test = self.splits
        placement = calibration.campaign_placement
        checks = (
            (train.duration_days == 20, "development_contract_train_duration"),
            (train.event_count == 66_000, "development_contract_train_event_count"),
            (train.campaign_count == 8, "development_contract_train_campaign_count"),
            (
                train.campaign_profile is CampaignProfile.BASELINE_REUSE,
                "development_contract_train_campaign_profile",
            ),
            (calibration.duration_days == 5, "development_contract_calibration_duration"),
            (
                calibration.event_count == 17_000,
                "development_contract_calibration_event_count",
            ),
            (
                self.scenario_counts(calibration)["attack"] == 340,
                "development_contract_calibration_attack_count",
            ),
            (
                calibration.campaign_count == 10,
                "development_contract_calibration_campaign_count",
            ),
            (
                self.scenario_counts(calibration)["attack"] // calibration.campaign_count == 34,
                "development_contract_calibration_campaign_size",
            ),
            (
                calibration.campaign_profile is CampaignProfile.BASELINE_REUSE,
                "development_contract_calibration_campaign_profile",
            ),
            (placement is not None, "development_contract_calibration_placement"),
            (
                placement is not None and placement.protected_boundary_basis_points == 6_000,
                "development_contract_boundary",
            ),
            (
                placement is not None and placement.minimum_campaigns_before_boundary == 5,
                "development_contract_campaigns_before",
            ),
            (
                placement is not None and placement.minimum_campaigns_after_boundary == 5,
                "development_contract_campaigns_after",
            ),
            (
                placement is not None and placement.minimum_gap_seconds == 300,
                "development_contract_campaign_gap",
            ),
            (
                placement is not None and placement.maximum_sampling_attempts_per_campaign == 4_096,
                "development_contract_sampling_attempts",
            ),
            (test.duration_days == 5, "development_contract_test_duration"),
            (test.event_count == 17_000, "development_contract_test_event_count"),
            (test.campaign_count == 3, "development_contract_test_campaign_count"),
            (
                test.campaign_profile is CampaignProfile.ENTITY_REUSE_SHIFT,
                "development_contract_test_campaign_profile",
            ),
            (
                self.controlled_test_shift.minimum_unique_entity_ratio_multiplier == 2,
                "development_contract_shift_multiplier",
            ),
            (
                self.controlled_test_shift.maximum_unique_network_ratio_basis_points == 5_000,
                "development_contract_shift_network_ratio",
            ),
            (
                self.controlled_test_shift.minimum_network_presence_basis_points == 9_000,
                "development_contract_shift_network_presence",
            ),
        )
        for valid, error in checks:
            if not valid:
                raise ValueError(error)

    @property
    def total_events(self) -> int:
        return sum(split.event_count for split in self.splits)

    def scenario_counts(self, split: SplitConfig) -> dict[str, int]:
        return {
            name: split.event_count * weight // 10_000
            for name, weight in sorted(self.scenario_weights.model_dump().items())
        }


def validate_profile_contract(config: GeneratorConfig) -> None:
    """Fail closed over the complete supported version and profile matrix."""

    version = config.config_schema_version
    profile = config.dataset_profile
    if version not in ("1.0.0", "1.1.0"):
        raise ValueError("unsupported_configuration_version")
    if profile not in ("development", "smoke"):
        raise ValueError("unsupported_dataset_profile")
    if len(config.splits) != 3:
        raise ValueError("profile_contract_split_structure_invalid")

    calibration = config.splits[1]
    placements = [split for split in config.splits if split.campaign_placement is not None]
    if version == "1.0.0":
        if placements:
            raise ValueError("campaign placement requires configuration schema 1.1.0")
        if profile == "development":
            return
        if profile == "smoke":
            return
        raise ValueError("unsupported_version_profile_combination")

    if placements != [calibration]:
        raise ValueError("configuration schema 1.1.0 requires calibration placement only")
    if profile == "development":
        config._validate_locked_development_contract()
        return
    if profile != "smoke":
        raise ValueError("unsupported_version_profile_combination")

    total_attack_events = sum(config.scenario_counts(split)["attack"] for split in config.splits)
    checks = (
        (config.total_events <= SMOKE_MAX_TOTAL_EVENTS, "smoke_contract_total_event_count"),
        (
            total_attack_events <= SMOKE_MAX_TOTAL_ATTACK_EVENTS,
            "smoke_contract_total_attack_count",
        ),
    )
    for valid, error in checks:
        if not valid:
            raise ValueError(error)

    for split in config.splits:
        split_checks = (
            (
                split.event_count <= SMOKE_MAX_EVENTS_PER_SPLIT,
                "smoke_contract_split_event_count",
            ),
            (
                config.scenario_counts(split)["attack"] <= SMOKE_MAX_ATTACK_EVENTS_PER_SPLIT,
                "smoke_contract_split_attack_count",
            ),
            (
                split.duration_days <= SMOKE_MAX_DURATION_DAYS_PER_SPLIT,
                "smoke_contract_split_duration",
            ),
            (
                split.campaign_count <= SMOKE_MAX_CAMPAIGNS_PER_SPLIT,
                "smoke_contract_campaign_count",
            ),
        )
        for valid, error in split_checks:
            if not valid:
                raise ValueError(error)

        placement = split.campaign_placement
        if placement is None:
            continue
        placement_checks = (
            (
                placement.minimum_campaigns_before_boundary
                <= SMOKE_MAX_PLACEMENT_CAMPAIGNS_PER_SIDE,
                "smoke_contract_campaigns_before_boundary",
            ),
            (
                placement.minimum_campaigns_after_boundary
                <= SMOKE_MAX_PLACEMENT_CAMPAIGNS_PER_SIDE,
                "smoke_contract_campaigns_after_boundary",
            ),
            (
                placement.maximum_sampling_attempts_per_campaign
                <= SMOKE_MAX_PLACEMENT_SAMPLING_ATTEMPTS,
                "smoke_contract_sampling_attempts",
            ),
        )
        for valid, error in placement_checks:
            if not valid:
                raise ValueError(error)

    if sum(split.duration_days for split in config.splits) > SMOKE_MAX_TOTAL_DURATION_DAYS:
        raise ValueError("smoke_contract_total_duration")


def validated_configuration_snapshot(config: GeneratorConfig) -> GeneratorConfig:
    """Rebuild an isolated strict snapshot before any generation-side operation."""

    raw = config.model_dump(mode="python")
    snapshot = GeneratorConfig.model_validate(raw, strict=True)
    validate_profile_contract(snapshot)
    return snapshot


def generator_version_for_config(config: GeneratorConfig) -> str:
    """Map each supported configuration schema to its generation algorithm."""

    return config.config_schema_version


def effective_configuration(config: GeneratorConfig) -> dict[str, Any]:
    """Serialize configuration exactly as its declared algorithm expects."""

    result = config.model_dump(mode="json")
    if config.config_schema_version == "1.0.0":
        for split in result["splits"]:
            split.pop("campaign_placement", None)
    return result


def configuration_fingerprint(config: GeneratorConfig) -> str | None:
    """Return the canonical namespace fingerprint used only by algorithm 1.1.0."""

    if config.config_schema_version == "1.0.0":
        return None
    return effective_configuration_fingerprint(effective_configuration(config))


def boundary_timestamp(
    start: datetime,
    end: datetime,
    boundary_basis_points: int,
) -> datetime:
    """Return the shared integer-millisecond timestamp boundary."""

    if not 1 <= boundary_basis_points <= 9_999:
        raise ValueError("boundary basis points must be between 1 and 9999")
    duration = end - start
    duration_ms = (
        duration.days * 86_400_000 + duration.seconds * 1_000 + duration.microseconds // 1_000
    )
    if duration_ms <= 0:
        raise ValueError("boundary duration must be positive")
    offset_ms = duration_ms * boundary_basis_points // 10_000
    return start + timedelta(milliseconds=offset_ms)


def load_generator_config(path: Path) -> GeneratorConfig:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("simulation_configuration_unreadable") from None
    config = GeneratorConfig.model_validate(raw)
    validate_profile_contract(config)
    return config
