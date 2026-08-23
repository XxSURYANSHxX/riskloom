import json
from datetime import UTC, datetime, timedelta
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

# Locked contract for the Day 5 counterfactual policy-validation batch. This profile exists only to
# produce a small, chronologically disjoint dataset that no Day 4 fitting partition has ever seen.
# It is deliberately NOT "development" so it can never be substituted for the approved training or
# held-out data, and every value below is pinned so the batch cannot be quietly reshaped.
POLICY_VALIDATION_START_AT = "2026-03-01T00:00:00Z"
POLICY_VALIDATION_TRAIN_DURATION_DAYS = 5
POLICY_VALIDATION_TRAIN_EVENT_COUNT = 5_000
POLICY_VALIDATION_TRAIN_CAMPAIGN_COUNT = 5
POLICY_VALIDATION_CALIBRATION_DURATION_DAYS = 3
POLICY_VALIDATION_CALIBRATION_EVENT_COUNT = 2_000
POLICY_VALIDATION_CALIBRATION_CAMPAIGN_COUNT = 4
POLICY_VALIDATION_TEST_DURATION_DAYS = 3
POLICY_VALIDATION_TEST_EVENT_COUNT = 2_000
POLICY_VALIDATION_TEST_CAMPAIGN_COUNT = 2
POLICY_VALIDATION_BOUNDARY_BASIS_POINTS = 5_000
POLICY_VALIDATION_CAMPAIGNS_PER_SIDE = 2
POLICY_VALIDATION_MINIMUM_GAP_SECONDS = 300
POLICY_VALIDATION_SAMPLING_ATTEMPTS = 4_096
POLICY_VALIDATION_TOTAL_EVENTS = 9_000

# Locked contract for the Gate H0 adversarial stress batch. This profile exists only to score the
# already-locked Day 4 model against attack traffic shaped to evade the mechanisms it relies on. It
# is deliberately neither "development" nor "policy-validation" so it can never be substituted for
# a fitting partition or the counterfactual batch, and it is the only profile permitted to carry an
# evasion shape at all.
# The evasion shape applies to the TEST split only, mirroring how campaign placement applies to
# calibration only. This is forced by two existing dataset invariants rather than chosen for
# convenience: `test_entity_rotation_shift_missing` requires the test split to carry at least twice
# the entity-uniqueness ratio of train and calibration, and `attack_network_coordination_too_sparse`
# caps unique attack networks at half the attack count. Diluting every split would violate both, so
# the baseline splits stay baseline and the measurement is taken on the test split's attacks.
ADVERSARIAL_START_AT = "2026-06-01T00:00:00Z"
ADVERSARIAL_TRAIN_DURATION_DAYS = 2
ADVERSARIAL_TRAIN_EVENT_COUNT = 2_000
ADVERSARIAL_TRAIN_CAMPAIGN_COUNT = 2
ADVERSARIAL_CALIBRATION_DURATION_DAYS = 2
ADVERSARIAL_CALIBRATION_EVENT_COUNT = 1_000
ADVERSARIAL_CALIBRATION_CAMPAIGN_COUNT = 4
ADVERSARIAL_TEST_DURATION_DAYS = 4
ADVERSARIAL_TEST_EVENT_COUNT = 6_000
ADVERSARIAL_TEST_CAMPAIGN_COUNT = 20
ADVERSARIAL_TOTAL_EVENTS = 9_000
ADVERSARIAL_BOUNDARY_BASIS_POINTS = 5_000
ADVERSARIAL_CAMPAIGNS_PER_SIDE = 2
ADVERSARIAL_MINIMUM_GAP_SECONDS = 300
ADVERSARIAL_SAMPLING_ATTEMPTS = 4_096


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


class EvasionShape(BaseModel):
    """How an adversarial campaign is reshaped to evade a specific detection mechanism.

    Available only to configuration schema 1.2.0 under the ``adversarial-stress`` profile. Every
    variant leaves the attack *volume* untouched and changes only its shape, so a drop in detection
    is attributable to the evasion rather than to there being less to detect.

    Each variant names the mechanism it attacks:

    ``slow_and_low``
        Stretches the campaign over ``duration_minutes`` instead of the usual 30-90, so per-window
        attempt counts stay near one. Targets the 17 ``*_60s`` and 22 ``*_300s`` velocity features.

    ``window_edge``
        Places events at exactly ``edge_window_seconds`` apart so each prior event lands precisely
        on the expiry cutoff. The engine's windows are ``(current_time - window, current_time]``,
        left-exclusive, so an event exactly one window back is already expired. Targets that rule.

    ``distributed_thin``
        Gives each event its own device and spreads the campaign across ``network_count`` networks
        instead of one. Targets the 14 device and 17 network reuse features. Instruments are
        already unique per event in the base generator, so there is nothing to dilute there.

    ``failure_camouflage``
        Overrides the campaign failure rate to ``failure_rate_basis_points``, modelling an attacker
        using pre-validated cards. Targets all 18 failure-derived features at once.
    """

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True, frozen=True, strict=True)

    variant: Literal["slow_and_low", "window_edge", "distributed_thin", "failure_camouflage"]
    duration_minutes: int | None = Field(default=None, ge=91, le=10_080)
    edge_window_seconds: int | None = Field(default=None)
    network_count: int | None = Field(default=None, ge=2, le=512)
    failure_rate_basis_points: int | None = Field(default=None, ge=0, le=10_000)

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        """Each variant requires its own parameter and forbids the others.

        Fail closed rather than ignoring a stray field: a config that sets ``network_count`` on a
        ``slow_and_low`` variant is not expressing an intent this generator can honour, and
        silently dropping it would make the recorded configuration a poor description of what ran.
        """

        required = {
            "slow_and_low": "duration_minutes",
            "window_edge": "edge_window_seconds",
            "distributed_thin": "network_count",
            "failure_camouflage": "failure_rate_basis_points",
        }[self.variant]
        provided = {
            name
            for name in (
                "duration_minutes",
                "edge_window_seconds",
                "network_count",
                "failure_rate_basis_points",
            )
            if getattr(self, name) is not None
        }
        if provided != {required}:
            raise ValueError("evasion_shape_parameters_do_not_match_variant")
        if self.variant == "window_edge" and self.edge_window_seconds not in (60, 300, 3_600):
            raise ValueError("evasion_shape_edge_window_must_be_a_feature_window")
        return self


class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    name: SplitName
    duration_days: int = Field(ge=1, le=365)
    event_count: int = Field(ge=100, le=10_000_000, multiple_of=100)
    campaign_count: int = Field(ge=1, le=10_000)
    campaign_profile: CampaignProfile
    campaign_placement: CampaignPlacement | None = None
    evasion_shape: EvasionShape | None = None


class GeneratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    config_schema_version: Literal["1.0.0", "1.1.0", "1.2.0"]
    dataset_profile: Literal["smoke", "development", "policy-validation", "adversarial-stress"]
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

    def _validate_locked_policy_validation_contract(self) -> None:
        train, calibration, test = self.splits
        placement = calibration.campaign_placement
        checks = (
            (
                self.start_at == datetime(2026, 3, 1, tzinfo=UTC),
                "policy_validation_contract_start_at",
            ),
            (train.duration_days == 5, "policy_validation_contract_train_duration"),
            (train.event_count == 5_000, "policy_validation_contract_train_event_count"),
            (train.campaign_count == 5, "policy_validation_contract_train_campaign_count"),
            (
                train.campaign_profile is CampaignProfile.BASELINE_REUSE,
                "policy_validation_contract_train_campaign_profile",
            ),
            (calibration.duration_days == 3, "policy_validation_contract_calibration_duration"),
            (
                calibration.event_count == 2_000,
                "policy_validation_contract_calibration_event_count",
            ),
            (
                calibration.campaign_count == 4,
                "policy_validation_contract_calibration_campaign_count",
            ),
            (
                calibration.campaign_profile is CampaignProfile.BASELINE_REUSE,
                "policy_validation_contract_calibration_campaign_profile",
            ),
            (placement is not None, "policy_validation_contract_calibration_placement"),
            (
                placement is not None and placement.protected_boundary_basis_points == 5_000,
                "policy_validation_contract_boundary",
            ),
            (
                placement is not None and placement.minimum_campaigns_before_boundary == 2,
                "policy_validation_contract_campaigns_before",
            ),
            (
                placement is not None and placement.minimum_campaigns_after_boundary == 2,
                "policy_validation_contract_campaigns_after",
            ),
            (
                placement is not None and placement.minimum_gap_seconds == 300,
                "policy_validation_contract_campaign_gap",
            ),
            (
                placement is not None and placement.maximum_sampling_attempts_per_campaign == 4_096,
                "policy_validation_contract_sampling_attempts",
            ),
            (test.duration_days == 3, "policy_validation_contract_test_duration"),
            (test.event_count == 2_000, "policy_validation_contract_test_event_count"),
            (test.campaign_count == 2, "policy_validation_contract_test_campaign_count"),
            (
                test.campaign_profile is CampaignProfile.ENTITY_REUSE_SHIFT,
                "policy_validation_contract_test_campaign_profile",
            ),
            (
                self.total_events == POLICY_VALIDATION_TOTAL_EVENTS,
                "policy_validation_contract_total_event_count",
            ),
        )
        for valid, error in checks:
            if not valid:
                raise ValueError(error)

    def _validate_locked_adversarial_contract(self) -> None:
        """Pin the Gate H0 batch so it cannot be quietly reshaped between runs.

        The scale, window and campaign structure are fixed here rather than left to the config file
        so that a reported degradation is attributable to the evasion shape alone. Attack weight is
        deliberately left at the standard 200 bp: raising prevalence would inflate precision and
        depress false-positive rate against the Gate B2 reference and make the comparison
        meaningless.
        """

        train, calibration, test = self.splits
        placement = calibration.campaign_placement
        shapes = [split.evasion_shape for split in self.splits]
        checks = (
            (
                self.start_at == datetime.fromisoformat(ADVERSARIAL_START_AT),
                "adversarial_contract_start_at",
            ),
            (self.total_events == ADVERSARIAL_TOTAL_EVENTS, "adversarial_contract_total_events"),
            (
                train.duration_days == ADVERSARIAL_TRAIN_DURATION_DAYS
                and train.event_count == ADVERSARIAL_TRAIN_EVENT_COUNT
                and train.campaign_count == ADVERSARIAL_TRAIN_CAMPAIGN_COUNT,
                "adversarial_contract_train_shape",
            ),
            (
                calibration.duration_days == ADVERSARIAL_CALIBRATION_DURATION_DAYS
                and calibration.event_count == ADVERSARIAL_CALIBRATION_EVENT_COUNT
                and calibration.campaign_count == ADVERSARIAL_CALIBRATION_CAMPAIGN_COUNT,
                "adversarial_contract_calibration_shape",
            ),
            (
                test.duration_days == ADVERSARIAL_TEST_DURATION_DAYS
                and test.event_count == ADVERSARIAL_TEST_EVENT_COUNT
                and test.campaign_count == ADVERSARIAL_TEST_CAMPAIGN_COUNT,
                "adversarial_contract_test_shape",
            ),
            (self.scenario_weights.attack == 200, "adversarial_contract_attack_weight"),
            (
                shapes[0] is None and shapes[1] is None,
                "adversarial_contract_baseline_splits_must_stay_baseline",
            ),
            (shapes[2] is not None, "adversarial_contract_test_evasion_required"),
            (placement is not None, "adversarial_contract_calibration_placement"),
            (
                placement is not None
                and placement.protected_boundary_basis_points == ADVERSARIAL_BOUNDARY_BASIS_POINTS,
                "adversarial_contract_boundary",
            ),
            (
                placement is not None
                and placement.minimum_campaigns_before_boundary == ADVERSARIAL_CAMPAIGNS_PER_SIDE,
                "adversarial_contract_campaigns_before",
            ),
            (
                placement is not None
                and placement.minimum_campaigns_after_boundary == ADVERSARIAL_CAMPAIGNS_PER_SIDE,
                "adversarial_contract_campaigns_after",
            ),
            (
                placement is not None
                and placement.minimum_gap_seconds == ADVERSARIAL_MINIMUM_GAP_SECONDS,
                "adversarial_contract_campaign_gap",
            ),
            (
                placement is not None
                and placement.maximum_sampling_attempts_per_campaign
                == ADVERSARIAL_SAMPLING_ATTEMPTS,
                "adversarial_contract_sampling_attempts",
            ),
            (
                train.campaign_profile is CampaignProfile.BASELINE_REUSE,
                "adversarial_contract_train_campaign_profile",
            ),
            (
                test.campaign_profile is CampaignProfile.ENTITY_REUSE_SHIFT,
                "adversarial_contract_test_campaign_profile",
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
    if version not in ("1.0.0", "1.1.0", "1.2.0"):
        raise ValueError("unsupported_configuration_version")
    if profile not in ("development", "smoke", "policy-validation", "adversarial-stress"):
        raise ValueError("unsupported_dataset_profile")

    # Two independent protections, not one. The evasion field is rejected outright outside schema
    # 1.2.0, and effective_configuration additionally removes it from any older canonical
    # configuration, so neither a mislabelled profile nor a mislabelled version can reach the
    # locked datasets.
    carries_evasion = any(split.evasion_shape is not None for split in config.splits)
    if version != "1.2.0" and carries_evasion:
        raise ValueError("evasion_shape_requires_configuration_schema_1_2_0")
    if profile != "adversarial-stress" and carries_evasion:
        raise ValueError("evasion_shape_requires_the_adversarial_stress_profile")
    if version == "1.2.0" and profile != "adversarial-stress":
        raise ValueError("configuration_schema_1_2_0_is_reserved_for_adversarial_stress")
    if profile == "adversarial-stress" and version != "1.2.0":
        raise ValueError("adversarial_stress_requires_configuration_schema_1_2_0")
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
    if profile == "policy-validation":
        config._validate_locked_policy_validation_contract()
        return
    if profile == "adversarial-stress":
        config._validate_locked_adversarial_contract()
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
    """Map each supported configuration schema to its generation algorithm.

    Schema 1.2.0 deliberately maps to the **1.1.0** algorithm rather than announcing a new one.
    Identifier construction, PRNG stream derivation and fingerprint binding are byte-for-byte what
    1.1.0 does; the only addition is a conditional campaign shape that no other profile can select.
    Claiming a new algorithm version would assert a change to the derivation that did not happen,
    and the configuration fingerprint already distinguishes an adversarial dataset from every other.
    """

    if config.config_schema_version == "1.2.0":
        return SIMULATION_ALGORITHM_VERSION
    return config.config_schema_version


def effective_configuration(config: GeneratorConfig) -> dict[str, Any]:
    """Serialize configuration exactly as its declared algorithm expects."""

    result = config.model_dump(mode="json")
    if config.config_schema_version == "1.0.0":
        for split in result["splits"]:
            split.pop("campaign_placement", None)
    if config.config_schema_version != "1.2.0":
        # The evasion field must never enter an older algorithm's canonical configuration. Every
        # PRNG stream in schema 1.1.0 is namespaced by this fingerprint, so a stray "evasion_shape":
        # null would change every identifier and every drawn value in the development and
        # policy-validation datasets -- including the one the locked Day 4 model was trained on.
        # Removing it here mirrors exactly how 1.0.0 removes campaign_placement above.
        for split in result["splits"]:
            split.pop("evasion_shape", None)
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
