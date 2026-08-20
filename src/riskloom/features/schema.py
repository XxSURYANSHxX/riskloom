from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

from riskloom.simulation.event_schema import EventId


class FeatureVector(BaseModel):
    """The exact version 1.0.0 numerical feature schema."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True, strict=True, frozen=True)

    amount_subunits: int = Field(ge=1, le=100_000_000)
    utc_hour: int = Field(ge=0, le=23)
    utc_day_of_week: int = Field(ge=0, le=6)
    channel_web: int = Field(ge=0, le=1)
    channel_mobile_web: int = Field(ge=0, le=1)
    channel_mobile_app: int = Field(ge=0, le=1)
    customer_token_missing: int = Field(ge=0, le=1)
    device_token_missing: int = Field(ge=0, le=1)
    network_token_missing: int = Field(ge=0, le=1)

    checkout_prior_attempt_count_3600s: int = Field(ge=0)
    checkout_previous_attempt_age_ms: int = Field(ge=0, lt=3_600_000)
    checkout_history_present: int = Field(ge=0, le=1)

    merchant_prior_attempt_count_60s: int = Field(ge=0)
    merchant_prior_failure_count_60s: int = Field(ge=0)
    merchant_distinct_instruments_60s: int = Field(ge=0)
    merchant_distinct_devices_60s: int = Field(ge=0)
    merchant_distinct_networks_60s: int = Field(ge=0)
    merchant_prior_attempt_count_300s: int = Field(ge=0)
    merchant_prior_failure_count_300s: int = Field(ge=0)
    merchant_distinct_instruments_300s: int = Field(ge=0)
    merchant_distinct_devices_300s: int = Field(ge=0)
    merchant_distinct_networks_300s: int = Field(ge=0)
    merchant_prior_attempt_count_3600s: int = Field(ge=0)
    merchant_prior_failure_count_3600s: int = Field(ge=0)
    merchant_distinct_instruments_3600s: int = Field(ge=0)
    merchant_distinct_devices_3600s: int = Field(ge=0)
    merchant_distinct_networks_3600s: int = Field(ge=0)

    device_prior_attempt_count_60s: int = Field(ge=0)
    device_prior_failure_count_60s: int = Field(ge=0)
    device_distinct_instruments_60s: int = Field(ge=0)
    device_distinct_sessions_60s: int = Field(ge=0)
    device_prior_attempt_count_300s: int = Field(ge=0)
    device_prior_failure_count_300s: int = Field(ge=0)
    device_distinct_instruments_300s: int = Field(ge=0)
    device_distinct_sessions_300s: int = Field(ge=0)
    device_prior_attempt_count_3600s: int = Field(ge=0)
    device_prior_failure_count_3600s: int = Field(ge=0)
    device_distinct_instruments_3600s: int = Field(ge=0)
    device_distinct_sessions_3600s: int = Field(ge=0)

    network_prior_attempt_count_60s: int = Field(ge=0)
    network_prior_failure_count_60s: int = Field(ge=0)
    network_distinct_instruments_60s: int = Field(ge=0)
    network_distinct_devices_60s: int = Field(ge=0)
    network_distinct_sessions_60s: int = Field(ge=0)
    network_prior_attempt_count_300s: int = Field(ge=0)
    network_prior_failure_count_300s: int = Field(ge=0)
    network_distinct_instruments_300s: int = Field(ge=0)
    network_distinct_devices_300s: int = Field(ge=0)
    network_distinct_sessions_300s: int = Field(ge=0)
    network_prior_attempt_count_3600s: int = Field(ge=0)
    network_prior_failure_count_3600s: int = Field(ge=0)
    network_distinct_instruments_3600s: int = Field(ge=0)
    network_distinct_devices_3600s: int = Field(ge=0)
    network_distinct_sessions_3600s: int = Field(ge=0)

    instrument_prior_attempt_count_300s: int = Field(ge=0)
    instrument_prior_failure_count_300s: int = Field(ge=0)
    instrument_distinct_devices_300s: int = Field(ge=0)
    instrument_distinct_networks_300s: int = Field(ge=0)
    instrument_distinct_merchants_300s: int = Field(ge=0)
    instrument_prior_attempt_count_3600s: int = Field(ge=0)
    instrument_prior_failure_count_3600s: int = Field(ge=0)
    instrument_distinct_devices_3600s: int = Field(ge=0)
    instrument_distinct_networks_3600s: int = Field(ge=0)
    instrument_distinct_merchants_3600s: int = Field(ge=0)

    session_prior_attempt_count_60s: int = Field(ge=0)
    session_prior_failure_count_60s: int = Field(ge=0)
    session_distinct_instruments_60s: int = Field(ge=0)
    session_prior_attempt_count_300s: int = Field(ge=0)
    session_prior_failure_count_300s: int = Field(ge=0)
    session_distinct_instruments_300s: int = Field(ge=0)

    merchant_prior_failure_rate_300s_bp: int = Field(ge=0, le=10_000)
    device_prior_failure_rate_300s_bp: int = Field(ge=0, le=10_000)
    network_prior_failure_rate_300s_bp: int = Field(ge=0, le=10_000)
    instrument_prior_failure_rate_300s_bp: int = Field(ge=0, le=10_000)
    session_prior_failure_rate_300s_bp: int = Field(ge=0, le=10_000)

    @model_validator(mode="after")
    def validate_coherent_indicators(self) -> Self:
        if self.channel_web + self.channel_mobile_web + self.channel_mobile_app != 1:
            raise ValueError("exactly one channel feature must be active")
        has_history = self.checkout_prior_attempt_count_3600s > 0
        if has_history != bool(self.checkout_history_present):
            raise ValueError("checkout history indicator does not match prior count")
        if not has_history and self.checkout_previous_attempt_age_ms != 0:
            raise ValueError("checkout age must use zero sentinel without history")

        family_relations = {
            "merchant": ("instruments", "devices", "networks"),
            "device": ("instruments", "sessions"),
            "network": ("instruments", "devices", "sessions"),
            "instrument": ("devices", "networks", "merchants"),
            "session": ("instruments",),
        }
        family_windows = {
            "merchant": (60, 300, 3_600),
            "device": (60, 300, 3_600),
            "network": (60, 300, 3_600),
            "instrument": (300, 3_600),
            "session": (60, 300),
        }
        for family, windows in family_windows.items():
            previous: dict[str, int] | None = None
            for window in windows:
                attempts = getattr(self, f"{family}_prior_attempt_count_{window}s")
                failures = getattr(self, f"{family}_prior_failure_count_{window}s")
                if failures > attempts:
                    raise ValueError("failure count cannot exceed attempt count")
                current = {"attempts": attempts, "failures": failures}
                for relation in family_relations[family]:
                    distinct = getattr(self, f"{family}_distinct_{relation}_{window}s")
                    if distinct > attempts:
                        raise ValueError("distinct count cannot exceed attempt count")
                    current[relation] = distinct
                if previous is not None and any(current[name] < previous[name] for name in current):
                    raise ValueError("wider-window count cannot be smaller")
                previous = current

            attempts_300 = getattr(self, f"{family}_prior_attempt_count_300s")
            failures_300 = getattr(self, f"{family}_prior_failure_count_300s")
            expected_rate = failures_300 * 10_000 // attempts_300 if attempts_300 else 0
            if getattr(self, f"{family}_prior_failure_rate_300s_bp") != expected_rate:
                raise ValueError("failure rate does not match prior counts")

        for family, missing in (
            ("device", self.device_token_missing),
            ("network", self.network_token_missing),
        ):
            if missing and any(
                value
                for name, value in self.model_dump().items()
                if name.startswith((f"{family}_prior_", f"{family}_distinct_"))
            ):
                raise ValueError("missing primary token cannot have rolling history")
        return self


FEATURE_NAMES = tuple(sorted(FeatureVector.model_fields))
FEATURE_COUNT = len(FEATURE_NAMES)

if FEATURE_COUNT != 75:
    raise RuntimeError("feature_schema_must_have_exactly_75_fields")


class FeatureRecord(BaseModel):
    """Canonical feature artifact row with safe join metadata."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True, strict=True, frozen=True)

    event_id: EventId
    occurred_at: datetime
    features: FeatureVector

    @model_validator(mode="after")
    def validate_utc_millisecond_timestamp(self) -> Self:
        offset = self.occurred_at.utcoffset()
        if offset is None or offset.total_seconds() != 0:
            raise ValueError("occurred_at must be timezone-aware UTC")
        if self.occurred_at.microsecond % 1_000:
            raise ValueError("occurred_at must use millisecond precision")
        return self

    @field_serializer("occurred_at")
    def serialize_occurred_at(self, value: datetime) -> str:
        return value.strftime("%Y-%m-%dT%H:%M:%S.") + f"{value.microsecond // 1_000:03d}Z"
