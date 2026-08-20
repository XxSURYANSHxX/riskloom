from datetime import UTC, datetime

from riskloom.features.config import FeatureConfig
from riskloom.features.schema import FeatureRecord, FeatureVector
from riskloom.features.state import EntitySnapshot, FeatureState
from riskloom.simulation.event_schema import Channel, CheckoutAttemptEvent, Outcome


class FeatureEngineInputError(ValueError):
    """Safe feature-engine input error."""


def _elapsed_milliseconds(current: datetime, previous: datetime) -> int:
    delta = current - previous
    return delta.days * 86_400_000 + delta.seconds * 1_000 + delta.microseconds // 1_000


def _failure_rate_basis_points(snapshot: EntitySnapshot) -> int:
    if snapshot.attempts == 0:
        return 0
    return snapshot.failures * 10_000 // snapshot.attempts


class FeatureEngine:
    """Deterministic compute-before-update temporal feature engine."""

    def __init__(self, config: FeatureConfig) -> None:
        self.config = config
        self._state = FeatureState()
        self._last_key: tuple[datetime, str] | None = None

    def process(self, event: CheckoutAttemptEvent) -> FeatureRecord:
        if type(event) is not CheckoutAttemptEvent:
            raise FeatureEngineInputError("feature_event_type_invalid")
        current_key = (event.occurred_at, event.event_id)
        if self._last_key is not None and current_key <= self._last_key:
            raise FeatureEngineInputError("feature_input_not_strictly_monotonic")

        eviction_checkpoint = self._state.evict(event.occurred_at)
        try:
            record = self._compute_record(event)
            self._state.observe(
                event.occurred_at,
                checkout=event.checkout_id,
                merchant=event.merchant_id,
                device=event.device_token,
                network=event.network_token,
                instrument=event.payment_instrument_token,
                session=event.session_token,
                failed=event.outcome is Outcome.FAILED,
            )
        except Exception:
            self._state.restore_eviction(eviction_checkpoint)
            raise
        self._last_key = current_key
        return record

    def _compute_record(self, event: CheckoutAttemptEvent) -> FeatureRecord:
        timestamp = event.occurred_at.astimezone(UTC)
        values: dict[str, int] = {
            "amount_subunits": event.amount_subunits,
            "utc_hour": timestamp.hour,
            "utc_day_of_week": timestamp.weekday(),
            "channel_web": int(event.channel is Channel.WEB),
            "channel_mobile_web": int(event.channel is Channel.MOBILE_WEB),
            "channel_mobile_app": int(event.channel is Channel.MOBILE_APP),
            "customer_token_missing": int(event.customer_token is None),
            "device_token_missing": int(event.device_token is None),
            "network_token_missing": int(event.network_token is None),
        }

        checkout = self._state.index("checkout_3600s").snapshot(event.checkout_id)
        values["checkout_prior_attempt_count_3600s"] = checkout.attempts
        values["checkout_history_present"] = int(checkout.attempts > 0)
        values["checkout_previous_attempt_age_ms"] = (
            _elapsed_milliseconds(event.occurred_at, checkout.last_occurred_at)
            if checkout.last_occurred_at is not None
            else 0
        )

        self._add_merchant_features(values, event)
        self._add_device_features(values, event)
        self._add_network_features(values, event)
        self._add_instrument_features(values, event)
        self._add_session_features(values, event)

        merchant_300 = self._state.index("merchant_300s").snapshot(event.merchant_id)
        device_300 = self._state.index("device_300s").snapshot(event.device_token)
        network_300 = self._state.index("network_300s").snapshot(event.network_token)
        instrument_300 = self._state.index("instrument_300s").snapshot(
            event.payment_instrument_token
        )
        session_300 = self._state.index("session_300s").snapshot(event.session_token)
        values.update(
            {
                "merchant_prior_failure_rate_300s_bp": _failure_rate_basis_points(merchant_300),
                "device_prior_failure_rate_300s_bp": _failure_rate_basis_points(device_300),
                "network_prior_failure_rate_300s_bp": _failure_rate_basis_points(network_300),
                "instrument_prior_failure_rate_300s_bp": _failure_rate_basis_points(instrument_300),
                "session_prior_failure_rate_300s_bp": _failure_rate_basis_points(session_300),
            }
        )
        vector = FeatureVector.model_validate(values)
        return FeatureRecord(
            event_id=event.event_id, occurred_at=event.occurred_at, features=vector
        )

    def _add_merchant_features(self, values: dict[str, int], event: CheckoutAttemptEvent) -> None:
        for window in (60, 300, 3_600):
            snapshot = self._state.index(f"merchant_{window}s").snapshot(event.merchant_id)
            values[f"merchant_prior_attempt_count_{window}s"] = snapshot.attempts
            values[f"merchant_prior_failure_count_{window}s"] = snapshot.failures
            values[f"merchant_distinct_instruments_{window}s"] = snapshot.distinct.get(
                "instruments", 0
            )
            values[f"merchant_distinct_devices_{window}s"] = snapshot.distinct.get("devices", 0)
            values[f"merchant_distinct_networks_{window}s"] = snapshot.distinct.get("networks", 0)

    def _add_device_features(self, values: dict[str, int], event: CheckoutAttemptEvent) -> None:
        for window in (60, 300, 3_600):
            snapshot = self._state.index(f"device_{window}s").snapshot(event.device_token)
            values[f"device_prior_attempt_count_{window}s"] = snapshot.attempts
            values[f"device_prior_failure_count_{window}s"] = snapshot.failures
            values[f"device_distinct_instruments_{window}s"] = snapshot.distinct.get(
                "instruments", 0
            )
            values[f"device_distinct_sessions_{window}s"] = snapshot.distinct.get("sessions", 0)

    def _add_network_features(self, values: dict[str, int], event: CheckoutAttemptEvent) -> None:
        for window in (60, 300, 3_600):
            snapshot = self._state.index(f"network_{window}s").snapshot(event.network_token)
            values[f"network_prior_attempt_count_{window}s"] = snapshot.attempts
            values[f"network_prior_failure_count_{window}s"] = snapshot.failures
            values[f"network_distinct_instruments_{window}s"] = snapshot.distinct.get(
                "instruments", 0
            )
            values[f"network_distinct_devices_{window}s"] = snapshot.distinct.get("devices", 0)
            values[f"network_distinct_sessions_{window}s"] = snapshot.distinct.get("sessions", 0)

    def _add_instrument_features(self, values: dict[str, int], event: CheckoutAttemptEvent) -> None:
        for window in (300, 3_600):
            snapshot = self._state.index(f"instrument_{window}s").snapshot(
                event.payment_instrument_token
            )
            values[f"instrument_prior_attempt_count_{window}s"] = snapshot.attempts
            values[f"instrument_prior_failure_count_{window}s"] = snapshot.failures
            values[f"instrument_distinct_devices_{window}s"] = snapshot.distinct.get("devices", 0)
            values[f"instrument_distinct_networks_{window}s"] = snapshot.distinct.get("networks", 0)
            values[f"instrument_distinct_merchants_{window}s"] = snapshot.distinct.get(
                "merchants", 0
            )

    def _add_session_features(self, values: dict[str, int], event: CheckoutAttemptEvent) -> None:
        for window in (60, 300):
            snapshot = self._state.index(f"session_{window}s").snapshot(event.session_token)
            values[f"session_prior_attempt_count_{window}s"] = snapshot.attempts
            values[f"session_prior_failure_count_{window}s"] = snapshot.failures
            values[f"session_distinct_instruments_{window}s"] = snapshot.distinct.get(
                "instruments", 0
            )

    def diagnostics(self) -> dict[str, object]:
        return self._state.diagnostics()
