from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from random import Random

from riskloom.simulation.config import (
    GeneratorConfig,
    SplitConfig,
    boundary_timestamp,
    configuration_fingerprint,
    generator_version_for_config,
    validated_configuration_snapshot,
)
from riskloom.simulation.event_schema import (
    Channel,
    CheckoutAttemptEvent,
    FailureCategory,
    Outcome,
)
from riskloom.simulation.identifiers import deterministic_identifier, random_stream
from riskloom.simulation.label_schema import (
    CampaignProfile,
    GeneratorMetadata,
    GroundTruthLabel,
    ScenarioType,
)


@dataclass(frozen=True, slots=True)
class GeneratedRecord:
    event: CheckoutAttemptEvent
    label: GroundTruthLabel


@dataclass(frozen=True, slots=True)
class CampaignWindow:
    start: datetime
    duration_ms: int

    @property
    def end(self) -> datetime:
        return self.start + timedelta(milliseconds=self.duration_ms)


def _allocate_weighted(count: int, weights: dict[str, int], rng: Random) -> list[str]:
    """Allocate a count using integer largest remainders, then deterministically shuffle."""

    total_weight = sum(weights.values())
    allocations: dict[str, int] = {}
    remainders: list[tuple[int, str]] = []
    allocated = 0
    for key, weight in sorted(weights.items()):
        numerator = count * weight
        allocations[key] = numerator // total_weight
        allocated += allocations[key]
        remainders.append((numerator % total_weight, key))
    for _, key in sorted(remainders, key=lambda item: (-item[0], item[1]))[: count - allocated]:
        allocations[key] += 1
    values = [key for key in sorted(allocations) for _ in range(allocations[key])]
    rng.shuffle(values)
    return values


def _split_into_groups(total: int, group_count: int) -> list[int]:
    base, remainder = divmod(total, group_count)
    return [base + (1 if index < remainder else 0) for index in range(group_count)]


def _timedelta_milliseconds(value: timedelta) -> int:
    return value.days * 86_400_000 + value.seconds * 1_000 + value.microseconds // 1_000


def _retry_group_sizes(total: int, minimum: int, maximum: int, rng: Random) -> list[int]:
    sizes: list[int] = []
    remaining = total
    while remaining:
        if remaining <= maximum:
            if remaining < minimum:
                sizes[-1] += remaining
            else:
                sizes.append(remaining)
            break
        size = rng.randint(minimum, maximum)
        if 0 < remaining - size < minimum:
            size -= minimum - (remaining - size)
        sizes.append(size)
        remaining -= size
    if sum(sizes) != total or any(not minimum <= size <= maximum for size in sizes):
        raise ValueError("retry_quota_cannot_form_valid_chains")
    return sizes


class SimulationGenerator:
    def __init__(self, config: GeneratorConfig, seed: int) -> None:
        validated_config = validated_configuration_snapshot(config)
        if type(seed) is not int or not 0 <= seed <= 2**63 - 1:
            raise ValueError("seed_out_of_range")
        self.config = validated_config
        self.seed = seed
        self.algorithm_version = generator_version_for_config(validated_config)
        self.configuration_fingerprint = configuration_fingerprint(validated_config)
        self.merchants = [
            self._identifier("mrc", "merchant", index)
            for index in range(validated_config.merchant_count)
        ]
        self.catalogs = self._build_catalogs()

    def generate(self) -> list[GeneratedRecord]:
        records: list[GeneratedRecord] = []
        split_start = self.config.start_at.astimezone(UTC)
        for split in self.config.splits:
            split_end = split_start + timedelta(days=split.duration_days)
            counts = self.config.scenario_counts(split)
            records.extend(self._normal(split, split_start, split_end, counts["normal"]))
            records.extend(self._retries(split, split_start, split_end, counts["legitimate_retry"]))
            records.extend(self._flash_sale(split, split_start, split_end, counts["flash_sale"]))
            records.extend(
                self._shared_infrastructure(
                    split,
                    split_start,
                    split_end,
                    counts["shared_infrastructure"],
                )
            )
            records.extend(
                self._legitimate_failures(
                    split,
                    split_start,
                    split_end,
                    counts["legitimate_failure"],
                )
            )
            records.extend(self._campaigns(split, split_start, split_end, counts["attack"]))
            split_start = split_end

        records.sort(key=lambda record: (record.event.occurred_at, record.event.event_id))
        if len(records) != self.config.total_events:
            raise ValueError("generated_event_count_mismatch")
        return records

    def _identifier(self, prefix: str, kind: str, *parts: str | int) -> str:
        return deterministic_identifier(
            prefix,
            self.seed,
            kind,
            *parts,
            algorithm_version=self.algorithm_version,
            configuration_fingerprint=self.configuration_fingerprint,
        )

    def _random_stream(self, split: str, component: str) -> Random:
        return random_stream(
            self.seed,
            split,
            component,
            algorithm_version=self.algorithm_version,
            configuration_fingerprint=self.configuration_fingerprint,
        )

    def _build_catalogs(self) -> dict[str, list[int]]:
        catalogs: dict[str, list[int]] = {}
        span = self.config.amount_maximum_subunits - self.config.amount_minimum_subunits
        denominator = self.config.merchant_catalog_points - 1
        for merchant_index, merchant_id in enumerate(self.merchants):
            rng = self._random_stream("global", f"catalog-{merchant_index}")
            step = max(1, span // denominator)
            values: list[int] = []
            for point in range(self.config.merchant_catalog_points):
                base = self.config.amount_minimum_subunits + span * point // denominator
                jitter = rng.randrange(max(1, step // 4)) if point not in (0, denominator) else 0
                values.append(min(self.config.amount_maximum_subunits, base + jitter))
            catalogs[merchant_id] = sorted(set(values))
        return catalogs

    def _time(self, start: datetime, end: datetime, rng: Random, margin_ms: int = 0) -> datetime:
        duration = end - start
        duration_ms = (
            duration.days * 86_400_000
            + duration.seconds * 1_000
            + duration.microseconds // 1_000
            - margin_ms
        )
        if duration_ms <= 0:
            raise ValueError("split_duration_too_short")
        return start + timedelta(milliseconds=rng.randrange(duration_ms))

    def _campaign_windows(
        self,
        split: SplitConfig,
        start: datetime,
        end: datetime,
    ) -> list[CampaignWindow]:
        placement = split.campaign_placement
        windows: list[CampaignWindow] = []
        if placement is None:
            for campaign_index in range(split.campaign_count):
                rng = self._random_stream(
                    split.name.value,
                    f"campaign-window-{campaign_index}",
                )
                duration_ms = rng.randint(30, 90) * 60 * 1_000
                windows.append(
                    CampaignWindow(
                        start=self._time(start, end, rng, margin_ms=duration_ms + 1),
                        duration_ms=duration_ms,
                    )
                )
            return windows

        boundary = boundary_timestamp(
            start,
            end,
            placement.protected_boundary_basis_points,
        )
        campaign_indices = list(range(split.campaign_count))
        assignment_rng = self._random_stream(
            split.name.value,
            "campaign-placement-sides",
        )
        assignment_rng.shuffle(campaign_indices)
        before_indices = frozenset(campaign_indices[: placement.minimum_campaigns_before_boundary])
        gap = timedelta(seconds=placement.minimum_gap_seconds)
        accepted: list[CampaignWindow] = []
        by_index: dict[int, CampaignWindow] = {}
        for campaign_index in range(split.campaign_count):
            rng = self._random_stream(
                split.name.value,
                f"campaign-window-{campaign_index}",
            )
            duration_ms = rng.randint(30, 90) * 60 * 1_000
            duration = timedelta(milliseconds=duration_ms)
            if campaign_index in before_indices:
                minimum_start = start
                maximum_start = boundary - duration
            else:
                minimum_start = boundary
                maximum_start = end - duration
            available_ms = _timedelta_milliseconds(maximum_start - minimum_start)
            if available_ms < 0:
                raise ValueError("campaign_placement_infeasible")
            selected: CampaignWindow | None = None
            for _ in range(placement.maximum_sampling_attempts_per_campaign):
                candidate = CampaignWindow(
                    start=minimum_start + timedelta(milliseconds=rng.randrange(available_ms + 1)),
                    duration_ms=duration_ms,
                )
                if all(
                    candidate.end + gap <= existing.start or existing.end + gap <= candidate.start
                    for existing in accepted
                ):
                    selected = candidate
                    break
            if selected is None:
                raise ValueError("campaign_placement_infeasible")
            accepted.append(selected)
            by_index[campaign_index] = selected

        ordered = sorted(accepted, key=lambda window: window.start)
        gaps_ms = [
            _timedelta_milliseconds(current.start - previous.end)
            for previous, current in zip(ordered, ordered[1:], strict=False)
        ]
        if len(gaps_ms) > 1 and len(set(gaps_ms)) == 1:
            raise ValueError("campaign_placement_must_be_irregular")
        return [by_index[index] for index in range(split.campaign_count)]

    def _pool_token(self, prefix: str, kind: str, size: int, rng: Random) -> str:
        return self._identifier(prefix, kind, rng.randrange(size))

    def _maybe_pool_token(
        self,
        prefix: str,
        kind: str,
        size: int,
        missing_basis_points: int,
        rng: Random,
    ) -> str | None:
        if rng.randrange(10_000) < missing_basis_points:
            return None
        return self._pool_token(prefix, kind, size, rng)

    def _channels(self, count: int, rng: Random) -> list[Channel]:
        values = _allocate_weighted(count, self.config.channel_weights.model_dump(), rng)
        return [Channel(value) for value in values]

    def _outcomes(
        self, count: int, failure_basis_points: int, rng: Random
    ) -> list[tuple[Outcome, FailureCategory | None]]:
        failure_count = (count * failure_basis_points + 5_000) // 10_000
        failures = _allocate_weighted(
            failure_count,
            self.config.failure_weights.model_dump(),
            rng,
        )
        outcomes: list[tuple[Outcome, FailureCategory | None]] = [
            (Outcome.FAILED, FailureCategory(value)) for value in failures
        ]
        outcomes.extend((Outcome.AUTHORIZED, None) for _ in range(count - failure_count))
        rng.shuffle(outcomes)
        return outcomes

    def _amount(self, merchant_id: str, rng: Random) -> int:
        catalog = self.catalogs[merchant_id]
        return catalog[rng.randrange(len(catalog))]

    def _event(
        self,
        *,
        split: SplitConfig,
        scenario: str,
        index: int,
        occurred_at: datetime,
        merchant_id: str,
        checkout_id: str,
        customer_token: str | None,
        device_token: str | None,
        network_token: str | None,
        session_token: str,
        instrument_token: str,
        outcome: Outcome,
        failure_category: FailureCategory | None,
        channel: Channel,
        rng: Random,
    ) -> CheckoutAttemptEvent:
        return CheckoutAttemptEvent(
            event_id=self._identifier("evt", "event", split.name.value, scenario, index),
            merchant_id=merchant_id,
            occurred_at=occurred_at,
            checkout_id=checkout_id,
            customer_token=customer_token,
            device_token=device_token,
            network_token=network_token,
            session_token=session_token,
            payment_instrument_token=instrument_token,
            amount_subunits=self._amount(merchant_id, rng),
            currency=self.config.currency,
            outcome=outcome,
            failure_category=failure_category,
            channel=channel,
        )

    def _label(
        self,
        event: CheckoutAttemptEvent,
        split: SplitConfig,
        scenario: ScenarioType,
        scenario_instance_id: str | None,
        campaign_id: str | None = None,
        campaign_profile: CampaignProfile | None = None,
    ) -> GroundTruthLabel:
        return GroundTruthLabel(
            event_id=event.event_id,
            split=split.name,
            is_attack=scenario is ScenarioType.CARD_TESTING_CAMPAIGN,
            campaign_id=campaign_id,
            scenario_type=scenario,
            generator_metadata=GeneratorMetadata(
                scenario_instance_id=scenario_instance_id,
                campaign_profile=campaign_profile,
            ),
        )

    def _ordinary_tokens(self, index: int, scenario: str, rng: Random) -> dict[str, str | None]:
        pools = self.config.entity_pools
        missing = self.config.missingness_rates
        return {
            "checkout_id": self._identifier("chk", "checkout", scenario, index),
            "customer_token": self._maybe_pool_token(
                "cus", "customer", pools.customers, missing.customer, rng
            ),
            "device_token": self._maybe_pool_token(
                "dev", "device", pools.devices, missing.device, rng
            ),
            "network_token": self._maybe_pool_token(
                "net", "network", pools.networks, missing.network, rng
            ),
            "session_token": self._identifier("ses", "session", scenario, index),
            "instrument_token": self._pool_token("pmt", "instrument", pools.instruments, rng),
        }

    def _normal(
        self, split: SplitConfig, start: datetime, end: datetime, count: int
    ) -> list[GeneratedRecord]:
        rng = self._random_stream(split.name.value, "normal")
        channels = self._channels(count, rng)
        outcomes = self._outcomes(count, self.config.outcome_rates.normal_failure, rng)
        records: list[GeneratedRecord] = []
        for index in range(count):
            merchant = self.merchants[rng.randrange(len(self.merchants))]
            tokens = self._ordinary_tokens(index, f"{split.name.value}-normal", rng)
            outcome, failure = outcomes[index]
            event = self._event(
                split=split,
                scenario="normal",
                index=index,
                occurred_at=self._time(start, end, rng),
                merchant_id=merchant,
                outcome=outcome,
                failure_category=failure,
                channel=channels[index],
                rng=rng,
                **tokens,  # type: ignore[arg-type]
            )
            records.append(
                GeneratedRecord(event, self._label(event, split, ScenarioType.NORMAL, None))
            )
        return records

    def _retries(
        self, split: SplitConfig, start: datetime, end: datetime, count: int
    ) -> list[GeneratedRecord]:
        rng = self._random_stream(split.name.value, "legitimate-retry")
        bounds = self.config.retry_bounds
        group_sizes = _retry_group_sizes(
            count,
            bounds.minimum_attempts,
            bounds.maximum_attempts,
            rng,
        )
        successful_chains = set(
            index
            for index, value in enumerate(
                _allocate_weighted(
                    len(group_sizes),
                    {
                        "successful": self.config.outcome_rates.retry_eventual_success,
                        "failed": 10_000 - self.config.outcome_rates.retry_eventual_success,
                    },
                    rng,
                )
            )
            if value == "successful"
        )
        channels = self._channels(len(group_sizes), rng)
        failure_values = _allocate_weighted(count, self.config.failure_weights.model_dump(), rng)
        records: list[GeneratedRecord] = []
        event_index = 0
        failure_index = 0
        for chain_index, size in enumerate(group_sizes):
            merchant = self.merchants[rng.randrange(len(self.merchants))]
            scenario_key = f"{split.name.value}-retry-{chain_index}"
            checkout = self._identifier("chk", "retry-checkout", scenario_key)
            session = self._identifier("ses", "retry-session", scenario_key)
            customer = self._maybe_pool_token(
                "cus",
                "customer",
                self.config.entity_pools.customers,
                self.config.missingness_rates.customer,
                rng,
            )
            device = self._maybe_pool_token(
                "dev",
                "device",
                self.config.entity_pools.devices,
                self.config.missingness_rates.device,
                rng,
            )
            network = self._maybe_pool_token(
                "net",
                "network",
                self.config.entity_pools.networks,
                self.config.missingness_rates.network,
                rng,
            )
            instrument = self._pool_token(
                "pmt", "instrument", self.config.entity_pools.instruments, rng
            )
            scenario_id = self._identifier("scn", "retry", scenario_key)
            maximum_chain_ms = bounds.maximum_gap_seconds * 1_000 * (size - 1)
            occurred = self._time(start, end, rng, margin_ms=maximum_chain_ms + 1)
            for attempt in range(size):
                succeeds = chain_index in successful_chains and attempt == size - 1
                if succeeds:
                    outcome, failure = Outcome.AUTHORIZED, None
                else:
                    outcome = Outcome.FAILED
                    failure = FailureCategory(failure_values[failure_index])
                    failure_index += 1
                event = self._event(
                    split=split,
                    scenario="legitimate-retry",
                    index=event_index,
                    occurred_at=occurred,
                    merchant_id=merchant,
                    checkout_id=checkout,
                    customer_token=customer,
                    device_token=device,
                    network_token=network,
                    session_token=session,
                    instrument_token=instrument,
                    outcome=outcome,
                    failure_category=failure,
                    channel=channels[chain_index],
                    rng=rng,
                )
                records.append(
                    GeneratedRecord(
                        event,
                        self._label(
                            event,
                            split,
                            ScenarioType.LEGITIMATE_RETRY,
                            scenario_id,
                        ),
                    )
                )
                event_index += 1
                occurred += timedelta(
                    seconds=rng.randint(bounds.minimum_gap_seconds, bounds.maximum_gap_seconds)
                )
        return records

    def _flash_sale(
        self, split: SplitConfig, start: datetime, end: datetime, count: int
    ) -> list[GeneratedRecord]:
        rng = self._random_stream(split.name.value, "flash-sale")
        window_count = min(3, max(1, count // 100))
        sizes = _split_into_groups(count, window_count)
        channels = self._channels(count, rng)
        outcomes = self._outcomes(count, self.config.outcome_rates.flash_sale_failure, rng)
        records: list[GeneratedRecord] = []
        event_index = 0
        for window_index, size in enumerate(sizes):
            scenario_id = self._identifier("scn", "flash-sale", split.name.value, window_index)
            merchant = self.merchants[rng.randrange(len(self.merchants))]
            shared_network = self._pool_token(
                "net", "network", self.config.entity_pools.networks, rng
            )
            duration_ms = rng.randint(30, 90) * 60 * 1_000
            window_start = self._time(start, end, rng, margin_ms=duration_ms + 1)
            for _ in range(size):
                tokens = self._ordinary_tokens(event_index, f"{split.name.value}-flash", rng)
                tokens["network_token"] = shared_network
                outcome, failure = outcomes[event_index]
                event = self._event(
                    split=split,
                    scenario="flash-sale",
                    index=event_index,
                    occurred_at=window_start + timedelta(milliseconds=rng.randrange(duration_ms)),
                    merchant_id=merchant,
                    outcome=outcome,
                    failure_category=failure,
                    channel=channels[event_index],
                    rng=rng,
                    **tokens,  # type: ignore[arg-type]
                )
                records.append(
                    GeneratedRecord(
                        event,
                        self._label(event, split, ScenarioType.FLASH_SALE, scenario_id),
                    )
                )
                event_index += 1
        return records

    def _shared_infrastructure(
        self, split: SplitConfig, start: datetime, end: datetime, count: int
    ) -> list[GeneratedRecord]:
        rng = self._random_stream(split.name.value, "shared-infrastructure")
        group_count = min(3, max(1, count // 50))
        sizes = _split_into_groups(count, group_count)
        channels = self._channels(count, rng)
        outcomes = self._outcomes(
            count,
            self.config.outcome_rates.shared_infrastructure_failure,
            rng,
        )
        records: list[GeneratedRecord] = []
        event_index = 0
        for group_index, size in enumerate(sizes):
            scenario_id = self._identifier(
                "scn", "shared-infrastructure", split.name.value, group_index
            )
            shared_network = self._pool_token(
                "net", "network", self.config.entity_pools.networks, rng
            )
            shared_device = self._pool_token("dev", "device", self.config.entity_pools.devices, rng)
            for local_index in range(size):
                merchant = self.merchants[rng.randrange(len(self.merchants))]
                tokens = self._ordinary_tokens(event_index, f"{split.name.value}-shared", rng)
                tokens["network_token"] = shared_network
                if local_index % 3 == 0:
                    tokens["device_token"] = shared_device
                outcome, failure = outcomes[event_index]
                event = self._event(
                    split=split,
                    scenario="shared-infrastructure",
                    index=event_index,
                    occurred_at=self._time(start, end, rng),
                    merchant_id=merchant,
                    outcome=outcome,
                    failure_category=failure,
                    channel=channels[event_index],
                    rng=rng,
                    **tokens,  # type: ignore[arg-type]
                )
                records.append(
                    GeneratedRecord(
                        event,
                        self._label(
                            event,
                            split,
                            ScenarioType.SHARED_INFRASTRUCTURE,
                            scenario_id,
                        ),
                    )
                )
                event_index += 1
        return records

    def _legitimate_failures(
        self, split: SplitConfig, start: datetime, end: datetime, count: int
    ) -> list[GeneratedRecord]:
        rng = self._random_stream(split.name.value, "legitimate-failure")
        channels = self._channels(count, rng)
        failures = _allocate_weighted(count, self.config.failure_weights.model_dump(), rng)
        records: list[GeneratedRecord] = []
        for index in range(count):
            merchant = self.merchants[rng.randrange(len(self.merchants))]
            tokens = self._ordinary_tokens(index, f"{split.name.value}-failure", rng)
            event = self._event(
                split=split,
                scenario="legitimate-failure",
                index=index,
                occurred_at=self._time(start, end, rng),
                merchant_id=merchant,
                outcome=Outcome.FAILED,
                failure_category=FailureCategory(failures[index]),
                channel=channels[index],
                rng=rng,
                **tokens,  # type: ignore[arg-type]
            )
            records.append(
                GeneratedRecord(
                    event,
                    self._label(event, split, ScenarioType.LEGITIMATE_FAILURE, None),
                )
            )
        return records

    def _campaigns(
        self, split: SplitConfig, start: datetime, end: datetime, count: int
    ) -> list[GeneratedRecord]:
        rng = self._random_stream(split.name.value, "campaign")
        sizes = _split_into_groups(count, split.campaign_count)
        windows = (
            None
            if self.config.config_schema_version == "1.0.0"
            else self._campaign_windows(split, start, end)
        )
        channels = self._channels(count, rng)
        # Failure camouflage is decided here, before the campaign loop, because outcomes are drawn
        # for the whole split at once. An attacker working from pre-validated cards produces the
        # legitimate failure rate rather than the 75% that card testing normally leaves behind.
        split_evasion = split.evasion_shape
        attack_failure_rate = self.config.outcome_rates.attack_failure
        if split_evasion is not None and split_evasion.variant == "failure_camouflage":
            assert split_evasion.failure_rate_basis_points is not None
            attack_failure_rate = split_evasion.failure_rate_basis_points
        outcomes = self._outcomes(count, attack_failure_rate, rng)
        records: list[GeneratedRecord] = []
        event_index = 0
        for campaign_index, size in enumerate(sizes):
            campaign_id = self._identifier("cmp", "campaign", split.name.value, campaign_index)
            scenario_id = self._identifier(
                "scn", "campaign-scenario", split.name.value, campaign_index
            )
            merchant_count = min(len(self.merchants), rng.randint(2, min(4, len(self.merchants))))
            campaign_merchants = rng.sample(self.merchants, merchant_count)
            # Every evasion branch below is strictly conditional and draws no random values when
            # absent. `rng` is one shared stream for the whole campaign loop, so a single
            # unconditional draw here would shift every subsequent value and silently change the
            # development, smoke and policy-validation datasets.
            evasion = split.evasion_shape
            shared_network = self._identifier(
                "net", "campaign-network", split.name.value, campaign_index
            )
            campaign_networks = [shared_network]
            if evasion is not None and evasion.variant == "distributed_thin":
                assert evasion.network_count is not None
                campaign_networks = [
                    self._identifier(
                        "net", "campaign-network", split.name.value, campaign_index, index
                    )
                    for index in range(evasion.network_count)
                ]
            baseline_devices = max(1, (size + 11) // 12)
            device_count = (
                min(size, baseline_devices * 4)
                if split.campaign_profile is CampaignProfile.ENTITY_REUSE_SHIFT
                else baseline_devices
            )
            if evasion is not None and evasion.variant == "distributed_thin":
                # One device per event: the reuse the detector keys on simply is not there.
                device_count = size
            devices = [
                self._identifier(
                    "dev",
                    "campaign-device",
                    split.name.value,
                    campaign_index,
                    index,
                )
                for index in range(device_count)
            ]
            if windows is None:
                duration_ms = rng.randint(30, 90) * 60 * 1_000
                window_start = self._time(start, end, rng, margin_ms=duration_ms + 1)
            else:
                window = windows[campaign_index]
                duration_ms = window.duration_ms
                window_start = window.start
            if evasion is not None:
                # A stretched or exactly-spaced campaign needs more room than the 30-90 minutes the
                # window was sized for, so the start is pulled back to keep every event inside its
                # split. `event_outside_labeled_split` is a hard dataset invariant; sliding the
                # start is preferable to truncating the campaign, which would silently reduce the
                # attack volume and make a detection drop unattributable.
                if evasion.variant == "slow_and_low":
                    assert evasion.duration_minutes is not None
                    duration_ms = evasion.duration_minutes * 60 * 1_000
                    required_ms = duration_ms
                elif evasion.variant == "window_edge":
                    assert evasion.edge_window_seconds is not None
                    required_ms = max(0, size - 1) * evasion.edge_window_seconds * 1_000
                else:
                    required_ms = duration_ms
                latest_start = end - timedelta(milliseconds=required_ms + 1)
                if window_start > latest_start:
                    window_start = max(start, latest_start)
                room_ms = int((end - window_start).total_seconds() * 1_000) - 1
                duration_ms = max(1, min(duration_ms, room_ms))
            missing_device_count = (size * self.config.missingness_rates.device + 5_000) // 10_000
            missing_network_count = (size * self.config.missingness_rates.network + 5_000) // 10_000
            missing_device_indices = set(rng.sample(range(size), missing_device_count))
            missing_network_indices = set(rng.sample(range(size), missing_network_count))
            # Deterministic offsets only for the window-edge variant. Everything else keeps
            # drawing inline below, at exactly the point in the loop the original did, so the
            # shared stream's interleaving with the other per-event draws is untouched.
            offsets_ms: list[int] | None = None
            if evasion is not None and evasion.variant == "window_edge":
                # Exactly one window apart, so at each event the previous one sits precisely on the
                # expiry cutoff. Windows are (current_time - window, current_time], left-exclusive,
                # so an event exactly one window back is already gone and the count reads zero.
                assert evasion.edge_window_seconds is not None
                step_ms = evasion.edge_window_seconds * 1_000
                offsets_ms = [index * step_ms for index in range(size)]
            for local_index in range(size):
                merchant = campaign_merchants[local_index % len(campaign_merchants)]
                device: str | None = devices[local_index % len(devices)]
                if local_index in missing_device_indices:
                    device = None
                session_part = (
                    local_index
                    if split.campaign_profile is CampaignProfile.ENTITY_REUSE_SHIFT
                    else local_index % len(devices)
                )
                customer = self._maybe_pool_token(
                    "cus",
                    "customer",
                    self.config.entity_pools.customers,
                    self.config.missingness_rates.customer,
                    rng,
                )
                network: str | None = campaign_networks[local_index % len(campaign_networks)]
                if local_index in missing_network_indices:
                    network = None
                outcome, failure = outcomes[event_index]
                event = self._event(
                    split=split,
                    scenario="campaign",
                    index=event_index,
                    occurred_at=window_start
                    + timedelta(
                        milliseconds=(
                            rng.randrange(duration_ms)
                            if offsets_ms is None
                            else offsets_ms[local_index]
                        )
                    ),
                    merchant_id=merchant,
                    checkout_id=self._identifier(
                        "chk", "campaign-checkout", split.name.value, event_index
                    ),
                    customer_token=customer,
                    device_token=device,
                    network_token=network,
                    session_token=self._identifier(
                        "ses",
                        "campaign-session",
                        split.name.value,
                        campaign_index,
                        session_part,
                    ),
                    instrument_token=self._identifier(
                        "pmt",
                        "campaign-instrument",
                        split.name.value,
                        event_index,
                    ),
                    outcome=outcome,
                    failure_category=failure,
                    channel=channels[event_index],
                    rng=rng,
                )
                records.append(
                    GeneratedRecord(
                        event,
                        self._label(
                            event,
                            split,
                            ScenarioType.CARD_TESTING_CAMPAIGN,
                            scenario_id,
                            campaign_id=campaign_id,
                            campaign_profile=split.campaign_profile,
                        ),
                    )
                )
                event_index += 1
        return records


def generate_records(config: GeneratorConfig, seed: int) -> list[GeneratedRecord]:
    return SimulationGenerator(config, seed).generate()
