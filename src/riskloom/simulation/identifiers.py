import hashlib
import json
import random
import re
from collections.abc import Mapping
from typing import Any
from uuid import UUID, uuid5

RISKLOOM_SIMULATION_NAMESPACE = UUID("9cb8e2f1-9c60-5f0d-a947-80ee21c37c0a")
LEGACY_SIMULATION_ALGORITHM_VERSION = "1.0.0"
SIMULATION_ALGORITHM_VERSION = "1.1.0"
SUPPORTED_SIMULATION_ALGORITHM_VERSIONS = frozenset(
    {LEGACY_SIMULATION_ALGORITHM_VERSION, SIMULATION_ALGORITHM_VERSION}
)
_FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _validate_seed(seed: int) -> None:
    if type(seed) is not int or not 0 <= seed <= 2**63 - 1:
        raise ValueError("seed_out_of_range")


def effective_configuration_fingerprint(effective_configuration: Mapping[str, Any]) -> str:
    """Hash canonical effective configuration without artifact or machine state."""

    canonical = (
        json.dumps(
            dict(effective_configuration),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _namespace_parts(
    algorithm_version: str,
    configuration_fingerprint: str | None,
) -> tuple[str, ...]:
    if algorithm_version not in SUPPORTED_SIMULATION_ALGORITHM_VERSIONS:
        raise ValueError("simulation_algorithm_version_unsupported")
    if algorithm_version == LEGACY_SIMULATION_ALGORITHM_VERSION:
        if configuration_fingerprint is not None:
            raise ValueError("legacy_configuration_fingerprint_forbidden")
        return (algorithm_version,)
    if (
        configuration_fingerprint is None
        or _FINGERPRINT_PATTERN.fullmatch(configuration_fingerprint) is None
    ):
        raise ValueError("configuration_fingerprint_required")
    return (algorithm_version, configuration_fingerprint)


def deterministic_identifier(
    prefix: str,
    seed: int,
    kind: str,
    *parts: str | int,
    algorithm_version: str = LEGACY_SIMULATION_ALGORITHM_VERSION,
    configuration_fingerprint: str | None = None,
) -> str:
    _validate_seed(seed)
    name = "/".join(
        (
            "riskloom",
            "simulation",
            *_namespace_parts(algorithm_version, configuration_fingerprint),
            str(seed),
            kind,
            *(str(part) for part in parts),
        )
    )
    return f"{prefix}_{uuid5(RISKLOOM_SIMULATION_NAMESPACE, name).hex}"


def derived_seed(
    seed: int,
    split: str,
    component: str,
    *,
    algorithm_version: str = LEGACY_SIMULATION_ALGORITHM_VERSION,
    configuration_fingerprint: str | None = None,
) -> int:
    _validate_seed(seed)
    material = "/".join(
        (
            "riskloom",
            "simulation",
            *_namespace_parts(algorithm_version, configuration_fingerprint),
            str(seed),
            split,
            component,
        )
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest(), byteorder="big")


def random_stream(
    seed: int,
    split: str,
    component: str,
    *,
    algorithm_version: str = LEGACY_SIMULATION_ALGORITHM_VERSION,
    configuration_fingerprint: str | None = None,
) -> random.Random:
    return random.Random(
        derived_seed(
            seed,
            split,
            component,
            algorithm_version=algorithm_version,
            configuration_fingerprint=configuration_fingerprint,
        )
    )
