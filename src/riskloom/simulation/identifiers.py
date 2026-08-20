import hashlib
import random
from uuid import UUID, uuid5

RISKLOOM_SIMULATION_NAMESPACE = UUID("9cb8e2f1-9c60-5f0d-a947-80ee21c37c0a")
SIMULATION_ALGORITHM_VERSION = "1.0.0"


def _validate_seed(seed: int) -> None:
    if type(seed) is not int or not 0 <= seed <= 2**63 - 1:
        raise ValueError("seed_out_of_range")


def deterministic_identifier(prefix: str, seed: int, kind: str, *parts: str | int) -> str:
    _validate_seed(seed)
    name = "/".join(
        (
            "riskloom",
            "simulation",
            SIMULATION_ALGORITHM_VERSION,
            str(seed),
            kind,
            *(str(part) for part in parts),
        )
    )
    return f"{prefix}_{uuid5(RISKLOOM_SIMULATION_NAMESPACE, name).hex}"


def derived_seed(seed: int, split: str, component: str) -> int:
    _validate_seed(seed)
    material = (  # noqa: UP012 - keep deterministic encoding explicit
        f"riskloom/simulation/{SIMULATION_ALGORITHM_VERSION}/{seed}/{split}/{component}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest(), byteorder="big")


def random_stream(seed: int, split: str, component: str) -> random.Random:
    return random.Random(derived_seed(seed, split, component))
