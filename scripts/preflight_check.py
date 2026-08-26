"""Report whether this checkout can actually start the service.

A fresh ``git clone`` cannot. Every artifact the application binds to at startup is Git-ignored, so
a clone contains the configuration files and no model at all, and the container will exit with a
``serving_*`` identity rather than serve. That is correct fail-closed behaviour, but it is a
confusing first experience, so this script names what is missing before anything is started.

It reports three states rather than two. A file can be present and valid, absent, or present and
*invalid* -- the last being the case that a bare existence check reports as success and startup
then rejects. Two defects made that concrete:

* ``training_report.json`` was not checked at all, yet ``load_locked_model`` requires the model
  directory to contain exactly ``model.json``, ``training_report.json`` and ``manifest.json``.
  Preflight passed on a directory the application refuses.
* Existence was checked, but not content. A truncated or edited artifact passed.

Both are fixed by pinning hashes and then running the same ``load_serving_bundle`` the application
runs at startup. This script reads only; it creates nothing, downloads nothing and installs nothing.
"""

import hashlib
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from riskloom.runtime_bundle import (  # noqa: E402
    EVALUATION_MEMBER,
    EXPECTED_SHA256,
    STARTUP_MEMBERS,
    RuntimeBundleError,
    verify_installation,
)

INSTALL_COMMAND = "uv run python scripts/runtime_bundle.py install"
# The primary command, and the only one an ordinary first-time reader needs: it downloads the pinned
# asset from the published v1.0.3-runtime GitHub Release, verifies it, and installs it.
RELEASE_TAG_NAME = "v1.0.3-runtime"
# Secondary, and deliberately described as such. It exists for someone who already holds a verified
# archive -- an air-gapped machine, a mirrored copy -- not as a workaround for the normal path.
OFFLINE_COMMAND = (
    "uv run python scripts/runtime_bundle.py install --archive <path to "
    "riskloom-runtime-artifacts.zip>"
)

TRACKED_CONFIGURATION = (
    ("configs/features/default.json", "feature configuration"),
    ("configs/modeling/default.json", "modeling configuration"),
)

RUNTIME_DESCRIPTIONS = {
    "artifacts/models/development/model.json": "locked model",
    "artifacts/models/development/training_report.json": "locked training report",
    "artifacts/models/development/manifest.json": "locked model manifest",
    "artifacts/features/development-v1.1.0-config-bound/manifest.json": "locked feature manifest",
    EVALUATION_MEMBER: "held-out evaluation (model panel and drift reference)",
}

# The description table and the allowlist must describe the same files. Checked here rather than
# left to a KeyError at the moment somebody is already debugging a failed startup.
assert set(RUNTIME_DESCRIPTIONS) == {*STARTUP_MEMBERS, EVALUATION_MEMBER}

PRESENT = "ok"
MISSING = "missing"
INVALID = "invalid"


def _state(relative: str) -> str:
    """Classify one pinned artifact as present-and-valid, absent, or present-but-invalid."""

    path = REPOSITORY_ROOT / relative
    if not path.exists():
        return MISSING
    if not path.is_file():
        return INVALID
    try:
        payload = path.read_bytes()
    except OSError:
        return INVALID
    if hashlib.sha256(payload).hexdigest() != EXPECTED_SHA256[relative]:
        return INVALID
    return PRESENT


def _print_group(title: str, rows: list[tuple[str, str, str]]) -> None:
    print(f"\n{title}")
    for relative, description, state in rows:
        print(f"  [{state:>7}] {relative}")
        print(f"            {description}")


def main() -> int:
    print("RiskLoom startup preflight")

    tracked: list[tuple[str, str, str]] = []
    for relative, description in TRACKED_CONFIGURATION:
        state = PRESENT if (REPOSITORY_ROOT / relative).is_file() else MISSING
        tracked.append((relative, f"{description} (in git)", state))
    _print_group("required configuration:", tracked)

    runtime: list[tuple[str, str, str]] = []
    for relative in STARTUP_MEMBERS:
        state = _state(relative)
        runtime.append((relative, f"{RUNTIME_DESCRIPTIONS[relative]} (git-ignored input)", state))
    _print_group("required to start:", runtime)

    evaluation_state = _state(EVALUATION_MEMBER)
    _print_group(
        "required for the complete product:",
        [
            (
                EVALUATION_MEMBER,
                f"{RUNTIME_DESCRIPTIONS[EVALUATION_MEMBER]} (git-ignored input)",
                evaluation_state,
            )
        ],
    )

    env_present = (REPOSITORY_ROOT / ".env").is_file()
    print("\noptional:")
    print(f"  [{('ok' if env_present else 'absent'):>7}] .env")
    print("            runtime configuration; `Copy-Item .env.example .env` is enough to start")

    blocking = [row for row in tracked + runtime if row[2] != PRESENT]
    if blocking:
        print(f"\n{len(blocking)} required file(s) missing or invalid. The service will not start.")
        for relative, _description, state in blocking:
            print(f"  {state:>7}  {relative}")
        print("\nThese are locked artifacts, deliberately excluded from version control. They are")
        print("inputs to the image, never built into it, and they cannot be regenerated here:")
        print("retraining would produce a different model and break the locked-artifact contract.")
        print("\nInstall the runtime bundle:")
        print(f"  {INSTALL_COMMAND}")
        print(
            f"\nThat downloads the pinned asset from the published {RELEASE_TAG_NAME} GitHub"
            " Release,"
        )
        print("verifies it against the hashes built into this source, and installs it.")
        print("\nAlready have a verified archive? Install it offline instead:")
        print(f"  {OFFLINE_COMMAND}")
        return 1

    # Hashes prove the right bytes are present. Only the real binding proves the service starts.
    try:
        verify_installation(require_evaluation=False)
    except RuntimeBundleError as error:
        print(f"\nStartup binding failed: {error}")
        print("\nReinstall the runtime bundle:")
        print(f"  {INSTALL_COMMAND}")
        print("\nor, from a verified archive you already hold:")
        print(f"  {OFFLINE_COMMAND}")
        return 1

    print("\nAll startup requirements present and the serving binding succeeds.")
    if evaluation_state != PRESENT:
        print(f"\nThe held-out evaluation is {evaluation_state}. The service will start and")
        print("score normally, but the dashboard's model panel answers 404 and the drift")
        print("endpoint has no reference. Install the complete bundle to enable them:")
        print(f"  {INSTALL_COMMAND}")
    print("\n`docker compose up --build -d` will serve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
