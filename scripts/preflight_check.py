"""Report whether this checkout can actually start the service.

A fresh ``git clone`` cannot. Every artifact the application binds to at startup is Git-ignored, so
a clone contains the configuration files and no model at all, and the container will exit with a
``serving_*`` identity rather than serve. That is correct fail-closed behaviour, but it is a
confusing first experience, so this script names the missing files before anything is started.

Run it before ``docker compose up``. It reads only; it creates nothing and downloads nothing.
"""

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

REQUIRED = (
    ("configs/features/default.json", "feature configuration", True),
    ("configs/modeling/default.json", "modeling configuration", True),
    ("artifacts/models/development/model.json", "locked model", False),
    ("artifacts/models/development/manifest.json", "locked model manifest", False),
    (
        "artifacts/features/development-v1.1.0-config-bound/manifest.json",
        "locked feature manifest",
        False,
    ),
)

OPTIONAL = (
    (
        "artifacts/evaluations/development/evaluation.json",
        "held-out evaluation (drift reference and model panel)",
    ),
    (".env", "runtime configuration"),
)


def main() -> int:
    print("RiskLoom startup preflight\n")
    missing_required: list[str] = []

    print("required to start:")
    for relative, description, tracked in REQUIRED:
        path = REPOSITORY_ROOT / relative
        present = path.exists()
        if not present:
            missing_required.append(relative)
        origin = "in git" if tracked else "git-ignored input"
        print(f"  [{'ok' if present else '--'}] {relative}")
        print(f"       {description} ({origin})")

    print("\noptional:")
    for relative, description in OPTIONAL:
        path = REPOSITORY_ROOT / relative
        state = "ok" if path.exists() else "absent"
        print(f"  [{state:>6}] {relative}  -- {description}")

    if not missing_required:
        print("\nAll startup requirements present. `docker compose up --build -d` will serve.")
        return 0

    print(f"\n{len(missing_required)} required file(s) missing. The service will refuse to start.")
    print("\nThese are locked artifacts, deliberately excluded from version control. They are")
    print("inputs to the image, never built into it, and they cannot be regenerated here:")
    print("retraining would produce a different model and break the locked-artifact contract.")
    print("\nCopy the artifact bundle into place, preserving these paths:")
    for relative in missing_required:
        print(f"  {relative}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
