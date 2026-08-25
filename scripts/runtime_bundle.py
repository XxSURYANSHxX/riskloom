"""Build, install and verify the RiskLoom runtime artifact bundle.

The bundle carries the five locked JSON artifacts a clean clone cannot obtain from Git. Everything
this command can reach is fixed in source: the release URL, the member allowlist and the expected
SHA-256 of every file. There is no argument that widens any of them.

    uv run python scripts/runtime_bundle.py build --output dist/riskloom-runtime-artifacts.zip
    uv run python scripts/runtime_bundle.py install
    uv run python scripts/runtime_bundle.py install --archive C:\\path\\to\\bundle.zip
    uv run python scripts/runtime_bundle.py verify --require-evaluation

``install`` without ``--archive`` downloads the one fixed release asset; with ``--archive`` it makes
no network request at all. Failures print a short stable identity and never artifact contents,
secrets or upstream response bodies.

Two tags are involved and they are deliberately different. The runtime artifacts live on their own
immutable release tag; the source snapshot carries the submission tag. Nothing here assumes they
match, so the source can be re-tagged without invalidating a published asset.

The runtime release is not published yet, so ``install`` without ``--archive`` currently fails with
``runtime_bundle_download_failed``. Use ``--archive`` until it exists.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from riskloom.runtime_bundle import (  # noqa: E402
    ASSET_NAME,
    BUNDLE_MEMBERS,
    EVALUATION_MEMBER,
    RELEASE_TAG,
    RELEASE_URL,
    STARTUP_MEMBERS,
    SUBMISSION_TAG,
    InstallationResult,
    RuntimeBundleError,
    build_bundle,
    install_from_archive,
    install_from_release,
    verify_installation,
)

DEFAULT_OUTPUT = Path("dist") / ASSET_NAME


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python scripts/runtime_bundle.py")
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="package the approved local artifacts")
    build.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    install = commands.add_parser("install", help="install the runtime artifacts")
    install.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="install from a local archive instead of downloading (no network access)",
    )
    install.add_argument(
        "--startup-only",
        action="store_true",
        help="install only the four startup artifacts, omitting the held-out evaluation",
    )

    verify = commands.add_parser("verify", help="check installed artifacts and the serving binding")
    verify.add_argument(
        "--require-evaluation",
        action="store_true",
        help="also require the held-out evaluation used by the model panel and drift reference",
    )
    return parser


def _report(result: InstallationResult) -> None:
    for member in result.installed:
        print(f"  installed  {member}")
    for member in result.already_present:
        print(f"  unchanged  {member}")
    if not result.evaluation_installed:
        print(f"\n  omitted    {EVALUATION_MEMBER}")
        print("             the dashboard model panel and drift reference will be unavailable")


def _build(output: Path) -> int:
    print(f"RiskLoom runtime bundle: building {len(BUNDLE_MEMBERS)} approved artifacts\n")
    path = build_bundle(output)
    print(f"  wrote {path}")
    print(f"\nIntended runtime release asset: {RELEASE_TAG}/{ASSET_NAME}")
    print(f"The source snapshot is tagged separately as {SUBMISSION_TAG}; the two are independent,")
    print("so the source can be re-tagged without invalidating a published runtime asset.")
    print("The archive is deterministic: the same approved inputs always produce these bytes.")
    print("\nThe runtime release is not published yet; use `install --archive` until it exists.")
    return 0


def _install(archive: Path | None, startup_only: bool) -> int:
    require_evaluation = not startup_only
    if archive is None:
        print("RiskLoom runtime bundle: downloading the fixed release asset")
        print(f"  {RELEASE_URL}\n")
        result = install_from_release(require_evaluation=require_evaluation)
    else:
        print(f"RiskLoom runtime bundle: installing from {archive} (no network access)\n")
        result = install_from_archive(archive, require_evaluation=require_evaluation)
    _report(result)
    print("\nServing binding verified. `docker compose up --build -d` will serve.")
    return 0


def _verify(require_evaluation: bool) -> int:
    members = BUNDLE_MEMBERS if require_evaluation else STARTUP_MEMBERS
    print(f"RiskLoom runtime bundle: verifying {len(members)} artifacts and the serving binding\n")
    verify_installation(require_evaluation=require_evaluation)
    for member in members:
        print(f"  ok  {member}")
    print("\nAll pinned hashes match and the serving binding succeeds.")
    return 0


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = _parser().parse_args(arguments)
    try:
        if parsed.command == "build":
            return _build(parsed.output)
        if parsed.command == "install":
            return _install(parsed.archive, parsed.startup_only)
        return _verify(parsed.require_evaluation)
    except RuntimeBundleError as error:
        print(f"\nrefused: {error}", file=sys.stderr)
        if parsed.command != "build":
            print(
                "\nInstall the published runtime bundle with:\n"
                "  uv run python scripts/runtime_bundle.py install",
                file=sys.stderr,
            )
        return 1


if __name__ == "__main__":
    sys.exit(main())
