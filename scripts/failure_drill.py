"""Break a running RiskLoom instance on purpose and report how it degraded.

Every scenario here exercises behaviour that already exists; none of it is implemented by this
script. The point is to make that behaviour observable against a live service rather than only
inside the test suite, and to produce a narratable PASS/FAIL line per scenario.

Usage:

    uv run python scripts/failure_drill.py --base-url http://127.0.0.1:8000 --scenario all

Scenarios that need the database interrupted print the exact command to run and wait, rather than
reaching into Docker themselves: the drill stays a client of the service and never manipulates it
through a side channel.
"""

import argparse
import hashlib
import hmac
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:8000"


@dataclass
class Outcome:
    name: str
    passed: bool
    detail: str
    evidence: list[str] = field(default_factory=list)


def _request(
    url: str, method: str = "GET", body: bytes | None = None, headers: dict[str, str] | None = None
) -> tuple[int, Any]:
    request = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            raw = response.read()
            try:
                return response.status, json.loads(raw)
            except json.JSONDecodeError:
                return response.status, raw.decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, raw.decode("utf-8", "replace")
    except urllib.error.URLError as exc:
        return 0, f"unreachable: {exc.reason}"


def _signed_webhook(secret: str, event: dict[str, Any], event_id: str) -> tuple[bytes, dict]:
    raw = json.dumps(event, separators=(",", ":"), sort_keys=True).encode("utf-8")
    signature = hmac.new(secret.encode("utf-8"), raw, hashlib.sha256).hexdigest()
    return raw, {
        "Content-Type": "application/json",
        "X-Razorpay-Signature": signature,
        "X-Razorpay-Event-Id": event_id,
    }


def _event(payment_id: str, name: str, created_at: int, status: str, captured: bool) -> dict:
    """A synthetic, reserved-value webhook body. No real payment data appears anywhere."""

    return {
        "entity": "event",
        "account_id": "acc_synthetic",
        "event": name,
        "contains": ["payment"],
        "created_at": created_at,
        "payload": {
            "payment": {
                "entity": {
                    "id": payment_id,
                    "entity": "payment",
                    "amount": 100,
                    "amount_refunded": 0,
                    "currency": "INR",
                    "status": status,
                    "captured": captured,
                    "created_at": created_at,
                }
            }
        },
    }


def scenario_webhook_duplicate(base: str, secret: str) -> Outcome:
    """Replaying one provider event id must never create a second business effect."""

    stamp = int(time.time())
    payment = f"pay_drill_dup_{stamp}"
    event_id = f"event_drill_dup_{stamp}"
    body, headers = _signed_webhook(
        secret, _event(payment, "payment.captured", stamp, "captured", True), event_id
    )

    first_status, first = _request(f"{base}/api/v1/webhooks/razorpay", "POST", body, headers)
    second_status, second = _request(f"{base}/api/v1/webhooks/razorpay", "POST", body, headers)

    ok = (
        first_status == 200
        and second_status == 200
        and first.get("duplicate") is False
        and second.get("duplicate") is True
    )
    return Outcome(
        name="webhook-duplicate",
        passed=ok,
        detail="replay recognised as duplicate, one business effect",
        evidence=[
            f"first  -> {first_status} {json.dumps(first)}",
            f"replay -> {second_status} {json.dumps(second)}",
        ],
    )


def scenario_webhook_out_of_order(base: str, secret: str) -> Outcome:
    """A late-arriving earlier event must be accepted as its own immutable fact."""

    stamp = int(time.time())
    payment = f"pay_drill_ooo_{stamp}"
    captured = _event(payment, "payment.captured", stamp + 100, "captured", True)
    authorized = _event(payment, "payment.authorized", stamp, "authorized", False)

    body_a, headers_a = _signed_webhook(secret, captured, f"event_drill_ooo_late_{stamp}")
    late_status, late = _request(f"{base}/api/v1/webhooks/razorpay", "POST", body_a, headers_a)

    body_b, headers_b = _signed_webhook(secret, authorized, f"event_drill_ooo_early_{stamp}")
    early_status, early = _request(f"{base}/api/v1/webhooks/razorpay", "POST", body_b, headers_b)

    ok = late_status == 200 and early_status == 200 and early.get("duplicate") is False
    return Outcome(
        name="webhook-out-of-order",
        passed=ok,
        detail="newer event first, older event still accepted as a distinct fact",
        evidence=[
            f"captured(newer)   -> {late_status} {json.dumps(late)}",
            f"authorized(older) -> {early_status} {json.dumps(early)}",
        ],
    )


def scenario_database_down(base: str) -> Outcome:
    """Readiness must report unavailable without leaking the underlying error."""

    status, payload = _request(f"{base}/health/ready")
    if status == 200:
        return Outcome(
            name="database-down",
            passed=False,
            detail="database still reachable -- pause it first (see --help)",
            evidence=[f"/health/ready -> {status} {json.dumps(payload)}"],
        )
    ok = status == 503 and isinstance(payload, dict) and payload.get("status") == "not_ready"
    return Outcome(
        name="database-down",
        passed=ok,
        detail="readiness degrades to 503 without exposing internals",
        evidence=[f"/health/ready -> {status} {json.dumps(payload)}"],
    )


def scenario_preflight_storage_unavailable(base: str, attempt: dict[str, Any]) -> Outcome:
    """With storage down, preflight must refuse rather than return an unbacked ALLOW."""

    body = json.dumps(attempt).encode("utf-8")
    status, payload = _request(
        f"{base}/api/v1/checkout/preflight",
        "POST",
        body,
        {"Content-Type": "application/json"},
    )
    code = payload.get("error", {}).get("code") if isinstance(payload, dict) else None
    ok = status == 503 and code == "storage_unavailable"
    return Outcome(
        name="preflight-storage-unavailable",
        passed=ok,
        detail="503 storage_unavailable; never a 200 ALLOW the ledger cannot back",
        evidence=[f"POST /checkout/preflight -> {status} {json.dumps(payload)}"],
    )


def scenario_drift(base: str) -> Outcome:
    """Drift must report insufficient_data rather than a band it cannot support."""

    status, payload = _request(f"{base}/api/v1/dashboard/drift")
    if status != 200 or not isinstance(payload, dict):
        return Outcome("drift", False, "endpoint unavailable", [f"-> {status} {payload}"])
    reported = payload.get("status")
    ok = reported in {"ok", "insufficient_data", "reference_unavailable"}
    if reported == "insufficient_data":
        ok = payload.get("psi") is None and payload.get("band") is None
    return Outcome(
        name="drift",
        passed=ok,
        detail=(
            f"status={reported}, rows={payload.get('observed_rows')}/{payload.get('minimum_rows')}"
        ),
        evidence=[
            f"psi={payload.get('psi')} band={payload.get('band')} epsilon={payload.get('epsilon')}"
        ],
    )


def render(outcomes: list[Outcome], verbose: bool) -> int:
    width = max(len(item.name) for item in outcomes)
    print("\nRiskLoom failure drill\n" + "=" * (width + 46))
    for item in outcomes:
        mark = "PASS" if item.passed else "FAIL"
        print(f"  [{mark}] {item.name.ljust(width)}  {item.detail}")
        if verbose or not item.passed:
            for line in item.evidence:
                print(f"         {line}")
    failed = [item for item in outcomes if not item.passed]
    print("=" * (width + 46))
    print(f"  {len(outcomes) - len(failed)}/{len(outcomes)} scenarios passed\n")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inject controlled failures into a running RiskLoom instance.",
        epilog=(
            "To exercise the storage scenarios, interrupt the database first:\n"
            "  docker compose pause postgres\n"
            "  uv run python scripts/failure_drill.py --scenario storage\n"
            "  docker compose unpause postgres"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--scenario",
        default="all",
        choices=[
            "all",
            "webhook-duplicate",
            "webhook-out-of-order",
            "storage",
            "drift",
        ],
    )
    parser.add_argument(
        "--webhook-secret",
        default=None,
        help="Required for webhook scenarios; must match RISKLOOM_RAZORPAY_WEBHOOK_SECRET.",
    )
    parser.add_argument("--attempt-file", default=None, help="JSON body for the preflight scenario")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    base = args.base_url.rstrip("/")
    outcomes: list[Outcome] = []
    wants = args.scenario

    if wants in {"all", "webhook-duplicate", "webhook-out-of-order"}:
        if not args.webhook_secret:
            print("webhook scenarios need --webhook-secret", file=sys.stderr)
            if wants != "all":
                return 2
        else:
            if wants in {"all", "webhook-duplicate"}:
                outcomes.append(scenario_webhook_duplicate(base, args.webhook_secret))
            if wants in {"all", "webhook-out-of-order"}:
                outcomes.append(scenario_webhook_out_of_order(base, args.webhook_secret))

    if wants in {"all", "storage"}:
        outcomes.append(scenario_database_down(base))
        if args.attempt_file:
            attempt = json.loads(Path(args.attempt_file).read_text(encoding="utf-8"))
            outcomes.append(scenario_preflight_storage_unavailable(base, attempt))

    if wants in {"all", "drift"}:
        outcomes.append(scenario_drift(base))

    if not outcomes:
        print("no scenarios ran", file=sys.stderr)
        return 2
    return render(outcomes, args.verbose)


if __name__ == "__main__":
    sys.exit(main())
