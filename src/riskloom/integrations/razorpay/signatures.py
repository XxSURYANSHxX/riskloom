import hashlib
import hmac
import re

from pydantic import SecretStr

_HEX_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def verify_webhook_signature(
    raw_body: bytes,
    received_signature: str,
    webhook_secret: SecretStr,
) -> bool:
    """Verify Razorpay's HMAC-SHA256 signature over the exact request bytes."""

    if _HEX_SHA256_PATTERN.fullmatch(received_signature) is None:
        return False
    expected = hmac.new(
        webhook_secret.get_secret_value().encode("utf-8"),
        raw_body,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, received_signature.casefold())


def sha256_digest(raw_body: bytes) -> str:
    return hashlib.sha256(raw_body).hexdigest()
