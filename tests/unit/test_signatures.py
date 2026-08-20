import hashlib
import hmac

from pydantic import SecretStr

from riskloom.integrations.razorpay.signatures import sha256_digest, verify_webhook_signature


def _signature(body: bytes, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def test_signature_uses_exact_raw_bytes(synthetic_raw_body: bytes) -> None:
    secret_value = "synthetic_webhook_secret_for_tests_only"
    secret = SecretStr(secret_value)
    signature = _signature(synthetic_raw_body, secret_value)

    assert verify_webhook_signature(synthetic_raw_body, signature, secret)
    assert verify_webhook_signature(synthetic_raw_body, signature.upper(), secret)
    assert not verify_webhook_signature(synthetic_raw_body + b" ", signature, secret)


def test_signature_rejects_non_hex_value(synthetic_raw_body: bytes) -> None:
    secret = SecretStr("synthetic_webhook_secret_for_tests_only")

    assert not verify_webhook_signature(synthetic_raw_body, "not-a-signature", secret)


def test_body_digest_is_exact(synthetic_raw_body: bytes) -> None:
    assert sha256_digest(synthetic_raw_body) == hashlib.sha256(synthetic_raw_body).hexdigest()
    assert sha256_digest(synthetic_raw_body) != sha256_digest(synthetic_raw_body + b" ")
