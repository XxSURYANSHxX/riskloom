import pytest
from pydantic import ValidationError

from riskloom.core.config import Settings, get_settings


def _values() -> dict[str, object]:
    return {
        "_env_file": None,
        "environment": "test",
        "database_url": "postgresql+asyncpg://riskloom:synthetic@localhost:5432/riskloom",
        "razorpay_key_id": "rzp_test_synthetic",
        "razorpay_key_secret": "synthetic_key_secret",
        "razorpay_webhook_secret": "synthetic_webhook_secret_at_least_32",
    }


def test_settings_accept_secure_test_configuration() -> None:
    settings = Settings(**_values())

    assert settings.environment == "test"
    assert settings.database_url.get_secret_value().startswith("postgresql+asyncpg://")
    assert "synthetic_key_secret" not in repr(settings)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("razorpay_key_id", "rzp_live_forbidden"),
        ("database_url", "sqlite+aiosqlite:///local.db"),
        ("database_url", "not-a-database-url"),
        ("razorpay_key_secret", ""),
        ("razorpay_webhook_secret", "too-short"),
    ],
)
def test_settings_reject_unsafe_values(field: str, value: str) -> None:
    values = _values()
    values[field] = value

    with pytest.raises(ValidationError) as exc_info:
        Settings(**values)

    if value:
        assert value not in str(exc_info.value)


def test_get_settings_reads_prefixed_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    environment = {
        "RISKLOOM_ENVIRONMENT": "test",
        "RISKLOOM_DATABASE_URL": (
            "postgresql+asyncpg://riskloom:synthetic@localhost:5432/riskloom"
        ),
        "RISKLOOM_RAZORPAY_KEY_ID": "rzp_test_from_environment",
        "RISKLOOM_RAZORPAY_KEY_SECRET": "synthetic_environment_secret",
        "RISKLOOM_RAZORPAY_WEBHOOK_SECRET": "synthetic_environment_webhook_secret",
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()

    try:
        assert get_settings().environment == "test"
    finally:
        get_settings.cache_clear()
