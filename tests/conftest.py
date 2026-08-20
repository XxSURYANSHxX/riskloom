import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from alembic.config import Config
from testcontainers.community.postgres import PostgresContainer

from alembic import command
from riskloom.core.config import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_WEBHOOK_SECRET = "synthetic_webhook_secret_for_tests_only"


def build_settings(database_url: str) -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        log_level="INFO",
        database_url=database_url,
        database_connect_timeout_seconds=3,
        razorpay_key_id="rzp_test_synthetic_key",
        razorpay_key_secret="synthetic_key_secret",
        razorpay_webhook_secret=SYNTHETIC_WEBHOOK_SECRET,
        webhook_max_body_bytes=4_096,
    )


def make_alembic_config(database_url: str) -> Config:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    return config


@pytest.fixture
def unit_settings() -> Settings:
    return build_settings("postgresql+asyncpg://riskloom:synthetic@127.0.0.1:55432/riskloom")


@pytest.fixture
def synthetic_event() -> dict[str, Any]:
    return {
        "entity": "event",
        "account_id": "acc_synthetic",
        "event": "payment.failed",
        "contains": ["payment"],
        "created_at": 1_700_000_100,
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_synthetic",
                    "entity": "payment",
                    "amount": 100,
                    "amount_refunded": 0,
                    "currency": "INR",
                    "status": "failed",
                    "order_id": "order_synthetic",
                    "invoice_id": None,
                    "international": False,
                    "method": "card",
                    "captured": False,
                    "created_at": 1_700_000_000,
                    "refund_status": None,
                    "error_code": "BAD_REQUEST_ERROR",
                    "error_source": "customer",
                    "error_step": "payment_authentication",
                    "error_reason": "incorrect_card_details",
                    "email": "person@example.invalid",
                    "contact": "+10000000000",
                    "ip_address": "192.0.2.1",
                    "description": "synthetic free form description",
                    "notes": {"free_form": "synthetic note"},
                    "token_id": "token_synthetic",
                    "vpa": "synthetic@invalid",
                    "acquirer_data": {"auth_code": "synthetic"},
                    "card": {
                        "number": "synthetic-pan-marker",
                        "cvv": "synthetic-cvv-marker",
                        "name": "Synthetic Cardholder",
                    },
                    "unknown_nested": {"unsafe": "must not persist"},
                }
            },
            "unknown_entity": {"entity": {"unsafe": "must not persist"}},
        },
        "unexpected_top_level": {"unsafe": "must not persist"},
    }


@pytest.fixture
def synthetic_raw_body(synthetic_event: dict[str, Any]) -> bytes:
    return json.dumps(synthetic_event, separators=(",", ":")).encode("utf-8")


@pytest.fixture(scope="session")
def postgres_url() -> Iterator[str]:
    container = PostgresContainer(
        image="postgres:16-alpine",
        username="riskloom_test",
        password="synthetic_database_password",
        dbname="riskloom_test",
        driver="asyncpg",
    )
    with container:
        yield container.get_connection_url()


@pytest.fixture(scope="session")
def migrated_database_url(postgres_url: str) -> Iterator[str]:
    config = make_alembic_config(postgres_url)
    command.upgrade(config, "head")
    yield postgres_url
    command.downgrade(config, "base")


@pytest.fixture
def integration_settings(migrated_database_url: str) -> Settings:
    return build_settings(migrated_database_url)
