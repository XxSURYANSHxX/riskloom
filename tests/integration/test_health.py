from collections.abc import AsyncIterator

import pytest
from fastapi.testclient import TestClient

from riskloom.api.dependencies import get_database
from riskloom.core.config import Settings
from riskloom.main import create_app

pytestmark = pytest.mark.integration


def test_health_endpoints_and_no_extra_public_routes(integration_settings: Settings) -> None:
    with TestClient(create_app(integration_settings)) as client:
        assert client.get("/health/live").json() == {"status": "ok"}

        ready = client.get("/health/ready")
        assert ready.status_code == 200
        assert ready.json() == {"status": "ready", "checks": {"database": "ok"}}

        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404


def test_readiness_hides_database_failure(integration_settings: Settings) -> None:
    class UnavailableDatabase:
        async def ping(self) -> None:
            raise RuntimeError("synthetic database failure")

        async def sessions(self) -> AsyncIterator[object]:
            if False:
                yield object()

        async def close(self) -> None:
            return None

    app = create_app(integration_settings)
    app.dependency_overrides[get_database] = UnavailableDatabase

    with TestClient(app) as client:
        response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "checks": {"database": "unavailable"},
    }
    assert "synthetic database failure" not in response.text
