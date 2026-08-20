import base64

import httpx
import pytest
import respx
from pydantic import ValidationError

from riskloom.core.config import Settings
from riskloom.integrations.razorpay.client import (
    RAZORPAY_API_BASE_URL,
    RazorpayOrdersClient,
    RazorpayOrdersError,
)


def _order_response() -> dict[str, object]:
    return {
        "id": "order_synthetic",
        "entity": "order",
        "amount": 5000,
        "amount_paid": 0,
        "amount_due": 5000,
        "currency": "INR",
        "receipt": "receipt-synthetic",
        "status": "created",
        "attempts": 0,
        "created_at": 1_700_000_000,
        "notes": {"ignored": "not retained"},
    }


@pytest.mark.asyncio
@respx.mock
async def test_orders_client_sends_typed_test_mode_request(unit_settings: Settings) -> None:
    route = respx.post(f"{RAZORPAY_API_BASE_URL}/v1/orders").mock(
        return_value=httpx.Response(200, json=_order_response())
    )

    async with RazorpayOrdersClient(unit_settings) as client:
        order = await client.create_order(5000, "INR", "receipt-synthetic")

    assert order.id == "order_synthetic"
    request = route.calls.last.request
    assert request.content == b'{"amount":5000,"currency":"INR","receipt":"receipt-synthetic"}'
    expected = base64.b64encode(b"rzp_test_synthetic_key:synthetic_key_secret").decode("ascii")
    assert request.headers["Authorization"] == f"Basic {expected}"


@pytest.mark.asyncio
@respx.mock
async def test_orders_client_maps_timeout_to_safe_error(unit_settings: Settings) -> None:
    respx.post(f"{RAZORPAY_API_BASE_URL}/v1/orders").mock(
        side_effect=httpx.ReadTimeout("synthetic timeout")
    )

    async with RazorpayOrdersClient(unit_settings) as client:
        with pytest.raises(RazorpayOrdersError, match="razorpay_orders_timeout"):
            await client.create_order(5000, "INR", "receipt-synthetic")


@pytest.mark.asyncio
@respx.mock
async def test_orders_client_hides_upstream_error_body(unit_settings: Settings) -> None:
    sensitive_marker = "person@example.invalid"
    respx.post(f"{RAZORPAY_API_BASE_URL}/v1/orders").mock(
        return_value=httpx.Response(400, json={"error": sensitive_marker})
    )

    async with RazorpayOrdersClient(unit_settings) as client:
        with pytest.raises(RazorpayOrdersError) as exc_info:
            await client.create_order(5000, "INR", "receipt-synthetic")

    assert str(exc_info.value) == "razorpay_orders_rejected"
    assert sensitive_marker not in str(exc_info.value)


@pytest.mark.asyncio
@respx.mock
async def test_orders_client_rejects_invalid_response(unit_settings: Settings) -> None:
    respx.post(f"{RAZORPAY_API_BASE_URL}/v1/orders").mock(
        return_value=httpx.Response(200, json={"entity": "order"})
    )

    async with RazorpayOrdersClient(unit_settings) as client:
        with pytest.raises(RazorpayOrdersError, match="razorpay_orders_invalid_response"):
            await client.create_order(5000, "INR", "receipt-synthetic")


@pytest.mark.asyncio
async def test_orders_client_validates_input_before_network(unit_settings: Settings) -> None:
    async with RazorpayOrdersClient(unit_settings) as client:
        with pytest.raises(ValidationError):
            await client.create_order(0, "inr", "")
