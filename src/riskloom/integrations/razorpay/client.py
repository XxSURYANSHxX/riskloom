from types import TracebackType

import httpx

from riskloom.core.config import Settings
from riskloom.integrations.razorpay.schemas import CreateOrderInput, RazorpayOrder

RAZORPAY_API_BASE_URL = "https://api.razorpay.com"


class RazorpayOrdersError(Exception):
    """Safe internal error whose message never includes an upstream body or credentials."""


class RazorpayOrdersClient:
    def __init__(
        self,
        settings: Settings,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            base_url=RAZORPAY_API_BASE_URL,
            auth=httpx.BasicAuth(
                settings.razorpay_key_id.get_secret_value(),
                settings.razorpay_key_secret.get_secret_value(),
            ),
            timeout=httpx.Timeout(5.0, connect=2.0),
            trust_env=False,
            headers={"Content-Type": "application/json"},
        )

    async def create_order(self, amount: int, currency: str, receipt: str) -> RazorpayOrder:
        request = CreateOrderInput(amount=amount, currency=currency, receipt=receipt)
        try:
            response = await self._client.post("/v1/orders", json=request.model_dump())
        except httpx.TimeoutException as exc:
            raise RazorpayOrdersError("razorpay_orders_timeout") from exc
        except httpx.RequestError as exc:
            raise RazorpayOrdersError("razorpay_orders_unavailable") from exc

        if response.is_error:
            raise RazorpayOrdersError("razorpay_orders_rejected")
        try:
            return RazorpayOrder.model_validate(response.json())
        except ValueError as exc:
            raise RazorpayOrdersError("razorpay_orders_invalid_response") from exc

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "RazorpayOrdersClient":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()
