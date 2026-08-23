from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url


class Settings(BaseSettings):
    """Environment-backed RiskLoom settings with secret-safe validation errors."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="RISKLOOM_",
        extra="ignore",
        hide_input_in_errors=True,
    )

    environment: Literal["local", "test"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    database_url: SecretStr
    database_connect_timeout_seconds: float = Field(default=3.0, gt=0, le=30)
    # Bounds every *statement*, not just connection establishment. Without it a database that
    # is frozen rather than refusing -- a paused container, a hung host -- leaves an already
    # pooled connection open and a live preflight waits forever instead of fail-safing to 503.
    database_command_timeout_seconds: float = Field(default=10.0, gt=0, le=60)
    razorpay_key_id: SecretStr
    razorpay_key_secret: SecretStr
    razorpay_webhook_secret: SecretStr = Field(min_length=32)
    webhook_max_body_bytes: int = Field(default=262_144, ge=1_024, le=1_048_576)

    # Day 6 live scoring. These name the locked, already-published artifacts the serving path
    # must bind to; the service fails closed at startup if any of them does not validate.
    feature_config_path: Path = Path("configs/features/default.json")
    modeling_config_path: Path = Path("configs/modeling/default.json")
    risk_model_directory: Path = Path("artifacts/models/development")
    feature_manifest_path: Path = Path(
        "artifacts/features/development-v1.1.0-config-bound/manifest.json"
    )
    # Hard cap on Razorpay order-creation attempts a single process may make. This makes the
    # project constraint of not generating bulk test-mode traffic a structural guarantee rather
    # than a promise: past the cap, an ALLOW fail-safes to REVIEW instead of attempting an order.
    # The cap counts attempts rather than successes, so a rejected attempt still consumes one.
    razorpay_max_orders_per_process: int = Field(default=5, ge=0, le=50)

    # Day 7 dashboard. The evaluation artifact is Git-ignored, so the dashboard treats its absence
    # as an ordinary state rather than an error.
    evaluation_artifact_path: Path = Path("artifacts/evaluations/development/evaluation.json")
    dashboard_static_directory: Path = Path("static")

    # Day 8 explanations. The API key is optional: a clone without one still starts and the
    # dashboard reports the feature as unconfigured, exactly as it does for a missing evaluation
    # artifact. The model id is a single value because Google revises recommended ids over time.
    gemini_api_key: SecretStr | None = None
    gemini_model: str = Field(default="gemini-3.6-flash", min_length=1, max_length=64)
    # 20s rather than 10s: the Flash tier spends thinking tokens before emitting the reply,
    # measured at ~222 thought tokens for a prompt this small.
    gemini_timeout_seconds: float = Field(default=20.0, gt=0, le=60)
    # Hard cap on Gemini calls a single process may make, mirroring the Razorpay order budget.
    # It counts attempts rather than successes: the unit is taken before the outbound request, so
    # a call that then fails still consumes one.
    gemini_max_calls_per_process: int = Field(default=5, ge=0, le=20)

    @field_validator("database_url")
    @classmethod
    def require_async_postgresql(cls, value: SecretStr) -> SecretStr:
        try:
            url = make_url(value.get_secret_value())
        except Exception as exc:
            raise ValueError("database URL must be a valid SQLAlchemy URL") from exc
        if url.drivername != "postgresql+asyncpg":
            raise ValueError("database URL must use the postgresql+asyncpg driver")
        if not url.database:
            raise ValueError("database URL must name a database")
        return value

    @field_validator("gemini_api_key", mode="before")
    @classmethod
    def treat_blank_key_as_unconfigured(cls, value: object) -> object:
        """A blank key means "not configured", not "configured with nothing".

        ``.env.example`` ships this variable empty, so a fresh clone that copies it would
        otherwise produce ``SecretStr('')`` -- truthy enough to build a client that then fails
        every call with an authentication error. Absent and blank must behave identically.
        """

        if isinstance(value, str) and not value.strip():
            return None
        if isinstance(value, SecretStr) and not value.get_secret_value().strip():
            return None
        return value

    @field_validator("razorpay_key_id")
    @classmethod
    def require_test_mode_key(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().startswith("rzp_test_"):
            raise ValueError("Razorpay key ID must be a test-mode key")
        return value

    @model_validator(mode="after")
    def require_non_empty_secrets(self) -> Self:
        if not self.razorpay_key_secret.get_secret_value().strip():
            raise ValueError("Razorpay key secret must not be empty")
        if not self.razorpay_webhook_secret.get_secret_value().strip():
            raise ValueError("Razorpay webhook secret must not be empty")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
