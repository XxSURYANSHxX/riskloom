import logging
import re
import sys
from collections.abc import Mapping
from typing import Any, cast
from uuid import uuid4

import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars
from structlog.types import EventDict, Processor

REDACTED = "[REDACTED]"
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_SENSITIVE_KEY_PARTS = (
    "authorization",
    "body",
    "cardholder",
    "contact",
    "credential",
    "cvv",
    "email",
    "ip_address",
    "key_secret",
    "password",
    "payload",
    "phone",
    "raw",
    "secret",
    "signature",
    "token",
    "vpa",
)


def _is_sensitive_key(key: str) -> bool:
    normalized = key.casefold().replace("-", "_")
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def _redact_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): REDACTED if _is_sensitive_key(str(key)) else _redact_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_value(item) for item in value]
    return value


def redact_sensitive_fields(_logger: Any, _method_name: str, event_dict: EventDict) -> EventDict:
    return cast(EventDict, _redact_value(event_dict))


def configure_logging(level: str) -> None:
    """Configure stdlib and structlog output as one-line JSON."""

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        timestamper,
        redact_sensitive_fields,
    ]
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level, force=True)
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=False,
    )


def request_id_from_header(value: str | None) -> str:
    if value is not None and _REQUEST_ID_PATTERN.fullmatch(value):
        return value
    return str(uuid4())


def bind_request_context(request_id: str) -> None:
    clear_contextvars()
    bind_contextvars(request_id=request_id)


def clear_request_context() -> None:
    clear_contextvars()
