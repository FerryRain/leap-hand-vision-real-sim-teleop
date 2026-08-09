"""Versioned JSON protocol shared by the Franka bridge server and client."""

from __future__ import annotations

import json
import math
from enum import Enum
from typing import Any, Sequence

PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 64 * 1024


class ProtocolError(ValueError):
    """Raised when a peer sends a malformed or unsupported message."""


def decode_message(raw: str | bytes) -> dict[str, Any]:
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ProtocolError("binary message is not valid UTF-8") from error
    if len(raw.encode("utf-8")) > MAX_MESSAGE_BYTES:
        raise ProtocolError("message exceeds 64 KiB limit")
    try:
        value = json.loads(
            raw,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ProtocolError(f"non-finite JSON number: {token}")
            ),
        )
    except json.JSONDecodeError as error:
        raise ProtocolError(f"invalid JSON: {error.msg}") from error
    if not isinstance(value, dict):
        raise ProtocolError("message must be a JSON object")
    message_type = value.get("type")
    if not isinstance(message_type, str) or not message_type:
        raise ProtocolError("message.type must be a non-empty string")
    return value


def encode_message(message: dict[str, Any]) -> str:
    return json.dumps(
        json_safe(message),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def vector3(message: dict[str, Any], key: str) -> tuple[float, float, float]:
    value = message.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ProtocolError(f"{key} must contain exactly three numbers")
    try:
        vector = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ProtocolError(f"{key} must contain numbers") from error
    if not all(math.isfinite(item) for item in vector):
        raise ProtocolError(f"{key} must contain finite numbers")
    return vector  # type: ignore[return-value]


def finite_float(
    message: dict[str, Any],
    key: str,
    *,
    default: float | None = None,
) -> float:
    value = message.get(key, default)
    if value is None:
        raise ProtocolError(f"missing numeric field {key}")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ProtocolError(f"{key} must be numeric") from error
    if not math.isfinite(result):
        raise ProtocolError(f"{key} must be finite")
    return result


def positive_int(message: dict[str, Any], key: str) -> int:
    value = message.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ProtocolError(f"{key} must be a non-negative integer")
    return value


def vector_norm(vector: Sequence[float]) -> float:
    return math.sqrt(math.fsum(float(value) ** 2 for value in vector))


def ack(request: dict[str, Any], **fields: Any) -> dict[str, Any]:
    response: dict[str, Any] = {
        "type": "ack",
        "for": request.get("type"),
    }
    request_id = request.get("request_id")
    if isinstance(request_id, str):
        response["request_id"] = request_id
    response.update(fields)
    return response


def error_response(
    request: dict[str, Any] | None,
    code: str,
    message: str,
) -> dict[str, Any]:
    response: dict[str, Any] = {
        "type": "error",
        "code": code,
        "message": message,
    }
    if request is not None:
        response["for"] = request.get("type")
        if isinstance(request.get("request_id"), str):
            response["request_id"] = request["request_id"]
    return response


def json_safe(value: Any) -> Any:
    """Convert Franky, NumPy, enum, and nested values into strict JSON data."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return json_safe(item_method())
        except (TypeError, ValueError):
            pass
    tolist_method = getattr(value, "tolist", None)
    if callable(tolist_method):
        try:
            return json_safe(tolist_method())
        except (TypeError, ValueError):
            pass
    return str(value)
