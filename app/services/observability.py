from __future__ import annotations

import json
import logging
from logging import Logger
from typing import Any, Dict
import time
from contextvars import ContextVar

# Context variable for per-request trace id
trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        payload: Dict[str, Any] = {
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "time": int(time.time() * 1000),
        }
        # Attach trace id from context if present
        tid = trace_id_var.get()
        if tid:
            payload["trace_id"] = tid
        # Include extra dict if provided
        if hasattr(record, "extra") and isinstance(getattr(record, "extra"), dict):
            payload.update(getattr(record, "extra"))
        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str = "app") -> Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def set_trace_id(tid: str) -> None:
    trace_id_var.set(tid)


def get_trace_id() -> str | None:
    return trace_id_var.get()


def redact_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    # Best-effort redaction for potentially sensitive fields
    redacted = dict(data)
    if "gps" in redacted and redacted["gps"]:
        redacted["gps"] = "[REDACTED]"
    if "pincode" in redacted and redacted["pincode"]:
        redacted["pincode"] = "[REDACTED]"
    if "token" in redacted:
        redacted["token"] = "[REDACTED]"
    return redacted
