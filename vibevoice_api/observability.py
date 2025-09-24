from __future__ import annotations

import contextvars
import time
from contextlib import contextmanager
from typing import Optional, List

from prometheus_client import Counter, Histogram, Gauge
import json
import os
from datetime import datetime
from vibevoice_api.config import CONFIG


# Context variables for per-request context
cv_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("vibevoice_request_id", default="-")
cv_hints: contextvars.ContextVar[Optional[List[str]]] = contextvars.ContextVar("vibevoice_hints", default=None)


def set_request_id(rid: str) -> None:
    cv_request_id.set(rid)


def get_request_id() -> str:
    return cv_request_id.get()


def set_hints_container() -> None:
    cv_hints.set([])


def add_hint(msg: str) -> None:
    lst = cv_hints.get()
    if lst is None:
        lst = []
        cv_hints.set(lst)
    lst.append(msg)
    HINTS_TOTAL.labels(type=_hint_type_from_msg(msg)).inc()


def get_hints() -> List[str]:
    return cv_hints.get() or []


def clear_hints() -> None:
    cv_hints.set(None)


def _hint_type_from_msg(msg: str) -> str:
    # Best-effort label extraction: prefix before ':'
    if ":" in msg:
        return msg.split(":", 1)[0]
    return "misc"


# Prometheus metrics
REQUEST_COUNT = Counter(
    "vibevoice_requests_total", "Total HTTP requests", ["endpoint", "method", "status"]
)

REQUEST_LATENCY = Histogram(
    "vibevoice_request_latency_seconds", "HTTP request latency", ["endpoint", "method"]
)

SYNTHESIS_LATENCY = Histogram(
    "vibevoice_synthesis_latency_seconds", "Synthesis latency", ["format"]
)

ACTIVE_INFERENCES = Gauge(
    "vibevoice_active_inferences", "Active inferences"
)

ERRORS_TOTAL = Counter(
    "vibevoice_errors_total", "Synthesis errors", ["type"]
)

ATTN_FALLBACK_TOTAL = Counter(
    "vibevoice_attn_fallback_total", "Attention backend fallbacks", ["from", "to"]
)

HINTS_TOTAL = Counter(
    "vibevoice_hints_total", "Hints emitted", ["type"]
)


@contextmanager
def observe_latency(hist: Histogram, *labels: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        hist.labels(*labels).observe(dt)


def _ensure_logs_dir() -> str:
    d = CONFIG.logs_dir
    os.makedirs(d, exist_ok=True)
    return d


def append_json_log(filename: str, obj: dict) -> None:
    d = _ensure_logs_dir()
    path = os.path.join(d, filename)
    obj = dict(obj)
    obj.setdefault("ts", datetime.utcnow().isoformat() + "Z")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

