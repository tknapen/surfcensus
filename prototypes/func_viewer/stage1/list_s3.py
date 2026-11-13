"""Anonymous, read-only requests to the two public buckets, at most 4 at a time, and counted.

A helper of list_runs.py, not a script. Every request of the listing goes through get(): it
holds a semaphore of MAX_REQUESTS, so the limit holds whatever the callers do with threads, and
it counts every request that is sent, retries included. Nothing is signed and no credential is
read: the requests carry a User-Agent header and nothing else of the caller.
"""

from __future__ import annotations

import sys
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

sys.path.append(str(Path(__file__).resolve().parents[1] / "survey"))  # survey.py, also under -I
from survey import DERIV, NS, RAW

__all__ = ["DERIV", "MAX_REQUESTS", "RAW", "get", "head", "list_prefix", "requests_made"]

MAX_REQUESTS = 4
RETRY_STATUS = (500, 502, 503, 504)
ATTEMPTS = 7

_slots = threading.BoundedSemaphore(MAX_REQUESTS)
_lock = threading.Lock()
_count = 0
_client = httpx.Client(timeout=60.0, headers={"User-Agent": "listing-survey/0.2"})


def requests_made() -> int:
    """How many requests this process has sent so far, retries included."""
    with _lock:
        return _count


def get(url: str, params: dict | None = None, method: str = "GET") -> httpx.Response:
    """GET (or HEAD) with retries on 5xx (S3 answers SlowDown with 503) and transport errors."""
    global _count
    last: object = None
    for attempt in range(ATTEMPTS):
        with _slots:
            with _lock:
                _count += 1
            try:
                r = _client.request(method, url, params=params)
            except httpx.HTTPError as exc:
                last = exc
                r = None
        if r is not None and r.status_code not in RETRY_STATUS:
            return r
        if r is not None:
            last = f"HTTP {r.status_code}"
        time.sleep(min(1.5 * 2**attempt, 30.0))  # outside the semaphore: a wait holds no slot
    raise RuntimeError(f"giving up on {url} {params}: {last}")


def head(base: str, key: str) -> tuple[int, int | None, str | None]:
    """(status, size, ETag without quotes) of one object, from a HEAD request."""
    r = get(f"{base}/{key}", method="HEAD")
    size = r.headers.get("content-length")
    etag = r.headers.get("etag")
    return r.status_code, int(size) if size else None, etag.strip('"') if etag else None


def list_prefix(base: str, prefix: str, delimiter: str | None = None):
    """Every key under a prefix: ([(key, size, etag, last_modified)], [common prefixes]).

    The ETag is returned without the quotes that S3 puts around it. Pages are read one after
    the other, since each needs the token of the one before.
    """
    keys: list[tuple[str, int, str, str]] = []
    prefixes: list[str] = []
    token = None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if delimiter:
            params["delimiter"] = delimiter
        if token:
            params["continuation-token"] = token
        r = get(base, params)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        for c in root.findall(f"{NS}Contents"):
            keys.append(
                (
                    c.findtext(f"{NS}Key") or "",
                    int(c.findtext(f"{NS}Size") or 0),
                    (c.findtext(f"{NS}ETag") or "").strip('"'),
                    c.findtext(f"{NS}LastModified") or "",
                )
            )
        for c in root.findall(f"{NS}CommonPrefixes"):
            prefixes.append(c.findtext(f"{NS}Prefix") or "")
        if root.findtext(f"{NS}IsTruncated") != "true":
            return keys, prefixes
        token = root.findtext(f"{NS}NextContinuationToken")
