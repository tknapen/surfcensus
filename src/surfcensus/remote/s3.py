"""An anonymous, read-only S3 client: listing, HEAD, ranged GET, download (work package RA).

Laid down by work package R0 (docs/dev/remote_plan.md): the records, the constructor, which
puts the guard of ``remote/transport.py`` under every request, and the signatures with the
behaviour each method must have. The methods are written by RA.

Behaviour that every method shares:

* Requests go through ``self.http``, an ``httpx.Client`` on the guarded transport with
  ``trust_env=False``. Nothing else of this module opens a connection.
* At most ``config.max_concurrency`` requests of this client at once (a semaphore of the
  client), under the limit of the process (``remote.MAX_CONCURRENCY``, held by the
  transport).
* Retries: statuses of ``remote.RETRIED_STATUSES`` (``SlowDown`` among them), connection
  errors, read timeouts, a body shorter than its ``Content-Length`` and a body whose MD5 is
  not its ETag (an ETag without ``-``) are tried again up to ``config.max_retries`` times,
  after :func:`backoff_delay` seconds slept through ``self.sleep``. When they run out the
  call raises ``SourceError("unreachable", ...)``. 400, 403, 404, 412 and 416 are not
  retried; an object request maps them through ``remote.STATUS_REASONS``.
* Errors are ``SourceError`` made with ``remote.source_error``: the detail names the key's
  place relative to the subject directory, or ``listing``, and no URL, bucket key or ETag.
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx

from surfcensus.config import RemoteConfig
from surfcensus.remote.transport import ReadOnlyTransport, default_transport, read_only

_RA = "work package RA: remote/s3.py"

Addressing = Literal["virtual", "path"]

S3_NAMESPACE = "http://s3.amazonaws.com/doc/2006-03-01/"
"""The XML namespace of ListObjectsV2 answers and of error bodies."""

RETRY_BASE_S = 0.5
"""The first backoff before full jitter, doubled per attempt."""

RETRY_CAP_S = 30.0
"""The longest backoff, also for a ``Retry-After`` that asks for more."""

CHUNK_BYTES = 1 << 20
"""A download writes and checks its cancel event per chunk of this size."""


@dataclass(frozen=True)
class S3Object:
    """One object of a listing."""

    key: str
    """The whole key in the bucket. Never written outside ``catalog.sqlite``."""
    size: int
    etag: str
    """Without quotes. The MD5 of the content when it holds no ``-`` (one-part upload)."""
    last_modified: str
    """``LastModified`` of the listing, ISO 8601 as S3 writes it."""


@dataclass(frozen=True)
class S3Head:
    """What the answer to a HEAD or GET says about the object."""

    size: int
    """The size of the whole object (from ``Content-Range`` for a ranged GET)."""
    etag: str
    """Without quotes."""
    last_modified: str
    """``Last-Modified`` of the answer, as sent."""
    version_id: str | None
    """``x-amz-version-id`` when the bucket sends it (to verify on the real bucket)."""


def backoff_delay(
    attempt: int, retry_after: float | None = None, *, rng: random.Random | None = None
) -> float:
    """Seconds to wait before retry number ``attempt`` (1 for the first): full jitter,
    uniform in ``[0, min(RETRY_CAP_S, RETRY_BASE_S * 2 ** (attempt - 1))]``, or
    ``retry_after`` when the answer sent one, at most ``RETRY_CAP_S``."""
    raise NotImplementedError(_RA)


class S3Client:
    """Anonymous, read-only access to one bucket.

    ``addressing`` defaults to ``path`` for a bucket name with a dot (``openneuro.org``) and
    ``virtual`` otherwise. ``transport=None`` takes ``remote.default_transport()``; a
    transport given (the fake of the tests) is put behind the guard as well. ``sleep`` is
    what a retry waits with, so a test can count the waits instead of sleeping.
    """

    def __init__(
        self,
        bucket: str,
        *,
        config: RemoteConfig,
        addressing: Addressing | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not bucket or "/" in bucket:
            raise ValueError(f"not a bucket name: {bucket!r}")
        self.bucket = bucket
        self.config = config
        self.addressing: Addressing = addressing or ("path" if "." in bucket else "virtual")
        self.sleep = sleep
        self.transport: ReadOnlyTransport = read_only(
            transport if transport is not None else default_transport()
        )
        # trust_env=False: proxies from the environment would be mounted beside the
        # transport and pass around the guard; default_transport() takes the proxy itself
        self.http = httpx.Client(
            transport=self.transport,
            trust_env=False,
            follow_redirects=False,
            timeout=config.timeout_s,
        )
        self.slots = threading.BoundedSemaphore(config.max_concurrency)
        """At most ``config.max_concurrency`` requests of this client at once."""

    @property
    def endpoint(self) -> str:
        """``https://<bucket>.s3.amazonaws.com`` (virtual) or
        ``https://s3.amazonaws.com/<bucket>`` (path); a key is appended after a slash."""
        if self.addressing == "virtual":
            return f"https://{self.bucket}.s3.amazonaws.com"
        return f"https://s3.amazonaws.com/{self.bucket}"

    def list(self, prefix: str, *, delimiter: str | None = None) -> Iterator[S3Object | str]:
        """Every object below ``prefix``, in the order of the bucket (UTF-8 binary), page
        after page of ListObjectsV2 (``list-type=2``, ``continuation-token``). With
        ``delimiter`` a common prefix is yielded as a ``str`` where it falls in that order.
        A page that cannot be had after the retries raises ``SourceError("unreachable",
        ...)``; 403 is ``unreadable``, 404 (no such bucket) ``not found``."""
        raise NotImplementedError(_RA)

    def head(self, key: str) -> S3Head:
        """HEAD of ``key``. 404 raises ``SourceError("changed", ...)`` for a key that a
        listing gave; see the module for the rest."""
        raise NotImplementedError(_RA)

    def get(
        self,
        key: str,
        *,
        byte_range: tuple[int, int] | None = None,
        if_match: str | None = None,
    ) -> tuple[bytes, S3Head]:
        """The content of ``key``, or its inclusive ``(first, last)`` range (206). The caller
        clamps ``last`` to the size and asks for no empty range; a 416 raises
        ``SourceError("invalid", ...)``. ``if_match`` (an ETag without quotes) is sent as
        ``If-Match: "<etag>"``; 412 raises ``SourceError("changed", ...)``. A body is checked
        against its ``Content-Length`` and a whole one, for an ETag without ``-``, against its
        MD5; a mismatch is retried as a body cut short is."""
        raise NotImplementedError(_RA)

    def download(
        self,
        key: str,
        dest: Path,
        *,
        expect: S3Object,
        cancel: threading.Event | None = None,
    ) -> S3Head:
        """Write ``key`` to ``dest``: GET with ``If-Match: expect.etag``, streamed in
        :data:`CHUNK_BYTES` to a temporary name beside ``dest``, checked for ``expect.size``
        and, for an ETag without ``-``, its MD5, then renamed to ``dest``. ``dest`` never
        holds a partial file. When ``cancel`` is set, or the call is interrupted, the
        temporary file is removed and the call raises (``KeyboardInterrupt`` passes
        through) within one chunk. 412, or an answer whose size or ETag is not that of
        ``expect``, raises ``SourceError("changed", ...)``; a body cut short or whose MD5
        does not match, once the retries ran out, ``SourceError("unreachable", ...)``."""
        raise NotImplementedError(_RA)

    def close(self) -> None:
        self.http.close()

    def __enter__(self) -> S3Client:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
