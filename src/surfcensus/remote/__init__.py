"""Anonymous, read-only access to public buckets (docs/dev/remote_plan.md).

What the modules of ``remote/`` share, so that the S3 client (``remote/s3.py``), the cache
(``remote/cache.py``) and the OpenNeuroDerivatives source can be written side by side:
the names of the bucket and its prefix, the limit on concurrent requests, the parsing of
``openneuro:dsXXXXXX``, the reasons that HTTP answers map to, and process-wide counters of
requests and bytes.

Importing this module loads the standard library only, so that a command can parse a path
without loading ``httpx``. The transport (:class:`ReadOnlyTransport`, :func:`default_transport`)
lives in ``remote/transport.py`` and is imported on first use of its names here.

Rules that every module of ``remote/`` follows:

* Read-only and anonymous. Every request goes through :class:`ReadOnlyTransport`, which
  refuses any method but GET and HEAD, any ``Authorization`` or security-token header and
  any signed query string, before any I/O. Nothing here reads a credential.
* At most :data:`MAX_CONCURRENCY` requests at a time in a process, whatever the settings say.
* No accession, ETag or object size leaves ``catalog.sqlite`` (decision D20): the text of an
  exception that names a URL is passed through :func:`source_error` before it becomes a
  ``SourceError``.
"""

from __future__ import annotations

import os
import re
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from surfcensus.remote.transport import ReadOnlyTransport, default_transport, read_only
    from surfcensus.sources.base import SourceError

__all__ = [
    "DERIVATIVES_BUCKET",
    "ENDPOINT_ENV",
    "FMRIPREP_PREFIX",
    "MAX_CONCURRENCY",
    "REMOTE_SCHEME",
    "RETRIED_STATUSES",
    "SLOW_DOWN",
    "STATUS_REASONS",
    "UNREACHABLE",
    "ReadOnlyTransport",
    "RemoteRef",
    "RemoteStats",
    "dataset_prefix",
    "default_transport",
    "parse_remote_path",
    "read_only",
    "record_bytes",
    "record_request",
    "request_slot",
    "reset_stats",
    "source_error",
    "stats",
    "take_slot",
]

DERIVATIVES_BUCKET = "openneuro-derivatives"
"""The OpenNeuroDerivatives bucket, listable anonymously (checked 2026-11-01)."""

FMRIPREP_PREFIX = "fmriprep/"
"""The prefix under which the bucket holds one ``dsXXXXXX-fmriprep/`` prefix per dataset. A
remote source is stored with this prefix and never with the prefix of one dataset, which
would put an accession into ``config.yaml`` (decision D47, item 1)."""

MAX_CONCURRENCY = 8
"""At most this many requests are open at once in a process, through every client and
thread (``RemoteConfig.max_concurrency`` may only lower it). The tests use 4."""

ENDPOINT_ENV = "SURFCENSUS_S3_ENDPOINT"
"""An ``http://127.0.0.1:PORT`` (or ``localhost``, ``[::1]``) that :func:`default_transport`
sends every bucket request to, path-style: how a test in a subprocess reaches the fake bucket
of ``tests/fakes3.py``. Any other host is refused."""

REMOTE_SCHEME = "openneuro"
"""``openneuro:dsXXXXXX`` names a dataset of :data:`DERIVATIVES_BUCKET` on the command line."""

_REMOTE_PATH = re.compile(r"openneuro:(ds[0-9]{6})")
_DATASET_PREFIX = "{accession}-fmriprep/"

STATUS_REASONS: dict[int, str] = {403: "unreadable", 404: "changed", 412: "changed", 416: "invalid"}
"""The ``SourceError`` reason of an HTTP status of a request for an object that a listing
named (decision D47, item 10). A 404 or 412 says that the object changed after it was listed;
a 403 that it may not be read. None of these is retried. A listing that cannot be had is
``not found`` (404) or ``unreadable`` (403); retries that run out are ``unreachable``."""

RETRIED_STATUSES: frozenset[int] = frozenset({500, 502, 503, 504})
"""Statuses that are retried, with exponential backoff and full jitter, up to
``RemoteConfig.max_retries``; besides them connection errors, read timeouts, a body cut short
and a body whose MD5 is not its ETag. ``SlowDown`` comes as a 503."""

SLOW_DOWN = "SlowDown"
"""The ``Code`` of S3's error body that asks a client to send fewer requests."""

UNREACHABLE = "unreachable"
"""The reason of a ``SourceError`` when the retries ran out on a connection error, a 5xx, or
a body that was cut short or did not match its ETag. It stops a phase with exit code 2 and
records nothing of the batch it happened in (decision D47, item 10)."""


@dataclass(frozen=True)
class RemoteRef:
    """A dataset of the OpenNeuroDerivatives bucket as a user names it: ``openneuro:ds000102``.

    The accession is the user's own argument: it may be printed on their terminal and is
    the name of the default run directory (:attr:`run_dir_name`), and it is stored in
    ``catalog.sqlite`` only (decision D47, item 1).
    """

    accession: str
    source: Literal["openneuro-derivatives"] = "openneuro-derivatives"

    def __post_init__(self) -> None:
        if not re.fullmatch(r"ds[0-9]{6}", self.accession):
            raise ValueError(f"an accession is ds and six digits, not {self.accession!r}")

    @property
    def text(self) -> str:
        """``openneuro:dsXXXXXX``, as it was typed."""
        return f"{REMOTE_SCHEME}:{self.accession}"

    @property
    def bucket(self) -> str:
        return DERIVATIVES_BUCKET

    @property
    def prefix(self) -> str:
        """The dataset's prefix in the bucket, ``fmriprep/dsXXXXXX-fmriprep/``: what a
        listing asks for. Never stored outside ``catalog.sqlite``."""
        return dataset_prefix(self.accession)

    @property
    def run_dir_name(self) -> str:
        """``dsXXXXXX-surfcensus-run``: the default run directory, in the current directory
        (decision D47, item 2)."""
        return f"{self.accession}-surfcensus-run"


def dataset_prefix(accession: str, prefix: str = FMRIPREP_PREFIX) -> str:
    """``fmriprep/dsXXXXXX-fmriprep/``: the prefix of one dataset below ``prefix``."""
    return prefix + _DATASET_PREFIX.format(accession=accession)


def parse_remote_path(text: str | os.PathLike[str]) -> RemoteRef | None:
    """The dataset that ``text`` names, or None when ``text`` is a path.

    ``openneuro:`` followed by ``ds`` and six ASCII digits is always the remote dataset, also
    when a folder of that name exists (decision D47, item 8). Any other text that starts with
    ``openneuro:`` raises ``SourceError("invalid", ...)``: it is meant as a dataset and names
    none. Every other text, ``./openneuro:x`` included, is a path.
    """
    value = os.fspath(text)
    if not value.startswith(f"{REMOTE_SCHEME}:"):
        return None
    found = _REMOTE_PATH.fullmatch(value)
    if found is None:
        from surfcensus.sources.base import SourceError

        raise SourceError(
            "invalid",
            f"{value!r} names no dataset: write openneuro:ds and the six digits of the "
            "accession, for example openneuro:ds000102",
        )
    return RemoteRef(found.group(1))


def source_error(reason: str, location: str, problem: object) -> SourceError:
    """A ``SourceError`` whose detail is ``location: problem``, with ``problem`` (an
    exception or a text, which may hold a URL) scrubbed: a URL or a path becomes its last
    component and an accession ``D-?`` (``privacy.scrub``). ``location`` is the file's place
    relative to the subject directory (``surf/lh.white``) or another text without a key."""
    from surfcensus.privacy import scrub
    from surfcensus.sources.base import SourceError

    return SourceError(reason, f"{location}: {scrub(str(problem))}")


# --- process-wide counters and the limit on concurrent requests -----------------------------


@dataclass(frozen=True)
class RemoteStats:
    """What this process has asked the buckets for since it started, or since
    :func:`reset_stats`."""

    requests: int
    """Requests that :class:`ReadOnlyTransport` passed on, retries included."""
    bytes: int
    """Bytes of response bodies read through it."""
    in_flight: int
    """Requests whose response is not closed yet."""
    high_water: int
    """The most requests that were open at once."""


_lock = threading.Lock()
_slots = threading.BoundedSemaphore(MAX_CONCURRENCY)
_requests = 0
_bytes = 0
_in_flight = 0
_high_water = 0


def stats() -> RemoteStats:
    """The counters of this process."""
    with _lock:
        return RemoteStats(_requests, _bytes, _in_flight, _high_water)


def reset_stats() -> None:
    """Set the counts of requests and bytes and the high-water mark to 0 (or to the requests
    open now). For tests and for a command that reports what one phase fetched."""
    global _requests, _bytes, _high_water
    with _lock:
        _requests, _bytes, _high_water = 0, 0, _in_flight


def record_request() -> None:
    """Count one request. Called by :class:`ReadOnlyTransport` for every request it passes on."""
    global _requests
    with _lock:
        _requests += 1


def record_bytes(n: int) -> None:
    """Count ``n`` bytes of a response body. Called by the transport as a body is read."""
    global _bytes
    with _lock:
        _bytes += n


def take_slot() -> Callable[[], None]:
    """Take one of the :data:`MAX_CONCURRENCY` slots of the process, blocking while all are
    taken, and return the function that gives it back (a second call does nothing).

    :class:`ReadOnlyTransport` takes a slot for every request and gives it back when the
    response is closed or its body has been read to the end, so no client and no thread can
    have more requests open. A caller that streams a body must close the response."""
    global _in_flight, _high_water
    _slots.acquire()
    with _lock:
        _in_flight += 1
        _high_water = max(_high_water, _in_flight)
    released = threading.Event()

    def release() -> None:
        global _in_flight
        with _lock:
            if released.is_set():
                return
            released.set()
            _in_flight -= 1
        _slots.release()

    return release


@contextmanager
def request_slot() -> Iterator[None]:
    """:func:`take_slot` for the length of a ``with`` block."""
    release = take_slot()
    try:
        yield
    finally:
        release()


_LAZY = {
    "ReadOnlyTransport": "ReadOnlyTransport",
    "default_transport": "default_transport",
    "read_only": "read_only",
}


def __getattr__(name: str) -> object:
    # the transport needs httpx, which a command that only parses a path does not load
    if name in _LAZY:
        from surfcensus.remote import transport

        return getattr(transport, _LAZY[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
