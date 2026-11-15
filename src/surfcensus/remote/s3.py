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

import email.utils
import hashlib
import math
import os
import random
import re
import secrets
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NoReturn, TypeVar
from urllib.parse import quote, unquote_plus
from xml.etree import ElementTree

import httpx

from surfcensus.config import RemoteConfig
from surfcensus.remote import (
    RETRIED_STATUSES,
    SLOW_DOWN,
    STATUS_REASONS,
    UNREACHABLE,
    source_error,
)
from surfcensus.remote.transport import ReadOnlyTransport, default_transport, read_only
from surfcensus.sources.base import SourceError

_T = TypeVar("_T")

_MD5 = re.compile(r"[0-9a-f]{32}")
_CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")

LISTING = "listing"
"""The location that the errors of a listing name."""

_ERROR_BODY_BYTES = 64 * 1024
"""At most this much of an error body is read, for its ``Code``."""

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
    if attempt < 1:
        raise ValueError(f"retries are numbered from 1, not {attempt}")
    if retry_after is not None and math.isfinite(retry_after):
        return min(RETRY_CAP_S, max(0.0, retry_after))
    # the exponent is bounded so that a large attempt number cannot overflow a float
    ceiling = min(RETRY_CAP_S, RETRY_BASE_S * 2.0 ** min(attempt - 1, 64))
    return (rng or random).uniform(0.0, ceiling)


class DownloadCancelled(SourceError):
    """A download stopped because its cancel event was set. The reason is ``unreachable``,
    so that code which does not look for this class stops its phase and records nothing,
    as for a fault of the network (decision D47, item 10)."""

    def __init__(self, reason: str = UNREACHABLE, detail: str = "the download was cancelled"):
        super().__init__(reason, detail)


class _Retry(Exception):
    """An answer that is tried again: a retried status, a body cut short, a body whose MD5
    is not its ETag. ``problem`` holds no URL, key or ETag."""

    def __init__(self, problem: str, retry_after: float | None = None) -> None:
        super().__init__(problem)
        self.problem = problem
        self.retry_after = retry_after


def _unquote(etag: str | None) -> str:
    return (etag or "").strip().strip('"')


def _md5_of(etag: str) -> str | None:
    """The MD5 that ``etag`` is, or None for an ETag of a multipart upload or another form."""
    value = etag.lower()
    return value if _MD5.fullmatch(value) else None


def _retry_after(value: str | None) -> float | None:
    """Seconds of a ``Retry-After`` header (a number or an HTTP date), or None."""
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if when.tzinfo is None:
            return None
        seconds = when.timestamp() - time.time()
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def _error_code(body: bytes) -> str | None:
    """The ``Code`` of an S3 error body, or None."""
    try:
        root = ElementTree.fromstring(body)
    except ElementTree.ParseError:
        return None
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] == "Code" and element.text:
            return element.text.strip()[:64]
    return None


def _content_range(value: str | None) -> tuple[int, int, int] | None:
    """``(first, last, total)`` of ``Content-Range: bytes first-last/total``, or None."""
    found = _CONTENT_RANGE.fullmatch((value or "").strip())
    if found is None:
        return None
    first, last, total = (int(group) for group in found.groups())
    if first > last or last >= total:
        return None
    return first, last, total


def _int_header(response: httpx.Response, name: str) -> int | None:
    value = response.headers.get(name)
    if value is None:
        return None
    try:
        number = int(value.strip())
    except ValueError:
        return None
    return number if number >= 0 else None


def _head_of(response: httpx.Response, size: int) -> S3Head:
    version = response.headers.get("x-amz-version-id")
    return S3Head(
        size=size,
        etag=_unquote(response.headers.get("etag")),
        last_modified=response.headers.get("last-modified", ""),
        version_id=version if version and version != "null" else None,
    )


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

    # --- requests -----------------------------------------------------------------------------

    def _url(self, key: str) -> str:
        """The URL of ``key``; ``""`` gives the bucket's root, where a listing is asked for."""
        return f"{self.endpoint}/{quote(key, safe='/~')}"

    def _call(
        self,
        method: str,
        url: str,
        *,
        location: str,
        handle: Callable[[httpx.Response], _T],
        params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        secret: tuple[str, ...] = (),
        cancel: threading.Event | None = None,
    ) -> _T:
        """One request, with the retries of the module.

        ``handle`` reads the streamed answer of an attempt and returns the result; it raises
        :class:`_Retry` for an answer that is tried again and ``SourceError`` for one that is
        not. The slot of the client is held from the request to the end of ``handle``, and
        the response is closed when ``handle`` returns or raises. ``secret`` are strings (the
        key, a prefix) that are taken out of the text of an exception before it becomes a
        detail, beside the URL."""
        hidden = (url, *secret, *(quote(part, safe="") for part in secret))
        attempt = 0
        while True:
            if cancel is not None and cancel.is_set():
                raise DownloadCancelled(UNREACHABLE, f"{location}: the download was cancelled")
            try:
                with (
                    self.slots,
                    self.http.stream(method, url, params=params, headers=headers) as response,
                ):
                    return handle(response)
            except _Retry as retry:
                problem, retry_after = retry.problem, retry.retry_after
            except httpx.ProxyError as exc:
                # the proxy refused the bucket: a fault of the machine, not of a request
                raise source_error(
                    UNREACHABLE, location, _scrubbed(f"the proxy refused: {exc}", hidden)
                ) from None
            except (httpx.NetworkError, httpx.TimeoutException, httpx.RemoteProtocolError) as exc:
                problem, retry_after = _scrubbed(f"{type(exc).__name__}: {exc}", hidden), None
            attempt += 1
            if attempt > self.config.max_retries:
                tries = "1 attempt" if attempt == 1 else f"{attempt} attempts"
                raise source_error(UNREACHABLE, location, f"{problem} ({tries})")
            self._pause(backoff_delay(attempt, retry_after), cancel)

    def _pause(self, delay: float, cancel: threading.Event | None) -> None:
        """Wait ``delay`` seconds before a retry. A call that can be cancelled waits on its
        cancel event when the client sleeps for real, so that a stop (Ctrl-C in a lease) does
        not wait out a backoff of up to ``RETRY_CAP_S``; the loop then raises
        :class:`DownloadCancelled`. An injected ``sleep`` is called as it is."""
        if cancel is not None and self.sleep is time.sleep:
            cancel.wait(delay)
        else:
            self.sleep(delay)

    def _status(
        self, response: httpx.Response, location: str, *, missing: str = "changed"
    ) -> NoReturn:
        """Raise for an answer whose status is not a success: :class:`_Retry` for a retried
        status, ``SourceError`` otherwise. ``missing`` is the reason of a 404."""
        status = response.status_code
        code = None
        if response.request.method != "HEAD":
            body = bytearray()
            try:
                for chunk in response.iter_bytes():
                    body += chunk
                    if len(body) >= _ERROR_BODY_BYTES:
                        break
            except httpx.HTTPError:
                pass
            code = _error_code(bytes(body))
        said = f"HTTP {status}" + (f" {code}" if code else "")
        if code == SLOW_DOWN:
            said += " (the bucket asks for fewer requests)"
        if status in RETRIED_STATUSES:
            raise _Retry(said, _retry_after(response.headers.get("retry-after")))
        if status == 404:
            raise source_error(missing, location, said)
        reason = STATUS_REASONS.get(status, "unreadable")
        raise source_error(reason, location, said)

    # --- listing ------------------------------------------------------------------------------

    def list(self, prefix: str, *, delimiter: str | None = None) -> Iterator[S3Object | str]:
        """Every object below ``prefix``, in the order of the bucket (UTF-8 binary), page
        after page of ListObjectsV2 (``list-type=2``, ``continuation-token``). With
        ``delimiter`` a common prefix is yielded as a ``str`` where it falls in that order.
        A page that cannot be had after the retries raises ``SourceError("unreachable",
        ...)``; 403 is ``unreadable``, 404 (no such bucket) ``not found``.

        Keys are asked for URL-encoded (``encoding-type=url``), which keeps a key with a
        character that XML 1.0 cannot hold readable, and decoded when the answer says
        ``EncodingType`` ``url``. Errors name the location ``listing``."""
        token: str | None = None
        seen: set[str] = set()
        while True:
            params = {"list-type": "2", "prefix": prefix, "encoding-type": "url"}
            if delimiter is not None:
                params["delimiter"] = delimiter
            if token is not None:
                params["continuation-token"] = token
            entries, token = self._call(
                "GET",
                self._url(""),
                location=LISTING,
                handle=self._read_page,
                params=params,
                secret=(prefix,),
            )
            yield from entries
            if token is None:
                return
            if token in seen:
                raise source_error("invalid", LISTING, "the bucket gave a continuation token twice")
            seen.add(token)

    def _read_page(self, response: httpx.Response) -> tuple[list[S3Object | str], str | None]:
        if response.status_code != 200:
            self._status(response, LISTING, missing="not found")
        body = _read_body(response)
        try:
            root = ElementTree.fromstring(body)
        except ElementTree.ParseError:
            # a page that does not parse was most likely cut short on the way
            raise _Retry("a listing page that is not XML") from None
        ns = {"s3": S3_NAMESPACE}

        def text(element: ElementTree.Element, name: str) -> str | None:
            found = element.find(f"s3:{name}", ns)
            if found is None:
                found = element.find(name)
            return found.text if found is not None else None

        encoded = (text(root, "EncodingType") or "").lower() == "url"

        def decoded(value: str) -> str:
            return unquote_plus(value) if encoded else value

        entries: list[S3Object | str] = []
        for element in list(root):
            tag = element.tag.rsplit("}", 1)[-1]
            if tag == "Contents":
                key, size = text(element, "Key"), text(element, "Size")
                if key is None or size is None or not size.strip().isdigit():
                    raise source_error(
                        "invalid", LISTING, "an entry of the listing has no key or size"
                    )
                entries.append(
                    S3Object(
                        key=decoded(key),
                        size=int(size),
                        etag=_unquote(text(element, "ETag")),
                        last_modified=text(element, "LastModified") or "",
                    )
                )
            elif tag == "CommonPrefixes":
                common = text(element, "Prefix")
                if common is not None:
                    entries.append(decoded(common))
        # S3 puts the objects of a page before its common prefixes; merged into bucket order
        entries.sort(key=lambda entry: (entry if isinstance(entry, str) else entry.key).encode())
        truncated = (text(root, "IsTruncated") or "false").strip().lower() == "true"
        if not truncated:
            return entries, None
        token = text(root, "NextContinuationToken")
        if not token:
            raise source_error("invalid", LISTING, "a truncated page without a continuation token")
        return entries, token

    # --- objects ------------------------------------------------------------------------------

    def head(self, key: str, *, location: str | None = None) -> S3Head:
        """HEAD of ``key``. 404 raises ``SourceError("changed", ...)`` for a key that a
        listing gave; see the module for the rest. ``location`` is what an error names
        (the file's place relative to the subject directory); by default the last component
        of the key."""
        where = location if location is not None else _basename(key)

        def handle(response: httpx.Response) -> S3Head:
            if response.status_code != 200:
                self._status(response, where)
            size = _int_header(response, "content-length")
            if size is None:
                raise source_error("invalid", where, "the answer has no Content-Length")
            return _head_of(response, size)

        return self._call("HEAD", self._url(key), location=where, handle=handle, secret=(key,))

    def get(
        self,
        key: str,
        *,
        byte_range: tuple[int, int] | None = None,
        if_match: str | None = None,
        location: str | None = None,
    ) -> tuple[bytes, S3Head]:
        """The content of ``key``, or its inclusive ``(first, last)`` range (206). The caller
        clamps ``last`` to the size and asks for no empty range; a 416 raises
        ``SourceError("invalid", ...)``. ``if_match`` (an ETag without quotes) is sent as
        ``If-Match: "<etag>"``; 412 raises ``SourceError("changed", ...)``. A body is checked
        against its ``Content-Length`` and a whole one, for an ETag without ``-``, against its
        MD5; a mismatch is retried as a body cut short is. ``location`` as for :meth:`head`.

        An answer of 200 to a ranged request (a server that ignores ``Range``) is checked as
        a whole body and cut to the range. An answer whose ETag is not ``if_match`` raises
        ``changed`` as a 412 does."""
        where = location if location is not None else _basename(key)
        headers: dict[str, str] = {}
        if byte_range is not None:
            first, last = byte_range
            if first < 0 or last < first:
                raise ValueError(f"not an inclusive byte range: {byte_range!r}")
            headers["Range"] = f"bytes={first}-{last}"
        if if_match is not None:
            headers["If-Match"] = f'"{_unquote(if_match)}"'

        def handle(response: httpx.Response) -> tuple[bytes, S3Head]:
            status = response.status_code
            if status not in (200, 206):
                self._status(response, where)
            body = _read_body(response)
            head = _head_of(response, len(body))
            if if_match is not None and head.etag != _unquote(if_match):
                raise source_error("changed", where, "the object's ETag is not the listed one")
            if status == 206:
                found = _content_range(response.headers.get("content-range"))
                if found is None or byte_range is None:
                    raise source_error("invalid", where, "a partial answer without a usable range")
                first, last, total = found
                if first != byte_range[0] or len(body) != last - first + 1:
                    raise _Retry("a partial answer of another length than its range")
                return body, _head_of(response, total)
            _check_md5(body, head.etag)
            if byte_range is not None:
                return body[byte_range[0] : byte_range[1] + 1], head
            return body, head

        return self._call(
            "GET", self._url(key), location=where, handle=handle, headers=headers, secret=(key,)
        )

    def download(
        self,
        key: str,
        dest: Path,
        *,
        expect: S3Object,
        cancel: threading.Event | None = None,
        location: str | None = None,
    ) -> S3Head:
        """Write ``key`` to ``dest``: GET with ``If-Match: expect.etag``, streamed in
        :data:`CHUNK_BYTES` to a temporary name beside ``dest``, checked for ``expect.size``
        and, for an ETag without ``-``, its MD5, then renamed to ``dest``. ``dest`` never
        holds a partial file. When ``cancel`` is set, or the call is interrupted, the
        temporary file is removed and the call raises (``KeyboardInterrupt`` passes
        through) within one chunk. 412, or an answer whose size or ETag is not that of
        ``expect``, raises ``SourceError("changed", ...)``; a body cut short or whose MD5
        does not match, once the retries ran out, ``SourceError("unreachable", ...)``.

        A cancel raises :class:`DownloadCancelled`, a ``SourceError`` with the reason
        ``unreachable``. The temporary name is ``.<name of dest>.<16 hex digits>.tmp`` in the
        directory of ``dest``. ``location`` as for :meth:`head`."""
        where = location if location is not None else _basename(key)
        dest = Path(dest)
        etag = _unquote(expect.etag)
        md5 = _md5_of(etag)

        def handle(response: httpx.Response) -> S3Head:
            if response.status_code != 200:
                self._status(response, where)
            head = _head_of(response, expect.size)
            if head.etag != etag:
                raise source_error("changed", where, "the object's ETag is not the listed one")
            length = _int_header(response, "content-length")
            if length is not None and length != expect.size:
                raise source_error("changed", where, "the object's size is not the listed one")
            temporary = dest.parent / f".{dest.name}.{secrets.token_hex(8)}.tmp"
            try:
                digest = hashlib.md5(usedforsecurity=False)
                written = 0
                with open(temporary, "xb") as out:
                    for chunk in response.iter_bytes(CHUNK_BYTES):
                        if cancel is not None and cancel.is_set():
                            raise DownloadCancelled(
                                UNREACHABLE, f"{where}: the download was cancelled"
                            )
                        out.write(chunk)
                        digest.update(chunk)
                        written += len(chunk)
                if written != expect.size:
                    raise _Retry(f"the body has {_size_word(written, expect.size)}")
                if md5 is not None and digest.hexdigest() != md5:
                    raise _Retry("the body's MD5 is not its ETag")
                os.replace(temporary, dest)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
            return head

        return self._call(
            "GET",
            self._url(key),
            location=where,
            handle=handle,
            headers={"If-Match": f'"{etag}"'},
            secret=(key,),
            cancel=cancel,
        )

    def close(self) -> None:
        self.http.close()

    def __enter__(self) -> S3Client:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# --- helpers of the bodies --------------------------------------------------------------------


def _basename(key: str) -> str:
    """The last component of ``key``: what an error names when its caller gave no place."""
    return key.rstrip("/").rsplit("/", 1)[-1] or "object"


def _scrubbed(text: str, hidden: tuple[str, ...]) -> str:
    """``text`` with every string of ``hidden`` (the URL, the key, a prefix) replaced by
    ``<url>``, longest first, so that no part of one is left."""
    for value in sorted(hidden, key=len, reverse=True):
        if value:
            text = text.replace(value, "<url>")
    return text


def _size_word(written: int, expected: int) -> str:
    return "fewer bytes than its listing says" if written < expected else "more bytes than listed"


def _read_body(response: httpx.Response) -> bytes:
    """The whole body of a streamed answer, checked against its ``Content-Length``: a body
    that is shorter (or longer) raises :class:`_Retry`."""
    body = bytearray()
    for chunk in response.iter_bytes():
        body += chunk
    expected = _int_header(response, "content-length")
    if expected is not None and len(body) != expected:
        raise _Retry("the body was cut short" if len(body) < expected else "the body is too long")
    return bytes(body)


def _check_md5(body: bytes, etag: str) -> None:
    """Raise :class:`_Retry` when ``etag`` is an MD5 and ``body`` does not have it."""
    md5 = _md5_of(etag)
    if md5 is not None and hashlib.md5(body, usedforsecurity=False).hexdigest() != md5:
        raise _Retry("the body's MD5 is not its ETag")
