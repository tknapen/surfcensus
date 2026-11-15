"""The one way a request leaves the package: :class:`ReadOnlyTransport` around an httpx
transport (docs/dev/remote_plan.md).

The guard sits in the transport, below every client, so no code path can send another
method, a credential or a signed URL: ``S3Client`` wraps whatever transport it is given in
it (:func:`read_only`) and makes its ``httpx.Client`` with ``trust_env=False``, because
proxies taken from the environment are mounted beside the transport and would pass around
it. The proxy of the environment, where there is one, is given to the transport itself by
:func:`default_transport`.
"""

from __future__ import annotations

import ipaddress
import os
from collections.abc import Callable, Iterator
from urllib.parse import parse_qsl, urlsplit

import httpx

from surfcensus.remote import ENDPOINT_ENV, record_bytes, record_request, take_slot
from surfcensus.sources.base import SourceError

ALLOWED_METHODS: frozenset[str] = frozenset({"GET", "HEAD"})

REFUSED_HEADERS: frozenset[str] = frozenset(
    {"authorization", "x-amz-security-token", "x-amz-content-sha256", "x-amz-date"}
)
"""Headers of a signed or authenticated request; none of them is ever sent."""

REFUSED_QUERY: frozenset[str] = frozenset(
    {
        "x-amz-signature",
        "x-amz-credential",
        "x-amz-security-token",
        "x-amz-algorithm",
        "awsaccesskeyid",
        "signature",
    }
)
"""Query parameters of a presigned URL (compared without case)."""

_S3_HOST = "s3.amazonaws.com"


class _Counted(httpx.SyncByteStream):
    """A response body that counts its bytes and gives the slot of its request back when it
    is closed or read to the end."""

    def __init__(self, inner: httpx.SyncByteStream, release: Callable[[], None]) -> None:
        self._inner = inner
        self._release = release

    def __iter__(self) -> Iterator[bytes]:
        try:
            for chunk in self._inner:
                record_bytes(len(chunk))
                yield chunk
        finally:
            self._release()

    def close(self) -> None:
        try:
            self._inner.close()
        finally:
            self._release()


def refusal(request: httpx.Request) -> str | None:
    """Why :class:`ReadOnlyTransport` refuses ``request``, or None when it may go."""
    if request.method not in ALLOWED_METHODS:
        return f"the method {request.method} (only GET and HEAD are sent)"
    sent = {name.lower() for name in request.headers}
    for name in sorted(sent & REFUSED_HEADERS):
        return f"the header {name} (requests are anonymous)"
    for name, _ in parse_qsl(request.url.query.decode("ascii", "replace"), keep_blank_values=True):
        if name.lower() in REFUSED_QUERY:
            return f"the query parameter {name} of a signed URL (requests are anonymous)"
    if request.url.userinfo:
        return "credentials in the URL (requests are anonymous)"
    return None


class ReadOnlyTransport(httpx.BaseTransport):
    """GET and HEAD only, anonymous, at most ``MAX_CONCURRENCY`` at once in the process.

    A request with another method, an ``Authorization`` or AWS security header, a signed
    query string or credentials in its URL raises ``PermissionError`` before the inner
    transport sees it. Every request that passes is counted (``remote.stats``) and holds a
    slot of the process (``remote.take_slot``) until its response is closed.
    """

    def __init__(self, inner: httpx.BaseTransport) -> None:
        if isinstance(inner, ReadOnlyTransport):
            inner = inner.inner
        self.inner = inner

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        why = refusal(request)
        if why is not None:
            raise PermissionError(f"refused before sending: {why}")
        release = take_slot()
        try:
            record_request()
            response = self.inner.handle_request(request)
        except BaseException:
            release()
            raise
        stream = response.stream
        if isinstance(stream, httpx.ByteStream):
            # a body in memory already (a mock transport): read it, so the slot goes back now,
            # also when the response was made from the stream and has no content yet
            try:
                record_bytes(len(response.read()))
            finally:
                release()
            return response
        if not isinstance(stream, httpx.SyncByteStream):
            release()
            raise TypeError("the transport returned an asynchronous body")
        response.stream = _Counted(stream, release)
        return response

    def close(self) -> None:
        self.inner.close()


def read_only(transport: httpx.BaseTransport) -> ReadOnlyTransport:
    """``transport`` behind the guard; a transport that is guarded already is returned as
    it is."""
    if isinstance(transport, ReadOnlyTransport):
        return transport
    return ReadOnlyTransport(transport)


def _loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


class _Endpoint(httpx.BaseTransport):
    """Sends every request to the bucket server of a test, path-style: a request for
    ``https://<bucket>.s3.amazonaws.com/<key>`` or ``https://s3.amazonaws.com/<bucket>/<key>``
    goes to ``<endpoint>/<bucket>/<key>``, with the same query and headers."""

    def __init__(self, endpoint: httpx.URL, inner: httpx.BaseTransport) -> None:
        self.endpoint = endpoint
        self.inner = inner

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        path = request.url.path
        if host.endswith("." + _S3_HOST):
            path = "/" + host[: -len(_S3_HOST) - 1] + path
        elif host != _S3_HOST:
            raise PermissionError(f"refused before sending: {host} is no bucket host")
        url = self.endpoint.copy_with(path=path, query=request.url.query or None)
        headers = httpx.Headers(request.headers)
        headers["Host"] = url.netloc.decode("ascii")
        moved = httpx.Request(request.method, url, headers=headers, stream=request.stream)
        return self.inner.handle_request(moved)

    def close(self) -> None:
        self.inner.close()


def _environment_proxy() -> str | None:
    """The HTTPS proxy of the environment, unless ``NO_PROXY`` exempts the bucket hosts."""
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if not proxy:
        return None
    exempt = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    for entry in (part.strip().lower() for part in exempt.split(",")):
        if entry == "*" or (entry and _S3_HOST.endswith(entry.lstrip("*").lstrip("."))):
            return None
    return proxy


def default_transport() -> ReadOnlyTransport:
    """The transport of a client that was given none: HTTPS to the bucket hosts, through the
    proxy of the environment where there is one (``HTTPS_PROXY``, the CA file of
    ``SSL_CERT_FILE``), behind :class:`ReadOnlyTransport`.

    With :data:`ENDPOINT_ENV` set every request goes to that endpoint instead, path-style
    and without a proxy: how a test in a subprocess reaches the fake bucket. The endpoint
    must be ``http`` on a loopback host; anything else raises ``SourceError("invalid",
    ...)``, so the variable cannot send requests elsewhere.
    """
    endpoint = os.environ.get(ENDPOINT_ENV)
    if endpoint:
        parts = urlsplit(endpoint)
        if parts.scheme != "http" or not parts.hostname or not _loopback(parts.hostname):
            raise SourceError(
                "invalid",
                f"{ENDPOINT_ENV} is honoured for http on a loopback host only, for tests; "
                "unset it to reach the bucket",
            )
        return read_only(_Endpoint(httpx.URL(endpoint), httpx.HTTPTransport(trust_env=False)))
    return read_only(httpx.HTTPTransport(proxy=_environment_proxy(), retries=0))
