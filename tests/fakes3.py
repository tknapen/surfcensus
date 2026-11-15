"""A fake S3 bucket server for tests (docs/dev/remote_plan.md, "Testing without the network").

Directories are buckets: every file below a bucket directory, links followed, is an object
whose key is its path relative to that directory. One handler answers what the package asks a
bucket: ListObjectsV2 (``list-type=2``, ``prefix``, ``delimiter``, ``CommonPrefixes``,
``continuation-token``, ``max-keys``, a page size of the fake's own), GET and HEAD with
``Range`` (206 and ``Content-Range``, 416) and ``If-Match`` (412), a quoted MD5 as ETag,
``Last-Modified``, and with ``version_ids`` an ``x-amz-version-id``. Any other method is 405.
A request with an ``Authorization`` header or a signed query string is answered 403 and kept
in :attr:`FakeS3.signed`, so a test can assert that none was sent.

The same handler is reachable two ways: as an ``httpx.MockTransport`` (:meth:`FakeS3.transport`)
for tests in the process, and as a ``ThreadingHTTPServer`` on 127.0.0.1 (:meth:`FakeS3.serve`)
for tests that run a command in a subprocess, which reaches it through
``SURFCENSUS_S3_ENDPOINT``. Requests may be virtual-hosted (``<bucket>.s3.amazonaws.com``) or
path-style (``s3.amazonaws.com/<bucket>/``, or the loopback server).

Faults are injected by key pattern (:meth:`FakeS3.fault`): 403, 500, 503, a 503 ``SlowDown`` body,
a body cut short, a connection reset. :meth:`FakeS3.mutate_after_list` changes the content of
an object once a listing has named it, without touching the file, so that its ETag at GET
differs from the listed one. A request log and a high-water mark of concurrent requests are
kept. The fake writes no file.
"""

from __future__ import annotations

import base64
import email.utils
import fnmatch
import hashlib
import json
import os
import socket
import struct
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from urllib.parse import parse_qsl, quote, unquote, urlsplit
from xml.sax.saxutils import escape

import httpx

if TYPE_CHECKING:
    from synth_tierb import SyntheticCohort  # pyrefly: ignore[missing-import]

NAMESPACE = "http://s3.amazonaws.com/doc/2006-03-01/"
S3_HOST = "s3.amazonaws.com"
FaultKind = Literal["403", "500", "503", "slowdown", "truncate", "reset"]
SIGNED_QUERY = {
    "x-amz-signature",
    "x-amz-credential",
    "x-amz-security-token",
    "x-amz-algorithm",
    "awsaccesskeyid",
    "signature",
}


@dataclass
class Fault:
    pattern: str
    kind: FaultKind
    times: int | None
    """How many more requests it hits; None for every one."""
    methods: frozenset[str] = frozenset({"GET", "HEAD", "LIST"})
    retry_after: float | None = None
    hits: int = 0


@dataclass(frozen=True)
class Logged:
    """One request as the fake saw it."""

    method: str
    """``GET``, ``HEAD``, ``LIST`` (a GET of a listing) or the method refused."""
    bucket: str | None
    key: str | None
    """The key of an object request, the ``prefix`` of a listing."""
    query: Mapping[str, str]
    range: str | None
    if_match: str | None
    status: int
    signed: bool
    """An ``Authorization`` header or a signed query string came with it."""


@dataclass
class _Reply:
    status: int
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""
    cut: int | None = None
    """Send this many bytes of ``body`` and drop the connection, with the full length in
    ``Content-Length``."""
    reset: bool = False


def _http_date(mtime: float) -> str:
    return email.utils.formatdate(mtime, usegmt=True)


def _iso(mtime: float) -> str:
    stamp = datetime.fromtimestamp(mtime, UTC)
    return stamp.strftime("%Y-%m-%dT%H:%M:%S.") + f"{stamp.microsecond // 1000:03d}Z"


def _error(status: int, code: str, message: str) -> _Reply:
    body = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f"<Error><Code>{code}</Code><Message>{escape(message)}</Message></Error>"
    ).encode()
    return _Reply(status, {"Content-Type": "application/xml"}, body)


class FakeS3:
    """Buckets served from directories. ``buckets`` maps a bucket name to its directory.

    ``page_size`` bounds the entries of one ListObjectsV2 page (S3's is 1000). ``latency_s``
    is slept at the start of every request. With ``version_ids`` every object answer has an
    ``x-amz-version-id`` derived from its content.
    """

    def __init__(
        self,
        buckets: Mapping[str, Path],
        *,
        page_size: int = 1000,
        latency_s: float = 0.0,
        version_ids: bool = False,
    ) -> None:
        self.buckets = {name: Path(path) for name, path in buckets.items()}
        self.page_size = page_size
        self.latency_s = latency_s
        self.version_ids = version_ids
        self.log: list[Logged] = []
        self.signed: list[Logged] = []
        self.high_water = 0
        self._in_flight = 0
        self._lock = threading.Lock()
        self._faults: list[Fault] = []
        self._mutations: dict[tuple[str, str], tuple[bytes, bool]] = {}
        self._md5: dict[tuple[str, int, int], str] = {}

    # --- what a test sets up ------------------------------------------------------------------

    def fault(
        self,
        pattern: str,
        kind: FaultKind,
        *,
        times: int | None = 1,
        methods: tuple[str, ...] = ("GET", "HEAD", "LIST"),
        retry_after: float | None = None,
    ) -> Fault:
        """Answer requests whose key (or listing prefix) matches ``pattern`` (``fnmatch``)
        with ``kind``, ``times`` times (None: always). ``LIST`` among ``methods`` stands for
        a listing."""
        made = Fault(pattern, kind, times, frozenset(methods), retry_after)
        with self._lock:
            self._faults.append(made)
        return made

    def mutate_after_list(self, bucket: str, key: str, content: bytes | None = None) -> None:
        """Once a listing has named ``key``, serve ``content`` (by default the file's bytes
        with one more byte) for it, with the ETag and size of that content: the object
        changed after it was listed. The file is not touched."""
        path = self._path(bucket, key)
        new = content if content is not None else path.read_bytes() + b"\n"
        with self._lock:
            self._mutations[(bucket, key)] = (new, False)

    def clear(self) -> None:
        """Forget the log, the faults and the high-water mark."""
        with self._lock:
            self.log.clear()
            self.signed.clear()
            self._faults.clear()
            self.high_water = 0

    @property
    def methods(self) -> set[str]:
        """The methods of every logged request (``LIST`` for a listing)."""
        return {entry.method for entry in self.log}

    # --- objects ------------------------------------------------------------------------------

    def _path(self, bucket: str, key: str) -> Path:
        root = self.buckets[bucket]
        path = root.joinpath(*key.split("/"))
        if ".." in key.split("/"):
            raise KeyError(key)
        return path

    def keys(self, bucket: str) -> list[str]:
        """Every key of ``bucket`` in the order S3 lists them (UTF-8 binary)."""
        root = self.buckets[bucket]
        found = []
        for directory, _, files in os.walk(root, followlinks=True):
            for name in files:
                path = Path(directory, name)
                if path.is_file():
                    found.append(path.relative_to(root).as_posix())
        return sorted(found, key=lambda key: key.encode())

    def _content(self, bucket: str, key: str) -> tuple[bytes, float] | None:
        with self._lock:
            mutation = self._mutations.get((bucket, key))
        try:
            path = self._path(bucket, key)
            if not path.is_file():
                return None
            stat = path.stat()
        except (KeyError, OSError):
            return None
        if mutation is not None and mutation[1]:
            return mutation[0], stat.st_mtime + 1
        return path.read_bytes(), stat.st_mtime

    def _listed_facts(self, bucket: str, key: str) -> tuple[int, str, float]:
        """Size, ETag (without quotes) and mtime as a listing gives them now."""
        with self._lock:
            mutation = self._mutations.get((bucket, key))
        path = self._path(bucket, key)
        stat = path.stat()
        if mutation is not None and mutation[1]:
            return len(mutation[0]), hashlib.md5(mutation[0]).hexdigest(), stat.st_mtime + 1
        cached = (str(path), stat.st_size, stat.st_mtime_ns)
        etag = self._md5.get(cached)
        if etag is None:
            etag = hashlib.md5(path.read_bytes()).hexdigest()
            self._md5[cached] = etag
        return stat.st_size, etag, stat.st_mtime

    def _object_headers(self, content: bytes, mtime: float) -> dict[str, str]:
        md5 = hashlib.md5(content).hexdigest()
        headers = {
            "ETag": f'"{md5}"',
            "Last-Modified": _http_date(mtime),
            "Accept-Ranges": "bytes",
            "Content-Type": "application/octet-stream",
        }
        if self.version_ids:
            headers["x-amz-version-id"] = "v" + hashlib.sha256(content).hexdigest()[:24]
        return headers

    # --- the handler --------------------------------------------------------------------------

    def respond(
        self, method: str, host: str, path: str, query: str, headers: Mapping[str, str]
    ) -> _Reply:
        """The answer to one request; the core of both ways in."""
        lower = {name.lower(): value for name, value in headers.items()}
        with self._lock:
            self._in_flight += 1
            self.high_water = max(self.high_water, self._in_flight)
        try:
            if self.latency_s:
                time.sleep(self.latency_s)
            return self._respond(method, host, path, query, lower)
        finally:
            with self._lock:
                self._in_flight -= 1

    def _respond(
        self, method: str, host: str, path: str, query: str, headers: Mapping[str, str]
    ) -> _Reply:
        params = dict(parse_qsl(query, keep_blank_values=True))
        signed = "authorization" in headers or any(k.lower() in SIGNED_QUERY for k in params)
        bucket, key = self._route(host, unquote(path))
        listing = method == "GET" and key == "" and params.get("list-type") == "2"
        logged_method = "LIST" if listing else method
        logged_key = params.get("prefix", "") if listing else key

        def done(reply: _Reply) -> _Reply:
            entry = Logged(
                logged_method,
                bucket,
                logged_key,
                params,
                headers.get("range"),
                headers.get("if-match"),
                reply.status,
                signed,
            )
            with self._lock:
                self.log.append(entry)
                if signed:
                    self.signed.append(entry)
            return reply

        if method not in ("GET", "HEAD"):
            return done(_error(405, "MethodNotAllowed", f"{method} is not served"))
        if signed:
            return done(_error(403, "AccessDenied", "the fake takes anonymous requests only"))
        if bucket is None or bucket not in self.buckets:
            return done(_error(404, "NoSuchBucket", "no such bucket"))
        fault = self._fault(logged_method, logged_key or "")
        if fault is not None:
            return done(self._faulty(fault, method, bucket, key, params, headers, listing))
        if listing:
            return done(self._list(bucket, params))
        if key == "":
            return done(_error(400, "InvalidRequest", "a listing needs list-type=2"))
        return done(self._object(method, bucket, key, headers))

    def _route(self, host: str, path: str) -> tuple[str | None, str]:
        host = host.split(":")[0].lower() if not host.startswith("[") else host
        if host.endswith("." + S3_HOST):
            return host[: -len(S3_HOST) - 1], path.lstrip("/")
        parts = path.lstrip("/").split("/", 1)
        if not parts[0]:
            return None, ""
        return parts[0], parts[1] if len(parts) > 1 else ""

    def _fault(self, method: str, key: str) -> Fault | None:
        with self._lock:
            for fault in self._faults:
                if method not in fault.methods or not fnmatch.fnmatchcase(key, fault.pattern):
                    continue
                if fault.times is not None and fault.hits >= fault.times:
                    continue
                fault.hits += 1
                return fault
        return None

    def _faulty(
        self,
        fault: Fault,
        method: str,
        bucket: str,
        key: str,
        params: Mapping[str, str],
        headers: Mapping[str, str],
        listing: bool,
    ) -> _Reply:
        if fault.kind == "403":
            return _error(403, "AccessDenied", "access denied")
        if fault.kind == "500":
            return _error(500, "InternalError", "we encountered an internal error")
        if fault.kind in ("503", "slowdown"):
            code = "SlowDown" if fault.kind == "slowdown" else "ServiceUnavailable"
            reply = _error(503, code, "please reduce your request rate")
            if fault.retry_after is not None:
                reply.headers["Retry-After"] = f"{fault.retry_after:g}"
            return reply
        if fault.kind == "reset":
            return _Reply(0, reset=True)
        reply = (
            self._list(bucket, params) if listing else self._object(method, bucket, key, headers)
        )
        if reply.status in (200, 206) and reply.body:
            reply.cut = max(0, len(reply.body) // 2)
        return reply

    def _list(self, bucket: str, params: Mapping[str, str]) -> _Reply:
        prefix = params.get("prefix", "")
        delimiter = params.get("delimiter") or None
        try:
            max_keys = int(params.get("max-keys", "1000"))
        except ValueError:
            return _error(400, "InvalidArgument", "max-keys is a number")
        size = max(1, min(max_keys, self.page_size, 1000))
        entries: list[tuple[str, bool]] = []  # name, is a common prefix
        seen: set[str] = set()
        for key in self.keys(bucket):
            if not key.startswith(prefix):
                continue
            rest = key[len(prefix) :]
            if delimiter is not None and delimiter in rest:
                common = prefix + rest[: rest.index(delimiter) + len(delimiter)]
                if common not in seen:
                    seen.add(common)
                    entries.append((common, True))
                continue
            entries.append((key, False))
        start = 0
        token = params.get("continuation-token")
        if token is not None:
            try:
                after = base64.urlsafe_b64decode(token.encode()).decode()
            except ValueError:
                return _error(400, "InvalidArgument", "the continuation token is not valid")
            start = next(
                (i for i, (name, _) in enumerate(entries) if name.encode() > after.encode()),
                len(entries),
            )
        elif "start-after" in params:
            after = params["start-after"]
            start = next(
                (i for i, (name, _) in enumerate(entries) if name.encode() > after.encode()),
                len(entries),
            )
        page = entries[start : start + size]
        truncated = start + size < len(entries)
        parts = [
            '<?xml version="1.0" encoding="UTF-8"?>\n',
            f'<ListBucketResult xmlns="{NAMESPACE}">',
            f"<Name>{escape(bucket)}</Name><Prefix>{escape(prefix)}</Prefix>",
            f"<KeyCount>{len(page)}</KeyCount><MaxKeys>{max_keys}</MaxKeys>",
        ]
        if delimiter is not None:
            parts.append(f"<Delimiter>{escape(delimiter)}</Delimiter>")
        parts.append(f"<IsTruncated>{'true' if truncated else 'false'}</IsTruncated>")
        if token is not None:
            parts.append(f"<ContinuationToken>{escape(token)}</ContinuationToken>")
        if truncated:
            last = page[-1][0]
            next_token = base64.urlsafe_b64encode(last.encode()).decode()
            parts.append(f"<NextContinuationToken>{next_token}</NextContinuationToken>")
        for name, common in page:
            if common:
                parts.append(f"<CommonPrefixes><Prefix>{escape(name)}</Prefix></CommonPrefixes>")
                continue
            object_size, etag, mtime = self._listed_facts(bucket, name)
            parts.append(
                f"<Contents><Key>{escape(name)}</Key><LastModified>{_iso(mtime)}</LastModified>"
                f"<ETag>&quot;{etag}&quot;</ETag><Size>{object_size}</Size>"
                "<StorageClass>STANDARD</StorageClass></Contents>"
            )
            with self._lock:
                mutation = self._mutations.get((bucket, name))
                if mutation is not None and not mutation[1]:
                    self._mutations[(bucket, name)] = (mutation[0], True)
        parts.append("</ListBucketResult>")
        return _Reply(200, {"Content-Type": "application/xml"}, "".join(parts).encode())

    def _object(self, method: str, bucket: str, key: str, headers: Mapping[str, str]) -> _Reply:
        found = self._content(bucket, key)
        if found is None:
            return _error(404, "NoSuchKey", "the specified key does not exist")
        content, mtime = found
        answer = self._object_headers(content, mtime)
        match = headers.get("if-match")
        if match is not None:
            wanted = {part.strip().strip('"') for part in match.split(",")}
            if "*" not in wanted and answer["ETag"].strip('"') not in wanted:
                return _error(412, "PreconditionFailed", "at least one precondition failed")
        total = len(content)
        wanted_range = headers.get("range")
        status, body = 200, content
        if (
            wanted_range is not None
            and wanted_range.startswith("bytes=")
            and "," not in wanted_range
        ):
            first_text, _, last_text = wanted_range[len("bytes=") :].partition("-")
            try:
                if first_text == "":
                    first = max(0, total - int(last_text))
                    last = total - 1
                else:
                    first = int(first_text)
                    last = int(last_text) if last_text else total - 1
            except ValueError:
                first, last = 0, total - 1
            else:
                if first >= total or first > last:
                    reply = _error(416, "InvalidRange", "the requested range is not satisfiable")
                    reply.headers["Content-Range"] = f"bytes */{total}"
                    return reply
                last = min(last, total - 1)
                status, body = 206, content[first : last + 1]
                answer["Content-Range"] = f"bytes {first}-{last}/{total}"
        answer["Content-Length"] = str(len(body))
        return _Reply(status, answer, b"" if method == "HEAD" else body)

    # --- in the process -------------------------------------------------------------------------

    def handle(self, request: httpx.Request) -> httpx.Response:
        """The handler of :meth:`transport`."""
        url = request.url
        reply = self.respond(
            request.method,
            url.netloc.decode("ascii"),
            url.raw_path.decode("ascii").split("?", 1)[0],
            url.query.decode("ascii"),
            dict(request.headers),
        )
        if reply.reset:
            raise httpx.ReadError("connection reset by the fake", request=request)
        headers = dict(reply.headers)
        if reply.cut is not None:
            # the full length is announced and half is sent
            headers["Content-Length"] = str(len(reply.body))
            return httpx.Response(
                reply.status, headers=headers, stream=_Cut(reply.body, reply.cut, request)
            )
        if request.method == "HEAD" and "Content-Length" in headers:
            return httpx.Response(reply.status, headers=headers, content=b"")
        return httpx.Response(reply.status, headers=headers, content=reply.body)

    def transport(self) -> httpx.MockTransport:
        """An ``httpx.MockTransport`` on :meth:`handle`."""
        return httpx.MockTransport(self.handle)

    # --- in a subprocess ------------------------------------------------------------------------

    @contextmanager
    def serve(self) -> Iterator[str]:
        """Serve the buckets on 127.0.0.1 at a free port for the length of the block; yields
        the endpoint (``http://127.0.0.1:PORT``) to set as ``SURFCENSUS_S3_ENDPOINT``."""
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: object) -> None:
                pass

            def _answer(self) -> None:
                parts = urlsplit(self.path)
                length = int(self.headers.get("Content-Length") or 0)
                if length:
                    self.rfile.read(length)
                reply = fake.respond(
                    self.command,
                    self.headers.get("Host", ""),
                    parts.path,
                    parts.query,
                    dict(self.headers.items()),
                )
                if reply.reset:
                    # SO_LINGER 0: the close sends a reset, not a clean end
                    self.connection.setsockopt(
                        socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
                    )
                    self.close_connection = True
                    return
                self.send_response(reply.status)
                for name, value in reply.headers.items():
                    if name != "Content-Length":
                        self.send_header(name, value)
                # a HEAD answer has no body and the length of the object
                length = reply.headers.get("Content-Length", str(len(reply.body)))
                self.send_header(
                    "Content-Length", length if reply.cut is None else str(len(reply.body))
                )
                if reply.cut is not None:
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(reply.body[: reply.cut])
                    self.wfile.flush()
                    self.close_connection = True
                    return
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(reply.body)

            do_GET = do_HEAD = do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _answer

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}"
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


class _Cut(httpx.SyncByteStream):
    """A body that ends early with a read error, as a dropped connection does."""

    def __init__(self, body: bytes, cut: int, request: httpx.Request) -> None:
        self._body = body
        self._cut = cut
        self._request = request

    def __iter__(self) -> Iterator[bytes]:
        if self._cut:
            yield self._body[: self._cut]
        raise httpx.RemoteProtocolError(
            "peer closed connection without sending complete message body", request=self._request
        )


# --- buckets made of the synthetic cohorts ----------------------------------------------------

FAKE_ACCESSION = "ds999999"
"""The accession of the fake dataset: no accession of the repository but ds000102 and ds003020
names a real one."""


def bucket_of(
    cohort: Path | SyntheticCohort,
    accession: str = FAKE_ACCESSION,
    *,
    root: Path,
    fmriprep_version: str = "23.0.2",
    fs_root: str = "sourcedata/freesurfer",
) -> Path:
    """A bucket directory under ``root`` that holds ``cohort`` (a ``SyntheticCohort``, or a
    subjects directory as a ``Path``) as an OpenNeuroDerivatives dataset:
    ``fmriprep/<accession>-fmriprep/{dataset_description.json, <fs_root> -> cohort}``. The
    ``fsaverage`` that the cohort links next to its subjects comes along. Returns the bucket
    directory, to give :class:`FakeS3` as ``{"openneuro-derivatives": path}``. Nothing of the
    cohort is written to."""
    # a Path has an attribute root as well: "/"
    subjects = cohort if isinstance(cohort, Path) else cohort.root
    dataset = root / "fmriprep" / f"{accession}-fmriprep"
    (dataset / fs_root).parent.mkdir(parents=True, exist_ok=True)
    (dataset / fs_root).symlink_to(subjects.resolve(), target_is_directory=True)
    description = {
        "Name": "fMRIPrep - fMRI PREProcessing workflow",
        "BIDSVersion": "1.4.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "fMRIPrep", "Version": fmriprep_version}],
    }
    (dataset / "dataset_description.json").write_text(json.dumps(description, indent=2))
    return root


def key_of(
    accession: str, label: str, location: str, fs_root: str = "sourcedata/freesurfer"
) -> str:
    """The bucket key of ``location`` of subject ``label`` in a dataset of :func:`bucket_of`."""
    return f"fmriprep/{accession}-fmriprep/{fs_root}/{label}/{quote(location, safe='/')}"
