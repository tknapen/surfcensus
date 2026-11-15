"""The anonymous, read-only S3 client of ``remote/s3.py`` against the fake bucket.

Listing (pages, delimiter, encoded keys), HEAD, ranged and conditional GET, downloads to a
temporary name, the retries with an injected sleep, the mapping of statuses to reasons, the
limit on concurrent requests, and what an error detail may hold. Nothing here touches the
network: every request goes to ``tests/fakes3.py`` or to a stand-in transport.
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import pytest
from fakes3 import FakeS3  # pyrefly: ignore[missing-import]

from surfcensus import remote
from surfcensus.config import RemoteConfig
from surfcensus.remote import s3 as s3mod
from surfcensus.remote.s3 import (
    CHUNK_BYTES,
    RETRY_CAP_S,
    DownloadCancelled,
    S3Client,
    S3Object,
    backoff_delay,
)
from surfcensus.sources.base import SourceError

BUCKET = "openneuro-derivatives"
PREFIX = "fmriprep/ds999999-fmriprep/sourcedata/freesurfer/sub-01/"
KEY = PREFIX + "surf/lh.white"
CONTENT = bytes(range(256)) * 40  # 10,240 bytes


def write(root: Path, key: str, content: bytes) -> Path:
    path = root.joinpath(*key.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


class Sleeps:
    """The sleep of a client under test: it records what it is asked to wait, and waits not."""

    def __init__(self) -> None:
        self.waits: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)


@pytest.fixture
def bucket(tmp_path: Path) -> Path:
    root = tmp_path / "bucket"
    write(root, KEY, CONTENT)
    write(root, PREFIX + "surf/rh.white", b"right")
    write(root, PREFIX + "mri/T1.mgz", b"t1" * 100)
    write(root, PREFIX + "scripts/recon-all.log", b"log line\n" * 1000)
    write(root, PREFIX + "label/odd name+plus é.txt", b"odd")
    write(root, "fmriprep/ds999999-fmriprep/dataset_description.json", b"{}")
    return root


@pytest.fixture
def fake(bucket: Path) -> FakeS3:
    return FakeS3({BUCKET: bucket})


def make(
    fake: FakeS3 | None = None,
    *,
    transport: httpx.BaseTransport | None = None,
    retries: int = 3,
    concurrency: int = 4,
    addressing: s3mod.Addressing | None = None,
) -> tuple[S3Client, Sleeps]:
    sleeps = Sleeps()
    chosen = transport if transport is not None else (fake or FakeS3({})).transport()
    client = S3Client(
        BUCKET,
        config=RemoteConfig(max_concurrency=concurrency, max_retries=retries, timeout_s=10),
        addressing=addressing,
        transport=chosen,
        sleep=sleeps,
    )
    return client, sleeps


def listed(client: S3Client, key: str) -> S3Object:
    found = [o for o in client.list(key) if isinstance(o, S3Object) and o.key == key]
    assert len(found) == 1
    return found[0]


def assert_private(error: SourceError, *extra: str) -> None:
    """The detail of ``error`` holds no URL, no key or prefix, and no ETag of the bucket."""
    detail = error.detail
    for bad in ("amazonaws", "http://", "https://", "fmriprep/", "ds999999", *extra):
        assert bad not in detail, (bad, detail)
    md5 = hashlib.md5(CONTENT).hexdigest()
    assert md5 not in detail


# --- backoff ------------------------------------------------------------------------------------


def test_backoff_is_full_jitter_doubling_to_the_cap() -> None:
    rng = random.Random(0)
    for attempt in range(1, 12):
        ceiling = min(RETRY_CAP_S, 0.5 * 2 ** (attempt - 1))
        values = [backoff_delay(attempt, rng=rng) for _ in range(200)]
        assert all(0 <= v <= ceiling for v in values)
        assert max(values) > ceiling * 0.8
    assert backoff_delay(10_000, rng=rng) <= RETRY_CAP_S
    assert backoff_delay(1, 7.0) == 7.0
    assert backoff_delay(1, 100.0) == RETRY_CAP_S
    assert backoff_delay(2, -3.0) == 0.0
    assert backoff_delay(1, float("nan"), rng=random.Random(1)) <= 0.5
    with pytest.raises(ValueError):
        backoff_delay(0)


# --- listing ------------------------------------------------------------------------------------


@pytest.mark.parametrize("page_size", [1000, 7])
def test_a_listing_of_2500_keys_goes_page_after_page(tmp_path: Path, page_size: int) -> None:
    root = tmp_path / "bucket"
    for index in range(2500):
        write(root, f"p/{index:04d}/f", b"")
    write(root, "q/outside", b"x")
    fake = FakeS3({BUCKET: root}, page_size=page_size)
    # the fake walks its directory for every page; the keys do not change in this test, so
    # one walk serves the 358 pages of size 7 (33 s without, on one machine)
    walked = fake.keys(BUCKET)
    fake.keys = lambda bucket: walked  # pyrefly: ignore[bad-assignment]
    client, sleeps = make(fake)
    objects = list(client.list("p/"))
    assert [o.key for o in objects if isinstance(o, S3Object)] == [
        k for k in fake.keys(BUCKET) if k.startswith("p/")
    ]
    assert len(objects) == 2500
    pages = [entry for entry in fake.log if entry.method == "LIST"]
    assert len(pages) == -(-2500 // page_size)
    assert all(entry.query["list-type"] == "2" for entry in pages)
    assert "continuation-token" not in pages[0].query
    assert all("continuation-token" in entry.query for entry in pages[1:])
    assert sleeps.waits == []


def test_a_listing_gives_size_etag_and_time(fake: FakeS3) -> None:
    client, _ = make(fake)
    obj = listed(client, KEY)
    assert obj.size == len(CONTENT)
    assert obj.etag == hashlib.md5(CONTENT).hexdigest()
    assert obj.last_modified.endswith("Z") and "T" in obj.last_modified


def test_a_delimiter_yields_common_prefixes_in_bucket_order(fake: FakeS3) -> None:
    client, _ = make(fake, addressing="path")
    found = list(client.list(PREFIX, delimiter="/"))
    assert found == [PREFIX + "label/", PREFIX + "mri/", PREFIX + "scripts/", PREFIX + "surf/"]
    top = list(client.list("fmriprep/ds999999-fmriprep/", delimiter="/"))
    assert [e if isinstance(e, str) else e.key for e in top] == [
        "fmriprep/ds999999-fmriprep/dataset_description.json",
        "fmriprep/ds999999-fmriprep/sourcedata/",
    ]


def test_objects_and_common_prefixes_of_a_page_are_merged_in_order(tmp_path: Path) -> None:
    root = tmp_path / "bucket"
    for key in ("a/b", "a/c/x", "a/d", "a/e/y", "a/f"):
        write(root, key, b"1")
    client, _ = make(FakeS3({BUCKET: root}))
    found = [e if isinstance(e, str) else e.key for e in client.list("a/", delimiter="/")]
    assert found == ["a/b", "a/c/", "a/d", "a/e/", "a/f"]


def page(
    entries: str, *, encoded: bool = False, truncated: str = "false", token: str = ""
) -> bytes:
    ns = s3mod.S3_NAMESPACE
    encoding = "<EncodingType>url</EncodingType>" if encoded else ""
    next_token = f"<NextContinuationToken>{token}</NextContinuationToken>" if token else ""
    return (
        f'<?xml version="1.0" encoding="UTF-8"?><ListBucketResult xmlns="{ns}">'
        f"<Name>{BUCKET}</Name>{encoding}<IsTruncated>{truncated}</IsTruncated>{next_token}"
        f"{entries}</ListBucketResult>"
    ).encode()


def contents(key: str) -> str:
    return (
        f"<Contents><Key>{key}</Key><LastModified>2026-01-01T00:00:00.000Z</LastModified>"
        "<ETag>&quot;abc&quot;</ETag><Size>3</Size></Contents>"
    )


def test_keys_of_an_url_encoded_answer_are_decoded() -> None:
    asked: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        asked.append(request.url)
        body = page(
            contents("p/odd%20name%2Bplus+%C3%A9.txt")
            + "<CommonPrefixes><Prefix>p/sub%2001/</Prefix></CommonPrefixes>",
            encoded=True,
        )
        return httpx.Response(200, content=body)

    client, _ = make(transport=httpx.MockTransport(handler))
    found = list(client.list("p/", delimiter="/"))
    assert found[0] == S3Object("p/odd name+plus é.txt", 3, "abc", "2026-01-01T00:00:00.000Z")
    assert found[1] == "p/sub 01/"
    assert asked[0].params["encoding-type"] == "url"
    plain = page(contents("p/a%20b"))
    client2, _ = make(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=plain)))
    assert [o.key for o in client2.list("p/") if isinstance(o, S3Object)] == ["p/a%20b"]


def test_objects_given_before_the_common_prefixes_of_a_page_come_in_bucket_order() -> None:
    # S3 writes the Contents of a page before its CommonPrefixes; the fake interleaves them
    body = page(
        contents("p/b") + contents("p/d") + "<CommonPrefixes><Prefix>p/a/</Prefix></CommonPrefixes>"
        "<CommonPrefixes><Prefix>p/c/</Prefix></CommonPrefixes>"
    )
    client, _ = make(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=body)))
    found = [e if isinstance(e, str) else e.key for e in client.list("p/", delimiter="/")]
    assert found == ["p/a/", "p/b", "p/c/", "p/d"]


def test_keys_with_spaces_plus_and_non_ascii_are_fetched(fake: FakeS3) -> None:
    client, _ = make(fake)
    obj = listed(client, PREFIX + "label/odd name+plus é.txt")
    body, head = client.get(obj.key, if_match=obj.etag)
    assert (body, head.size, head.etag) == (b"odd", 3, obj.etag)


def test_a_listing_that_repeats_its_token_or_has_none_is_invalid() -> None:
    looping = page(contents("p/a"), truncated="true", token="t1")
    client, _ = make(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=looping)))
    with pytest.raises(SourceError) as caught:
        list(client.list("p/"))
    assert caught.value.reason == "invalid"
    endless = page(contents("p/a"), truncated="true")
    client, _ = make(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=endless)))
    with pytest.raises(SourceError, match="continuation token"):
        list(client.list("p/"))


def test_a_listing_of_no_such_bucket_is_not_found(bucket: Path) -> None:
    client = S3Client(
        "no-such-bucket", config=RemoteConfig(max_retries=1), transport=FakeS3({}).transport()
    )
    with pytest.raises(SourceError) as caught:
        list(client.list("p/"))
    assert caught.value.reason == "not found"
    assert caught.value.detail.startswith("listing: HTTP 404 NoSuchBucket")


def test_a_forbidden_listing_is_unreadable_and_not_retried() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(403, content=b"<Error><Code>AccessDenied</Code></Error>")

    client, sleeps = make(transport=httpx.MockTransport(handler))
    with pytest.raises(SourceError) as caught:
        list(client.list(PREFIX))
    assert (caught.value.reason, len(calls), sleeps.waits) == ("unreadable", 1, [])
    assert caught.value.detail == "listing: HTTP 403 AccessDenied"


@pytest.mark.parametrize("kind", ["500", "503", "slowdown", "reset", "truncate"])
def test_a_listing_page_is_retried(fake: FakeS3, kind: str) -> None:
    fake.fault(PREFIX, kind, times=2, methods=("LIST",))  # pyrefly: ignore[bad-argument-type]
    client, sleeps = make(fake)
    assert len(list(client.list(PREFIX))) == 5
    assert len(sleeps.waits) == 2
    assert len([e for e in fake.log if e.method == "LIST"]) == 3


def test_virtual_and_path_addressing(fake: FakeS3) -> None:
    hosts: list[str] = []
    inner = fake.transport()

    def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(f"{request.url.host}{request.url.raw_path.decode().split('?')[0]}")
        return inner.handle_request(request)

    for addressing in ("virtual", "path"):
        client, _ = make(
            transport=httpx.MockTransport(handler),
            addressing=addressing,  # pyrefly: ignore[bad-argument-type]
        )
        assert len(list(client.list(PREFIX))) == 5
        assert client.get(KEY)[0] == CONTENT
    assert hosts[0] == f"{BUCKET}.s3.amazonaws.com/"
    assert hosts[1] == f"{BUCKET}.s3.amazonaws.com/{KEY}"
    assert hosts[2] == f"s3.amazonaws.com/{BUCKET}/"
    assert hosts[3] == f"s3.amazonaws.com/{BUCKET}/{KEY}"
    dotted = S3Client("openneuro.org", config=RemoteConfig(), transport=inner)
    assert dotted.addressing == "path"


# --- HEAD and GET --------------------------------------------------------------------------------


def test_head_gives_size_etag_time_and_version(bucket: Path) -> None:
    fake = FakeS3({BUCKET: bucket}, version_ids=True)
    client, _ = make(fake)
    head = client.head(KEY)
    assert head.size == len(CONTENT)
    assert head.etag == hashlib.md5(CONTENT).hexdigest()
    assert head.last_modified.endswith("GMT")
    assert head.version_id is not None and head.version_id.startswith("v")
    assert make(FakeS3({BUCKET: bucket}))[0].head(KEY).version_id is None
    assert fake.methods == {"HEAD"}


def test_a_range_gives_206_its_bytes_and_the_whole_size(fake: FakeS3) -> None:
    client, _ = make(fake)
    body, head = client.get(KEY, byte_range=(100, 1123))
    assert body == CONTENT[100:1124]
    assert head.size == len(CONTENT)
    assert fake.log[-1].status == 206 and fake.log[-1].range == "bytes=100-1123"
    body, head = client.get(KEY, byte_range=(10_000, 10_239))
    assert body == CONTENT[10_000:] and head.size == len(CONTENT)
    with pytest.raises(ValueError):
        client.get(KEY, byte_range=(5, 4))


def test_an_answer_of_200_to_a_range_is_cut_to_the_range() -> None:
    md5 = hashlib.md5(CONTENT).hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"ETag": f'"{md5}"'}, content=CONTENT)

    client, _ = make(transport=httpx.MockTransport(handler))
    body, head = client.get(KEY, byte_range=(5, 9))
    assert (body, head.size) == (CONTENT[5:10], len(CONTENT))


def test_a_whole_get_is_checked_and_sends_if_match(fake: FakeS3) -> None:
    client, _ = make(fake)
    obj = listed(client, KEY)
    body, head = client.get(KEY, if_match=obj.etag)
    assert body == CONTENT and head.etag == obj.etag and head.size == len(CONTENT)
    assert fake.log[-1].if_match == f'"{obj.etag}"'


def test_an_if_match_mismatch_is_changed_and_not_retried(fake: FakeS3) -> None:
    client, sleeps = make(fake)
    with pytest.raises(SourceError) as caught:
        client.get(KEY, if_match="0" * 32, location="surf/lh.white")
    assert caught.value.reason == "changed"
    assert caught.value.detail == "surf/lh.white: HTTP 412 PreconditionFailed"
    assert sleeps.waits == [] and len(fake.log) == 1
    assert_private(caught.value)


def test_an_object_that_changed_after_its_listing_is_changed(fake: FakeS3, tmp_path: Path) -> None:
    fake.mutate_after_list(BUCKET, KEY)
    client, _ = make(fake)
    obj = listed(client, KEY)  # the listing names the old facts and arms the mutation
    with pytest.raises(SourceError) as caught:
        client.get(KEY, if_match=obj.etag)
    assert caught.value.reason == "changed"
    with pytest.raises(SourceError) as caught:
        client.download(KEY, tmp_path / "lh.white", expect=obj)
    assert caught.value.reason == "changed"
    assert not (tmp_path / "lh.white").exists()


def status_transport(status: int, code: str = "X") -> httpx.MockTransport:
    body = f"<Error><Code>{code}</Code></Error>".encode()
    return httpx.MockTransport(lambda request: httpx.Response(status, content=body))


@pytest.mark.parametrize(
    ("status", "code", "reason"),
    [
        (404, "NoSuchKey", "changed"),
        (403, "AccessDenied", "unreadable"),
        (412, "PreconditionFailed", "changed"),
        (416, "InvalidRange", "invalid"),
        # what says something about the way to the bucket and not about the object: no row
        # of a subject (a proxy's 407, a 401, a redirect, a request the server refuses)
        (400, "InvalidRequest", "unreachable"),
        (401, "X", "unreachable"),
        (407, "X", "unreachable"),
        (301, "PermanentRedirect", "unreachable"),
    ],
)
def test_statuses_map_to_reasons_without_retry(
    status: int, code: str, reason: str, tmp_path: Path
) -> None:
    for call in ("head", "get", "download"):
        client, sleeps = make(transport=status_transport(status, code))
        with pytest.raises(SourceError) as caught:
            if call == "head":
                client.head(KEY, location="surf/lh.white")
            elif call == "get":
                client.get(KEY, location="surf/lh.white")
            else:
                obj = S3Object(KEY, 10, "a" * 32, "")
                client.download(KEY, tmp_path / "x", expect=obj, location="surf/lh.white")
        assert caught.value.reason == reason, (call, status)
        assert sleeps.waits == []
        said = f"HTTP {status}" if call == "head" else f"HTTP {status} {code}"
        assert caught.value.detail == f"surf/lh.white: {said}"
        assert list(tmp_path.iterdir()) == []
        assert_private(caught.value)


def test_the_fake_gives_404_and_416(fake: FakeS3) -> None:
    client, sleeps = make(fake)
    with pytest.raises(SourceError) as caught:
        client.head(PREFIX + "surf/lh.gone")
    assert (caught.value.reason, caught.value.detail) == ("changed", "lh.gone: HTTP 404")
    with pytest.raises(SourceError) as caught:
        client.get(KEY, byte_range=(1 << 20, (1 << 20) + 5))
    assert caught.value.reason == "invalid"
    assert sleeps.waits == []


@pytest.mark.parametrize("kind", ["500", "503", "slowdown", "reset", "truncate"])
def test_object_requests_are_retried(fake: FakeS3, tmp_path: Path, kind: str) -> None:
    client, sleeps = make(fake)
    obj = listed(client, KEY)
    for call in ("get", "download", "head"):
        if call == "head" and kind == "truncate":
            continue  # a HEAD has no body to cut
        fake.fault(KEY, kind, times=2)  # pyrefly: ignore[bad-argument-type]
        sleeps.waits.clear()
        if call == "get":
            assert client.get(KEY)[0] == CONTENT
        elif call == "head":
            assert client.head(KEY).size == len(CONTENT)
        else:
            dest = tmp_path / f"{kind}.bin"
            client.download(KEY, dest, expect=obj)
            assert dest.read_bytes() == CONTENT
            assert sorted(p.name for p in tmp_path.iterdir()) == sorted(["bucket", f"{kind}.bin"])
        assert len(sleeps.waits) == 2, call
        assert all(0 <= wait <= 1.0 for wait in sleeps.waits)


def test_retries_that_run_out_are_unreachable(fake: FakeS3) -> None:
    fake.fault(KEY, "reset", times=None)
    client, sleeps = make(fake, retries=3)
    with pytest.raises(SourceError) as caught:
        client.get(KEY, location="surf/lh.white")
    assert caught.value.reason == "unreachable"
    assert caught.value.detail.startswith("surf/lh.white: ReadError")
    assert caught.value.detail.endswith("(4 attempts)")
    assert len(sleeps.waits) == 3 and len(fake.log) == 4
    assert_private(caught.value)
    fake.clear()
    fake.fault(PREFIX, "503", times=None, methods=("LIST",))
    with pytest.raises(SourceError) as caught:
        list(client.list(PREFIX))
    assert caught.value.reason == "unreachable"
    assert caught.value.detail == "listing: HTTP 503 ServiceUnavailable (4 attempts)"


def test_no_retry_at_all_with_max_retries_0(fake: FakeS3) -> None:
    fake.fault(KEY, "500", times=1)
    client, sleeps = make(fake, retries=0)
    with pytest.raises(SourceError) as caught:
        client.get(KEY)
    assert caught.value.reason == "unreachable"
    assert caught.value.detail.endswith("(1 attempt)")
    assert sleeps.waits == []


def test_retry_after_is_honoured_up_to_the_cap(fake: FakeS3) -> None:
    client, sleeps = make(fake)
    fake.fault(KEY, "slowdown", times=1, retry_after=7)
    fake.fault(KEY, "slowdown", times=1, retry_after=600)
    assert client.get(KEY)[0] == CONTENT
    assert sleeps.waits == [7.0, RETRY_CAP_S]


def test_a_body_whose_md5_is_not_its_etag_is_retried_then_changed(tmp_path: Path) -> None:
    """Once the retries ran out it is the object that does not have its listed bytes: a row
    of its subject (``changed``), not a bucket that cannot be reached."""
    md5 = hashlib.md5(CONTENT).hexdigest()
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, headers={"ETag": f'"{md5}"'}, content=b"x" * len(CONTENT))

    client, sleeps = make(transport=httpx.MockTransport(handler), retries=2)
    with pytest.raises(SourceError) as caught:
        client.get(KEY)
    assert caught.value.reason == "changed"
    assert "MD5" in caught.value.detail and "(3 attempts)" in caught.value.detail
    assert (len(calls), len(sleeps.waits)) == (3, 2)
    assert_private(caught.value)
    calls.clear()
    with pytest.raises(SourceError) as caught:
        client.download(KEY, tmp_path / "out", expect=S3Object(KEY, len(CONTENT), md5, ""))
    assert caught.value.reason == "changed"
    assert len(calls) == 3
    assert list(tmp_path.iterdir()) == []


class Short(httpx.SyncByteStream):
    """A body that ends early without an error, which a server or proxy can give."""

    def __iter__(self) -> Iterator[bytes]:
        yield b"x" * 50


def test_a_body_shorter_than_its_length_is_retried_also_without_a_read_error(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        headers = {"ETag": '"abcdef-2"'}
        if len(calls) <= 3:
            headers["Content-Length"] = "100"  # the get: a length is announced
        return httpx.Response(200, headers=headers, stream=Short())

    client, sleeps = make(transport=httpx.MockTransport(handler), retries=2)
    with pytest.raises(SourceError) as caught:
        client.get(KEY)
    assert caught.value.reason == "unreachable" and "cut short" in caught.value.detail
    assert (len(calls), len(sleeps.waits)) == (3, 2)
    # the download: no length announced, and fewer bytes than listed
    with pytest.raises(SourceError) as caught:
        client.download(KEY, tmp_path / "x", expect=S3Object(KEY, 100, "abcdef-2", ""))
    assert caught.value.reason == "unreachable" and "fewer bytes" in caught.value.detail
    assert len(calls) == 6 and list(tmp_path.iterdir()) == []


def test_a_multipart_etag_is_not_checked_as_an_md5() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"ETag": '"abcdef-2"'}, content=b"payload")

    client, _ = make(transport=httpx.MockTransport(handler))
    body, head = client.get(KEY)
    assert (body, head.etag) == (b"payload", "abcdef-2")


def test_a_proxy_refusal_is_unreachable_without_retries() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ProxyError("403 Forbidden", request=request)

    client, sleeps = make(transport=httpx.MockTransport(handler))
    with pytest.raises(SourceError) as caught:
        client.head(KEY, location="surf/lh.white")
    assert caught.value.reason == "unreachable"
    assert caught.value.detail == "surf/lh.white: the proxy refused: 403 Forbidden"
    assert sleeps.waits == []


def test_a_key_in_the_text_of_an_exception_is_not_in_the_detail() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"cannot reach {request.url} for {KEY}", request=request)

    client, _ = make(transport=httpx.MockTransport(handler), retries=1)
    with pytest.raises(SourceError) as caught:
        client.get(KEY)
    assert caught.value.reason == "unreachable"
    assert_private(caught.value, "lh.white" + "/")
    with pytest.raises(SourceError) as caught:
        list(client.list(PREFIX))
    assert_private(caught.value)


# --- downloads -----------------------------------------------------------------------------------


def test_a_download_writes_a_temporary_name_and_renames_it(fake: FakeS3, tmp_path: Path) -> None:
    client, _ = make(fake)
    obj = listed(client, KEY)
    out = tmp_path / "out"
    out.mkdir()
    dest = out / "lh.white"
    seen: list[list[str]] = []
    inner = fake.transport()

    class Watching(httpx.SyncByteStream):
        def __init__(self, body: bytes) -> None:
            self.body = body

        def __iter__(self) -> Iterator[bytes]:
            for start in range(0, len(self.body), 1000):
                seen.append(sorted(os.listdir(out)))
                yield self.body[start : start + 1000]

    def handler(request: httpx.Request) -> httpx.Response:
        answer = inner.handle_request(request)
        return httpx.Response(
            answer.status_code, headers=answer.headers, stream=Watching(answer.read())
        )

    watched, _ = make(transport=httpx.MockTransport(handler))
    head = watched.download(KEY, dest, expect=obj)
    assert dest.read_bytes() == CONTENT
    assert head.size == len(CONTENT) and head.etag == obj.etag
    assert os.listdir(out) == ["lh.white"]
    later = [names for names in seen if names]
    assert later and all(
        len(names) == 1 and names[0].startswith(".lh.white.") and names[0].endswith(".tmp")
        for names in later
    )


def test_a_download_replaces_an_earlier_file_only_when_whole(fake: FakeS3, tmp_path: Path) -> None:
    client, _ = make(fake, retries=0)
    obj = listed(client, KEY)
    (tmp_path / "out").mkdir()
    dest = tmp_path / "out" / "lh.white"
    dest.write_bytes(b"earlier")
    fake.fault(KEY, "truncate", times=1)
    with pytest.raises(SourceError):
        client.download(KEY, dest, expect=obj)
    assert dest.read_bytes() == b"earlier"
    assert os.listdir(tmp_path / "out") == ["lh.white"]
    client.download(KEY, dest, expect=obj)
    assert dest.read_bytes() == CONTENT
    assert fake.log[-1].if_match == f'"{obj.etag}"'


def test_a_download_of_another_size_than_listed_is_changed(fake: FakeS3, tmp_path: Path) -> None:
    client, sleeps = make(fake)
    obj = listed(client, KEY)
    wrong = S3Object(obj.key, obj.size + 1, obj.etag, obj.last_modified)
    (tmp_path / "out").mkdir()
    with pytest.raises(SourceError) as caught:
        client.download(KEY, tmp_path / "out" / "x", expect=wrong)
    assert caught.value.reason == "changed" and sleeps.waits == []
    assert list((tmp_path / "out").iterdir()) == []


def chunked(
    chunks: int, *, after: Callable[[int], None]
) -> tuple[httpx.MockTransport, list[int], bytes]:
    """A transport that serves ``chunks`` chunks of ``CHUNK_BYTES`` and calls ``after(i)``
    once chunk ``i`` was handed out; ``sent`` counts the chunks handed out."""
    body = os.urandom(chunks * CHUNK_BYTES)
    md5 = hashlib.md5(body).hexdigest()
    sent: list[int] = []

    class Stream(httpx.SyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            for index in range(chunks):
                sent.append(index)
                yield body[index * CHUNK_BYTES : (index + 1) * CHUNK_BYTES]
                after(index)

    def handler(request: httpx.Request) -> httpx.Response:
        headers = {"ETag": f'"{md5}"', "Content-Length": str(len(body))}
        return httpx.Response(200, headers=headers, stream=Stream())

    return httpx.MockTransport(handler), sent, body


def test_a_cancel_stops_within_one_chunk_and_leaves_no_file(tmp_path: Path) -> None:
    cancel = threading.Event()

    def after(index: int) -> None:
        if index == 1:
            cancel.set()

    transport, sent, body = chunked(8, after=after)
    client, sleeps = make(transport=transport)
    obj = S3Object(KEY, len(body), hashlib.md5(body).hexdigest(), "")
    with pytest.raises(DownloadCancelled) as caught:
        client.download(KEY, tmp_path / "big", expect=obj, cancel=cancel)
    assert isinstance(caught.value, SourceError) and caught.value.reason == "unreachable"
    assert len(sent) <= 3
    assert list(tmp_path.iterdir()) == []
    assert sleeps.waits == []
    # a cancel that is set before the call sends nothing
    with pytest.raises(DownloadCancelled):
        client.download(KEY, tmp_path / "big", expect=obj, cancel=cancel)
    assert len(sent) <= 3


def test_a_cancel_during_the_wait_before_a_retry_ends_the_wait(tmp_path: Path) -> None:
    """A client that sleeps for real waits on the cancel event before a retry, so a stop does
    not wait out the backoff: here the bucket asks for 30 s and the cancel is set while the
    first answer is made."""
    cancel = threading.Event()
    asked: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        asked.append(request.method)
        cancel.set()
        return httpx.Response(503, headers={"Retry-After": "30"}, content=b"")

    client = S3Client(
        BUCKET,
        config=RemoteConfig(max_concurrency=4, max_retries=3, timeout_s=10),
        transport=httpx.MockTransport(handler),
    )
    obj = S3Object(KEY, 5, "0" * 32, "")
    started = time.monotonic()
    with pytest.raises(DownloadCancelled):
        client.download(KEY, tmp_path / "big", expect=obj, cancel=cancel)
    # far below the 30 s asked for; the bound only tells a wait that ended from one that did not
    assert time.monotonic() - started < 15
    assert asked == ["GET"] and list(tmp_path.iterdir()) == []


def test_a_429_is_retried_like_a_503() -> None:
    answers = iter([429, 429, 200])
    md5 = hashlib.md5(CONTENT).hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        status = next(answers)
        if status != 200:
            return httpx.Response(status, content=b"slow down")
        return httpx.Response(200, headers={"ETag": f'"{md5}"'}, content=CONTENT)

    client, sleeps = make(transport=httpx.MockTransport(handler), retries=3)
    body, _ = client.get(KEY)
    assert body == CONTENT and len(sleeps.waits) == 2


def test_a_redirect_is_not_followed(tmp_path: Path) -> None:
    asked: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        asked.append(request.url.host)
        return httpx.Response(301, headers={"Location": "https://elsewhere.example/x"})

    client, _ = make(transport=httpx.MockTransport(handler))
    for call in (
        lambda: client.get(KEY),
        lambda: client.head(KEY),
        lambda: client.download(KEY, tmp_path / "x", expect=S3Object(KEY, 5, "0" * 32, "")),
        lambda: list(client.list(PREFIX)),
    ):
        asked.clear()
        with pytest.raises(SourceError) as caught:
            call()
        assert caught.value.reason == "unreachable"
        assert asked == [f"{BUCKET}.s3.amazonaws.com"]  # one request, none to the other host
        assert "elsewhere" not in caught.value.detail


def test_an_answer_with_another_etag_than_the_listed_one_is_changed(tmp_path: Path) -> None:
    """Also when the body is the listed one: the header says which object was sent."""
    md5 = hashlib.md5(CONTENT).hexdigest()
    other = "f" * 32
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, headers={"ETag": f'"{other}"'}, content=CONTENT)

    client, sleeps = make(transport=httpx.MockTransport(handler))
    with pytest.raises(SourceError) as caught:
        client.download(KEY, tmp_path / "out", expect=S3Object(KEY, len(CONTENT), md5, ""))
    assert caught.value.reason == "changed" and "ETag" in caught.value.detail
    assert (len(calls), sleeps.waits) == (1, []) and list(tmp_path.iterdir()) == []
    with pytest.raises(SourceError) as caught:
        client.get(KEY, if_match=md5)
    assert caught.value.reason == "changed"


def test_an_answer_without_an_etag_is_not_the_buckets(tmp_path: Path) -> None:
    """A page that something on the way sends with 200 (a proxy, a portal) has no ETag: the
    way to the bucket is at fault, not the object, and nothing is retried."""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, content=b"<html>sign in</html>")

    client, sleeps = make(transport=httpx.MockTransport(handler))
    with pytest.raises(SourceError) as caught:
        client.download(KEY, tmp_path / "out", expect=S3Object(KEY, 20, "0" * 32, ""))
    assert caught.value.reason == "unreachable" and "no ETag" in caught.value.detail
    assert (len(calls), sleeps.waits) == (1, []) and list(tmp_path.iterdir()) == []
    with pytest.raises(SourceError) as caught:
        client.get(KEY, if_match="0" * 32)
    assert caught.value.reason == "unreachable"


def test_a_stopped_client_starts_and_retries_nothing(tmp_path: Path) -> None:
    asked: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        asked.append(request.method)
        return httpx.Response(503, headers={"Retry-After": "30"}, content=b"")

    client = S3Client(
        BUCKET,
        config=RemoteConfig(max_concurrency=4, max_retries=3, timeout_s=10),
        transport=httpx.MockTransport(handler),
    )
    # stopped while the first answer is on its way: the wait of 30 s ends and no retry is made
    timer = threading.Timer(0.2, client.stop)
    timer.start()
    started = time.monotonic()
    with pytest.raises(DownloadCancelled):
        list(client.list(PREFIX))
    timer.join()
    assert time.monotonic() - started < 15 and asked == ["GET"]
    # and nothing is started while it is stopped, by any call
    for call in (
        lambda: client.get(KEY),
        lambda: client.head(KEY),
        lambda: client.download(KEY, tmp_path / "x", expect=S3Object(KEY, 5, "0" * 32, "")),
    ):
        with pytest.raises(DownloadCancelled):
            call()
    assert asked == ["GET"] and list(tmp_path.iterdir()) == []
    client.stopped.clear()
    with pytest.raises(SourceError):
        client.head(KEY)
    assert len(asked) > 1


def test_a_stop_ends_a_body_that_is_being_read() -> None:
    client_box: list[S3Client] = []

    def body() -> Iterator[bytes]:
        yield b"a" * 1000
        client_box[0].stop()
        yield b"b" * 1000

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"ETag": '"x-1"'}, content=body())

    client, _ = make(transport=httpx.MockTransport(handler))
    client_box.append(client)
    with pytest.raises(DownloadCancelled):
        client.get(KEY)


def test_a_read_that_stalls_is_given_up_after_the_stall_bound() -> None:
    client, _ = make()
    assert client.http.timeout.read == 10  # the fixture's timeout_s, below the bound
    slow = S3Client(BUCKET, config=RemoteConfig(timeout_s=120), transport=FakeS3({}).transport())
    assert slow.http.timeout.read == s3mod.READ_STALL_S == 30
    assert slow.http.timeout.connect == 120


def test_no_request_url_reaches_a_log(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    """``httpx`` logs every request with its URL at INFO; for the bucket hosts those lines
    are dropped, since the URL holds the accession, the key and the subject label."""
    write(tmp_path / BUCKET, KEY, CONTENT)
    fake = FakeS3({BUCKET: tmp_path / BUCKET})
    client, _ = make(fake)
    with caplog.at_level(logging.DEBUG):
        client.get(KEY)
        list(client.list(PREFIX))
        # a request of someone else's client to another host is logged as before
        other = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
        other.get("https://example.org/page")
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "example.org/page" in text
    assert "ds999999" not in text and "sub-01" not in text and "amazonaws" not in text


def test_an_interrupt_during_a_download_passes_and_leaves_no_file(tmp_path: Path) -> None:
    def after(index: int) -> None:
        if index == 0:
            raise KeyboardInterrupt

    transport, _, body = chunked(4, after=after)
    client, _ = make(transport=transport)
    obj = S3Object(KEY, len(body), hashlib.md5(body).hexdigest(), "")
    with pytest.raises(KeyboardInterrupt):
        client.download(KEY, tmp_path / "big", expect=obj)
    assert list(tmp_path.iterdir()) == []


def test_a_download_in_chunks_is_whole(tmp_path: Path) -> None:
    transport, sent, body = chunked(3, after=lambda index: None)
    client, _ = make(transport=transport)
    obj = S3Object(KEY, len(body), hashlib.md5(body).hexdigest(), "")
    client.download(KEY, tmp_path / "big", expect=obj)
    assert (tmp_path / "big").read_bytes() == body and len(sent) == 3


# --- the limit, the guard, the slots ------------------------------------------------------------


def test_sixteen_parallel_calls_keep_four_open(bucket: Path, tmp_path: Path) -> None:
    fake = FakeS3({BUCKET: bucket}, latency_s=0.05)
    client, _ = make(fake, concurrency=4)
    obj = listed(client, KEY)
    errors: list[BaseException] = []

    def work(index: int) -> None:
        try:
            if index % 4 == 0:
                client.download(KEY, tmp_path / f"d{index}", expect=obj)
            elif index % 4 == 1:
                client.head(KEY)
            elif index % 4 == 2:
                list(client.list(PREFIX))
            else:
                client.get(KEY, byte_range=(0, 9))
        except BaseException as exc:  # collected for the assertion
            errors.append(exc)

    before = remote.stats().in_flight
    threads = [threading.Thread(target=work, args=(i,)) for i in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert errors == []
    assert 2 <= fake.high_water <= 4
    assert remote.stats().in_flight == before


def test_requests_are_anonymous_get_and_head_only(fake: FakeS3, tmp_path: Path) -> None:
    client, _ = make(fake)
    obj = listed(client, KEY)
    client.head(KEY)
    client.get(KEY, byte_range=(0, 1), if_match=obj.etag)
    client.download(KEY, tmp_path / "x", expect=obj)
    fake.fault(KEY, "reset", times=1)
    client.get(KEY)
    assert fake.signed == []
    assert not any(entry.signed for entry in fake.log)
    assert fake.methods <= {"GET", "HEAD", "LIST"}


def test_every_streamed_answer_gives_its_slot_back(fake: FakeS3, tmp_path: Path) -> None:
    before = remote.stats().in_flight
    client, _ = make(fake, retries=1)
    obj = listed(client, KEY)
    fake.fault(KEY, "truncate", times=None)
    for call in (lambda: client.get(KEY), lambda: client.download(KEY, tmp_path / "x", expect=obj)):
        with pytest.raises(SourceError):
            call()
    fake.clear()
    with pytest.raises(SourceError):
        client.get(KEY, if_match="0" * 32)
    assert remote.stats().in_flight == before


# --- over a socket -------------------------------------------------------------------------------


def test_over_a_socket_cut_bodies_and_resets_are_retried(
    fake: FakeS3, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with fake.serve() as endpoint:
        monkeypatch.setenv(remote.ENDPOINT_ENV, endpoint)
        sleeps = Sleeps()
        client = S3Client(
            BUCKET, config=RemoteConfig(max_concurrency=4, max_retries=3), sleep=sleeps
        )
        with client:
            obj = listed(client, KEY)
            fake.fault(KEY, "truncate", times=1)
            fake.fault(KEY, "reset", times=1)
            assert client.get(KEY)[0] == CONTENT
            fake.fault(KEY, "truncate", times=1)
            client.download(KEY, tmp_path / "x", expect=obj)
            assert (tmp_path / "x").read_bytes() == CONTENT
            assert client.head(KEY).size == len(CONTENT)
    assert len(sleeps.waits) == 3
    assert fake.signed == []
