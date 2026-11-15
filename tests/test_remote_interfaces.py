"""What the packages of the remote stage share (docs/dev/remote_plan.md, work package R0):
the names, parameters and fields of ``remote/``, ``sources/openneuro_derivatives.py`` and
the new rules of ``sources/patterns.py``; the constants; the stubs and who writes them; and
what R0 wrote whole: the parsing of ``openneuro:dsXXXXXX``, the read-only transport, the
counters and the limit of the process, the names of the cache, the stored form of a remote
dataset in a local run (decision D47, item 1) and how a run is made, opened and rebuilt on it.

The names were fixed before the bodies were written, so that RA, RB, RC and RE can be written
side by side. A package that renames or reorders something fails here; a change that is meant
goes here as well, in the same change. A stub that is written keeps its name and parameters,
and its line in ``STUBS`` goes.

Nothing here touches the network: every request goes to the fake of ``tests/fakes3.py``.
"""

from __future__ import annotations

import dataclasses
import inspect
import pickle
import threading
from collections.abc import Collection, Iterator
from importlib import import_module
from pathlib import Path

import httpx
import pytest
import yaml
from fakes3 import FakeS3  # pyrefly: ignore[missing-import]
from synth_tree import write_light_subject  # pyrefly: ignore[missing-import]

from surfcensus import privacy, remote
from surfcensus.config import (
    ConfigError,
    RemoteConfig,
    RunConfig,
    SourceConfig,
    new_census_config,
    new_local_config,
    remote_dataset_config,
    run_key,
    with_changes,
)
from surfcensus.remote.cache import CACHE_DIR, Cache, CacheBudgetError
from surfcensus.remote.s3 import S3Client, S3Head, S3Object
from surfcensus.remote.transport import ReadOnlyTransport, default_transport, read_only, refusal
from surfcensus.run import Run, from_config, remote_index
from surfcensus.sources.base import DatasetRecord, FileRole, RemoteSource, SourceError
from surfcensus.sources.openneuro_derivatives import (
    OpenNeuroDerivatives,
    accession_of,
    dataset_root,
)
from surfcensus.sources.patterns import (
    is_subject_candidate_keys,
    key_variant,
    parse_pipeline_version,
    resolve_subject_files,
    roles_from_keys,
)
from surfcensus.store import Catalog

ACCESSION = "ds999999"
OTHER = "ds999998"
BUCKET = "openneuro-derivatives"
SALT = privacy.local_salt_from_hex("cd" * 32)
DATASET_KEY = privacy.HashKey(bytes(range(32)), "dataset-key")

INTERFACE: dict[str, dict[str, tuple[str, ...] | None]] = {
    # module -> name -> parameter names of a function, or None for a class or a constant
    "surfcensus.remote": {
        "DERIVATIVES_BUCKET": None,
        "RAW_BUCKET": None,
        "raw_prefix": ("accession",),
        "FMRIPREP_PREFIX": None,
        "MAX_CONCURRENCY": None,
        "ENDPOINT_ENV": None,
        "REMOTE_SCHEME": None,
        "STATUS_REASONS": None,
        "RETRIED_STATUSES": None,
        "SLOW_DOWN": None,
        "UNREACHABLE": None,
        "RemoteRef": None,
        "RemoteStats": None,
        "ReadOnlyTransport": None,
        "parse_remote_path": ("text",),
        "dataset_prefix": ("accession", "prefix"),
        "source_error": ("reason", "location", "problem"),
        "stats": (),
        "reset_stats": (),
        "record_request": (),
        "record_bytes": ("n",),
        "take_slot": (),
        "request_slot": (),
        "default_transport": (),
        "read_only": ("transport",),
    },
    "surfcensus.remote.transport": {
        "ALLOWED_METHODS": None,
        "REFUSED_HEADERS": None,
        "REFUSED_QUERY": None,
        "ReadOnlyTransport": None,
        "refusal": ("request",),
        "read_only": ("transport",),
        "default_transport": (),
    },
    "surfcensus.remote.s3": {
        "S3_NAMESPACE": None,
        "RETRY_BASE_S": None,
        "RETRY_CAP_S": None,
        "CHUNK_BYTES": None,
        "S3Object": None,
        "S3Head": None,
        "S3Client": None,
        "backoff_delay": ("attempt", "retry_after", "rng"),
    },
    "surfcensus.remote.cache": {
        "CACHE_DIR": None,
        "LEASES": None,
        "RETAINED": None,
        "LOCK_FILE": None,
        "DISK_MARGIN_BYTES": None,
        "Cache": None,
        "CacheBudgetError": None,
    },
    "surfcensus.sources.openneuro_derivatives": {
        "DATASET_PREFIX": None,
        "FS_ROOTS": None,
        "OpenNeuroDerivatives": None,
        "accession_of": ("common_prefix", "prefix"),
        "dataset_root": ("bucket", "accession", "prefix"),
        # the EPI interface of a dataset of the bucket (decision D50)
        "FUNC_TREE": None,
        "FUNC_TREE_RAW": None,
        "FUNC_TREE_DERIV": None,
        "FUNC_TREE_FS": None,
        "FUNC_TREE_OBJECTS": None,
        "FuncListing": None,
        "FuncTree": None,
        "tree_objects": ("tree_root",),
    },
    "surfcensus.func.discover": {
        "remote_identity": ("record", "connector"),
        "func_source_of_tree": ("record", "tree"),
    },
    "surfcensus.sources.patterns": {
        "roles_from_keys": ("names",),
        "key_variant": ("role", "location"),
        "is_subject_candidate_keys": ("names", "label"),
        "parse_pipeline_version": ("raw",),
    },
    "surfcensus.sources.base": {"RemoteSource": None},
    "surfcensus.config": {
        "new_local_config": ("root", "subjects", "overrides"),
        "remote_dataset_config": ("config", "accession", "subjects", "bucket"),
    },
    "surfcensus.privacy": {
        "scrub": ("text", "pseudonyms", "roots", "mask_subjects"),
        "scrub_with": ("catalog", "text"),
    },
    "surfcensus.run": {
        "remote_index": ("config", "accession"),
        "from_config": (
            "config",
            "run_config",
            "run_dir",
            "root",
            "identity",
            "transport",
            "secret",
        ),
    },
}

METHODS: dict[type, dict[str, tuple[str, ...]]] = {
    S3Client: {
        "__init__": ("bucket", "config", "addressing", "transport", "sleep"),
        "list": ("prefix", "delimiter"),
        "head": ("key", "location"),
        "get": ("key", "byte_range", "if_match", "location"),
        "download": ("key", "dest", "expect", "cancel", "location"),
        "close": (),
    },
    Cache: {
        "__init__": ("root", "max_bytes", "secret", "retain_roles"),
        "name": ("bucket", "key", "etag"),
        "path_for": ("bucket", "key", "etag", "retained"),
        "lease": ("client", "objects", "retain"),
        "retained": ("client", "obj"),
        "leased_bytes": (),
        "forget": ("bucket", "key", "etag"),
        "prune": (),
    },
    OpenNeuroDerivatives: {
        "__init__": (
            "client",
            "cache",
            "prefix",
            "accession",
            "subjects",
            "pseudonym",
            "raw_client",
        ),
        "bind": ("cache", "pseudonym"),
        "config": (),
        "datasets": ("only",),
        "subjects": ("dataset",),
        "read_bytes": ("subject", "role", "byte_range"),
        "lease": ("subject", "roles"),
        "template_dir": ("dataset",),
        "func_listing": ("dataset",),
        "func_size": ("dataset",),
        "func_tree": ("dataset",),
        "func_stems": ("dataset",),
        "stop": (),
        "close": (),
    },
}

RECORDS = {
    S3Object: ("key", "size", "etag", "last_modified"),
    S3Head: ("size", "etag", "last_modified", "version_id"),
    remote.RemoteRef: ("accession", "source"),
    remote.RemoteStats: ("requests", "bytes", "in_flight", "high_water"),
}


@pytest.mark.parametrize("module", sorted(INTERFACE))
def test_the_shared_names_exist_with_their_parameters(module: str) -> None:
    loaded = import_module(module)
    for name, parameters in INTERFACE[module].items():
        assert hasattr(loaded, name), f"{module}.{name}"
        if parameters is not None:
            found = tuple(inspect.signature(getattr(loaded, name)).parameters)
            assert found == parameters, f"{module}.{name}"


def test_the_methods_keep_their_parameters() -> None:
    for cls, methods in METHODS.items():
        for name, parameters in methods.items():
            found = tuple(inspect.signature(getattr(cls, name)).parameters)[1:]
            assert found == parameters, f"{cls.__name__}.{name}"


def test_the_records_keep_their_fields() -> None:
    for cls, fields in RECORDS.items():
        assert tuple(f.name for f in dataclasses.fields(cls)) == fields, cls.__name__
        assert cls.__dataclass_params__.frozen, cls.__name__  # pyrefly: ignore[missing-attribute]


def test_the_constants() -> None:
    assert remote.DERIVATIVES_BUCKET == "openneuro-derivatives"
    assert remote.FMRIPREP_PREFIX == "fmriprep/"
    assert remote.MAX_CONCURRENCY == 8
    assert remote.ENDPOINT_ENV == "SURFCENSUS_S3_ENDPOINT"
    assert remote.STATUS_REASONS == {
        403: "unreadable",
        404: "changed",
        412: "changed",
        416: "invalid",
    }
    assert remote.RETRIED_STATUSES == {429, 500, 502, 503, 504}
    assert (remote.SLOW_DOWN, remote.UNREACHABLE) == ("SlowDown", "unreachable")
    assert CACHE_DIR == "cache"


# --- no stub is left ---------------------------------------------------------------------------


def test_no_stub_of_the_remote_stage_is_left() -> None:
    """R0 laid the client, the cache and the source down as stubs that named their work
    package; RA, RB and RE wrote them."""
    import surfcensus.remote.cache as cache_module
    import surfcensus.remote.s3 as s3_module
    import surfcensus.sources.openneuro_derivatives as source_module

    for module in (s3_module, cache_module, source_module):
        text = inspect.getsource(module)
        assert "NotImplementedError" not in text, module


# --- openneuro:dsXXXXXX ----------------------------------------------------------------------


def test_a_remote_path_is_openneuro_and_an_accession_of_ascii_digits() -> None:
    ref = remote.parse_remote_path("openneuro:ds000102")
    assert ref == remote.RemoteRef("ds000102")
    assert (ref.text, ref.bucket, ref.prefix, ref.run_dir_name) == (
        "openneuro:ds000102",
        "openneuro-derivatives",
        "fmriprep/ds000102-fmriprep/",
        "ds000102-surfcensus-run",
    )
    assert remote.parse_remote_path(Path("openneuro:ds003020")) == remote.RemoteRef("ds003020")
    for path in ("/data/ds000102", "ds000102", "./openneuro:ds000102", "OpenNeuro:ds000102"):
        assert remote.parse_remote_path(path) is None
    for text in (
        "openneuro:",
        "openneuro:ds00010",
        "openneuro:ds0001022",
        "openneuro:ds000102/",
        "openneuro:DS000102",
        "openneuro:ds" + "".join(chr(0x660 + int(d)) for d in "000102"),  # Arabic-Indic
    ):
        with pytest.raises(SourceError) as caught:
            remote.parse_remote_path(text)
        assert caught.value.reason == "invalid"
    with pytest.raises(ValueError):
        remote.RemoteRef("ds12")


def test_an_error_text_with_a_url_is_scrubbed() -> None:
    exc = httpx.ConnectError(
        "GET https://openneuro-derivatives.s3.amazonaws.com/fmriprep/ds999999-fmriprep/"
        "sourcedata/freesurfer/sub-01/surf/lh.white failed"
    )
    error = remote.source_error("unreachable", "surf/lh.white", exc)
    assert error.reason == "unreachable"
    assert error.detail == "surf/lh.white: GET lh.white failed"
    assert ACCESSION not in str(error) and "amazonaws" not in str(error)


# --- the transport: read-only, anonymous, counted, at most 8 at once --------------------------


class Chunks(httpx.SyncByteStream):
    def __iter__(self) -> Iterator[bytes]:
        yield b"abc"
        yield b"def"


class Streaming(httpx.BaseTransport):
    """Answers with a body that is read from the transport, as a connection's is."""

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=Chunks())


class Recording(httpx.BaseTransport):
    def __init__(self) -> None:
        self.seen: list[httpx.Request] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(request)
        return httpx.Response(200, content=b"abcdef")


@pytest.mark.parametrize(
    ("method", "headers", "url", "why"),
    [
        ("POST", {}, "https://b.s3.amazonaws.com/k", "the method POST"),
        ("PUT", {}, "https://b.s3.amazonaws.com/k", "the method PUT"),
        ("DELETE", {}, "https://b.s3.amazonaws.com/k", "the method DELETE"),
        ("GET", {"Authorization": "AWS x:y"}, "https://b.s3.amazonaws.com/k", "authorization"),
        ("HEAD", {"X-Amz-Security-Token": "t"}, "https://b.s3.amazonaws.com/k", "security-token"),
        ("GET", {}, "https://b.s3.amazonaws.com/k?X-Amz-Signature=s", "signed URL"),
        ("GET", {}, "https://b.s3.amazonaws.com/k?AWSAccessKeyId=a&Signature=s", "signed URL"),
        ("GET", {}, "https://user:pw@b.s3.amazonaws.com/k", "authorization"),  # httpx: Basic
    ],
)
def test_the_transport_refuses_before_any_io(
    method: str, headers: dict[str, str], url: str, why: str
) -> None:
    inner = Recording()
    before = remote.stats()
    with httpx.Client(transport=read_only(inner), trust_env=False) as http:
        with pytest.raises(PermissionError, match=why):
            http.request(method, url, headers=headers)
    assert inner.seen == []
    after = remote.stats()
    assert (after.requests, after.in_flight) == (before.requests, before.in_flight)


def test_the_transport_counts_requests_and_bytes_and_gives_its_slot_back() -> None:
    inner = Recording()
    guarded = read_only(inner)
    assert read_only(guarded) is guarded and ReadOnlyTransport(guarded).inner is inner
    assert refusal(httpx.Request("GET", "https://b.s3.amazonaws.com/k?prefix=x")) is None
    remote.reset_stats()
    with httpx.Client(transport=guarded, trust_env=False) as http:
        assert http.get("https://b.s3.amazonaws.com/k").content == b"abcdef"
        assert http.head("https://b.s3.amazonaws.com/k").status_code == 200
    with httpx.Client(transport=read_only(Streaming()), trust_env=False) as http:
        with http.stream("GET", "https://b.s3.amazonaws.com/k") as streamed:
            assert remote.stats().in_flight == 1
            next(streamed.iter_raw())
        assert remote.stats().in_flight == 0
        assert http.get("https://b.s3.amazonaws.com/k").content == b"abcdef"
        assert remote.stats().in_flight == 0
    counted = remote.stats()
    assert counted.requests == 4 and counted.bytes >= 15 and counted.in_flight == 0


class Unread(httpx.BaseTransport):
    """Answers with a body that is in memory and was handed over as a stream, so the
    response has no content until it is read."""

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=httpx.ByteStream(b"abcdef"))


def test_a_body_in_memory_that_was_not_read_yet_is_read_and_its_slot_given_back() -> None:
    remote.reset_stats()
    with httpx.Client(transport=read_only(Unread()), trust_env=False) as http:
        for _ in range(remote.MAX_CONCURRENCY + 1):  # a leaked slot each would block here
            assert http.get("https://b.s3.amazonaws.com/k").content == b"abcdef"
    counted = remote.stats()
    assert (counted.requests, counted.bytes, counted.in_flight) == (9, 54, 0)


def test_no_more_than_eight_requests_are_open_in_the_process(tmp_path: Path) -> None:
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "k").write_bytes(b"x")
    fake = FakeS3({"b": tmp_path / "b"}, latency_s=0.25)
    remote.reset_stats()
    clients = [
        httpx.Client(transport=read_only(fake.transport()), trust_env=False) for _ in range(3)
    ]
    threads = [
        threading.Thread(target=clients[i % 3].get, args=("https://b.s3.amazonaws.com/k",))
        for i in range(14)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for client in clients:
        client.close()
    assert len(fake.log) == 14
    # the slot is taken before the inner transport is asked, so the server never sees more
    assert 2 <= fake.high_water <= remote.stats().high_water <= remote.MAX_CONCURRENCY


def test_the_endpoint_of_the_tests_is_loopback_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "k").write_bytes(b"over the socket")
    fake = FakeS3({BUCKET: tmp_path / "b"})
    for refused in ("http://10.0.0.1:9", "https://127.0.0.1:9", "http://example.org:9", "ftp://x"):
        monkeypatch.setenv(remote.ENDPOINT_ENV, refused)
        with pytest.raises(SourceError, match="loopback"):
            default_transport()
    with fake.serve() as endpoint:
        monkeypatch.setenv(remote.ENDPOINT_ENV, endpoint)
        transport = default_transport()
        assert isinstance(transport, ReadOnlyTransport)
        with httpx.Client(transport=transport, trust_env=False) as http:
            virtual = http.get(f"https://{BUCKET}.s3.amazonaws.com/k")
            path = http.get(f"https://s3.amazonaws.com/{BUCKET}/k")
            with pytest.raises(PermissionError):
                http.get("https://example.org/k")
            with pytest.raises(PermissionError):
                http.post(f"https://{BUCKET}.s3.amazonaws.com/k")
    assert virtual.content == path.content == b"over the socket"
    assert fake.methods == {"GET"}
    monkeypatch.delenv(remote.ENDPOINT_ENV)
    assert isinstance(default_transport().inner, httpx.HTTPTransport)  # no request made


def test_the_endpoint_of_the_tests_keeps_the_timeouts_of_a_request() -> None:
    """The request that goes on to the test server carries the extensions of the one it
    replaces, which is where ``httpx`` keeps the timeouts: without them a stalled test
    server would hold a request for good."""
    from surfcensus.remote import transport as transport_module

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=b"x")

    moved = transport_module._Endpoint(
        httpx.URL("http://127.0.0.1:9"), httpx.MockTransport(handler)
    )
    with httpx.Client(transport=moved, trust_env=False, timeout=httpx.Timeout(7, read=3)) as http:
        http.get(f"https://{BUCKET}.s3.amazonaws.com/k?list-type=2")
    (request,) = seen
    assert request.url == f"http://127.0.0.1:9/{BUCKET}/k?list-type=2"
    assert request.extensions["timeout"] == {"connect": 7, "read": 3, "write": 7, "pool": 7}


@pytest.mark.parametrize(
    ("no_proxy", "exempt"),
    [
        ("", False),
        ("*", True),
        ("amazonaws.com", True),
        (".amazonaws.com", True),
        ("*.s3.amazonaws.com", True),
        ("s3.amazonaws.com", True),
        ("localhost, 127.0.0.1 ,amazonaws.com", True),
        ("ws.com", False),  # no dot boundary: another domain
        ("aws.com", False),
        ("example.org,localhost", False),
        # one bucket's own host exempts nothing: the transport serves every bucket
        (f"{BUCKET}.s3.amazonaws.com", False),
    ],
)
def test_no_proxy_exempts_the_bucket_hosts_at_a_dot(
    no_proxy: str, exempt: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from surfcensus.remote import transport as transport_module

    for name in ("HTTPS_PROXY", "https_proxy", "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    assert transport_module._environment_proxy() is None
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:3128")
    monkeypatch.setenv("NO_PROXY", no_proxy)
    assert transport_module._environment_proxy() == (
        None if exempt else "http://proxy.invalid:3128"
    )


def test_the_client_puts_the_guard_under_every_request(monkeypatch: pytest.MonkeyPatch) -> None:
    # a proxy of the environment is mounted beside the transport of a client that trusts the
    # environment, and would pass around the guard
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    inner = Recording()
    client = S3Client(BUCKET, config=RemoteConfig(max_concurrency=4), transport=inner)
    assert isinstance(client.transport, ReadOnlyTransport) and client.transport.inner is inner
    assert client.http._mounts == {}  # pyrefly: ignore[missing-attribute]
    assert client.http.get(client.endpoint + "/k").content == b"abcdef"
    assert inner.seen[0].url.host == "openneuro-derivatives.s3.amazonaws.com"
    with pytest.raises(PermissionError):
        client.http.delete(client.endpoint + "/k")
    assert (client.addressing, S3Client("openneuro.org", config=RemoteConfig()).addressing) == (
        "virtual",
        "path",
    )
    assert S3Client("openneuro.org", config=RemoteConfig(), transport=inner).endpoint == (
        "https://s3.amazonaws.com/openneuro.org"
    )
    client.close()
    with pytest.raises(ValueError):
        RemoteConfig(max_concurrency=remote.MAX_CONCURRENCY + 1)


# --- the cache: names keyed with the run secret -----------------------------------------------


def test_the_names_of_the_cache_are_keyed_and_keep_only_the_file_name(tmp_path: Path) -> None:
    key = f"fmriprep/{ACCESSION}-fmriprep/sourcedata/freesurfer/sub-CANARY7/surf/lh.pial.T1"
    etag = "0123456789abcdef0123456789abcdef"
    cache = Cache(tmp_path / "cache", max_bytes=100, secret=bytes(range(32)))
    other = Cache(tmp_path / "cache", max_bytes=100, secret=bytes(range(1, 33)))
    retained = cache.path_for(BUCKET, key, etag, retained=True)
    leased = cache.path_for(BUCKET, key, etag, retained=False)
    h = privacy.cache_name(bytes(range(32)), BUCKET, key, etag)
    assert retained == tmp_path / "cache" / "retained" / h[:2] / h / "lh.pial.T1"
    assert leased.parent.name == h and leased.name == "lh.pial.T1"
    assert leased.parent.parent == cache.area and cache.area.parent.name == "leases"
    assert other.path_for(BUCKET, key, etag, retained=True) != retained
    for path in (retained, leased):
        relative = path.relative_to(tmp_path).as_posix()
        for secret_text in (ACCESSION, "sub-CANARY7", etag, "fmriprep", "freesurfer"):
            assert secret_text not in relative
    assert "secret" not in repr(cache) and bytes(range(32)).hex() not in repr(cache)
    with pytest.raises(ValueError):
        Cache(tmp_path, max_bytes=1, secret=b"short")
    error = CacheBudgetError("a lease of 5 bytes would take this process over 4")
    assert isinstance(error, ConfigError) and error.reason == "invalid"
    assert pickle.loads(pickle.dumps(error)).detail == error.detail


# --- the stored form of a remote dataset (decision D47, item 1) --------------------------------


def test_a_remote_source_config_names_no_dataset_but_by_pseudonym() -> None:
    SourceConfig(kind="openneuro-derivatives", bucket=BUCKET, prefix="fmriprep/")
    SourceConfig(
        kind="openneuro-derivatives",
        bucket=BUCKET,
        prefix="fmriprep/",
        dataset="D-0123abcd",
        subjects=("sub-0*",),
    )
    with pytest.raises(ValueError, match="names no dataset"):
        SourceConfig(
            kind="openneuro-derivatives", bucket=BUCKET, prefix="fmriprep/ds000102-fmriprep/"
        )
    with pytest.raises(ValueError, match="has no subjects"):
        SourceConfig(kind="openneuro-derivatives", bucket=BUCKET, subjects=("a",))
    with pytest.raises(ValueError, match="has no field_strength, root"):
        SourceConfig(
            kind="openneuro-derivatives", bucket=BUCKET, dataset="D-1", root="/x", field_strength=3
        )


def test_a_local_run_stores_one_remote_dataset_by_its_pseudonym(tmp_path: Path) -> None:
    config = new_local_config(None)
    assert config.sources == ()
    stored = remote_dataset_config(config, ACCESSION, subjects=("sub-0*", "sub-0*"))
    pseudonym = privacy.dataset_pseudonym(run_key(config), "openneuro-derivatives", ACCESSION)
    assert stored == SourceConfig(
        kind="openneuro-derivatives",
        bucket=BUCKET,
        prefix="fmriprep/",
        dataset=pseudonym,
        subjects=("sub-0*",),
    )
    with_source = with_changes(config, sources=(stored,))
    assert remote_index(with_source, ACCESSION) == 0
    assert remote_index(with_source, OTHER) is None
    assert remote_index(new_census_config(DATASET_KEY), ACCESSION) is None
    text = yaml.safe_dump(with_source.model_dump(mode="json"))
    assert ACCESSION not in text and "999999" not in text
    for bad in (
        dict(sources=(SourceConfig(kind="openneuro-derivatives", bucket=BUCKET),)),
        dict(sources=(stored, SourceConfig(kind="local", root="/data"))),
        dict(sources=(SourceConfig(kind="openneuro-embedded", bucket=BUCKET, dataset="D-1"),)),
    ):
        with pytest.raises(ConfigError, match="a local run"):
            with_changes(config, **bad)
    with pytest.raises(ConfigError, match="census run"):
        remote_dataset_config(new_census_config(DATASET_KEY), ACCESSION)
    with pytest.raises(ConfigError, match="stored by bucket and prefix"):
        with_changes(new_census_config(DATASET_KEY), sources=(stored,))
    with pytest.raises(ConfigError, match="subject globs"):
        new_local_config(None, subjects=("a",))


def test_the_pseudonym_of_a_remote_dataset_is_pinned_on_a_fixed_salt() -> None:
    assert privacy.dataset_pseudonym(SALT, "openneuro-derivatives", ACCESSION) == "D-8d961300"


def test_a_source_of_one_dataset_is_stored_only_once_bound(tmp_path: Path) -> None:
    client = S3Client(BUCKET, config=RemoteConfig(), transport=FakeS3({}).transport())
    census = OpenNeuroDerivatives(client=client)
    assert census.config() == SourceConfig(
        kind="openneuro-derivatives", bucket=BUCKET, prefix="fmriprep/"
    )
    one = OpenNeuroDerivatives(client=client, accession=ACCESSION, subjects="sub-01")
    assert isinstance(one, RemoteSource) and one.globs == ("sub-01",)
    with pytest.raises(ConfigError, match="pseudonym"):
        one.config()
    cache = Cache(tmp_path, max_bytes=1, secret=bytes(32))
    one.bind(cache=cache, pseudonym="D-0123abcd")
    assert one.cache is cache and one.config().dataset == "D-0123abcd"
    one.bind(cache=cache, pseudonym="D-0123abcd")
    with pytest.raises(ConfigError, match="another dataset"):
        one.bind(cache=cache, pseudonym="D-ffffffff")
    with pytest.raises(ValueError):
        census.bind(cache=cache, pseudonym="D-0123abcd")
    for bad in (
        dict(prefix="fmriprep"),
        dict(prefix="fmriprep/ds000102-fmriprep/"),
        dict(accession="ds12"),
        dict(subjects=("a",)),
        dict(pseudonym="D-1"),
    ):
        with pytest.raises(ValueError):
            OpenNeuroDerivatives(client=client, **bad)  # pyrefly: ignore[bad-argument-type]
    assert accession_of("fmriprep/ds000102-fmriprep/") == "ds000102"
    assert accession_of("fmriprep/ds000102-mriqc/") is None
    assert accession_of("xcpd/ds000102-fmriprep/") is None
    assert dataset_root(BUCKET, ACCESSION) == f"s3://{BUCKET}/fmriprep/{ACCESSION}-fmriprep/"


# --- a run on a remote dataset: created, opened, rebuilt ---------------------------------------


def record_of(accession: str) -> DatasetRecord:
    return DatasetRecord(
        source="openneuro-derivatives",
        identity=accession,
        accession=accession,
        root=dataset_root(BUCKET, accession),
        fs_root="sourcedata/freesurfer",
        layout="fmriprep",
        pipeline="fmriprep",
        pipeline_version="23.0.2",
    )


class Listed(OpenNeuroDerivatives):
    """The source of RE with its listing given, as RE will list it, and nothing else."""

    listings = 0

    def datasets(self, only: Collection[str] | None = None) -> Iterator[DatasetRecord]:
        type(self).listings += 1
        assert self.accession is not None
        yield record_of(self.accession)


@pytest.fixture
def fake() -> FakeS3:
    return FakeS3({BUCKET: Path("/nonexistent")})


def one_dataset(fake: FakeS3, accession: str = ACCESSION, globs: tuple[str, ...] = ()) -> Listed:
    client = S3Client(BUCKET, config=RemoteConfig(max_concurrency=4), transport=fake.transport())
    return Listed(client=client, accession=accession, subjects=globs)


def files_text(run_dir: Path) -> str:
    """Every file of the run directory but the catalog, as text."""
    return "\n".join(
        path.read_text(errors="replace")
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and not path.name.startswith("catalog.sqlite")
    )


def test_a_run_on_a_remote_dataset_keeps_its_accession_in_the_catalog_only(
    tmp_path: Path, fake: FakeS3
) -> None:
    run_dir = tmp_path / f"{ACCESSION}-surfcensus-run"
    source = one_dataset(fake, globs=("sub-0*",))
    with Run(run_dir, source, transport=fake.transport()) as run:
        stored = run.config.sources[0]
        assert stored.kind == "openneuro-derivatives" and stored.subjects == ("sub-0*",)
        assert stored.dataset == privacy.dataset_pseudonym(
            run_key(run.config), "openneuro-derivatives", ACCESSION
        )
        assert source.pseudonym == stored.dataset and source.cache is not None
        assert source.cache.root == run_dir / CACHE_DIR
        rows = run._catalog.query("SELECT pseudonym, accession, identity, root FROM datasets")
        assert [tuple(row) for row in rows] == [
            (stored.dataset, ACCESSION, ACCESSION, dataset_root(BUCKET, ACCESSION))
        ]
        # the accession becomes the pseudonym in a recorded detail; labels are kept
        assert (
            run.context().scrub(f"{ACCESSION}: sub-01 failed") == f"{stored.dataset}: sub-01 failed"
        )
    text = files_text(run_dir)
    assert ACCESSION not in text and "fmriprep/ds" not in text
    assert fake.log == []  # making the run sent nothing (the listing is RE's, given here)

    # opened again without a source: rebuilt from the pseudonym and the catalog
    with Run(run_dir, transport=fake.transport()) as run:
        rebuilt = run._source(0)
        assert isinstance(rebuilt, OpenNeuroDerivatives) and not isinstance(rebuilt, Listed)
        assert (rebuilt.accession, rebuilt.globs, rebuilt.pseudonym) == (
            ACCESSION,
            ("sub-0*",),
            stored.dataset,
        )
        assert rebuilt.config() == stored and rebuilt.cache is not None
        assert isinstance(rebuilt.client.transport, ReadOnlyTransport)
    assert fake.log == []

    # the same dataset again, with other globs: the run is found and takes them
    listings = Listed.listings
    with Run(run_dir, one_dataset(fake, globs=("sub-01",)), transport=fake.transport()) as run:
        assert run.config.sources[0].subjects == ("sub-01",)
    assert Listed.listings == listings  # the row is there: nothing listed again
    assert "sub-01" in (run_dir / "config.yaml").read_text()


def test_another_dataset_or_a_path_is_another_root(tmp_path: Path, fake: FakeS3) -> None:
    run_dir = tmp_path / "run"
    Run(run_dir, one_dataset(fake), transport=fake.transport()).close()
    with pytest.raises(ConfigError) as caught:
        Run(run_dir, one_dataset(fake, OTHER), transport=fake.transport())
    assert caught.value.reason == "other root"
    assert OTHER not in str(caught.value) and ACCESSION not in str(caught.value)
    assert "--run-dir" in str(caught.value)
    subjects = tmp_path / "subjects"
    write_light_subject(subjects / "sub-01")
    from surfcensus.sources.local import LocalSource

    with pytest.raises(ConfigError) as caught:
        Run(run_dir, LocalSource(subjects))
    assert caught.value.reason == "other root" and ACCESSION not in str(caught.value)
    with pytest.raises(ConfigError, match="does not move"):
        Run(run_dir, one_dataset(fake), relocate=True, transport=fake.transport())


def test_a_run_whose_dataset_row_is_missing_gets_it_from_the_next_call(
    tmp_path: Path, fake: FakeS3
) -> None:
    run_dir = tmp_path / "run"
    Run(run_dir, one_dataset(fake), transport=fake.transport()).close()
    with Catalog(run_dir / "catalog.sqlite") as catalog, catalog.write() as conn:
        conn.execute("DELETE FROM datasets")  # as a creation that stopped before the row
    with Run(run_dir, transport=fake.transport()) as run:
        with pytest.raises(ConfigError, match="openneuro:dsXXXXXX"):
            run._source(0)
    with Run(run_dir, one_dataset(fake), transport=fake.transport()) as run:
        assert run._source(0).config() == run.config.sources[0]
        assert run._catalog.query("SELECT accession FROM datasets")[0]["accession"] == ACCESSION


def test_from_config_checks_the_accession_against_the_pseudonym(tmp_path: Path) -> None:
    config = new_local_config(None)
    stored = remote_dataset_config(config, ACCESSION)
    config = with_changes(config, sources=(stored,))
    secret = bytes(32)
    built = from_config(stored, config, tmp_path, identity=ACCESSION, secret=secret)
    assert isinstance(built, OpenNeuroDerivatives) and built.config() == stored
    with pytest.raises(ConfigError, match="needs the run secret"):
        from_config(stored, config, tmp_path, identity=ACCESSION)
    with pytest.raises(ConfigError, match=r"catalog\.sqlite only"):
        from_config(stored, config, tmp_path, secret=secret)
    with pytest.raises(ConfigError) as caught:
        from_config(stored, config, tmp_path, identity=OTHER, secret=secret)
    assert caught.value.reason == "key"
    census = SourceConfig(kind="openneuro-derivatives", bucket=BUCKET, prefix="fmriprep/")
    every = from_config(census, new_census_config(DATASET_KEY), tmp_path, secret=secret)
    assert isinstance(every, OpenNeuroDerivatives) and every.accession is None
    with pytest.raises(NotImplementedError, match="O9"):
        from_config(
            SourceConfig(kind="openneuro-embedded", bucket="openneuro.org"),
            new_census_config(DATASET_KEY),
            tmp_path,
            secret=secret,
        )
    with pytest.raises(ConfigError, match="created from its source"):
        Run(tmp_path / "new", config=config)


def test_a_census_run_stores_the_bucket_and_prefix_only(tmp_path: Path, fake: FakeS3) -> None:
    client = S3Client(BUCKET, config=RemoteConfig(max_concurrency=4), transport=fake.transport())
    every = OpenNeuroDerivatives(client=client)
    run_dir = tmp_path / "census"
    with Run(run_dir, every, key=DATASET_KEY, transport=fake.transport()) as run:
        assert run.config.sources == (
            SourceConfig(kind="openneuro-derivatives", bucket=BUCKET, prefix="fmriprep/"),
        )
        assert every.cache is not None and every.pseudonym is None
    with Run(run_dir, transport=fake.transport()) as run:
        rebuilt = run._source(0)
        assert isinstance(rebuilt, OpenNeuroDerivatives) and rebuilt.accession is None
    with pytest.raises(ConfigError, match="census run lists every dataset"):
        Run(tmp_path / "census2", one_dataset(fake), key=DATASET_KEY)
    assert fake.log == []


def test_scrub_with_names_the_pseudonym_of_a_remote_dataset_in_a_local_run(tmp_path: Path) -> None:
    with Catalog.create(tmp_path / "catalog.sqlite", mode="local", key=SALT) as catalog:
        catalog.upsert_dataset(record_of(ACCESSION), "D-8d961300", None)
        text = (
            f"{ACCESSION}: s3://{BUCKET}/fmriprep/{ACCESSION}-fmriprep/sub-01/x failed for sub-01"
        )
        assert privacy.scrub_with(catalog, text) == "D-8d961300: x failed for sub-01"
    assert privacy.scrub("ds000102 sub-01", pseudonyms={}, mask_subjects=False) == "D-? sub-01"
    assert privacy.scrub("ds000102 sub-01", pseudonyms={}) == "D-? sub-?"


# --- one rule for the roles of a directory and of a listing ----------------------------------


def test_a_listing_and_a_directory_give_the_same_roles(tmp_path: Path) -> None:
    subject = tmp_path / "sub-01"
    write_light_subject(subject)
    on_disk = resolve_subject_files(subject)
    names = [path.relative_to(subject).as_posix() for path in subject.rglob("*") if path.is_file()]
    assert roles_from_keys(names) == {
        role: path.relative_to(subject).as_posix() for role, path in on_disk.items()
    }
    fs7 = roles_from_keys(["surf/lh.pial.T1", "surf/lh.pial.T2", "mri/T1.mgz"])
    assert fs7 == {FileRole.T1: "mri/T1.mgz", FileRole.LH_PIAL: "surf/lh.pial.T2"}
    assert key_variant(FileRole.LH_PIAL, "surf/lh.pial.T1") == "lh.pial.T1"
    assert key_variant(FileRole.T1, "mri/T1.mgz") is None
    assert is_subject_candidate_keys(["mri/T1.mgz", "surf/lh.white"], "sub-01")
    assert not is_subject_candidate_keys(["mri/T1.mgz", "surf/lh.white"], "fsaverage")
    assert not is_subject_candidate_keys(["mri/T1.mgz", "label/lh.aparc.annot"], "sub-01")
    assert not is_subject_candidate_keys(["mri", "surf"], "sub-01")


def test_the_pipeline_version_is_read_by_one_rule() -> None:
    description = b'{"GeneratedBy": [{"Name": "fMRIPrep", "Version": "21.0.2"}]}'
    assert parse_pipeline_version(description) == "21.0.2"
    assert parse_pipeline_version(b"\xff\xfe not json") is None
    assert parse_pipeline_version(b'{"PipelineDescription": {"Version": "1.5.0"}}') == "1.5.0"


def test_runconfig_accepts_what_remote_dataset_config_makes() -> None:
    config = new_local_config(None)
    stored = remote_dataset_config(config, ACCESSION)
    assert isinstance(
        RunConfig.model_validate(with_changes(config, sources=(stored,)).model_dump()), RunConfig
    )
