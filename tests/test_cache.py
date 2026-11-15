"""The download cache of a run (work package RB; docs/dev/remote_plan.md, section 5).

Leases are tested against a stand-in client: an ``S3Client`` over the fake bucket whose
``download`` writes bytes it was given, through a temporary name beside ``dest``, and can be
told to fail, to block until released and to watch its cancel event. No test touches the
network. The tests with processes synchronise through files and use timeouts only to guard
against a hang.
"""

from __future__ import annotations

import _thread
import logging
import os
import re
import signal
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fakes3 import FakeS3  # pyrefly: ignore[missing-import]

from surfcensus import privacy
from surfcensus.config import RemoteConfig
from surfcensus.remote import cache as cache_module
from surfcensus.remote.cache import LEASES, LOCK_FILE, RETAINED, Cache, CacheBudgetError
from surfcensus.remote.s3 import S3Client, S3Head, S3Object
from surfcensus.sources.base import FileRole, SourceError

SECRET = bytes(range(32))
BUCKET = "fake-bucket"
ACCESSION = "ds999999"
PREFIX = f"fmriprep/{ACCESSION}-fmriprep/sourcedata/freesurfer/sub-01/"
HANG_S = 60.0
"""Only a guard against a hang: nothing waits this long when the code works."""

needs_locks = pytest.mark.skipif(cache_module.fcntl is None, reason="needs fcntl.flock")


def _obj(name: str, content: bytes) -> S3Object:
    import hashlib

    return S3Object(PREFIX + name, len(content), hashlib.md5(content).hexdigest(), "x")


class StandIn(S3Client):
    """``S3Client`` whose ``download`` writes what it was given and records what it did."""

    def __init__(self, contents: dict[str, bytes], *, max_concurrency: int = 4) -> None:
        super().__init__(
            BUCKET,
            config=RemoteConfig(max_concurrency=max_concurrency),
            transport=FakeS3({}).transport(),
        )
        self.contents = contents
        self.calls: list[str] = []
        self.cancelled: list[str] = []
        self.fail: dict[str, BaseException] = {}
        self.gate: threading.Event | None = None
        self.started = threading.Semaphore(0)
        self.fail_after = 0
        """A failing download waits until this many downloads have started."""
        self._count = threading.Condition()
        self.n_started = 0
        self.open = 0
        self.high = 0

    def download(
        self,
        key: str,
        dest: Path,
        *,
        expect: S3Object,
        cancel: threading.Event | None = None,
    ) -> S3Head:
        with self._count:
            self.calls.append(key)
            self.open += 1
            self.high = max(self.high, self.open)
            self.n_started += 1
            self._count.notify_all()
        tmp = dest.with_name(f".{dest.name}.part")
        try:
            self.started.release()
            if key in self.fail:
                with self._count:
                    assert self._count.wait_for(lambda: self.n_started >= self.fail_after, HANG_S)
                raise self.fail[key]
            content = self.contents[key]
            tmp.write_bytes(content[: len(content) // 2])  # a partial file, as a stream leaves
            gate = self.gate
            deadline = time.monotonic() + HANG_S
            while gate is not None and not gate.wait(0.01):
                if cancel is not None and cancel.is_set():
                    with self._count:
                        self.cancelled.append(key)
                    raise SourceError("unreachable", f"{dest.name}: cancelled")
                assert time.monotonic() < deadline, "the gate was never opened"
            tmp.write_bytes(content)
            os.replace(tmp, dest)
            return S3Head(len(content), expect.etag, expect.last_modified, None)
        finally:
            tmp.unlink(missing_ok=True)
            with self._count:
                self.open -= 1


def _contents(n: int, size: int = 1000) -> dict[str, bytes]:
    return {PREFIX + f"f{i}.bin": bytes([i]) * (size + i) for i in range(n)}


def _objects(contents: dict[str, bytes]) -> dict[FileRole, S3Object]:
    roles = list(FileRole)
    return {
        roles[i]: _obj(key[len(PREFIX) :], data) for i, (key, data) in enumerate(contents.items())
    }


def _files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.name != LOCK_FILE)


@pytest.fixture(autouse=True)
def _ctrl_c_as_in_a_terminal() -> Iterator[None]:
    """SIGINT raises ``KeyboardInterrupt`` in the test process, also when the suite was
    started with SIGINT ignored (``cmd &`` from a shell without job control, ``nohup``, some
    schedulers): ``_thread.interrupt_main`` does nothing for an ignored signal."""
    before = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, before)


@pytest.fixture
def cache(tmp_path: Path) -> Iterator[Cache]:
    made = Cache(tmp_path / "cache", max_bytes=10**9, secret=SECRET)
    yield made
    cache_module._AREAS.pop(os.path.abspath(made.area), None)


def _cache(tmp_path: Path, *, max_bytes: int, retain_roles: tuple[FileRole, ...] = ()) -> Cache:
    made = Cache(tmp_path / "cache", max_bytes=max_bytes, secret=SECRET, retain_roles=retain_roles)
    cache_module._AREAS.pop(os.path.abspath(made.area), None)
    return made


# --- names --------------------------------------------------------------------------------------


def test_no_name_under_the_cache_holds_a_key_path_or_an_accession(cache: Cache) -> None:
    contents = _contents(3)
    client = StandIn(contents)
    objects = _objects(contents)
    with cache.lease(client, objects) as paths:
        leased = _files(cache.root)
        assert {p.parent.parent for p in paths.values()} == {cache.area}
    with cache.lease(client, objects, retain=True):
        pass
    retained = _files(cache.root)
    for path in leased + retained:
        relative = path.relative_to(cache.root).as_posix()
        assert (
            ACCESSION not in relative
            and "sourcedata" not in relative
            and "fmriprep" not in relative
        )
    names = {o.key: privacy.cache_name(SECRET, BUCKET, o.key, o.etag) for o in objects.values()}
    unkeyed = {
        o.key: privacy.cache_name(bytes(32), BUCKET, o.key, o.etag) for o in objects.values()
    }
    for key, h in names.items():
        assert (cache.root / RETAINED / h[:2] / h / key.rsplit("/", 1)[1]).is_file()
        assert unkeyed[key] != h
    pattern = re.compile(
        r"(leases/[^/]+-\d+/[0-9a-f]{32}|retained/[0-9a-f]{2}/[0-9a-f]{32})/f\d\.bin"
    )
    assert all(pattern.fullmatch(p.relative_to(cache.root).as_posix()) for p in leased + retained)


# --- leases -------------------------------------------------------------------------------------


def test_a_lease_downloads_concurrently_within_the_limit_and_removes_on_exit(cache: Cache) -> None:
    contents = _contents(7)
    client = StandIn(contents, max_concurrency=3)
    client.gate = threading.Event()

    fourth: list[bool] = []

    def open_gate() -> None:
        for _ in range(3):  # three downloads are open at once before any may end
            assert client.started.acquire(timeout=HANG_S)
        # a fourth may not start while the three are held; the wait only gives a lease that
        # breaks the limit the time to show it, and the result does not depend on it otherwise
        fourth.append(client.started.acquire(timeout=0.5))
        assert client.gate is not None
        client.gate.set()

    opener = threading.Thread(target=open_gate)
    opener.start()
    with cache.lease(client, _objects(contents)) as paths:
        assert cache.leased_bytes() == sum(len(v) for v in contents.values())
        for role, path in paths.items():
            assert path.read_bytes() == contents[_objects(contents)[role].key]
    opener.join(HANG_S)
    assert fourth == [False] and client.high == 3 and sorted(client.calls) == sorted(contents)
    assert _files(cache.root) == [] and cache.leased_bytes() == 0
    assert [p.name for p in cache.area.iterdir()] == [LOCK_FILE]


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_a_lease_removes_its_files_when_its_body_raises(
    cache: Cache, error: type[BaseException]
) -> None:
    contents = _contents(3)
    client = StandIn(contents)
    with pytest.raises(error), cache.lease(client, _objects(contents)) as paths:
        assert all(p.is_file() for p in paths.values())
        raise error()
    assert _files(cache.root) == [] and cache.leased_bytes() == 0


def test_a_failed_download_cancels_the_others_and_raises_its_own_error(cache: Cache) -> None:
    contents = _contents(4)
    client = StandIn(contents, max_concurrency=4)
    client.gate = threading.Event()  # never opened: the others wait until they are cancelled
    objects = _objects(contents)
    failing = list(objects.values())[2]
    cause = SourceError("changed", "f2.bin: the object changed")
    client.fail[failing.key] = cause
    client.fail_after = 4  # it fails once all four are open
    with pytest.raises(SourceError) as raised, cache.lease(client, objects):
        pytest.fail("the lease did not fail")
    assert raised.value is cause
    assert sorted(client.cancelled) == sorted(o.key for o in objects.values() if o is not failing)
    assert _files(cache.root) == [] and cache.leased_bytes() == 0


def test_ctrl_c_while_downloads_run_cancels_them_and_removes_the_files(cache: Cache) -> None:
    contents = _contents(3)
    client = StandIn(contents)
    client.gate = threading.Event()

    def interrupt() -> None:
        for _ in range(3):
            assert client.started.acquire(timeout=HANG_S)
        _thread.interrupt_main()

    thread = threading.Thread(target=interrupt)
    thread.start()
    with pytest.raises(KeyboardInterrupt), cache.lease(client, _objects(contents)):
        pytest.fail("the lease was not interrupted")
    thread.join(HANG_S)
    assert sorted(client.cancelled) == sorted(contents)
    assert _files(cache.root) == [] and cache.leased_bytes() == 0


def test_a_stop_while_the_pool_starts_a_thread_still_waits_for_its_download(
    cache: Cache, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ThreadPoolExecutor starts a thread and notes it in two steps. A Ctrl-C taken in between
    # (seen once in CI on macOS: the lease ended while one download had not stopped) leaves a
    # running download that the pool does not wait for, and whose future the lease never got.
    # The stand-in pool makes that state on the last submit: the download runs, its thread is
    # taken out of the pool's list, and the submit ends with the interrupt.
    contents = _contents(3)
    last = sorted(contents)[-1]

    class SlowToStop(StandIn):
        """The last download takes a while to end once it is cancelled, as one that is in the
        middle of a chunk does, so a lease that does not wait for it ends before it."""

        def download(self, key, dest, *, expect, cancel=None):  # type: ignore[no-untyped-def]
            try:
                return super().download(key, dest, expect=expect, cancel=cancel)
            finally:
                if key == last and not self.gate.is_set():  # pyrefly: ignore[missing-attribute]
                    time.sleep(0.5)
                    self.stopped.append(key)

    client = SlowToStop(contents)
    client.gate = threading.Event()
    client.stopped = []  # pyrefly: ignore[missing-attribute]

    class Interrupted(cache_module.ThreadPoolExecutor):
        def submit(self, fn, /, *args, **kwargs):  # type: ignore[no-untyped-def]
            before = set(self._threads)
            future = super().submit(fn, *args, **kwargs)
            if len(self._threads) < len(contents):
                return future
            for _ in contents:
                assert client.started.acquire(timeout=HANG_S)
            self._threads -= self._threads - before
            raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(cache_module, "ThreadPoolExecutor", Interrupted)
        with pytest.raises(KeyboardInterrupt), cache.lease(client, _objects(contents)):
            pytest.fail("the lease was not interrupted")
    # every download had stopped when the lease ended, the one of the unknown thread too
    assert client.stopped == [last]  # pyrefly: ignore[missing-attribute]
    assert client.open == 0
    assert sorted(client.cancelled) == sorted(contents)
    assert _files(cache.root) == [] and cache.leased_bytes() == 0
    # and nothing waits for an entry of that lease: the same files can be leased again
    client.gate.set()
    with cache.lease(client, _objects(contents)) as paths:
        assert sorted(p.read_bytes() for p in paths.values()) == sorted(contents.values())


def test_retained_roles_are_kept_and_used_again_without_a_request(tmp_path: Path) -> None:
    contents = _contents(3)
    objects = _objects(contents)
    keep = next(iter(objects))
    cache = _cache(tmp_path, max_bytes=10**9, retain_roles=(keep,))
    client = StandIn(contents)
    with cache.lease(client, objects):
        pass
    kept = cache.path_for(BUCKET, objects[keep].key, objects[keep].etag, retained=True)
    assert _files(cache.root) == [kept] and kept.read_bytes() == contents[objects[keep].key]
    client.calls.clear()
    with cache.lease(client, objects) as paths:
        assert paths[keep] == kept
    assert sorted(client.calls) == sorted(o.key for r, o in objects.items() if r != keep)
    client.calls.clear()
    with cache.lease(client, {keep: objects[keep]}) as paths:
        assert paths == {keep: kept}
    assert client.calls == []


def test_retain_keeps_every_role(cache: Cache) -> None:
    contents = _contents(3)
    objects = _objects(contents)
    client = StandIn(contents)
    with cache.lease(client, objects, retain=True):
        pass
    client.calls.clear()
    with cache.lease(client, objects) as paths:
        assert all(RETAINED in p.parts for p in paths.values())
    assert client.calls == [] and len(_files(cache.root)) == 3


def test_a_retained_download_is_staged_outside_retained(cache: Cache) -> None:
    """While ``Cache.retained`` downloads, its partial file lies in the lease area of the
    process: ``retained/`` holds nothing of the object until the whole file is renamed in."""
    contents = _contents(1, size=50_000)
    (obj,) = _objects(contents).values()
    client = StandIn(contents)
    client.gate = threading.Event()
    done: list[Path] = []
    thread = threading.Thread(target=lambda: done.append(cache.retained(client, obj)))
    thread.start()
    try:
        assert client.started.acquire(timeout=HANG_S)
        deadline = time.monotonic() + HANG_S
        partial: list[Path] = []
        while not partial and time.monotonic() < deadline:
            partial = [p for p in cache.root.rglob("*") if p.is_file() and p.name != LOCK_FILE]
        assert partial
        assert all(RETAINED not in p.relative_to(cache.root).parts for p in partial)
        assert not any(p.is_file() for p in (cache.root / RETAINED).rglob("*"))
    finally:
        client.gate.set()
        thread.join(HANG_S)
    (kept,) = done
    assert RETAINED in kept.relative_to(cache.root).parts
    assert kept.read_bytes() == contents[obj.key]
    assert _files(cache.root) == [kept]


def test_retained_never_holds_a_partial_file(cache: Cache) -> None:
    contents = _contents(2, size=200_000)
    objects = _objects(contents)
    later = _contents(3, size=50_000)
    third = _objects(later)[list(FileRole)[2]]
    client = StandIn(contents)
    client.gate = threading.Event()
    seen: list[tuple[int, int]] = []
    stop = threading.Event()
    sizes = {
        privacy.cache_name(SECRET, BUCKET, o.key, o.etag): o.size
        for o in [*objects.values(), third]
    }

    def watch() -> None:  # whatever it sees in retained/ must be whole
        while not stop.is_set():
            for path in (cache.root / RETAINED).rglob("*.bin"):
                try:
                    seen.append((path.stat().st_size, sizes[path.parent.name]))
                except FileNotFoundError:
                    pass

    def open_gate() -> None:
        for _ in range(2):
            assert client.started.acquire(timeout=HANG_S)
        assert client.gate is not None
        client.gate.set()

    watcher = threading.Thread(target=watch)
    watcher.start()
    try:
        opener = threading.Thread(target=open_gate)
        opener.start()
        with cache.lease(client, objects, retain=True):
            assert not any((cache.root / RETAINED).rglob("*.bin"))
        opener.join(HANG_S)
        cache.retained(StandIn(later), third)
    finally:
        stop.set()
        watcher.join(HANG_S)
    assert len(_files(cache.root / RETAINED)) == 3
    assert all(size == want for size, want in seen)


def test_the_budget_raises_before_any_download(tmp_path: Path) -> None:
    contents = _contents(3, size=100)
    objects = _objects(contents)
    roles = list(objects)
    cache = _cache(tmp_path, max_bytes=250)
    client = StandIn(contents)
    with pytest.raises(CacheBudgetError, match="over the cache's budget"):
        with cache.lease(client, objects):
            pass
    assert client.calls == [] and cache.leased_bytes() == 0
    with cache.lease(client, {roles[0]: objects[roles[0]]}):
        assert cache.leased_bytes() == 100
        with pytest.raises(CacheBudgetError):
            with cache.lease(client, {roles[1]: objects[roles[1]], roles[2]: objects[roles[2]]}):
                pass
        # the same object again is shared and counted once
        with cache.lease(client, {roles[0]: objects[roles[0]], roles[1]: objects[roles[1]]}) as p:
            assert cache.leased_bytes() == 201 and p[roles[0]].is_file()
        assert cache.leased_bytes() == 100 and _files(cache.root) != []
    assert cache.leased_bytes() == 0 and _files(cache.root) == []
    assert client.calls.count(objects[roles[0]].key) == 1


def test_two_threads_asking_for_one_retained_object_give_one_download(cache: Cache) -> None:
    contents = _contents(1)
    obj = next(iter(_objects(contents).values()))
    client = StandIn(contents)
    client.gate = threading.Event()
    results: list[Path] = []
    threads = [
        threading.Thread(target=lambda: results.append(cache.retained(client, obj)))
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    assert client.started.acquire(timeout=HANG_S)
    client.gate.set()
    for thread in threads:
        thread.join(HANG_S)
    assert client.calls == [obj.key] and len(set(results)) == 1 and len(results) == 2
    assert results[0].read_bytes() == contents[obj.key] and cache.leased_bytes() == 0
    assert _files(cache.root) == [results[0]]


def test_forget_removes_one_retained_version_and_its_empty_directories(cache: Cache) -> None:
    contents = _contents(1)
    obj = next(iter(_objects(contents).values()))
    kept = cache.retained(StandIn(contents), obj)
    cache.forget(BUCKET, obj.key, obj.etag)
    cache.forget(BUCKET, obj.key, obj.etag)
    assert not kept.exists() and not kept.parent.parent.exists()
    assert (cache.root / RETAINED).is_dir()


@needs_locks
def test_the_first_use_clears_what_a_dead_process_with_this_pid_left(cache: Cache) -> None:
    (cache.area / ("0" * 32)).mkdir(parents=True)
    stale = cache.area / ("0" * 32) / "stale.bin"
    stale.write_bytes(b"x")
    contents = _contents(1)
    with cache.lease(StandIn(contents), _objects(contents)):
        assert not stale.exists()


# --- prune, and processes ---------------------------------------------------------------------

_HOLDER = """
import hashlib, os, sys, time, threading
from pathlib import Path
import multiprocessing

from fakes3 import FakeS3
from surfcensus.config import RemoteConfig
from surfcensus.remote.cache import Cache
from surfcensus.remote.s3 import S3Client, S3Head, S3Object
from surfcensus.sources.base import FileRole

ROOT, SIGNALS = Path(sys.argv[1]), Path(sys.argv[2])
WAIT = 60.0

def wait_for(name):
    deadline = time.monotonic() + WAIT
    while not (SIGNALS / name).exists():
        if time.monotonic() > deadline:
            sys.exit(3)
        time.sleep(0.02)

def child(signals):
    (Path(signals) / "child.pid").write_text(str(os.getpid()))
    deadline = time.monotonic() + WAIT
    while not (Path(signals) / "release").exists() and time.monotonic() < deadline:
        time.sleep(0.02)

class Client(S3Client):
    def download(self, key, dest, *, expect, cancel=None):
        dest.write_bytes(b"y" * expect.size)
        return S3Head(expect.size, expect.etag, expect.last_modified, None)

if __name__ == "__main__":
    config = RemoteConfig(max_concurrency=4)
    client = Client("fake-bucket", config=config, transport=FakeS3({}).transport())
    cache = Cache(ROOT, max_bytes=10**9, secret=bytes(range(32)))
    obj = S3Object("p/sub-01/surf/lh.white", 1000, "e" * 32, "x")
    with cache.lease(client, {FileRole.LH_WHITE: obj}):
        if sys.argv[3] == "spawn":
            spawn = multiprocessing.get_context("spawn")
            process = spawn.Process(target=child, args=(str(SIGNALS),))
            process.start()
            deadline = time.monotonic() + WAIT
            while not (SIGNALS / "child.pid").exists() and time.monotonic() < deadline:
                time.sleep(0.02)
        (SIGNALS / "ready").write_text(str(cache.area))
        wait_for("release")
"""

_PRUNER = """
import sys
from pathlib import Path
from surfcensus.remote.cache import Cache
print(Cache(Path(sys.argv[1]), max_bytes=1, secret=bytes(range(32))).prune())
"""


def _python(tmp_path: Path, script: str, *args: str) -> subprocess.Popen[str]:
    path = tmp_path / f"script_{abs(hash(script))}.py"
    path.write_text(textwrap.dedent(script))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).parent), env.get("PYTHONPATH", "")])
    return subprocess.Popen(
        [sys.executable, str(path), *args],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait_file(path: Path, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + HANG_S
    while not path.exists():
        if process.poll() is not None:
            pytest.fail(f"the process ended with {process.returncode}: {process.communicate()[1]}")
        assert time.monotonic() < deadline, "the process did not get ready"
        time.sleep(0.02)


@needs_locks
def test_prune_leaves_the_live_area_of_another_process_and_areas_of_other_hosts(
    tmp_path: Path, cache: Cache, caplog: pytest.LogCaptureFixture
) -> None:
    signals = tmp_path / "signals"
    signals.mkdir()
    holder = _python(tmp_path, _HOLDER, str(cache.root), str(signals), "plain")
    try:
        _wait_file(signals / "ready", holder)
        live = Path((signals / "ready").read_text())
        foreign = cache.root / LEASES / "another-host-12"
        foreign.mkdir()
        (foreign / "x.bin").write_bytes(b"x")
        before = _files(live)
        assert len(before) == 1
        with caplog.at_level(logging.INFO, logger="surfcensus"):
            assert cache.prune() == 0
        assert _files(live) == before and (foreign / "x.bin").exists()
        assert "another-host-12" in caplog.text
    finally:
        (signals / "release").write_text("")
        assert holder.wait(HANG_S) == 0, holder.communicate()[1]
    # its process ended in order: the area holds only its lock, which prune now takes
    assert _files(live) == [] and (live / LOCK_FILE).exists()
    assert cache.prune() == 0 and not live.exists()


@needs_locks
def test_prune_after_kill_9_removes_the_dead_area_while_a_spawned_child_lives(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cache"
    signals = tmp_path / "signals"
    signals.mkdir()
    parent = _python(tmp_path, _HOLDER, str(root), str(signals), "spawn")
    child_pid = None
    try:
        _wait_file(signals / "ready", parent)
        child_pid = int((signals / "child.pid").read_text())
        dead = Path((signals / "ready").read_text())
        assert len(_files(dead)) == 1
        os.kill(parent.pid, signal.SIGKILL)
        parent.wait(HANG_S)
        os.kill(child_pid, 0)  # the spawned child lives on
        pruner = _python(tmp_path, _PRUNER, str(root))
        out, err = pruner.communicate(timeout=HANG_S)
        assert pruner.returncode == 0, err
        assert int(out.strip()) == 1 and not dead.exists()
        os.kill(child_pid, 0)  # it was alive while its parent's area was pruned
    finally:
        (signals / "release").write_text("")
        if parent.poll() is None:
            parent.kill()
            parent.wait(HANG_S)
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


# --- through the real client over the fake bucket ---------------------------------------------


def _bucket(tmp_path: Path, contents: dict[str, bytes]) -> FakeS3:
    root = tmp_path / "bucket"
    for key, data in contents.items():
        (root / key).parent.mkdir(parents=True, exist_ok=True)
        (root / key).write_bytes(data)
    return FakeS3({BUCKET: root})


def test_a_lease_through_the_s3_client_over_the_fake_bucket(cache: Cache, tmp_path: Path) -> None:
    contents = _contents(5, size=3 << 20)  # several chunks each
    fake = _bucket(tmp_path, contents)
    objects = _objects(contents)
    keep = next(iter(objects))
    client = S3Client(BUCKET, config=RemoteConfig(max_concurrency=4), transport=fake.transport())
    with cache.lease(client, objects) as paths:
        for role, path in paths.items():
            assert path.read_bytes() == contents[objects[role].key]
    assert _files(cache.root) == [] and cache.leased_bytes() == 0
    assert {entry.method for entry in fake.log} == {"GET"} and not fake.signed
    assert sorted(entry.key or "" for entry in fake.log) == sorted(contents)
    assert {entry.if_match for entry in fake.log} == {f'"{o.etag}"' for o in objects.values()}
    assert fake.high_water <= 4
    kept = cache.retained(client, objects[keep])
    assert kept.read_bytes() == contents[objects[keep].key]
    assert _files(cache.root) == [kept]
    client.close()


def test_an_object_changed_after_its_listing_fails_the_lease_with_nothing_left(
    cache: Cache, tmp_path: Path
) -> None:
    contents = _contents(3)
    fake = _bucket(tmp_path, contents)
    objects = _objects(contents)
    changed = list(objects.values())[1]
    fake.mutate_after_list(BUCKET, changed.key, b"other bytes")
    list(S3Client(BUCKET, config=RemoteConfig(), transport=fake.transport()).list(PREFIX))
    client = S3Client(BUCKET, config=RemoteConfig(max_concurrency=4), transport=fake.transport())
    with pytest.raises(SourceError) as raised, cache.lease(client, objects):
        pytest.fail("the lease did not fail")
    assert raised.value.reason == "changed"
    assert ACCESSION not in str(raised.value) and changed.etag not in str(raised.value)
    assert _files(cache.root) == [] and cache.leased_bytes() == 0
