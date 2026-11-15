"""The download cache of a run: ``<run>/cache/`` (work package RB).

Laid down by work package R0 (docs/dev/remote_plan.md): the layout of the directory, the
names of its files (keyed with the run secret, so that no name can be matched to the bucket
listing), the constructor and the signatures with the behaviour each method must have.

Layout::

    cache/
      leases/<host>-<pid>/.lock            held (flock) by the process while it lives
      leases/<host>-<pid>/<h>/<basename>   a file leased for one batch, removed after it
      retained/<h[:2]>/<h>/<basename>      Tier A, the template and ``retain_roles``

``h`` is ``privacy.cache_name(secret, bucket, key, etag)`` and ``basename`` the last
component of the key (``lh.pial.T1``), so that the renderer reads a file under the name of
its role's candidate (``patterns.variant_name``). Every download is written to a temporary
name in this process's lease area, checked, and renamed: ``retained/`` never holds a partial
file. The cache belongs to one run.
"""

from __future__ import annotations

import errno
import os
import re
import socket
import threading
import time
from collections.abc import Collection, Iterator, Mapping
from concurrent.futures import FIRST_EXCEPTION, Future, ThreadPoolExecutor, wait
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from surfcensus import privacy
from surfcensus.config import ConfigError
from surfcensus.log import get_logger
from surfcensus.sources.base import FileRole, SourceError

try:
    import fcntl
except ImportError:  # Windows has no flock: see _lock_file
    fcntl = None

try:
    import msvcrt  # pyrefly: ignore[missing-import]
except ImportError:
    msvcrt = None

if TYPE_CHECKING:
    from surfcensus.remote.s3 import S3Client, S3Head, S3Object

CACHE_DIR = "cache"
"""The directory of the cache inside a run directory."""

LEASES = "leases"
RETAINED = "retained"
LOCK_FILE = ".lock"

DISK_MARGIN_BYTES = 500_000_000
"""Free space that a call on remote data leaves beside one batch (decision D47, item 9)."""


class CacheBudgetError(ConfigError):
    """A lease would take this process over ``max_bytes``: it raises and does not wait
    (decision D47, item 9). ``Run`` sizes its batches so that this does not happen; when it
    does, the run stops with exit code 2 like any other fault of the machine. Its reason is
    ``invalid``."""

    def __init__(self, detail: str) -> None:
        super().__init__("invalid", detail)

    def __reduce__(self) -> tuple[type[CacheBudgetError], tuple[str]]:
        return type(self), (self.detail,)


def _host() -> str:
    """The host name as a path component: letters, digits, dots and hyphens."""
    return re.sub(r"[^A-Za-z0-9.-]", "_", socket.gethostname()) or "host"


_POLL_S = 0.1
"""How long a lease waits on its downloads at a time, so that Ctrl-C in the main thread is
taken between two waits."""

_LOCK_WAIT_S = 10.0
"""How long taking the lock of this process's own area waits for a ``prune`` of another
process that holds it while it removes what a dead process of the same pid left."""


def _lock_file(descriptor: int) -> bool:
    """Take an exclusive lock on the open file without waiting: True when it is held now,
    False when another process holds it. ``OSError`` when the file system refuses locks.
    ``flock`` where there is ``fcntl``; on Windows ``msvcrt.locking`` of the first byte
    (written, not run). Without either, ``OSError(ENOLCK)``."""
    if fcntl is not None:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True
    if msvcrt is not None:  # pragma: no cover - not run on this platform
        try:
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EDEADLK):
                return False
            raise
        return True
    raise OSError(errno.ENOLCK, "no file locks on this platform")


@dataclass
class _Entry:
    """One object downloaded into the lease area, shared by the open leases of this process
    that asked for it. The file goes when the last of them ends."""

    size: int
    retained: Path
    count: int = 1
    retain: bool = False
    ready: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None


@dataclass
class _Area:
    """What this process knows of one lease area: the descriptor of its lock (-1 when it
    runs without one), the bytes of its live entries, and the entries by path."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    descriptor: int | None = None
    """None until the area is first used; -1 when the lock could not be had."""
    bytes: int = 0
    entries: dict[Path, _Entry] = field(default_factory=dict)
    retaining: dict[Path, threading.Lock] = field(default_factory=dict)


_AREAS: dict[str, _Area] = {}
_AREAS_LOCK = threading.Lock()


def _area_state(area: Path) -> _Area:
    """The state of ``area`` in this process: shared by every ``Cache`` of one run in it (the
    second ``Run`` of ``review`` makes a second one), since they share the area and its lock,
    and ``flock`` taken twice by one process on two descriptors conflicts with itself."""
    with _AREAS_LOCK:
        return _AREAS.setdefault(os.path.abspath(area), _Area())


def _remove_tree(path: Path) -> int:
    """Remove ``path`` and everything in it without following links; the number of files and
    links removed. A path that is gone already counts 0."""
    if path.is_symlink() or path.is_file():
        try:
            path.unlink()
        except FileNotFoundError:
            return 0
        return 1
    if not path.is_dir():
        return 0
    removed = 0
    for child in list(path.iterdir()):
        removed += _remove_tree(child)
    try:
        path.rmdir()
    except OSError:
        pass
    return removed


def _rmdir_empty(path: Path) -> None:
    try:
        path.rmdir()
    except OSError:
        pass


class Cache:
    """The cache of one run, as one process uses it.

    ``max_bytes`` bounds the bytes that this process holds in leases at once;
    ``retain_roles`` are kept after their lease beside Tier A and the template
    (``CacheConfig.retain_roles``). ``secret`` is the run secret (32 bytes).
    """

    def __init__(
        self,
        root: Path,
        *,
        max_bytes: int,
        secret: bytes,
        retain_roles: Collection[FileRole] = (),
    ) -> None:
        if len(secret) != 32:
            raise ValueError("the cache is keyed with the run secret, 32 bytes")
        if max_bytes < 1:
            raise ValueError(f"max_bytes is at least 1, not {max_bytes}")
        self.root = Path(root)
        self.max_bytes = max_bytes
        self.retain_roles = frozenset(FileRole(role) for role in retain_roles)
        self._secret = secret
        self.area = self.root / LEASES / f"{_host()}-{os.getpid()}"
        """This process's lease area. The pid is read when the cache is made: a worker
        process that only reads leased paths has none of its own."""

    def __repr__(self) -> str:  # the secret stays out of a traceback
        return f"Cache({str(self.root)!r}, max_bytes={self.max_bytes})"

    def name(self, bucket: str, key: str, etag: str) -> str:
        """``privacy.cache_name`` of one version of one object: 32 hex characters that cannot
        be computed without the run secret."""
        return privacy.cache_name(self._secret, bucket, key, etag)

    def path_for(self, bucket: str, key: str, etag: str, *, retained: bool) -> Path:
        """Where the object is put: ``retained/<h[:2]>/<h>/<basename>`` or
        ``leases/<host>-<pid>/<h>/<basename>``."""
        h = self.name(bucket, key, etag)
        basename = PurePosixPath(key).name
        if not basename or basename in (".", ".."):
            raise ValueError("a key names a file")
        if retained:
            return self.root / RETAINED / h[:2] / h / basename
        return self.area / h / basename

    def lease(
        self,
        client: S3Client,
        objects: Mapping[FileRole, S3Object],
        *,
        retain: bool = False,
    ) -> AbstractContextManager[dict[FileRole, Path]]:
        """Local paths of ``objects`` while the context is open.

        A retained copy (of a retained role, or of an earlier lease with ``retain``) is used
        as it is, with no request. The others are downloaded concurrently (at most the
        client's ``max_concurrency``) into this process's area by ``client.download``. On
        exit, also on an exception and on ``KeyboardInterrupt``, what this lease downloaded is
        removed, but the roles of ``retain_roles`` and, with ``retain``, every role, which
        are moved to ``retained/`` by rename. Raises :class:`CacheBudgetError` when the
        bytes this process holds in leases plus those asked for exceed ``max_bytes``
        (decision D47, item 9: it does not wait). A ``SourceError`` of a download is raised
        after the other downloads of the lease were cancelled and their files removed.

        Two open leases of this process that ask for one version of one object share its
        file in the area: the first downloads it, the second waits for it, the bytes are
        counted once, and the file goes (or moves to ``retained/``) when the last of them
        ends. The budget is that of the area, so of every ``Cache`` of this run in this
        process."""
        return self._lease(client, objects, retain)

    @contextmanager
    def _lease(
        self, client: S3Client, objects: Mapping[FileRole, S3Object], retain: bool
    ) -> Iterator[dict[FileRole, Path]]:
        paths: dict[FileRole, Path] = {}
        fetch: dict[FileRole, S3Object] = {}
        for role, obj in objects.items():
            kept = self.path_for(client.bucket, obj.key, obj.etag, retained=True)
            if _is_whole(kept, obj.size):
                paths[role] = kept
            else:
                fetch[role] = obj
        if not fetch:
            yield paths
            return
        state = self._area()
        joined: list[tuple[Path, _Entry]] = []
        made: dict[Path, tuple[S3Object, _Entry]] = {}
        with state.lock:
            need = 0
            seen: set[Path] = set()
            for obj in fetch.values():
                dest = self.path_for(client.bucket, obj.key, obj.etag, retained=False)
                if dest not in seen and dest not in state.entries:
                    need += obj.size
                seen.add(dest)
            if state.bytes + need > self.max_bytes:
                raise CacheBudgetError(
                    f"a lease of {need} bytes would take this process to "
                    f"{state.bytes + need} bytes of leased files, over the cache's budget "
                    f"of {self.max_bytes} (cache.max_bytes); give a smaller batch or a larger "
                    "budget"
                )
            for role, obj in fetch.items():
                dest = self.path_for(client.bucket, obj.key, obj.etag, retained=False)
                keep = retain or role in self.retain_roles
                entry = state.entries.get(dest)
                if entry is None:
                    entry = _Entry(
                        size=obj.size,
                        retained=self.path_for(client.bucket, obj.key, obj.etag, retained=True),
                    )
                    state.entries[dest] = entry
                    state.bytes += obj.size
                    made[dest] = (obj, entry)
                else:
                    entry.count += 1
                entry.retain = entry.retain or keep
                joined.append((dest, entry))
                paths[role] = dest
        try:
            self._download_all(client, made)
            mine = {id(entry) for _, entry in made.values()}
            for _, entry in joined:
                if id(entry) not in mine:
                    _wait_ready(entry)
            yield paths
        finally:
            self._release(state, joined)

    def _download_all(self, client: S3Client, made: Mapping[Path, tuple[S3Object, _Entry]]) -> None:
        """Download what this lease made entries for, at most the client's ``max_concurrency``
        at once, waiting in short steps so that ``KeyboardInterrupt`` is taken. The first
        error cancels the others; after they ended it is raised."""
        if not made:
            return
        cancel = threading.Event()
        workers = max(1, min(len(made), client.config.max_concurrency))
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="surfcensus-cache")
        futures: dict[Future[S3Head], tuple[Path, S3Object, _Entry]] = {}
        errors: list[BaseException] = []
        try:
            for dest, (obj, entry) in made.items():
                dest.parent.mkdir(parents=True, exist_ok=True)
                future = pool.submit(client.download, obj.key, dest, expect=obj, cancel=cancel)
                futures[future] = (dest, obj, entry)
            pending = set(futures)
            while pending:
                done, pending = wait(pending, timeout=_POLL_S, return_when=FIRST_EXCEPTION)
                for future in done:
                    self._settle(future, futures[future], errors, cancel)
                if errors:
                    cancel.set()
            if errors:
                raise errors[0]
        except BaseException as exc:
            cancel.set()
            for future in futures:
                future.cancel()
            _drain(pool)
            for future, target in futures.items():
                entry = target[2]
                if not entry.ready.is_set():
                    if future.done() and not future.cancelled():
                        self._settle(future, target, errors, cancel)
                    # a download that ended well after the stop is not retained
                    entry.error = entry.error or exc
                    entry.ready.set()
            raise
        finally:
            pool.shutdown(wait=True)

    def _settle(
        self,
        future: Future[S3Head],
        target: tuple[Path, S3Object, _Entry],
        errors: list[BaseException],
        cancel: threading.Event,
    ) -> None:
        """Record how one download ended on its entry; an error that came before the cancel
        joins ``errors``."""
        dest, obj, entry = target
        if entry.ready.is_set() or future.cancelled():
            return
        exc = future.exception()
        if exc is None and not _is_whole(dest, obj.size):
            # the client renames only a checked file to dest; this guards retained/ as well
            exc = SourceError(
                "changed", f"{PurePosixPath(obj.key).name}: the download is not the size listed"
            )
        if exc is not None:
            entry.error = exc
            # an error that comes after the cancel is the cancel's echo, not the cause
            if not cancel.is_set():
                errors.append(exc)
        entry.ready.set()

    def _release(self, state: _Area, joined: list[tuple[Path, _Entry]]) -> None:
        """End this lease's hold on its entries; the last holder removes the file, or moves
        it to ``retained/`` when it is whole and some lease asked to keep it."""
        last: list[tuple[Path, _Entry]] = []
        with state.lock:
            for dest, entry in joined:
                entry.count -= 1
                if entry.count == 0:
                    if state.entries.get(dest) is entry:
                        del state.entries[dest]
                    state.bytes -= entry.size
                    last.append((dest, entry))
        for dest, entry in last:
            if entry.retain and entry.error is None and _is_whole(dest, entry.size):
                _move_to_retained(dest, entry.retained)
            _remove_tree(dest.parent)

    def retained(self, client: S3Client, obj: S3Object) -> Path:
        """The retained copy of ``obj``, downloaded first when it is not there: for Tier A
        files, the template and ``retain_roles``. Not counted against ``max_bytes``.

        The download goes to a directory of its own in this process's area and is renamed
        into ``retained/``, so ``retained/`` never holds a partial file. Two threads of this
        process that ask for one object wait for each other and give one download; two
        processes may both download it, and the second rename replaces the first with the
        same bytes."""
        kept = self.path_for(client.bucket, obj.key, obj.etag, retained=True)
        if _is_whole(kept, obj.size):
            return kept
        state = self._area()
        with state.lock:
            one = state.retaining.setdefault(kept, threading.Lock())
        with one:
            if _is_whole(kept, obj.size):
                return kept
            h = self.name(client.bucket, obj.key, obj.etag)
            staging = self.area / f"{h}.retain"
            dest = staging / kept.name
            try:
                staging.mkdir(parents=True, exist_ok=True)
                client.download(obj.key, dest, expect=obj)
                if not _is_whole(dest, obj.size):
                    raise SourceError(
                        "changed", f"{kept.name}: the download is not the size listed"
                    )
                kept.parent.mkdir(parents=True, exist_ok=True)
                os.replace(dest, kept)
            finally:
                _remove_tree(staging)
        return kept

    def leased_bytes(self) -> int:
        """The bytes this process holds in open leases now (of every ``Cache`` of this run in
        this process, which share the area)."""
        state = _area_state(self.area)
        with state.lock:
            return state.bytes

    def forget(self, bucket: str, key: str, etag: str) -> None:
        """Remove the retained copy of one version of an object, if any (for
        ``Run.prune_cache``), and its directories when they are empty."""
        kept = self.path_for(bucket, key, etag, retained=True)
        try:
            kept.unlink()
        except FileNotFoundError:
            pass
        _rmdir_empty(kept.parent)
        _rmdir_empty(kept.parent.parent)

    def prune(self) -> int:
        """Remove what dead processes of this host left: every lease area under ``leases/``
        whose ``.lock`` can be taken (its process is gone), and temporary files there.
        Never the area of a live process, and never an area of another host (those are
        logged). Returns the number of files removed. Called when a remote source is
        built.

        An area without a ``.lock`` is left: its process may be making it now, and a
        process puts no file into its area before it holds the lock. The count leaves out
        the ``.lock`` files. Where there are no file locks nothing is removed."""
        leases = self.root / LEASES
        try:
            names = sorted(child.name for child in leases.iterdir())
        except FileNotFoundError:
            return 0
        if fcntl is None and msvcrt is None:
            get_logger().info("the cache is not pruned: this platform has no file locks")
            return 0
        host = _host()
        own = os.path.abspath(self.area)
        removed = 0
        for name in names:
            area = leases / name
            if area.is_symlink() or not area.is_dir():
                continue
            owner, _, pid = name.rpartition("-")
            if owner != host or not pid.isdigit():
                get_logger().info(
                    "the cache keeps the lease area %s, which is not of this host", name
                )
                continue
            if os.path.abspath(area) == own and _area_state(self.area).descriptor is not None:
                continue
            removed += _prune_area(area)
        if removed:
            get_logger().info("removed %d files that ended processes left in the cache", removed)
        return removed

    def hold_area(self) -> Path:
        """This process's lease area with its lock held, for a caller that writes a directory
        of its own there (the staging tree of a template) and renames it into ``retained/``:
        an area without its lock could be pruned by another process or cleared by the first
        lease of this one. The lock is held for the life of the process."""
        self._area()
        return self.area

    def _area(self) -> _Area:
        """This process's area, its lock taken on first use: what a dead process with the same
        pid left there is removed then, since no lease of this process exists yet."""
        state = _area_state(self.area)
        with state.lock:
            if state.descriptor is None:
                state.descriptor = _take_own_lock(self.area)
                for child in list(self.area.iterdir()):
                    if child.name != LOCK_FILE:
                        _remove_tree(child)
        return state


def _is_whole(path: Path, size: int) -> bool:
    try:
        info = path.stat()
    except OSError:
        return False
    return path.is_file() and info.st_size == size


def _wait_ready(entry: _Entry) -> None:
    """Wait for the lease that downloads ``entry``, in short steps, so that Ctrl-C is taken."""
    while not entry.ready.wait(_POLL_S):
        pass
    if entry.error is not None:
        raise entry.error


def _drain(pool: ThreadPoolExecutor) -> None:
    """Wait for the downloads of a lease that was stopped; they end within one chunk once the
    cancel event is set. A second Ctrl-C does not leave a download writing into the area that
    the lease is about to remove: it is taken after they ended."""
    interrupted: KeyboardInterrupt | None = None
    while True:
        try:
            pool.shutdown(wait=True)
            break
        except KeyboardInterrupt as exc:
            interrupted = exc
    if interrupted is not None:
        raise interrupted


def _move_to_retained(dest: Path, kept: Path) -> None:
    """Rename a whole file from the lease area into ``retained/``: one rename on one file
    system, so ``retained/`` never sees it partial. Where the rename fails (the two lie on
    two file systems) the file is not retained, and a later call downloads it again."""
    try:
        kept.parent.mkdir(parents=True, exist_ok=True)
        os.replace(dest, kept)
    except OSError as exc:
        get_logger().warning("a file could not be kept in the cache: %s", exc.strerror)


def _take_own_lock(area: Path) -> int:
    """Create ``area`` and hold ``area/.lock`` for the life of the process; -1 when the file
    system or the platform has no locks (logged; ``prune`` of another process cannot tell
    this area from a dead one then). Waits up to :data:`_LOCK_WAIT_S` for a ``prune`` of
    another process that holds the lock, and checks that the file locked is the one under the
    name, since that ``prune`` removes the file before it lets go."""
    deadline = time.monotonic() + _LOCK_WAIT_S
    while True:
        area.mkdir(parents=True, exist_ok=True)
        lock = area / LOCK_FILE
        descriptor = os.open(lock, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            held = _lock_file(descriptor)
        except OSError as exc:
            os.close(descriptor)
            get_logger().warning(
                "the lease area of this process is not locked (%s); another process's "
                "pruning could remove its files",
                exc.strerror or exc,
            )
            return -1
        if held:
            try:
                same = os.fstat(descriptor).st_ino == os.stat(lock).st_ino
            except FileNotFoundError:
                same = False
            if same:
                return descriptor
        os.close(descriptor)
        if time.monotonic() > deadline:
            raise ConfigError(
                "invalid",
                f"the lease area {area.name} of the cache is held by another process",
            )
        time.sleep(0.05)


def _prune_area(area: Path) -> int:
    """Remove a lease area of this host when its lock can be taken; the files removed."""
    lock = area / LOCK_FILE
    try:
        descriptor = os.open(lock, os.O_RDWR)
    except FileNotFoundError:
        return 0
    except OSError:
        return 0
    try:
        try:
            if not _lock_file(descriptor):
                return 0  # its process lives
        except OSError:
            return 0
        removed = 0
        for child in list(area.iterdir()):
            if child.name != LOCK_FILE:
                removed += _remove_tree(child)
        # the file goes while it is locked, so a process that opened it meanwhile finds that
        # the file it locked is no longer under the name, and makes another
        try:
            lock.unlink()
        except FileNotFoundError:
            pass
        _rmdir_empty(area)
        return removed
    finally:
        os.close(descriptor)
