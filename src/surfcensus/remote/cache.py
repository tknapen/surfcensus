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

import os
import re
import socket
from collections.abc import Collection, Mapping
from contextlib import AbstractContextManager
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from surfcensus import privacy
from surfcensus.config import ConfigError
from surfcensus.sources.base import FileRole

if TYPE_CHECKING:
    from surfcensus.remote.s3 import S3Client, S3Object

_RB = "work package RB: remote/cache.py"

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
        after the other downloads of the lease were cancelled and their files removed."""
        raise NotImplementedError(_RB)

    def retained(self, client: S3Client, obj: S3Object) -> Path:
        """The retained copy of ``obj``, downloaded first when it is not there: for Tier A
        files, the template and ``retain_roles``. Not counted against ``max_bytes``."""
        raise NotImplementedError(_RB)

    def leased_bytes(self) -> int:
        """The bytes this process holds in open leases now."""
        raise NotImplementedError(_RB)

    def forget(self, bucket: str, key: str, etag: str) -> None:
        """Remove the retained copy of one version of an object, if any (for
        ``Run.prune_cache``)."""
        raise NotImplementedError(_RB)

    def prune(self) -> int:
        """Remove what dead processes of this host left: every lease area under ``leases/``
        whose ``.lock`` can be taken (its process is gone), and temporary files there.
        Never the area of a live process, and never an area of another host (those are
        logged). Returns the number of files removed. Called when a remote source is
        built."""
        raise NotImplementedError(_RB)
