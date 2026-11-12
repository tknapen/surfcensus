"""The catalog of a run (``catalog.sqlite``) and the parquet tables under ``tables/``.

Stub of milestone 2: the names and signatures are final, the bodies and the SQL are written
by work package 2.C.

Two classes read the same file. :class:`Catalog` is the private side: accessions, roots,
file locations, input signatures, failure details. :class:`PublicCatalog` is what the code
downstream of inventory may hold; it returns none of those.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Collection, Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow as pa

from surfcensus.privacy import HashKey
from surfcensus.records import Provenance, RenderRow, WorkRow
from surfcensus.sources.base import DatasetRecord, FileRole, SubjectRecord

SCHEMA_VERSION = 1
CATALOG_FILE = "catalog.sqlite"

_PACKAGE = "work package 2.C (store.py)"


class Catalog:
    """The private catalog.

    Imported only by ``remote/``, ``inventory/``, ``enrich/``, ``privacy.py``, ``run.py``,
    ``cli.py`` and (milestone 5) ``calibration_transfer.py``.
    """

    def __init__(self, path: Path, *, readonly: bool = False) -> None:
        """Open an existing catalog and apply pending migrations. A file written by a later
        version raises ``ConfigError("newer schema", ...)``."""
        raise NotImplementedError(_PACKAGE)

    @classmethod
    def create(cls, path: Path, *, mode: Literal["local", "census"], key: HashKey) -> Catalog:
        """A new catalog; also draws the run secret."""
        raise NotImplementedError(_PACKAGE)

    def close(self) -> None:
        raise NotImplementedError(_PACKAGE)

    def __enter__(self) -> Catalog:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def write(self) -> AbstractContextManager[sqlite3.Connection]:
        """One ``BEGIN IMMEDIATE`` transaction: committed on success, rolled back otherwise."""
        raise NotImplementedError(_PACKAGE)

    def meta(self, key: str) -> str | None:
        raise NotImplementedError(_PACKAGE)

    def set_meta(self, key: str, value: str) -> None:
        raise NotImplementedError(_PACKAGE)

    def run_secret(self) -> bytes:
        raise NotImplementedError(_PACKAGE)

    def upsert_dataset(
        self, record: DatasetRecord, pseudonym: str, n_subjects_listed: int | None
    ) -> int:
        """Returns the ``dataset_id``."""
        raise NotImplementedError(_PACKAGE)

    def upsert_subject(
        self,
        uid: str,
        dataset_id: int,
        record: SubjectRecord,
        provenance: Provenance,
        inventory_hash: str,
    ) -> None:
        """Replaces the subject's ``files`` rows in the same transaction."""
        raise NotImplementedError(_PACKAGE)

    def record_dataset_file(
        self,
        dataset_id: int,
        role: str,
        bucket: str | None,
        key: str,
        *,
        size: int,
        etag: str,
        version_id: str | None,
        sha256: str | None,
    ) -> None:
        raise NotImplementedError(_PACKAGE)

    def mark_absent(self, dataset_id: int, present_labels: Collection[str]) -> list[str]:
        """Set ``present = 0`` on the dataset's subjects that are not listed any more;
        returns their ``subject_uid``s."""
        raise NotImplementedError(_PACKAGE)

    def relocate(self, dataset_id: int, record: DatasetRecord) -> None:
        """New ``root``, ``fs_root`` and ``template_location``; ``identity`` stays."""
        raise NotImplementedError(_PACKAGE)

    def dataset_ids(self, only: Collection[str] | None = None) -> list[int]:
        """``only``: accessions or pseudonyms."""
        raise NotImplementedError(_PACKAGE)

    def pseudonyms(self, only: Collection[str] | None = None) -> list[str]:
        """Accessions or pseudonyms -> sorted pseudonyms; an unknown value raises
        ``ConfigError("invalid", ...)``."""
        raise NotImplementedError(_PACKAGE)

    def dataset_record(self, dataset_id: int) -> DatasetRecord:
        raise NotImplementedError(_PACKAGE)

    def subject_record(self, uid: str) -> SubjectRecord:
        """Equal to the record the source yielded at inventory."""
        raise NotImplementedError(_PACKAGE)

    def inventory_hash(self, uid: str) -> str | None:
        raise NotImplementedError(_PACKAGE)

    def upsert_template(
        self,
        template_id: str,
        mesh: str,
        origin: str,
        dataset_id: int | None,
        build_stamp: str | None,
        files: Mapping[str, Mapping[str, object]],
    ) -> None:
        raise NotImplementedError(_PACKAGE)

    def failures(self, phase: str | None = None) -> pd.DataFrame:
        """Columns ``unit``, ``phase``, ``reason``, ``detail``; for ``status --failures``."""
        raise NotImplementedError(_PACKAGE)

    def public(self) -> PublicCatalog:
        raise NotImplementedError(_PACKAGE)


class PublicCatalog:
    """What ``metrics/``, ``render/``, ``app/``, ``export/``, ``select/``, ``reference/`` and
    ``calibration.py`` may hold: the same file, without anything private."""

    @classmethod
    def open(cls, run_dir: Path, *, readonly: bool = True) -> PublicCatalog:
        """Needs no key. Read-only by default: no migration is applied, and a catalog older
        than this version raises ``ConfigError("invalid", ...)``, a newer one
        ``ConfigError("newer schema", ...)``."""
        raise NotImplementedError(_PACKAGE)

    def close(self) -> None:
        raise NotImplementedError(_PACKAGE)

    def __enter__(self) -> PublicCatalog:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def mode(self) -> Literal["local", "census"]:
        raise NotImplementedError(_PACKAGE)

    def run_id(self) -> str:
        raise NotImplementedError(_PACKAGE)

    def meta(self, key: str) -> str | None:
        """Only ``run_id``, ``mode``, ``key_fingerprint``, ``surfcensus_version`` and
        ``template_id``."""
        raise NotImplementedError(_PACKAGE)

    def datasets(self) -> pd.DataFrame:
        """The view ``public_datasets``, ordered by pseudonym."""
        raise NotImplementedError(_PACKAGE)

    def subjects(self) -> pd.DataFrame:
        """The view ``public_subjects``, ordered by ``subject_uid``."""
        raise NotImplementedError(_PACKAGE)

    def subject_uids(
        self,
        *,
        datasets: Collection[str] | None = None,
        present: bool | None = True,
        complete: bool | None = None,
        tier_a: bool | None = None,
        tier_b: bool | None = None,
    ) -> list[str]:
        """``datasets``: pseudonyms. Ordered by pseudonym, then ``subject_uid``."""
        raise NotImplementedError(_PACKAGE)

    def dataset_of(self, uid: str) -> str:
        """The pseudonym of the subject's dataset."""
        raise NotImplementedError(_PACKAGE)

    def roles(self, uid: str) -> dict[FileRole, str | None]:
        """Role -> variant, for the roles the subject has."""
        raise NotImplementedError(_PACKAGE)

    def input_signature(
        self, uid: str, roles: Collection[FileRole], extra: Sequence[str] = ()
    ) -> str:
        """``sources.base.input_signature`` over the subject's stored files. The caller may
        pass it back to ``work_done``, ``record_work`` and ``record_renders``, and to
        nothing else."""
        raise NotImplementedError(_PACKAGE)

    def set_file_digests(self, uid: str, digests: Mapping[FileRole, str]) -> None:
        """Store SHA-256 digests of files that were read in full. Write-only."""
        raise NotImplementedError(_PACKAGE)

    def work(self, phase: str) -> pd.DataFrame:
        """Columns ``unit``, ``status``, ``reason``, ``output``, ``finished_at``."""
        raise NotImplementedError(_PACKAGE)

    def work_done(
        self, phase: str, unit: str, config_hash: str, input_signature: str
    ) -> bool | None:
        """True: done and matching. False: failed and matching. None: no row, or a row
        that does not match."""
        raise NotImplementedError(_PACKAGE)

    def record_work(self, rows: Sequence[WorkRow]) -> None:
        raise NotImplementedError(_PACKAGE)

    def renders(self) -> pd.DataFrame:
        """Columns ``subject_uid``, ``config_hash``, ``tile_hash``, ``render_key``,
        ``video``, ``status``, ``reason``, ``media_path``, ``n_frames``, ``template_id``;
        and ``detail`` in a local run only."""
        raise NotImplementedError(_PACKAGE)

    def render(self, uid: str) -> RenderRow | None:
        raise NotImplementedError(_PACKAGE)

    def record_renders(self, rows: Sequence[RenderRow]) -> None:
        raise NotImplementedError(_PACKAGE)

    def selection(self) -> pd.DataFrame:
        raise NotImplementedError(_PACKAGE)

    def replace_selection(self, rows: pd.DataFrame, reasons: Collection[str]) -> None:
        """Replaces the rows of those reasons."""
        raise NotImplementedError(_PACKAGE)


def write_table(
    path: Path, frame: pd.DataFrame, schema: pa.Schema, *, sort_by: Sequence[str]
) -> None:
    """Cast ``frame`` to ``schema`` (a missing or extra column is an error), sort it and write
    zstd parquet atomically, without the pandas index and metadata."""
    raise NotImplementedError(_PACKAGE)


def read_table(path: Path, schema: pa.Schema | None = None) -> pd.DataFrame:
    """An absent file with a ``schema`` gives an empty frame of that schema."""
    raise NotImplementedError(_PACKAGE)


def upsert_part(
    run_dir: Path, table: str, dataset: str, frame: pd.DataFrame, *, replace_uids: Collection[str]
) -> Path:
    """``tables/_parts/<table>/<pseudonym>.parquet``: drop the existing rows of
    ``replace_uids``, append ``frame``, write."""
    raise NotImplementedError(_PACKAGE)


def merge_parts(run_dir: Path, table: str, *, present: Collection[str]) -> Path:
    """Concatenate the parts of ``table``, drop rows whose ``subject_uid`` is not in
    ``present``, sort by ``schemas.KEYS[table]`` and write ``tables/<table>.parquet``."""
    raise NotImplementedError(_PACKAGE)
