"""A run: a run directory, its sources and the phases that fill it.

``Run`` is the Python API of docs/architecture.md; the CLI is a thin layer over it. It holds
the configuration, the private catalog and the sources, and only delegates: each phase
method imports the phase it calls and hands it the public side of the run
(``records.RunContext``), never the run itself.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import sqlite3
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from surfcensus import privacy
from surfcensus.config import (
    CONFIG_FILE,
    ConfigError,
    RunConfig,
    SourceConfig,
    load_config,
    new_census_config,
    new_local_config,
    remote_dataset_config,
    run_key,
    with_changes,
    write_config,
)
from surfcensus.log import get_logger
from surfcensus.privacy import HashKey
from surfcensus.records import FUNC_INVENTORY_PHASE, METRIC_STAGES, PhaseReport, RunContext
from surfcensus.sources.base import (
    HEMIS,
    TIER_B,
    TIER_C,
    DatasetRecord,
    FileRole,
    Hemi,
    RemoteSource,
    Source,
    SourceError,
)
from surfcensus.store import (
    CATALOG_FILE,
    Catalog,
    PublicCatalog,
    UnfinishedCatalog,
    media_dir,
)

if TYPE_CHECKING:
    import httpx
    import numpy as np
    import pandas as pd

    from surfcensus.records import SessionWork
    from surfcensus.remote.cache import Cache
    from surfcensus.templates import Template, TemplateNotFound


RENDER_LEASED: tuple[FileRole, ...] = (*TIER_C, FileRole.LH_SPHERE_REG, FileRole.RH_SPHERE_REG)
"""The roles that a render leases at most (both ``sphere.reg`` with a template), for the
disk bound of a render on remote data."""


def other_root_hint(
    old: str,
    held: Sequence[tuple[Path, str] | None],
    here: Path | None,
    pipeline: str | None,
    new_run: str,
) -> str:
    """What to say after the root ``old`` of a run that a path of another root was given
    for. ``held`` is what each stored root holds now (``sources.local.subjects_held``),
    ``here`` and ``pipeline`` the subjects directory and pipeline of the path, ``new_run``
    the option that makes a new run. ``--relocate`` is suggested only where it is right:
    data that left the stored root, or the same subjects in the same pipeline under
    another name. A stored root that still holds subjects of another directory is another
    dataset (decision D35, review of the quick start)."""
    if any(place is None for place in held):
        return (
            "which holds no subjects now: pass --relocate if they moved to this path, or "
            f"{new_run} for a new run"
        )
    if any(place == (here, pipeline) for place in held):
        return (
            "where the subjects of this path lie as well, named another way: give "
            f"{old} as the path to go on with that run, --relocate to name the subjects "
            f"as this path from now on (they keep their identifiers), or {new_run} for a "
            "new run"
        )
    if any(place is not None and place[0] == here for place in held):
        return (
            "where the subjects of this path lie as well, in a layout that gives them "
            f"other identifiers: give {old} as the path to go on with that run, or "
            f"{new_run} for a new run"
        )
    return (
        "which still holds its subjects, so this path is another dataset: give it a run "
        f"directory of its own with {new_run}"
    )


def remote_index(config: RunConfig, accession: str) -> int | None:
    """The place in ``config.sources`` of the dataset ``accession`` of the OpenNeuroDerivatives
    bucket, in a local run that stores it; None otherwise.

    ``config.yaml`` holds the dataset's pseudonym and not its accession (decision D47, item
    1). The pseudonym is ``HMAC(salt, "openneuro-derivatives:<accession>")`` with the salt of
    ``config.yaml``, so the accession that a user types again gives it again: that is how a
    second ``review openneuro:dsXXXXXX`` recognises its run, without the catalog."""
    if config.mode != "local":
        return None
    expected = privacy.dataset_pseudonym(run_key(config), "openneuro-derivatives", accession)
    for index, stored in enumerate(config.sources):
        if stored.kind == "openneuro-derivatives" and stored.dataset == expected:
            return index
    return None


def epi_with(
    config: RunConfig,
    *,
    enabled: bool | None = None,
    runs: Sequence[str] | None = None,
    raw: Path | str | None = None,
) -> RunConfig:
    """``config`` with the options of the EPI interface that were given (``--epi/--no-epi``,
    ``--epi-runs``, ``--raw``); None keeps what the run has, as for ``--subjects`` and
    ``--video``. The section ``epi`` is written into ``config.yaml`` only when it then differs
    from the defaults it stands for, so a call that gives no option, or one that gives
    ``--epi`` to a run that never had the section, leaves the bytes of ``config.yaml`` as
    they were (``config.dump_config``), and options that bring the settings back to the
    defaults remove the section. ``runs`` of ``("*",)`` stores no glob (every run). A stored
    ``raw`` cannot be cleared, only replaced. ``raw`` is stored absolute, links resolved. Raises
    ``ConfigError("invalid", ...)`` for a raw dataset that is not a directory and for a
    census run with the interface on (``RunConfig`` refuses it)."""
    from surfcensus.config import EpiConfig, epi_settings

    if enabled is None and runs is None and raw is None:
        return config
    current = epi_settings(config)
    update: dict[str, object] = {}
    if enabled is not None:
        update["enabled"] = enabled
    if runs is not None:
        # ``--epi-runs '*'`` selects every run, which is what no glob means: it clears them
        update["runs"] = () if set(runs) == {"*"} else tuple(dict.fromkeys(runs))
    if raw is not None:
        place = Path(raw)
        if not place.is_dir():
            raise ConfigError("not found", f"--raw {raw}: no such directory")
        update["raw"] = str(place.resolve())
    try:
        changed = EpiConfig.model_validate(current.model_dump() | update)
    except ValueError as exc:
        raise ConfigError("invalid", " ".join(str(exc).split())) from exc
    if changed == current:
        return config
    defaults = EpiConfig() if config.mode == "local" else EpiConfig(enabled=False)
    if changed == defaults:
        # back to the defaults (``--no-epi`` then ``--epi``): no section, so config.yaml has
        # the bytes it had before an EPI option was given (review of step 2)
        return with_changes(config, epi=None)
    return with_changes(config, epi=changed.model_dump())


def epi_run_stems(sources: Sequence[Source], raw: Path | str | None) -> list[str] | None:
    """The derivative stems (``sub-01_ses-1_task-rest_run-1``) of every functional run the
    local ``sources`` give, from listings alone, for the check that an ``--epi-runs`` glob
    matches a run before it is stored; None when no dataset has a functional part."""
    from surfcensus.func import connectors as connectors_
    from surfcensus.func.discover import folders_of, func_source_of
    from surfcensus.func.inventory import list_participant, participants_of

    stems: list[str] | None = None
    for source in sources:
        if source.kind != "local":
            continue
        for record in source.datasets():
            func = func_source_of(record, None if raw is None else Path(raw))
            if func is None:
                continue
            stems = [] if stems is None else stems
            connector = connectors_.connector(func.connector)
            folders = folders_of(func, Path(record.root) / record.fs_root)
            for participant in participants_of(folders):
                listing = list_participant(folders, connector, participant)
                stems += [run.stem for run in listing.deriv_runs]
    return stems


def refuse_unmatched_epi_runs(stems: Sequence[str] | None, globs: Sequence[str]) -> None:
    """``ConfigError("invalid", ...)`` when a glob of ``--epi-runs`` matches no run of
    ``stems`` (:func:`epi_run_stems`), before it is stored, as for ``--subjects``."""
    import fnmatch

    if not globs:
        return
    if stems is None:
        raise ConfigError(
            "invalid", "--epi-runs: this folder has no functional runs (no fMRIPrep derivative)"
        )
    unmatched = [g for g in globs if not any(fnmatch.fnmatchcase(s, g) for s in stems)]
    if unmatched:
        raise ConfigError(
            "invalid",
            f"--epi-runs: no functional run matches {', '.join(unmatched)} (matched against "
            "the BIDS name of the preprocessed series, sub-01_ses-1_task-rest_run-1)",
        )


@dataclass(frozen=True)
class _StoredLocal:
    """A local source as the run holds it."""

    index: int
    """Position in ``config.sources``."""
    root: str
    identity: str
    pseudonym: str


class Run:
    """``Run(run_dir, source)`` creates a run; ``Run(run_dir)`` opens an existing one.

    Creating, in a directory that is empty or absent: ``config.yaml`` and the catalog with
    its run secret. A local run gets a new salt; a census run is created with the dataset
    key as ``key`` and stores its fingerprint only. ``config`` is the configuration to
    start from (``config.new_local_config``); its sources are replaced by those given.
    Of two processes that create a run in one directory at the same time, one is refused
    and changes nothing. A place where no run can be made (no permission, a read-only or
    full disk) raises ``ConfigError("invalid", ...)``. The catalog gets its name when it is
    whole (``Catalog.create``). A creation that was stopped, also by a kill, leaves at most
    a ``config.yaml`` without a catalog, or with a catalog in which no run was made yet;
    the next opening makes the run in it with the salt of that ``config.yaml``, and for a
    census run needs the key to do so.

    Opening needs no key, also for a census run; ``Run.inventory`` of a census run does.
    The sources are rebuilt from ``config.yaml`` when a phase first needs them, so a run
    whose data is not mounted can still be opened. A ``source`` given on opening is matched
    to a stored one by kind and, for a local source, by root; its subject globs and field
    strength then replace the stored ones. A local source whose root matches none raises
    ``ConfigError("other root", ...)``, unless ``relocate``: then the stored root of the
    run's one local source is replaced by the root of the source given, and nothing else
    of it, and every ``subject_uid`` stays as it was. With several local sources
    ``relocate`` is the pseudonym of the dataset that moved. A ``config`` given on opening
    replaces the stored one; it must be of the same run (same mode and identity).
    ``migrate=False`` opens an existing catalog without migrating it (``store.Catalog``), for
    ``surfcensus status``; a phase that needs what a migration adds then raises.

    A run directory inside a directory that a local source reads is refused
    (``ConfigError("invalid", ...)``), before anything is written: its subjects directory,
    and what a subject or a template in it links to (``sources.local.source_places``). A
    source is only read.
    """

    run_dir: Path
    config: RunConfig

    def __init__(
        self,
        run_dir: Path | str,
        source: Source | Sequence[Source] | None = None,
        *,
        config: RunConfig | None = None,
        key: HashKey | None = None,
        relocate: bool | str = False,
        transport: httpx.BaseTransport | None = None,
        migrate: bool = True,
    ) -> None:
        self.run_dir = Path(run_dir)
        self._key = key
        self._migrate = migrate
        self._transport = transport
        self._built: dict[int, Source] = {}
        self._template: Template | None = None
        self._template_looked_for = False
        self._no_template: tuple[str, int] | None = None  # the message, the level it was logged at
        if source is None:
            given: tuple[Source, ...] = ()
        elif isinstance(source, Sequence):
            given = tuple(source)
        else:
            given = (source,)
        if (self.run_dir / CONFIG_FILE).is_file():
            self._open(given, config, relocate)
        else:
            if relocate:
                raise ConfigError("missing", f"{self.run_dir} holds no run to relocate")
            self._create(given, config)

    # creating and opening

    def _create(self, given: tuple[Source, ...], config: RunConfig | None) -> None:
        key = self._key
        census = config.mode == "census" if config is not None else key is not None
        if not census and key is not None:
            raise ConfigError("key", "a local run draws its own salt; a key is for a census run")
        if not census and not given and (config is None or not config.sources):
            raise ConfigError(
                "missing", f"{self.run_dir} holds no run; give a source to create one"
            )
        try:
            occupied = sorted(n for n in os.listdir(self.run_dir) if not n.startswith("."))
        except FileNotFoundError:
            occupied = []
        except NotADirectoryError:
            raise ConfigError("invalid", f"{self.run_dir} is not a directory") from None
        if occupied:
            raise ConfigError(
                "invalid",
                f"{self.run_dir} holds no run and is not empty ({occupied[0]}, ...); a run is "
                "created in a new or empty directory",
            )

        if census:
            if key is None:
                raise ConfigError("key", "a census run is created with the dataset key")
            base = config if config is not None else new_census_config(key)
            hash_key = run_key(base, key)
            if config is not None and config.sources and not given:
                raise ConfigError(
                    "invalid", "the sources of a new census run are given as sources, not stored"
                )
        else:
            if not given:
                assert config is not None  # checked above
                if any(stored.kind != "local" for stored in config.sources):
                    # its accession would be in the catalog, which a new run does not have
                    raise ConfigError(
                        "invalid", "a new run on a remote dataset is created from its source"
                    )
                given = tuple(
                    from_config(stored, config, self.run_dir, transport=self._transport)
                    for stored in config.sources
                )
            if config is not None:
                base = config
            elif _one_dataset(given[0]):
                base = new_local_config(None)  # the source is stored once the salt is drawn
            else:
                base = new_local_config(given[0].config().root or ".")
            hash_key = run_key(base)
        stored = []
        self.config = base  # what the refusal below reads; the sources follow
        for source in given:
            self._refuse_inside(source)
            if _one_dataset(source):
                assert isinstance(source, RemoteSource) and source.accession is not None
                if census:
                    raise ConfigError(
                        "invalid",
                        "a census run lists every dataset of its prefix; a source of one "
                        "dataset is for a local run",
                    )
                # decision D47, item 1: stored by its pseudonym, the accession in the catalog
                stored.append(
                    remote_dataset_config(
                        base, source.accession, subjects=source.globs, bucket=source.bucket
                    )
                )
                continue
            described = source.config()
            if census and described.kind == "local":
                assert described.identity is not None  # a source knows its identity
                described = SourceConfig(
                    kind="local",
                    dataset=privacy.dataset_pseudonym(hash_key, "local", described.identity),
                    subjects=described.subjects,
                    field_strength=described.field_strength,
                )
            stored.append(described)
        if len(set(stored)) != len(stored):
            raise ConfigError("invalid", "the same source was given twice")
        self.config = with_changes(base, sources=tuple(stored))

        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            # exclusive: a process that creates a run here at the same time is refused, or
            # refuses this one, before either can replace the other's salt
            write_config(self.run_dir, self.config, exclusive=True)
            # exist_ok: a process that opens the run before its catalog is there makes the
            # catalog itself, with the salt just written; that catalog is then this run's
            self._catalog = Catalog.create(
                self.run_dir / CATALOG_FILE, mode=self.config.mode, key=hash_key, exist_ok=True
            )
        except (OSError, sqlite3.OperationalError) as exc:
            raise ConfigError(
                "invalid", f"no run can be made in {self.run_dir}: {' '.join(str(exc).split())}"
            ) from exc
        for index, (source, described) in enumerate(zip(given, stored, strict=True)):
            self._built[index] = source
            if isinstance(source, RemoteSource):
                self._bind(source, described)
            if census and described.kind == "local":
                # the root of a local source of a census run is in the catalog only
                assert described.dataset is not None
                for dataset in source.datasets():
                    self._catalog.upsert_dataset(dataset, described.dataset, None)
            elif described.kind != "local" and described.dataset is not None:
                # and so is the accession of the remote dataset of a local run
                self._remote_row(source, described.dataset)

    def _bind(self, source: RemoteSource, stored: SourceConfig) -> None:
        """Give a remote source the cache of this run and, in a local run, its pseudonym, and
        remove what ended processes left in the cache (:meth:`_pruned`)."""
        cache = self._remote_cache()
        source.bind(cache=cache, pseudonym=stored.dataset)
        self._pruned(cache)

    @staticmethod
    def _pruned(cache: Cache | None) -> None:
        """``Cache.prune``: the lease areas of processes of this host that ended without
        cleaning up (killed, say) are removed when a remote source is built. Areas of live
        processes and of other hosts are left."""
        if cache is not None:
            cache.prune()

    def _remote_cache(self) -> Cache:
        """The cache of this run's downloads, keyed with the run secret."""
        from surfcensus.remote.cache import CACHE_DIR, Cache

        return Cache(
            self.run_dir / CACHE_DIR,
            max_bytes=self.config.cache.max_bytes,
            secret=self._catalog.run_secret(),
            retain_roles=self.config.cache.retain_roles,
        )

    def _remote_row(self, source: Source, pseudonym: str) -> None:
        """Write the dataset row of the remote dataset of a local run when the catalog has
        none: the one place its accession is kept (decision D47, item 1). A run whose
        creation stopped before the row was written gets it from the next call that names
        the dataset. Lists the dataset (a few requests) only when the row is missing."""
        found = self._catalog.query(
            "SELECT 1 FROM datasets WHERE source = ? AND pseudonym = ?",
            (source.kind, pseudonym),
        )
        if found:
            return
        accession = getattr(source, "accession", None)
        for dataset in source.datasets():
            if dataset.identity != accession:
                raise SourceError("invalid", "the source listed a dataset it was not given")
            self._catalog.upsert_dataset(dataset, pseudonym, None)

    def _open(
        self, given: tuple[Source, ...], config: RunConfig | None, relocate: bool | str
    ) -> None:
        stored = load_config(self.run_dir)
        if config is not None and (config.mode, config.identity) != (stored.mode, stored.identity):
            raise ConfigError(
                "invalid",
                f"the configuration given is not of the run in {self.run_dir}: its mode or "
                "identity differs",
            )
        self.config = config if config is not None else stored
        if self.config.mode == "local" and self._key is not None:
            raise ConfigError("key", "a local run is keyed with its own salt, not with a key")
        self._catalog = self._open_catalog()
        try:
            if relocate and not given:
                raise ConfigError("invalid", "to relocate, give the source at its new place")
            for source in given:
                self._adopt(source, relocate)
            if self.config != stored:
                write_config(self.run_dir, self.config)
        except BaseException:
            self._catalog.close()
            raise

    def _open_catalog(self) -> Catalog:
        path = self.run_dir / CATALOG_FILE
        identity = self.config.identity
        try:
            catalog = Catalog(path, migrate=self._migrate)
        except ConfigError as exc:
            if exc.reason != "missing" and not isinstance(exc, UnfinishedCatalog):
                raise
            # a run whose creation stopped between its two files, or in the second one
            if self.config.mode == "census" and self._key is None:
                raise ConfigError(
                    "key",
                    f"{self.run_dir} has no finished {CATALOG_FILE}; making it needs the key",
                ) from None
            catalog = Catalog.create(
                path, mode=self.config.mode, key=run_key(self.config, self._key), exist_ok=True
            )
        found = (catalog.meta("mode"), catalog.meta("key_fingerprint"))
        if found != (self.config.mode, identity.key_fingerprint):
            catalog.close()
            raise ConfigError(
                "invalid",
                f"{CONFIG_FILE} and {CATALOG_FILE} in {self.run_dir} are of different runs",
            )
        return catalog

    def _stored_locals(self) -> list[_StoredLocal]:
        """The local sources of the run with where the run believes them to be."""
        found = []
        for index, stored in enumerate(self.config.sources):
            if stored.kind != "local":
                continue
            if self.config.mode == "local":
                assert stored.root is not None  # RunConfig checks it
                identity = stored.identity or stored.root
                pseudonym = privacy.dataset_pseudonym(run_key(self.config), "local", identity)
                found.append(_StoredLocal(index, stored.root, identity, pseudonym))
                continue
            rows = self._catalog.query(
                "SELECT root, identity FROM datasets WHERE source = 'local' AND pseudonym = ?",
                (stored.dataset,),
            )
            if not rows:
                raise ConfigError(
                    "invalid", f"the catalog has no dataset {stored.dataset} of {CONFIG_FILE}"
                )
            assert stored.dataset is not None  # RunConfig checks it
            found.append(_StoredLocal(index, rows[0]["root"], rows[0]["identity"], stored.dataset))
        return found

    def _adopt(self, source: Source, relocate: bool | str) -> None:
        """Match a source given on opening to a stored one and take over what it says."""
        if source.kind != "local":
            self._adopt_remote(source, relocate)
            return
        described = source.config()
        locals_ = self._stored_locals()
        match = next((s for s in locals_ if s.root == described.root), None)
        moved = match is None
        if match is None:
            match = self._relocation_target(locals_, relocate, source)
        current = self.config.sources[match.index]
        # on a move the root alone changes; the subject set and field strength stay as stored
        subjects = current.subjects if moved else described.subjects
        field_strength = current.field_strength if moved else described.field_strength
        if self.config.mode == "local":
            updated = SourceConfig(
                kind="local",
                root=described.root,
                identity=match.identity,
                subjects=subjects,
                field_strength=field_strength,
            )
        else:
            updated = SourceConfig(
                kind="local",
                dataset=match.pseudonym,
                subjects=subjects,
                field_strength=field_strength,
            )
        if updated != current:
            sources = list(self.config.sources)
            sources[match.index] = updated
            self.config = with_changes(self.config, sources=tuple(sources))
        if (described.identity, described.subjects, described.field_strength) == (
            match.identity,
            subjects,
            field_strength,
        ):
            self._refuse_inside(source)
            self._built[match.index] = source
        else:  # the source given is not the one to use: rebuild it as the run stores it
            self._built.pop(match.index, None)
            source = self._source(match.index, root=described.root)
        if moved:
            rows = self._catalog.query(
                "SELECT dataset_id, pipeline FROM datasets WHERE source = 'local' AND identity = ?",
                (match.identity,),
            )
            for dataset in source.datasets():
                # A row of another pipeline is what the data was before an inventory found
                # it in the layout of this one; its subjects are absent. It follows to the
                # new root and is left alone otherwise. With no row of this pipeline,
                # Catalog.relocate refuses the move.
                ids = [row["dataset_id"] for row in rows]
                found = [row["dataset_id"] for row in rows if row["pipeline"] == dataset.pipeline]
                for dataset_id in found or ids:
                    self._catalog.relocate(dataset_id, dataset)
                    self._relocate_func(dataset_id, dataset)
                earlier = [dataset_id for dataset_id in ids if dataset_id not in found]
                if found and earlier:
                    with self._catalog.write() as conn:
                        conn.executemany(
                            "UPDATE datasets SET root = ? WHERE dataset_id = ?",
                            [(dataset.root, dataset_id) for dataset_id in earlier],
                        )

    def _adopt_remote(self, source: Source, relocate: bool | str) -> None:
        """Match a remote source given on opening to a stored one. In a local run the source
        names its dataset by accession, and the stored source by pseudonym
        (:func:`remote_index`); the globs of the source given replace the stored ones, as
        for a local source. A dataset of another accession is ``other root``."""
        if relocate:
            raise ConfigError(
                "invalid", "a remote dataset does not move; --relocate is for local data"
            )
        if self.config.mode == "local" and _one_dataset(source):
            assert isinstance(source, RemoteSource) and source.accession is not None
            index = remote_index(self.config, source.accession)
            if index is None:
                raise ConfigError(
                    "other root",
                    f"this run directory belongs to {self._named_sources()}, not to the "
                    "dataset given: give it a run directory of its own with --run-dir DIR",
                )
            updated = remote_dataset_config(
                self.config, source.accession, subjects=source.globs, bucket=source.bucket
            )
            if updated != self.config.sources[index]:
                sources = list(self.config.sources)
                sources[index] = updated
                self.config = with_changes(self.config, sources=tuple(sources))
            self._refuse_inside(source)
            self._built[index] = source
            self._bind(source, updated)
            assert updated.dataset is not None
            self._remote_row(source, updated.dataset)
            return
        described = source.config()
        matches = [i for i, s in enumerate(self.config.sources) if s == described]
        if not matches:
            if self.config.mode == "local":
                raise ConfigError(
                    "other root",
                    f"this run directory belongs to {self._named_sources()}: give the "
                    "source a run directory of its own with --run-dir DIR",
                )
            raise ConfigError("invalid", f"this run has no such {described.kind} source")
        self._built[matches[0]] = source
        if isinstance(source, RemoteSource):
            self._bind(source, described)

    def _named_sources(self) -> str:
        """The stored sources as a message names them: a local root in a local run, else the
        pseudonym; a remote dataset by its pseudonym, never by its accession."""
        names = []
        for stored in self.config.sources:
            if stored.kind == "local" and self.config.mode == "local":
                names.append(str(stored.root))
            elif stored.kind == "local":
                names.append(str(stored.dataset))
            elif stored.dataset is not None:
                names.append(f"{stored.dataset}, a dataset of the {stored.bucket} bucket")
            else:
                names.append(f"the {stored.bucket} bucket")
        return ", ".join(names) or "no source"

    def _relocation_target(
        self, locals_: Sequence[_StoredLocal], relocate: bool | str, source: Source
    ) -> _StoredLocal:
        """The stored source that ``source``, a local source at another root, stands for.

        Without ``relocate`` this refuses (``other root``) and says what to give: when a
        stored root still holds its subjects, ``source`` is another dataset or the same
        subjects named another way, and ``--relocate`` is suggested only for the second
        within one pipeline. With ``relocate`` a stored root that still holds subjects of
        another subjects directory is refused (``invalid``): two datasets would be merged
        under one set of identifiers (decision D35, review of the quick start)."""
        from surfcensus.sources.local import subjects_held

        if not locals_:
            if self.config.mode == "local" and self.config.sources:
                # a run on a remote dataset, given a path
                raise ConfigError(
                    "other root",
                    f"this run directory belongs to {self._named_sources()}, not to this "
                    "path: give it a run directory of its own with --run-dir DIR",
                )
            raise ConfigError("invalid", "this run has no local source")
        local = self.config.mode == "local"

        def named(stored: _StoredLocal) -> str:
            # a census run keeps its roots out of sight: name its datasets by pseudonym
            return stored.root if local else stored.pseudonym

        given = getattr(source, "subjects_dir", None)
        here = Path(given).resolve() if isinstance(given, Path) else None
        dataset = getattr(source, "dataset", None)
        pipeline = getattr(dataset, "pipeline", None)
        held = {stored.index: subjects_held(stored.root) for stored in locals_}
        if not relocate:
            old = ", ".join(named(stored) for stored in locals_)
            hint = other_root_hint(old, list(held.values()), here, pipeline, "--run-dir DIR")
            raise ConfigError("other root", f"this run directory belongs to {old}, {hint}")
        if isinstance(relocate, str):
            chosen = [s for s in locals_ if s.pseudonym == relocate]
            if not chosen:
                raise ConfigError("invalid", f"this run has no local dataset {relocate}")
            target = chosen[0]
        elif len(locals_) > 1:
            names = ", ".join(sorted(s.pseudonym for s in locals_))
            raise ConfigError(
                "invalid",
                f"this run has {len(locals_)} local sources ({names}); say which one moved: "
                "--relocate D-xxxxxxxx",
            )
        else:
            target = locals_[0]
        place = held[target.index]
        if place is not None and place[0] != here:
            raise ConfigError(
                "invalid",
                f"--relocate: the data of this run are still where they were "
                f"({named(target)}), so this path looks like another dataset, not the same "
                "data moved; give it a run directory of its own with --run-dir DIR. If that "
                "place holds a copy that this run should no longer read, move or rename the "
                "copy first, then --relocate takes the run to this path",
            )
        return target

    def _template_places(self) -> list[tuple[str, Path]]:
        """The fsaverage directories outside the subjects directories that this call may
        read a template from: the one given for the call and that of a FreeSurfer install.
        The directory a pinned template was once built from is not stored (the run keeps
        the template by content), so it is known only to the call that names it."""
        from surfcensus.sources.local import freesurfer_template_places

        places = freesurfer_template_places()
        if self.config.metrics.fsaverage_dir is not None:
            places.append(("the fsaverage directory", Path(self.config.metrics.fsaverage_dir)))
        return places

    def _raw_places(self) -> list[tuple[str, Path]]:
        """The participant folders of the raw dataset named with ``--raw`` (``epi.raw``),
        which the functional side reads."""
        from surfcensus.config import epi_settings
        from surfcensus.func.discover import participant_places

        raw = epi_settings(self.config).raw
        if raw is None:
            return []
        return participant_places((("the raw dataset", Path(raw)),))

    def _refuse_inside(self, source: Source) -> None:
        """Refuse a run directory that is, or lies inside, what ``source`` reads, or the
        participant folders of a raw dataset named with ``--raw``; and a raw dataset that
        lies inside the run directory."""
        from surfcensus.config import epi_settings
        from surfcensus.sources.local import refuse_in_sources

        raw = epi_settings(self.config).raw
        if raw is not None and Path(raw).resolve().is_relative_to(self.run_dir.resolve()):
            raise ConfigError(
                "invalid", f"--raw {raw} lies inside the run directory {self.run_dir}"
            )
        also = [*self._template_places(), *self._raw_places()]
        subjects_dir = getattr(source, "subjects_dir", None)
        if not isinstance(subjects_dir, Path):
            # a remote source reads nothing on this machine but the fsaverage directories
            # of this call and of a FreeSurfer install, which are only read as well
            if also:
                refuse_in_sources(self.run_dir, [], what="run", also=also)
            return
        refuse_in_sources(self.run_dir, [subjects_dir], what="run", also=also)

    def set_epi(
        self,
        *,
        enabled: bool | None = None,
        runs: Sequence[str] | None = None,
        raw: Path | str | None = None,
    ) -> bool:
        """Store the options of the EPI interface that were given (:func:`epi_with`) in
        ``config.yaml``; returns whether it changed. A glob of ``runs`` that matches no
        functional run is refused before it is stored (:func:`refuse_unmatched_epi_runs`),
        and a raw dataset whose participant folders hold the run directory is refused."""
        changed = epi_with(self.config, enabled=enabled, runs=runs, raw=raw)
        if changed == self.config:
            return False
        from surfcensus.config import epi_settings

        settings = epi_settings(changed)
        if runs:
            stems = epi_run_stems(self._sources(), settings.raw)
            refuse_unmatched_epi_runs(stems, runs)
        earlier = self.config
        self.config = changed
        try:
            for index in range(len(self.config.sources)):
                if index in self._built:
                    self._refuse_inside(self._built[index])
            write_config(self.run_dir, changed)
        except BaseException:
            self.config = earlier
            raise
        return True

    def refuse_output_in_sources(self, out: Path | str) -> None:
        """Raise ``ConfigError("invalid", ...)`` when ``out`` is, or lies inside, a directory
        that a local source of the run reads: its subjects directory, the directory that a
        subject or a template in it links to (``sources.local.source_places``), the
        participant folders of its functional derivative and raw dataset
        (``func.discover.stored_read_places``), or the fsaverage directory of this call or of
        a FreeSurfer install. ``out`` is compared as it is written and with links resolved:
        sources are only read. Where the source cannot be built now (the data is not
        mounted), ``out`` is compared with its stored root."""
        from surfcensus.func.discover import stored_read_places
        from surfcensus.sources.local import refuse_in_sources

        places = []
        for stored in self._stored_locals():
            try:
                place = getattr(self._source(stored.index), "subjects_dir", None)
            except SourceError:
                place = None
            places.append(place if isinstance(place, Path) else Path(stored.root))
        also = [*self._template_places(), *stored_read_places(self.config)]
        refuse_in_sources(out, places, what="output", also=also)

    # sources

    def _source(self, index: int, *, root: str | None = None) -> Source:
        """The source of ``config.sources[index]``, built on first use."""
        if index not in self._built:
            stored = self.config.sources[index]
            identity = None
            secret = None
            if stored.kind == "local":
                local = next(s for s in self._stored_locals() if s.index == index)
                root, identity = root or local.root, local.identity
            else:
                secret = self._catalog.run_secret()
                if stored.dataset is not None:
                    identity = self._remote_accession(stored)
            source = from_config(
                stored,
                self.config,
                self.run_dir,
                root=root,
                identity=identity,
                transport=self._transport,
                secret=secret,
            )
            self._refuse_inside(source)
            if stored.kind != "local":
                self._pruned(getattr(source, "cache", None))
            self._built[index] = source
        return self._built[index]

    def _remote_accession(self, stored: SourceConfig) -> str:
        """The accession of the remote dataset that ``stored`` names by pseudonym, from the
        catalog, the one place it is kept (decision D47, item 1)."""
        rows = self._catalog.query(
            "SELECT accession FROM datasets WHERE source = ? AND pseudonym = ?",
            (stored.kind, stored.dataset),
        )
        if not rows or rows[0]["accession"] is None:
            raise ConfigError(
                "invalid",
                f"the catalog has no dataset {stored.dataset} of {CONFIG_FILE}: run the "
                "command again with the dataset named (openneuro:dsXXXXXX), which records it",
            )
        return rows[0]["accession"]

    def _sources(self) -> list[Source]:
        """Every source of the run. Raises ``SourceError`` when one cannot be built, for
        example because its root is gone."""
        return [self._source(index) for index in range(len(self.config.sources))]

    def _source_of(self, dataset: DatasetRecord) -> Source:
        for source in self._sources():
            if source.kind != dataset.source:
                continue
            if source.kind != "local" or source.config().identity == dataset.identity:
                return source
        raise SourceError("not found", "no source of this run holds the subject's dataset")

    def _lease(
        self, subject_uid: str, roles: Collection[FileRole]
    ) -> AbstractContextManager[Mapping[FileRole, Path]]:
        record = self._catalog.subject_record(subject_uid)
        return self._source_of(record.dataset).lease(record, roles)

    def _remote_roles_bytes(self, roles: Collection[FileRole]) -> list[int]:
        """Per present subject of a remote dataset, the bytes of its listed files of
        ``roles``, from the private catalog: what one lease of the subject downloads at most."""
        if not roles or all(stored.kind == "local" for stored in self.config.sources):
            return []
        marks = ", ".join("?" for _ in roles)
        rows = self._catalog.query(
            "SELECT SUM(f.size) AS total FROM files f "
            "JOIN subjects s ON s.subject_uid = f.subject_uid "
            "JOIN datasets d ON d.dataset_id = s.dataset_id "
            f"WHERE d.source != 'local' AND s.present = 1 AND f.role IN ({marks}) "
            "GROUP BY f.subject_uid",
            tuple(FileRole(role).value for role in roles),
        )
        return [int(row["total"] or 0) for row in rows]

    @contextmanager
    def _remote_batches(self, roles: Collection[FileRole], what: str) -> Iterator[None]:
        """The disk bound of a phase on remote data (docs/dev/remote_plan.md, section 5).

        One batch leases at most ``cache.batch_subjects`` subjects at once, each at most the
        largest subject's files of ``roles``. When that exceeds ``cache.max_bytes`` the batch
        size is lowered for the length of the block (one line at info level), so that a
        lease never raises ``CacheBudgetError``; a single subject over the budget is refused
        (``ConfigError("invalid", ...)``), as is a call when the disk of the cache has less
        free space than one batch and ``DISK_MARGIN_BYTES``. Nothing happens for a run
        without a remote dataset."""
        from surfcensus.remote.cache import CACHE_DIR, DISK_MARGIN_BYTES

        sizes = self._remote_roles_bytes(roles)
        largest = max(sizes, default=0)
        if largest == 0:
            yield
            return
        settings = self.config.cache
        if largest > settings.max_bytes:
            raise ConfigError(
                "invalid",
                f"the {what} of one subject lease {size_text(largest)} of source files, more than "
                f"the cache's budget of {size_text(settings.max_bytes)} (cache.max_bytes in "
                f"{CONFIG_FILE}): raise cache.max_bytes",
            )
        batch = min(settings.batch_subjects, max(1, settings.max_bytes // largest))
        free = free_bytes(self.run_dir / CACHE_DIR)
        need = batch * largest
        if free is not None and free < need + DISK_MARGIN_BYTES:
            raise ConfigError(
                "invalid",
                f"the disk of {self.run_dir} has {size_text(free)} free, and the {what} need "
                f"{size_text(need)} for a batch of {batch} subject(s) beside a margin of "
                f"{size_text(DISK_MARGIN_BYTES)}: make room, or lower cache.batch_subjects in "
                f"{CONFIG_FILE}",
            )
        if batch == settings.batch_subjects:
            yield
            return
        get_logger().info(
            "batches of %d subjects for the %s, so that one batch fits cache.max_bytes",
            batch,
            what,
        )
        earlier = self.config
        self.config = with_changes(earlier, cache=settings.model_dump() | {"batch_subjects": batch})
        try:
            yield
        finally:
            if self.config.cache.batch_subjects == batch:
                self.config = with_changes(
                    self.config,
                    cache=self.config.cache.model_dump()
                    | {"batch_subjects": earlier.cache.batch_subjects},
                )

    def prune_cache(self) -> int:
        """Remove what the run no longer needs from the retained cache of its remote
        datasets; returns the number of files removed.

        An object is kept while a ``files`` row of a present subject names it (key and
        ETag, as ``Cache.name`` keys it): Tier A files and ``cache.retain_roles`` of the
        subjects the run has now. The rest is a file of an earlier version of a subject or
        of a subject that left, or the ``fsaverage`` next to the subjects with its tree;
        it is removed only once the run's pinned template reads from ``templates/<id>/``,
        since before that the rest may be the template the next call is built from.
        ``Run.metrics`` calls this at its end on a run with a remote dataset."""
        from surfcensus.remote.cache import CACHE_DIR, Cache
        from surfcensus.templates import TEMPLATES_DIR, read_template

        remote = [i for i, stored in enumerate(self.config.sources) if stored.kind != "local"]
        if not remote or not (self.run_dir / CACHE_DIR).is_dir():
            return 0
        pinned = self._catalog.meta("template_id")
        if pinned is None:
            return 0
        try:
            if read_template(self.run_dir / TEMPLATES_DIR / pinned).template_id != pinned:
                return 0
        except (OSError, ValueError):
            return 0
        cache = Cache(
            self.run_dir / CACHE_DIR,
            max_bytes=self.config.cache.max_bytes,
            secret=self._catalog.run_secret(),
        )
        rows = self._catalog.query(
            "SELECT d.root, s.fs_dir, f.location, f.etag_or_mtime FROM files f "
            "JOIN subjects s ON s.subject_uid = f.subject_uid "
            "JOIN datasets d ON d.dataset_id = s.dataset_id "
            "WHERE d.source != 'local' AND s.present = 1"
        )
        keep: set[str] = set()
        for row in rows:
            bucket, _, base = str(row["root"]).removeprefix("s3://").partition("/")
            key = f"{base}{row['fs_dir']}/{row['location']}"
            keep.add(cache.name(bucket, key, row["etag_or_mtime"]))
        removed = cache.prune_retained(keep)
        if removed:
            get_logger().info(
                "removed %d files from the cache that the run no longer needs", removed
            )
        return removed

    def _pseudonyms(self, datasets: Collection[str] | None) -> list[str] | None:
        """Accessions or pseudonyms, as a user names datasets, to pseudonyms."""
        return None if datasets is None else self._catalog.pseudonyms(datasets)

    # phases

    def inventory(
        self, *, datasets: Collection[str] | None = None, force: bool = False
    ) -> PhaseReport:
        """List the sources and record their subjects (phase 2).

        ``datasets``: accessions or pseudonyms, here and in the other phases; a value the
        run does not know raises ``ConfigError("invalid", ...)``. A census run needs to
        have been opened with its key: ``ConfigError("key", ...)`` otherwise. Raises
        ``SourceError`` when a source cannot be built.
        """
        from surfcensus.inventory.subjects import run_inventory

        key = run_key(self.config, self._key)
        return run_inventory(
            self._catalog,
            self._sources(),
            self.config,
            key,
            datasets=self._pseudonyms(datasets),
            force=force,
        )

    def template(self, *, retemplate: bool = False) -> Template | None:
        """The fsaverage template of the run (``templates.find_template``), or None for a
        run that has none.

        Looked for, in this order: the directory of ``metrics.fsaverage_dir`` when it is set,
        ``$FREESURFER_HOME/subjects/fsaverage``, the ``fsaverage`` next to the subjects of
        the first dataset, in pseudonym order, that has one, whether or not the inventory
        has seen it. The first use stores the template under ``templates/<template_id>/``,
        records it in the catalog and pins the run to it (``meta.template_id``); from then
        on the run keeps that template whatever the environment looks like. ``retemplate``
        looks again and pins what it finds.

        A run that never had a template gets None, after one warning that says where a
        template was looked for and how to give one. A run that is pinned never does:
        ``TemplateError("changed", ...)`` is raised when its template can be neither read
        from the cache nor built again, or when the directory given builds another one.
        ``TemplateNotFound`` is raised for a directory that was given, cannot be used and
        has no pinned template to stand in for it.
        """
        return self._find_template(retemplate, logging.WARNING)

    def _find_template(self, retemplate: bool, level: int) -> Template | None:
        """:meth:`template`, saying that there is none at ``level``: a render mentions it
        only to those who ask for more output, because the metrics say it aloud."""
        from surfcensus.templates import (
            TEMPLATE_GONE,
            TEMPLATES_DIR,
            TemplateError,
            TemplateNotFound,
            find_template,
            store_template,
        )

        if self._template_looked_for and not retemplate:
            if self._no_template is not None and level > self._no_template[1]:
                get_logger().log(level, self._no_template[0])
                self._no_template = (self._no_template[0], level)
            return self._template
        pinned = self._catalog.meta("template_id")
        cache = self.run_dir / TEMPLATES_DIR
        adjacent_id: int | None = None

        def find(adjacent: Path | None) -> Template:
            return find_template(
                config=self.config.metrics,
                adjacent=adjacent,
                cache_dir=cache,
                pinned=pinned,
                retemplate=retemplate,
            )

        try:
            # The fsaverage next to the subjects is looked for only when nothing before it
            # gives the template: a pinned template whose templates/<id>/ reads, the
            # directory of this call and that of a FreeSurfer install. For a remote dataset
            # looking for it downloads it (about 40 MB), so a run that holds its template
            # lists and fetches nothing for it.
            try:
                found = find(None)
            except TemplateNotFound:
                if self.config.metrics.fsaverage_dir is not None:
                    raise
                adjacent, adjacent_id = self._adjacent_template()
                found = find(adjacent)
            except TemplateError as exc:
                if pinned is None or retemplate or TEMPLATE_GONE not in str(exc):
                    raise
                adjacent, adjacent_id = self._adjacent_template()
                if adjacent is None:
                    raise
                found = find(adjacent)
        except TemplateNotFound as exc:
            if self.config.metrics.fsaverage_dir is not None:
                raise  # a directory that was asked for and cannot be used is no "none found"
            if pinned is not None:
                raise TemplateError(
                    "changed",
                    "the template this run was scored with is no longer available; "
                    "pass --fsaverage DIR or --retemplate",
                ) from exc
            self._no_template = (self._no_template_message(exc), level)
            get_logger().log(level, self._no_template[0])
            self._template, self._template_looked_for = None, True
            return None
        store_template(cache, found)  # written again when it does not read back
        if found.template_id != pinned:
            self._catalog.upsert_template(
                found.template_id,
                found.mesh,
                found.origin,
                adjacent_id if found.origin == "adjacent" else None,
                found.build_stamp,
                {
                    name: {"size": size, "sha256": digest}
                    for name, (size, digest) in found.files.items()
                },
            )
            self._catalog.set_meta("template_id", found.template_id)
        self._template, self._template_looked_for, self._no_template = found, True, None
        return found

    def _adjacent_template(self) -> tuple[Path | None, int | None]:
        """The ``fsaverage`` next to the subjects of the first local dataset, in pseudonym
        order, that has one with every template file, and that dataset's row.

        The directory is looked for where the inventory recorded it and, for a dataset
        without such a record, at its usual place next to the subjects: an ``fsaverage``
        that was put there after the inventory is found without a second inventory. When
        no dataset has a usable one, the first directory that was recorded or exists is
        returned, so that the lookup can say what it lacks.
        """
        from surfcensus.sources.local import TEMPLATE_NAME, template_files_missing

        rows = self._catalog.query(
            "SELECT dataset_id, source, root, fs_root, template_location FROM datasets "
            "WHERE root IS NOT NULL ORDER BY pseudonym, source, pipeline"
        )
        incomplete: tuple[Path | None, int | None] = (None, None)
        for row in rows:
            if row["source"] != "local":
                # a dataset of a bucket: its fsaverage, when its listing has every template
                # file, downloaded into the retained cache (``template_dir``), whose files
                # are those of a local copy of it, so the template id is the same
                found = self._remote_template(row["dataset_id"])
                if found is not None:
                    return found, row["dataset_id"]
                continue
            recorded = row["template_location"]
            location = recorded or (PurePosixPath(row["fs_root"]) / TEMPLATE_NAME).as_posix()
            directory = Path(row["root"]) / location
            try:
                missing = template_files_missing(directory)
            except OSError:
                missing = ("unreadable",)
            if missing == ():
                return directory, row["dataset_id"]
            if incomplete[0] is None and (recorded is not None or missing is not None):
                incomplete = (directory, row["dataset_id"])
        return incomplete

    def _remote_template(self, dataset_id: int) -> Path | None:
        """The ``fsaverage`` next to the subjects of the remote dataset ``dataset_id`` as a
        directory in the retained cache (``template_dir`` of its source), or None when its
        listing lacks a template file or no source of the run holds the dataset."""
        record = self._catalog.dataset_record(dataset_id)
        if record.template_location is None:
            return None
        try:
            source = self._source_of(record)
        except SourceError:
            return None
        template_dir = getattr(source, "template_dir", None)
        if template_dir is None:
            return None
        found = template_dir(record)
        return found if isinstance(found, Path) else None

    def _no_template_message(self, exc: TemplateNotFound) -> str:
        names = {
            "explicit": "the directory given",
            "freesurfer_home": "$FREESURFER_HOME/subjects/fsaverage",
            "adjacent": "fsaverage next to the subjects",
            "bundled": "bundled copy",
        }
        lines = ["No fsaverage template found. Looked in:"]
        for origin, where, state in exc.looked_in:
            # a census run keeps its paths out of sight
            place = where if self.config.mode == "local" else names.get(origin, origin)
            lines.append(f"  {place}   ({state})")
        lines += [
            "Thickness is not compared per location, and click marks will not map to "
            "fsaverage. Subjects are ranked on",
            "the raw ceiling fraction and the occipital-to-control thickness ratio (decision D16).",
            "To enable: pass --fsaverage /path/to/fsaverage (any FreeSurfer install has one "
            "in subjects/).",
        ]
        return "\n".join(lines)

    def metrics(
        self,
        *,
        datasets: Collection[str] | None = None,
        stages: Collection[str] | None = None,
        jobs: int = 1,
        force: bool = False,
        retry_failed: bool = False,
        retemplate: bool = False,
    ) -> PhaseReport:
        """Compute the metrics that are pending (phase 4); see
        ``metrics.pipeline.run_metrics``.

        ``stages`` are names of ``records.ALL_METRIC_STAGES``. Without them the four stages
        of ``records.METRIC_STAGES`` run, or, when ``datasets`` are given, only the two that
        work per subject: a stage on the whole run is not restricted by a dataset.
        ``retemplate`` is that of :meth:`template`.

        Raises ``TemplateError`` when the run's template is gone, ``ConfigError`` for a
        stage that does not exist, and ``SourceError`` when a subject has to be read and a
        source cannot be built, before anything is recorded against a subject.
        """
        from surfcensus.metrics.pipeline import (
            PER_SUBJECT_STAGES,
            metrics_pending,
            run_metrics,
        )

        pseudonyms = self._pseudonyms(datasets)
        if stages is None:
            stages = METRIC_STAGES if datasets is None else PER_SUBJECT_STAGES
        template = self.template(retemplate=retemplate)
        context = self.context()
        pending = metrics_pending(
            context,
            template=template,
            datasets=pseudonyms,
            stages=stages,
            force=force,
            retry_failed=retry_failed,
        )
        if pending:
            self._sources()  # a source that is gone is not a property of its subjects
        with self._remote_batches(TIER_B if pending else (), "metrics"):
            report = run_metrics(
                self.context(),
                template=template,
                datasets=pseudonyms,
                stages=stages,
                jobs=jobs,
                force=force,
                retry_failed=retry_failed,
            )
        self.prune_cache()
        return report

    def render(
        self,
        *,
        jobs: int = 1,
        datasets: Collection[str] | None = None,
        subjects: Sequence[str] | None = None,
        all_subjects: bool | None = None,
        video: bool | None = None,
        force: bool = False,
        retry_failed: bool = False,
        native: bool = True,
        stop: Callable[[], bool] | None = None,
    ) -> PhaseReport:
        """Render the pending subjects into ``media/<subject_uid>/`` (phase 6); see
        ``render.batch.run_render``.

        With a template (:meth:`template`), the vertex map of each render carries the
        fsaverage counterparts of its vertices, and the call ends with the ``native`` stage
        of the metrics, which gives the cluster centroids their native position. Without
        one the renders have no counterparts, which is said at info level only: the metrics
        are where a missing template is reported.

        ``subjects`` are ``subject_uid``s; in a local run a subject may also be named by
        its label. Raises ``SourceError`` when there is something to render and a source
        cannot be built, before anything is recorded against a subject. A call that finds
        every render recorded needs no source.

        ``native=False`` leaves the ``native`` stage out, for a caller that renders a run
        in several calls and ends with one that has it. ``stop`` is asked before each batch
        of renders; once it answers true the call returns with what was rendered, the rest
        stays pending and the ``native`` stage is left out as well (``surfcensus review``
        ends its renderer this way).
        """
        from surfcensus.metrics.pipeline import update_native
        from surfcensus.render.batch import render_pending, run_render

        template = self._find_template(False, logging.INFO)
        context = self.context()
        pseudonyms = self._pseudonyms(datasets)
        uids = None if subjects is None else self._subject_uids(subjects)
        pending = render_pending(
            context,
            template=template,
            datasets=pseudonyms,
            subjects=uids,
            all_subjects=all_subjects,
            video=video,
            force=force,
            retry_failed=retry_failed,
        )
        if pending:
            self._sources()  # a source that is gone is not a property of its subjects
        roles = RENDER_LEASED if pending else ()
        with self._remote_batches(roles, "renders"):
            report = run_render(
                self.context(),
                template=template,
                jobs=jobs,
                datasets=pseudonyms,
                subjects=uids,
                all_subjects=all_subjects,
                video=video,
                force=force,
                retry_failed=retry_failed,
                stop=stop,
            )
        if not native or (stop is not None and stop()):
            return report
        # the cluster centroids of the subjects just rendered get their native position
        located = update_native(context, template=template)
        return dataclasses.replace(report, notes=(*report.notes, *located.notes))

    def func_inventory(self, *, force: bool = False) -> PhaseReport:
        """Phase 2f, the functional inventory (``func.inventory.run_func_inventory``), after
        :meth:`inventory`: a separate call so that the surface report keeps its numbers. It
        reads listings, sidecars, reports, transforms, masks and the headers of the series,
        never a volume. A run without a functional source, with ``--no-epi`` or of a census
        gives an empty report and builds no source; otherwise a source that cannot be built
        raises ``SourceError`` before anything is recorded."""
        from surfcensus.config import epi_settings
        from surfcensus.func.inventory import run_func_inventory

        if self.config.mode != "local" or not epi_settings(self.config).enabled:
            return PhaseReport(phase=FUNC_INVENTORY_PHASE, done=0, skipped=0, failed=0, failures=())
        return run_func_inventory(
            self._catalog, self._sources(), self.config, run_key(self.config), force=force
        )

    def pack(
        self,
        *,
        jobs: int = 1,
        sessions: Collection[str] | None = None,
        force: bool = False,
        retry_failed: bool = False,
        stop: Callable[[], bool] | None = None,
        waiting: Callable[[], None] | None = None,
    ) -> PhaseReport:
        """Pack the pending functional sessions into ``epi/`` (``func.batch.run_pack``), with
        the run's lease of a session bound into the context (:meth:`context`).

        ``sessions`` are ``session_uid``s. What is pending is decided from the catalog and the
        run directory (``func.batch.pack_pending``) before any source is looked at, so a run
        whose packs are all made needs no source and works while the data is not mounted.
        When something is pending, the FreeSurfer sources and the derivative of every
        functional source are looked for first, and a folder that is gone raises
        ``SourceError("not found", ...)`` before anything is recorded: a missing disk is not
        a property of its sessions. A census run is refused (``ConfigError("invalid", ...)``):
        the EPI phases do not run there. ``stop`` is that of :meth:`render`; ``waiting`` is
        called when Ctrl-C arrives while sessions are packed in worker processes, which are
        waited for (``batches.run_batches``)."""
        from surfcensus.config import epi_settings
        from surfcensus.func.batch import pack_pending, run_pack

        if self.config.mode != "local":
            raise ConfigError("invalid", "a census run has no EPI interface; nothing is packed")
        context = self.context()
        if not epi_settings(self.config).enabled:
            return run_pack(context, jobs=jobs, sessions=sessions)  # removes nothing, packs nothing
        pending = pack_pending(context, sessions=sessions, force=force, retry_failed=retry_failed)
        if pending:
            self._sources()  # a source that is gone is not a property of its sessions
            for root in self._func_roots():
                if not Path(root).is_dir():
                    raise SourceError(
                        "not found",
                        f"the derivative {root} of this run's functional sessions is not there",
                    )
        return run_pack(
            context,
            jobs=jobs,
            sessions=sessions,
            force=force,
            retry_failed=retry_failed,
            stop=stop,
            waiting=waiting,
        )

    def _func_roots(self) -> list[str]:
        """The derivative roots of the run's present functional sources, from the catalog."""
        rows = self._catalog.query(
            "SELECT deriv_root FROM func_sources WHERE present = 1 ORDER BY func_id"
        )
        return [row["deriv_root"] for row in rows]

    def func_folders(self) -> list[dict[str, object]]:
        """The present functional sources of a local run with their roots, for the lines that
        ``review`` and ``inventory`` print on the terminal of whoever has the data:
        ``deriv_root``, ``raw_root``, ``raw_found``, ``connector`` and ``pipeline_version``.
        Empty for a census run, whose roots stay in the catalog."""
        if self.config.mode != "local" or not self.public().has_func():
            return []
        rows = self._catalog.query(
            "SELECT deriv_root, raw_root, raw_found, connector, pipeline_version "
            "FROM func_sources WHERE present = 1 ORDER BY func_id"
        )
        return [dict(row) for row in rows]

    def func_order_lines(self) -> list[str]:
        """``func.inventory.func_order_lines``: which rule ordered the sessions of each
        participant with more than one (decision D42); empty in a census run."""
        from surfcensus.func.inventory import func_order_lines

        if self.config.mode != "local" or not self.public().has_func():
            return []
        return func_order_lines(self._catalog)

    def func_fetch_lines(self) -> list[str]:
        """``func.inventory.func_fetch_lines``: the functional runs with git-annex
        placeholders and the commands that fetch their files; empty in a census run."""
        from surfcensus.func.inventory import func_fetch_lines

        if self.config.mode != "local" or not self.public().has_func():
            return []
        return func_fetch_lines(self._catalog)

    def _relocate_func(self, dataset_id: int, dataset: DatasetRecord) -> None:
        """Move the functional source of a dataset that moved (``relocate``): the derivative
        and the raw dataset are found again by rule from the dataset at its new place and
        their roots replace the stored ones; nothing else changes, so every
        ``participant_uid``, ``session_uid`` and ``run_uid`` stays, since the derivative's
        identity is kept (decision D21). A raw dataset named with ``--raw`` keeps its name. A
        dataset whose functional part is not found at the new place is left as it is: the
        next functional inventory marks its sessions absent."""
        from surfcensus.config import epi_settings
        from surfcensus.func.discover import func_source_of

        if self.config.mode != "local":
            return
        stored = self._catalog.func_source(dataset_id)
        if stored is None:
            return
        raw = epi_settings(self.config).raw
        try:
            found = func_source_of(dataset, None if raw is None else Path(raw))
        except (SourceError, OSError):
            found = None
        if found is None or found.connector != stored.connector:
            return
        rows = self._catalog.query(
            "SELECT func_id FROM func_sources WHERE dataset_id = ? AND connector = ?",
            (dataset_id, stored.connector),
        )
        for row in rows:
            self._catalog.relocate_func(
                row["func_id"], found.deriv_root, found.raw_root, found.raw_found
            )

    def _subject_uids(self, names: Sequence[str]) -> list[str]:
        named = [names] if isinstance(names, str) else list(names)
        if self.config.mode != "local":
            return named
        rows = self._catalog.query("SELECT subject_uid, subject_label, present FROM subjects")
        uids = {row["subject_uid"] for row in rows}
        by_label: dict[str, list[str]] = {}
        for row in rows:
            if row["present"]:
                by_label.setdefault(row["subject_label"], []).append(row["subject_uid"])
        resolved = []
        for name in named:
            matches = by_label.get(name, [])
            if name not in uids and len(matches) > 1:
                raise ConfigError(
                    "invalid",
                    f"{name} names subjects of {len(matches)} datasets; use a subject_uid",
                )
            resolved.append(matches[0] if name not in uids and matches else name)
        return resolved

    def scores(self) -> pd.DataFrame:
        """The primary rows of ``tables/scores.parquet``, one per present subject, by
        ``primary_rank``: first the subjects scored against a reference, by rank, then those
        without one, by rank. In a local run the frame also has ``subject_label``, after
        ``subject_uid``. Empty before ``metrics`` has run."""
        import pandas as pd

        from surfcensus.store import primary_scores

        scores = primary_scores(self.run_dir)
        if self.config.mode == "local":
            subjects = self.public().subjects()
            label = dict(zip(subjects["subject_uid"], subjects["subject_label"], strict=True))
            labels = pd.Series([label.get(uid) for uid in scores["subject_uid"]], dtype=object)
            scores.insert(1, "subject_label", labels)
        return scores

    def clusters(self) -> pd.DataFrame:
        """``clusters.parquet`` left-joined with ``clusters_native.parquet``: every outlier
        cluster that was kept, with the native position of its centroid where a render
        gives one. Empty before ``metrics`` has found a reference."""
        from surfcensus.store import clusters_with_native

        return clusters_with_native(self.run_dir)

    def zmap(self, subject_uid: str, reference_id: str | None = None) -> dict[Hemi, np.ndarray]:
        """The subject's z-map per hemisphere on the template mesh, against ``reference_id``
        or, by default, the reference of its primary score. z-maps are not stored: this
        reads the subject's thickness map and the reference and computes
        ``metrics.fsaverage.zmap``.

        Raises ``KeyError`` for a subject that has no score against that reference,
        ``ConfigError("missing", ...)`` when the map or the reference is not on disk,
        ``ConfigError("invalid", ...)`` for a reference of the run that cannot be read (the
        next ``metrics`` builds it again), and ``ValueError``, naming the file, for a map
        that cannot be read or is not on the mesh of the template.
        """
        from surfcensus.metrics.fsaverage import global_median, read_maps, zmap
        from surfcensus.metrics.pipeline import REFERENCE_DIR, map_path, reference_dir
        from surfcensus.reference.load import read_reference, shipped_reference
        from surfcensus.reference.strata import SHIPPED_PREFIX
        from surfcensus.schemas import NO_REFERENCE, SCHEMAS
        from surfcensus.store import read_table, table_path

        scores = read_table(table_path(self.run_dir, "scores"), SCHEMAS["scores"])
        rows = scores[scores["subject_uid"] == subject_uid]
        if reference_id is None:
            rows = rows[rows["primary"].to_numpy(dtype=bool)]
        else:
            rows = rows[rows["reference_id"] == reference_id]
        if rows.empty or rows["reference_id"].iloc[0] == NO_REFERENCE:
            against = "a reference" if reference_id is None else reference_id
            raise KeyError(f"{subject_uid} has no z-map against {against}")
        reference_id, stratum = rows["reference_id"].iloc[0], rows["stratum"].iloc[0]
        template = self.template()
        if template is None:
            raise ConfigError("missing", "the run has no template")
        try:
            if reference_id.startswith(SHIPPED_PREFIX):
                reference = shipped_reference()
            else:
                try:
                    reference = read_reference(reference_dir(self.run_dir, reference_id))
                except ValueError as exc:
                    raise ConfigError(
                        "invalid",
                        f"the reference {reference_id} in {REFERENCE_DIR} cannot be read; "
                        "run `surfcensus metrics`, which builds it again",
                    ) from exc
            maps = read_maps(
                map_path(self.run_dir, subject_uid), len(template.hemis[HEMIS[0]].cortex)
            )
        except FileNotFoundError as exc:
            raise ConfigError(
                "missing", f"the map or the reference of {subject_uid} is not in the run"
            ) from exc
        if reference is None:
            raise ConfigError("missing", f"the run has no reference {reference_id}")
        return {
            hemi: zmap(
                maps[hemi],
                global_median(maps[hemi], template.hemis[hemi]),
                reference.strata[stratum][hemi],
            )
            for hemi in HEMIS
        }

    def public(self) -> PublicCatalog:
        """The public side of the catalog, on the run's connection."""
        return self._catalog.public()

    def context(self) -> RunContext:
        """What phase functions receive in place of the run. In a local run it carries the
        lease of a functional session (``func_lease``: ``func.inventory.session_work`` over
        what the private catalog knows of the session, with the EPI settings of this
        configuration) and ``pack_key`` with the run secret bound; a census run has neither,
        since the EPI phases do not run there."""
        catalog = self._catalog
        secret = catalog.run_secret()
        config = self.config
        func_lease = None
        if config.mode == "local":
            from surfcensus.config import epi_settings

            epi = epi_settings(config)

            @contextmanager
            def func_lease(session_uid: str) -> Iterator[SessionWork]:
                from surfcensus.func.inventory import session_work

                yield session_work(catalog.func_lease_facts(session_uid), epi)

        return RunContext(
            run_dir=self.run_dir,
            config=config,
            public=catalog.public(),
            lease=self._lease,
            render_key=lambda tile_hash, digests: privacy.render_key(secret, tile_hash, digests),
            scrub=lambda text: privacy.scrub_with(catalog, text),
            func_lease=func_lease,
            pack_key=lambda pack_hash, digests: privacy.pack_key(secret, pack_hash, digests),
        )

    def media_dir(self, subject_uid: str) -> Path:
        return media_dir(self.run_dir, subject_uid)

    def close(self) -> None:
        """Close the catalog, and give up the lease area of a remote source's cache when no
        lease of it is open (``Cache.release``), so that a call that ended leaves nothing
        under ``cache/leases/``."""
        for source in self._built.values():
            cache = getattr(source, "cache", None) if source.kind != "local" else None
            if cache is not None:
                cache.release()
        self._catalog.close()

    def __enter__(self) -> Run:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def free_bytes(path: Path) -> int | None:
    """The free space of the file system that holds ``path`` (or its nearest existing
    parent); None when it cannot be told."""
    import shutil

    place = Path(path)
    while not place.exists() and place.parent != place:
        place = place.parent
    try:
        return shutil.disk_usage(place).free
    except OSError:
        return None


def size_text(n: int) -> str:
    """A number of bytes as people read it: ``0 MB``, ``45 kB``, ``40.8 MB``, ``1.25 GB``
    (powers of ten)."""
    if n <= 0:
        return "0 MB"
    if n < 1_000_000:
        return f"{max(1, round(n / 1e3))} kB"
    if n < 1_000_000_000:
        return f"{n / 1e6:.1f} MB"
    return f"{n / 1e9:.2f} GB"


def from_config(
    config: SourceConfig,
    run_config: RunConfig,
    run_dir: Path,
    *,
    root: str | None = None,
    identity: str | None = None,
    transport: httpx.BaseTransport | None = None,
    secret: bytes | None = None,
) -> Source:
    """The source that ``config`` describes.

    ``root`` and ``identity`` override those of ``config``: a local source in a census run
    is stored without either, and both come from the catalog.

    A remote source (``openneuro-derivatives``) takes its client from ``run_config.remote``
    and ``transport`` and its cache from ``run_dir`` and ``secret``, the run secret, which
    it needs (``ConfigError("invalid", ...)`` without). Stored with a ``dataset`` (the local
    run on one dataset, decision D47, item 1), ``identity`` is the dataset's accession from
    the catalog: it must be given and must give the stored pseudonym with the run's salt
    (``ConfigError("invalid", ...)`` or ``ConfigError("key", ...)`` otherwise). Stored
    without one (a census run), the source lists every dataset of its prefix. The source
    is built without a request. ``openneuro-embedded`` is not built (O9).
    """
    if config.kind != "local":
        return _remote_from_config(config, run_config, run_dir, identity, transport, secret)
    from surfcensus.sources.local import LocalSource

    path = root if root is not None else config.root
    if path is None:
        raise ConfigError("invalid", "a local source needs a root")
    return LocalSource(
        path,
        subjects=config.subjects,
        identity=identity if identity is not None else config.identity,
        field_strength=config.field_strength,
        # a stored root is the folder that was found: it is not searched below derivatives/
        nested=False,
    )


def _one_dataset(source: Source) -> bool:
    """Whether ``source`` is a remote source of one dataset: what a local run on a remote
    dataset is made of."""
    return isinstance(source, RemoteSource) and source.accession is not None


def _remote_from_config(
    config: SourceConfig,
    run_config: RunConfig,
    run_dir: Path,
    accession: str | None,
    transport: httpx.BaseTransport | None,
    secret: bytes | None,
) -> Source:
    if config.kind != "openneuro-derivatives":
        raise NotImplementedError(f"the {config.kind} source is not built (decision O9)")
    if secret is None:
        raise ConfigError("invalid", "a remote source needs the run secret for its cache")
    if config.dataset is not None:
        if accession is None:
            raise ConfigError(
                "invalid", f"the accession of {config.dataset} is in {CATALOG_FILE} only"
            )
        expected = privacy.dataset_pseudonym(run_key(run_config), config.kind, accession)
        if expected != config.dataset:
            raise ConfigError("key", f"the dataset given is not {config.dataset} of {CONFIG_FILE}")
    from surfcensus.remote import FMRIPREP_PREFIX
    from surfcensus.remote.cache import CACHE_DIR, Cache
    from surfcensus.remote.s3 import S3Client
    from surfcensus.sources.openneuro_derivatives import OpenNeuroDerivatives

    assert config.bucket is not None  # SourceConfig checks it
    client = S3Client(config.bucket, config=run_config.remote, transport=transport)
    cache = Cache(
        run_dir / CACHE_DIR,
        max_bytes=run_config.cache.max_bytes,
        secret=secret,
        retain_roles=run_config.cache.retain_roles,
    )
    return OpenNeuroDerivatives(
        client=client,
        cache=cache,
        prefix=config.prefix or FMRIPREP_PREFIX,
        accession=accession if config.dataset is not None else None,
        subjects=config.subjects,
        pseudonym=config.dataset,
    )
