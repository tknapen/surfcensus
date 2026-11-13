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
from collections.abc import Collection, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
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
    run_key,
    with_changes,
    write_config,
)
from surfcensus.log import get_logger
from surfcensus.privacy import HashKey
from surfcensus.records import METRIC_STAGES, PhaseReport, RunContext
from surfcensus.sources.base import HEMIS, DatasetRecord, FileRole, Hemi, Source, SourceError
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

    from surfcensus.templates import Template, TemplateNotFound

_PER_SUBJECT_STAGES = ("regional", "fsaverage")


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

    A run directory inside the subjects directory of a local source is refused
    (``ConfigError("invalid", ...)``): a source is only read.
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
    ) -> None:
        self.run_dir = Path(run_dir)
        self._key = key
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
                given = tuple(
                    from_config(stored, config, self.run_dir, transport=self._transport)
                    for stored in config.sources
                )
            base = config if config is not None else new_local_config(given[0].config().root or ".")
            hash_key = run_key(base)
        stored = []
        for source in given:
            self._refuse_inside(source)
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
            if census and described.kind == "local":
                # the root of a local source of a census run is in the catalog only
                assert described.dataset is not None
                for dataset in source.datasets():
                    self._catalog.upsert_dataset(dataset, described.dataset, None)

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
            catalog = Catalog(path)
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
        described = source.config()
        if described.kind != "local":
            matches = [i for i, s in enumerate(self.config.sources) if s == described]
            if not matches:
                raise ConfigError("invalid", f"this run has no such {described.kind} source")
            self._built[matches[0]] = source
            return
        locals_ = self._stored_locals()
        match = next((s for s in locals_ if s.root == described.root), None)
        moved = match is None
        if match is None:
            match = self._relocation_target(locals_, relocate)
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
                earlier = [dataset_id for dataset_id in ids if dataset_id not in found]
                if found and earlier:
                    with self._catalog.write() as conn:
                        conn.executemany(
                            "UPDATE datasets SET root = ? WHERE dataset_id = ?",
                            [(dataset.root, dataset_id) for dataset_id in earlier],
                        )

    def _relocation_target(
        self, locals_: Sequence[_StoredLocal], relocate: bool | str
    ) -> _StoredLocal:
        if not locals_:
            raise ConfigError("invalid", "this run has no local source")
        if not relocate:
            # a census run keeps its roots out of sight: name its datasets by pseudonym
            local = self.config.mode == "local"
            old = ", ".join(s.root if local else s.pseudonym for s in locals_)
            raise ConfigError(
                "other root",
                f"this run directory belongs to {old}; pass --run-dir for a new run, or "
                "--relocate if the data moved",
            )
        if isinstance(relocate, str):
            named = [s for s in locals_ if s.pseudonym == relocate]
            if not named:
                raise ConfigError("invalid", f"this run has no local dataset {relocate}")
            return named[0]
        if len(locals_) > 1:
            names = ", ".join(sorted(s.pseudonym for s in locals_))
            raise ConfigError(
                "invalid",
                f"this run has {len(locals_)} local sources ({names}); say which one moved: "
                "--relocate D-xxxxxxxx",
            )
        return locals_[0]

    def _refuse_inside(self, source: Source) -> None:
        subjects_dir = getattr(source, "subjects_dir", None)
        if not isinstance(subjects_dir, Path):
            return
        if self.run_dir.resolve().is_relative_to(subjects_dir.resolve()):
            raise ConfigError(
                "invalid",
                f"the run directory {self.run_dir} lies inside the subjects directory "
                f"{subjects_dir}, which is only read; choose a run directory outside it",
            )

    # sources

    def _source(self, index: int, *, root: str | None = None) -> Source:
        """The source of ``config.sources[index]``, built on first use."""
        if index not in self._built:
            stored = self.config.sources[index]
            identity = None
            if stored.kind == "local":
                local = next(s for s in self._stored_locals() if s.index == index)
                root, identity = root or local.root, local.identity
            source = from_config(
                stored,
                self.config,
                self.run_dir,
                root=root,
                identity=identity,
                transport=self._transport,
            )
            self._refuse_inside(source)
            self._built[index] = source
        return self._built[index]

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
        the first dataset, in pseudonym order, that has one. The first use stores the
        template under ``templates/<template_id>/``, records it in the catalog and pins the
        run to it (``meta.template_id``); from then on the run keeps that template whatever
        the environment looks like. ``retemplate`` looks again and pins what it finds.

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
        cache = self.run_dir / "templates"
        adjacent, adjacent_id = self._adjacent_template()
        try:
            found = find_template(
                config=self.config.metrics,
                adjacent=adjacent,
                cache_dir=cache,
                pinned=pinned,
                retemplate=retemplate,
            )
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
        """The ``fsaverage`` next to the subjects of the first dataset, in pseudonym order,
        that the inventory found one for, and that dataset's row."""
        rows = self._catalog.query(
            "SELECT dataset_id, source, root, template_location FROM datasets "
            "WHERE template_location IS NOT NULL ORDER BY pseudonym, source, pipeline"
        )
        for row in rows:
            if row["source"] == "local" and row["root"] is not None:
                return Path(row["root"]) / row["template_location"], row["dataset_id"]
        return None, None

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
        from surfcensus.metrics.pipeline import metrics_pending, run_metrics

        pseudonyms = self._pseudonyms(datasets)
        if stages is None:
            stages = METRIC_STAGES if datasets is None else _PER_SUBJECT_STAGES
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
        return run_metrics(
            context,
            template=template,
            datasets=pseudonyms,
            stages=stages,
            jobs=jobs,
            force=force,
            retry_failed=retry_failed,
        )

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
        report = run_render(
            context,
            template=template,
            jobs=jobs,
            datasets=pseudonyms,
            subjects=uids,
            all_subjects=all_subjects,
            video=video,
            force=force,
            retry_failed=retry_failed,
        )
        # the cluster centroids of the subjects just rendered get their native position
        native = update_native(context, template=template)
        return dataclasses.replace(report, notes=(*report.notes, *native.notes))

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
        """The primary rows of ``tables/scores.parquet``, one per present subject: first the
        subjects scored against a reference, by rank, then those without one, by rank.
        Empty before ``metrics`` has run."""
        from surfcensus.schemas import NO_REFERENCE, SCHEMAS
        from surfcensus.store import read_table, table_path

        scores = read_table(table_path(self.run_dir, "scores"), SCHEMAS["scores"])
        scores = scores[scores["primary"].to_numpy(dtype=bool)]
        order = scores.assign(unreferenced=scores["reference_id"] == NO_REFERENCE)
        order = order.sort_values(["unreferenced", "rank", "subject_uid"])
        return order.drop(columns="unreferenced").reset_index(drop=True)

    def clusters(self) -> pd.DataFrame:
        """``clusters.parquet`` left-joined with ``clusters_native.parquet``: every outlier
        cluster that was kept, with the native position of its centroid where a render
        gives one. Empty before ``metrics`` has found a reference."""
        from surfcensus.schemas import KEYS, SCHEMAS
        from surfcensus.store import read_table, table_path

        clusters = read_table(table_path(self.run_dir, "clusters"), SCHEMAS["clusters"])
        native = read_table(table_path(self.run_dir, "clusters_native"), SCHEMAS["clusters_native"])
        return clusters.merge(native, how="left", on=list(KEYS["clusters"]))

    def zmap(self, subject_uid: str, reference_id: str | None = None) -> dict[Hemi, np.ndarray]:
        """The subject's z-map per hemisphere on the template mesh, against ``reference_id``
        or, by default, the reference of its primary score. z-maps are not stored: this
        reads the subject's thickness map and the reference and computes
        ``metrics.fsaverage.zmap``.

        Raises ``KeyError`` for a subject that has no score against that reference, and
        ``ConfigError("missing", ...)`` when the map or the reference is not on disk.
        """
        from surfcensus.metrics.fsaverage import global_median, read_maps, zmap
        from surfcensus.metrics.pipeline import map_path, reference_dir
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
        try:
            if reference_id.startswith(SHIPPED_PREFIX):
                reference = shipped_reference()
            else:
                reference = read_reference(reference_dir(self.run_dir, reference_id))
            maps = read_maps(map_path(self.run_dir, subject_uid))
        except FileNotFoundError as exc:
            raise ConfigError(
                "missing", f"the map or the reference of {subject_uid} is not in the run"
            ) from exc
        if template is None or reference is None:
            raise ConfigError("missing", f"the run has no template or no reference {reference_id}")
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
        """What phase functions receive in place of the run."""
        catalog = self._catalog
        secret = catalog.run_secret()
        return RunContext(
            run_dir=self.run_dir,
            config=self.config,
            public=catalog.public(),
            lease=self._lease,
            render_key=lambda tile_hash, digests: privacy.render_key(secret, tile_hash, digests),
            scrub=lambda text: privacy.scrub_with(catalog, text),
        )

    def media_dir(self, subject_uid: str) -> Path:
        return media_dir(self.run_dir, subject_uid)

    def close(self) -> None:
        self._catalog.close()

    def __enter__(self) -> Run:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def from_config(
    config: SourceConfig,
    run_config: RunConfig,
    run_dir: Path,
    *,
    root: str | None = None,
    identity: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> Source:
    """The source that ``config`` describes.

    ``root`` and ``identity`` override those of ``config``: a local source in a census run
    is stored without either, and both come from the catalog. The remote kinds arrive in
    milestone 5; they take their client and cache from ``run_config``, ``run_dir`` and
    ``transport``.
    """
    if config.kind != "local":
        raise NotImplementedError(f"the {config.kind} source arrives in milestone 5")
    from surfcensus.sources.local import LocalSource

    path = root if root is not None else config.root
    if path is None:
        raise ConfigError("invalid", "a local source needs a root")
    return LocalSource(
        path,
        subjects=config.subjects,
        identity=identity if identity is not None else config.identity,
        field_strength=config.field_strength,
    )
