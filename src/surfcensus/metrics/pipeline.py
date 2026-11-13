"""Phase 4 for a run: the metric stages, their skip tests and their tables. See
docs/phases.md.

Two stages work per subject, on leased files and batch by batch (``batches.run_batches``):
``regional`` (the parcel rows of the stats files) and ``fsaverage`` (the thickness map on
the template mesh, the resampling facts and the ceiling counts). Their rows go to the part of
the subject's dataset under ``tables/_parts/``. The other stages work on the run and read no
source file: ``reference`` (the per-vertex reference of every stratum and the outlier
clusters of every subject scored against it), ``score`` (one row per subject and reference),
``native`` (cluster centroids on the native surface, from the vertex maps of the renders) and
``merge``.

A unit is skipped when its ledger row says ``done`` for the stage's present identity and
input signature and its output is there. Files are written first and the ledger row last.
The two stages that read source files store the SHA-256 of each file they read in the
catalog, so that a row traces to the bytes it was computed from. For the same reason they
remove what they hold of a subject that no longer has their files.

An output of the run that is damaged is not a failure of a subject: a map that cannot be
read stops the phase and is removed, to be made again; a vertex map that cannot be read is
named in a note.

A phase function receives a ``RunContext`` and never the run, so it names subjects by
``subject_uid`` and datasets by pseudonym.
"""

from __future__ import annotations

import hashlib
import math
import shutil
import time
import zipfile
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa

from surfcensus._io import canonical_json, read_npz, sha256_file
from surfcensus.batches import WorkerError, run_batches
from surfcensus.config import ConfigError, MetricsConfig, stage_hashes
from surfcensus.fsio.annot import read_annotation
from surfcensus.fsio.morph import read_morph
from surfcensus.fsio.stats import read_stats
from surfcensus.fsio.surfaces import read_surface
from surfcensus.log import get_logger
from surfcensus.metrics import MetricError, scoring
from surfcensus.metrics.ceiling import ceiling_counts, native_region, thickness_above_max
from surfcensus.metrics.clusters import find_clusters, native_centroids
from surfcensus.metrics.fsaverage import (
    assign_strata,
    build_reference,
    global_median,
    read_maps,
    resample_hemi,
    stratum_members,
    write_maps,
    zmap,
)
from surfcensus.metrics.regional import regional_rows
from surfcensus.records import (
    ALL_METRIC_STAGES,
    METRIC_STAGES,
    PhaseReport,
    RunContext,
    WorkRow,
)
from surfcensus.reference import load
from surfcensus.reference.load import (
    INDEX_JSON,
    Reference,
    ReferenceMaps,
    applicable_stratum,
    read_index,
    shipped_reference,
    stratum_size,
    write_reference,
)
from surfcensus.reference.strata import SET_PREFIX, members_sha256, set_reference_id
from surfcensus.schemas import (
    CLUSTERS,
    CLUSTERS_NATIVE,
    FLAG_OF_REASON,
    KEYS,
    SCHEMAS,
    SCORES,
    VERTEX_SUMMARY,
    intensity_schema,
    join_flags,
    split_flags,
)
from surfcensus.sources.base import HEMIS, TIER_B, FileRole, Hemi, SourceError
from surfcensus.store import (
    media_dir,
    merge_parts,
    part_path,
    read_table,
    table_path,
    upsert_part,
    write_table,
)
from surfcensus.templates import (
    REGION_NAMES,
    TEMPLATES_DIR,
    Template,
    load_template,
    store_template,
)

PHASE = "metrics"
PER_SUBJECT_STAGES: tuple[str, ...] = ("regional", "fsaverage")
"""The stages that read source files, one subject at a time; the only ones that a dataset
filter restricts."""
RUN_UNIT = "run"
"""The one unit of a stage that works on the whole run."""
FSAVG_DIR = "tables/fsavg"
REFERENCE_DIR = "tables/reference"
SOURCE_CHANGED = "source changed"
METRIC_FAILURE = "metric failure"
STALE_REFERENCE = "reference not current"
PART_TABLES: tuple[str, ...] = ("regional", "vertex_summary", "intensity")
"""The tables that are written as one part per dataset."""

_STATS: dict[Hemi, FileRole] = {"lh": FileRole.LH_STATS, "rh": FileRole.RH_STATS}
_THICKNESS: dict[Hemi, FileRole] = {"lh": FileRole.LH_THICKNESS, "rh": FileRole.RH_THICKNESS}
_SPHERE: dict[Hemi, FileRole] = {"lh": FileRole.LH_SPHERE_REG, "rh": FileRole.RH_SPHERE_REG}
_ANNOT: dict[Hemi, FileRole] = {"lh": FileRole.LH_ANNOT, "rh": FileRole.RH_ANNOT}
_REGIONAL_ROLES: tuple[FileRole, ...] = tuple(_STATS.values())
_VERTEX_MAP = "vertex_map.npz"
_SUBJECT_VALUES = (
    "subject_uid",
    "dataset",
    "field_strength",
    "fs_major",
    "holes_lh",
    "holes_rh",
    "trunc_lh_1",
    "trunc_lh_2",
    "trunc_rh_1",
    "trunc_rh_2",
)
"""What the scores take from the catalog; part of the input signature of the ``score`` unit."""
_MAX_NAMED = 5
_NULLABLE_INTS = {pa.int8(): "Int8", pa.int16(): "Int16", pa.int32(): "Int32"}


def map_path(run_dir: Path | str, subject_uid: str) -> Path:
    """``tables/fsavg/<subject_uid>.npz`` of a run directory: the subject's thickness maps."""
    return Path(run_dir) / FSAVG_DIR / f"{subject_uid}.npz"


def reference_dir(run_dir: Path | str, reference_id: str) -> Path:
    """``tables/reference/<reference_id>`` of a run directory."""
    return Path(run_dir) / REFERENCE_DIR / reference_id


def frame_of(rows: Sequence[Mapping[str, object]], schema: pa.Schema) -> pd.DataFrame:
    """``rows`` as a frame in the column types that a table of ``schema`` is read back in:
    text as ``object`` with None, floats as float32 with NaN for a missing value, integers
    that may be missing as pandas nullable integers."""
    data: dict[str, object] = {}
    for column in schema:
        values = [row[column.name] for row in rows]
        if pa.types.is_string(column.type):
            data[column.name] = pd.Series(values, dtype=object)
        elif pa.types.is_floating(column.type):
            floats = [math.nan if value is None else float(value) for value in values]  # pyrefly: ignore[bad-argument-type]
            data[column.name] = np.asarray(floats, dtype=np.float32)
        elif pa.types.is_boolean(column.type):
            data[column.name] = np.asarray(values, dtype=bool)
        elif column.nullable:
            data[column.name] = pd.array(values, dtype=_NULLABLE_INTS[column.type])
        else:
            data[column.name] = np.asarray(values, dtype=column.type.to_pandas_dtype())
    return pd.DataFrame(data)


# --- what a call did ------------------------------------------------------------------------


@dataclass
class _Tally:
    done: int = 0
    skipped: int = 0
    failures: dict[tuple[str, str], str] = field(default_factory=dict)
    """``(stage, unit)`` -> reason, for every unit in the failed state."""
    notes: list[str] = field(default_factory=list)

    def report(self, phase: str, started: float) -> PhaseReport:
        failures = tuple(
            (unit, f"{stage}: {reason}") for (stage, unit), reason in sorted(self.failures.items())
        )
        return PhaseReport(
            phase=phase,
            done=self.done,
            skipped=self.skipped,
            failed=len(failures),
            failures=failures,
            notes=tuple(dict.fromkeys(self.notes)),
            seconds=time.perf_counter() - started,
        )


def _failure(exc: BaseException) -> tuple[str, str]:
    """Reason and unscrubbed detail of the ledger row for an exception of a lease or of the
    work on one subject."""
    if isinstance(exc, MetricError):
        return exc.reason, exc.detail
    if isinstance(exc, SourceError):
        reasons = {"changed": SOURCE_CHANGED, "not found": "missing file"}
        return reasons.get(exc.reason, "unreadable"), exc.detail
    if isinstance(exc, WorkerError):  # carries the name of the class that was raised
        return METRIC_FAILURE, str(exc)
    return METRIC_FAILURE, f"{type(exc).__name__}: {exc}"


def _signature(lines: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16]


def _some(uids: Sequence[str]) -> str:
    """The first of ``uids`` by name, and how many more there are."""
    more = f" and {len(uids) - _MAX_NAMED} more" if len(uids) > _MAX_NAMED else ""
    return ", ".join(uids[:_MAX_NAMED]) + more


# --- the two stages that read source files --------------------------------------------------


@dataclass(frozen=True)
class _Outcome:
    """What the work on one subject hands back to the process that records it."""

    frame: pd.DataFrame
    digests: dict[FileRole, str]
    """SHA-256 of every source file that the rows were computed from; each is read whole."""
    has_map: bool = False


@dataclass(frozen=True)
class _Stage:
    """A per-subject stage of one call."""

    name: str
    table: str
    roles: tuple[FileRole, ...]
    config_hash: str
    units: list[str]
    work: Callable[[str, Mapping[FileRole, Path]], _Outcome]
    left: list[str]
    """The present subjects of the datasets that are not units: they have none of the files
    that make a subject a unit of the stage, and may have had them at an earlier call."""


@dataclass
class _Plan:
    signatures: dict[str, str]
    pending: list[str]
    skipped: int
    failures: dict[str, str]


def _recorded(ctx: RunContext, stage: str) -> dict[str, tuple[str | None, str | None]]:
    """Unit -> (reason, output) of the stage's ledger rows."""
    frame = ctx.public.work(stage)
    return {
        unit: (reason, output)
        for unit, reason, output in zip(
            frame["unit"], frame["reason"], frame["output"], strict=True
        )
    }


def _plan(ctx: RunContext, stage: _Stage, *, force: bool, retry_failed: bool) -> _Plan:
    """Decide from the ledger and the listing in the catalog which subjects are pending.
    Opens no source file and writes nothing."""
    public = ctx.public
    recorded = _recorded(ctx, stage.name)
    plan = _Plan({}, [], 0, {})
    for uid in stage.units:
        signature = public.input_signature(uid, stage.roles)
        plan.signatures[uid] = signature
        state = None if force else public.work_done(stage.name, uid, stage.config_hash, signature)
        reason, output = recorded.get(uid, (None, None))
        if state is False and not retry_failed:
            plan.failures[uid] = reason or METRIC_FAILURE
        elif state is True and _output_there(ctx, stage, uid, output):
            plan.skipped += 1
        else:
            plan.pending.append(uid)
    return plan


def _output_there(ctx: RunContext, stage: _Stage, uid: str, output: str | None) -> bool:
    if isinstance(output, str):
        return (Path(ctx.run_dir) / output).is_file()
    return part_path(ctx.run_dir, stage.table, ctx.public.dataset_of(uid)).is_file()


def _run_stage(
    ctx: RunContext, stage: _Stage, tally: _Tally, *, jobs: int, force: bool, retry_failed: bool
) -> None:
    public = ctx.public
    plan = _plan(ctx, stage, force=force, retry_failed=retry_failed)
    tally.skipped += plan.skipped
    tally.failures.update({(stage.name, uid): reason for uid, reason in plan.failures.items()})
    empty = SCHEMAS[stage.table].empty_table().to_pandas()
    _drop_left(ctx, stage, empty)

    def commit(outcomes: Sequence[tuple[str, object]]) -> None:
        touched: dict[str, list[str]] = {}
        frames: dict[str, list[pd.DataFrame]] = {}
        digests: dict[str, dict[FileRole, str]] = {}
        rows: list[WorkRow] = []
        for uid, outcome in outcomes:
            dataset = public.dataset_of(uid)
            touched.setdefault(dataset, []).append(uid)
            output = None
            if isinstance(outcome, BaseException):
                reason, detail = _failure(outcome)
                status, text = "failed", ctx.scrub(detail)
                tally.failures[(stage.name, uid)] = reason
                has_map = False
            else:
                result = cast(_Outcome, outcome)
                has_map = result.has_map
                frames.setdefault(dataset, []).append(result.frame)
                digests[uid] = result.digests
                status, reason, text = "done", None, None
                tally.failures.pop((stage.name, uid), None)
                tally.done += 1
                if has_map:
                    output = f"{FSAVG_DIR}/{uid}.npz"
            if stage.name == "fsaverage" and not has_map:
                # the map of an earlier state of the subject must not pass for a present one
                map_path(ctx.run_dir, uid).unlink(missing_ok=True)
            rows.append(
                WorkRow(
                    phase=stage.name,
                    unit=uid,
                    config_hash=stage.config_hash,
                    input_signature=plan.signatures[uid],
                    status=status,
                    reason=reason,
                    detail=text,
                    output=output,
                )
            )
        for dataset in sorted(touched):
            parts = [frame for frame in frames.get(dataset, []) if len(frame)]
            upsert_part(
                ctx.run_dir,
                stage.table,
                dataset,
                pd.concat(parts, ignore_index=True) if parts else empty,
                replace_uids=touched[dataset],
            )
        # Before the ledger rows: a unit that is done is skipped from then on, so digests
        # written after its row could be lost for good between the two.
        for uid, read in digests.items():
            public.set_file_digests(uid, read)
        public.record_work(rows)  # last: a row never stands for rows that are not written

    run_batches(
        plan.pending,
        lambda uid: ctx.lease(uid, stage.roles),
        stage.work,
        commit,
        batch_subjects=ctx.config.cache.batch_subjects,
        jobs=jobs,
        progress=stage.name if plan.pending else None,
    )


def _drop_left(ctx: RunContext, stage: _Stage, empty: pd.DataFrame) -> None:
    """Remove what the stage holds of the subjects that are no longer its units: their rows
    in the part of their dataset, their map and their ledger row. A subject whose stats or
    thickness files are gone would otherwise be scored on the rows of files that no longer
    exist. Reads no source file. The ledger rows go last, as they are written last: a call
    that is interrupted finds them and removes the rest again."""
    if not stage.left:
        return
    public = ctx.public
    by_dataset: dict[str, list[str]] = {}
    for uid in stage.left:
        by_dataset.setdefault(public.dataset_of(uid), []).append(uid)
        if stage.name == "fsaverage":
            map_path(ctx.run_dir, uid).unlink(missing_ok=True)
    dropped: set[str] = set()
    for dataset, uids in sorted(by_dataset.items()):
        path = part_path(ctx.run_dir, stage.table, dataset)
        if not path.is_file():
            continue
        held = set(read_table(path, SCHEMAS[stage.table])["subject_uid"]) & set(uids)
        if held:
            upsert_part(ctx.run_dir, stage.table, dataset, empty, replace_uids=sorted(held))
            dropped |= held
    recorded = set(_recorded(ctx, stage.name)) & set(stage.left)
    if recorded:
        public.forget_work(stage.name, recorded)
    if dropped or recorded:
        get_logger().info(
            "%s: %d subject(s) no longer have the files of this stage; what it held of them "
            "was removed",
            stage.name,
            len(dropped | recorded),
        )


def _unreadable(path: Path, exc: Exception) -> MetricError:
    """A source file that is there and cannot be read fails its subject. The text of an
    ``OSError`` holds the path, so only its reason is kept."""
    if isinstance(exc, OSError):
        return MetricError("unreadable", f"{path.name}: {exc.strerror or type(exc).__name__}")
    text = " ".join(str(exc).split())
    return MetricError("unreadable", text if text.startswith(path.name) else f"{path.name}: {text}")


def _read(
    reader: Callable[[Path], Any],
    files: Mapping[FileRole, Path],
    role: FileRole,
    digests: dict[FileRole, str],
) -> Any:
    """The content of the subject's file of ``role`` and, in ``digests``, the SHA-256 of its
    bytes: size and modification time of a local file do not say which bytes were parsed."""
    path = files[role]
    try:
        digests[role] = sha256_file(path)
        return reader(path)
    except (OSError, ValueError) as exc:
        raise _unreadable(path, exc) from exc


def _regional_one(uid: str, files: Mapping[FileRole, Path]) -> _Outcome:
    """What a worker does for one subject in the ``regional`` stage: the parcel rows of the
    stats files it has, one hemisphere's when the other file is missing. Top-level, so that
    it can be sent to a process."""
    frames = []
    digests: dict[FileRole, str] = {}
    for hemi in HEMIS:
        role = _STATS[hemi]
        if role not in files:
            continue
        stats = _read(read_stats, files, role, digests)
        try:
            frames.append(regional_rows(uid, hemi, stats))
        except ValueError as exc:
            raise _unreadable(files[role], exc) from exc
    if not frames:
        return _Outcome(SCHEMAS["regional"].empty_table().to_pandas(), digests)
    return _Outcome(pd.concat(frames, ignore_index=True), digests)


def _fraction(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _hemisphere(
    uid: str,
    hemi: Hemi,
    files: Mapping[FileRole, Path],
    template: Template | None,
    config: MetricsConfig,
    digests: dict[FileRole, str],
) -> tuple[dict[str, object] | None, np.ndarray | None]:
    """The row of ``vertex_summary`` of one hemisphere and, when it has one, its map. The
    digests of the files it read are added to ``digests``."""
    thickness_file = files.get(_THICKNESS[hemi])
    if thickness_file is None:
        return None, None
    thickness = _read(read_morph, files, _THICKNESS[hemi], digests)
    finite = thickness[np.isfinite(thickness)]
    if finite.size == 0:
        raise MetricError("unreadable", f"{thickness_file.name}: no thickness value")
    flags: set[str] = set()
    annot = None
    if _ANNOT[hemi] in files:
        annot = _read(read_annotation, files, _ANNOT[hemi], digests)
    else:
        flags.add("no_annot")
    if thickness_above_max(thickness, max_mm=config.thickness_max_mm):
        flags.add("thickness_above_max")

    hemi_map = smap = None
    if template is None:
        flags.add("no_template")
    elif _SPHERE[hemi] not in files:
        flags.add("no_sphere_reg")
    else:
        sphere = _read(read_surface, files, _SPHERE[hemi], digests)
        try:
            hemi_map, smap = resample_hemi(
                thickness, sphere, template.hemis[hemi], annot=annot, config=config
            )
        except MetricError as exc:
            if exc.reason not in FLAG_OF_REASON:
                raise
            flags.add(FLAG_OF_REASON[exc.reason])
            get_logger().info("%s %s: no map (%s)", uid, hemi, exc)
        else:
            if hemi_map.flipped:
                flags.add("sphere_flipped")
    if annot is not None and len(annot.labels) != len(thickness):
        flags.add("vertex_count_mismatch")
        annot = None

    row: dict[str, object] = dict.fromkeys(VERTEX_SUMMARY.names)
    row.update(
        subject_uid=uid,
        hemi=hemi,
        n_native=len(thickness),
        thickness_max_mm=float(finite.max()),
    )
    if hemi_map is not None and template is not None:
        median = hemi_map.global_median
        row.update(
            template_id=template.template_id,
            n_fallback=hemi_map.n_fallback,
            n_missing_cortex=hemi_map.n_missing_cortex,
            global_median_mm=median if math.isfinite(median) else None,
        )
    limits = {"max_mm": config.thickness_max_mm, "margin_mm": config.ceiling_margin_mm}
    template_hemi = None if smap is None or template is None else template.hemis[hemi]
    kind = "none"
    for name in ("search", "control"):
        region, found = native_region(
            name, annot=annot, smap=smap, template_hemi=template_hemi, regions=config.regions
        )
        if region is None:
            continue
        counts = ceiling_counts(thickness, region, **limits)
        row.update(
            {
                f"{name}_n": counts.n_region,
                f"{name}_n_exact": counts.n_exact,
                f"{name}_n_near": counts.n_near,
                f"{name}_n_zero": counts.n_zero,
            }
        )
        fraction = _fraction(counts.n_near + counts.n_zero, counts.n_region)
        if name == "search":
            kind = found
            row["ceiling_frac"] = fraction
        else:
            row["control_ceiling_frac"] = fraction
    row["region_kind"] = kind
    if annot is not None:
        # the same count on the subject's own occipital parcels, whether or not it has a map
        region, _ = native_region(
            "search", annot=annot, smap=None, template_hemi=None, regions=config.regions
        )
        if region is not None:
            counts = ceiling_counts(thickness, region, **limits)
            row.update(
                annot_search_n=counts.n_region,
                annot_search_n_near=counts.n_near,
                annot_search_n_zero=counts.n_zero,
                ceiling_frac_annot=_fraction(counts.n_near + counts.n_zero, counts.n_region),
            )
    row["flags"] = join_flags(flags)
    return row, None if hemi_map is None else hemi_map.thickness


def _fsaverage_one(
    uid: str,
    files: Mapping[FileRole, Path],
    *,
    template_dir: str | None,
    config: MetricsConfig,
    out: str,
) -> _Outcome:
    """What a worker does for one subject in the ``fsaverage`` stage: the rows of
    ``vertex_summary`` and, written here, the map file. Top-level, so that it can be sent to
    a process."""
    template = None if template_dir is None else load_template(template_dir)
    rows = []
    maps: dict[Hemi, np.ndarray] = {}
    digests: dict[FileRole, str] = {}
    for hemi in HEMIS:
        row, values = _hemisphere(uid, hemi, files, template, config, digests)
        if row is not None:
            rows.append(row)
        if values is not None:
            maps[hemi] = values
    if maps:
        path = Path(out) / f"{uid}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_maps(path, maps)
    return _Outcome(frame_of(rows, VERTEX_SUMMARY), digests, has_map=bool(maps))


def _units(
    ctx: RunContext, datasets: Collection[str] | None, any_of: Collection[FileRole]
) -> tuple[list[str], list[str]]:
    """The present subjects of ``datasets`` that have at least one file of ``any_of``, and
    those that have none."""
    wanted = set(any_of)
    units: list[str] = []
    left: list[str] = []
    for uid in ctx.public.subject_uids(datasets=datasets):
        (units if wanted & set(ctx.public.roles(uid)) else left).append(uid)
    return units, left


def _stages(
    ctx: RunContext,
    template: Template | None,
    datasets: Collection[str] | None,
    stages: Collection[str],
    *,
    for_work: bool,
) -> list[_Stage]:
    config = ctx.config
    hashes = stage_hashes(config, template_id=None if template is None else template.template_id)
    found = []
    if "regional" in stages:
        # one stats file is enough (docs/metrics.md); ``tier_a`` asks for both
        units, left = _units(ctx, datasets, _REGIONAL_ROLES)
        found.append(
            _Stage(
                "regional",
                "regional",
                _REGIONAL_ROLES,
                hashes["regional"],
                units,
                _regional_one,
                left,
            )
        )
    if "fsaverage" in stages:
        units, left = _units(ctx, datasets, tuple(_THICKNESS.values()))
        template_dir = None
        if template is not None and for_work:
            # where the workers read the template from
            template_dir = str(store_template(Path(ctx.run_dir) / TEMPLATES_DIR, template))
        work = partial(
            _fsaverage_one,
            template_dir=template_dir,
            config=config.metrics,
            out=str(Path(ctx.run_dir) / FSAVG_DIR),
        )
        found.append(
            _Stage("fsaverage", "vertex_summary", TIER_B, hashes["fsaverage"], units, work, left)
        )
    return found


def _checked_stages(stages: Collection[str]) -> list[str]:
    named = [stages] if isinstance(stages, str) else list(stages)
    unknown = [name for name in named if name not in ALL_METRIC_STAGES]
    if unknown:
        raise ConfigError(
            "invalid",
            f"no metric stage {', '.join(unknown)}; the stages are {', '.join(ALL_METRIC_STAGES)}",
        )
    return [name for name in ALL_METRIC_STAGES if name in named]


def metrics_pending(
    ctx: RunContext,
    *,
    template: Template | None,
    datasets: Collection[str] | None = None,
    stages: Collection[str] = METRIC_STAGES,
    force: bool = False,
    retry_failed: bool = False,
) -> int:
    """How many subjects ``run_metrics`` would read source files of when called with the same
    arguments. Opens no source file and records nothing; zero means that the call needs no
    source."""
    wanted = _checked_stages(stages)
    return sum(
        len(_plan(ctx, stage, force=force, retry_failed=retry_failed).pending)
        for stage in _stages(ctx, template, datasets, wanted, for_work=False)
    )


def run_metrics(
    ctx: RunContext,
    *,
    template: Template | None,
    datasets: Collection[str] | None = None,
    stages: Collection[str] = METRIC_STAGES,
    jobs: int = 1,
    force: bool = False,
    retry_failed: bool = False,
) -> PhaseReport:
    """Run the metric stages named in ``stages`` (names from ``records.ALL_METRIC_STAGES``)
    on what is pending, in the order of that list whatever the order given.

    ``datasets`` (pseudonyms) restricts the two per-subject stages. The run-level stages use
    every present subject and ignore it. ``template`` is the run's template, or None for a
    run without one: no map is made then, no reference is built and the scores are the
    ranking of decision D16.

    ``regional``: every subject with a stats file, of one hemisphere or of both.
    ``fsaverage``: every subject with a thickness file. Each stores the SHA-256 of the
    source files it read in the catalog, before the ledger row of the subject. A hemisphere
    whose map cannot be made (no ``sphere.reg``, a degenerate sphere, low coverage, a vertex
    count that differs) gets a flag and no map; a subject fails only when a file that is
    there cannot be read (``unreadable``) or differs from what the inventory listed
    (``source changed``). A failed subject loses its rows and its map, is recorded in the
    ledger with reason and scrubbed detail, and is not tried again with the same files and
    settings unless ``retry_failed``. ``force`` ignores the ledger.

    A present subject that no longer has a stats file, or a thickness file, is no unit of
    the stage any more: its rows, its map and its ledger row are removed, without a source
    being read, so that it is scored as a subject without those files.

    The report counts the units worked on now (``done``), skipped, and in the failed state
    when the call returns (``failed``; ``failures`` holds ``(unit, "<stage>: <reason>")``).

    Raises ``ConfigError("invalid", ...)`` for a stage that does not exist, and for a map
    of the run that the ``reference`` stage cannot read: the file is removed and the next
    call makes it again.
    """
    started = time.perf_counter()
    wanted = _checked_stages(stages)
    tally = _Tally()
    for stage in _stages(ctx, template, datasets, wanted, for_work=True):
        _run_stage(ctx, stage, tally, jobs=jobs, force=force, retry_failed=retry_failed)
    if "reference" in wanted:
        _reference_stage(ctx, template, tally, force=force, retry_failed=retry_failed)
    if "score" in wanted:
        _score_stage(ctx, template, tally, force=force, retry_failed=retry_failed)
    if "native" in wanted:
        _native_stage(ctx, template, tally, force=force)
    if "merge" in wanted:
        _merge_stage(ctx, tally)
    return tally.report(PHASE, started)


# --- who is scored against what -------------------------------------------------------------


@dataclass(frozen=True)
class _Applied:
    """A reference and the subjects that are scored against it."""

    reference_id: str
    kind: str
    strata: dict[str, str]
    """``subject_uid`` -> the stratum of the reference that the subject is compared with."""


@dataclass(frozen=True)
class _Scored:
    """Which subjects of the run have a current map, and which references apply to them."""

    with_map: pd.DataFrame
    """``subject_uid``, ``dataset``, ``field_strength``, ``fs_major`` of the present subjects
    with a current map, ordered by ``subject_uid``."""
    set_strata: dict[str, str]
    """``subject_uid`` -> its stratum within the set; no entry for a subject without one."""
    shipped: Reference | None
    shipped_strata: dict[str, str]
    signature: str
    """Input signature of the ``reference`` unit."""


def _present(ctx: RunContext) -> pd.DataFrame:
    subjects = ctx.public.subjects()
    return subjects[subjects["present"].to_numpy(dtype=bool)].reset_index(drop=True)


def _optional(value: object, kind: Callable[[Any], Any]) -> Any:
    if value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value)):
        return None
    return kind(value)


def _scored(ctx: RunContext, template: Template | None) -> _Scored:
    """Decided from the ledger, the catalog and the names of the map files; no map is read."""
    public, config = ctx.public, ctx.config
    columns = ["subject_uid", "dataset", "field_strength", "fs_major"]
    if template is None:
        return _Scored(pd.DataFrame(columns=columns), {}, None, {}, _signature(["no template"]))
    fsaverage_hash = stage_hashes(config, template_id=template.template_id)["fsaverage"]
    present = _present(ctx)
    signatures: dict[str, str] = {}
    for uid in present["subject_uid"]:
        signature = public.input_signature(uid, TIER_B)
        current = public.work_done("fsaverage", uid, fsaverage_hash, signature) is True
        if current and map_path(ctx.run_dir, uid).is_file():
            signatures[uid] = signature
    with_map = present.loc[present["subject_uid"].isin(signatures), columns]
    with_map = with_map.sort_values("subject_uid").reset_index(drop=True)
    lines = [
        "\t".join(map(str, (signatures[row[0]], *row)))
        for row in with_map.itertuples(index=False, name=None)
    ]
    assigned = assign_strata(with_map, min_n=config.metrics.min_reference_n)
    set_strata = {str(uid): str(name) for uid, name in assigned.items() if isinstance(name, str)}

    shipped = shipped_reference()
    shipped_strata: dict[str, str] = {}
    if shipped is not None:
        spheres = {hemi: template.files[f"surf/{hemi}.sphere.reg"][1] for hemi in HEMIS}
        if shipped.mesh != template.mesh or dict(shipped.sphere_sha256) != spheres:
            shipped = None  # made on another fsaverage: its vertices are not this run's
    if shipped is not None:
        for row in with_map.itertuples(index=False):
            stratum = applicable_stratum(
                shipped, _optional(row.field_strength, float), _optional(row.fs_major, int)
            )
            if stratum is not None:
                shipped_strata[str(row.subject_uid)] = stratum
        # by content as well: its maps and percentiles decide the z-maps and the scores
        files = sorted(path for path in load.SHIPPED_DIR.iterdir() if path.is_file())
        content = _signature([f"{path.name}\t{sha256_file(path)}" for path in files])
        lines.append(f"shipped\t{shipped.reference_id}\t{content}")
    return _Scored(with_map, set_strata, shipped, shipped_strata, _signature(sorted(lines)))


def _reference_row(ctx: RunContext) -> tuple[str | None, str | None]:
    return _recorded(ctx, "reference").get(RUN_UNIT, (None, None))


def _set_reference_id(ctx: RunContext) -> str | None:
    """The id of the set reference that the ``reference`` unit last built; None for none."""
    _, output = _reference_row(ctx)
    if not isinstance(output, str):
        return None
    name = Path(output).parent.name
    return name if name.startswith(SET_PREFIX) else None


def _applied(ctx: RunContext, scored: _Scored) -> list[_Applied]:
    """The references that the tables on disk were computed with, the primary one of this
    run first: the shipped one in a local run, the one built within the set in a census."""
    found = []
    set_id = _set_reference_id(ctx)
    if scored.set_strata and set_id is not None:
        found.append(_Applied(set_id, "set", scored.set_strata))
    if scored.shipped is not None and scored.shipped_strata:
        found.append(_Applied(scored.shipped.reference_id, "shipped", scored.shipped_strata))
    if ctx.public.mode() == "local":
        found.sort(key=lambda applied: applied.kind != "shipped")
    return found


def _reference_hash(ctx: RunContext, template: Template | None) -> str:
    template_id = None if template is None else template.template_id
    return stage_hashes(ctx.config, template_id=template_id)["reference"]


def _reference_current(ctx: RunContext, template: Template | None, scored: _Scored) -> bool:
    done = ctx.public.work_done(
        "reference", RUN_UNIT, _reference_hash(ctx, template), scored.signature
    )
    if done is not True:
        return False
    _, output = _reference_row(ctx)
    if not isinstance(output, str):
        return True
    run_dir = Path(ctx.run_dir)
    return (run_dir / output).is_file() and table_path(run_dir, "clusters").is_file()


# --- reference and clusters -----------------------------------------------------------------


def _build_set_reference(
    ctx: RunContext,
    template: Template,
    scored: _Scored,
    maps: Mapping[str, Mapping[Hemi, np.ndarray]],
    medians: Mapping[str, Mapping[Hemi, float]],
) -> Reference:
    """The per-vertex reference of every stratum that a subject is assigned to, built from
    every member of the stratum and written to ``tables/reference/<id>/``."""
    config = ctx.config.metrics
    stage = _reference_hash(ctx, template)
    digests = {
        uid: sha256_file(map_path(ctx.run_dir, uid)) for uid in scored.with_map["subject_uid"]
    }
    strata: dict[str, dict[Hemi, ReferenceMaps]] = {}
    members: dict[str, tuple[str, ...]] = {}
    member_digests: dict[str, str] = {}
    for name in sorted(set(scored.set_strata.values())):
        uids = stratum_members(scored.with_map, name)
        members[name] = tuple(uids)
        member_digests[name] = members_sha256({uid: digests[uid] for uid in uids})
        strata[name] = {}
        for hemi in HEMIS:
            # a hemisphere without a map is NaN throughout, and so is its row here
            stack = np.stack(
                [maps[uid][hemi] - np.float32(medians[uid][hemi]) for uid in uids]
            ).astype(np.float32, copy=False)
            strata[name][hemi] = build_reference(
                stack,
                template.hemis[hemi],
                template.adjacency(hemi),
                min_vertex_n=config.min_vertex_n,
                smooth_iterations=config.mad_smooth_iterations,
                mad_floor_mm=config.mad_floor_mm,
            )
    reference = Reference(
        reference_id=set_reference_id(stage, member_digests),
        kind="set",
        mesh=template.mesh,
        template_id=template.template_id,
        sphere_sha256={hemi: template.files[f"surf/{hemi}.sphere.reg"][1] for hemi in HEMIS},
        config_hash=stage,
        strata=strata,
        members=members,
        metric_norms={},
        parcel_norms={},
    )
    write_reference(
        reference_dir(ctx.run_dir, reference.reference_id), reference, members_sha256=member_digests
    )
    return reference


def _cluster_rows(
    ctx: RunContext,
    template: Template,
    reference: Reference,
    strata: Mapping[str, str],
    maps: Mapping[str, Mapping[Hemi, np.ndarray]],
    medians: Mapping[str, Mapping[Hemi, float]],
) -> list[dict[str, object]]:
    config = ctx.config.metrics
    rows: list[dict[str, object]] = []
    for uid in sorted(strata):
        for hemi in HEMIS:
            median = medians[uid][hemi]
            if not math.isfinite(median):
                continue
            half = template.hemis[hemi]
            z = zmap(maps[uid][hemi], median, reference.strata[strata[uid]][hemi])
            for region in REGION_NAMES:
                found = find_clusters(
                    z,
                    half.regions[region],
                    half,
                    template.adjacency(hemi),
                    threshold=config.cluster_z,
                    min_vertices=config.cluster_min_vertices,
                )
                for rank, cluster in enumerate(found[: config.clusters_kept], start=1):
                    rows.append(
                        {
                            "subject_uid": uid,
                            "reference_id": reference.reference_id,
                            "hemi": hemi,
                            "region": region,
                            "rank": rank,
                            "n_vertices": len(cluster.vertices),
                            "area_mm2": cluster.area_mm2,
                            "mass": cluster.mass,
                            "excess_mass": cluster.excess_mass,
                            "peak_z": cluster.peak_z,
                            "peak_vertex": cluster.peak_vertex,
                            "centroid_vertex": cluster.centroid_vertex,
                        }
                    )
    return rows


def _remove_stale_references(ctx: RunContext, keep: str | None) -> None:
    """A reference built within the run is replaced by the next one; one of another kind
    (the rated reference of a later pass) is left alone."""
    root = Path(ctx.run_dir) / REFERENCE_DIR
    if not root.is_dir():
        return
    for path in sorted(root.iterdir()):
        if path.is_dir() and path.name.startswith(SET_PREFIX) and path.name != keep:
            shutil.rmtree(path)


def _merge(ctx: RunContext, table: str) -> Path:
    return merge_parts(ctx.run_dir, table, present=ctx.public.subject_uids())


def _read_run_maps(ctx: RunContext, uids: Sequence[str]) -> dict[str, dict[Hemi, np.ndarray]]:
    """The maps of ``uids``. One that cannot be read is an output of the run that was
    damaged, not a property of its subject: every such map is removed, so that the
    ``fsaverage`` stage finds it missing and makes it again, and the phase stops.

    Raises ``ConfigError("invalid", ...)`` that names the subjects.
    """
    maps: dict[str, dict[Hemi, np.ndarray]] = {}
    damaged: list[str] = []
    for uid in uids:
        try:
            maps[uid] = read_maps(map_path(ctx.run_dir, uid))
        except (OSError, ValueError):
            damaged.append(uid)
    if damaged:
        for uid in damaged:
            map_path(ctx.run_dir, uid).unlink(missing_ok=True)
        raise ConfigError(
            "invalid",
            f"the fsaverage map of {len(damaged)} subject(s) in {FSAVG_DIR} could not be read "
            f"({_some(damaged)}) and was removed; run the metrics again to make it anew",
        )
    return maps


def _reference_stage(
    ctx: RunContext, template: Template | None, tally: _Tally, *, force: bool, retry_failed: bool
) -> None:
    """The stage ``reference``: the set reference, when the run has enough subjects with a
    map, and ``clusters.parquet`` for every reference that applies."""
    public = ctx.public
    if template is None:
        return  # a run without a template has no map; the score stage says so
    _merge(ctx, "vertex_summary")
    scored = _scored(ctx, template)
    stage = _reference_hash(ctx, template)
    if not force:
        if _reference_current(ctx, template, scored):
            tally.skipped += 1
            return
        failed = public.work_done("reference", RUN_UNIT, stage, scored.signature) is False
        if failed and not retry_failed:
            tally.failures[("reference", RUN_UNIT)] = _reference_row(ctx)[0] or METRIC_FAILURE
            return

    uids = [str(uid) for uid in scored.with_map["subject_uid"]]
    maps = _read_run_maps(ctx, uids)
    medians = {
        uid: {hemi: global_median(maps[uid][hemi], template.hemis[hemi]) for hemi in HEMIS}
        for uid in uids
    }
    rows: list[dict[str, object]] = []
    output = None
    built = None
    if scored.set_strata:
        built = _build_set_reference(ctx, template, scored, maps, medians)
        rows += _cluster_rows(ctx, template, built, scored.set_strata, maps, medians)
        output = f"{REFERENCE_DIR}/{built.reference_id}/{INDEX_JSON}"
    if scored.shipped is not None and scored.shipped_strata:
        rows += _cluster_rows(ctx, template, scored.shipped, scored.shipped_strata, maps, medians)
        output = output or f"tables/{table_path(ctx.run_dir, 'clusters').name}"
    clusters = table_path(ctx.run_dir, "clusters")
    if output is None:
        clusters.unlink(missing_ok=True)  # no reference applies: there is no cluster table
    else:
        clusters.parent.mkdir(parents=True, exist_ok=True)
        write_table(clusters, frame_of(rows, CLUSTERS), CLUSTERS, sort_by=KEYS["clusters"])
    _remove_stale_references(ctx, None if built is None else built.reference_id)
    public.record_work(
        [
            WorkRow(
                phase="reference",
                unit=RUN_UNIT,
                config_hash=stage,
                input_signature=scored.signature,
                status="done",
                output=output,
            )
        ]
    )
    tally.failures.pop(("reference", RUN_UNIT), None)
    tally.done += 1


# --- scores ---------------------------------------------------------------------------------


def _digest(path: Path) -> str:
    return sha256_file(path) if path.is_file() else "-"


def _score_signature(
    ctx: RunContext, subjects: pd.DataFrame, applied: Sequence[_Applied], scored: _Scored
) -> str:
    """Over the content of the three tables the scores are computed from, what they take
    from the catalog, and who is scored against which reference."""
    values = [
        [_optional(value, lambda v: v if isinstance(v, str) else float(v)) for value in row]
        for row in subjects[list(_SUBJECT_VALUES)].itertuples(index=False, name=None)
    ]
    payload = {
        "tables": {
            name: _digest(table_path(ctx.run_dir, name))
            for name in ("regional", "vertex_summary", "clusters")
        },
        "subjects": values,
        "references": [[a.reference_id, sorted(a.strata.items())] for a in applied],
        "reference_input": scored.signature,
    }
    return _signature([canonical_json(payload)])


def _too_few_note(ctx: RunContext, template: Template | None, scored: _Scored) -> str:
    ranked = "Ranked on raw ceiling fraction and occipital-to-control thickness"
    if template is None:
        return (
            "No fsaverage template: thickness is not compared per location. "
            f"{ranked} (decision D16). To enable: pass --fsaverage DIR "
            "(any FreeSurfer install has one in subjects/)."
        )
    needed = ctx.config.metrics.min_reference_n
    return (
        f"{len(scored.with_map)} subjects with a map: too few for per-location scoring "
        f"({needed} needed). {ranked}."
    )


def _score_frame(
    ctx: RunContext,
    template: Template | None,
    subjects: pd.DataFrame,
    applied: Sequence[_Applied],
    scored: _Scored,
    score_hash: str,
) -> pd.DataFrame:
    config = ctx.config
    run_dir = Path(ctx.run_dir)
    vertex_summary = read_table(table_path(run_dir, "vertex_summary"), SCHEMAS["vertex_summary"])
    regional = read_table(table_path(run_dir, "regional"), SCHEMAS["regional"])
    clusters = read_table(table_path(run_dir, "clusters"), CLUSTERS) if applied else None
    metrics = scoring.subject_metrics(
        vertex_summary,
        clusters,
        regional,
        subjects,
        config.metrics.regions,
        mass=config.scoring.mass,
        zmaps={a.reference_id: sorted(a.strata) for a in applied} if applied else None,
    )
    if not applied:
        frame = scoring.fallback_rank(metrics)
    else:
        strata = pd.Series(scored.set_strata, dtype=object)
        frame = scoring.composite(metrics, strata, config.scoring)
        shipped = scored.shipped
        if shipped is not None and shipped.metric_norms:
            # against a shipped reference that carries them, a component is placed among the
            # percentiles of the reference population, stratum by stratum
            pieces = []
            placed_rows = np.zeros(len(metrics), dtype=bool)
            of_shipped = (metrics["reference_id"] == shipped.reference_id).to_numpy()
            for stratum, norms in sorted(shipped.metric_norms.items()):
                if not all(name in norms for name in config.scoring.components):
                    continue
                members = {u for u, name in scored.shipped_strata.items() if name == stratum}
                chosen = of_shipped & metrics["subject_uid"].isin(members).to_numpy()
                if chosen.any():
                    pieces.append(
                        scoring.composite(metrics[chosen], strata, config.scoring, norms=norms)
                    )
                    placed_rows |= chosen
            if pieces:
                frame = pd.concat([frame[~placed_rows], *pieces]).sort_values(
                    ["subject_uid", "reference_id"], ignore_index=True
                )
    frame = scoring.add_rank(frame)

    order = {a.reference_id: position for position, a in enumerate(applied)}
    stratum_of = {(uid, a.reference_id): name for a in applied for uid, name in a.strata.items()}
    sizes: dict[tuple[str, str], int] = {}
    for a in applied:
        if a.kind == "shipped" and scored.shipped is not None:
            for name in set(a.strata.values()):
                sizes[(a.reference_id, name)] = stratum_size(scored.shipped, name)
        else:
            index = read_index(reference_dir(run_dir, a.reference_id))
            for name in set(a.strata.values()):
                sizes[(a.reference_id, name)] = index.strata[name].n
    keys = list(zip(frame["subject_uid"], frame["reference_id"], strict=True))
    first = (
        pd.DataFrame(
            {"uid": [uid for uid, _ in keys], "order": [order.get(r, -1) for _, r in keys]}
        )
        .groupby("uid")["order"]
        .transform("min")
    )
    frame["primary"] = np.asarray([order.get(r, -1) for _, r in keys]) == first.to_numpy()
    strata_column = [stratum_of.get(key) for key in keys]
    frame["stratum"] = pd.Series(strata_column, index=frame.index, dtype=object)
    frame["n_reference"] = pd.array(
        [
            None if name is None else sizes[(reference, name)]
            for (_, reference), name in zip(keys, strata_column, strict=True)
        ],
        dtype="Int32",
    )
    frame["score_config_hash"] = pd.Series(
        [score_hash] * len(frame), index=frame.index, dtype=object
    )
    # A subject with fewer than two rows of vertex_summary lacks a thickness file, or its
    # fsaverage unit failed: the flag says that thickness gave the score nothing, and the
    # ledger says why.
    rows_of = vertex_summary.groupby("subject_uid").size()
    frame["flags"] = pd.Series(
        [
            join_flags(
                [*split_flags(flags or ""), *(("no_thickness",) if rows_of.get(uid, 0) < 2 else ())]
            )
            for uid, flags in zip(frame["subject_uid"], frame["flags"], strict=True)
        ],
        index=frame.index,
        dtype=object,
    )
    return frame[list(SCORES.names)]


def _score_stage(
    ctx: RunContext, template: Template | None, tally: _Tally, *, force: bool, retry_failed: bool
) -> None:
    """The stage ``score``: ``scores.parquet``, one row per present subject and reference."""
    public = ctx.public
    for table in ("regional", "vertex_summary"):
        _merge(ctx, table)
    scored = _scored(ctx, template)
    applied = _applied(ctx, scored) if template is not None else []
    template_id = None if template is None else template.template_id
    score_hash = stage_hashes(ctx.config, template_id=template_id, with_reference=bool(applied))[
        "score"
    ]
    subjects = _present(ctx).sort_values("subject_uid").reset_index(drop=True)
    signature = _score_signature(ctx, subjects, applied, scored)
    if template is None or not (scored.set_strata or scored.shipped_strata):
        tally.notes.append(_too_few_note(ctx, template, scored))

    scores = table_path(ctx.run_dir, "scores")
    if not force:
        state = public.work_done("score", RUN_UNIT, score_hash, signature)
        if state is True and scores.is_file():
            tally.skipped += 1
            return
        if state is False and not retry_failed:
            reason, _ = _recorded(ctx, "score").get(RUN_UNIT, (None, None))
            tally.failures[("score", RUN_UNIT)] = reason or METRIC_FAILURE
            return
    if template is not None and not _reference_current(ctx, template, scored):
        # Scores from clusters of another state of the run would look like any others. This
        # says something about the order of the calls, not about the inputs, so no row is
        # recorded: the next call looks again.
        tally.failures[("score", RUN_UNIT)] = STALE_REFERENCE
        return
    frame = _score_frame(ctx, template, subjects, applied, scored, score_hash)
    scores.parent.mkdir(parents=True, exist_ok=True)
    write_table(scores, frame, SCORES, sort_by=KEYS["scores"])
    public.record_work(
        [
            WorkRow(
                phase="score",
                unit=RUN_UNIT,
                config_hash=score_hash,
                input_signature=signature,
                status="done",
                output="tables/scores.parquet",
            )
        ]
    )
    tally.failures.pop(("score", RUN_UNIT), None)
    tally.done += 1


# --- native centroids and merging -----------------------------------------------------------


def update_native(
    ctx: RunContext, *, template: Template | None, force: bool = False
) -> PhaseReport:
    """The stage ``native``: ``clusters_native.parquet`` from ``clusters.parquet`` and the
    vertex maps of the done renders. Reads no source file.

    A cluster gets a row when its subject has a done render that carries counterparts on
    this template and the centroid, or a mesh neighbour of it, is the counterpart of a
    vertex in that render's vertex map. Without a template or a cluster table there is no
    table, and one that an earlier state left is removed. ``clusters.parquet`` is only read.

    A vertex map that cannot be read leaves its subject without rows and is named in a note
    of the report; it fails no unit, and every later call reads the file again, so a render
    made anew is picked up.
    """
    started = time.perf_counter()
    tally = _Tally()
    _native_stage(ctx, template, tally, force=force)
    return tally.report("native", started)


def _native_stage(
    ctx: RunContext, template: Template | None, tally: _Tally, *, force: bool
) -> None:
    public = ctx.public
    run_dir = Path(ctx.run_dir)
    clusters_file = table_path(run_dir, "clusters")
    native_file = table_path(run_dir, "clusters_native")
    if template is None or not clusters_file.is_file():
        native_file.unlink(missing_ok=True)
        return
    renders = public.renders()
    usable = renders[
        (renders["status"] == "done") & (renders["template_id"] == template.template_id)
    ]
    lines = [
        f"{uid}\t{key}\t{template.template_id}"
        for uid, key in zip(usable["subject_uid"], usable["render_key"], strict=True)
    ]
    signature = _signature([sha256_file(clusters_file), *sorted(lines)])
    stage = stage_hashes(ctx.config, template_id=template.template_id)["native"]
    done = public.work_done("native", RUN_UNIT, stage, signature) is True
    if done and not force and native_file.is_file():
        tally.skipped += 1
        return

    clusters = read_table(clusters_file, CLUSTERS)
    rendered = set(usable["subject_uid"])
    adjacency = {hemi: template.adjacency(hemi) for hemi in HEMIS}
    frames = []
    damaged: list[str] = []
    for uid, rows in clusters.groupby("subject_uid", sort=True):
        path = media_dir(run_dir, str(uid)) / _VERTEX_MAP
        if uid not in rendered or not path.is_file():
            continue
        try:
            vertex_map = read_npz(path)
        except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile):
            damaged.append(str(uid))  # its centroids get no row
            continue
        frames.append(native_centroids(rows.reset_index(drop=True), vertex_map, adjacency))
    if frames:
        native = pd.concat(frames, ignore_index=True)
    else:
        native = CLUSTERS_NATIVE.empty_table().to_pandas().drop(columns=["template_id"])
    native["template_id"] = pd.Series([template.template_id] * len(native), dtype=object)
    write_table(native_file, native, CLUSTERS_NATIVE, sort_by=KEYS["clusters_native"])
    if damaged:
        tally.notes.append(
            f"media/<subject_uid>/{_VERTEX_MAP} of {len(damaged)} rendered subject(s) cannot be "
            f"read ({_some(damaged)}): their cluster centroids have no native position. "
            "Remove the file and run `surfcensus render`, which makes such a render again."
        )
        # a signature that no state of the run gives: the next call reads the files again
        signature = _signature([signature, *(f"damaged\t{uid}" for uid in damaged)])
    public.record_work(
        [
            WorkRow(
                phase="native",
                unit=RUN_UNIT,
                config_hash=stage,
                input_signature=signature,
                status="done",
                output="tables/clusters_native.parquet",
            )
        ]
    )
    tally.done += 1


def merge_tables(ctx: RunContext) -> PhaseReport:
    """The stage ``merge``: every ``tables/<table>.parquet`` that has parts, from its parts,
    without the rows of subjects that are no longer present."""
    started = time.perf_counter()
    tally = _Tally()
    _merge_stage(ctx, tally)
    return tally.report("merge", started)


def _merge_stage(ctx: RunContext, tally: _Tally) -> None:
    for table in PART_TABLES:
        if not (Path(ctx.run_dir) / "tables" / "_parts" / table).is_dir():
            continue
        schema = None
        if table == "intensity":  # its columns follow from the depths of this run
            schema = intensity_schema(len(ctx.config.intensity.depths))
        merge_parts(ctx.run_dir, table, present=ctx.public.subject_uids(), schema=schema)
        tally.done += 1
