"""A run directory with the EPI interface in it, for the tests of the EPI pages of the main
app (work package E2.D; docs/epi_interface.md, "The app in a run directory").

Until the phases of step 2 are bound into ``Run`` (E2.Z: ``Run.func_inventory`` over
``func.inventory``, ``Run.pack`` over ``func.batch``), this module writes what they leave in a
run directory by itself, from step 1's packs (``tests/epi_packs.py``):

* ``epi/<participant_uid>/`` of two participants: a copy of the ``--out`` directory of step 1
  under the ids of the run (participant ``FULL``, both sessions packed), and a copy of it under
  other ids with only its first session's directory (participant ``HALF``: ``stream.json``
  lists both sessions, as the pack phase writes it while the second is still to be made);
* the rows of the catalog through ``store.Catalog`` (``upsert_func_source``,
  ``upsert_func_participant``, ``upsert_func_session``) and ``PublicCatalog.record_packs``:
  every run of ``session.json`` with its reason, a ``done`` pack for every packed run that is on
  disk, and in the second session of ``HALF`` one ``failed`` pack and the others none.

The ids of the packs are 16 hex characters already (step 1's ``func_id``), so they serve as
``session_uid`` and ``run_uid``; the catalog rows are the facts the page and the index read,
not what an inventory of the synthetic folder would give (no files, no anchor, no frame check
but the flag). When the real phases exist, :func:`add_epi` can be replaced by
``Run.func_inventory`` and ``Run.pack`` on ``synth_func.write_review_folder``.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from epi_packs import PackedStream  # pyrefly: ignore[missing-import]

from surfcensus.config import EpiConfig
from surfcensus.func import STREAM_JSON
from surfcensus.func.model import SessionInfo, StreamInfo, info_text, read_info
from surfcensus.records import (
    ConnectorMatch,
    Display,
    FrameDisplacement,
    FuncRunRow,
    FuncSessionRow,
    FuncSourceRecord,
    Grid,
    PackRow,
    RunEntities,
)
from surfcensus.store import EPI_DIR, Catalog, PublicCatalog

FULL = "f0" * 8
"""The participant whose two sessions are packed."""
HALF = "a5" * 8
"""The participant whose second session is still to be packed: ``stream.json`` lists it, the
run directory has no ``session.json`` of it, its runs have no pack or a failed one."""
LABELS = {FULL: "sub-01", HALF: "sub-07"}

_HEX = re.compile(r"\b[0-9a-f]{16}\b")


@dataclass(frozen=True)
class EpiRun:
    run_dir: Path
    participants: tuple[str, ...]
    sessions: dict[str, tuple[str, ...]]
    """participant -> its sessions, in stream order."""
    shown: dict[str, tuple[str, ...]]
    """participant -> the sessions that have a ``session.json`` in the run directory."""
    runs: dict[str, tuple[str, ...]]
    """session -> its runs as ``session.json`` lists them."""
    packed: dict[str, tuple[str, ...]]
    """session -> its runs with a ``done`` pack."""
    failed: tuple[str, ...]
    names: tuple[str, ...]
    """Every run name and session label of the stream: none may reach a rating route."""


def _other(uid: str) -> str:
    return hashlib.sha256(f"half:{uid}".encode()).hexdigest()[:16]


def _copy_stream(
    source: Path, target: Path, ids: dict[str, str], sessions: tuple[str, ...], label: str
) -> None:
    """Copy step 1's ``--out`` to ``target`` with every id renamed by ``ids``; only the
    session directories of ``sessions`` (old ids) are copied, ``stream.json`` lists them all."""
    target.mkdir(parents=True)
    for session in sessions:
        shutil.copytree(source / session, target / ids[session])
    for path in sorted(target.rglob("*"), key=lambda p: -len(p.parts)):
        if path.is_dir() and path.name in ids:
            path.rename(path.with_name(ids[path.name]))
    for path in target.rglob("*.json"):
        text = path.read_text(encoding="utf-8")
        path.write_text(_HEX.sub(lambda m: ids.get(m.group(0), m.group(0)), text), encoding="utf-8")
    stream = read_info(source / STREAM_JSON, StreamInfo)
    assert stream is not None
    changed = {"sessions": tuple(ids[s] for s in stream.sessions), "label": label}
    (target / STREAM_JSON).write_text(
        info_text(stream.model_copy(update=changed)), encoding="utf-8"
    )


def _runs(info: SessionInfo) -> tuple[FuncRunRow, ...]:
    rows = []
    for run in info.runs:
        packable = run.reason is None
        frame = None
        if run.frame is not None:
            frame = FrameDisplacement(
                mean_mm=run.frame.mean_mm,
                max_mm=run.frame.max_mm,
                rotation_deg=run.frame.rotation_deg,
                translation_mm=run.frame.translation_mm,
                translation_ras=run.frame.translation_ras,
                shear_scale=run.frame.shear_scale,
                n_voxels=100,
                flagged=run.frame.flagged,
            )
        grid = next((g for g in info.grids if g.grid_key == run.grid_key), None)
        rows.append(
            FuncRunRow(
                run_uid=run.run,
                session_uid=info.session,
                run_key=run.name,
                name=run.name,
                entities=RunEntities(sub="01"),
                session_order=run.session_order,
                acquisition_order=run.acquisition_order,
                anchor=run.anchor,
                selected=True,
                packable=packable,
                reason=run.reason,
                detail=run.detail,
                missing_roles=(),
                n_volumes=run.n_volumes,
                tr_s=run.tr_s,
                echoes=1,
                display=Display("anat", "T1w"),
                spaces=("T1w",),
                grid=None
                if grid is None or not packable
                else Grid(
                    grid.shape,
                    (
                        (grid.voxel_mm[0], 0.0, 0.0, 0.0),
                        (0.0, grid.voxel_mm[1], 0.0, 0.0),
                        (0.0, 0.0, grid.voxel_mm[2], 0.0),
                        (0.0, 0.0, 0.0, 1.0),
                    ),
                ),
                raw_affine=None,
                reference_image=None,
                rigid=None,
                dof=None,
                corrections=None,
                frame=frame,
                files=(),
            )
        )
    return tuple(rows)


def add_epi(run_dir: Path, packed: PackedStream, *, subject_uid: str | None = None) -> EpiRun:
    """Write the EPI interface of two participants into a local run directory (see the
    module). ``subject_uid`` is the FreeSurfer subject of the first session of ``FULL``."""
    epi = run_dir / EPI_DIR
    full_ids = {s: s for s in packed.sessions}
    for info in packed.infos.values():
        full_ids.update({r.run: r.run for r in info.runs})
    half_ids = {old: _other(old) for old in full_ids}
    _copy_stream(packed.out, epi / FULL, full_ids, packed.sessions, LABELS[FULL])
    _copy_stream(packed.out, epi / HALF, half_ids, packed.sessions[:1], LABELS[HALF])

    config = EpiConfig()
    sessions: dict[str, tuple[str, ...]] = {}
    shown: dict[str, tuple[str, ...]] = {}
    runs: dict[str, tuple[str, ...]] = {}
    done: dict[str, tuple[str, ...]] = {}
    failed: list[str] = []
    names: set[str] = set()
    pack_rows: list[PackRow] = []
    with Catalog(run_dir / "catalog.sqlite") as catalog:
        dataset_id = int(catalog.query("SELECT dataset_id FROM datasets ORDER BY dataset_id")[0][0])
        func_id = catalog.upsert_func_source(
            dataset_id,
            FuncSourceRecord(
                connector="fmriprep",
                match=ConnectorMatch("fmriprep", "21.0.2", "scanner-T1w"),
                deriv_root=str(packed.dataset.deriv_root),
                deriv_identity=str(packed.dataset.deriv_root),
                raw_root=None,
                deriv_found="surface layout",
                raw_found=None,
            ),
        )
        for participant, ids, on_disk in (
            (FULL, full_ids, packed.sessions),
            (HALF, half_ids, packed.sessions[:1]),
        ):
            catalog.upsert_func_participant(
                func_id, participant, LABELS[participant], len(packed.sessions)
            )
            sessions[participant] = tuple(ids[s] for s in packed.sessions)
            shown[participant] = tuple(ids[s] for s in on_disk)
            for order, old in enumerate(packed.sessions):
                text = info_text(packed.infos[old])
                info = SessionInfo.model_validate_json(
                    _HEX.sub(lambda m, ids=ids: ids.get(m.group(0), m.group(0)), text)
                )
                rows = _runs(info)
                names.update(r.name for r in rows)
                names.add(info.label)
                label = info.label.rpartition("ses-")[2] if "ses-" in info.label else None
                catalog.upsert_func_session(
                    FuncSessionRow(
                        session_uid=info.session,
                        participant_uid=participant,
                        session=label,
                        stream_order=order,
                        subject_uid=subject_uid if (participant == FULL and order == 0) else None,
                        fs_subject=None,
                        anchor=None,
                        anchor_note=None,
                        frame=None,
                        n_grids=len(info.grids),
                        problem=None,
                        inventory_hash="0" * 16,
                        files=(),
                        runs=rows,
                    )
                )
                runs[info.session] = tuple(r.run_uid for r in rows)
                packable = [r.run_uid for r in rows if r.packable]
                if old in on_disk:
                    done[info.session] = tuple(packable)
                    pack_rows += [
                        PackRow(
                            run_uid=uid,
                            session_uid=info.session,
                            config_hash=config.pack_hash(),
                            input_signature="0" * 16,
                            status="done",
                            pack_key="1" * 16,
                            path=f"{EPI_DIR}/{participant}/{info.session}/{uid}",
                        )
                        for uid in packable
                    ]
                else:
                    done[info.session] = ()
                    failed.append(packable[0])
                    pack_rows.append(
                        PackRow(
                            run_uid=packable[0],
                            session_uid=info.session,
                            config_hash=config.pack_hash(),
                            input_signature="0" * 16,
                            status="failed",
                            reason="pack failure",
                            detail="OSError",
                        )
                    )
    with PublicCatalog.open(run_dir, readonly=False) as public:
        public.record_packs(pack_rows)
    return EpiRun(
        run_dir=run_dir,
        participants=(FULL, HALF),
        sessions=sessions,
        shown=shown,
        runs=runs,
        packed=done,
        failed=tuple(failed),
        names=tuple(sorted(names)),
    )
