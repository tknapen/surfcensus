"""What the packages of step 3 of the EPI interface share (docs/epi_interface.md, "Step 3: the
plan (E3.0)"): the shapes of the subject pages and their Literals, the four routes that work
package E3.A adds to ``app.schemas.ROUTES``, the two review columns, the recorded state of a
run (``func.batch.recorded_states``), the module ``func.status`` that E3.B writes, the
constants of the functional export tables, and the fixtures of step 3.

The names were fixed before the bodies were written (work package E3.0), so that E3.A and E3.B
can be written side by side. A package that renames or reorders something fails here; a change
that is meant goes here as well, in the same change. A stub that is written keeps its name and
parameters; E3.B removes the test that the stubs raise when it writes them, and E3.A, when it
appends :data:`SUBJECT_ROUTES` to ``ROUTES``, keeps the test that pins them.
"""

from __future__ import annotations

import dataclasses
import inspect
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, get_args

import pandas as pd
import pytest
from epi_packs import PackedStream, make_stream  # pyrefly: ignore[missing-import]
from epi_run_fixture import (  # pyrefly: ignore[missing-import]
    CORRECTIONS,
    FULL,
    HALF,
    UNCHECKED,
    add_epi,
    session_ids,
)
from synth_func import func_counts, write_review_folder  # pyrefly: ignore[missing-import]

from surfcensus import schemas as table_schemas
from surfcensus import store
from surfcensus.app import epi_schemas as shapes
from surfcensus.app import schemas
from surfcensus.app.schemas import Route
from surfcensus.config import EpiConfig
from surfcensus.export import tables
from surfcensus.func import batch, status
from surfcensus.func.batch import RecordedState, recorded_states
from surfcensus.func.model import CorrectionsInfo, corrections_info
from surfcensus.records import AffineChoice, AnchorRule
from surfcensus.store import PublicCatalog

# --- the shapes -------------------------------------------------------------------------------

SURFACE = Literal["rated", "unratable", "rendered", "render failed", "not rendered", "none"]
STATE = Literal["packed", "pending", "failed", "cannot be packed", "not selected"]
RULE = Literal["acquisition time", "session label"]

FIELDS: dict[type, tuple[tuple[str, Any], ...]] = {
    # shape -> its fields in order, with their types
    shapes.SubjectRowOut: (
        ("key", str),
        ("kind", Literal["subject", "participant"]),
        ("subject_uid", str | None),
        ("participant_uid", str | None),
        ("label", str | None),
        ("dataset", str),
        ("surface", SURFACE),
        ("surface_reason", str | None),
        ("n_sessions", int),
        ("n_runs", int),
        ("n_packable", int),
        ("n_packed", int),
        ("n_pending", int),
        ("n_failed", int),
        ("n_flagged", int),
        ("url", str),
        ("epi_url", str | None),
    ),
    shapes.SubjectsOut: (
        ("subjects", tuple[shapes.SubjectRowOut, ...]),
        ("n_pending", int),
    ),
    shapes.FuncFrameOut: (
        ("mean_mm", float),
        ("max_mm", float),
        ("rotation_deg", float),
        ("translation_mm", float),
        ("flagged", bool | None),
    ),
    shapes.FuncRunOut: (
        ("run_uid", str),
        ("name", str | None),
        ("acquisition_order", int | None),
        ("session_order", int | None),
        ("anchor", bool),
        ("selected", bool),
        ("n_volumes", int | None),
        ("tr_s", float | None),
        ("echoes", int),
        ("grid_key", str | None),
        ("grid_shape", tuple[int, int, int] | None),
        ("voxel_mm", tuple[float, float, float] | None),
        ("oblique_deg", float | None),
        ("raw_affine", AffineChoice | None),
        ("reference_image", bool | None),
        ("rigid", bool | None),
        ("dof", int | None),
        ("corrections", CorrectionsInfo | None),
        ("frame", shapes.FuncFrameOut | None),
        ("state", STATE),
        ("reason", str | None),
        ("detail", str | None),
        ("epi_url", str | None),
    ),
    shapes.FuncSessionOut: (
        ("session_uid", str),
        ("participant_uid", str),
        ("participant", str | None),
        ("label", str),
        ("stream_order", int),
        ("stream_rule", RULE),
        ("subject_uid", str | None),
        ("fs_subject", str | None),
        ("anchor_run_uid", str | None),
        ("anchor_name", str | None),
        ("anchor_rule", AnchorRule | None),
        ("anchor_note", str | None),
        ("anchor_packed", bool),
        ("frame_limit_mm", float | None),
        ("shared_frame_plausible", bool | None),
        ("anchor_is_outlier", bool | None),
        ("n_grids", int),
        ("problem", str | None),
        ("shown", bool),
        ("runs", tuple[shapes.FuncRunOut, ...]),
    ),
    shapes.SubjectPageOut: (
        ("subject", shapes.SubjectRowOut),
        ("review_url", str | None),
        ("rate_url", str),
        ("index_url", str),
        ("epi_index_url", str),
        ("sessions", tuple[shapes.FuncSessionOut, ...]),
    ),
}


@pytest.mark.parametrize("shape", list(FIELDS), ids=lambda s: s.__name__)
def test_the_shapes_of_the_subject_pages_have_their_fields_in_order(shape: Any) -> None:
    assert issubclass(shape, shapes._Shape)
    assert shape.model_config.get("frozen") and shape.model_config.get("extra") == "forbid"
    got = tuple((name, field.annotation) for name, field in shape.model_fields.items())
    assert got == FIELDS[shape]
    # every field is required: a page never gets a shape with a default filled in for it
    assert all(field.is_required() for field in shape.model_fields.values())


def test_the_literals_are_those_of_the_catalog_and_the_tables() -> None:
    assert shapes.SurfaceState == SURFACE
    assert shapes.RunState == STATE
    assert get_args(shapes.RunState) == table_schemas.PACK_STATES
    assert get_args(batch.RecordedStateName) == table_schemas.PACK_STATES
    assert set(get_args(shapes.StreamRule)) == set(store.SESSION_ORDERS)
    assert get_args(shapes.RowKind) == ("subject", "participant")


def _row(**changes: Any) -> dict[str, Any]:
    base = {
        "key": "0" * 16,
        "kind": "subject",
        "subject_uid": "0" * 16,
        "participant_uid": FULL,
        "label": "sub-001",
        "dataset": "D-00000000",
        "surface": "rendered",
        "surface_reason": None,
        "n_sessions": 1,
        "n_runs": 2,
        "n_packable": 2,
        "n_packed": 1,
        "n_pending": 1,
        "n_failed": 0,
        "n_flagged": 0,
        "url": "/subject/" + "0" * 16,
        "epi_url": f"/epi/{FULL}",
    }
    return base | changes


def test_a_subject_page_body_validates_and_a_wrong_one_does_not() -> None:
    run = {
        "run_uid": "1" * 16,
        "name": "ses-1_task-rest_run-1",
        "acquisition_order": 0,
        "session_order": 0,
        "anchor": True,
        "selected": True,
        "n_volumes": 10,
        "tr_s": 2.0,
        "echoes": 1,
        "grid_key": "a" * 12,
        "grid_shape": [10, 11, 12],
        "voxel_mm": [3.0, 3.0, 3.0],
        "oblique_deg": 0.0,
        "raw_affine": "header, unchecked",
        "reference_image": False,
        "rigid": True,
        "dof": 6,
        "corrections": corrections_info(CORRECTIONS).model_dump(mode="json"),
        "frame": {
            "mean_mm": 0.0,
            "max_mm": 0.0,
            "rotation_deg": 0.0,
            "translation_mm": 0.0,
            "flagged": False,
        },
        "state": "packed",
        "reason": None,
        "detail": None,
        "epi_url": f"/epi/{FULL}#run={'1' * 16}",
    }
    session = {
        "session_uid": "2" * 16,
        "participant_uid": FULL,
        "participant": "sub-01",
        "label": "1",
        "stream_order": 0,
        "stream_rule": "acquisition time",
        "subject_uid": "0" * 16,
        "fs_subject": "sub-01",
        "anchor_run_uid": "1" * 16,
        "anchor_name": "ses-1_task-rest_run-1",
        "anchor_rule": "name_order",
        "anchor_note": None,
        "anchor_packed": True,
        "frame_limit_mm": 10.0,
        "shared_frame_plausible": True,
        "anchor_is_outlier": False,
        "n_grids": 1,
        "problem": None,
        "shown": True,
        "runs": [run],
    }
    page = {
        "subject": _row(),
        "review_url": "/review#subject=" + "0" * 16,
        "rate_url": "/",
        "index_url": "/subjects",
        "epi_index_url": "/epi",
        "sessions": [session],
    }
    shapes.SubjectPageOut.model_validate(page)
    shapes.SubjectsOut.model_validate({"subjects": [_row()], "n_pending": 1})
    participant = _row(kind="participant", subject_uid=None, key=HALF, surface="none")
    shapes.SubjectRowOut.model_validate(participant)
    for wrong in (
        {"surface": "done"},
        {"kind": "session"},
        {"n_runs": -1},
        {"extra": 1},
    ):
        with pytest.raises(ValueError):
            shapes.SubjectRowOut.model_validate(_row(**wrong))
    with pytest.raises(ValueError):
        shapes.FuncRunOut.model_validate(run | {"state": "done"})
    with pytest.raises(ValueError):
        shapes.FuncSessionOut.model_validate(session | {"stream_rule": "label"})


# --- the routes and the rating screen -----------------------------------------------------------


def test_the_four_routes_of_the_subject_pages_are_laid_out() -> None:
    assert schemas.SUBJECT_ROUTES == (
        Route("GET", "/subjects", unblinded=True),
        Route("GET", "/subject/{key}", unblinded=True),
        Route("GET", "/api/subjects", shapes.SubjectsOut, unblinded=True),
        Route("GET", "/api/subjects/{key}", shapes.SubjectPageOut, unblinded=True),
    )
    others = {(r.method, r.path) for r in schemas.ROUTES if r not in schemas.SUBJECT_ROUTES}
    for route in schemas.SUBJECT_ROUTES:
        # a path of its own, and none that the routes of the EPI pages would take for theirs
        assert (route.method, route.path) not in others
        assert "epi" not in route.path
        assert route.unblinded and route.request is None


def test_the_review_columns_of_step_3_are_named_in_its_shape() -> None:
    named = re.findall(r"``(\w+)``", schemas.ReviewOut.__doc__ or "")
    assert named.index("func_sessions") > named.index("n_marks")
    assert named.index("func_packed") == named.index("func_sessions") + 1


def test_the_rating_routes_get_nothing_new() -> None:
    # the link to a subject page is made by the page from what it has (decision D44)
    assert tuple(schemas.SessionOut.model_fields) == (
        "mode",
        "rater",
        "app_version",
        "protocol",
        "protocol_sha256",
        "require_marks",
        "settings",
        "examples",
        "epi_url",
    )
    assert tuple(schemas.AssignmentOut.model_fields) == (
        "assignment_id",
        "kind",
        "subject_uid",
        "media_base",
        "render_key",
        "render",
        "has_vertex_map",
        "previous",
    )


# --- the recorded state of a run ----------------------------------------------------------------


def test_recorded_states_has_its_signature_and_fields() -> None:
    assert tuple(inspect.signature(recorded_states).parameters) == ("public", "pack_hash")
    assert [f.name for f in dataclasses.fields(RecordedState)] == ["state", "reason", "detail"]
    assert RecordedState.__dataclass_params__.frozen  # type: ignore[attr-defined]


def _frames(runs: list[dict[str, Any]], packs: list[dict[str, Any]]) -> Any:
    def frame(columns: Any, rows: list[dict[str, Any]]) -> pd.DataFrame:
        return store._frame(columns, [[row.get(name) for name, _ in columns] for row in rows])

    pack_columns = (*store._PACK_COLUMNS, ("detail", "str"))
    return SimpleNamespace(
        func_runs=lambda: frame(store._FUNC_RUN_COLUMNS, runs),
        packs=lambda: frame(pack_columns, packs),
    )


def _run(uid: str, **changes: Any) -> dict[str, Any]:
    return {
        "run_uid": uid,
        "session_uid": "s" * 16,
        "acquisition_order": int(uid[0], 16),
        "selected": True,
        "packable": True,
        "anchor": False,
        "echoes": 1,
        "present": True,
    } | changes


def test_recorded_states_is_what_the_catalog_records_under_the_present_settings() -> None:
    now, old = "a" * 12, "b" * 12
    public = _frames(
        [
            _run("1" * 16),
            _run("2" * 16),
            _run("3" * 16),
            _run("4" * 16),
            _run("5" * 16, packable=False, reason="multi-echo", detail="3 echoes"),
            _run("6" * 16, selected=False, packable=True, reason="not selected"),
            _run("7" * 16, present=False),
            _run("8" * 16),
        ],
        [
            {"run_uid": "1" * 16, "config_hash": now, "status": "done"},
            {
                "run_uid": "2" * 16,
                "config_hash": now,
                "status": "failed",
                "reason": "pack failure",
                "detail": "OSError",
            },
            {"run_uid": "3" * 16, "config_hash": old, "status": "done"},
            {"run_uid": "5" * 16, "config_hash": now, "status": "done"},
            {"run_uid": "6" * 16, "config_hash": now, "status": "done"},
            {"run_uid": "7" * 16, "config_hash": now, "status": "done"},
            {"run_uid": "8" * 16, "config_hash": now, "status": "running"},
        ],
    )
    states = recorded_states(public, now)
    assert list(states) == ["1" * 16, "2" * 16, "3" * 16, "4" * 16, "5" * 16, "6" * 16, "8" * 16]
    assert states["1" * 16] == RecordedState("packed", None, None)
    assert states["2" * 16] == RecordedState("failed", "pack failure", "OSError")
    assert states["3" * 16] == RecordedState("pending", None, None)  # another pack setting
    assert states["4" * 16] == RecordedState("pending", None, None)
    assert states["5" * 16] == RecordedState("cannot be packed", "multi-echo", "3 echoes")
    assert states["6" * 16] == RecordedState("not selected", "not selected", None)
    assert states["8" * 16] == RecordedState("pending", None, None)
    assert recorded_states(_frames([], []), now) == {}


@pytest.fixture(scope="module")
def packed(tmp_path_factory: pytest.TempPathFactory) -> PackedStream:
    return make_stream(tmp_path_factory.mktemp("epistep3"), n_volumes=10)


def test_recorded_states_on_the_run_of_the_epi_pages(
    app_run: Path, packed: PackedStream, small_run_uids: dict[str, str]
) -> None:
    epi = add_epi(app_run, packed, subject_uid=small_run_uids["sub-001"])
    with PublicCatalog.open(app_run) as public:
        states = recorded_states(public, EpiConfig().pack_hash())
        runs = public.func_runs()
    assert list(states) == list(runs["run_uid"])
    for session, uids in epi.runs.items():
        for uid in uids:
            state = states[uid]
            if uid in epi.packed[session]:
                assert state.state == "packed"
            elif uid in epi.failed:
                assert state == RecordedState("failed", "pack failure", "OSError")
            elif state.state != "cannot be packed":
                assert state.state == "pending"
    assert {s.state for s in states.values()} >= {"packed", "pending", "failed"}
    assert recorded_states(public_on(app_run), "0" * 12).keys() == states.keys()
    assert all(
        s.state in ("pending", "cannot be packed", "not selected")
        for s in recorded_states(public_on(app_run), "0" * 12).values()
    )


def public_on(run_dir: Path) -> Any:
    with PublicCatalog.open(run_dir) as public:
        return _frames(public.func_runs().to_dict("records"), public.packs().to_dict("records"))


# --- func.status and the export -----------------------------------------------------------------


def test_func_status_is_laid_out_for_e3b() -> None:
    assert status.STATUS_PREFIXES == ("func_inventory:", "pack:", "func checks:")
    assert get_args(status.FailureRow) == (str, str, str, str | None)
    signatures = {
        "status_lines": ("ctx",),
        "failure_rows": ("ctx",),
        "unit_names": ("public",),
    }
    for name, parameters in signatures.items():
        function = getattr(status, name)
        assert tuple(inspect.signature(function).parameters) == parameters
        # the stubs until E3.B writes them; E3.B removes these three lines
        with pytest.raises(NotImplementedError, match=r"work package E3\.B: func/status\.py"):
            function(None)


BANNED = {"input_signature", "content_digest", "pack_key", "fd", "dvars", "non_steady", "traces"}


def test_the_functional_export_tables_are_laid_out() -> None:
    assert tables.FUNC_TABLES == ("func_sessions", "func_runs")
    assert not set(tables.FUNC_TABLES) & set(tables.EXPORT_TABLES)
    assert tables.EXPORT_TABLES == (
        "subjects",
        "scores",
        "selection",
        "ratings",
        "marks",
        "final_marks",
        "clusters",
        "datasets",
        "behaviour",
        "skips",
    )
    assert tables.EPI_MANIFEST_KEY == "epi"
    assert tables.MANIFEST_SCHEMA_VERSION == 1
    view = tuple(name for name, _ in store._FUNC_RUN_COLUMNS)
    assert tables.FUNC_RUNS_COLUMNS[: len(view)] == view
    assert tables.FUNC_RUNS_COLUMNS[len(view) :] == (
        "participant_uid",
        "pack_state",
        "pack_reason",
        "pack_detail",
        "pack_n_volumes",
        "pack_bytes",
        "packed_at",
    )
    sessions = {name for name, _ in store._FUNC_SESSION_COLUMNS}
    assert sessions <= set(tables.FUNC_SESSIONS_COLUMNS)
    assert set(tables.FUNC_SESSIONS_COLUMNS) - sessions == {
        "dataset",
        "participant",
        "stream_rule",
        "n_packed",
        "n_failed",
        "n_pending",
    }
    assert tables.FUNC_SESSIONS_COLUMNS[:6] == (
        "dataset",
        "participant_uid",
        "participant",
        "session_uid",
        "session_label",
        "stream_order",
    )
    for columns in (tables.FUNC_SESSIONS_COLUMNS, tables.FUNC_RUNS_COLUMNS):
        assert len(set(columns)) == len(columns)
        assert not set(columns) & BANNED


# --- the fixtures of step 3 ---------------------------------------------------------------------


def test_add_epi_gives_each_session_its_subject_and_every_run_its_facts(
    app_run: Path, packed: PackedStream, small_run_uids: dict[str, str]
) -> None:
    ids = session_ids(packed)
    assert len(ids[FULL]) == len(ids[HALF]) == 2 and not set(ids[FULL]) & set(ids[HALF])
    with pytest.raises(ValueError, match="no such session"):
        add_epi(app_run, packed, subject_uids={"f" * 16: None})
    wanted = {ids[FULL][0]: None, ids[FULL][1]: small_run_uids["sub-000"]}
    wanted[ids[HALF][0]] = small_run_uids["sub-002"]
    epi = add_epi(app_run, packed, subject_uid=small_run_uids["sub-001"], subject_uids=wanted)
    with PublicCatalog.open(epi.run_dir) as public:
        sessions = public.func_sessions()
        runs = public.func_runs()
    got = {
        str(uid): (None if not isinstance(sub, str) else sub)
        for uid, sub in zip(sessions["session_uid"], sessions["subject_uid"], strict=True)
    }
    assert got == wanted | {ids[HALF][1]: None}
    for session, group in runs.groupby("session_uid"):
        first = group["acquisition_order"].min()
        affine = dict(zip(group["acquisition_order"], group["raw_affine"], strict=True))
        assert affine.pop(first) == UNCHECKED, session
        assert set(affine.values()) <= {"header"}
    for text in runs["corrections"]:
        assert CorrectionsInfo.model_validate_json(text) == corrections_info(CORRECTIONS)
    assert set(runs["sdc"]) == {"applied"} and set(runs["slice_timing"]) == {"not applied"}
    assert runs["rigid"].astype(bool).all() and set(runs["dof"]) == {6}
    assert runs["flagged"].fillna(False).astype(bool).any()


def test_a_review_folder_can_have_a_freesurfer_subject_without_functional_data(
    tmp_path: Path,
) -> None:
    dataset = write_review_folder(tmp_path, extra_fs=("sub-03",))
    assert sorted(dataset.subjects) == ["sub-01", "sub-02", "sub-03"]
    copy = dataset.fs_subjects_dir / "sub-03"
    assert dataset.subjects["sub-03"].path == copy
    assert (copy / "surf" / "lh.pial").is_file() and (copy / "mri" / "T1.mgz").is_file()
    assert not (dataset.deriv_root / "sub-03").exists()
    assert not (dataset.raw_root / "sub-03").exists()
    counts = func_counts(dataset)
    assert counts.fs_subjects == ("sub-01", "sub-02", "sub-03")
    assert "sub-03" not in counts.participants
    plain = func_counts(write_review_folder(tmp_path / "plain"))
    assert (counts.runs, counts.packable, counts.reasons) == (
        plain.runs,
        plain.packable,
        plain.reasons,
    )
    for taken in ("sub-02", "sub-01"):
        with pytest.raises(ValueError, match="not a new FreeSurfer subject"):
            write_review_folder(tmp_path / taken, extra_fs=(taken,))
    one = write_review_folder(tmp_path / "one", without_fs=("sub-02",), extra_fs=("sub-04",))
    assert sorted(one.subjects) == ["sub-01", "sub-04"]
