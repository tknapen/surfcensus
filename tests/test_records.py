"""Tests of surfcensus/records.py: the plain records shared between modules."""

from __future__ import annotations

import dataclasses
import pickle
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import pytest
from synth_tree import write_light_subject  # pyrefly: ignore[missing-import]

from surfcensus import records
from surfcensus.config import new_local_config
from surfcensus.inventory import provenance
from surfcensus.records import (
    PhaseReport,
    Provenance,
    RenderResult,
    RenderRow,
    RunContext,
    WorkRow,
)
from surfcensus.render import RenderConfig, render_subject
from surfcensus.sources.patterns import resolve_subject_files

WORK_COLUMNS = (
    "phase",
    "unit",
    "config_hash",
    "input_signature",
    "status",
    "reason",
    "detail",
    "output",
    "surfcensus_version",
    "finished_at",
)
RENDER_COLUMNS = (
    "subject_uid",
    "config_hash",
    "input_signature",
    "tile_hash",
    "render_key",
    "video",
    "status",
    "reason",
    "detail",
    "media_path",
    "n_frames",
    "template_id",
    "rendered_at",
)


def names(record: type) -> list[str]:
    return [field.name for field in dataclasses.fields(record)]


def test_the_rows_have_the_columns_of_their_tables() -> None:
    assert sorted(names(WorkRow)) == sorted(WORK_COLUMNS)
    assert sorted(names(RenderRow)) == sorted(RENDER_COLUMNS)


def test_a_row_to_be_recorded_needs_only_what_the_caller_knows() -> None:
    done = WorkRow("regional", "0123456789abcdef", "abcdefabcdef", "0011223344556677", "done")
    assert (done.reason, done.detail, done.output) == (None, None, None)
    assert done.surfcensus_version is None and done.finished_at is None
    failed = RenderRow(
        "0123456789abcdef",
        "abcdefabcdef",
        "0011223344556677",
        video=False,
        status="failed",
        reason="missing surface",
        detail="lh_pial",
    )
    assert failed.tile_hash is None and failed.render_key is None and failed.n_frames is None
    assert failed.rendered_at is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        failed.status = "done"  # pyrefly: ignore[read-only]


def test_provenance_is_what_the_inventory_stores_and_is_re_exported() -> None:
    assert provenance.Provenance is Provenance
    assert names(Provenance) == [
        "build_stamp",
        "fs_version",
        "fs_major",
        "flags",
        "n_invocations",
        "finished",
        "t2pial",
        "flairpial",
        "hires",
        "expert",
        "xopts",
        "edited",
        "pial_variant",
        "holes",
        "truncations",
        "participant",
        "longitudinal",
    ]
    unknown = Provenance(
        build_stamp=None,
        fs_version=None,
        fs_major=None,
        flags=(),
        n_invocations=None,
        finished=None,
        t2pial=None,
        flairpial=None,
        hires=None,
        expert=None,
        xopts=None,
        edited=None,
        pial_variant="pial",
        holes=None,
        truncations={"lh": None, "rh": (93, 434)},
        participant="sub-01",
        longitudinal=False,
    )
    assert unknown == dataclasses.replace(unknown)
    assert unknown.truncations["rh"] == (93, 434)


def test_a_phase_report_counts_units_and_names_failures() -> None:
    report = PhaseReport("render", done=2, skipped=5, failed=1, failures=(("ab" * 8, "x"),))
    assert report.notes == () and report.seconds == 0.0
    assert names(PhaseReport) == [
        "phase",
        "done",
        "skipped",
        "failed",
        "failures",
        "notes",
        "seconds",
    ]


def test_a_render_result_crosses_a_process_boundary(tmp_path: Path) -> None:
    write_light_subject(tmp_path / "sub-01")
    info = render_subject(
        resolve_subject_files(tmp_path / "sub-01"),
        tmp_path / "media",
        RenderConfig(panel_px=64),
        video=False,
    )
    bare = RenderResult(info)
    assert bare.extras == {}
    rows = pd.DataFrame({"subject_uid": ["0123456789abcdef"], "hemi": ["lh"]})
    result = RenderResult(info, {"intensity": rows})
    back = pickle.loads(pickle.dumps(result))
    assert back.info == info
    pd.testing.assert_frame_equal(back.extras["intensity"], rows)
    # each result has its own extras
    assert RenderResult(info).extras is not bare.extras


def test_a_run_context_holds_the_public_side_of_a_run(tmp_path: Path) -> None:
    assert names(RunContext) == [
        "run_dir",
        "config",
        "public",
        "lease",
        "render_key",
        "scrub",
        "func_lease",  # step 2 of the EPI interface (E2.0), None by default
        "pack_key",
    ]
    public = object()  # stands for a PublicCatalog, which this module names for the checker only
    context = RunContext(
        run_dir=tmp_path,
        config=new_local_config(tmp_path),
        public=public,  # pyrefly: ignore[bad-argument-type]
        lease=lambda uid, roles: nullcontext({}),
        render_key=lambda tile_hash, digests: "0" * 16,
        scrub=lambda text: text,
    )
    assert context.public is public
    with context.lease("0123456789abcdef", ()) as paths:
        assert paths == {}
    assert not any(name.startswith("_catalog") or name == "catalog" for name in names(RunContext))


def test_the_module_names_other_modules_types_for_the_checker_only() -> None:
    for name in ("PublicCatalog", "RunConfig", "RenderInfo", "pd", "PackInfo", "SessionInfo"):
        assert not hasattr(records, name)
