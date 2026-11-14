"""One registration per session (``func.session``, work package E1.B): the anchor, the order of
the session's stream, the raw affine, the frame check and the plan of a session, against the
closed forms of ``tests/synth_func.py`` and a recomputation in the test with nibabel and numpy.

What the frame check reports is a statement about the raw data (D36): how far a run lies,
drawn through the anchor's transform, from where its own stored transform puts it.

``plan_session`` is tested with a small connector written here that answers from the fixture's
known values; the test that uses the fMRIPrep connector of package E1.A skips while that
connector is not written. Two rules of ``func.bids`` (E1.A) are used by the session module: the
reading of ``AcquisitionTime`` and of a sidecar. While those are not written, the tests put in
their place the rule of stage 1 (``prototypes/func_viewer/stage1/session.py``), written here.

The grid ruling in force: the runs of a session whose preprocessed series lie on different
grids are all packed, each grid a group of its own; no run gets ``runs on different grids``.
The fixture still labels its run on another grid with that reason, so the expected plan is
computed here from the design with that run packable.
"""

from __future__ import annotations

import datetime
import json
import math
import re
from collections.abc import Iterator, Mapping
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from synth_func import (  # pyrefly: ignore[missing-import]
    FSNATIVE_TO_T1W,
    GRID_AFFINE,
    GRID_B_SHAPE,
    GRID_SHAPE,
    SyntheticFuncDataset,
    SyntheticRun,
    apply,
    rigid,
    write_func_dataset,
)

from surfcensus.func import bids, func_id, session
from surfcensus.func.geometry import grid_of
from surfcensus.records import (
    NO_PREPROCESSED_SERIES,
    RUNS_ON_DIFFERENT_GRIDS,
    STANDARD_SPACE_ONLY,
    AnchorCandidate,
    ConnectorMatch,
    Corrections,
    DerivRun,
    Display,
    FuncFileRef,
    Grid,
    Matrix4,
    Motion,
    RawRun,
    RigidPath,
    RunEntities,
    Statement,
    to_matrix4,
)
from surfcensus.sources.base import FuncRole

# --- the rules of func.bids, while E1.A has not written them ------------------------------


def _stage1_seconds(value: object) -> tuple[str, float] | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    m = re.fullmatch(r"(\d{1,2}):(\d{1,2}):(\d{1,2}(?:\.\d+)?)", text)
    if m:
        h, minute, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
        return ("time", h * 3600 + minute * 60 + s) if h < 24 and minute < 60 and s < 61 else None
    try:
        stamp = datetime.datetime.fromisoformat(text)
    except ValueError:
        return None
    if "T" not in text and " " not in text:
        return None
    epoch = datetime.datetime(1970, 1, 1)
    return "datetime", (stamp.replace(tzinfo=None) - epoch).total_seconds()


def _stage1_sidecar_time(path: Path) -> str | None:
    try:
        value = json.loads(Path(path).read_text()).get("AcquisitionTime")
    except (OSError, ValueError, AttributeError):
        return None
    return value if isinstance(value, str) else None


def _written(function: object, *args: object) -> bool:
    try:
        function(*args)  # pyrefly: ignore[not-callable]
    except NotImplementedError:
        return False
    except Exception:  # noqa: BLE001  # written, and refusing this input
        return True
    return True


@pytest.fixture(autouse=True)
def bids_rules(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    if not _written(bids.acquisition_seconds, "9:5:3.5"):
        monkeypatch.setattr(bids, "acquisition_seconds", _stage1_seconds)
    if not _written(bids.sidecar_time, Path("/nonexistent.json")):
        monkeypatch.setattr(bids, "sidecar_time", _stage1_sidecar_time)
    yield


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> SyntheticFuncDataset:
    return write_func_dataset(tmp_path_factory.mktemp("func"))


def m4(matrix: np.ndarray) -> Matrix4:
    return to_matrix4(matrix)


# --- the anchor and the order -------------------------------------------------------------


def cand(run: str, order: tuple, time: str | None) -> AnchorCandidate:
    return AnchorCandidate(run=run, order=order, acquisition_time=time)


def test_the_anchor_by_acquisition_time() -> None:
    runs = [
        cand("a", ("motor", 1), "9:20:1.25"),
        cand("b", ("rest", 1), "9:12:40.0"),
        cand("c", ("rest", 2), "9:5:3.5"),  # without leading zeros, as sidecars write it
        cand("d", ("rest", 3), "9:31:10"),
    ]
    anchor = session.pick_anchor(runs)
    assert (anchor.run, anchor.rule) == ("c", session.RULE_TIME)
    assert "9:5:3.5" in anchor.why
    assert session.acquisition_order(runs, anchor) == ("c", "b", "a", "d")
    # the order of the input does not matter
    assert session.acquisition_order(runs[::-1], session.pick_anchor(runs[::-1])) == (
        "c",
        "b",
        "a",
        "d",
    )


def test_the_anchor_by_name_when_a_time_is_missing_or_unreadable() -> None:
    for missing in (None, "not a time", "25:00:00", "2026-11-14"):
        runs = [
            cand("x", ("rest", 2), "9:00:00"),
            cand("y", ("nback", 3), missing),
            cand("z", ("nback", 1), "8:00:00"),
        ]
        anchor = session.pick_anchor(runs)
        assert (anchor.run, anchor.rule) == ("z", session.RULE_NAME), missing
        assert "of 3 runs" in anchor.why
        assert session.acquisition_order(runs, anchor) == ("z", "y", "x")


def test_times_of_two_kinds_go_by_name() -> None:
    runs = [cand("x", ("a", 2), "9:00:00"), cand("y", ("a", 1), "2026-11-14T08:00:00")]
    anchor = session.pick_anchor(runs)
    assert (anchor.run, anchor.rule) == ("y", session.RULE_NAME)
    assert "time of day" in anchor.why


def test_equal_times_go_by_name_and_iso_times_are_read() -> None:
    runs = [
        cand("late", ("a", 1), "2026-11-14T10:00:00"),
        cand("b2", ("b", 2), "2026-11-14T09:00:00"),
        cand("b1", ("b", 1), "2026-11-14T09:00:00"),
    ]
    anchor = session.pick_anchor(runs)
    assert (anchor.run, anchor.rule) == ("b1", session.RULE_TIME)
    assert session.acquisition_order(runs, anchor) == ("b1", "b2", "late")


def test_the_order_is_never_rearranged_around_the_anchor() -> None:
    """The stream of a session is its runs in acquisition order (D38): an anchor that the
    rule would not put first is refused, not moved to the front."""
    runs = [cand("a", ("x", 1), "9:00:00"), cand("b", ("x", 2), "8:00:00")]
    anchor = session.pick_anchor(runs)
    assert session.acquisition_order(runs, anchor) == ("b", "a")
    wrong = type(anchor)(run="a", rule=anchor.rule, why="")
    with pytest.raises(ValueError, match="first"):
        session.acquisition_order(runs, wrong)
    with pytest.raises(ValueError, match="not among"):
        session.acquisition_order(runs, type(anchor)(run="q", rule=anchor.rule, why=""))
    with pytest.raises(ValueError):
        session.pick_anchor([])


# --- the raw affine -----------------------------------------------------------------------


def test_the_reference_image_decides_the_raw_affine(dataset: SyntheticFuncDataset) -> None:
    """Each run of the design: the choice it was written for, and the affine taken is the one
    the raw values were made with (``R_r``), also where the header's sform is wrong (H06)."""
    seen = set()
    for run in dataset.runs:
        if "raw_bold" not in run.files:
            continue
        found = session.raw_affine_choice(run.files["raw_bold"], run.files.get("reference"))
        assert found.choice == run.affine_choice, run.stem
        assert found.reference_image == ("reference" in run.files)
        np.testing.assert_allclose(np.array(found.matrix), run.raw_affine, atol=1e-4)
        seen.add(found.choice)
    assert seen == {"header", "qform", "header, unchecked"}
    h06 = dataset.run("sub-01", "1", "rest", "1")
    assert np.abs(h06.header_affine - h06.raw_affine).max() > 1.0  # the sform is wrong there


def test_a_reference_that_agrees_with_neither_gives_its_own_affine(
    dataset: SyntheticFuncDataset, tmp_path: Path
) -> None:
    run = dataset.run("sub-01", "1", "rest", "2")
    ref_img = nib.load(run.files["reference"])
    moved = ref_img.affine.copy()
    moved[:3, 3] += 3.0
    path = tmp_path / "ref_boldref.nii.gz"
    nib.save(nib.Nifti1Image(np.asarray(ref_img.dataobj), moved), path)
    found = session.raw_affine_choice(run.files["raw_bold"], path)
    assert found.choice == "reference"
    np.testing.assert_array_equal(np.array(found.matrix), nib.load(path).affine)
    # just inside the tolerance it is the header's
    near = ref_img.affine.copy()
    near[0, 3] += 0.5 * session.AFFINE_TOLERANCE
    nib.save(nib.Nifti1Image(np.asarray(ref_img.dataobj), near), path)
    assert session.raw_affine_choice(run.files["raw_bold"], path).choice == "header"
    # a reference on another grid leaves it unchecked, with the reference seen
    nib.save(nib.Nifti1Image(np.zeros((3, 3, 3), np.float32), ref_img.affine), path)
    found = session.raw_affine_choice(run.files["raw_bold"], path)
    assert (found.choice, found.reference_image) == ("header, unchecked", True)


# --- the frame check ----------------------------------------------------------------------


def independent_displacement(
    run_matrix: np.ndarray, anchor_matrix: np.ndarray, mask: np.ndarray, affine: np.ndarray
) -> tuple[float, float]:
    points = apply(affine, np.argwhere(mask).astype(float))
    moved = apply(np.linalg.inv(anchor_matrix) @ run_matrix, points)
    dist = np.sqrt(((moved - points) ** 2).sum(axis=1))
    return float(dist.mean()), float(dist.max())


def test_a_shift_and_a_rotation_in_closed_form() -> None:
    mask = np.zeros((9, 9, 9), bool)
    mask[2:7, 2:7, 2:7] = True
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    anchor = rigid((3.0, -2.0, 5.0), (4.0, 5.0, -6.0))
    shift = anchor @ rigid((0, 0, 0), (3.0, 0.0, 4.0))  # the run's head moved 5 mm in the anatomy
    d = session.frame_displacement(m4(shift), m4(anchor), mask, affine)
    assert d.mean_mm == pytest.approx(5.0, abs=1e-9) and d.max_mm == pytest.approx(5.0, abs=1e-9)
    assert d.rotation_deg == pytest.approx(0.0, abs=1e-6)
    assert d.translation_ras == pytest.approx((3.0, 0.0, 4.0), abs=1e-9)
    assert d.shear_scale == pytest.approx(0.0, abs=1e-12) and d.n_voxels == 125
    assert not d.flagged
    turned = anchor @ rigid((0.0, 0.0, 10.0), (0.0, 0.0, 0.0), centre=(8.0, 8.0, 8.0))
    d = session.frame_displacement(m4(turned), m4(anchor), mask, affine)
    assert d.rotation_deg == pytest.approx(10.0, abs=1e-9)
    assert d.translation_mm == pytest.approx(0.0, abs=1e-9)  # about the mask's centroid
    mean, top = independent_displacement(turned, anchor, mask, affine)
    assert (d.mean_mm, d.max_mm) == pytest.approx((mean, top), abs=1e-9)
    scaled = anchor @ rigid((0, 0, 0), (0, 0, 0), centre=(8.0, 8.0, 8.0), scale=1.01)
    assert session.frame_displacement(m4(scaled), m4(anchor), mask, affine).shear_scale == (
        pytest.approx(0.01, abs=1e-9)
    )
    with pytest.raises(ValueError, match="empty"):
        session.frame_displacement(m4(shift), m4(anchor), np.zeros((2, 2, 2), bool), affine)


def test_the_flag_is_above_the_limit_not_at_it() -> None:
    mask = np.zeros((5, 5, 5), bool)
    mask[1:4, 1:4, 1:4] = True
    affine = np.eye(4)
    anchor = np.eye(4)
    at = rigid((0, 0, 0), (0.0, 10.0, 0.0), centre=(0, 0, 0))
    d = session.frame_displacement(m4(at), m4(anchor), mask, affine, limit_mm=10.0)
    assert d.max_mm == 10.0 and not d.flagged
    above = rigid((0, 0, 0), (0.0, 10.0 + 1e-9, 0.0), centre=(0, 0, 0))
    assert session.frame_displacement(m4(above), m4(anchor), mask, affine, 10.0).flagged
    assert session.frame_displacement(m4(at), m4(anchor), mask, affine, 9.999).flagged


def test_the_frame_checks_of_the_design(dataset: SyntheticFuncDataset) -> None:
    """The fixture's sessions: the flagged runs, the outlier anchor, and every displacement
    recomputed here over the run's brain mask."""
    for expected in dataset.sessions:
        runs = {r.stem: r for r in dataset.runs_of(expected.participant, expected.session)}
        transforms = {stem: m4(runs[stem].anat_to_scanner) for stem in expected.order}
        masks = {}
        for stem in expected.order:
            img = nib.load(runs[stem].files["pre_mask"])
            masks[stem] = (np.asarray(img.dataobj) > 0, img.affine)
        check = session.check_frame(transforms, expected.anchor, masks)
        assert check.anchor == expected.anchor and check.limit_mm == 10.0
        assert check.flagged == expected.flagged
        assert check.shared_frame_plausible == (not expected.flagged)
        assert check.anchor_is_outlier == expected.anchor_is_outlier
        assert list(check.runs) == list(expected.order)
        own = check.runs[expected.anchor]
        assert (own.mean_mm, own.max_mm, own.flagged) == (0.0, 0.0, False)
        for stem in expected.order:
            if stem == expected.anchor:
                continue
            mean, top = independent_displacement(
                runs[stem].anat_to_scanner,
                runs[expected.anchor].anat_to_scanner,
                *masks[stem],
            )
            got = check.runs[stem]
            assert (got.mean_mm, got.max_mm) == pytest.approx((mean, top), rel=1e-9, abs=1e-9)
    ses2 = dataset.session("sub-01", "2")
    assert ses2.anchor_is_outlier and len(ses2.flagged) == 2  # the design's outlier case


def test_an_outlier_needs_runs_that_agree_with_each_other() -> None:
    mask = np.ones((3, 3, 3), bool)
    eye = np.eye(4)
    far = {
        "a": eye,
        "b": rigid((0, 0, 0), (0, 0, 20.0), centre=(0, 0, 0)),
        "c": rigid((0, 0, 0), (0, 0, -20.0), centre=(0, 0, 0)),
    }
    masks = {k: (mask, eye) for k in far}
    check = session.check_frame({k: m4(v) for k, v in far.items()}, "a", masks)
    assert check.flagged == ("b", "c") and not check.anchor_is_outlier  # b and c are 40 mm apart
    one = {"a": eye, "b": far["b"]}
    assert not session.check_frame({k: m4(v) for k, v in one.items()}, "a", masks).anchor_is_outlier
    with pytest.raises(ValueError):
        session.check_frame({"a": m4(eye)}, "z", masks)


# --- the plan of a session ----------------------------------------------------------------


KEY = b"test key for the ids of runs"


def entities(stem: str) -> RunEntities:
    parts = dict(p.split("-", 1) for p in stem.split("_") if "-" in p)
    return RunEntities(
        sub=parts["sub"],
        ses=parts.get("ses"),
        task=parts.get("task"),
        run=parts.get("run"),
        echo=parts.get("echo"),
    )


def order_of(stem: str) -> tuple[str | int, ...]:
    e = entities(stem)
    return (e.task or "", "", "", "", "", int(e.run or 0), int(e.echo or 0))


def ref(role: FuncRole, base: str, root: Path, path: Path) -> FuncFileRef:
    stat = path.stat()
    return FuncFileRef(
        role=role,
        base=base,  # pyrefly: ignore[bad-argument-type]
        location=path.relative_to(root).as_posix(),
        size=stat.st_size,
        etag_or_mtime=str(stat.st_mtime_ns),
    )


DERIV_ROLES = (
    FuncRole.PRE_BOLD,
    FuncRole.PRE_SIDECAR,
    FuncRole.PRE_MASK,
    FuncRole.XFM_TO_ANAT,
    FuncRole.REFERENCE,
    FuncRole.CONFOUNDS,
    FuncRole.SUMMARY_REPORT,
    FuncRole.VALIDATION_REPORT,
    FuncRole.SDC_FIGURE,
)


def deriv_run(dataset: SyntheticFuncDataset, run: SyntheticRun) -> DerivRun:
    files = {
        role: ref(role, "deriv", dataset.deriv_root, run.files[role.value])
        for role in DERIV_ROLES
        if role.value in run.files
    }
    spaces = ("T1w",) if run.design.reason not in ("native space only",) else ("",)
    return DerivRun(
        entities=entities(run.stem),
        key=run.key,
        order=order_of(run.stem),
        stem=run.stem,
        spaces=spaces,
        files=files,
    )


def raw_runs(dataset: SyntheticFuncDataset, run: SyntheticRun) -> list[RawRun]:
    out = []
    names = [k for k in run.files if k.startswith("raw_bold_echo")] or ["raw_bold"]
    if "raw_bold" not in run.files:
        return []
    for name in names:
        path = run.files[name]
        stem = path.name[: -len("_bold.nii.gz")]
        files = {FuncRole.RAW_BOLD: ref(FuncRole.RAW_BOLD, "raw", dataset.raw_root, path)}
        sidecar = path.with_name(stem + "_bold.json")
        if sidecar.exists():
            files[FuncRole.RAW_SIDECAR] = ref(
                FuncRole.RAW_SIDECAR, "raw", dataset.raw_root, sidecar
            )
        out.append(
            RawRun(
                entities=entities(stem), key=run.key, order=order_of(stem), stem=stem, files=files
            )
        )
    return out


class FakeConnector:
    """A connector that answers from the fixture's known values (``S_r``), not from its files."""

    name = "fake"

    def __init__(self, dataset: SyntheticFuncDataset) -> None:
        self.dataset = dataset
        self.by_stem = {r.stem: r for r in dataset.runs}
        self.calls: list[str] = []

    def detect(self, folder: Path) -> ConnectorMatch | None:
        return (
            ConnectorMatch("fake", "1.0", "scanner-T1w")
            if folder == self.dataset.deriv_root
            else None
        )

    def list_runs(self, folder: Path, participant: str, match: ConnectorMatch) -> list[DerivRun]:
        return [
            deriv_run(self.dataset, r)
            for r in self.dataset.runs
            if r.design.participant == participant
        ]

    def session_files(
        self, folder: Path, participant: str, session: str | None, match: ConnectorMatch
    ) -> Mapping[FuncRole, FuncFileRef]:
        path = self.dataset.fsnative_files[participant]
        return {FuncRole.XFM_ANAT_FSNATIVE: ref(FuncRole.XFM_ANAT_FSNATIVE, "deriv", folder, path)}

    def display(self, run: DerivRun) -> Display:
        reason = self.by_stem[run.stem].design.reason
        if reason == "standard space only":
            return Display(None, reason=STANDARD_SPACE_ONLY)
        if reason == "native space only":
            return Display("native", space="")
        if reason == "no preprocessed series":
            return Display(None, reason=NO_PREPROCESSED_SERIES)
        return Display("anat", space="T1w")

    def mask(self, run: DerivRun) -> FuncRole:
        return FuncRole.PRE_MASK

    def raw_to_anat(self, run: DerivRun, files: Mapping[FuncRole, Path]) -> RigidPath:
        self.calls.append(run.stem)
        known = self.by_stem[run.stem]
        return RigidPath(
            matrix=m4(known.anat_to_scanner),
            dof=known.design.dof,
            shear_scale=0.0,
            rigid=True,
            source=FuncRole.XFM_TO_ANAT,
            reference=FuncRole.REFERENCE if FuncRole.REFERENCE in files else None,
        )

    def anat_to_fsnative(self, files: Mapping[FuncRole, Path]) -> Matrix4:
        return m4(FSNATIVE_TO_T1W)

    def motion(self, run: DerivRun, files: Mapping[FuncRole, Path]) -> Motion | None:
        return None

    def corrections(self, run: DerivRun, files: Mapping[FuncRole, Path]) -> Corrections:
        unknown = Statement("unknown", None, None)
        return Corrections(unknown, unknown, None, unknown, unknown, None, None)

    def repetition_time(self, run: DerivRun, files: Mapping[FuncRole, Path]) -> float | None:
        return None


def expected_reason(run: SyntheticRun) -> str | None:
    """The fixture's reason under the grid ruling: a run on another grid is packed."""
    return None if run.design.reason == RUNS_ON_DIFFERENT_GRIDS else run.design.reason


def seconds(text: str) -> float:
    h, m, s = text.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def expected_order(dataset: SyntheticFuncDataset, participant: str, ses: str | None) -> list[str]:
    packable = [r for r in dataset.runs_of(participant, ses) if expected_reason(r) is None]
    by_name = sorted(packable, key=lambda r: (r.design.task, int(r.design.run)))
    if all(r.design.time is not None and "raw_sidecar" in r.files for r in packable):
        return [r.stem for r in sorted(by_name, key=lambda r: seconds(r.design.time or ""))]
    return [r.stem for r in by_name]


def plan_for(dataset: SyntheticFuncDataset, connector: object, participant: str, ses: str | None):  # noqa: ANN201
    mine = dataset.runs_of(participant, ses)
    deriv = [deriv_run(dataset, r) for r in mine]
    raw = [x for r in mine for x in raw_runs(dataset, r)]
    ids = {r.key: func_id(KEY, r.key) for r in mine}
    plan = session.plan_session(
        participant,
        ses,
        deriv,
        raw,
        connector,  # pyrefly: ignore[bad-argument-type]
        dataset.deriv_root,
        dataset.raw_root,
        ids,
        participant,
    )
    return plan, ids


@pytest.mark.parametrize("key", [("sub-01", "1"), ("sub-01", "2"), ("sub-02", None)])
def test_the_plan_of_each_session(dataset: SyntheticFuncDataset, key: tuple) -> None:
    participant, ses = key
    connector = FakeConnector(dataset)
    plan, ids = plan_for(dataset, connector, participant, ses)
    mine = {r.stem: r for r in dataset.runs_of(participant, ses)}
    stem_of = {ids[r.key]: r.stem for r in mine.values()}
    assert (plan.participant, plan.session, plan.fs_subject) == (participant, ses, participant)
    order = expected_order(dataset, participant, ses)
    packed = [r for r in plan.runs if r.reason is None]
    # the stream of the session (D38): the packable runs first, in acquisition order exactly
    assert [stem_of[r.run] for r in packed] == order
    assert [stem_of[r.run] for r in plan.runs[: len(packed)]] == order
    rest = [stem_of[r.run] for r in plan.runs[len(packed) :]]
    assert rest == sorted(rest, key=lambda s: order_of(s))
    assert plan.anchor is not None and stem_of[plan.anchor.run] == order[0]
    rule = "acquisition_time" if participant == "sub-01" else "name_order"
    assert plan.anchor.rule == rule
    for item in plan.runs:
        run = mine[stem_of[item.run]]
        assert item.reason == expected_reason(run), run.stem
        assert item.name == run.name and item.deriv.stem == run.stem
        assert (item.raw is None) == ("raw_bold" not in run.files)
    # the frame check is that of the transforms the connector gave, the anchor's for every run
    assert plan.frame is not None
    assert [stem_of[r] for r in plan.frame.runs] == order
    assert sorted(connector.calls) == sorted(order)
    fixture = dataset.session(participant, ses)
    assert tuple(stem_of[r] for r in plan.frame.flagged) == fixture.flagged
    assert plan.frame.anchor_is_outlier == fixture.anchor_is_outlier
    # the grids: one per distinct grid, named by its first run's
    assert set(plan.grid_of) == {r.run for r in packed}
    for run_id, key_ in plan.grid_of.items():
        grid = plan.grids[key_]
        assert grid == grid_of(mine[stem_of[run_id]].files["pre_bold"])
    assert FuncRole.XFM_ANAT_FSNATIVE in plan.files


def test_runs_on_two_grids_are_two_groups(dataset: SyntheticFuncDataset) -> None:
    plan, ids = plan_for(dataset, FakeConnector(dataset), "sub-02", None)
    other = next(r for r in dataset.runs if r.design.case == "other grid")
    assert {g.shape for g in plan.grids.values()} == {GRID_SHAPE, GRID_B_SHAPE}
    key_b = plan.grid_of[ids[other.key]]
    assert plan.grids[key_b].shape == GRID_B_SHAPE
    assert [r for r, k in plan.grid_of.items() if k == key_b] == [ids[other.key]]
    assert Grid(GRID_SHAPE, m4(GRID_AFFINE)).key in plan.grids
    assert all(r.reason != RUNS_ON_DIFFERENT_GRIDS for r in plan.runs)


def test_a_run_whose_mask_is_on_another_grid_has_no_brain_mask(
    dataset: SyntheticFuncDataset, tmp_path: Path
) -> None:
    import shutil

    moved = write_func_dataset(tmp_path)
    run = moved.run("sub-02", None, "nback", "1")
    other = next(r for r in moved.runs if r.design.case == "other grid")
    shutil.copyfile(other.files["pre_mask"], run.files["pre_mask"])
    plan, ids = plan_for(moved, FakeConnector(moved), "sub-02", None)
    item = next(r for r in plan.runs if r.run == ids[run.key])
    assert item.reason == "no brain mask"
    assert plan.anchor is not None and plan.anchor.run != ids[run.key]


def test_the_plan_with_the_fmriprep_connector(dataset: SyntheticFuncDataset) -> None:
    """The plan through the connector of package E1.A and its BIDS listing equals the plan
    through the fake connector. Skips while E1.A's functions are not written."""
    from surfcensus.func.connectors import fmriprep

    connector = fmriprep.make()
    try:
        match = connector.detect(dataset.deriv_root)
        assert match is not None
        deriv = connector.list_runs(dataset.deriv_root, "sub-01", match)
        raw = bids.list_raw_runs(dataset.raw_root, "sub-01")
    except NotImplementedError as error:
        pytest.skip(f"not written yet: {error}")
    mine = [r for r in deriv if r.entities.ses == "1"]
    ids = {r.key: func_id(KEY, r.key) for r in mine}
    plan = session.plan_session(
        "sub-01",
        "1",
        mine,
        [r for r in raw if r.entities.ses == "1"],
        connector,
        dataset.deriv_root,
        dataset.raw_root,
        ids,
        "sub-01",
    )
    fake, fake_ids = plan_for(dataset, FakeConnector(dataset), "sub-01", "1")
    assert [r.name for r in plan.runs] == [r.name for r in fake.runs]
    assert [r.reason for r in plan.runs] == [r.reason for r in fake.runs]
    assert plan.anchor is not None and fake.anchor is not None
    assert (plan.anchor.rule, plan.grid_of.values().__len__()) == (
        fake.anchor.rule,
        len(fake.grid_of),
    )
    assert plan.frame is not None and fake.frame is not None
    for a, b in zip(plan.frame.runs.values(), fake.frame.runs.values(), strict=True):
        assert a.max_mm == pytest.approx(b.max_mm, abs=1e-6)
    assert math.isfinite(plan.frame.limit_mm)
