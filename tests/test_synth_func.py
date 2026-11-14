"""The synthetic EPI folder of ``tests/synth_func.py`` holds what it says.

Checked with nibabel, numpy and scipy alone, against the closed forms of the fixture and
with readers written here (an ITK reader, a confounds reader, a report reader), so that these
tests hold while the package's own readers are written (step 1 of docs/epi_interface.md).
"""

from __future__ import annotations

import csv
import html
import json
import math
import re
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from scipy import ndimage
from synth_annex import fetch  # pyrefly: ignore[missing-import]
from synth_func import (  # pyrefly: ignore[missing-import]
    BRAIN_CENTRE,
    DESIGN,
    FLB,
    FSNATIVE_TO_T1W,
    GENERATIONS,
    GRID_AFFINE,
    GRID_B_SHAPE,
    GRID_SHAPE,
    N_VOLUMES,
    PLACEHOLDER_ROLES,
    PLACEHOLDER_RUN,
    RAW_SHAPE,
    SPLINE_TOLERANCE,
    TR_S,
    VERSIONS,
    SyntheticFuncDataset,
    SyntheticRun,
    apply,
    brain_mask,
    expected_pre,
    expected_raw_stream,
    power_fd,
    raw_sample_voxels,
    series_value,
    spline_interior,
    subject_world,
    tree_bytes,
    voxel_centres,
    write_func_dataset,
)

LPS = np.diag([-1.0, -1.0, 1.0, 1.0])


@pytest.fixture(scope="module", params=GENERATIONS)
def dataset(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> SyntheticFuncDataset:
    return write_func_dataset(tmp_path_factory.mktemp("func"), generation=request.param)


def itk_ras(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(RAS point map, centre in LPS) of a single affine ITK text transform, read here."""
    params = fixed = None
    n = 0
    for line in path.read_text().splitlines():
        if line.startswith("Transform:"):
            n += 1
        elif line.startswith("Parameters:"):
            params = np.array(line.split(":", 1)[1].split(), float)
        elif line.startswith("FixedParameters:"):
            fixed = np.array(line.split(":", 1)[1].split(), float)
    assert n == 1 and params is not None and fixed is not None and params.size == 12
    m, t = params[:9].reshape(3, 3), params[9:]
    lps = np.eye(4)
    lps[:3, :3] = m
    lps[:3, 3] = t + fixed - m @ fixed
    return LPS @ lps @ LPS, fixed


def report_line(text: str, label: str) -> str | None:
    found = re.search(r"<li>\s*" + re.escape(label) + r"\s*:\s*(.*?)</li>", text, re.S)
    if found is None:
        return None
    return " ".join(html.unescape(re.sub(r"<[^>]+>", "", found.group(1))).split())


def anchor_of(dataset: SyntheticFuncDataset, run: SyntheticRun) -> SyntheticRun:
    session = dataset.session(run.design.participant, run.design.session)
    return next(r for r in dataset.runs if r.stem == session.anchor)


# --- the layout ---------------------------------------------------------------------------


def test_the_folders_and_the_counts(dataset: SyntheticFuncDataset) -> None:
    raw, deriv = dataset.raw_root, dataset.deriv_root
    assert deriv == raw / "derivatives" / "fmriprep"
    assert dataset.fs_subjects_dir == deriv / "sourcedata" / "freesurfer"
    description = json.loads((deriv / "dataset_description.json").read_text())
    assert description["GeneratedBy"][0] == {
        "Name": "fMRIPrep",
        "Version": VERSIONS[dataset.generation],
    }
    assert json.loads((raw / "dataset_description.json").read_text())["BIDSVersion"]
    # 19 runs in the design, 18 with a raw series, of which one has two echoes
    assert len(DESIGN) == len(dataset.runs) == 19
    raw_files = sorted(raw.glob("sub-*/**/func/*_bold.nii.gz"))
    assert len(raw_files) == 19
    assert sum(r.raw_stem is not None for r in dataset.runs) == 18
    assert len(sorted(deriv.glob("sub-*/**/func/*_space-T1w_desc-preproc_bold.nii.gz"))) == 16
    assert len(sorted(deriv.glob("sub-*/**/func/*_space-MNI*_desc-preproc_bold.nii.gz"))) == 1
    native = sorted(deriv.glob("sub-*/**/func/*_task-*_run-?_desc-preproc_bold.nii.gz"))
    assert len(native) == 2  # the native-only run, and one beside its space-T1w series
    assert len(sorted(deriv.glob("sub-*/figures/*_desc-summary_bold.html"))) == 18
    assert len(sorted(deriv.glob("sub-*/figures/*_desc-validation_bold.html"))) == 18
    assert len(sorted(deriv.glob("sub-*/figures/*_desc-sdc_bold.svg"))) == 1
    assert len(sorted(deriv.glob("sub-*/**/func/*_desc-confounds_timeseries.tsv"))) == 18
    assert [r.packable for r in dataset.runs].count(True) == 11
    reasons = sorted(r.design.reason for r in dataset.runs if not r.packable)
    assert reasons == sorted(
        [
            "multi-echo",
            "native space only",
            "no brain mask",
            "no preprocessed series",
            "no raw series",
            "no transform to the anatomy",
            "standard space only",
            "volume counts differ",
        ]
    )
    assert [(s.participant, s.session) for s in dataset.sessions] == [
        ("sub-01", "1"),
        ("sub-01", "2"),
        ("sub-02", None),
    ]
    assert (raw / "sub-01" / "ses-2" / "func").is_dir() and (raw / "sub-02" / "func").is_dir()
    for name in ("sub-01", "sub-02"):
        subject = dataset.fs_subjects_dir / name
        for part in (
            "mri/T1.mgz",
            "surf/lh.white",
            "surf/rh.white",
            "surf/lh.pial",
            "surf/rh.pial",
        ):
            assert (subject / part).is_file()
        assert dataset.fsnative_files[name].is_file()
        assert dataset.fsnative_files[name].parent == deriv / name / "anat"
    # files that every run of the design names lie where the record says
    for run in dataset.runs:
        for path in run.files.values():
            assert path.is_file(), path


def test_the_names_are_those_of_the_generation(dataset: SyntheticFuncDataset) -> None:
    deriv = dataset.deriv_root
    scanner = sorted(deriv.rglob("*_from-scanner_to-T1w_mode-image_xfm.txt"))
    boldref = sorted(deriv.rglob("*_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt"))
    hmc = sorted(deriv.rglob("*_from-orig_to-boldref_mode-image_desc-hmc_xfm.txt"))
    plain_ref = sorted(deriv.rglob("*_run-?_boldref.nii.gz"))
    coreg_ref = sorted(deriv.rglob("*_desc-coreg_boldref.nii.gz"))
    with_xfm = sum("xfm_to_anat" in r.files for r in dataset.runs)
    with_ref = sum("reference" in r.files for r in dataset.runs)
    assert (with_xfm, with_ref) == (18, 18)
    if dataset.generation == "scanner-T1w":
        assert (len(scanner), len(boldref), len(hmc)) == (with_xfm, 0, 0)
        assert (len(plain_ref), len(coreg_ref)) == (with_ref, 0)
        assert len(sorted(deriv.rglob("*_from-T1w_to-scanner_mode-image_xfm.txt"))) == with_xfm
    else:
        assert (len(scanner), len(boldref), len(hmc)) == (0, with_xfm, with_xfm)
        assert (len(plain_ref), len(coreg_ref)) == (0, with_ref)
        assert len(sorted(deriv.rglob("*_desc-hmc_boldref.nii.gz"))) == with_ref


def test_a_dataset_can_lie_beside_its_derivative(tmp_path: Path) -> None:
    dataset = write_func_dataset(tmp_path, raw_inside=False)
    assert (dataset.raw_root, dataset.deriv_root) == (tmp_path / "raw", tmp_path / "fmriprep")
    assert not (dataset.raw_root / "derivatives").exists()
    assert dataset.fs_subjects_dir == tmp_path / "fmriprep" / "sourcedata" / "freesurfer"


# --- transforms and headers ---------------------------------------------------------------


def test_the_transforms_are_the_known_point_maps_with_a_centre(
    dataset: SyntheticFuncDataset,
) -> None:
    for run in dataset.runs:
        if "xfm_to_anat" not in run.files:
            assert run.design.reason == "no transform to the anatomy"
            continue
        path = run.files["xfm_to_anat"]
        matrix, centre = itk_ras(path)
        np.testing.assert_allclose(matrix, run.anat_to_scanner, atol=1e-12)
        assert np.abs(centre).max() > 1.0  # written with a centre that is not the origin
        # S_r = B_s . D_r, and D_r is the design's displacement about the brain centre
        frame = dataset.session(run.design.participant, run.design.session).frame
        np.testing.assert_allclose(np.linalg.inv(frame) @ matrix, run.displacement, atol=1e-12)
        inverse = sorted(path.parent.glob(run.stem + "_from-T1w_to-*_xfm.txt"))
        assert len(inverse) == 1
        np.testing.assert_allclose(itk_ras(inverse[0])[0] @ matrix, np.eye(4), atol=1e-12)
    for path in dataset.fsnative_files.values():
        np.testing.assert_allclose(itk_ras(path)[0], FSNATIVE_TO_T1W, atol=1e-12)
    centre = np.array([*BRAIN_CENTRE, 1.0])
    for run in dataset.runs:  # the rigid parts are rigid, the 9-dof one is scaled by 0.4 %
        singular = np.linalg.svd(run.displacement[:3, :3], compute_uv=False)
        scale = 1.004 if run.design.dof == 9 else 1.0
        np.testing.assert_allclose(singular, scale, atol=1e-12)
        if not np.any(run.design.displacement[0]) and scale == 1.0:
            np.testing.assert_allclose(
                run.displacement @ centre - centre, [*run.design.displacement[1], 0.0], atol=1e-12
            )


def test_the_raw_headers_and_the_arbiter(dataset: SyntheticFuncDataset) -> None:
    choices = []
    for run in dataset.runs:
        if "raw_bold" not in run.files:
            continue
        img = nib.load(run.files["raw_bold"])
        np.testing.assert_allclose(img.affine, run.header_affine, atol=1e-6)
        np.testing.assert_allclose(img.header.get_qform(), run.header_qform, atol=1e-4)
        assert img.shape[:3] == RAW_SHAPE
        # the arbiter rule, written here: the affine that the reference image agrees with
        if "reference" not in run.files:
            choice = "header, unchecked"
        else:
            reference = nib.load(run.files["reference"])
            assert reference.shape[:3] == img.shape[:3]
            np.testing.assert_allclose(reference.affine, run.raw_affine, atol=1e-6)
            if np.allclose(reference.affine, img.affine, atol=1e-4):
                choice = "header"
            elif np.allclose(reference.affine, img.header.get_qform(), atol=1e-4):
                choice = "qform"
            else:
                choice = "reference"
        assert choice == run.affine_choice, run.stem
        choices.append(choice)
        # the case of H06: qform and sform 20 mm and 5 degrees apart, one of them true
        if run.design.header != "agree":
            gap = np.abs(img.affine - img.header.get_qform())[:3, 3].max()
            assert gap > 4.0
    assert choices.count("qform") == 1  # H06: sub-01 ses-1 task-rest run-1
    assert choices.count("header, unchecked") == 1  # sub-01 ses-1 task-rest run-3
    assert choices.count("header") == len(choices) - 2


def test_the_series_hold_the_field(dataset: SyntheticFuncDataset) -> None:
    raw_voxels = voxel_centres(RAW_SHAPE)
    for run in dataset.runs:
        if "raw_bold" in run.files:
            img = nib.load(run.files["raw_bold"])
            assert img.shape[3] == run.n_raw_volumes == N_VOLUMES
            assert img.header.get_zooms()[3] == pytest.approx(TR_S)
            points = apply(np.linalg.inv(run.anat_to_scanner) @ run.raw_affine, raw_voxels)
            for t in (0, N_VOLUMES - 1):
                got = np.asarray(img.dataobj[..., t]).ravel(order="F")
                np.testing.assert_allclose(got, series_value(points, t), rtol=2e-7, atol=1e-4)
        if (
            run.design.reason in (None, "no brain mask", "volume counts differ")
            and run.design.case != "other grid"
        ):
            img = nib.load(run.files["pre_bold"])
            assert img.shape[:3] == GRID_SHAPE and img.shape[3] == run.n_pre_volumes
            np.testing.assert_allclose(img.affine, GRID_AFFINE, atol=1e-6)
            voxels = voxel_centres(GRID_SHAPE)
            for t in (0, img.shape[3] - 1):
                got = np.asarray(img.dataobj[..., t]).ravel(order="F")
                np.testing.assert_allclose(got, expected_pre(voxels, t), rtol=2e-7, atol=1e-4)
    differ = next(r for r in dataset.runs if r.design.reason == "volume counts differ")
    assert (differ.n_raw_volumes, differ.n_pre_volumes) == (N_VOLUMES, N_VOLUMES - 2)
    other = next(r for r in dataset.runs if r.design.case == "other grid")
    assert nib.load(other.files["pre_bold"]).shape[:3] == GRID_B_SHAPE
    assert nib.load(other.files["pre_mask"]).shape == GRID_B_SHAPE


def test_a_resampled_value_is_known_in_closed_form(dataset: SyntheticFuncDataset) -> None:
    """The raw stream of every packable run through its anchor's transform, sampled here by
    scipy's cubic spline with the edge repeated, equals the closed form inside the raw field
    of view; for the anchor it equals the preprocessed series."""
    for run in dataset.runs:
        if not run.packable:
            continue
        # the run on another grid is sampled on its own grid (decision D38)
        voxels = voxel_centres(GRID_B_SHAPE if run.design.case == "other grid" else GRID_SHAPE)
        mask = np.asarray(nib.load(run.files["pre_mask"]).dataobj).ravel(order="F") > 0
        anchor = anchor_of(dataset, run)
        u = raw_sample_voxels(run, anchor, voxels)
        inside = spline_interior(u)
        assert (inside & mask).sum() > 400, run.stem
        series = nib.load(run.files["raw_bold"])
        for t in (0, 5):
            volume = np.asarray(series.dataobj[..., t], np.float64)
            got = ndimage.map_coordinates(volume, u.T, order=3, mode="nearest")
            want = expected_raw_stream(run, anchor, voxels, t)
            assert np.abs(got - want)[inside].max() < SPLINE_TOLERANCE, run.stem
            if run is anchor or np.allclose(anchor.displacement, run.displacement):
                # the anchor, and a run that did not move against it (the run on another grid)
                np.testing.assert_allclose(want, expected_pre(voxels, t), atol=1e-9)
            else:  # the head moved: the raw stream is not the preprocessed series
                assert np.abs(want - expected_pre(voxels, t))[mask].max() > 0.1


# --- what the derivative states -----------------------------------------------------------


def test_the_brain_mask_and_the_surfaces(dataset: SyntheticFuncDataset) -> None:
    subject = dataset.subjects["sub-01"]
    mask = np.asarray(nib.load(dataset.runs[0].files["pre_mask"]).dataobj) > 0
    np.testing.assert_array_equal(mask, brain_mask(subject, GRID_SHAPE, GRID_AFFINE))
    assert mask.sum() == 1249  # of 4,284 voxels: two ellipsoids of 8 mm voxels
    t1 = nib.load(dataset.fs_subjects_dir / "sub-01" / "mri" / "T1.mgz")
    tkr_to_scanner = t1.affine @ np.linalg.inv(t1.header.get_vox2ras_tkr())
    for hemi in ("lh", "rh"):
        vertices, _ = nib.freesurfer.read_geometry(
            str(dataset.fs_subjects_dir / "sub-01" / "surf" / f"{hemi}.pial")
        )
        world = apply(FSNATIVE_TO_T1W @ tkr_to_scanner, vertices)
        np.testing.assert_allclose(world, subject_world(subject, vertices), atol=1e-4)
        voxel = apply(np.linalg.inv(GRID_AFFINE), world)
        assert (voxel > 0).all() and (voxel < np.asarray(GRID_SHAPE) - 1).all()


def test_the_confounds_tables(dataset: SyntheticFuncDataset) -> None:
    for run in dataset.runs:
        if "confounds" not in run.files:
            assert run.design.reason == "no preprocessed series"
            continue
        with run.files["confounds"].open() as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        assert len(rows) == run.n_pre_volumes
        names = ("trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z")
        params = np.array([[float(row[n]) for n in names] for row in rows])
        np.testing.assert_array_equal(params, run.motion)
        fd = power_fd(params)
        assert rows[0]["framewise_displacement"] == "n/a" and rows[0]["std_dvars"] == "n/a"
        got = np.array([float(row["framewise_displacement"]) for row in rows[1:]])
        np.testing.assert_allclose(got, fd[1:], rtol=0, atol=1e-15)
        assert 0.0 < got.max() < 0.5
        flags = [int(row["non_steady_state_outlier00"]) for row in rows]
        assert flags == [1] + [0] * (len(rows) - 1)


def test_the_summary_reports_in_fmriprep_wording(dataset: SyntheticFuncDataset) -> None:
    seen = set()
    for run in dataset.runs:
        report = run.files.get("summary_report")
        if report is None:
            assert not run.design.report
            continue
        text = report.read_text()
        stc = report_line(text, "Slice timing correction")
        sdc = report_line(text, "Susceptibility distortion correction")
        registration = report_line(text, "Registration")
        assert stc == run.design.stc[dataset.generation]
        assert sdc == run.design.sdc
        assert registration == (
            f"FreeSurfer bbregister (boundary-based registration, BBR) - {run.design.dof} dof"
        )
        assert ("sdc_figure" in run.files) == (sdc != "None")
        seen.add(stc)
    flb = next(r for r in dataset.runs if r.design.sdc == FLB)
    assert report_line(
        flb.files["summary_report"].read_text(), "Susceptibility distortion correction"
    ) == ('FLB ("fieldmap-less", SyN-based)')
    if dataset.generation == "scanner-T1w":
        assert seen == {"Applied", "Not applied"}
    else:
        assert seen == {"Applied", "Not applied", "n/a", "Skipped (too few volumes)"}


def test_the_sidecars_and_the_anchors(dataset: SyntheticFuncDataset) -> None:
    def seconds(text: str) -> float:
        hours, minutes, rest = text.split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(rest)

    for session in dataset.sessions:
        # the rule is decided over every run of the session, packable or not, and the anchor
        # is the first run by it that has a stored transform
        runs = dataset.runs_of(session.participant, session.session)
        times = {}
        for run in runs:
            sidecar = run.files.get("raw_sidecar")
            value = json.loads(sidecar.read_text()).get("AcquisitionTime") if sidecar else None
            times[run.stem] = value
        by_name = sorted(runs, key=lambda r: (r.design.task, int(r.design.run)))
        if all(times.values()):
            assert session.rule == "acquisition_time"
            order = sorted(by_name, key=lambda r: seconds(times[r.stem]))
        else:
            assert session.rule == "name_order"
            order = by_name
        assert session.order == tuple(r.stem for r in order if r.packable)
        assert session.anchor == next(r.stem for r in order if "xfm_to_anat" in r.files)
    # sub-02: the first run by name order is multi-echo and not packed; it is the anchor
    assert dataset.session("sub-02", None).anchor == "sub-02_task-me_run-1"
    # in ses-1 the time rule and the name rule disagree about the anchor
    first_by_name = dataset.run("sub-01", "1", "motor", "1").stem
    assert dataset.session("sub-01", "1").anchor == "sub-01_ses-1_task-rest_run-2" != first_by_name
    # times without leading zeros, as real sidecars write them
    assert dataset.run("sub-01", "1", "rest", "2").design.time == "9:5:3.5"
    # an inherited sidecar at the top of the raw dataset, and a raw name with run-02
    inherited = json.loads((dataset.raw_root / "task-nback_bold.json").read_text())
    assert inherited == {"RepetitionTime": TR_S, "TaskName": "nback"}
    renamed = dataset.run("sub-02", None, "nback", "2")
    assert renamed.raw_stem == "sub-02_task-nback_run-02" and renamed.stem.endswith("_run-2")
    assert renamed.key == "task-nback_run-2" and "raw_sidecar" not in renamed.files
    assert dataset.run("sub-01", "1", "rest", "1").key == "ses-1_task-rest_run-1"


def test_the_frame_displacements(dataset: SyntheticFuncDataset) -> None:
    """Each run's displacement against its anchor over the brain mask, recomputed from the
    written transforms, gives the session's flags and its outlier."""
    mask = np.asarray(nib.load(dataset.runs[0].files["pre_mask"]).dataobj) > 0
    points = apply(GRID_AFFINE, np.argwhere(mask).astype(float))
    for session in dataset.sessions:
        runs = {r.stem: r for r in dataset.runs_of(session.participant, session.session)}
        framed = list(dict.fromkeys([session.anchor, *session.order]))
        matrices = {stem: itk_ras(runs[stem].files["xfm_to_anat"])[0] for stem in framed}

        def largest(a: str, b: str, matrices: dict[str, np.ndarray] = matrices) -> float:
            moved = apply(np.linalg.inv(matrices[a]) @ matrices[b], points)
            return float(np.sqrt(((moved - points) ** 2).sum(axis=1)).max())

        others = [s for s in session.order if s != session.anchor]
        flagged = tuple(s for s in others if largest(session.anchor, s) > 10.0)
        assert flagged == session.flagged
        outlier = len(others) >= 2 and len(flagged) == len(others)
        if outlier:
            outlier = all(
                largest(a, b) <= 10.0 for i, a in enumerate(others) for b in others[i + 1 :]
            )
        assert outlier == session.anchor_is_outlier
    assert dataset.session("sub-01", "1").flagged == ("sub-01_ses-1_task-rest_run-3",)
    assert dataset.session("sub-01", "2").anchor_is_outlier
    assert not dataset.session("sub-02", None).flagged
    shift = dataset.run("sub-01", "1", "motor", "1")
    assert largest_shift(shift, points) == pytest.approx(3.0, abs=1e-9)


def largest_shift(run: SyntheticRun, points: np.ndarray) -> float:
    moved = apply(run.displacement, points)
    return float(np.sqrt(((moved - points) ** 2).sum(axis=1)).max())


# --- placeholders, cost -------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["link", "pointer"])
def test_placeholders_lack_their_content_and_fetch_puts_it_back(tmp_path: Path, kind: str) -> None:
    dataset = write_func_dataset(tmp_path, annex=kind)  # pyrefly: ignore[bad-argument-type]
    run = dataset.run(*PLACEHOLDER_RUN)
    assert len(dataset.placeholders) == len(PLACEHOLDER_ROLES[kind])
    for placeholder, role in zip(dataset.placeholders, PLACEHOLDER_ROLES[kind], strict=True):
        path = run.files[role]
        assert placeholder.path == path
        if kind == "link":
            assert path.is_symlink() and not path.exists()
            assert ".git/annex/objects/" in str(path.readlink())
        else:
            assert not path.is_symlink() and path.stat().st_size <= 32 * 1024
            assert path.read_text().startswith("/annex/objects/MD5E-s")
    for root in (dataset.raw_root, dataset.deriv_root):
        assert (root / ".git" / "annex" / "objects").is_dir()
    fetch(dataset.placeholders)
    assert all(p.path.read_bytes() == p.data for p in dataset.placeholders)


def test_what_one_dataset_costs(tmp_path: Path) -> None:
    start = time.perf_counter()
    dataset = write_func_dataset(tmp_path)
    seconds = time.perf_counter() - start
    size = tree_bytes(dataset.root)
    # measured 2026-11-14 on the shared 4-CPU machine: 0.55 to 0.75 s, 8.2 to 8.7 MB
    assert size < 12_000_000
    assert seconds < 30  # a bound against a runaway, not a measurement
    assert math.isfinite(seconds)
