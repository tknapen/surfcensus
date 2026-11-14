"""The fMRIPrep connector (``surfcensus.func.connectors.fmriprep``) on the synthetic folder of
``tests/synth_func.py`` in both generations of names, and on excerpts written here: summary
reports in the wording of fMRIPrep's ``interfaces/reports.py`` of each version family the spec
names, and confounds tables with the newer and the older column names."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from synth_func import (  # pyrefly: ignore[missing-import]
    FLB,
    FSNATIVE_TO_T1W,
    GENERATIONS,
    MNI,
    TR_S,
    SyntheticFuncDataset,
    SyntheticRun,
    power_fd,
    write_func_dataset,
)

from surfcensus.func import bids
from surfcensus.func.connectors import connector, detect
from surfcensus.func.connectors.base import ConnectorError
from surfcensus.func.connectors.fmriprep import (
    BOLDREF_T1W,
    MOTION_SOURCE,
    SCANNER_T1W,
    FmriprepConnector,
    generation_of,
    read_confounds,
    read_summary_report,
    shear_scale,
)
from surfcensus.records import ConnectorMatch, DerivRun
from surfcensus.sources.base import FuncRole

DERIV_ROLES = {
    role.value
    for role in FuncRole
    if role not in (FuncRole.RAW_BOLD, FuncRole.RAW_SIDECAR, FuncRole.XFM_ANAT_FSNATIVE)
}
DISPLAYED_ELSEWHERE = {"no preprocessed series", "standard space only", "native space only"}


@pytest.fixture(scope="module", params=GENERATIONS)
def dataset(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> SyntheticFuncDataset:
    return write_func_dataset(tmp_path_factory.mktemp("func"), generation=request.param)


def runs_of(dataset: SyntheticFuncDataset) -> dict[str, DerivRun]:
    found = detect(dataset.deriv_root)
    assert found is not None
    made, match = found
    out = {}
    for participant in ("sub-01", "sub-02"):
        for run in made.list_runs(dataset.deriv_root, participant, match):
            assert run.stem not in out
            out[run.stem] = run
    return out


def paths(dataset: SyntheticFuncDataset, run: DerivRun) -> dict[FuncRole, Path]:
    return {role: dataset.deriv_root / ref.location for role, ref in run.files.items()}


# --- versions and detection ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("version", "generation"),
    [
        ("20.2.3", SCANNER_T1W),
        ("21.0.2", SCANNER_T1W),
        ("23.0.2", SCANNER_T1W),
        ("23.1.4", SCANNER_T1W),
        ("23.2.0", BOLDREF_T1W),
        ("23.2.0rc1", BOLDREF_T1W),
        ("23.10.0", BOLDREF_T1W),  # minor numbers compare as numbers
        ("v24.1.1", BOLDREF_T1W),
        ("25.2.3.dev12+g0123abc", BOLDREF_T1W),
        ("100.0", BOLDREF_T1W),
        ("1.5.0", SCANNER_T1W),
        (None, SCANNER_T1W),
        ("", SCANNER_T1W),
        ("latest", SCANNER_T1W),
        ("\u0662\u0663.\u0662", SCANNER_T1W),
    ],
)
def test_the_generation_of_a_version(version: str | None, generation: str) -> None:
    assert generation_of(version) == generation


def test_a_derivative_is_detected_with_its_version(dataset: SyntheticFuncDataset) -> None:
    found = detect(dataset.deriv_root)
    assert found is not None
    made, match = found
    assert made.name == "fmriprep"
    assert match == ConnectorMatch("fmriprep", dataset.version, dataset.generation)
    assert detect(dataset.raw_root) is None  # a raw dataset names no pipeline
    assert detect(dataset.fs_subjects_dir) is None
    assert detect(dataset.root / "absent") is None


def copy_tree(dataset: SyntheticFuncDataset, tmp_path: Path) -> Path:
    """The derivative without its series (small): every name, the text files whole."""
    target = tmp_path / "fmriprep"
    target.mkdir()
    for path in sorted(dataset.deriv_root.rglob("*")):
        if "sourcedata" in path.parts:
            continue
        out = target / path.relative_to(dataset.deriv_root)
        if path.is_dir():
            out.mkdir(parents=True, exist_ok=True)
        elif path.suffix in (".gz", ".nii"):
            out.write_bytes(b"")
        else:
            shutil.copyfile(path, out)
    return target


def test_what_a_description_must_say(dataset: SyntheticFuncDataset, tmp_path: Path) -> None:
    folder = copy_tree(dataset, tmp_path)
    description = folder / "dataset_description.json"
    old = dataset.generation == "scanner-T1w"

    def write(data: object) -> ConnectorMatch | None:
        description.write_text(json.dumps(data))
        return FmriprepConnector().detect(folder)

    # older versions name the pipeline in PipelineDescription (to verify on a real file)
    match = write({"PipelineDescription": {"Name": "fMRIPrep", "Version": "20.1.1"}})
    assert match is not None and match.pipeline_version == "20.1.1"
    # the first entry that names fMRIPrep, whatever the case of its name
    match = write({"GeneratedBy": [{"Name": "MRIQC"}, {"Name": "fmriprep", "Version": "23.2.1"}]})
    assert match == ConnectorMatch("fmriprep", "23.2.1", BOLDREF_T1W)
    # without a version that parses, the listing decides the generation
    for stated in (None, "unknown"):
        entry = {"Name": "fMRIPrep"} | ({} if stated is None else {"Version": stated})
        match = write({"GeneratedBy": [entry]})
        assert match is not None and match.pipeline_version == stated
        assert match.generation == (SCANNER_T1W if old else BOLDREF_T1W)
    for wrong in (
        {"GeneratedBy": [{"Name": "QSIPrep", "Version": "1.0"}]},
        {"Name": "fMRIPrep - fMRI PREProcessing workflow"},  # the title is not the generator
        [{"Name": "fMRIPrep"}],
        {"GeneratedBy": "fMRIPrep"},
    ):
        assert write(wrong) is None
    for text in ("{", "", "\xff", "/annex/objects/MD5E-s10--0123456789abcdef.json\n"):
        description.write_text(text)
        assert FmriprepConnector().detect(folder) is None
    description.unlink()
    assert FmriprepConnector().detect(folder) is None
    # a description without any func/ folder is no functional derivative
    write({"GeneratedBy": [{"Name": "fMRIPrep", "Version": "21.0.2"}]})
    for func in sorted(folder.glob("sub-*/**/func"), reverse=True):
        shutil.rmtree(func)
    assert FmriprepConnector().detect(folder) is None


# --- the listing --------------------------------------------------------------------------


def expected_display(run: SyntheticRun) -> str | None:
    reason = run.design.reason
    return reason if reason in DISPLAYED_ELSEWHERE else None


def test_every_synthetic_run_is_listed_with_its_files(dataset: SyntheticFuncDataset) -> None:
    listed = runs_of(dataset)
    assert sorted(listed) == sorted(r.stem for r in dataset.runs)
    made = connector("fmriprep")
    old = dataset.generation == "scanner-T1w"
    for run in dataset.runs:
        found = listed[run.stem]
        assert found.key == run.key == bids.run_key(found.entities)
        assert found.order == bids.name_order(found.entities)
        assert found.entities.ses == run.design.session and found.echoes == 1
        expected = {role for role in run.files if role in DERIV_ROLES}
        if expected_display(run) is not None:
            expected -= {"pre_bold", "pre_mask", "pre_sidecar"}
        assert {role.value for role in found.files} == expected, run.stem
        for role, ref in found.files.items():
            path = run.files[role.value]
            st = path.stat()
            assert (
                ref.base == "deriv"
                and ref.location == path.relative_to(dataset.deriv_root).as_posix()
            )
            assert (ref.size, ref.etag_or_mtime) == (st.st_size, str(st.st_mtime_ns))
        # which of several candidate names was listed
        if not old and FuncRole.REFERENCE in found.files:
            assert found.files[FuncRole.REFERENCE].variant == "desc-coreg_boldref"
        if not old and FuncRole.PRE_BOLD in found.files:
            assert found.files[FuncRole.PRE_BOLD].variant == "space-T1w_desc-preproc_bold"
            if FuncRole.PRE_MASK in found.files:
                assert found.files[FuncRole.PRE_MASK].variant == "space-T1w_desc-brain_mask"
        if old:
            assert all(ref.variant is None for ref in found.files.values())
        # the spaces of the preprocessed series and the display grid
        reason = run.design.reason
        spaces = {
            "no preprocessed series": (),
            "standard space only": (MNI,),
            "native space only": ("",),
        }.get(reason or "", ("", "T1w") if run.design.native else ("T1w",))
        assert found.spaces == spaces, run.stem
        display = made.display(found)
        if expected_display(run) is None:
            assert (display.layout, display.space, display.reason) == ("anat", "T1w", None)
            assert made.mask(found) == FuncRole.PRE_MASK
        else:
            assert (display.layout, display.reason) == (None, expected_display(run))


def test_the_runs_come_in_name_order(dataset: SyntheticFuncDataset) -> None:
    made = connector("fmriprep")
    match = made.detect(dataset.deriv_root)
    assert match is not None
    runs = made.list_runs(dataset.deriv_root, "sub-02", match)
    assert [r.order for r in runs] == sorted(r.order for r in runs)
    runs = made.list_runs(dataset.deriv_root, "sub-01", match)
    assert [r.entities.ses for r in runs] == ["1"] * 4 + ["2"] * 3
    assert made.list_runs(dataset.deriv_root, "sub-03", match) == []
    assert made.list_runs(dataset.deriv_root, "../sub-01", match) == []


def test_names_that_are_no_run_do_not_break_the_listing(
    dataset: SyntheticFuncDataset, tmp_path: Path
) -> None:
    folder = copy_tree(dataset, tmp_path)
    func = folder / "sub-02" / "func"
    for name in (
        "sub-02_task-rest_run-01_x.txt",
        "sub-02_task-r_e_run-1_bold.nii.gz",
        "sub-02_run-1_desc-preproc_bold.nii.gz",  # no task
        "sub-02_task-rest_run-\u0661_desc-preproc_bold.nii.gz",
        "sub-01_task-rest_run-1_desc-preproc_bold.nii.gz",  # another participant
        "sub-02_ses-1_task-rest_run-1_desc-preproc_bold.nii.gz",  # names a session
        "sub-02_",
        "sub-",
        "_",
        "sub-02_task-rest_run-9.nii.gz",
    ):
        (func / name).write_bytes(b"")
    os.close(os.open(os.fsencode(func) + b"/sub-02_task-r\xe9st_run-1_bold.nii.gz", os.O_CREAT))
    (func / "sub-02_task-dir_run-1_desc-preproc_bold.nii.gz").mkdir()
    os.symlink(tmp_path / "absent", func / "sub-02_task-gone_run-1_desc-preproc_bold.nii.gz")
    made = FmriprepConnector()
    match = made.detect(folder)
    assert match is not None
    stems = [r.stem for r in made.list_runs(folder, "sub-02", match)]
    assert stems == sorted(
        (r.stem for r in dataset.runs if r.design.participant == "sub-02"),
        key=lambda stem: bids.name_order(bids.parse_entities(stem)),
    )


def test_a_derivative_folder_that_cannot_be_listed_is_an_error(
    dataset: SyntheticFuncDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    made = FmriprepConnector()
    match = made.detect(dataset.deriv_root)
    assert match is not None
    func = dataset.deriv_root / "sub-02" / "func"
    real = os.scandir

    def scandir(path: object) -> object:
        if Path(str(path)) == func:
            raise PermissionError(13, "Permission denied", str(path))
        return real(path)  # pyrefly: ignore[no-matching-overload]

    monkeypatch.setattr(bids.os, "scandir", scandir)
    with pytest.raises(ConnectorError) as raised:
        made.list_runs(dataset.deriv_root, "sub-02", match)
    assert raised.value.reason == "unreadable"
    assert raised.value.detail == "sub-02/func: Permission denied"


def test_the_transform_to_fsnative_of_a_session(
    dataset: SyntheticFuncDataset, tmp_path: Path
) -> None:
    made = FmriprepConnector()
    match = made.detect(dataset.deriv_root)
    assert match is not None
    for participant, session in (("sub-01", "1"), ("sub-01", "2"), ("sub-02", None)):
        files = made.session_files(dataset.deriv_root, participant, session, match)
        ref = files[FuncRole.XFM_ANAT_FSNATIVE]
        assert (
            ref.location
            == f"{participant}/anat/{participant}_from-T1w_to-fsnative_mode-image_xfm.txt"
        )
        matrix = made.anat_to_fsnative(
            {FuncRole.XFM_ANAT_FSNATIVE: dataset.deriv_root / ref.location}
        )
        np.testing.assert_allclose(np.asarray(matrix), FSNATIVE_TO_T1W, atol=1e-12)
    # the session's own anat/ folder comes first
    folder = copy_tree(dataset, tmp_path)
    anat = folder / "sub-01" / "ses-2" / "anat"
    anat.mkdir()
    shutil.copyfile(
        dataset.fsnative_files["sub-01"],
        anat / "sub-01_ses-2_from-T1w_to-fsnative_mode-image_xfm.txt",
    )
    files = made.session_files(folder, "sub-01", "2", match)
    assert files[FuncRole.XFM_ANAT_FSNATIVE].location.startswith("sub-01/ses-2/anat/")
    files = made.session_files(folder, "sub-01", "1", match)
    assert files[FuncRole.XFM_ANAT_FSNATIVE].location.startswith("sub-01/anat/")
    assert made.session_files(folder, "sub-03", None, match) == {}
    with pytest.raises(ConnectorError) as raised:
        made.anat_to_fsnative({})
    assert raised.value.reason == "invalid"


# --- what the files say -------------------------------------------------------------------


def test_the_stored_transform_of_every_run(dataset: SyntheticFuncDataset) -> None:
    made = FmriprepConnector()
    listed = runs_of(dataset)
    for run in dataset.runs:
        found = listed[run.stem]
        if FuncRole.XFM_TO_ANAT not in found.files:
            assert run.design.reason == "no transform to the anatomy"
            with pytest.raises(ConnectorError):
                made.raw_to_anat(found, paths(dataset, found))
            continue
        path = made.raw_to_anat(found, paths(dataset, found))
        np.testing.assert_allclose(np.asarray(path.matrix), run.anat_to_scanner, atol=1e-11)
        assert path.source == FuncRole.XFM_TO_ANAT
        assert path.reference == (FuncRole.REFERENCE if run.design.reference else None)
        assert path.dof == (run.design.dof if run.design.report else None)
        scale = run.design.displacement[2]
        assert path.shear_scale == pytest.approx(abs(scale - 1.0), abs=1e-9)
        assert path.rigid == (scale == 1.0)


def test_shear_and_scale() -> None:
    def matrix(a: np.ndarray) -> tuple:
        m = np.eye(4)
        m[:3, :3] = a
        return tuple(tuple(float(x) for x in row) for row in m)

    assert shear_scale(matrix(np.eye(3))) == 0.0
    assert shear_scale(matrix(np.diag([1.0, 1.02, 0.97]))) == pytest.approx(0.03)
    rng = np.random.default_rng(3)
    for _ in range(20):
        a = rng.normal(size=(3, 3))
        expected = np.abs(np.linalg.svd(a, compute_uv=False) - 1).max()
        assert shear_scale(matrix(a)) == pytest.approx(expected, rel=1e-9, abs=1e-9)
    shear = np.eye(3)
    shear[0, 1] = 0.01
    assert shear_scale(matrix(shear)) > 1e-3


def test_the_motion_of_every_run(dataset: SyntheticFuncDataset) -> None:
    made = FmriprepConnector()
    listed = runs_of(dataset)
    for run in dataset.runs:
        found = listed[run.stem]
        motion = made.motion(found, paths(dataset, found))
        if run.n_pre_volumes is None:
            assert motion is None
            continue
        assert motion is not None and motion.source == FuncRole.CONFOUNDS
        assert len(motion.fd) == len(motion.dvars) == run.n_pre_volumes
        fd = power_fd(run.motion)
        assert motion.fd[0] is None and motion.dvars[0] is None
        np.testing.assert_allclose(np.asarray(motion.fd[1:], float), fd[1:], rtol=0, atol=1e-12)
        assert motion.dvars[1:] == tuple(1.0 + 0.125 * t for t in range(1, run.n_pre_volumes))
        assert motion.non_steady == (0,)


def test_the_corrections_of_every_run(dataset: SyntheticFuncDataset) -> None:
    made = FmriprepConnector()
    listed = runs_of(dataset)
    for run in dataset.runs:
        found = listed[run.stem]
        c = made.corrections(found, paths(dataset, found))
        assert (c.motion.state, c.motion.note) == ("applied", MOTION_SOURCE)
        assert c.sdc_figure_listed == (run.design.sdc != "None")
        if not run.design.report:
            for statement in (c.sdc, c.slice_timing, c.registration):
                assert (
                    statement.state == "unknown"
                    and statement.note == "no summary report of the run"
                )
            assert c.dof is None
            continue
        words = run.design.stc[dataset.generation]
        state = {
            "Applied": "applied",
            "Not applied": "not applied",
            "Skipped (too few volumes)": "not applied",
            "n/a": "unknown",
        }[words]
        assert (c.slice_timing.state, c.slice_timing.words) == (state, words)
        assert c.slice_timing.source == FuncRole.SUMMARY_REPORT
        if run.design.sdc == "None":
            assert (c.sdc.state, c.sdc_method) == ("not applied", None)
        else:
            assert (c.sdc.state, c.sdc_method, c.sdc.note) == ("applied", FLB, None)
        assert c.registration.state == "applied" and c.dof == run.design.dof
        assert c.registration.words == (
            f"FreeSurfer bbregister (boundary-based registration, BBR) - {run.design.dof} dof"
        )


def test_the_repetition_time(dataset: SyntheticFuncDataset, tmp_path: Path) -> None:
    made = FmriprepConnector()
    run = runs_of(dataset)["sub-02_task-nback_run-1"]
    files = paths(dataset, run)
    assert made.repetition_time(run, files) == TR_S
    # without a sidecar, or with one that says nothing usable, the header gives it
    sidecar = tmp_path / "sidecar.json"
    for text in ('{"RepetitionTime": true}', '{"RepetitionTime": -2}', "{", "[]"):
        sidecar.write_text(text)
        assert made.repetition_time(run, files | {FuncRole.PRE_SIDECAR: sidecar}) == TR_S
    without = {role: path for role, path in files.items() if role != FuncRole.PRE_SIDECAR}
    assert made.repetition_time(run, without) == TR_S
    # a header in milliseconds
    img = nib.Nifti1Image(np.zeros((2, 2, 2, 3), np.float32), np.eye(4))
    img.header.set_zooms((1.0, 1.0, 1.0, 1500.0))
    img.header.set_xyzt_units("mm", "msec")
    nib.save(img, tmp_path / "ms.nii.gz")
    assert made.repetition_time(run, {FuncRole.PRE_BOLD: tmp_path / "ms.nii.gz"}) == 1.5
    (tmp_path / "bad.nii.gz").write_bytes(b"not a NIfTI file")
    assert made.repetition_time(run, {FuncRole.PRE_BOLD: tmp_path / "bad.nii.gz"}) is None
    assert made.repetition_time(run, {}) is None


# --- reports, in fMRIPrep's wording ---------------------------------------------------------


def report(family: str, stc: str, sdc: str, registration: str) -> str:
    """The summary segment of ``FUNCTIONAL_TEMPLATE`` of ``interfaces/reports.py``: 20.2.3
    (no orientation line, a confounds section), 21.0.2 to 23.0.2 (an orientation line and a
    confounds section), 23.2.0 and later (no confounds section)."""
    orientation = "" if family == "20.2" else "\t\t\t<li>Original orientation: LAS</li>\n"
    confounds = (
        "\t\t<details>\n\t\t\t<summary>Confounds collected</summary><br />\n"
        "\t\t\t<p>global_signal, csf, white_matter, framewise_displacement.</p>\n"
        "\t\t</details>\n"
        if family in ("20.2", "21.0")
        else ""
    )
    return (
        '\t\t<details open>\n\t\t<summary>Summary</summary>\n\t\t<ul class="elem-desc">\n'
        f"{orientation}"
        "\t\t\t<li>Repetition time (TR): 2s</li>\n"
        "\t\t\t<li>Phase-encoding (PE) direction: Anterior-Posterior</li>\n"
        "\t\t\t<li>Single-echo EPI sequence.</li>\n"
        f"\t\t\t<li>Slice timing correction: {stc}</li>\n"
        f"\t\t\t<li>Susceptibility distortion correction: {sdc}</li>\n"
        f"\t\t\t<li>Registration: {registration}</li>\n"
        "\t\t\t<li>Non-steady-state volumes: 2 (Confirmed: 2 automatically detected)</li>\n"
        "\t\t</ul>\n\t\t</details>\n"
        f"{confounds}"
    )


BBR = "FreeSurfer <code>bbregister</code> (boundary-based registration, BBR) - {} dof"
COREG = "FreeSurfer <code>mri_coreg</code> - {} dof"
FLIRT_BBR = "FSL <code>flirt</code> with boundary-based registration (BBR) metric - {} dof"
FLIRT = "FSL <code>flirt</code> rigid registration - 6 dof"
PRECOMPUTED = "Precomputed affine transformation"


@pytest.mark.parametrize(
    ("family", "stc", "sdc", "registration", "expected"),
    [
        # slice timing: Applied, Not applied and Skipped in every version (20.2.3 reports.py
        # lines 183 to 185), n/a from 23.2.0 (lines 234 to 241)
        ("20.2", "Applied", "None", BBR.format(6), ("applied", "not applied", "applied", 6)),
        ("21.0", "Not applied", FLB, BBR.format(9), ("not applied", "applied", "applied", 9)),
        (
            "21.0",
            "Skipped (too few volumes)",
            "None",
            COREG.format(6),
            ("not applied", "not applied", "applied", 6),
        ),
        ("23.2", "n/a", "None", FLIRT_BBR.format(12), ("unknown", "not applied", "applied", 12)),
        ("23.2", "Applied", FLB, FLIRT, ("applied", "applied", "applied", 6)),
        (
            "25.2",
            "Not applied",
            "None",
            PRECOMPUTED,
            ("not applied", "not applied", "applied", None),
        ),
    ],
)
def test_report_lines_in_fmripreps_wording(
    family: str, stc: str, sdc: str, registration: str, expected: tuple
) -> None:
    c = read_summary_report(report(family, stc, sdc, registration), None)
    assert (c.slice_timing.state, c.sdc.state, c.registration.state, c.dof) == expected
    assert c.slice_timing.words == stc and c.slice_timing.source == FuncRole.SUMMARY_REPORT
    assert c.sdc.words == sdc and c.sdc_method == (None if sdc == "None" else sdc)
    assert c.registration.words == registration.replace("<code>", "").replace("</code>", "")
    fallback = registration in (COREG.format(6), FLIRT)
    assert (c.registration.note is not None) == fallback
    if stc.startswith("Skipped"):
        assert c.slice_timing.note == "fMRIPrep skipped it"
    if stc == "n/a":
        assert "n/a" in (c.slice_timing.note or "")


def test_what_a_report_does_not_say_is_unknown() -> None:
    c = read_summary_report("<html><body>no list here</body></html>", None)
    for statement, label in (
        (c.slice_timing, "Slice timing correction"),
        (c.sdc, "Susceptibility distortion correction"),
        (c.registration, "Registration"),
    ):
        assert statement.state == "unknown" and statement.source is None
        assert statement.note == f"the summary report has no line '{label}'"
    c = read_summary_report(report("21.0", "Sometimes", "", ""), True)
    assert (c.slice_timing.state, c.slice_timing.words) == ("unknown", "Sometimes")
    assert c.sdc.state == "unknown" and c.registration.state == "unknown"
    assert c.sdc_figure_listed is True
    # the listing's figure beside the report's words
    c = read_summary_report(report("21.0", "Applied", "None", BBR.format(6)), True)
    assert c.sdc.state == "not applied" and c.sdc.note is not None
    c = read_summary_report(report("21.0", "Applied", FLB, BBR.format(6)), False)
    assert c.sdc.state == "applied" and c.sdc.note is not None
    # entities, tags, line breaks and case
    text = report("21.0", "applied", "FMB (fieldmap-based) &amp;\n <b>phase</b>", BBR.format(6))
    c = read_summary_report(text.replace("Slice timing", "slice TIMING"), None)
    assert c.slice_timing.state == "applied"
    assert c.sdc_method == "FMB (fieldmap-based) & phase"
    for nonsense in ("", "<li>" * 10_000, "\x00\uffff<li>Registration:</li>"):
        read_summary_report(nonsense, None)


def test_a_report_that_cannot_be_read_gives_unknown(
    dataset: SyntheticFuncDataset, tmp_path: Path
) -> None:
    made = FmriprepConnector()
    run = runs_of(dataset)["sub-01_ses-1_task-rest_run-3"]  # with the distortion figure
    files = paths(dataset, run)
    for content in (b"\xff\xfe", b""):
        (tmp_path / "r.html").write_bytes(content)
        c = made.corrections(run, files | {FuncRole.SUMMARY_REPORT: tmp_path / "r.html"})
        assert c.slice_timing.state == "unknown"
    c = made.corrections(run, files | {FuncRole.SUMMARY_REPORT: tmp_path / "absent.html"})
    assert c.slice_timing.note == "the summary report of the run cannot be read"
    # without a report the listed distortion figure says that distortion correction ran
    assert c.sdc.state == "applied" and c.sdc.source == FuncRole.SDC_FIGURE
    without = {role: path for role, path in files.items() if role != FuncRole.SUMMARY_REPORT}
    c = made.corrections(run, without)
    assert (c.sdc.state, c.sdc_method, c.slice_timing.state) == ("applied", None, "unknown")


# --- confounds tables -----------------------------------------------------------------------


def test_confounds_with_the_newer_and_the_older_names(tmp_path: Path) -> None:
    path = tmp_path / "sub-01_task-rest_desc-confounds_timeseries.tsv"
    path.write_text(
        "global_signal\tstd_dvars\tdvars\tframewise_displacement\t"
        "non_steady_state_outlier00\tnon_steady_state_outlier01\n"
        "1\tn/a\tn/a\tn/a\t1\t0\n"
        "2\t1.25\t20\t0.0625\t0\t1\n"
        "3\tnan\t21\t0.125\t0\t0\n"
        "4\t1.5\n"
    )
    motion = read_confounds(path)
    assert motion.fd == (None, 0.0625, 0.125, None)
    assert motion.dvars == (None, 1.25, None, 1.5)
    assert motion.non_steady == (0, 1) and motion.source == FuncRole.CONFOUNDS
    # the CamelCase names of tables before fMRIPrep renamed them (to verify)
    path.write_text(
        "stdDVARS\tFramewiseDisplacement\tNonSteadyStateOutlier00\nn/a\tn/a\t1\n2\t0.5\t0\n"
    )
    motion = read_confounds(path)
    assert (motion.fd, motion.dvars, motion.non_steady) == ((None, 0.5), (None, 2.0), (0,))
    # no non-steady-state column: no such volume
    path.write_text("framewise_displacement\tstd_dvars\n0.1\t1.0\n")
    assert read_confounds(path).non_steady == ()


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"framewise_displacement\n0.1\n",
        b"trans_x\tstd_dvars\n0\t1\n",
        b"/annex/objects/MD5E-s4096--0123456789abcdef0123456789abcdef.tsv\n",
        b"\xff\xfe\x00\x01",
    ],
)
def test_a_table_that_is_not_one(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "sub-01_desc-confounds_timeseries.tsv"
    path.write_bytes(content)
    with pytest.raises(ConnectorError) as raised:
        read_confounds(path)
    assert raised.value.reason == "invalid" and raised.value.detail.startswith(path.name)
    assert str(tmp_path) not in raised.value.detail


def test_a_table_that_cannot_be_read(tmp_path: Path) -> None:
    with pytest.raises(ConnectorError) as raised:
        read_confounds(tmp_path / "absent.tsv")
    assert raised.value.reason == "unreadable" and raised.value.detail.startswith("absent.tsv: ")
    assert str(tmp_path) not in raised.value.detail


# --- placeholders ---------------------------------------------------------------------------


def test_placeholders_are_no_files_here(tmp_path: Path) -> None:
    linked = write_func_dataset(tmp_path / "link", annex="link")
    made = FmriprepConnector()
    match = made.detect(linked.deriv_root)
    assert match is not None
    run = next(
        r
        for r in made.list_runs(linked.deriv_root, "sub-01", match)
        if r.stem == "sub-01_ses-2_task-rest_run-2"
    )
    # a git-annex link without content is not listed; step 2 names it as not fetched
    assert FuncRole.PRE_BOLD not in run.files and FuncRole.CONFOUNDS not in run.files
    raw = {r.stem for r in bids.list_raw_runs(linked.raw_root, "sub-01")}
    assert "sub-01_ses-2_task-rest_run-2" not in raw
    pointed = write_func_dataset(tmp_path / "pointer", annex="pointer")
    match = made.detect(pointed.deriv_root)
    assert match is not None
    run = next(
        r
        for r in made.list_runs(pointed.deriv_root, "sub-01", match)
        if r.stem == "sub-01_ses-2_task-rest_run-2"
    )
    files = paths(pointed, run)
    c = made.corrections(run, files)
    assert c.slice_timing.state == "unknown"
    with pytest.raises(ConnectorError) as raised:
        made.motion(run, files)
    assert raised.value.reason == "invalid"


# --- with the session plan of package E1.B --------------------------------------------------


def test_the_listings_give_the_plan_the_reasons_of_the_design(
    dataset: SyntheticFuncDataset,
) -> None:
    from surfcensus.func.session import plan_session

    made = connector("fmriprep")
    match = made.detect(dataset.deriv_root)
    assert match is not None
    for participant, session in (("sub-01", "1"), ("sub-01", "2"), ("sub-02", None)):
        deriv = [
            r
            for r in made.list_runs(dataset.deriv_root, participant, match)
            if r.entities.ses == session
        ]
        raw = [
            r
            for r in bids.list_raw_runs(dataset.raw_root, participant)
            if r.entities.ses == session
        ]
        ids = {r.key: f"{i:016x}" for i, r in enumerate(deriv)}
        try:
            plan = plan_session(
                participant,
                session,
                deriv,
                raw,
                made,
                dataset.deriv_root,
                dataset.raw_root,
                ids,
                participant,
                10.0,
            )
        except NotImplementedError:
            pytest.skip("func/session.py (package E1.B) is not written yet")
        # every reason that the listing and the headers decide; a run on another grid is
        # E1.B's to rule on (the spec has it both as a reason and as a group of its own)
        reasons = {r.deriv.stem: r.reason for r in plan.runs}
        expected = {r.stem: r.design.reason for r in dataset.runs_of(participant, session)}
        for stem, reason in expected.items():
            if reason != "runs on different grids":
                assert reasons[stem] == reason, stem
