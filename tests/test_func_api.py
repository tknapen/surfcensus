"""``func.api``: the folders found by rule, and the sessions of one subject packed into a
directory without a run directory (docs/epi_interface.md, Step 1; decisions D36 and D38).

The synthetic dataset of ``synth_func.py`` is read in place. What a pack holds is checked
against a recomputation in this file from the files alone (an ITK reader and the arbiter rule
written here, nibabel, ``scipy.ndimage.map_coordinates``), as stage 1's ``check_geometry.py``
does; only the 8-bit rule of the format (``pack.quantise``) and the decoder (``packread``) are
the package's.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from scipy import ndimage
from synth_annex import snapshot_links  # pyrefly: ignore[missing-import]
from synth_func import (  # pyrefly: ignore[missing-import]
    SyntheticFuncDataset,
    SyntheticRun,
    write_func_dataset,
)

from surfcensus.config import ConfigError, EpiConfig
from surfcensus.func import PACK_JSON, SESSION_JSON, STREAM_JSON, api, packread
from surfcensus.func.model import PackInfo, SessionInfo, StreamInfo, read_info
from surfcensus.func.pack import quantise
from surfcensus.sources.base import SourceError

KEY = b"a fixed key of the api tests"
LPS = np.diag([-1.0, -1.0, 1.0, 1.0])


# --- fixtures and helpers -------------------------------------------------------------------


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> SyntheticFuncDataset:
    return write_func_dataset(tmp_path_factory.mktemp("func-api") / "data")


@pytest.fixture(scope="module")
def packed(
    dataset: SyntheticFuncDataset, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Path]:
    """Both participants packed once, each into a directory of its own, with the fixed key."""
    out = {}
    before = snapshot_links(dataset.root)
    for participant in ("sub-01", "sub-02"):
        target = tmp_path_factory.mktemp(f"out-{participant}")
        api.epi_subject(dataset.raw_root, participant, target, key=KEY)
        out[participant] = target
    assert snapshot_links(dataset.root) == before
    return out


def stream_of(out: Path) -> StreamInfo:
    info = read_info(out / STREAM_JSON, StreamInfo)
    assert info is not None
    return info


def sessions_of(out: Path) -> list[SessionInfo]:
    infos = []
    for sid in stream_of(out).sessions:
        info = read_info(out / sid / SESSION_JSON, SessionInfo)
        assert info is not None
        infos.append(info)
    return infos


def pack_of(out: Path, session: SessionInfo, name: str) -> tuple[Path, PackInfo]:
    run = next(r for r in session.runs if r.name == name)
    info = read_info(out / session.session / run.run / PACK_JSON, PackInfo)
    assert info is not None, name
    return out / session.session, info


def itk_ras(path: Path) -> np.ndarray:
    """An ITK text transform as a RAS point map (nitransforms' convention), read here."""
    params = fixed = None
    for line in path.read_text().splitlines():
        if line.startswith("Parameters:"):
            params = np.array(line.split(":", 1)[1].split(), float)
        elif line.startswith("FixedParameters:"):
            fixed = np.array(line.split(":", 1)[1].split(), float)
    assert params is not None and fixed is not None
    m = np.eye(4)
    m[:3, :3] = params[:9].reshape(3, 3)
    m[:3, 3] = params[9:] + fixed - m[:3, :3] @ fixed
    return LPS @ m @ LPS


def arbitrated_affine(raw: Path, reference: Path | None) -> np.ndarray:
    """The raw affine by the rule of stage 1, written here: the header's when the reference
    image agrees, else the qform when it agrees, else the reference's; the header's without
    a reference."""
    img = nib.load(raw)
    if reference is None:
        return np.asarray(img.affine, float)
    ref = np.asarray(nib.load(reference).affine, float)
    if np.abs(img.affine - ref).max() <= 1e-4:
        return np.asarray(img.affine, float)
    qform, code = img.get_qform(coded=True)
    if code and np.abs(qform - ref).max() <= 1e-4:
        return np.asarray(qform, float)
    return ref


def independent_raw_volume(
    run: SyntheticRun, anchor: SyntheticRun, t: int, window: tuple[float, float]
) -> np.ndarray:
    pre = nib.load(run.files["pre_bold"])
    shape = pre.shape[:3]
    grid = np.asarray(pre.affine, float)
    affine = arbitrated_affine(run.files["raw_bold"], run.files.get("reference"))
    s_a = itk_ras(anchor.files["xfm_to_anat"])
    ijk = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1).reshape(-1, 3)
    homogeneous = np.c_[ijk, np.ones(len(ijk))]
    u = (np.linalg.inv(affine) @ s_a @ grid @ homogeneous.T)[:3]
    raw = nib.load(run.files["raw_bold"])
    volume = np.asarray(raw.dataobj[..., t], np.float64)
    values = ndimage.map_coordinates(volume, u, order=3, mode="nearest")
    top = np.asarray(volume.shape, float)[:, None] - 1
    values[((u < 0) | (u > top)).any(axis=0)] = 0.0
    return quantise(values.reshape(shape), *window)


def write_copy(tmp_path: Path, **kwargs: object) -> SyntheticFuncDataset:
    return write_func_dataset(tmp_path / "data", **kwargs)  # pyrefly: ignore[bad-argument-type]


# --- the folders ----------------------------------------------------------------------------


def test_the_folders_of_every_starting_point(dataset: SyntheticFuncDataset) -> None:
    deriv, raw, fs = dataset.deriv_root, dataset.raw_root, dataset.fs_subjects_dir
    for start in (raw, raw / "derivatives", deriv, fs, fs / "sub-01"):
        found = api.find_folders(start)
        assert found.deriv_root.resolve() == deriv.resolve(), start
        assert found.raw_root is not None and found.raw_root.resolve() == raw.resolve(), start
        assert found.fs_subjects_dir is not None
        assert found.fs_subjects_dir.resolve() == fs.resolve(), start
        assert (found.match.connector, found.match.pipeline_version) == ("fmriprep", "21.0.2")
        assert found.match.generation == "scanner-T1w"


def test_the_second_generation_is_found(tmp_path: Path) -> None:
    ds = write_copy(tmp_path, generation="boldref-T1w")
    found = api.find_folders(ds.raw_root)
    assert found.match.generation == "boldref-T1w" and found.match.pipeline_version == "25.2.3"


def test_side_by_side_needs_raw_and_takes_it(tmp_path: Path) -> None:
    ds = write_copy(tmp_path, raw_inside=False)
    found = api.find_folders(ds.deriv_root)
    assert found.raw_root is None and found.fs_subjects_dir is not None
    found = api.find_folders(ds.deriv_root, raw=ds.raw_root)
    assert found.raw_root == ds.raw_root
    with pytest.raises(SourceError) as err:
        api.find_folders(ds.deriv_root, raw=ds.deriv_root / "sub-01")
    assert err.value.reason == "layout" and "raw BIDS" in err.value.detail
    with pytest.raises(SourceError) as err:
        api.find_folders(ds.deriv_root, raw=tmp_path / "nowhere")
    assert err.value.reason == "not found"


def test_a_derivative_without_freesurfer_subjects_and_raw_in_sourcedata(tmp_path: Path) -> None:
    ds = write_copy(tmp_path, raw_inside=False)
    shutil.rmtree(ds.fs_subjects_dir)
    (ds.deriv_root / "sourcedata").rmdir()
    shutil.move(str(ds.raw_root), str(ds.deriv_root / "sourcedata" / "raw"))
    found = api.find_folders(ds.deriv_root)
    assert found.fs_subjects_dir is None
    assert found.raw_root == ds.deriv_root / "sourcedata" / "raw"
    # a BIDS root whose derivatives/ holds such a derivative, by the folder's content
    root = tmp_path / "bids"
    (root / "derivatives").mkdir(parents=True)
    shutil.move(str(ds.deriv_root), str(root / "derivatives" / "fmriprep-21.0.2"))
    found = api.find_folders(root)
    assert found.deriv_root == root / "derivatives" / "fmriprep-21.0.2"


def test_no_derivative_says_where_it_looked(tmp_path: Path) -> None:
    (tmp_path / "ds" / "derivatives" / "other").mkdir(parents=True)
    with pytest.raises(SourceError) as err:
        api.find_folders(tmp_path / "ds")
    assert err.value.reason == "layout"
    assert "looked at ., derivatives/other" in err.value.detail
    assert "fmriprep" in err.value.detail
    with pytest.raises(SourceError) as err:
        api.find_folders(tmp_path / "missing")
    assert err.value.reason == "not found"


# --- what is packed -------------------------------------------------------------------------


def test_the_stream_and_the_sessions_of_sub_01(
    dataset: SyntheticFuncDataset, packed: dict[str, Path]
) -> None:
    stream = stream_of(packed["sub-01"])
    assert stream.label == "sub-01" and stream.order == "session label"
    infos = sessions_of(packed["sub-01"])
    assert [i.label for i in infos] == ["sub-01 ses-1", "sub-01 ses-2"]
    for info in infos:
        want = dataset.session("sub-01", info.label.split("ses-")[1])
        names = {r.stem: r.name for r in dataset.runs}
        assert [r.name for r in info.runs if r.reason is None] == [names[s] for s in want.order]
        assert info.runs[0].anchor and info.runs[0].name == names[want.anchor]
        assert info.registration is not None
        assert info.registration.rule == want.rule
        assert info.registration.anchor_is_outlier == want.anchor_is_outlier
        flagged = {r.name for r in info.runs if r.frame is not None and r.frame.flagged}
        assert flagged == {names[s] for s in want.flagged}
        assert info.contours_from == "sub-01"
        assert all(g.contours is not None for g in info.grids)
        # one stretch of the stream: first indices one after the other
        firsts = [r.first for r in info.runs if r.reason is None]
        assert firsts == [12 * n for n in range(len(firsts))]
    # the pack of a run that is not the anchor lists the anchor's transform and the surfaces
    _, pack = pack_of(packed["sub-01"], infos[0], "ses-1_task-rest_run-1")
    roles = [i.role for i in pack.inputs]
    assert "anchor_xfm_to_anat" in roles and "xfm_anat_fsnative" in roles
    assert {"t1", "lh_white", "rh_white", "lh_pial", "rh_pial"} <= set(roles)
    assert {i.base for i in pack.inputs if i.role == "lh_pial"} == {"freesurfer"}
    _, anchor = pack_of(packed["sub-01"], infos[0], "ses-1_task-rest_run-2")
    assert "anchor_xfm_to_anat" not in [i.role for i in anchor.inputs]


def test_every_reason_of_sub_02_and_two_grids_in_one_session(
    dataset: SyntheticFuncDataset, packed: dict[str, Path]
) -> None:
    (info,) = sessions_of(packed["sub-02"])
    assert info.label == "sub-02"
    got = {r.name: r.reason for r in info.runs}
    want = {r.name: r.design.reason for r in dataset.runs_of("sub-02", None)}
    assert got == want
    assert got["task-rest_run-5"] is None  # on another grid, packed on its own
    assert len(info.grids) == 2
    grid_of = {r.name: r.grid_key for r in info.runs if r.reason is None}
    assert len(set(grid_of.values())) == 2
    other = next(g for g in info.grids if g.grid_key == grid_of["task-rest_run-5"])
    assert other.shape == (17, 18, 15) and other.contours is not None
    assert got["task-me_run-1"] == "multi-echo"
    assert got["task-native_run-1"] == "native space only"
    assert got["task-std_run-1"] == "standard space only"


def test_the_decoded_streams_against_a_recomputation(
    dataset: SyntheticFuncDataset, packed: dict[str, Path]
) -> None:
    """One run that is not the anchor per session, and the run on another grid: the raw
    stream within one gray level of the recomputation (99.5 % equal) and the preprocessed
    stream equal to the stored series quantised."""
    cases = [
        ("sub-01", "1", "rest", "1"),  # rotated, the qform is the true affine (H06)
        ("sub-01", "1", "rest", "3"),  # 12 mm, no reference image: the header, unchecked
        ("sub-01", "2", "rest", "3"),
        ("sub-02", None, "rest", "5"),  # on another grid
    ]
    for participant, ses, task, number in cases:
        run = dataset.run(participant, ses, task, number)
        session = dataset.session(participant, ses)
        anchor = next(r for r in dataset.runs if r.stem == session.anchor)
        infos = sessions_of(packed[participant])
        info = next(i for i in infos if i.label.endswith(f"ses-{ses}" if ses else participant))
        session_dir, pack = pack_of(packed[participant], info, run.name)
        pre = nib.load(run.files["pre_bold"])
        for t in (0, 7, 11):
            mine = independent_raw_volume(run, anchor, t, pack.window)
            got = packread.decode_volume(session_dir, pack, "raw", t)
            difference = np.abs(mine.astype(int) - got.astype(int))
            assert difference.max() <= 1, (run.name, t)
            assert (difference == 0).mean() >= 0.995, (run.name, t)
            stored = np.asarray(pre.dataobj[..., t], np.float32)
            want = quantise(stored, *pack.window)
            np.testing.assert_array_equal(packread.decode_volume(session_dir, pack, "pre", t), want)
        expected = {"rest_1": "qform", "rest_3": "header, unchecked"}.get(f"{task}_{number}")
        if ses == "1" and expected:
            assert pack.raw_affine == expected


def test_the_two_streams_of_the_anchor_agree_and_others_do_not(
    dataset: SyntheticFuncDataset, packed: dict[str, Path]
) -> None:
    """The anchor's raw stream is its preprocessed series where the spline is exact (the
    fixture's pipeline placed it perfectly); a run that moved is not, which is the signal
    (D36) and is never removed."""
    infos = sessions_of(packed["sub-01"])
    session_dir, anchor = pack_of(packed["sub-01"], infos[0], "ses-1_task-rest_run-2")
    _, moved = pack_of(packed["sub-01"], infos[0], "ses-1_task-rest_run-3")
    raw = packread.decode_volume(session_dir, anchor, "raw", 3).astype(int)
    pre = packread.decode_volume(session_dir, anchor, "pre", 3).astype(int)
    mask = np.asarray(nib.load(dataset.run("sub-01", "1", "rest", "2").files["pre_mask"]).dataobj)
    inside = mask > 0
    assert np.median(np.abs(raw - pre)[inside]) <= 1
    raw_m = packread.decode_volume(session_dir, moved, "raw", 3).astype(int)
    pre_m = packread.decode_volume(session_dir, moved, "pre", 3).astype(int)
    # 12 mm in a field that changes by about 2 per mm, under a window of about 1,500
    assert np.median(np.abs(raw_m - pre_m)[inside]) >= 3


def test_only_json_and_bin_files_are_written_and_no_volume_is_saved(
    dataset: SyntheticFuncDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved: list[object] = []

    def refuse(*args: object, **kwargs: object) -> None:
        saved.append(args)
        raise AssertionError("an image volume was written")

    monkeypatch.setattr(nib, "save", refuse)
    monkeypatch.setattr(nib.filebasedimages.FileBasedImage, "to_filename", refuse)
    monkeypatch.setattr(nib.filebasedimages.FileBasedImage, "to_file_map", refuse)
    out = tmp_path / "out"
    api.epi_subject(dataset.raw_root, "02", out)
    assert saved == []
    suffixes = {p.suffix for p in out.rglob("*") if p.is_file()}
    assert suffixes == {".json", ".bin"}
    assert not [p for p in out.rglob("*") if p.name.endswith((".nii", ".nii.gz", ".mgz"))]


def test_nothing_is_written_into_the_sources(tmp_path: Path) -> None:
    ds = write_copy(tmp_path, annex="link")
    fs_store = tmp_path / "store"
    fs_store.mkdir()
    shutil.move(str(ds.fs_subjects_dir / "sub-01"), str(fs_store / "sub-01"))
    (ds.fs_subjects_dir / "sub-01").symlink_to(fs_store / "sub-01")
    before = snapshot_links(tmp_path)
    out = tmp_path / "out"
    api.epi_subject(ds.raw_root, "sub-01", out)
    after = snapshot_links(tmp_path)
    assert {k: v for k, v in after.items() if not k.startswith("out")} == before


# --- refusals ----------------------------------------------------------------------------


def test_an_output_in_a_folder_that_is_read_is_refused(tmp_path: Path) -> None:
    ds = write_copy(tmp_path, raw_inside=False)
    store = tmp_path / "store"
    store.mkdir()
    shutil.move(str(ds.fs_subjects_dir / "sub-02"), str(store / "sub-02"))
    (ds.fs_subjects_dir / "sub-02").symlink_to(store / "sub-02")
    before = snapshot_links(tmp_path)
    for out in (
        ds.raw_root / "x",
        ds.deriv_root,
        ds.deriv_root / "sub-02" / "packs",
        ds.fs_subjects_dir / "packs",
        store / "sub-02" / "packs",
        ds.fs_subjects_dir / "sub-02" / "packs",
    ):
        with pytest.raises(ConfigError) as err:
            api.epi_subject(ds.deriv_root, "sub-02", out, raw=ds.raw_root)
        assert err.value.reason == "invalid" and "only read" in err.value.detail, out
    assert snapshot_links(tmp_path) == before
    # beside the folders is fine
    api.epi_subject(ds.deriv_root, "sub-02", tmp_path / "out", raw=ds.raw_root)


def test_the_other_refusals(dataset: SyntheticFuncDataset, tmp_path: Path) -> None:
    with pytest.raises(SourceError) as err:
        api.epi_subject(dataset.raw_root, "sub-09", tmp_path / "a")
    assert err.value.reason == "not found"
    with pytest.raises(ConfigError, match="no session 3; sessions: 1, 2"):
        api.epi_subject(dataset.raw_root, "sub-01", tmp_path / "b", sessions=["3"])
    with pytest.raises(ConfigError, match="no BIDS participant"):
        api.epi_subject(dataset.raw_root, "sub-0_1", tmp_path / "c")
    with pytest.raises(ConfigError, match="no run of sub-02 matches"):
        api.epi_subject(dataset.raw_root, "sub-02", tmp_path / "d", runs=["*task-none*"])
    with pytest.raises(ConfigError, match="several sessions"):
        api.epi_session(dataset.raw_root, "sub-01", tmp_path / "e")
    # an output that holds the packs of another participant
    api.epi_subject(dataset.raw_root, "sub-02", tmp_path / "f", runs=["*nback_run-1"])
    with pytest.raises(ConfigError, match="holds the packs of sub-02"):
        api.epi_subject(dataset.raw_root, "sub-01", tmp_path / "f")
    assert not list((tmp_path).glob("[abcde]"))


# --- what is not there ---------------------------------------------------------------------


def test_no_raw_dataset_lists_every_run_and_names_raw(tmp_path: Path) -> None:
    ds = write_copy(tmp_path, raw_inside=False)
    lines: list[str] = []
    stream = api.epi_subject(ds.deriv_root, "sub-01", tmp_path / "out", progress=lines.append)
    infos = sessions_of(tmp_path / "out")
    assert len(stream.sessions) == 2 and len(infos) == 2
    for info in infos:
        assert info.registration is None and info.grids == ()
        assert {r.reason for r in info.runs} == {"no raw series"}
        assert all("--raw DIR" in (r.detail or "") for r in info.runs)
    assert "no raw dataset found (give it with --raw DIR)" in lines[0]
    assert "0 can be packed" in lines[0]


def test_a_session_without_a_freesurfer_subject_is_packed_without_contours(
    tmp_path: Path,
) -> None:
    ds = write_copy(tmp_path)
    shutil.rmtree(ds.fs_subjects_dir / "sub-02")
    lines: list[str] = []
    api.epi_subject(ds.raw_root, "sub-02", tmp_path / "out", progress=lines.append)
    (info,) = sessions_of(tmp_path / "out")
    assert info.contours_from is None and all(g.contours is None for g in info.grids)
    assert sum(r.reason is None for r in info.runs) == 4
    assert any("without contours: no FreeSurfer subject of this session" in x for x in lines)


def test_a_sessionwise_freesurfer_subject_is_taken_first(tmp_path: Path) -> None:
    ds = write_copy(tmp_path)
    shutil.copytree(ds.fs_subjects_dir / "sub-01", ds.fs_subjects_dir / "sub-01_ses-2")
    api.epi_subject(ds.raw_root, "sub-01", tmp_path / "out")
    assert [i.contours_from for i in sessions_of(tmp_path / "out")] == ["sub-01", "sub-01_ses-2"]


@pytest.mark.parametrize("kind", ["link", "pointer"])
def test_files_without_content_are_listed_and_named(tmp_path: Path, kind: str) -> None:
    ds = write_copy(tmp_path, annex=kind)
    before = snapshot_links(ds.root)
    api.epi_subject(ds.raw_root, "sub-01", tmp_path / "out")
    assert snapshot_links(ds.root) == before
    infos = sessions_of(tmp_path / "out")
    held = next(r for r in infos[1].runs if r.name == "ses-2_task-rest_run-2")
    assert held.reason == "not fetched" and held.detail is not None
    assert "git-annex placeholder" in held.detail
    wanted = {
        "link": ("_bold.nii.gz", "desc-preproc_bold.nii.gz", "confounds_timeseries.tsv"),
        "pointer": ("desc-summary_bold.html", "confounds_timeseries.tsv"),
    }[kind]
    assert all(name in held.detail for name in wanted)
    assert not os.path.isabs(held.detail.split(": ", 1)[1].split(",")[0])
    assert sum(r.reason is None for r in infos[1].runs) == 2


# --- ids, determinism and a second call ------------------------------------------------------


def test_a_second_call_keeps_the_ids_and_drops_a_session_left_out(
    dataset: SyntheticFuncDataset, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    first = api.epi_subject(dataset.raw_root, "sub-01", out)
    runs_before = {r.name: r.run for i in sessions_of(out) for r in i.runs}
    again = api.epi_subject(dataset.raw_root, "sub-01", out)
    assert again.sessions == first.sessions
    assert {r.name: r.run for i in sessions_of(out) for r in i.runs} == runs_before
    one = api.epi_subject(dataset.raw_root, "sub-01", out, sessions=["ses-1"])
    assert one.sessions == first.sessions[:1]
    assert not (out / first.sessions[1]).exists()
    # without a stream.json a call draws new ids
    other = api.epi_subject(dataset.raw_root, "sub-01", tmp_path / "fresh")
    assert set(other.sessions).isdisjoint(first.sessions)


def _digests(out: Path) -> dict[str, str]:
    found = {}
    for info in sessions_of(out):
        for run in info.runs:
            pack = read_info(out / info.session / run.run / PACK_JSON, PackInfo)
            if pack is not None:
                found[f"{info.label} {run.name}"] = pack.content_digest
    return found


def test_the_content_digest_is_the_same_in_two_processes(
    dataset: SyntheticFuncDataset, packed: dict[str, Path], tmp_path: Path
) -> None:
    script = (
        "import sys\nfrom pathlib import Path\nfrom surfcensus.func import api\n"
        "api.epi_subject(Path(sys.argv[1]), 'sub-01', Path(sys.argv[2]),"
        " key=sys.argv[3].encode())\n"
    )
    out = tmp_path / "other"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"} | {"PYTHONHASHSEED": "7"}
    subprocess.run(
        [sys.executable, "-c", script, str(dataset.raw_root), str(out), KEY.decode()],
        check=True,
        env=env,
    )
    assert stream_of(out).sessions == stream_of(packed["sub-01"]).sessions
    mine = _digests(packed["sub-01"])
    assert len(mine) == 7 and _digests(out) == mine


def test_the_runs_named_and_one_session(dataset: SyntheticFuncDataset, tmp_path: Path) -> None:
    info = api.epi_session(dataset.raw_root, "sub-01", tmp_path / "one", session="2", key=KEY)
    assert info.label == "sub-01 ses-2" and stream_of(tmp_path / "one").sessions == (info.session,)
    stream = api.epi_subject(
        dataset.raw_root, "sub-02", tmp_path / "rest", runs=["*task-rest_run-5"], key=KEY
    )
    (only,) = sessions_of(tmp_path / "rest")
    assert len(stream.sessions) == 1
    assert [r.name for r in only.runs if r.reason is None] == ["task-rest_run-5"]
    # the runs left out are still runs of the session, listed and not packed
    left = [r for r in only.runs if r.reason is not None]
    assert len(left) == 11 and all(r.reason == "not selected" for r in left)
    assert all(r.detail and "--epi-runs" in r.detail for r in left)


def test_what_it_says(dataset: SyntheticFuncDataset, tmp_path: Path) -> None:
    lines: list[str] = []
    api.epi_subject(dataset.raw_root, "sub-02", tmp_path / "out", progress=lines.append)
    assert lines[0].startswith(
        "Functional: 12 runs in 1 session of sub-02 in derivatives/fmriprep (fMRIPrep 21.0.2),"
        f" raw series in {dataset.raw_root} (BIDS); 4 can be packed (T1w space), 8 cannot: "
    )
    # the anchor is the first run by name order, task-me run-1, which is multi-echo
    assert lines[1] == (
        "sub-02: the anchor task-me_run-1 is not packed (multi-echo); the raw stream of every"
        " packed run is drawn through its stored registration."
    )
    assert lines[2].startswith("EPI packs: 4 runs to pack, about ") and "an estimate" in lines[2]
    runs = [x for x in lines if x.startswith("sub-02: task-")]
    assert len(runs) == 4 and all(" 12 volumes, " in x for x in runs)
    assert (
        lines[-1].startswith("sub-02: 4 of 12 runs packed, ")
        and "contours from sub-02" in lines[-1]
    )


def test_a_participant_without_a_packable_run_gives_a_stream(tmp_path: Path) -> None:
    ds = write_copy(tmp_path)
    stream = api.epi_subject(ds.raw_root, "sub-02", tmp_path / "out", runs=["*task-std*"])
    (info,) = sessions_of(tmp_path / "out")
    reasons = {r.name: r.reason for r in info.runs}
    assert len(stream.sessions) == 1 and reasons["task-std_run-1"] == "standard space only"
    assert set(reasons.values()) == {"standard space only", "not selected"}


def test_the_config_is_used(dataset: SyntheticFuncDataset, tmp_path: Path) -> None:
    config = EpiConfig(frame_limit_mm=20.0)
    api.epi_subject(dataset.raw_root, "sub-01", tmp_path / "out", config=config, key=KEY)
    first = sessions_of(tmp_path / "out")[0]
    assert first.registration is not None and first.registration.limit_mm == 20.0
    assert not any(r.frame and r.frame.flagged for r in first.runs)


def _walk(out: Path) -> Iterator[Path]:
    yield from out.rglob("*")


def test_stream_json_is_written_last(
    dataset: SyntheticFuncDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[bool] = []
    real = api.pack_.pack_session

    def watch(out: Path, *args: object, **kwargs: object) -> object:
        seen.append((Path(out) / STREAM_JSON).exists())
        return real(out, *args, **kwargs)  # pyrefly: ignore[bad-argument-type]

    monkeypatch.setattr(api.pack_, "pack_session", watch)
    out = tmp_path / "out"
    api.epi_subject(dataset.raw_root, "sub-01", out)
    first = (out / STREAM_JSON).read_bytes()
    # a second call keeps the earlier stream.json, so that a page served from ``out`` keeps a
    # whole stream, until the new one is renamed over it at the end
    api.epi_subject(dataset.raw_root, "sub-01", out)
    assert seen == [False, False, True, True]
    assert json.loads((out / STREAM_JSON).read_text())["label"] == "sub-01"
    assert (out / STREAM_JSON).read_bytes() == first  # the same ids, the same stream
    assert len(list(_walk(out))) > 0


def test_the_order_of_sessions_compares_numbers_as_numbers() -> None:
    labels = ["10", "2", "pre", None, "1b", "1a", "01"]
    assert sorted(labels, key=api.session_key) == [None, "01", "1a", "1b", "2", "10", "pre"]


# --- one anchor per session, whatever is packed (decision D36; data review F1) -----------------

ANCHOR = "ses-1_task-rest_run-2"
"""The anchor of sub-01 ses-1 by AcquisitionTime (9:5:3.5), not the first in name order."""


@pytest.fixture(scope="module")
def ses1(dataset: SyntheticFuncDataset, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """sub-01 ses-1 packed whole, with the fixed key."""
    out = tmp_path_factory.mktemp("ses1-full")
    api.epi_subject(dataset.raw_root, "sub-01", out, sessions=["1"], key=KEY)
    return out


def _others(info: SessionInfo) -> dict[str, object]:
    """The frame of every packed run but the anchor, by name."""
    return {r.name: r.frame for r in info.runs if r.reason is None and not r.anchor}


def test_a_filter_does_not_move_the_anchor(
    dataset: SyntheticFuncDataset, ses1: Path, tmp_path: Path
) -> None:
    out = tmp_path / "motor"
    api.epi_subject(dataset.raw_root, "sub-01", out, sessions=["1"], runs=["*task-motor*"], key=KEY)
    (full,) = sessions_of(ses1)
    (info,) = sessions_of(out)
    anchor = next(r for r in info.runs if r.anchor)
    # the anchor is the session's, not the one run that was selected
    assert anchor.name == ANCHOR and anchor.reason == "not selected"
    assert info.registration is not None and info.registration.anchor == anchor.run
    assert not info.registration.anchor_packed and info.registration.rule == "acquisition_time"
    # every run of the session is listed, in the same order of the session as without a filter
    order = {r.name: r.session_order for r in full.runs}
    assert {r.name: r.session_order for r in info.runs} == order
    assert sorted(order.values()) == list(range(4))
    # the motor run is drawn through the anchor's transform: its pack is that of the full
    # session, value for value (the content digest covers the defining JSON, whose inputs
    # name the anchor's transform)
    _, mine = pack_of(out, info, "ses-1_task-motor_run-1")
    _, whole = pack_of(ses1, full, "ses-1_task-motor_run-1")
    assert mine.anchor_run == anchor.run and mine.content_digest == whole.content_digest
    assert any(i.role == "anchor_xfm_to_anat" for i in mine.inputs)
    assert _others(info) == {"ses-1_task-motor_run-1": _others(full)["ses-1_task-motor_run-1"]}
    motor = dataset.run("sub-01", "1", "motor", "1")
    for t in (0, mine.n_volumes - 1):
        expected = independent_raw_volume(
            motor, dataset.run("sub-01", "1", "rest", "2"), t, mine.window
        )
        decoded = packread.decode_volume(out / info.session, mine, "raw", t)
        assert np.array_equal(decoded, expected)
    # the first index follows the packed runs
    assert [r.first for r in info.runs if r.reason is None] == [0]


@pytest.mark.parametrize("case", ["no confounds", "no mask", "fewer volumes", "dangling raw"])
def test_an_anchor_that_cannot_be_packed_is_the_anchor_all_the_same(
    ses1: Path, tmp_path: Path, case: str
) -> None:
    ds = write_copy(tmp_path)
    run = ds.run("sub-01", "1", "rest", "2")
    expected = {
        "no confounds": "no confounds table",
        "no mask": "no brain mask",
        "fewer volumes": "volume counts differ",
        "dangling raw": "no raw series",
    }[case]
    if case == "no confounds":
        run.files["confounds"].unlink()
    elif case == "no mask":
        run.files["pre_mask"].unlink()
    elif case == "fewer volumes":
        img = nib.load(run.files["raw_bold"])
        data = np.asarray(img.dataobj)[..., :-2]
        run.files["raw_bold"].unlink()
        nib.save(nib.Nifti1Image(data, None, img.header), run.files["raw_bold"])
    else:
        run.files["raw_bold"].unlink()
        run.files["raw_bold"].symlink_to(tmp_path / "absent.nii.gz")
    out = tmp_path / "out"
    api.epi_subject(ds.raw_root, "sub-01", out, sessions=["1"], key=KEY)
    (full,) = sessions_of(ses1)
    (info,) = sessions_of(out)
    anchor = next(r for r in info.runs if r.anchor)
    assert anchor.name == ANCHOR and anchor.reason == expected
    assert info.registration is not None and not info.registration.anchor_packed
    if case == "dangling raw":
        assert anchor.detail is not None and "is a link whose target is missing" in anchor.detail
    # the frame numbers of the other runs are those of the full session, and so are their packs
    assert _others(info) == _others(full)
    assert {r.name: r.session_order for r in info.runs} == {
        r.name: r.session_order for r in full.runs
    }
    for name in _others(full):
        assert (
            pack_of(out, info, name)[1].content_digest
            == pack_of(ses1, full, name)[1].content_digest
        )


@pytest.mark.parametrize("case", ["no transform", "unreadable transform"])
def test_without_a_readable_transform_the_next_run_is_the_anchor(tmp_path: Path, case: str) -> None:
    ds = write_copy(tmp_path)
    xfm = ds.run("sub-01", "1", "rest", "2").files["xfm_to_anat"]
    if case == "no transform":
        xfm.unlink()
    else:
        xfm.write_text("#Insight Transform File V1.0\nthis is no transform\n")
    out = tmp_path / "out"
    lines: list[str] = []
    api.epi_subject(ds.raw_root, "sub-01", out, sessions=["1"], key=KEY, progress=lines.append)
    (info,) = sessions_of(out)
    first = next(r for r in info.runs if r.name == ANCHOR)
    anchor = next(r for r in info.runs if r.anchor)
    # the next run by AcquisitionTime (9:12:40.0)
    assert anchor.name == "ses-1_task-rest_run-1" and anchor.reason is None
    assert not first.anchor
    if case == "no transform":
        assert first.reason == "no transform to the anatomy"
    else:
        assert first.reason == "pack failure" and first.detail is not None
        assert first.detail.startswith("its stored transform to the anatomy cannot be read")
    reg = info.registration
    assert (
        reg is not None
        and reg.anchor_packed
        and reg.note
        == (
            f"the first run by the rule ({ANCHOR}) has no readable stored transform to the "
            "anatomy, so the anchor is the next run, ses-1_task-rest_run-1"
        )
    )
    assert f"sub-01 ses-1: {reg.note}." in lines
    assert first.session_order == 0 and anchor.session_order == 1


# --- a second call into the same --out (page review 2, data review F5) ------------------------


def _mtimes(out: Path) -> dict[str, int]:
    return {
        p.relative_to(out).as_posix(): p.stat().st_mtime_ns
        for p in out.rglob("*")
        if p.is_file() and p.name not in (SESSION_JSON, STREAM_JSON)
    }


def test_a_second_call_keeps_the_packs_that_are_current(
    dataset: SyntheticFuncDataset, tmp_path: Path
) -> None:
    ds = write_copy(tmp_path)
    out = tmp_path / "out"
    api.epi_subject(ds.raw_root, "sub-01", out, key=KEY)
    before = _mtimes(out)
    digests = _digests(out)
    lines: list[str] = []
    api.epi_subject(ds.raw_root, "sub-01", out, key=KEY, progress=lines.append)
    # nothing is packed again: every pack file and every contour file is as it was
    assert _mtimes(out) == before and _digests(out) == digests
    assert any(
        x.startswith("EPI packs: 0 runs to pack, 7 kept from an earlier call") for x in lines
    )
    assert sum("kept (its pack is current)" in x for x in lines) == 7
    assert "sub-01 ses-1: 4 of 4 runs packed (4 kept, 0 packed now)," in "\n".join(lines)
    # a changed input of one run packs that run again, and only it
    sessions = {i.label: i for i in sessions_of(out)}
    motor = ds.run("sub-01", "1", "motor", "1")
    table = motor.files["confounds"]
    table.write_text(table.read_text() + "\n")  # another SHA-256, the same rows
    lines.clear()
    api.epi_subject(ds.raw_root, "sub-01", out, key=KEY, progress=lines.append)
    again = [x for x in lines if "volumes, packed," in x]
    assert len(again) == 1 and again[0].startswith("sub-01 ses-1: ses-1_task-motor_run-1: ")
    # a changed anchor transform packs every run of its session again (an input of each)
    anchor = ds.run("sub-01", "1", "rest", "2").files["xfm_to_anat"]
    anchor.write_text(anchor.read_text() + "\n")
    lines.clear()
    api.epi_subject(ds.raw_root, "sub-01", out, key=KEY, progress=lines.append)
    ses1 = [x for x in lines if x.startswith("sub-01 ses-1: ses-1_")]
    assert len(ses1) == 4 and all("volumes, packed," in x for x in ses1)
    assert all("kept" in x for x in lines if x.startswith("sub-01 ses-2: ses-2_task-rest_run-3"))
    # a pack that lost a file is packed again; force packs everything again
    (info,) = [sessions[k] for k in sessions if k == "sub-01 ses-2"]
    run = next(r for r in info.runs if r.reason is None)
    pack = read_info(out / info.session / run.run / PACK_JSON, PackInfo)
    assert pack is not None
    (out / info.session / pack.streams["raw"].chunks[0].file).unlink()
    lines.clear()
    api.epi_subject(ds.raw_root, "sub-01", out, key=KEY, progress=lines.append)
    assert sum("volumes, packed," in x for x in lines) == 1
    lines.clear()
    api.epi_subject(ds.raw_root, "sub-01", out, key=KEY, progress=lines.append, force=True)
    assert sum("volumes, packed," in x for x in lines) == 7


def test_runs_that_leave_a_session_lose_their_pack_after_session_json(
    dataset: SyntheticFuncDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    api.epi_subject(dataset.raw_root, "sub-01", out, sessions=["1"], key=KEY)
    (info,) = sessions_of(out)
    session_dir = out / info.session
    packed = {r.name: r.run for r in info.runs if r.reason is None}
    assert all((session_dir / run).is_dir() for run in packed.values())
    # the earlier session.json stays in place while the second call packs: a reader of the
    # directory sees a whole session at every moment
    seen: list[bool] = []
    real = api.pack_.pack_run

    def watch(session_dir_: Path, *args: object, **kwargs: object) -> object:
        old = read_info(Path(session_dir_) / SESSION_JSON, SessionInfo)
        listed = [] if old is None else [r.run for r in old.runs if r.reason is None]
        seen.append(bool(listed) and all((Path(session_dir_) / r).is_dir() for r in listed))
        return real(session_dir_, *args, **kwargs)  # pyrefly: ignore[bad-argument-type]

    monkeypatch.setattr(api.pack_, "pack_run", watch)
    api.epi_subject(dataset.raw_root, "sub-01", out, sessions=["1"], runs=["*task-motor*"], key=KEY)
    assert seen == []  # the motor run's pack is current: nothing is packed
    (after,) = sessions_of(out)
    # the runs not selected are listed and lose their packs; the selected one keeps its own
    kept = [r for r in after.runs if r.reason is None]
    assert [r.name for r in kept] == ["ses-1_task-motor_run-1"]
    for name, run in packed.items():
        assert (session_dir / run).is_dir() == (name == "ses-1_task-motor_run-1")
    api.epi_subject(dataset.raw_root, "sub-01", out, sessions=["1"], key=KEY, force=True)
    assert seen and all(seen)


# --- links, missing targets and uncompressed series (data review F2, F3, F4) -------------------


def test_an_out_where_a_linked_input_lies_is_refused(tmp_path: Path) -> None:
    """Per-file links, as a DataLad or git-annex clone has them: the directory that holds the
    target of a linked input is read, so nothing is written there."""
    ds = write_copy(tmp_path)
    store = tmp_path / "store" / "objects"
    store.mkdir(parents=True)
    for role in ("raw_bold", "pre_mask"):
        source = ds.run("sub-01", "2", "rest", "3").files[role]
        target = store / role / source.name
        target.parent.mkdir()
        shutil.move(source, target)
        source.symlink_to(target)
    before = snapshot_links(tmp_path / "store")
    for out in (store / "raw_bold" / "o", store / "pre_mask"):
        with pytest.raises(ConfigError) as caught:
            api.epi_subject(ds.raw_root, "sub-01", out)
        assert "leads into" in str(caught.value) and "which is only read" in str(caught.value)
        assert not out.exists() or out == store / "pre_mask"
    assert snapshot_links(tmp_path / "store") == before
    # beside the store is not inside it
    api.epi_subject(ds.raw_root, "sub-01", tmp_path / "store" / "packs", sessions=["2"])
    assert (tmp_path / "store" / "packs" / STREAM_JSON).is_file()


@pytest.mark.parametrize("which", ["raw_bold", "pre_bold"])
def test_a_series_that_is_a_link_without_target_is_said_to_be_one(
    tmp_path: Path, which: str
) -> None:
    ds = write_copy(tmp_path)
    run = ds.run("sub-01", "2", "rest", "3")
    path = run.files[which]
    path.unlink()
    path.symlink_to(tmp_path / "nowhere" / path.name)
    api.epi_subject(ds.raw_root, "sub-01", tmp_path / "out", sessions=["2"])
    (info,) = sessions_of(tmp_path / "out")
    item = next(r for r in info.runs if r.name == run.name)
    what = "raw" if which == "raw_bold" else "preprocessed"
    assert item.reason == f"no {what} series"
    assert item.detail == f"the {what} series {path.name} is a link whose target is missing"


def test_a_derivative_without_a_preprocessed_series_says_only_that(tmp_path: Path) -> None:
    ds = write_copy(tmp_path)
    api.epi_subject(ds.raw_root, "sub-02", tmp_path / "out", runs=["*task-min*"])
    (info,) = sessions_of(tmp_path / "out")
    item = next(r for r in info.runs if r.name == "task-min_run-1")
    assert item.reason == "no preprocessed series" and item.detail is None


def test_an_uncompressed_derivative_is_read_like_a_compressed_one(
    ses1: Path, tmp_path: Path
) -> None:
    ds = write_copy(tmp_path)
    run = ds.run("sub-01", "1", "motor", "1")
    for role in ("pre_bold", "pre_mask", "reference"):
        path = run.files[role]
        img = nib.load(path)
        nib.save(img, path.with_name(path.name[: -len(".gz")]))
        path.unlink()
    out = tmp_path / "out"
    api.epi_subject(ds.raw_root, "sub-01", out, sessions=["1"], key=KEY)
    (info,) = sessions_of(out)
    (full,) = sessions_of(ses1)
    name = "ses-1_task-motor_run-1"
    item = next(r for r in info.runs if r.name == name)
    assert item.reason is None
    mine, whole = pack_of(out, info, name)[1], pack_of(ses1, full, name)[1]
    assert mine.raw_affine == whole.raw_affine and mine.window == whole.window
    for stream in ("raw", "pre"):
        for t in (0, mine.n_volumes - 1):
            assert np.array_equal(
                packread.decode_volume(out / info.session, mine, stream, t),
                packread.decode_volume(ses1 / full.session, whole, stream, t),
            )


def test_what_it_says_about_the_registration(dataset: SyntheticFuncDataset, tmp_path: Path) -> None:
    lines: list[str] = []
    api.epi_subject(dataset.raw_root, "sub-01", tmp_path / "out", progress=lines.append)
    flagged = [x for x in lines if "flagged (raw data)" in x]
    assert len(flagged) == 3, lines
    assert flagged[0].startswith(
        "sub-01 ses-1: flagged (raw data): ses-1_task-rest_run-3, drawn through the "
        f"registration of the anchor {ANCHOR}, lies up to 12."
    )
    assert flagged[0].endswith("beyond the limit of 10 mm.")
    assert (
        "sub-01 ses-2: the anchor ses-2_task-rest_run-1 is the outlier (raw data): every other"
        " run is flagged against it and they agree with each other, so the anchor, or its"
        " stored registration, is the odd one."
    ) in lines
    assert f"raw series in {dataset.raw_root} (BIDS)" in lines[0]


def test_an_interrupt_between_the_two_renames_puts_the_earlier_pack_back(
    dataset: SyntheticFuncDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    api.epi_subject(dataset.raw_root, "sub-01", out, sessions=["2"], key=KEY)
    (info,) = sessions_of(out)
    run = next(r for r in info.runs if r.reason is None)
    session_dir = out / info.session
    before = (session_dir / run.run / PACK_JSON).read_bytes()
    real = os.replace

    def interrupted(source: object, target: object) -> None:
        if Path(str(source)).parent.name.endswith(".tmp") and Path(str(target)).name == run.run:
            raise KeyboardInterrupt  # after the earlier pack was moved aside
        real(source, target)  # pyrefly: ignore[bad-argument-type]

    monkeypatch.setattr(api.pack_.os, "replace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        api.epi_subject(dataset.raw_root, "sub-01", out, sessions=["2"], key=KEY, force=True)
    monkeypatch.undo()
    assert (session_dir / run.run / PACK_JSON).read_bytes() == before
    assert packread.pack_on_disk(session_dir, run.run)
