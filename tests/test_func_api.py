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
    assert [r.name for r in only.runs] == ["task-rest_run-5"] and len(stream.sessions) == 1


def test_what_it_says(dataset: SyntheticFuncDataset, tmp_path: Path) -> None:
    lines: list[str] = []
    api.epi_subject(dataset.raw_root, "sub-02", tmp_path / "out", progress=lines.append)
    assert lines[0].startswith(
        "Functional: 12 runs in 1 session of sub-02 in derivatives/fmriprep (fMRIPrep 21.0.2),"
        " raw series in . (BIDS); 4 can be packed (T1w space), 8 cannot: "
    )
    assert lines[1].startswith("EPI packs: 4 runs to pack, about ") and "an estimate" in lines[1]
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
    assert len(stream.sessions) == 1 and [r.reason for r in info.runs] == ["standard space only"]


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
    api.epi_subject(dataset.raw_root, "sub-01", out)  # an earlier stream.json is removed first
    assert seen == [False, False, False, False]
    assert json.loads((out / STREAM_JSON).read_text())["label"] == "sub-01"
    assert len(list(_walk(out))) > 0


def test_the_order_of_sessions_compares_numbers_as_numbers() -> None:
    labels = ["10", "2", "pre", None, "1b", "1a", "01"]
    assert sorted(labels, key=api.session_key) == [None, "01", "1a", "1b", "2", "10", "pre"]
