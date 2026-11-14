"""``func.api.epi_subject`` end to end on the worked example (network): ``ds000102`` sub-01,
the two runs of ``task-flanker``, laid out as a BIDS dataset root with the fMRIPrep derivative
below ``derivatives/`` and its FreeSurfer subject in ``sourcedata/freesurfer/``.

The files are those that the network test of ``func/pack.py`` fetches into the pytest cache
(``ds000102-sub-01-func``), the FreeSurfer subject of ``conftest.real_subject``, and four small
files fetched here into ``ds000102-sub-01-func-bids``: the derivative's
``dataset_description.json``, the two summary reports and the raw dataset's
``task-flanker_bold.json``; anonymous, read-only GETs, at most four at a time. The BIDS tree is
made of links into the cache, under ``tmp_path``.

What is checked: the folders found from the dataset root; the anchor by name order (no raw
sidecar has ``AcquisitionTime``) and the frame numbers of stage 1 (0.829 and 1.087 mm, within
0.002); the contours cut from the FreeSurfer subject; and every decoded volume of both streams
of both runs equal to those of a pack made directly by ``pack.pack_run`` with the inputs built
as the network test of ``func/pack.py`` builds them (that test shows them equal to stage 1's
own pack), with numpy's voxel map in both.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from synth_annex import snapshot_links  # pyrefly: ignore[missing-import]
from test_func_pack import (  # pyrefly: ignore[missing-import]
    DERIV_URL,
    FSNATIVE,
    RAW_URL,
    STEM,
    _fetch_worked_example,
)

from surfcensus.config import EpiConfig
from surfcensus.func import PACK_JSON, SESSION_JSON, STREAM_JSON, api, geometry, itk, session
from surfcensus.func.connectors import fmriprep
from surfcensus.func.model import PackInfo, SessionInfo, StreamInfo, read_info
from surfcensus.func.pack import pack_run
from surfcensus.func.packread import stream_bytes
from surfcensus.records import DerivRun, RunPackInput, to_matrix4
from surfcensus.sources.base import FuncRole

pytestmark = pytest.mark.network

EXTRA = {  # local name -> (url, size from the bucket listing of 2026-11-14)
    "dataset_description.json": (DERIV_URL + "dataset_description.json", 504),
    "run-1_desc-summary_bold.html": (
        DERIV_URL + "sub-01/figures/" + STEM.format(n=1) + "_desc-summary_bold.html",
        3_517,
    ),
    "run-2_desc-summary_bold.html": (
        DERIV_URL + "sub-01/figures/" + STEM.format(n=2) + "_desc-summary_bold.html",
        3_431,
    ),
    "task-flanker_bold.json": (RAW_URL + "task-flanker_bold.json", 878),
}
EXPECTED_FRAME = {"mean_mm": 0.829, "max_mm": 1.087}
TOLERANCE = 0.002


def _fetch_extra(root: Path) -> Path:
    import httpx

    def get(item: tuple[str, tuple[str, int]]) -> None:
        name, (url, size) = item
        path = root / name
        if path.is_file() and path.stat().st_size == size:
            return
        with httpx.Client(timeout=120.0) as client:
            content = client.get(url).raise_for_status().content
        assert len(content) == size, f"{name}: {len(content)} bytes, expected {size}"
        part = path.with_name(path.name + ".part")
        part.write_bytes(content)
        part.replace(path)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(get, EXTRA.items()))
    return root


def _link(path: Path, target: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(target)


def bids_root(root: Path, data: Path, extra: Path, subject: Path) -> Path:
    """The worked example as a BIDS dataset root of links into the cache."""
    ds = root / "ds000102"
    deriv = ds / "derivatives" / "fmriprep"
    _link(ds / "task-flanker_bold.json", extra / "task-flanker_bold.json")
    (ds / "dataset_description.json").write_text(
        '{"Name": "worked example", "BIDSVersion": "1.0.2"}\n'
    )
    _link(deriv / "dataset_description.json", extra / "dataset_description.json")
    for n in (1, 2):
        stem = STEM.format(n=n)
        _link(ds / "sub-01" / "func" / f"{stem}_bold.nii.gz", data / f"raw/run-{n}_bold.nii.gz")
        func = deriv / "sub-01" / "func"
        for suffix in (
            "space-T1w_desc-preproc_bold.nii.gz",
            "space-T1w_desc-preproc_bold.json",
            "space-T1w_desc-brain_mask.nii.gz",
            "boldref.nii.gz",
            "desc-confounds_timeseries.tsv",
            "from-scanner_to-T1w_mode-image_xfm.txt",
        ):
            _link(func / f"{stem}_{suffix}", data / f"deriv/run-{n}_{suffix}")
        _link(
            deriv / "sub-01" / "figures" / f"{stem}_desc-summary_bold.html",
            extra / f"run-{n}_desc-summary_bold.html",
        )
    _link(
        deriv / "sub-01" / "anat" / "sub-01_from-T1w_to-fsnative_mode-image_xfm.txt",
        data / FSNATIVE[0],
    )
    _link(deriv / "sourcedata" / "freesurfer" / "sub-01", subject)
    return ds


def test_the_worked_example_from_the_dataset_root(
    request: pytest.FixtureRequest, real_subject: Path, tmp_path: Path
) -> None:
    data = _fetch_worked_example(request.config.cache.mkdir("ds000102-sub-01-func"))
    extra = _fetch_extra(request.config.cache.mkdir("ds000102-sub-01-func-bids"))
    ds = bids_root(tmp_path / "bids", data, extra, real_subject)
    before = snapshot_links(tmp_path / "bids")

    found = api.find_folders(ds)
    assert found.deriv_root.resolve() == (ds / "derivatives" / "fmriprep").resolve()
    assert found.raw_root is not None and found.raw_root.resolve() == ds.resolve()
    assert found.match.pipeline_version == "21.0.2"

    out = tmp_path / "out"
    config = EpiConfig(portable_voxel_map=False)
    lines: list[str] = []
    stream = api.epi_subject(ds, "sub-01", out, config=config, progress=lines.append)
    assert snapshot_links(tmp_path / "bids") == before
    assert read_info(out / STREAM_JSON, StreamInfo) == stream and len(stream.sessions) == 1
    assert lines[0].startswith(
        "Functional: 2 runs in 1 session of sub-01 in derivatives/fmriprep (fMRIPrep 21.0.2),"
        " raw series in . (BIDS); 2 can be packed (T1w space)"
    )
    info = read_info(out / stream.sessions[0] / SESSION_JSON, SessionInfo)
    assert info is not None and info.registration is not None
    assert info.registration.rule == "name_order"
    assert [r.name for r in info.runs] == ["task-flanker_run-1", "task-flanker_run-2"]
    assert info.runs[0].anchor and info.contours_from == "sub-01"
    assert info.grids[0].contours is not None and info.grids[0].shape == (49, 62, 36)
    frame = info.runs[1].frame
    assert frame is not None and not frame.flagged
    for name, value in EXPECTED_FRAME.items():
        assert abs(getattr(frame, name) - value) <= TOLERANCE, name
    assert [r.n_volumes for r in info.runs] == [146, 146] and info.runs[1].first == 146

    # the decoded streams against a pack made directly from the same files
    session_dir = out / info.session
    connector = fmriprep.make()
    anchor_xfm = data / "deriv/run-1_from-scanner_to-T1w_mode-image_xfm.txt"
    anchor_matrix = itk.read_itk_file(anchor_xfm)
    direct_dir = tmp_path / "direct" / info.session
    for n, run in ((1, info.runs[0]), (2, info.runs[1])):
        pack = read_info(session_dir / run.run / PACK_JSON, PackInfo)
        assert pack is not None and pack.raw_affine == "header"
        files = {
            FuncRole.RAW_BOLD: data / f"raw/run-{n}_bold.nii.gz",
            FuncRole.PRE_BOLD: data / f"deriv/run-{n}_space-T1w_desc-preproc_bold.nii.gz",
            FuncRole.PRE_SIDECAR: data / f"deriv/run-{n}_space-T1w_desc-preproc_bold.json",
            FuncRole.PRE_MASK: data / f"deriv/run-{n}_space-T1w_desc-brain_mask.nii.gz",
            FuncRole.XFM_TO_ANAT: data / f"deriv/run-{n}_from-scanner_to-T1w_mode-image_xfm.txt",
            FuncRole.REFERENCE: data / f"deriv/run-{n}_boldref.nii.gz",
            FuncRole.CONFOUNDS: data / f"deriv/run-{n}_desc-confounds_timeseries.tsv",
        }
        rigid = connector.raw_to_anat(_deriv_run(found, connector, n), files)
        direct = pack_run(
            direct_dir,
            RunPackInput(
                run=run.run,
                name=run.name,
                files=files,
                grid=geometry.grid_of(files[FuncRole.PRE_BOLD]),
                display_to_anat=to_matrix4(np.eye(4)),
                anchor_run=info.runs[0].run,
                anchor_matrix=anchor_matrix,
                raw_affine=session.raw_affine_choice(
                    files[FuncRole.RAW_BOLD], files[FuncRole.REFERENCE]
                ),
                motion=connector.motion(_deriv_run(found, connector, n), files),
                corrections=connector.corrections(_deriv_run(found, connector, n), files),
                rigid=rigid,
                tr_s=pack.tr_s,
            ),
            config,
        )
        assert direct.window == pack.window and direct.sd_window == pack.sd_window
        for stream_name in ("raw", "pre"):
            ours = list(stream_bytes(session_dir, pack, stream_name))
            theirs = list(stream_bytes(direct_dir, direct, stream_name))
            assert len(ours) == len(theirs) == 146
            assert [t for t in range(146) if ours[t] != theirs[t]] == [], (n, stream_name)


def _deriv_run(found: api.FuncFolders, connector: fmriprep.FmriprepConnector, n: int) -> DerivRun:
    runs = connector.list_runs(found.deriv_root, "sub-01", found.match)
    return next(r for r in runs if r.entities.run == str(n))
