"""A run on two real subjects from OpenNeuroDerivatives, each linked into an fMRIPrep layout.

``ds000102`` sub-01 (FreeSurfer 6.0.1) and ``ds003020`` sub-UTS01 (FreeSurfer 7.3.2, pial
surfaces as ``?h.pial.T1`` only). Run with ``uv run pytest -m network tests/test_network_run.py
-q -s`` to see the numbers of the FreeSurfer 7 render.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import nibabel.freesurfer as fs
import pytest
from synth import snapshot_tree  # pyrefly: ignore[missing-import]

from surfcensus.inventory.provenance import decode_text, parse_recon_log
from surfcensus.render.encode import read_image
from surfcensus.render.model import PANEL_ORDER, RenderInfo
from surfcensus.run import Run
from surfcensus.sources.local import local

pytestmark = pytest.mark.network

FLAGS = ("t2pial", "flairpial", "hires", "expert", "xopts", "edited")


def linked(tmp_path: Path, subject: Path, label: str, fmriprep: str) -> Path:
    """An fMRIPrep output whose one subject is a link to the cached files."""
    study = tmp_path / "study"
    subjects_dir = study / "sourcedata" / "freesurfer"
    subjects_dir.mkdir(parents=True)
    (subjects_dir / label).symlink_to(subject, target_is_directory=True)
    description = {"Name": "fMRIPrep", "GeneratedBy": [{"Name": "fMRIPrep", "Version": fmriprep}]}
    (study / "dataset_description.json").write_text(json.dumps(description))
    return study


def inventoried(run: Run) -> dict[str, object]:
    report = run.inventory()
    assert (report.done, report.skipped, report.failed) == (1, 0, 0)
    (row,) = run.public().subjects().to_dict("records")
    return row


def main_log_invocations(subject: Path) -> int:
    text = decode_text((subject / "scripts" / "recon-all.log").read_bytes())
    return len(parse_recon_log(text).invocations)


def test_inventory_of_the_freesurfer_6_subject(real_subject: Path, tmp_path: Path) -> None:
    before = snapshot_tree(real_subject)
    study = linked(tmp_path, real_subject, "sub-01", "21.0.2")
    with Run(tmp_path / "run", local(study)) as run:
        row = inventoried(run)
        (dataset,) = run.public().datasets().to_dict("records")
        assert run.inventory().skipped == 1

    assert (dataset["layout"], dataset["pipeline"], dataset["pipeline_version"]) == (
        "fmriprep",
        "fmriprep",
        "21.0.2",
    )
    assert (row["subject_label"], row["participant"]) == ("sub-01", "sub-01")
    assert (row["complete"], row["tier_a"], row["tier_b"]) == (True, True, True)
    assert row["missing_roles"] == "" and row["problem"] is None
    assert (row["fs_version"], row["fs_major"], row["pial_variant"]) == ("6.0.1", 6, "pial")
    # four invocations in recon-all.log and two in each per-hemisphere log
    assert main_log_invocations(real_subject) == 4 and row["n_invocations"] == 8
    assert row["recon_finished"] == True  # noqa: E712  (a pandas boolean)
    assert [bool(row[name]) for name in FLAGS] == [False] * 6
    assert (row["holes_lh"], row["holes_rh"]) == (41, 26)
    aseg = (real_subject / "stats" / "aseg.stats").read_text()
    assert re.search(r"^# Measure lhSurfaceHoles, .*, 41, unitless$", aseg, re.MULTILINE)
    assert re.search(r"^# Measure rhSurfaceHoles, .*, 26, unitless$", aseg, re.MULTILINE)
    assert (row["trunc_lh_1"], row["trunc_lh_2"]) == (39, 225)
    assert (row["trunc_rh_1"], row["trunc_rh_2"]) == (93, 434)
    flags = json.loads(row["recon_flags"])
    assert len(flags) > 10 and flags == sorted(flags)
    assert not any("/" in flag or "ds000102" in flag for flag in flags)
    assert "ds000102" not in json.dumps(row, default=str)
    assert snapshot_tree(real_subject) == before


def test_the_freesurfer_7_subject_is_inventoried_and_rendered(
    real_subject_fs7: Path, tmp_path: Path
) -> None:
    before = snapshot_tree(real_subject_fs7)
    study = linked(tmp_path, real_subject_fs7, "sub-UTS01", "23.0.1")
    with Run(tmp_path / "run", local(study)) as run:
        row = inventoried(run)
        start = time.perf_counter()
        report = run.render()
        seconds = time.perf_counter() - start
        media = run.media_dir(row["subject_uid"])
        (render,) = run.public().renders().to_dict("records")
        assert run.render().skipped == 1

    # inventory: FreeSurfer 7 writes ?h.pial.T1, and the bucket holds no ?h.pial key
    assert (row["fs_version"], row["fs_major"], row["pial_variant"]) == ("7.3.2", 7, "pial.T1")
    assert (row["complete"], row["tier_a"], row["tier_b"]) == (True, True, True)
    assert main_log_invocations(real_subject_fs7) == 5 and row["n_invocations"] == 9
    assert row["recon_finished"] == True  # noqa: E712
    flags = json.loads(row["recon_flags"])
    assert "-noT2pial" in flags and "-noFLAIRpial" in flags
    assert [bool(row[name]) for name in FLAGS] == [False] * 6
    assert (row["trunc_lh_1"], row["trunc_lh_2"]) == (22, 230)
    assert (row["trunc_rh_1"], row["trunc_rh_2"]) == (23, 325)
    assert (row["holes_lh"], row["holes_rh"]) == (4, 15)
    # over-maximum thickness is clamped at exactly 5.0 in FreeSurfer 7.3.2 as well
    for hemi in ("lh", "rh"):
        thickness = fs.read_morph_data(str(real_subject_fs7 / "surf" / f"{hemi}.thickness"))
        assert thickness.max() == 5.0

    # render: the first run of the renderer on FreeSurfer 7 data
    assert (report.done, report.failed) == (1, 0), render
    info = RenderInfo.model_validate_json((media / "render.json").read_text())
    assert info.inputs["lh_pial"].name == "lh.pial.T1"
    assert info.inputs["rh_pial"].name == "rh.pial.T1"
    assert info.volume.shape == (256, 256, 256) and info.volume.voxel_size_mm == (1.0, 1.0, 1.0)
    assert [len(info.panels[name].slices) for name in PANEL_ORDER] == [61, 54, 61]
    # the poles lie at the back of the head, left and right of the midline
    (lx, ly, _), (rx, ry, _) = info.poles["lh"].tkr, info.poles["rh"].tkr
    assert lx < 0 < rx and ly < -60 and ry < -60
    drawn = 0
    tiles = 0
    for name in PANEL_ORDER:
        for position in range(len(info.panels[name].slices)):
            tiles += 1
            drawn += bool(
                read_image(media / info.tiles.contours_path(name, position))[..., 3].any()
            )
    assert drawn > 0.8 * tiles
    assert not (media / "anim.mp4").exists() and (media / "poster.webp").is_file()
    assert render["n_frames"] == 120 and len(render["render_key"]) == 16
    total = sum(p.stat().st_size for p in media.rglob("*") if p.is_file())
    assert 1e6 < total < 15e6 and seconds < 120
    print(
        f"\nFreeSurfer 7.3.2 subject: rendered in {seconds:.1f} s, {total / 1e6:.2f} MB, "
        f"{drawn} of {tiles} tiles with contours; poles lh {info.poles['lh'].tkr}, "
        f"rh {info.poles['rh'].tkr}; window {info.window}"
    )
    assert snapshot_tree(real_subject_fs7) == before
