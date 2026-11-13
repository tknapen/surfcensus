"""``review`` end to end on a real subject from OpenNeuroDerivatives: ``ds000102`` sub-01
(FreeSurfer 6.0.1) in a subjects directory with the real ``fsaverage`` next to it.

Run with ``uv run pytest -m network tests/test_network_review.py -q -s`` to see the times.
"""

from __future__ import annotations

import time
from pathlib import Path

import nibabel.freesurfer as fs
import numpy as np
import pandas as pd
import pytest
from synth import snapshot_tree  # pyrefly: ignore[missing-import]

pytest.importorskip("fastapi")
pytest.importorskip("sqlmodel")

from fastapi.testclient import TestClient

from surfcensus.app.server import create_app
from surfcensus.export.tables import EXPORT_TABLES, MANIFEST, export_run
from surfcensus.render import api
from surfcensus.render.model import read_render
from surfcensus.render.vertexmap import read_vertex_map
from surfcensus.review import rateable, render_counts, review

pytestmark = pytest.mark.network


@pytest.fixture
def subjects_dir(tmp_path: Path, real_subject: Path, real_template: Path) -> Path:
    """A subjects directory whose one subject and whose template are links to the cache."""
    root = tmp_path / "data" / "subjects"
    root.mkdir(parents=True)
    (root / "sub-01").symlink_to(real_subject, target_is_directory=True)
    (root / "fsaverage").symlink_to(real_template, target_is_directory=True)
    return root


def test_review_of_the_real_subject_from_the_files_to_the_export(
    subjects_dir: Path,
    real_subject: Path,
    real_template: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    before = snapshot_tree(real_subject), snapshot_tree(real_template)
    calls: list[Path] = []
    original = api.render_subject

    def recording(files: object, out: Path, *args: object, **kwargs: object) -> object:
        calls.append(Path(out))
        return original(files, out, *args, **kwargs)

    monkeypatch.setattr(api, "render_subject", recording)

    started = time.perf_counter()
    with review(subjects_dir, serve=False, jobs=1) as run:
        first = time.perf_counter() - started
        run_dir = run.run_dir
        assert run_dir == subjects_dir.parent / "surfcensus-run"
        (row,) = run.public().subjects().to_dict("records")
        assert (row["subject_label"], row["fs_version"]) == ("sub-01", "6.0.1")
        assert row["complete"] and row["tier_a"] and row["tier_b"]
        # one subject: no reference, ranked as decision D16 says
        (score,) = run.scores().to_dict("records")
        assert score["score_kind"] == "fallback_rank" and score["primary_rank"] == 1
        assert score["occ_ctrl_ratio"] == pytest.approx(0.811, abs=0.001)
        assert run.template() is not None
        selection = run.public().selection()
        assert selection["subject_uid"].tolist() == [row["subject_uid"]]
        assert selection["reason"].tolist() == ["all"]
        assert render_counts(run) == (1, 0, 0) and rateable(run) == 1
        uid = row["subject_uid"]
        info = read_render(run.media_dir(uid))
        assert info is not None and info.template is not None
        vertex_map = read_vertex_map(run.media_dir(uid) / "vertex_map.npz")
        # rendered with counterparts: every vertex of the map has an fsaverage6 vertex
        assert (vertex_map["lh_fsavg6"] >= 0).all() and (vertex_map["rh_fsavg6"] >= 0).all()
    said = capsys.readouterr().err
    assert "Found 1 subject (a subjects directory, FreeSurfer 6.0.1)." in said
    assert "Rendered: 1 of 1 subject." in said
    assert len(calls) == 1

    started = time.perf_counter()
    review(subjects_dir, serve=False, jobs=1).close()
    second = time.perf_counter() - started
    assert len(calls) == 1  # the second call rendered nothing

    # a rating with one mark on the real tiles, through the app
    with TestClient(create_app(run_dir), base_url="http://127.0.0.1") as client:
        version = client.get("/api/session").json()["protocol"]["version"]
        shown = client.get("/api/next").json()["assignment"]
        panel = shown["render"]["panels"]["coronal"]
        mark = {
            "client_id": "n1",
            "class": "B",
            "panel": "coronal",
            "slice": len(panel["slices"]) // 2,
            "px": 100.5,
            "py": 180.5,
        }
        body = {
            "assignment_id": shown["assignment_id"],
            "protocol_version": version,
            "values": {"A": 0, "B": 1, "C": 0},
            "marks": [mark],
            "client_ts": "2026-11-13T12:00:00.000Z",
            "ms_on_subject": 20000,
        }
        answer = client.post("/api/rating", json=body)
        assert answer.status_code == 200, answer.text
        (saved,) = answer.json()["saved"]["marks"]

    paths = export_run(run_dir)
    assert [path.name for path in paths] == [f"{name}.csv" for name in EXPORT_TABLES] + [MANIFEST]
    tables = {path.stem: pd.read_csv(path, dtype={"subject_uid": str}) for path in paths[:-1]}
    assert tables["subjects"]["final_B"].tolist() == [1]
    assert tables["subjects"]["fs_version"].tolist() == ["6.0.1"]
    assert tables["ratings"]["ms_on_subject"].tolist() == [20000]
    (exported,) = tables["marks"].to_dict("records")
    assert (exported["class"], exported["panel"], exported["hemi"]) == (
        "B",
        "coronal",
        saved["hemi"],
    )
    # the stored vertex is the nearest one of the surface file, at the stored distance
    point = np.array([exported["tkr_x"], exported["tkr_y"], exported["tkr_z"]])
    nearest = {}
    for hemi in ("lh", "rh"):
        coords = fs.read_geometry(str(real_subject / "surf" / f"{hemi}.pial"))[0]
        distance = np.linalg.norm(coords.astype(np.float64) - point, axis=1)
        nearest[hemi] = (float(distance.min()), int(distance.argmin()))
    hemi = min(nearest, key=lambda h: nearest[h][0])
    if nearest[hemi][0] <= 5.0:
        assert exported["hemi"] == hemi and exported["pial_vertex"] == nearest[hemi][1]
        assert exported["pial_dist_mm"] == pytest.approx(nearest[hemi][0], abs=1e-6)
        assert exported["fsavg6_vertex"] >= 0
    else:
        assert pd.isna(exported["pial_vertex"])

    assert (snapshot_tree(real_subject), snapshot_tree(real_template)) == before
    print(f"\nreview of ds000102 sub-01: {first:.1f} s the first time, {second:.1f} s the second")
