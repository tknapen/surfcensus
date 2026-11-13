"""Metrics and renders with a template on real subjects from OpenNeuroDerivatives.

``ds000102`` sub-01 (FreeSurfer 6.0.1) and ``ds003020`` sub-UTS01 (FreeSurfer 7.3.2) with the
``fsaverage`` directory that lies next to the subjects of ``ds000102``. Run with
``uv run pytest -m network tests/test_network_metrics.py -q -s`` to see the numbers.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pytest
from synth import snapshot_tree  # pyrefly: ignore[missing-import]

from surfcensus.config import MetricsConfig
from surfcensus.fsio.annot import read_annotation
from surfcensus.fsio.morph import read_morph
from surfcensus.fsio.stats import aparc_frame, read_stats, surface_holes
from surfcensus.fsio.surfaces import read_surface
from surfcensus.metrics.ceiling import ceiling_counts, native_region
from surfcensus.metrics.fsaverage import read_maps, resample_hemi
from surfcensus.metrics.pipeline import map_path
from surfcensus.render import RenderConfig, render_subject, three_plane
from surfcensus.render.model import RenderInfo
from surfcensus.render.vertexmap import read_vertex_map
from surfcensus.run import Run
from surfcensus.schemas import SCHEMAS
from surfcensus.sources.base import HEMIS
from surfcensus.sources.local import local
from surfcensus.sources.patterns import resolve_subject_files
from surfcensus.store import CATALOG_FILE, Catalog, read_table, table_path
from surfcensus.templates import Template, build_template

pytestmark = pytest.mark.network

SURF2SURF = Path(__file__).parent / "data" / "surf2surf_ds000102_sub-01_lh.npy"
"""Thickness of sub-01 ``lh`` on fsaverage6 as ``mri_surf2surf`` gives it. Made once by a
developer with FreeSurfer (docs/dev/contract.md, 9.3); the comparison is skipped without it."""
METRIC_ROLES = {
    f"{hemi}_{kind}" for hemi in HEMIS for kind in ("stats", "thickness", "annot", "sphere_reg")
}
"""The file roles that the ``regional`` and ``fsaverage`` stages read."""


@pytest.fixture(scope="module")
def template(real_template: Path) -> Template:
    return build_template(real_template, "adjacent")


def study_of(tmp_path: Path, subjects: dict[str, Path], fsaverage: Path) -> Path:
    """An fMRIPrep output whose subjects and ``fsaverage`` are links to the cached files."""
    study = tmp_path / "study"
    subjects_dir = study / "sourcedata" / "freesurfer"
    subjects_dir.mkdir(parents=True)
    for label, path in subjects.items():
        (subjects_dir / label).symlink_to(path, target_is_directory=True)
    (subjects_dir / "fsaverage").symlink_to(fsaverage, target_is_directory=True)
    description = {"Name": "fMRIPrep", "GeneratedBy": [{"Name": "fMRIPrep", "Version": "21.0.2"}]}
    (study / "dataset_description.json").write_text(json.dumps(description))
    return study


def test_stats_of_the_freesurfer_6_subject(real_subject: Path) -> None:
    lh = read_stats(real_subject / "stats" / "lh.aparc.stats")
    assert lh.measures["MeanThickness"] == (2.39047, "mm")
    assert lh.measures["NumVert"] == (140304.0, "unitless")
    assert "CortexVol" in lh.measures  # the measure line without its comma
    assert all(
        len(aparc_frame(read_stats(real_subject / "stats" / f"{hemi}.aparc.stats"))) == 34
        for hemi in HEMIS
    )
    assert surface_holes(read_stats(real_subject / "stats" / "aseg.stats")) == (41, 26)


def test_the_map_of_the_freesurfer_6_subject(real_subject: Path, template: Template) -> None:
    config = MetricsConfig()
    at_ceiling = {}
    for hemi in HEMIS:
        thickness = read_morph(real_subject / "surf" / f"{hemi}.thickness")
        sphere = read_surface(real_subject / "surf" / f"{hemi}.sphere.reg")
        annot = read_annotation(real_subject / "label" / f"{hemi}.aparc.annot")
        half = template.hemis[hemi]
        started = time.perf_counter()
        hemi_map, smap = resample_hemi(thickness, sphere, half, annot=annot, config=config)
        seconds = time.perf_counter() - started
        n_cortex = int(half.cortex.sum())
        covered = n_cortex - hemi_map.n_missing_cortex
        search, kind = native_region(
            "search", annot=annot, smap=smap, template_hemi=half, regions=config.regions
        )
        assert search is not None and kind == "template"
        counts = ceiling_counts(thickness, search)
        at_ceiling[hemi] = int((thickness == 5.0).sum())
        print(
            f"\nsub-01 {hemi}: {hemi_map.n_fallback} fallbacks, {covered} of {n_cortex} cortex "
            f"vertices with a value, mean {np.nanmean(hemi_map.thickness):.3f} mm, global "
            f"median {hemi_map.global_median:.3f} mm, native search region {counts.n_region} "
            f"vertices ({counts.n_near} at 4.9 mm or more, {counts.n_zero} at 0), {seconds:.2f} s"
        )
        assert hemi_map.n_fallback <= 10 and not hemi_map.flipped
        assert covered >= 0.995 * n_cortex
        assert 2.3 < hemi_map.global_median < 2.6
        assert np.isnan(hemi_map.thickness[~half.cortex]).all()
        if hemi == "lh":
            assert (hemi_map.n_fallback, covered) == (3, 37_420)
            assert np.nanmean(hemi_map.thickness) == pytest.approx(2.43, abs=0.02)
            assert (counts.n_region, counts.n_near, counts.n_zero) == (21_484, 0, 0)
            assert counts.fraction == 0
    assert at_ceiling == {"lh": 16, "rh": 41}


def test_agreement_with_mri_surf2surf(real_subject: Path, template: Template) -> None:
    if not SURF2SURF.is_file():
        pytest.skip(f"{SURF2SURF.name} has not been made: it needs a FreeSurfer installation")
    theirs = np.load(SURF2SURF).astype(np.float64).ravel()
    half = template.hemis["lh"]
    hemi_map, _ = resample_hemi(
        read_morph(real_subject / "surf" / "lh.thickness"),
        read_surface(real_subject / "surf" / "lh.sphere.reg"),
        half,
        annot=read_annotation(real_subject / "label" / "lh.aparc.annot"),
        config=MetricsConfig(),
    )
    both = half.cortex & np.isfinite(hemi_map.thickness)
    difference = np.abs(hemi_map.thickness[both] - theirs[both])
    print(f"\nmri_surf2surf: mean |d| {difference.mean():.4f} mm, max {difference.max():.3f} mm")
    # the tolerance is set here once the first comparison has been seen


def test_render_with_the_template_has_counterparts_and_the_same_tiles(
    real_subject: Path, real_template: Path, template: Template, tmp_path: Path
) -> None:
    files = resolve_subject_files(real_subject)
    without = render_subject(files, tmp_path / "without", video=False)
    started = time.perf_counter()
    info = render_subject(files, tmp_path / "with", template=template, video=False)
    seconds = time.perf_counter() - started
    assert without.template is None and without.fsaverage_poles is None
    assert info.template is not None and info.template.template_id == template.template_id
    assert info.vertex_map is not None and info.vertex_map.fsaverage == "fsaverage6"
    assert {"lh_sphere_reg", "rh_sphere_reg"} <= set(info.inputs)

    vmap = read_vertex_map(tmp_path / "with" / "vertex_map.npz")
    bare = read_vertex_map(tmp_path / "without" / "vertex_map.npz")
    for hemi in HEMIS:
        counterparts = vmap[f"{hemi}_fsavg6"]
        assert counterparts.min() >= 0 and counterparts.max() < 40_962
        assert (bare[f"{hemi}_fsavg6"] == -1).all()
        np.testing.assert_array_equal(vmap[f"{hemi}_index"], bare[f"{hemi}_index"])
    assert info.fsaverage_poles is not None
    for hemi in HEMIS:
        pole = info.fsaverage_poles[hemi]
        assert pole.template_vertex == template.hemis[hemi].pole_vertex
        print(
            f"\nsub-01 {hemi}: the fsaverage pole lies {pole.distance_mm:.2f} mm from the "
            f"geometric pole (native vertex {pole.native_vertex})"
        )
        assert 0 <= pole.distance_mm < 10
    assert info.fsaverage_poles["lh"].distance_mm < 3

    # the tiles do not depend on the template
    tiles = sorted(p.name for p in (tmp_path / "with" / "tiles").iterdir())
    assert len(tiles) == 352
    for name in (*(f"tiles/{tile}" for tile in tiles), "poster.webp"):
        assert (tmp_path / "with" / name).read_bytes() == (tmp_path / "without" / name).read_bytes()
    size = sum(p.stat().st_size for p in (tmp_path / "with").rglob("*") if p.is_file())
    print(f"\nsub-01 with the template: {seconds:.1f} s, {size / 1e6:.1f} MB without video")
    assert seconds < 60 and size < 12e6

    # the template next to the subject is found without being named
    study = study_of(tmp_path, {"sub-01": real_subject}, real_template)
    found = three_plane(
        study / "sourcedata" / "freesurfer" / "sub-01",
        tmp_path / "auto",
        RenderConfig(panel_px=64),
        video=False,
    )
    assert found.template is not None and found.template.template_id == template.template_id


def test_a_run_on_both_real_subjects_with_the_template_next_to_them(
    real_subject: Path, real_subject_fs7: Path, real_template: Path, tmp_path: Path
) -> None:
    """Inventory, metrics and render through a run: two subjects are too few for a
    reference, so they are ranked as decision D16 says, with maps and counterparts."""
    before = {path: snapshot_tree(path) for path in (real_subject, real_subject_fs7, real_template)}
    study = study_of(
        tmp_path, {"sub-01": real_subject, "sub-UTS01": real_subject_fs7}, real_template
    )
    run_dir = tmp_path / "run"
    with Run(run_dir, local(study)) as run:
        assert run.inventory().done == 2
        subjects = run.public().subjects()
        assert subjects["complete"].all() and subjects["tier_a"].all() and subjects["tier_b"].all()
        uids = dict(zip(subjects["subject_label"], subjects["subject_uid"], strict=True))
        started = time.perf_counter()
        report = run.metrics()
        seconds = time.perf_counter() - started
        assert (report.done, report.skipped, report.failed) == (2 + 2 + 2, 0, 0)
        assert report.notes[0].startswith("2 subjects with a map: too few for per-location")
        template = run.template()
        assert template is not None and template.origin == "adjacent"
        assert template.template_id == "b710aa6e0467" and template.area_source == "white.avg.area"

        scores = run.scores().set_index("subject_uid")
        assert set(scores["score_kind"]) == {"fallback_rank"} and len(scores) == 2
        assert scores.loc[uids["sub-01"], "occ_ctrl_ratio"] == pytest.approx(0.811, abs=0.001)
        assert scores.loc[uids["sub-UTS01"], "occ_ctrl_ratio"] == pytest.approx(0.828, abs=0.001)
        assert set(scores["ceiling_region"]) == {"template"} and set(scores["flags"]) == {""}
        assert tuple(scores.loc[uids["sub-01"], ["holes_lh", "holes_rh"]]) == (41, 26)
        assert tuple(scores.loc[uids["sub-UTS01"], ["holes_lh", "holes_rh"]]) == (4, 15)

        summary = read_table(table_path(run_dir, "vertex_summary"), SCHEMAS["vertex_summary"])
        assert (summary["thickness_max_mm"] == 5.0).all()
        assert (summary["n_fallback"] <= 10).all() and (summary["n_missing_cortex"] < 190).all()
        assert set(summary["flags"]) == {""} and set(summary["region_kind"]) == {"template"}
        for label, uid in uids.items():
            maps = read_maps(map_path(run_dir, uid))
            rows = summary[summary["subject_uid"] == uid].set_index("hemi")
            print(
                f"\n{label}: "
                + "; ".join(
                    f"{hemi} mean {np.nanmean(maps[hemi]):.3f} mm, {rows.loc[hemi, 'n_fallback']} "
                    f"fallbacks, {rows.loc[hemi, 'n_missing_cortex']} cortex vertices without a "
                    f"value, ceiling fraction {rows.loc[hemi, 'ceiling_frac']:.2g}"
                    for hemi in HEMIS
                )
            )
            assert all(2.3 < np.nanmean(maps[hemi]) < 2.7 for hemi in HEMIS)
        print(f"\nmetrics of two subjects: {seconds:.1f} s")

        # before any render: the files the metrics read are in the catalog by content
        with Catalog(run_dir / CATALOG_FILE, readonly=True) as catalog:
            stored = catalog.query(
                "SELECT subject_uid, role, location, sha256 FROM files WHERE sha256 IS NOT NULL"
            )
        paths = {uids["sub-01"]: real_subject, uids["sub-UTS01"]: real_subject_fs7}
        for uid in paths:
            assert METRIC_ROLES <= {row["role"] for row in stored if row["subject_uid"] == uid}
        for row in stored:
            content = (paths[row["subject_uid"]] / row["location"]).read_bytes()
            assert row["sha256"] == hashlib.sha256(content).hexdigest(), row["location"]

        rendered = run.render()
        assert (rendered.done, rendered.failed) == (2, 0)
        renders = run.public().renders()
        assert set(renders["template_id"]) == {template.template_id}
        for label, uid in uids.items():
            info = RenderInfo.model_validate_json((run.media_dir(uid) / "render.json").read_text())
            vmap = read_vertex_map(run.media_dir(uid) / "vertex_map.npz")
            assert info.fsaverage_poles is not None and set(info.fsaverage_poles) == set(HEMIS)
            for hemi in HEMIS:
                assert vmap[f"{hemi}_fsavg6"].min() >= 0
                distance = info.fsaverage_poles[hemi].distance_mm
                print(f"\n{label} {hemi}: fsaverage pole {distance:.2f} mm from the geometric pole")

        # again: nothing is read, computed or rendered
        assert run.inventory().skipped == 2
        again = run.metrics()
        assert (again.done, again.skipped, again.failed) == (0, 6, 0)
        assert (run.render().done, run.render().skipped) == (0, 2)
    for path, snapshot in before.items():
        assert snapshot_tree(path) == snapshot
