"""Thickness on the template mesh: the resampled map, the stored file, the per-vertex
reference, z-maps and strata.

Subjects are those of ``synth_tierb``, whose registration, thickness and parcels are known
in closed form, on templates of ``synth_template`` (an order-5 directory, used as
fsaverage4 and fsaverage5). "One vertex ring" around a patch is the longest edge of the
subject's registered sphere: a template point lies in a face of that sphere, and every
corner of the face is within one edge of the point.

The last test reads ``ds000102`` sub-01 and the ``fsaverage`` next to it from the pytest
cache and skips when the files are absent.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping
from pathlib import Path

import nibabel.freesurfer as fs
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from synth import (  # pyrefly: ignore[missing-import]
    SyntheticSubject,
    write_synthetic_subject,
)
from synth_template import (  # pyrefly: ignore[missing-import]
    SyntheticTemplate,
    write_synthetic_template,
)
from synth_tierb import (  # pyrefly: ignore[missing-import]
    SyntheticCohort,
    ThickPatch,
    add_tier_b,
    cohort_patches,
    expected_thickness,
)

from surfcensus._io import read_npz, sha256_file, write_npz
from surfcensus.config import MetricsConfig
from surfcensus.fsio import registration
from surfcensus.fsio.annot import Annotation, read_annotation
from surfcensus.fsio.morph import read_morph
from surfcensus.fsio.surfaces import Surface, read_surface
from surfcensus.metrics import MetricError
from surfcensus.metrics import fsaverage as fa
from surfcensus.metrics.fsaverage import (
    HemiMap,
    assign_strata,
    build_reference,
    global_median,
    read_maps,
    resample_hemi,
    stratum_members,
    write_maps,
    zmap,
)
from surfcensus.reference.load import Reference, ReferenceMaps, read_reference, write_reference
from surfcensus.reference.strata import members_sha256, set_reference_id
from surfcensus.sources.base import HEMIS
from surfcensus.templates import Template, TemplateHemi, build_template

CONFIG4 = MetricsConfig(mesh="fsaverage4")
CONFIG5 = MetricsConfig(mesh="fsaverage5")
FLOAT16_STEP_MM = 0.002
"""Half the spacing of float16 between 4 and 8, rounded up: the most a stored thickness
differs from the computed one."""


@pytest.fixture(scope="module")
def small(tmp_path_factory: pytest.TempPathFactory) -> SyntheticTemplate:
    """An ``fsaverage`` directory of order 5. Read-only."""
    return write_synthetic_template(tmp_path_factory.mktemp("small"), order=5)


@pytest.fixture(scope="module")
def tpl4(small: SyntheticTemplate) -> Template:
    """2,562 vertices per hemisphere."""
    return build_template(small.path, "explicit", mesh="fsaverage4")


@pytest.fixture(scope="module")
def tpl5(small: SyntheticTemplate) -> Template:
    """10,242 vertices per hemisphere."""
    return build_template(small.path, "explicit", mesh="fsaverage5")


@pytest.fixture(scope="module")
def plain(tmp_path_factory: pytest.TempPathFactory, small: SyntheticTemplate) -> SyntheticSubject:
    """A subject of 2,562 vertices with Tier B, without noise or patch. Read-only."""
    subject = write_synthetic_subject(
        tmp_path_factory.mktemp("plain"), voxel_size=10.0, shape=(16, 16, 16), subdivisions=4
    )
    add_tier_b(subject, small)
    return subject


def tier_b(subject: SyntheticSubject, hemi: str) -> tuple[np.ndarray, Surface, Annotation]:
    return (
        read_morph(subject.path / "surf" / f"{hemi}.thickness"),
        read_surface(subject.path / "surf" / f"{hemi}.sphere.reg"),
        read_annotation(subject.path / "label" / f"{hemi}.aparc.annot"),
    )


def mapped(
    subject: SyntheticSubject, hemi: str, template: Template, config: MetricsConfig
) -> HemiMap:
    thickness, sphere, annot = tier_b(subject, hemi)
    return resample_hemi(thickness, sphere, template.hemis[hemi], annot=annot, config=config)[0]


def longest_edge_deg(sphere: Surface) -> float:
    """The longest edge of a registered sphere, as an angle."""
    unit = sphere.vertices / np.linalg.norm(sphere.vertices, axis=1, keepdims=True)
    cosine = min(
        float((unit[sphere.faces[:, a]] * unit[sphere.faces[:, b]]).sum(axis=1).min())
        for a, b in ((0, 1), (1, 2), (2, 0))
    )
    return math.degrees(math.acos(cosine))


def angle_deg(directions: np.ndarray, centre: tuple[float, float, float]) -> np.ndarray:
    c = np.asarray(centre) / np.linalg.norm(centre)
    return np.degrees(np.arccos(np.clip(directions @ c, -1.0, 1.0)))


def as_float16(values: np.ndarray) -> np.ndarray:
    return values.astype(np.float16).astype(np.float32)


# --- resample_hemi ---


@pytest.mark.parametrize(
    ("subdivisions", "mesh", "tolerance_mm"),
    [(4, "fsaverage4", 0.02), (4, "fsaverage5", 0.02), (5, "fsaverage4", 0.008)],
)
@pytest.mark.parametrize("warp", [0.0, 0.15])
def test_the_map_of_a_subject_without_noise_is_the_closed_form(
    small: SyntheticTemplate,
    tpl4: Template,
    tpl5: Template,
    tmp_path: Path,
    subdivisions: int,
    mesh: str,
    tolerance_mm: float,
    warp: float,
) -> None:
    """Measured: at most 0.016 mm from a sphere of 2,562 vertices and 0.005 mm from one of
    10,242, a mean of 0.0005 mm. Where the template's vertices are the subject's own (no
    warp, fsaverage4) only the rounding to float16 is left, at most 0.001 mm."""
    template = {"fsaverage4": tpl4, "fsaverage5": tpl5}[mesh]
    subject = write_synthetic_subject(
        tmp_path, voxel_size=10.0, shape=(16, 16, 16), subdivisions=subdivisions
    )
    add_tier_b(subject, small, warp=warp)
    for hemi in HEMIS:
        half = template.hemis[hemi]
        thickness, sphere, annot = tier_b(subject, hemi)
        hemi_map, smap = resample_hemi(
            thickness, sphere, half, annot=annot, config=MetricsConfig(mesh=mesh)
        )
        values = hemi_map.thickness
        assert values.dtype == np.float32 and values.shape == (len(half.sphere),)
        has = np.isfinite(values)
        assert not has[~half.cortex].any()
        error = np.abs(values[has] - expected_thickness(half.sphere[has]))
        assert error.max() < tolerance_mm
        assert error.mean() < 0.001
        assert hemi_map.n_missing_cortex == int(half.cortex.sum()) - int(has.sum())
        assert has.sum() > 0.99 * half.cortex.sum()
        assert (hemi_map.n_fallback, hemi_map.flipped) == (0, False)
        assert hemi_map.global_median == global_median(values, half)
        assert (smap.n_source, smap.n_target) == (len(thickness), len(half.sphere))
        assert smap.flipped is False


def test_the_map_holds_the_values_of_the_float16_file(
    plain: SyntheticSubject, tpl4: Template, tmp_path: Path
) -> None:
    half = tpl4.hemis["lh"]
    thickness, sphere, annot = tier_b(plain, "lh")
    hemi_map, smap = resample_hemi(thickness, sphere, half, annot=annot, config=CONFIG4)
    values = hemi_map.thickness
    assert np.array_equal(values, as_float16(values), equal_nan=True)
    # they are the interpolated values rounded once, not something else that float16 holds
    valid = (thickness > 0) & annot.cortex()
    unrounded = registration.resample(thickness, smap, valid=valid, min_valid_weight=0.5)
    unrounded[~half.cortex] = np.nan
    assert np.array_equal(values, as_float16(unrounded), equal_nan=True)
    assert np.nanmax(np.abs(values - unrounded)) <= 0.001  # thickness is below 4 mm here

    write_maps(tmp_path / "uid.npz", {"lh": values})
    assert np.array_equal(read_maps(tmp_path / "uid.npz")["lh"], values, equal_nan=True)
    # so the median of the stored map is the one that was returned
    assert global_median(read_maps(tmp_path / "uid.npz")["lh"], half) == hemi_map.global_median


def test_a_constant_field_stays_constant(plain: SyntheticSubject, tpl4: Template) -> None:
    half = tpl4.hemis["lh"]
    thickness, sphere, _ = tier_b(plain, "lh")
    constant = np.full_like(thickness, 2.75)
    hemi_map, _ = resample_hemi(constant, sphere, half, annot=None, config=CONFIG4)
    assert np.array_equal(hemi_map.thickness[half.cortex], np.full(half.cortex.sum(), 2.75))
    assert np.isnan(hemi_map.thickness[~half.cortex]).all()
    assert hemi_map.n_missing_cortex == 0
    assert hemi_map.global_median == 2.75


def test_zeros_and_values_outside_cortex_do_not_leak(
    plain: SyntheticSubject, tpl4: Template
) -> None:
    half = tpl4.hemis["lh"]
    thickness, sphere, annot = tier_b(plain, "lh")
    cortex = annot.cortex()
    assert (thickness[~cortex] == 0).all() and (thickness[cortex] > 0).all()
    planted = thickness.copy()
    rng = np.random.default_rng(7)
    inside = np.flatnonzero(cortex)
    zeroed, not_a_number, infinite = np.split(rng.permutation(inside)[:120], [100, 110])
    planted[zeroed] = 0.0
    planted[not_a_number] = np.nan
    planted[infinite] = np.inf
    planted[~cortex] = 9.0  # the medial wall, where the annotation has no label
    good = cortex.copy()
    good[np.concatenate([zeroed, not_a_number, infinite])] = False
    low, high = float(planted[good].min()), float(planted[good].max())
    assert 1.9 < low and high < 3.1

    hemi_map, _ = resample_hemi(planted, sphere, half, annot=annot, config=CONFIG4)
    values = hemi_map.thickness[np.isfinite(hemi_map.thickness)]
    assert values.size > 0.9 * half.cortex.sum()
    # a weighted mean of counted vertices lies between the smallest and the largest of them
    assert values.min() >= low - 0.001 and values.max() <= high + 0.001
    # and stays near the field, although one vertex in twenty around it was unusable
    has = np.isfinite(hemi_map.thickness)
    assert np.abs(hemi_map.thickness[has] - expected_thickness(half.sphere[has])).max() < 0.05

    # without the annotation nothing says that the 9.0 is not cortex
    leaking, _ = resample_hemi(planted, sphere, half, annot=None, config=CONFIG4)
    assert np.nanmax(leaking.thickness) > 4.0
    # and a zero is never averaged in, with or without it
    assert np.nanmin(leaking.thickness) >= low - 0.001


def test_the_config_reaches_the_sphere_map_and_the_weights(
    plain: SyntheticSubject, tpl4: Template, monkeypatch: pytest.MonkeyPatch
) -> None:
    half = tpl4.hemis["lh"]
    thickness, sphere, annot = tier_b(plain, "lh")
    seen: list[int] = []
    build = fa.build_sphere_map

    def recording(source: Surface, target: np.ndarray, *, k: int = 8) -> registration.SphereMap:
        seen.append(k)
        return build(source, target, k=k)

    monkeypatch.setattr(fa, "build_sphere_map", recording)
    missing = {
        weight: resample_hemi(
            thickness,
            sphere,
            half,
            annot=annot,
            config=MetricsConfig(mesh="fsaverage4", knn=5, min_valid_weight=weight),
        )[0].n_missing_cortex
        for weight in (0.0, 0.5, 1.0)
    }
    assert seen == [5, 5, 5]
    # the template cortex next to the medial wall lies in faces with a corner on the wall
    assert 0 <= missing[0.0] < missing[0.5] < missing[1.0]


def test_low_coverage_raises_and_the_bound_is_the_config(
    plain: SyntheticSubject, tpl4: Template
) -> None:
    half = tpl4.hemis["lh"]
    thickness, sphere, annot = tier_b(plain, "lh")
    posterior_only = np.where(sphere.vertices[:, 1] < 0, thickness, 0.0).astype(np.float32)
    with pytest.raises(MetricError) as caught:
        resample_hemi(posterior_only, sphere, half, annot=annot, config=CONFIG4)
    assert caught.value.reason == "low coverage"
    n_cortex = int(half.cortex.sum())
    assert f"of {n_cortex} template cortex vertices" in caught.value.detail
    assert "90% are needed" in caught.value.detail

    lenient = MetricsConfig(mesh="fsaverage4", min_coverage=0.3)
    hemi_map, _ = resample_hemi(posterior_only, sphere, half, annot=annot, config=lenient)
    n_value = n_cortex - hemi_map.n_missing_cortex
    assert 0.4 * n_cortex < n_value < 0.6 * n_cortex
    assert caught.value.detail.startswith(f"{n_value} of ")

    just_met = MetricsConfig(mesh="fsaverage4", min_coverage=(n_value - 0.5) / n_cortex)
    resample_hemi(posterior_only, sphere, half, annot=annot, config=just_met)
    just_missed = MetricsConfig(mesh="fsaverage4", min_coverage=(n_value + 0.5) / n_cortex)
    with pytest.raises(MetricError, match="low coverage"):
        resample_hemi(posterior_only, sphere, half, annot=annot, config=just_missed)

    # an annotation without cortex leaves nothing, whatever the thickness
    no_cortex = Annotation(labels=np.full(len(thickness), -1, dtype=np.int32), names=annot.names)
    with pytest.raises(MetricError, match="low coverage: 0 of"):
        resample_hemi(thickness, sphere, half, annot=no_cortex, config=CONFIG4)


def test_too_many_fallbacks_are_a_degenerate_sphere(
    plain: SyntheticSubject, tpl4: Template
) -> None:
    half = tpl4.hemis["lh"]
    thickness, sphere, annot = tier_b(plain, "lh")
    # a sphere that lost the faces of its anterior half: template points there lie in none
    centre_y = sphere.vertices[sphere.faces].mean(axis=1)[:, 1]
    holed = Surface(vertices=sphere.vertices, faces=sphere.faces[centre_y < 0])
    with pytest.raises(MetricError) as caught:
        resample_hemi(thickness, holed, half, annot=annot, config=CONFIG4)
    assert caught.value.reason == "degenerate sphere"
    assert "template vertices lie in no face of the sphere" in caught.value.detail

    anything = MetricsConfig(mesh="fsaverage4", max_fallback_frac=1.0)
    hemi_map, smap = resample_hemi(thickness, holed, half, annot=annot, config=anything)
    n_target = len(half.sphere)
    assert hemi_map.n_fallback == int(smap.fallback.sum())
    assert 0.4 * n_target < hemi_map.n_fallback < 0.6 * n_target
    assert caught.value.detail.startswith(f"{hemi_map.n_fallback} of {n_target} ")

    allowed = MetricsConfig(
        mesh="fsaverage4", max_fallback_frac=(hemi_map.n_fallback + 0.5) / n_target
    )
    assert resample_hemi(thickness, holed, half, annot=annot, config=allowed)[0].n_fallback
    refused = MetricsConfig(
        mesh="fsaverage4", max_fallback_frac=(hemi_map.n_fallback - 0.5) / n_target
    )
    with pytest.raises(MetricError, match="degenerate sphere"):
        resample_hemi(thickness, holed, half, annot=annot, config=refused)


@pytest.mark.parametrize(
    ("change", "says"),
    [
        (lambda v: v * np.where(np.arange(len(v)) % 3 == 0, 1.5, 1.0)[:, None], "median radius"),
        (lambda v: np.where(np.arange(len(v))[:, None] == 5, 0.0, v), "zero or not finite"),
        (lambda v: np.where(np.arange(len(v))[:, None] == 5, np.nan, v), "zero or not finite"),
    ],
)
def test_a_sphere_that_is_none_is_a_degenerate_sphere(
    plain: SyntheticSubject,
    tpl4: Template,
    change: Callable[[np.ndarray], np.ndarray],
    says: str,
) -> None:
    thickness, sphere, annot = tier_b(plain, "lh")
    broken = Surface(vertices=change(sphere.vertices), faces=sphere.faces)
    with pytest.raises(MetricError) as caught:
        resample_hemi(thickness, broken, tpl4.hemis["lh"], annot=annot, config=CONFIG4)
    assert caught.value.reason == "degenerate sphere"
    assert says in caught.value.detail
    assert not caught.value.detail.startswith(("degenerate", ":"))


def test_a_sphere_wound_inward_gives_the_same_map_and_says_so(
    small: SyntheticTemplate, plain: SyntheticSubject, tpl4: Template, tmp_path: Path
) -> None:
    flipped = write_synthetic_subject(tmp_path, voxel_size=10.0, shape=(16, 16, 16), subdivisions=4)
    add_tier_b(flipped, small, flip_winding=True)
    for hemi in HEMIS:
        outward, inward = (mapped(subject, hemi, tpl4, CONFIG4) for subject in (plain, flipped))
        assert (outward.flipped, inward.flipped) == (False, True)
        assert inward.n_fallback == 0
        assert np.array_equal(inward.thickness, outward.thickness, equal_nan=True)
        assert inward.global_median == outward.global_median


def test_files_of_different_vertex_counts_are_a_mismatch(
    plain: SyntheticSubject, tpl4: Template
) -> None:
    half = tpl4.hemis["lh"]
    thickness, sphere, annot = tier_b(plain, "lh")
    n = len(thickness)
    short_annot = Annotation(labels=annot.labels[:-1], names=annot.names)
    other_sphere = Surface(vertices=sphere.vertices[:642], faces=sphere.faces[:10] % 642)
    cases = [
        ((thickness[:-1], sphere, annot), f"thickness {n - 1}, sphere.reg {n}, annotation {n}"),
        ((thickness, sphere, short_annot), f"thickness {n}, sphere.reg {n}, annotation {n - 1}"),
        ((thickness, other_sphere, annot), f"thickness {n}, sphere.reg 642, annotation {n}"),
        ((thickness, other_sphere, None), f"thickness {n}, sphere.reg 642 vertices"),
    ]
    for (values, surface, labels), says in cases:
        with pytest.raises(MetricError) as caught:
            resample_hemi(values, surface, half, annot=labels, config=CONFIG4)
        assert caught.value.reason == "vertex count mismatch"
        assert says in caught.value.detail


# --- the session cohort ---


@pytest.fixture(scope="module")
def cohort_maps(
    synth_cohort: SyntheticCohort, tpl5: Template, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, dict[str, np.ndarray]]:
    """The maps of the cohort's subjects that have a ``sphere.reg``, on fsaverage5, written
    as files and read back: subject name -> hemisphere -> map."""
    directory = tmp_path_factory.mktemp("fsavg")
    maps = {}
    for subject in synth_cohort.subjects:
        name = subject.path.name
        if name in synth_cohort.no_sphere_reg:
            continue
        write_maps(
            directory / f"{name}.npz",
            {hemi: mapped(subject, hemi, tpl5, CONFIG5).thickness for hemi in HEMIS},
        )
        maps[name] = read_maps(directory / f"{name}.npz")
    return maps


def reference_of(
    maps: Mapping[str, Mapping[str, np.ndarray]], template: Template, hemi: str, **settings: float
) -> ReferenceMaps:
    half = template.hemis[hemi]
    stack = np.stack([m[hemi] - global_median(m[hemi], half) for m in maps.values()])
    chosen = {"min_vertex_n": 20, "smooth_iterations": 2, "mad_floor_mm": 0.05} | settings
    return build_reference(stack, half, template.adjacency(hemi), **chosen)


def test_the_cohort_maps_follow_expected_thickness(
    synth_cohort: SyntheticCohort,
    cohort_maps: dict[str, dict[str, np.ndarray]],
    tpl5: Template,
) -> None:
    assert len(cohort_maps) == 31 and "sub-009" not in cohort_maps
    spreads = []
    for subject in synth_cohort.subjects:
        name = subject.path.name
        if name not in cohort_maps:
            continue
        patch = synth_cohort.bad.get(name)
        for hemi in HEMIS:
            half = tpl5.hemis[hemi]
            values = cohort_maps[name][hemi]
            has = np.isfinite(values)
            assert has.sum() > 0.97 * half.cortex.sum()
            away = has.copy()
            if patch is not None:
                ring = longest_edge_deg(read_surface(subject.path / "surf" / f"{hemi}.sphere.reg"))
                angle = angle_deg(half.sphere, patch.centre)
                well_inside = has & (angle <= patch.radius_deg - ring)
                assert well_inside.sum() >= 3
                assert (values[well_inside] == 5.0).all()
                away &= np.abs(angle - patch.radius_deg) > ring
            residual = values[away] - expected_thickness(half.sphere[away], patch)
            # the subject's offset (SD 0.15 mm), then noise of SD 0.1 mm per source vertex,
            # which interpolation averages
            assert abs(np.median(residual)) < 0.75
            spreads.append(float(np.std(residual)))
    assert 0.04 < min(spreads) and max(spreads) < 0.1


def test_z_of_the_cohorts_patches(
    synth_cohort: SyntheticCohort,
    cohort_maps: dict[str, dict[str, np.ndarray]],
    tpl5: Template,
) -> None:
    """With noise, a few vertices of every subject exceed the threshold by chance."""
    for hemi in HEMIS:
        half = tpl5.hemis[hemi]
        ref = reference_of(cohort_maps, tpl5, hemi)
        # all 31 at most vertices; fewer next to the medial wall, none on it
        assert np.median(ref.n[half.cortex]) == 31 and ref.n.max() == 31
        assert (ref.n[~half.cortex] == 0).all()
        assert (ref.n[half.cortex] < 31).mean() < 0.05
        chance = []
        for subject in synth_cohort.subjects:
            name = subject.path.name
            if name not in cohort_maps:
                continue
            values = cohort_maps[name][hemi]
            z = zmap(values, global_median(values, half), ref)
            above = np.isfinite(z) & (z > 3.0)
            patch = synth_cohort.bad.get(name)
            if patch is None:
                chance.append(int(above.sum()))
                continue
            ring = longest_edge_deg(read_surface(subject.path / "surf" / f"{hemi}.sphere.reg"))
            angle = angle_deg(half.sphere, patch.centre)
            inside = np.isfinite(z) & (angle <= patch.radius_deg - ring)
            assert inside.sum() >= 3 and above[inside].all()
            assert z[inside].min() > 15
            chance.append(int((above & (angle >= patch.radius_deg + ring)).sum()))
        assert max(chance) < 0.01 * len(half.sphere)


# --- a cohort without noise: the patch and nothing else ---


@pytest.fixture(scope="module")
def quiet_cohort(
    tmp_path_factory: pytest.TempPathFactory, small: SyntheticTemplate, tpl5: Template
) -> tuple[dict[str, dict[str, np.ndarray]], ThickPatch, float]:
    """24 subjects of 2,562 vertices that differ in warp and offset only; ``sub-07`` has a
    5 mm patch of 16 degrees. Returns the maps, the patch and the ring of ``sub-07``."""
    root = tmp_path_factory.mktemp("quiet")
    rng = np.random.default_rng(11)
    patch = cohort_patches()["medial"]
    maps, ring = {}, 0.0
    for position, warp in enumerate(np.linspace(0.05, 0.25, 24)):
        subject = write_synthetic_subject(
            root, name=f"sub-{position:02d}", voxel_size=10.0, shape=(16, 16, 16), subdivisions=4
        )
        add_tier_b(
            subject,
            small,
            warp=float(warp),
            patch=patch if position == 7 else None,
            offset_mm=float(rng.normal(0.0, 0.15)),
        )
        maps[subject.path.name] = {
            hemi: mapped(subject, hemi, tpl5, CONFIG5).thickness for hemi in HEMIS
        }
        if position == 7:
            ring = longest_edge_deg(read_surface(subject.path / "surf" / "lh.sphere.reg"))
    return maps, patch, ring


def test_z_exceeds_the_threshold_in_the_patch_and_nowhere_else(
    quiet_cohort: tuple[dict[str, dict[str, np.ndarray]], ThickPatch, float], tpl5: Template
) -> None:
    maps, patch, ring = quiet_cohort
    assert 4.0 < ring < 8.0
    for hemi in HEMIS:
        half = tpl5.hemis[hemi]
        ref = reference_of(maps, tpl5, hemi)
        angle = angle_deg(half.sphere, patch.centre)
        for name, by_hemi in maps.items():
            z = zmap(by_hemi[hemi], global_median(by_hemi[hemi], half), ref)
            # z where the subject has a value and at least 20 of the 24 have one
            has = np.isfinite(by_hemi[hemi])
            assert np.array_equal(np.isfinite(z), has & (ref.n >= 20))
            assert (has & (ref.n < 20)).sum() < 0.01 * has.sum()
            above = np.isfinite(z) & (z > 3.0)
            if name != "sub-07":
                assert not above.any(), name
                assert np.nanmax(np.abs(z)) < 1.0
                continue
            inside = np.isfinite(z) & (angle <= patch.radius_deg - ring)
            outside = angle >= patch.radius_deg + ring
            assert inside.sum() > 50
            assert above[inside].all() and not above[outside].any()
            # 2.5 mm above the reference, against a floored MAD of 0.05 mm
            assert z[inside].min() > 25
            assert np.nanmax(np.abs(z[outside])) < 1.0


# --- global_median ---


def test_the_global_median_excludes_the_search_region(tpl4: Template) -> None:
    half = tpl4.hemis["lh"]
    rng = np.random.default_rng(3)
    values = rng.uniform(1.5, 4.0, size=len(half.sphere)).astype(np.float32)
    counted = half.cortex & ~half.regions["search"]
    assert counted.sum() > 500 and half.regions["search"].sum() > 500
    expected = float(np.median(values[counted].astype(np.float64)))
    assert global_median(values, half) == expected

    elsewhere = values.copy()
    elsewhere[half.regions["search"]] = 5.0
    elsewhere[~half.cortex] = 0.0
    assert global_median(elsewhere, half) == expected
    assert global_median(values + np.where(counted, 1.0, 0.0).astype(np.float32), half) > expected

    holes = values.copy()
    holes[np.flatnonzero(counted)[::2]] = np.nan
    remaining = counted & np.isfinite(holes)
    assert global_median(holes, half) == float(np.median(holes[remaining].astype(np.float64)))


def test_the_global_median_needs_a_fortieth_of_the_mesh(tpl4: Template) -> None:
    half = tpl4.hemis["lh"]
    needed = len(half.sphere) // 40
    assert needed == 64
    counted = np.flatnonzero(half.cortex & ~half.regions["search"])
    values = np.full(len(half.sphere), np.nan, dtype=np.float32)
    values[half.regions["search"]] = 3.0  # never counted
    values[counted[: needed - 1]] = 2.5
    assert math.isnan(global_median(values, half))
    values[counted[needed - 1]] = 2.5
    assert global_median(values, half) == 2.5
    assert math.isnan(global_median(np.full(len(half.sphere), np.nan, dtype=np.float32), half))
    with pytest.raises(ValueError, match="2561 vertices on a template of 2562"):
        global_median(values[:-1], half)


# --- write_maps, read_maps ---


def test_maps_round_trip_within_the_step_of_float16(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    maps = {hemi: rng.uniform(0.1, 5.0, size=40_962).astype(np.float32) for hemi in HEMIS}
    maps["lh"][::7] = np.nan
    maps["rh"][0] = 5.0
    path = tmp_path / "fsavg" / "0123456789abcdef.npz"
    write_maps(path, maps)
    assert path.stat().st_size == 164_342  # 164 kB per subject on fsaverage6
    stored = read_npz(path)
    assert list(stored) == ["lh_thickness", "rh_thickness"]
    assert all(array.dtype == np.dtype("<f2") for array in stored.values())
    back = read_maps(path)
    assert list(back) == ["lh", "rh"]
    worst = 0.0
    for hemi in HEMIS:
        assert back[hemi].dtype == np.float32
        assert np.array_equal(np.isnan(back[hemi]), np.isnan(maps[hemi]))
        assert np.array_equal(back[hemi], as_float16(maps[hemi]), equal_nan=True)
        worst = max(worst, float(np.nanmax(np.abs(back[hemi] - maps[hemi]))))
    assert 0.0015 < worst <= FLOAT16_STEP_MM
    assert back["rh"][0] == 5.0


def test_map_files_are_byte_identical(tmp_path: Path) -> None:
    rng = np.random.default_rng(6)
    maps = {hemi: rng.uniform(0.1, 5.0, size=2562).astype(np.float32) for hemi in HEMIS}
    maps["lh"][:40] = np.nan
    write_maps(tmp_path / "a.npz", maps)
    digest = sha256_file(tmp_path / "a.npz")

    write_maps(tmp_path / "a.npz", maps)
    assert sha256_file(tmp_path / "a.npz") == digest
    write_maps(tmp_path / "b.npz", {"rh": maps["rh"], "lh": maps["lh"]})
    assert sha256_file(tmp_path / "b.npz") == digest
    # float64 input, and a NaN with its sign bit set as arithmetic can leave it
    other_nan = maps["lh"].astype(np.float64)
    other_nan[:40] = -np.nan
    assert np.signbit(other_nan[0])
    write_maps(tmp_path / "c.npz", {"lh": other_nan, "rh": maps["rh"].astype(np.float64)})
    assert sha256_file(tmp_path / "c.npz") == digest
    # what was read, written again
    write_maps(tmp_path / "d.npz", read_maps(tmp_path / "a.npz"))
    assert sha256_file(tmp_path / "d.npz") == digest
    assert [path.name for path in sorted(tmp_path.iterdir())] == [
        "a.npz",
        "b.npz",
        "c.npz",
        "d.npz",
    ]

    changed = {"lh": maps["lh"].copy(), "rh": maps["rh"]}
    changed["lh"][100] += 0.01
    write_maps(tmp_path / "e.npz", changed)
    assert sha256_file(tmp_path / "e.npz") != digest


def test_one_usable_hemisphere_stores_nan_for_the_other(tmp_path: Path) -> None:
    lh = np.linspace(1.0, 4.0, 2562, dtype=np.float32)
    write_maps(tmp_path / "one.npz", {"lh": lh})
    back = read_maps(tmp_path / "one.npz")
    assert np.array_equal(back["lh"], as_float16(lh))
    assert back["rh"].shape == lh.shape and np.isnan(back["rh"]).all()
    write_maps(tmp_path / "explicit.npz", {"lh": lh, "rh": np.full_like(lh, np.nan)})
    assert sha256_file(tmp_path / "explicit.npz") == sha256_file(tmp_path / "one.npz")

    write_maps(tmp_path / "other.npz", {"rh": lh})
    back = read_maps(tmp_path / "other.npz")
    assert np.isnan(back["lh"]).all() and np.array_equal(back["rh"], as_float16(lh))


def test_maps_that_cannot_be_written(tmp_path: Path) -> None:
    lh = np.ones(2562, dtype=np.float32)
    with pytest.raises(ValueError, match="neither hemisphere"):
        write_maps(tmp_path / "x.npz", {})
    with pytest.raises(ValueError, match="on one mesh"):
        write_maps(tmp_path / "x.npz", {"lh": lh, "rh": lh[:642]})
    assert list(tmp_path.iterdir()) == []

    # a value that float16 cannot hold is missing, not infinite
    huge = lh.copy()
    huge[:3] = (1e6, np.inf, -np.inf)
    write_maps(tmp_path / "huge.npz", {"lh": huge, "rh": lh})
    back = read_maps(tmp_path / "huge.npz")["lh"]
    assert np.isnan(back[:3]).all() and (back[3:] == 1.0).all()


def test_reading_a_file_that_is_not_a_map(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_maps(tmp_path / "absent.npz")

    def refused(name: str, says: str) -> None:
        with pytest.raises(ValueError, match=says) as caught:
            read_maps(tmp_path / name)
        assert str(tmp_path) not in str(caught.value) and name in str(caught.value)

    (tmp_path / "garbage.npz").write_bytes(b"not an archive")
    refused("garbage.npz", "cannot be read as an archive")
    (tmp_path / "empty.npz").write_bytes(b"")
    refused("empty.npz", "cannot be read as an archive")
    lh = np.ones(12, dtype="<f2")
    write_npz(tmp_path / "keys.npz", {"lh_thickness": lh, "rh_z": lh})
    refused("keys.npz", "not the two thickness maps")
    write_npz(tmp_path / "order.npz", {"rh_thickness": lh, "lh_thickness": lh})
    refused("order.npz", "not the two thickness maps")
    write_npz(tmp_path / "one.npz", {"lh_thickness": lh})
    refused("one.npz", "not the two thickness maps")
    write_npz(tmp_path / "type.npz", {"lh_thickness": lh.astype("<f4"), "rh_thickness": lh})
    refused("type.npz", "lh_thickness is not one axis of float16")
    write_npz(tmp_path / "axes.npz", {"lh_thickness": lh, "rh_thickness": lh.reshape(3, 4)})
    refused("axes.npz", "rh_thickness is not one axis of float16")
    write_npz(tmp_path / "length.npz", {"lh_thickness": lh, "rh_thickness": lh[:5]})
    refused("length.npz", "12 and 5 vertices")


# --- build_reference ---


def numpy_reference(
    stack: np.ndarray, cortex: np.ndarray, min_vertex_n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median, MAD and count per vertex, one column at a time."""
    n_vertices = stack.shape[1]
    median, mad = np.full(n_vertices, np.nan), np.full(n_vertices, np.nan)
    n = np.zeros(n_vertices, dtype=np.int64)
    for vertex in np.flatnonzero(cortex):
        column = stack[:, vertex].astype(np.float64)
        column = column[np.isfinite(column)]
        n[vertex] = len(column)
        if len(column) >= max(min_vertex_n, 1):
            median[vertex] = np.median(column)
            mad[vertex] = np.median(np.abs(column - median[vertex]))
    return median, mad, n


def random_stack(half: TemplateHemi, n_subjects: int, seed: int) -> np.ndarray:
    """Values everywhere, the medial wall included, with a third of them missing: NaN for
    most, an infinity of either sign for some."""
    rng = np.random.default_rng(seed)
    stack = rng.normal(0.0, 0.4, size=(n_subjects, len(half.sphere))).astype(np.float32)
    stack[rng.random(stack.shape) < 0.3] = np.nan
    stack[rng.random(stack.shape) < 0.02] = np.inf
    stack[rng.random(stack.shape) < 0.02] = -np.inf
    stack[:, 10] = np.nan
    stack[1:, 11] = np.nan
    return stack


@pytest.mark.parametrize("n_subjects", [1, 2, 11, 12])
@pytest.mark.parametrize("min_vertex_n", [1, 6])
def test_the_reference_is_numpys_median_and_mad(
    tpl4: Template, n_subjects: int, min_vertex_n: int
) -> None:
    half = tpl4.hemis["lh"]
    stack = random_stack(half, n_subjects, seed=n_subjects)
    ref = build_reference(
        stack,
        half,
        tpl4.adjacency("lh"),
        min_vertex_n=min_vertex_n,
        smooth_iterations=0,
        mad_floor_mm=0.0,
    )
    median, mad, n = numpy_reference(stack, half.cortex, min_vertex_n)
    assert (ref.median.dtype, ref.mad.dtype, ref.n.dtype) == (np.float32, np.float32, np.int32)
    assert np.array_equal(ref.n, n)
    assert np.array_equal(ref.median, median.astype(np.float32), equal_nan=True)
    assert np.array_equal(ref.mad, mad.astype(np.float32), equal_nan=True)
    # outside cortex nothing is read, although the stack has values there
    assert np.isfinite(stack[:, ~half.cortex]).any()
    assert (ref.n[~half.cortex] == 0).all() and np.isnan(ref.median[~half.cortex]).all()
    if n_subjects >= 11:
        assert {int(count) % 2 for count in np.unique(ref.n)} == {0, 1}  # both kinds of median
    assert np.array_equal(np.isfinite(ref.median), np.isfinite(ref.mad))
    assert np.array_equal(np.isfinite(ref.median), half.cortex & (n >= min_vertex_n) & (n > 0))


def test_a_vertex_with_too_few_values_has_no_reference(tpl4: Template) -> None:
    half = tpl4.hemis["lh"]
    a, b, c = np.flatnonzero(half.cortex)[[5, 50, 500]]
    stack = np.tile(np.linspace(-1.0, 1.0, 9, dtype=np.float32)[:, None], (1, len(half.sphere)))
    stack[:4, a] = np.nan  # 5 values
    stack[:5, b] = np.nan  # 4 values
    stack[:, c] = np.inf  # none that is finite
    ref = build_reference(
        stack, half, tpl4.adjacency("lh"), min_vertex_n=5, smooth_iterations=0, mad_floor_mm=0.0
    )
    assert (ref.n[a], ref.n[b], ref.n[c]) == (5, 4, 0)
    assert ref.median[a] == np.float32(0.5) and ref.mad[a] == np.float32(0.25)
    assert np.isnan(ref.median[[b, c]]).all() and np.isnan(ref.mad[[b, c]]).all()
    assert ref.median[np.flatnonzero(half.cortex)[6]] == 0.0

    # smoothing fills neither of them, and the floor does not either
    smoothed = build_reference(
        stack, half, tpl4.adjacency("lh"), min_vertex_n=5, smooth_iterations=3, mad_floor_mm=0.3
    )
    assert np.array_equal(np.isfinite(smoothed.mad), np.isfinite(ref.mad))
    assert np.isnan(smoothed.median[[b, c]]).all()


def test_smoothing_leaves_a_constant_map_unchanged(tpl4: Template) -> None:
    half = tpl4.hemis["lh"]
    # every vertex has the same nine values, so one MAD everywhere: 0.1234 * 2
    column = (np.arange(-4, 5) * 0.1234).astype(np.float32)
    stack = np.tile(column[:, None], (1, len(half.sphere)))
    without = np.flatnonzero(half.cortex)[::9]
    stack[:5, without] = np.nan
    settings = {"min_vertex_n": 5, "mad_floor_mm": 0.0}
    plain_ref = build_reference(stack, half, tpl4.adjacency("lh"), smooth_iterations=0, **settings)
    constant = plain_ref.mad[np.isfinite(plain_ref.mad)]
    assert constant.size > 2000 and (constant == constant[0]).all() and constant[0] > 0.2
    for iterations in (1, 2, 5):
        smoothed = build_reference(
            stack, half, tpl4.adjacency("lh"), smooth_iterations=iterations, **settings
        )
        assert np.array_equal(smoothed.mad, plain_ref.mad, equal_nan=True)
        assert np.array_equal(smoothed.median, plain_ref.median, equal_nan=True)
        assert np.array_equal(smoothed.n, plain_ref.n)


def neighbour_mean(mad: np.ndarray, adjacency: sparse.csr_matrix) -> np.ndarray:
    """One smoothing step, vertex by vertex."""
    out = np.full(len(mad), np.nan)
    for vertex in np.flatnonzero(np.isfinite(mad)):
        around = adjacency.indices[adjacency.indptr[vertex] : adjacency.indptr[vertex + 1]]
        values = np.concatenate([[mad[vertex]], mad[around]])
        out[vertex] = values[np.isfinite(values)].mean()
    return out


@pytest.mark.parametrize("iterations", [1, 2])
def test_smoothing_is_the_mean_over_a_vertex_and_its_neighbours(
    tpl4: Template, iterations: int
) -> None:
    half = tpl4.hemis["lh"]
    adjacency = tpl4.adjacency("lh")
    stack = random_stack(half, 12, seed=40)
    lonely = int(np.flatnonzero(half.cortex)[300])
    around = adjacency.indices[adjacency.indptr[lonely] : adjacency.indptr[lonely + 1]]
    stack[:, around] = np.nan  # a vertex none of whose neighbours has a MAD
    settings = {"min_vertex_n": 6, "mad_floor_mm": 0.0}
    raw = build_reference(stack, half, adjacency, smooth_iterations=0, **settings)
    smoothed = build_reference(stack, half, adjacency, smooth_iterations=iterations, **settings)

    _, mad, _ = numpy_reference(stack, half.cortex, 6)
    expected = mad
    for _ in range(iterations):
        expected = neighbour_mean(expected, adjacency)
    has = np.isfinite(expected)
    assert np.array_equal(np.isfinite(smoothed.mad), has)
    assert np.allclose(smoothed.mad[has], expected[has], rtol=1e-6, atol=0.0)
    assert np.isfinite(raw.mad[lonely]) and smoothed.mad[lonely] == raw.mad[lonely]
    # the medians are not smoothed
    assert np.array_equal(smoothed.median, raw.median, equal_nan=True)
    # a smoother map: less spread between vertices
    assert np.nanstd(smoothed.mad) < 0.7 * np.nanstd(raw.mad)


def test_the_floor_is_applied_after_smoothing(tpl4: Template) -> None:
    half = tpl4.hemis["lh"]
    adjacency = tpl4.adjacency("lh")
    stack = random_stack(half, 12, seed=41)
    settings = {"min_vertex_n": 6, "smooth_iterations": 1}
    free = build_reference(stack, half, adjacency, mad_floor_mm=0.0, **settings)
    floor = float(np.nanmedian(free.mad))
    floored = build_reference(stack, half, adjacency, mad_floor_mm=floor, **settings)
    has = np.isfinite(free.mad)
    assert np.array_equal(np.isfinite(floored.mad), has)
    assert np.array_equal(floored.mad[has], np.maximum(free.mad[has], np.float32(floor)))
    assert 0.3 < (floored.mad[has] == np.float32(floor)).mean() < 0.7

    # a spike of 1 mm among zeros: its own vertex and its neighbours share it, the rest is 0
    spike = int(np.flatnonzero(half.cortex)[700])
    around = adjacency.indices[adjacency.indptr[spike] : adjacency.indptr[spike + 1]]
    assert half.cortex[around].all() and len(around) == 6
    column = np.array([0.0] * 9, dtype=np.float32)
    stack = np.tile(column[:, None], (1, len(half.sphere)))
    stack[:, spike] = np.array([-3, -2, -1, -1, 0, 1, 1, 2, 3], dtype=np.float32)
    spiked = build_reference(
        stack, half, adjacency, min_vertex_n=5, smooth_iterations=1, mad_floor_mm=0.05
    )
    assert spiked.mad[spike] == pytest.approx(1 / 7)
    assert spiked.mad[around] == pytest.approx(1 / 7)
    rest = half.cortex.copy()
    rest[[spike, *around]] = False
    assert (spiked.mad[rest] == np.float32(0.05)).all()
    assert np.isnan(spiked.mad[~half.cortex]).all()


def test_a_reference_of_no_subjects_and_inputs_of_another_mesh(tpl4: Template) -> None:
    half = tpl4.hemis["lh"]
    adjacency = tpl4.adjacency("lh")
    settings = {"min_vertex_n": 1, "smooth_iterations": 2, "mad_floor_mm": 0.05}
    empty = build_reference(
        np.empty((0, len(half.sphere)), dtype=np.float32), half, adjacency, **settings
    )
    assert np.isnan(empty.median).all() and np.isnan(empty.mad).all() and not empty.n.any()
    with pytest.raises(ValueError, match="642 vertices on a template of 2562"):
        build_reference(np.zeros((3, 642), dtype=np.float32), half, adjacency, **settings)
    with pytest.raises(ValueError, match="adjacency of shape"):
        build_reference(
            np.zeros((3, len(half.sphere)), dtype=np.float32),
            half,
            sparse.csr_matrix((642, 642)),
            **settings,
        )


def test_a_reference_does_not_depend_on_the_block_size(
    tpl4: Template, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    half = tpl4.hemis["lh"]
    stack = random_stack(half, 12, seed=42)
    settings = {"min_vertex_n": 6, "smooth_iterations": 2, "mad_floor_mm": 0.05}
    whole = build_reference(stack, half, tpl4.adjacency("lh"), **settings)
    monkeypatch.setattr(fa, "_BLOCK_VERTICES", 97)
    blocks = build_reference(stack, half, tpl4.adjacency("lh"), **settings)
    # nor on where the stack lies: a file mapped into memory, as a census may need
    on_disk = np.memmap(tmp_path / "stack.f4", dtype=np.float32, mode="w+", shape=stack.shape)
    on_disk[:] = stack
    mapped_stack = build_reference(on_disk, half, tpl4.adjacency("lh"), **settings)
    for other in (blocks, mapped_stack):
        for field in dataclasses.fields(ReferenceMaps):
            assert np.array_equal(
                getattr(other, field.name), getattr(whole, field.name), equal_nan=True
            )
    assert np.array_equal(on_disk, stack, equal_nan=True)  # the stack itself is not changed


def test_a_built_reference_is_one_the_reference_files_take(
    cohort_maps: dict[str, dict[str, np.ndarray]], tpl5: Template, tmp_path: Path
) -> None:
    maps = {hemi: reference_of(cohort_maps, tpl5, hemi) for hemi in HEMIS}
    uids = tuple(sorted(cohort_maps))
    digests = {"all": members_sha256({uid: "0" * 64 for uid in uids})}
    ref = Reference(
        reference_id=set_reference_id("0123456789ab", digests),
        kind="set",
        mesh=tpl5.mesh,
        template_id=tpl5.template_id,
        sphere_sha256={hemi: tpl5.files[f"surf/{hemi}.sphere.reg"][1] for hemi in HEMIS},
        config_hash="0123456789ab",
        strata={"all": maps},
        members={"all": uids},
        metric_norms={},
        parcel_norms={},
    )
    write_reference(tmp_path / "ref", ref, members_sha256=digests)
    back = read_reference(tmp_path / "ref")
    for hemi in HEMIS:
        for field in dataclasses.fields(ReferenceMaps):
            assert np.array_equal(
                getattr(back.strata["all"][hemi], field.name),
                getattr(maps[hemi], field.name),
                equal_nan=True,
            )


# --- zmap ---


def test_z_is_the_residual_over_the_scaled_mad() -> None:
    rng = np.random.default_rng(9)
    n = 500
    thickness = rng.uniform(1.5, 5.0, size=n).astype(np.float32)
    ref = ReferenceMaps(
        median=rng.normal(0.0, 0.3, size=n).astype(np.float32),
        mad=rng.uniform(0.05, 0.4, size=n).astype(np.float32),
        n=np.full(n, 30, dtype=np.int32),
    )
    thickness[:10] = np.nan
    ref.median[5:20] = np.nan
    ref.mad[15:30] = np.nan
    z = zmap(thickness, 2.47, ref)
    assert z.dtype == np.float32 and z.shape == (n,)
    expected = (thickness.astype(np.float64) - 2.47 - ref.median) / (
        1.4826 * ref.mad.astype(np.float64)
    )
    assert np.isnan(z[:30]).all() and np.isfinite(z[30:]).all()
    assert np.array_equal(z, expected.astype(np.float32), equal_nan=True)
    assert np.abs(z[30:]).max() > 5

    # one MAD above the median is 1 / 1.4826
    one = ReferenceMaps(
        median=np.array([0.25], dtype=np.float32),
        mad=np.array([0.5], dtype=np.float32),
        n=np.array([30], dtype=np.int32),
    )
    assert zmap(np.array([3.25], dtype=np.float32), 2.5, one)[0] == np.float32(0.5 / 0.7413)
    assert zmap(np.array([2.25], dtype=np.float32), 2.5, one)[0] == np.float32(-0.5 / 0.7413)


def test_z_is_nan_where_a_term_is_missing_or_the_scale_is_zero() -> None:
    thickness = np.array([3.0, 3.0, 3.0, 3.0], dtype=np.float32)
    ref = ReferenceMaps(
        median=np.zeros(4, dtype=np.float32),
        mad=np.array([0.1, 0.0, 0.1, 0.1], dtype=np.float32),
        n=np.full(4, 30, dtype=np.int32),
    )
    z = zmap(thickness, 2.5, ref)
    assert np.isfinite(z[[0, 2, 3]]).all() and np.isnan(z[1])  # not infinite
    assert np.isnan(zmap(thickness, float("nan"), ref)).all()
    with pytest.raises(ValueError, match="a map of 3 vertices and a reference of 4"):
        zmap(thickness[:3], 2.5, ref)


# --- strata ---


def frame(rows: list[tuple[str, str, float | None, int | None]]) -> pd.DataFrame:
    """Subjects as ``PublicCatalog.subjects()`` types them: nullable numbers."""
    return pd.DataFrame(
        {
            "subject_uid": pd.Series([row[0] for row in rows], dtype=object),
            "dataset": pd.Series([row[1] for row in rows], dtype=object),
            "field_strength": pd.Series([row[2] for row in rows], dtype="Float64"),
            "fs_major": pd.Series([row[3] for row in rows], dtype="Int64"),
        }
    )


def dataset_rows(
    pseudonym: str, n: int, field_strength: float | None, fs_major: int | None
) -> list[tuple[str, str, float | None, int | None]]:
    return [(f"{pseudonym[2:]}{k:04d}", pseudonym, field_strength, fs_major) for k in range(n)]


def test_a_small_dataset_is_scored_against_its_field_and_version() -> None:
    """The example of the contract: three datasets of 40, 40 and 10 subjects at 3 T."""
    rows = (
        dataset_rows("D-aaaaaaaa", 40, 3.0, 6)
        + dataset_rows("D-bbbbbbbb", 40, 2.89, 6)
        + dataset_rows("D-cccccccc", 10, 3.0, 6)
    )
    subjects = frame(rows).sample(frac=1.0, random_state=1).reset_index(drop=True)
    strata = assign_strata(subjects, min_n=30)
    assert strata.index.name == "subject_uid" and strata.name == "stratum"
    assert list(strata.index) == list(subjects["subject_uid"])
    by_dataset = dict(zip(subjects["dataset"], strata, strict=True))
    assert by_dataset == {
        "D-aaaaaaaa": "dataset:D-aaaaaaaa",
        "D-bbbbbbbb": "dataset:D-bbbbbbbb",
        "D-cccccccc": "field:3|fs:6",
    }
    assert strata.value_counts().to_dict() == {
        "dataset:D-aaaaaaaa": 40,
        "dataset:D-bbbbbbbb": 40,
        "field:3|fs:6": 10,
    }
    # the reference of a stratum is built from everyone who matches it, assigned or not
    members = stratum_members(subjects, "field:3|fs:6")
    assert len(members) == 90 and members == sorted(row[0] for row in rows)
    assert stratum_members(subjects, "all") == members
    assert stratum_members(subjects, "fs:6") == members
    assert stratum_members(subjects, "dataset:D-cccccccc") == [
        f"cccccccc{k:04d}" for k in range(10)
    ]
    assert stratum_members(subjects, "dataset:D-dddddddd") == []
    assert stratum_members(subjects, "field:7|fs:6") == []


def test_strata_fall_back_in_the_documented_order() -> None:
    rows = (
        dataset_rows("D-aaaaaaaa", 5, 3.0, 7)  # field and version: 5 + 3 = 8
        + dataset_rows("D-bbbbbbbb", 3, 3.1, 7)
        + dataset_rows("D-cccccccc", 4, 7.0, 7)  # alone in its field: all
        + dataset_rows("D-dddddddd", 2, None, 7)  # field unknown: all
        + dataset_rows("D-eeeeeeee", 2, 3.0, None)  # version unknown: all
        + dataset_rows("D-ffffffff", 6, 1.5, 6)  # its own dataset
    )
    subjects = frame(rows)

    def assigned(min_n: int) -> dict[str, object]:
        strata = assign_strata(subjects, min_n=min_n)
        return dict(zip(subjects["dataset"], strata, strict=True))

    assert assigned(6) == {
        "D-aaaaaaaa": "field:3|fs:7",
        "D-bbbbbbbb": "field:3|fs:7",
        "D-cccccccc": "all",
        "D-dddddddd": "all",
        "D-eeeeeeee": "all",
        "D-ffffffff": "dataset:D-ffffffff",
    }
    # the dataset comes first where it exists, whatever else exists
    assert assigned(5)["D-aaaaaaaa"] == "dataset:D-aaaaaaaa"
    assert assigned(5)["D-bbbbbbbb"] == "field:3|fs:7"
    # one subject is a stratum when one is enough
    assert set(assigned(1).values()) == {f"dataset:{row[1]}" for row in rows}
    # 22 subjects in all: the count needed is met exactly, then missed
    assert set(assigned(22).values()) == {"all"}
    assert set(assigned(23).values()) == {None}
    assert assign_strata(subjects, min_n=23).isna().all()

    assert stratum_members(subjects, "field:3|fs:7") == sorted(
        row[0] for row in rows if row[1] in ("D-aaaaaaaa", "D-bbbbbbbb")
    )
    assert len(stratum_members(subjects, "fs:7")) == 14
    assert len(stratum_members(subjects, "all")) == 22
    assert stratum_members(subjects, "field:1.5|fs:6") == [f"ffffffff{k:04d}" for k in range(6)]


def test_strata_read_missing_values_in_any_form() -> None:
    plain_frame = pd.DataFrame(
        {
            "subject_uid": ["a", "b", "c", "d"],
            "dataset": ["D-1", "D-1", "D-1", "D-1"],
            "field_strength": [3.0, np.nan, None, 2.9],
            "fs_major": [6.0, 6.0, 6.0, np.nan],  # a float column, as pandas makes of None
        }
    )
    assert assign_strata(plain_frame, min_n=4).tolist() == ["dataset:D-1"] * 4
    assert assign_strata(plain_frame, min_n=5).tolist() == [None] * 4
    assert stratum_members(plain_frame, "field:3|fs:6") == ["a"]
    assert stratum_members(plain_frame, "fs:6") == ["a", "b", "c"]

    empty = frame([])
    assert assign_strata(empty, min_n=1).empty
    assert stratum_members(empty, "all") == []


def test_strata_refuse_what_they_cannot_read() -> None:
    subjects = frame(dataset_rows("D-aaaaaaaa", 3, 3.0, 6))
    for name in ("", "ALL", "D-aaaaaaaa", "set-0123456789ab", "none"):
        with pytest.raises(ValueError, match="not the name of a stratum"):
            stratum_members(subjects, name)
    with pytest.raises(ValueError, match="lack the columns field_strength, fs_major"):
        assign_strata(subjects[["subject_uid", "dataset"]], min_n=1)
    twice = pd.concat([subjects, subjects.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="listed twice"):
        assign_strata(twice, min_n=1)
    with pytest.raises(ValueError, match="listed twice"):
        stratum_members(twice, "all")


# --- the real subject ---

REAL_TEMPLATE_FILES = (
    "surf/lh.sphere.reg",
    "surf/rh.sphere.reg",
    "surf/lh.white",
    "surf/rh.white",
    "label/lh.cortex.label",
    "label/rh.cortex.label",
    "label/lh.aparc.annot",
    "label/rh.aparc.annot",
    "surf/lh.white.avg.area.mgh",
    "surf/rh.white.avg.area.mgh",
)
REAL_TIER_B = tuple(
    name.format(h=hemi)
    for hemi in HEMIS
    for name in ("surf/{h}.thickness", "surf/{h}.sphere.reg", "label/{h}.aparc.annot")
)
REAL_MAP = {  # ds000102 sub-01 on fsaverage6, measured 2026-11-13
    "lh": {
        "n_fallback": 3,
        "n_value": 37_420,
        "n_cortex": 37_487,
        "mean": 2.4251,
        "median": 2.4609,
    },
    "rh": {
        "n_fallback": 2,
        "n_value": 37_455,
        "n_cortex": 37_482,
        "mean": 2.4211,
        "median": 2.4551,
    },
}


def check_real_map(subject: Path, fsaverage: Path, scratch: Path) -> None:
    template = build_template(fsaverage, "adjacent")
    config = MetricsConfig()
    maps = {}
    for hemi in HEMIS:
        half = template.hemis[hemi]
        thickness = read_morph(subject / "surf" / f"{hemi}.thickness")
        sphere = read_surface(subject / "surf" / f"{hemi}.sphere.reg")
        annot = read_annotation(subject / "label" / f"{hemi}.aparc.annot")
        hemi_map, smap = resample_hemi(thickness, sphere, half, annot=annot, config=config)
        values = hemi_map.thickness
        has = np.isfinite(values)
        measured = REAL_MAP[hemi]
        assert int(half.cortex.sum()) == measured["n_cortex"]
        assert hemi_map.n_fallback == measured["n_fallback"]
        assert int(has.sum()) == measured["n_value"]
        assert hemi_map.n_missing_cortex == measured["n_cortex"] - measured["n_value"]
        assert has.sum() >= 0.995 * half.cortex.sum()
        assert hemi_map.flipped is False
        assert float(values[has].mean(dtype=np.float64)) == pytest.approx(
            measured["mean"], abs=5e-4
        )
        assert hemi_map.global_median == pytest.approx(measured["median"], abs=5e-4)
        assert 0.1 < values[has].min() and values[has].max() <= 5.0

        valid = np.isfinite(thickness) & (thickness > 0) & annot.cortex()
        unrounded = registration.resample(thickness, smap, valid=valid, min_valid_weight=0.5)
        assert np.abs(unrounded[has] - values[has]).max() <= FLOAT16_STEP_MM

        # without the annotation the medial wall's zeros still do not count, and the template
        # cortex is covered
        bare, _ = resample_hemi(thickness, sphere, half, annot=None, config=config)
        assert bare.n_missing_cortex <= 5
        assert bare.global_median == pytest.approx(hemi_map.global_median, abs=0.01)
        maps[hemi] = values
    write_maps(scratch / "real.npz", maps)
    assert (scratch / "real.npz").stat().st_size == 164_342
    back = read_maps(scratch / "real.npz")
    assert all(np.array_equal(back[hemi], maps[hemi], equal_nan=True) for hemi in HEMIS)


def test_cached_real_subject_map(cached_subjects: Path, tmp_path: Path) -> None:
    subject, fsaverage = cached_subjects / "ds000102-sub-01", cached_subjects / "ds000102-fsaverage"
    wanted = [subject / name for name in REAL_TIER_B]
    wanted += [fsaverage / name for name in REAL_TEMPLATE_FILES]
    if not all(path.is_file() for path in wanted):
        pytest.skip("the real subject's Tier B files or its fsaverage are not in the pytest cache")
    # the files as they were listed: a sphere of 148,332 vertices on lh
    assert fs.read_geometry(str(subject / "surf" / "lh.sphere.reg"))[0].shape == (148_332, 3)
    check_real_map(subject, fsaverage, tmp_path)
