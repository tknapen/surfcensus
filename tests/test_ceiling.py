"""metrics/ceiling: counts at the thickness ceiling and at zero, and the native regions they
are counted in.

The counts are tested on arrays written here. The regions are tested twice: on hand-made
maps and annotations of a few vertices, and on a synthetic subject registered to a small
synthetic template, where the expected mask follows from the closed-form rules of
``tests/synth_template.py`` and ``tests/synth_tierb.py`` and not from the package. The tests
at the end read the real subject, and the real fsaverage, where the network tests have put
them in the pytest cache, and are skipped elsewhere.
"""

from __future__ import annotations

import dataclasses
import inspect
import logging
import math
import typing
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest
from jaxtyping import TypeCheckError
from synth import SyntheticSubject, write_synthetic_subject  # pyrefly: ignore[missing-import]
from synth_template import (  # pyrefly: ignore[missing-import]
    DESIKAN_NAMES,
    MEDIAL_WALL_M,
    SyntheticTemplate,
    direction,
    expected_regions,
    medial_coordinate,
    parcel_of,
    template_mesh,
    write_synthetic_template,
)
from synth_tierb import (  # pyrefly: ignore[missing-import]
    ThickPatch,
    add_tier_b,
    in_patch,
    subject_directions,
    warp_directions,
)

from surfcensus import schemas
from surfcensus.config import RegionConfig
from surfcensus.fsio.annot import Annotation, read_annotation
from surfcensus.fsio.morph import read_morph
from surfcensus.fsio.registration import SphereMap, build_sphere_map
from surfcensus.fsio.surfaces import read_surface
from surfcensus.metrics import MetricError
from surfcensus.metrics.ceiling import (
    ABOVE_MAX_TOLERANCE_MM,
    CeilingCounts,
    RegionKind,
    ceiling_counts,
    native_region,
    thickness_above_max,
)
from surfcensus.sources.base import HEMIS, Hemi
from surfcensus.sources.patterns import TEMPLATE_FILES
from surfcensus.templates import REGION_NAMES, RegionName, Template, TemplateHemi, build_template

REGIONS = RegionConfig()


def f32(*values: float) -> np.ndarray:
    return np.array(values, dtype=np.float32)


def everywhere(n: int) -> np.ndarray:
    return np.ones(n, dtype=bool)


# ---------------------------------------------------------------------------------------
# CeilingCounts


def test_the_fraction_is_near_plus_zero_over_the_region() -> None:
    counts = CeilingCounts(n_region=200, n_exact=3, n_near=7, n_zero=5)
    assert counts.fraction == 12 / 200
    # the exact count is part of the near count and is not added again
    assert CeilingCounts(n_region=10, n_exact=4, n_near=4, n_zero=0).fraction == 0.4
    assert CeilingCounts(n_region=10, n_exact=0, n_near=0, n_zero=0).fraction == 0.0
    assert isinstance(counts.fraction, float)


def test_an_empty_region_has_no_fraction() -> None:
    assert math.isnan(CeilingCounts(n_region=0, n_exact=0, n_near=0, n_zero=0).fraction)


def test_counts_are_frozen_and_hold_plain_integers() -> None:
    counts = ceiling_counts(f32(4.89, 4.9, 5.0, 0.0), everywhere(4))
    assert all(type(value) is int for value in dataclasses.astuple(counts))
    with pytest.raises(dataclasses.FrozenInstanceError):
        counts.n_zero = 2  # pyrefly: ignore[read-only]
    assert [field.name for field in dataclasses.fields(CeilingCounts)] == [
        "n_region",
        "n_exact",
        "n_near",
        "n_zero",
    ]


# ---------------------------------------------------------------------------------------
# ceiling_counts


def test_counts_at_489_49_50_and_0() -> None:
    counts = ceiling_counts(f32(4.89, 4.9, 5.0, 0.0), everywhere(4))
    assert counts == CeilingCounts(n_region=4, n_exact=1, n_near=2, n_zero=1)
    assert counts.fraction == 0.75
    # one at a time: which value is counted as what
    assert ceiling_counts(f32(4.89), everywhere(1)) == CeilingCounts(1, 0, 0, 0)
    assert ceiling_counts(f32(4.9), everywhere(1)) == CeilingCounts(1, 0, 1, 0)
    assert ceiling_counts(f32(5.0), everywhere(1)) == CeilingCounts(1, 1, 1, 0)
    assert ceiling_counts(f32(0.0), everywhere(1)) == CeilingCounts(1, 0, 0, 1)
    assert ceiling_counts(f32(2.5), everywhere(1)) == CeilingCounts(1, 0, 0, 0)


def test_only_vertices_of_the_region_are_counted() -> None:
    thickness = f32(5.0, 5.0, 4.95, 0.0, 0.0, 2.5, 4.95, 2.0)
    region = np.array([True, False, True, True, False, True, False, False])
    assert ceiling_counts(thickness, region) == CeilingCounts(4, 1, 2, 1)
    assert ceiling_counts(thickness, ~region) == CeilingCounts(4, 1, 2, 1)
    assert ceiling_counts(thickness, everywhere(8)) == CeilingCounts(8, 2, 4, 2)


def test_an_empty_region_gives_zero_counts_and_nan() -> None:
    counts = ceiling_counts(f32(5.0, 0.0, 4.95), np.zeros(3, dtype=bool))
    assert counts == CeilingCounts(0, 0, 0, 0)
    assert math.isnan(counts.fraction)
    none = ceiling_counts(np.empty(0, dtype=np.float32), np.empty(0, dtype=bool))
    assert none == CeilingCounts(0, 0, 0, 0) and math.isnan(none.fraction)


def test_the_limits_are_the_nearest_values_of_the_file_type() -> None:
    # float32(4.9) lies above the float64 4.9 and float32(4.7) below the float64 4.7: a value
    # that the file stores for the limit counts either way
    assert float(np.float32(4.9)) > 4.9 and float(np.float32(4.7)) < 4.7
    below_49, below_47 = (np.nextafter(np.float32(x), np.float32(0)) for x in (4.9, 4.7))
    below_5 = np.nextafter(np.float32(5.0), np.float32(0))
    thickness = np.array([4.9, below_49, 4.7, below_47, 5.0, below_5], dtype=np.float32)
    assert ceiling_counts(thickness, everywhere(6)) == CeilingCounts(6, 1, 3, 0)
    wide = ceiling_counts(thickness, everywhere(6), margin_mm=0.3)
    assert wide == CeilingCounts(6, 1, 5, 0)
    # numpy scalars for the limits (a config read from elsewhere) do not move them
    assert (
        ceiling_counts(thickness, everywhere(6), max_mm=np.float64(5.0), margin_mm=np.float64(0.3))
        == wide
    )
    # and one representable step above the maximum is still at the maximum
    assert ceiling_counts(
        np.array([np.nextafter(np.float32(5.0), np.float32(9))]), everywhere(1)
    ) == CeilingCounts(1, 1, 1, 0)


def test_float64_thickness_is_compared_as_float64() -> None:
    thickness = np.array([4.9, np.nextafter(4.9, 0.0), 5.0, 0.0, 4.7])
    assert ceiling_counts(thickness, everywhere(5)) == CeilingCounts(5, 1, 2, 1)
    # between the float32 4.7 and the float64 4.7: below the limit as float64, and the
    # value the limit itself has as float32
    between = np.array([4.6999999])
    assert float(np.float32(4.7)) < between[0] < 4.7
    assert ceiling_counts(between, everywhere(1), margin_mm=0.3).n_near == 0
    assert ceiling_counts(between.astype(np.float32), everywhere(1), margin_mm=0.3).n_near == 1
    assert ceiling_counts(np.array([4.7]), everywhere(1), margin_mm=0.3).n_near == 1


def test_margin_and_maximum_are_arguments() -> None:
    thickness = f32(3.5, 3.89, 3.9, 4.0, 4.5, 4.9, 5.0, 0.0)
    assert ceiling_counts(thickness, everywhere(8), max_mm=4.0) == CeilingCounts(8, 4, 5, 1)
    assert ceiling_counts(thickness, everywhere(8), max_mm=4.0, margin_mm=0.5) == CeilingCounts(
        8, 4, 7, 1
    )
    # without a margin the two counts are one
    assert ceiling_counts(thickness, everywhere(8), margin_mm=0.0) == CeilingCounts(8, 1, 1, 1)
    assert ceiling_counts(thickness, everywhere(8), max_mm=5, margin_mm=1) == CeilingCounts(
        8, 1, 4, 1
    )


def test_near_includes_exact_and_zero_is_apart() -> None:
    rng = np.random.default_rng(17)
    thickness = rng.uniform(0.0, 5.2, size=5000).astype(np.float32)
    thickness[rng.random(5000) < 0.05] = 0.0
    thickness[rng.random(5000) < 0.05] = 5.0
    region = rng.random(5000) < 0.6
    counts = ceiling_counts(thickness, region)
    inside = thickness[region].astype(np.float64)
    assert counts.n_region == int(region.sum())
    assert counts.n_exact == int((inside >= 5.0).sum()) > 0
    # no value of this sample lies between float32(4.9) and 4.9
    assert counts.n_near == int((inside >= 4.9).sum()) > counts.n_exact
    assert counts.n_zero == int((inside == 0.0).sum()) > 0
    assert counts.fraction == (counts.n_near + counts.n_zero) / counts.n_region


def test_nan_is_a_vertex_of_the_region_and_nothing_else() -> None:
    thickness = f32(np.nan, np.nan, 5.0, 0.0, np.inf, -0.0)
    counts = ceiling_counts(thickness, everywhere(6))
    assert counts == CeilingCounts(n_region=6, n_exact=2, n_near=2, n_zero=2)
    assert counts.fraction == 4 / 6


def test_arrays_of_different_length_are_refused() -> None:
    thickness, region = f32(1.0, 2.0, 3.0), everywhere(4)
    with pytest.raises((ValueError, TypeCheckError)):
        ceiling_counts(thickness, region)
    # the function's own check, which is all there is without the type-checking import hook
    with pytest.raises(ValueError, match="3 thickness values for a region of 4 vertices"):
        inspect.unwrap(ceiling_counts)(thickness, region)


@pytest.mark.parametrize(
    ("max_mm", "margin_mm"), [(5.0, -0.1), (5.0, 5.0), (5.0, 6.0), (0.0, 0.0), (-1.0, 0.0)]
)
def test_a_margin_outside_zero_to_the_maximum_is_refused(max_mm: float, margin_mm: float) -> None:
    with pytest.raises(ValueError, match="margin must be at least 0 and below the maximum"):
        ceiling_counts(f32(0.0, 5.0), everywhere(2), max_mm=max_mm, margin_mm=margin_mm)


def test_a_margin_just_below_the_maximum_counts_zero_once() -> None:
    # were the margin the maximum, a vertex at 0 would be near the ceiling and at zero
    counts = ceiling_counts(f32(0.0, 0.2, 5.0), everywhere(3), margin_mm=4.9)
    assert counts == CeilingCounts(3, 1, 2, 1)
    assert counts.fraction == 1.0


def test_the_input_is_not_changed() -> None:
    thickness, region = f32(4.95, 0.0, 5.0, 2.0), np.array([True, True, False, True])
    before = thickness.copy(), region.copy()
    thickness.setflags(write=False)
    region.setflags(write=False)
    ceiling_counts(thickness, region)
    assert np.array_equal(thickness, before[0]) and np.array_equal(region, before[1])


# ---------------------------------------------------------------------------------------
# thickness_above_max


def test_the_above_maximum_flag() -> None:
    assert ABOVE_MAX_TOLERANCE_MM == 0.01
    assert not thickness_above_max(f32(0.0, 2.5, 5.0))
    assert not thickness_above_max(f32(2.5, 5.005))
    # the value the file stores for 5.01 lies above the float64 5.01 and does not exceed it
    assert float(np.float32(5.01)) > 5.01
    assert not thickness_above_max(f32(2.5, 5.01))
    assert thickness_above_max(np.array([np.nextafter(np.float32(5.01), np.float32(9))]))
    assert thickness_above_max(f32(2.5, 5.02, 0.0))
    assert thickness_above_max(f32(6.2))
    assert thickness_above_max(np.array([5.011])) and not thickness_above_max(np.array([5.01]))


def test_the_flag_follows_the_maximum_that_is_given() -> None:
    thickness = f32(1.0, 4.5)
    assert thickness_above_max(thickness, max_mm=4.0)
    assert not thickness_above_max(thickness, max_mm=4.5)
    assert not thickness_above_max(thickness)
    assert thickness_above_max(f32(5.5)) and not thickness_above_max(f32(5.5), max_mm=6.0)


def test_the_flag_ignores_nan_and_needs_a_number() -> None:
    assert not thickness_above_max(f32(np.nan, 3.0))
    assert thickness_above_max(f32(np.nan, 7.0, np.nan))
    assert thickness_above_max(f32(1.0, np.inf))
    assert not thickness_above_max(f32(np.nan, np.nan))
    assert not thickness_above_max(np.empty(0, dtype=np.float32))
    assert type(thickness_above_max(f32(6.0))) is bool


def test_counts_are_still_reported_above_the_maximum() -> None:
    # recon-all ran with a maximum of 7: the counts at 5 are what they are, and the flag says
    # that they do not mark truncated searches
    thickness = f32(2.0, 4.95, 5.0, 6.2, 7.0, 0.0)
    assert thickness_above_max(thickness)
    assert ceiling_counts(thickness, everywhere(6)) == CeilingCounts(6, 3, 4, 1)
    assert ceiling_counts(thickness, everywhere(6), max_mm=7.0) == CeilingCounts(6, 1, 1, 1)
    assert not thickness_above_max(thickness, max_mm=7.0)


# ---------------------------------------------------------------------------------------
# native_region on hand-made maps


def tiny_hemi(regions: Mapping[str, Sequence[bool]]) -> TemplateHemi:
    """A template hemisphere of which only the region masks mean anything."""
    n = len(next(iter(regions.values())))
    sphere = np.zeros((n, 3))
    sphere[:, 2] = 1.0
    return TemplateHemi(
        sphere=sphere,
        faces=np.zeros((1, 3), dtype=np.int32),
        white=np.zeros((n, 3), dtype=np.float32),
        cortex=np.ones(n, dtype=bool),
        parcel=np.zeros(n, dtype=np.int16),
        area=np.ones(n, dtype=np.float32),
        regions={
            typing.cast(RegionName, name): np.array(mask, dtype=bool)
            for name, mask in regions.items()
        },
        pole_vertex=0,
    )


def tiny_map(nearest_target: Sequence[int], n_target: int) -> SphereMap:
    """A sphere map of which only the nearest target of each source vertex means anything."""
    weights = np.zeros((n_target, 3))
    weights[:, 0] = 1.0
    return SphereMap(
        n_source=len(nearest_target),
        n_target=n_target,
        corners=np.zeros((n_target, 3), dtype=np.int32),
        weights=weights,
        fallback=np.zeros(n_target, dtype=bool),
        nearest_source=np.zeros(n_target, dtype=np.int32),
        nearest_target=np.array(nearest_target, dtype=np.int32),
        flipped=False,
    )


def annotation(*parcels: str | None, names: Sequence[str] = DESIKAN_NAMES) -> Annotation:
    """One vertex per argument, labelled with that parcel; None for a vertex without a label."""
    labels = [-1 if parcel is None else names.index(parcel) for parcel in parcels]
    return Annotation(labels=np.array(labels, dtype=np.int32), names=tuple(names))


FIVE_REGIONS: dict[str, list[bool]] = {
    "search": [True, True, True, False, False],
    "medial": [True, False, False, False, False],
    "inferior": [False, True, False, False, False],
    "lateral": [False, False, True, False, False],
    "control": [False, False, False, True, False],
}
NEAREST = [4, 0, 0, 1, 2, 3, 3, 2]
"""Template vertex nearest each of eight native vertices."""


def test_region_kinds_are_those_of_the_tables() -> None:
    assert typing.get_args(RegionKind) == schemas.REGION_KINDS == ("template", "annot", "none")


def test_template_kind_takes_the_region_of_the_nearest_template_vertex() -> None:
    hemi, smap = tiny_hemi(FIVE_REGIONS), tiny_map(NEAREST, 5)
    expected = {
        "search": [False, True, True, True, True, False, False, True],
        "medial": [False, True, True, False, False, False, False, False],
        "inferior": [False, False, False, True, False, False, False, False],
        "lateral": [False, False, False, False, True, False, False, True],
        "control": [False, False, False, False, False, True, True, False],
    }
    assert set(expected) == set(REGION_NAMES)
    for name in REGION_NAMES:
        mask, kind = native_region(name, annot=None, smap=smap, template_hemi=hemi, regions=REGIONS)
        assert kind == "template"
        assert mask is not None and mask.dtype == np.bool_
        assert mask.tolist() == expected[name], name


def test_template_kind_keeps_only_what_the_annotation_calls_cortex() -> None:
    hemi, smap = tiny_hemi(FIVE_REGIONS), tiny_map(NEAREST, 5)
    annot = annotation(
        "cuneus",
        "cuneus",
        "unknown",
        "corpuscallosum",
        None,
        "superiorfrontal",
        None,
        # cortex of a parcel that is in no region: the template decides the region
        "insula",
    )
    mask, kind = native_region(
        "search", annot=annot, smap=smap, template_hemi=hemi, regions=REGIONS
    )
    assert kind == "template"
    assert mask is not None
    assert mask.tolist() == [False, True, False, False, False, False, False, True]
    control, _ = native_region(
        "control", annot=annot, smap=smap, template_hemi=hemi, regions=REGIONS
    )
    assert control is not None
    assert control.tolist() == [False, False, False, False, False, True, False, False]


def test_template_kind_does_not_change_the_template_or_the_map() -> None:
    hemi, smap = tiny_hemi(FIVE_REGIONS), tiny_map(NEAREST, 5)
    annot = annotation(*["cuneus"] * 7, None)
    region = hemi.regions["search"].copy()
    first, _ = native_region("search", annot=annot, smap=smap, template_hemi=hemi, regions=REGIONS)
    assert first is not None
    first[:] = True
    assert np.array_equal(hemi.regions["search"], region)
    assert smap.nearest_target.tolist() == NEAREST
    # also where native and template mesh are one and nothing is masked: a new array
    same_mesh, _ = native_region(
        "search", annot=None, smap=tiny_map([0, 1, 2, 3, 4], 5), template_hemi=hemi, regions=REGIONS
    )
    assert same_mesh is not None and same_mesh.tolist() == FIVE_REGIONS["search"]
    assert not np.shares_memory(same_mesh, hemi.regions["search"])
    again, _ = native_region("search", annot=annot, smap=smap, template_hemi=hemi, regions=REGIONS)
    assert again is not None and again.tolist() == [
        False,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
    ]


def test_an_annotation_of_another_length_is_a_vertex_count_mismatch() -> None:
    hemi, smap = tiny_hemi(FIVE_REGIONS), tiny_map(NEAREST, 5)
    annot = annotation(*["cuneus"] * 7)
    with pytest.raises(MetricError) as caught:
        native_region("search", annot=annot, smap=smap, template_hemi=hemi, regions=REGIONS)
    assert caught.value.reason == "vertex count mismatch"
    assert caught.value.detail == "the annotation has 7 vertices and the sphere 8"
    assert schemas.FLAG_OF_REASON[caught.value.reason] == "vertex_count_mismatch"


def test_a_map_onto_another_mesh_is_refused() -> None:
    hemi = tiny_hemi(FIVE_REGIONS)
    with pytest.raises(ValueError, match="6 target vertices and the template mesh 5"):
        native_region(
            "search", annot=None, smap=tiny_map([0, 1], 6), template_hemi=hemi, regions=REGIONS
        )


def test_template_kind_takes_its_regions_from_the_template_not_the_config() -> None:
    hemi, smap = tiny_hemi(FIVE_REGIONS), tiny_map(NEAREST, 5)
    other = RegionConfig(occipital=("insula",), control=("bankssts",))
    annot = annotation(*["cuneus"] * 8)
    for regions in (REGIONS, other):
        mask, kind = native_region(
            "search", annot=annot, smap=smap, template_hemi=hemi, regions=regions
        )
        assert kind == "template" and mask is not None
        assert mask.tolist() == [False, True, True, True, True, False, False, True]


ANNOTATED = annotation(
    "cuneus",
    "lateraloccipital",
    "lingual",
    "pericalcarine",
    "precuneus",
    "fusiform",
    "caudalmiddlefrontal",
    "rostralmiddlefrontal",
    "superiorfrontal",
    "insula",
    "unknown",
    "corpuscallosum",
    None,
)


def test_annot_kind_is_the_subjects_own_parcels() -> None:
    search, kind = native_region(
        "search", annot=ANNOTATED, smap=None, template_hemi=None, regions=REGIONS
    )
    assert kind == "annot" and search is not None and search.dtype == np.bool_
    # the occipital parcels alone: precuneus and fusiform are cut on the template only
    assert search.tolist() == [True] * 4 + [False] * 9
    control, kind = native_region(
        "control", annot=ANNOTATED, smap=None, template_hemi=None, regions=REGIONS
    )
    assert kind == "annot" and control is not None
    assert control.tolist() == [False] * 6 + [True] * 3 + [False] * 4


def test_annot_kind_follows_the_region_config() -> None:
    regions = RegionConfig(occipital=("cuneus", "insula"), control=("precuneus",))
    search, _ = native_region(
        "search", annot=ANNOTATED, smap=None, template_hemi=None, regions=regions
    )
    control, _ = native_region(
        "control", annot=ANNOTATED, smap=None, template_hemi=None, regions=regions
    )
    assert search is not None and np.flatnonzero(search).tolist() == [0, 9]
    assert control is not None and np.flatnonzero(control).tolist() == [4]


@pytest.mark.parametrize("name", ["medial", "inferior", "lateral"])
def test_the_annotation_cannot_give_the_parts_of_the_search_region(name: RegionName) -> None:
    assert native_region(name, annot=ANNOTATED, smap=None, template_hemi=None, regions=REGIONS) == (
        None,
        "none",
    )


@pytest.mark.parametrize("name", REGION_NAMES)
def test_without_template_and_annotation_there_is_no_region(name: RegionName) -> None:
    assert native_region(name, annot=None, smap=None, template_hemi=None, regions=REGIONS) == (
        None,
        "none",
    )


def test_a_template_without_a_map_and_a_map_without_a_template_are_no_template() -> None:
    hemi, smap = tiny_hemi(FIVE_REGIONS), tiny_map(NEAREST, 5)
    annot = annotation(*["cuneus"] * 3, *["insula"] * 5)
    halves: tuple[tuple[SphereMap | None, TemplateHemi | None], ...] = ((smap, None), (None, hemi))
    for only_map, only_hemi in halves:
        mask, kind = native_region(
            "search", annot=annot, smap=only_map, template_hemi=only_hemi, regions=REGIONS
        )
        assert kind == "annot" and mask is not None
        assert mask.tolist() == [True] * 3 + [False] * 5
        assert native_region(
            "search", annot=None, smap=only_map, template_hemi=only_hemi, regions=REGIONS
        ) == (None, "none")
        assert native_region(
            "medial", annot=annot, smap=only_map, template_hemi=only_hemi, regions=REGIONS
        ) == (None, "none")


def test_a_parcel_without_vertices_gives_a_region_that_may_be_empty() -> None:
    annot = annotation("insula", "superiorfrontal", None)
    search, kind = native_region(
        "search", annot=annot, smap=None, template_hemi=None, regions=REGIONS
    )
    assert kind == "annot" and search is not None and not search.any()
    counts = ceiling_counts(f32(5.0, 5.0, 0.0), search)
    assert counts == CeilingCounts(0, 0, 0, 0) and math.isnan(counts.fraction)


def test_an_annotation_whose_table_lacks_a_parcel_gives_no_region(
    caplog: pytest.LogCaptureFixture,
) -> None:
    names = tuple(n for n in DESIKAN_NAMES if n not in ("lingual", "pericalcarine"))
    annot = annotation("cuneus", "lateraloccipital", "superiorfrontal", names=names)
    with caplog.at_level(logging.WARNING, logger="surfcensus"):
        assert native_region(
            "search", annot=annot, smap=None, template_hemi=None, regions=REGIONS
        ) == (None, "none")
    assert [record.getMessage() for record in caplog.records] == [
        "The annotation has no parcel lingual, pericalcarine: no search region from it."
    ]
    assert caplog.records[0].levelno == logging.WARNING
    # the control parcels are all there, and the template does not ask the table for parcels
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="surfcensus"):
        control, kind = native_region(
            "control", annot=annot, smap=None, template_hemi=None, regions=REGIONS
        )
        assert kind == "annot" and control is not None
        assert control.tolist() == [False, False, True]
        mask, kind = native_region(
            "search",
            annot=annot,
            smap=tiny_map([0, 3, 1], 5),
            template_hemi=tiny_hemi(FIVE_REGIONS),
            regions=REGIONS,
        )
        assert kind == "template" and mask is not None and mask.tolist() == [True, False, True]
    assert caplog.records == []


# ---------------------------------------------------------------------------------------
# native_region and the counts on a synthetic subject

WARP = 0.15
TEMPLATE_ORDER = 4
PATCH = ThickPatch(
    centre=direction("lh", from_pole_deg=40.0, azimuth_deg=180.0), radius_deg=14.0, value_mm=5.0
)
"""On the left hemisphere, across the lateral edge of the posterior cap: lateraloccipital
inside the cap, precuneus above and fusiform below in the ring in front of it. So it lies
inside the template's search region, in all three of its parts, and only partly inside the
occipital parcels. On the right hemisphere the same direction is medial of the pole:
pericalcarine, precuneus and fusiform, and no lateraloccipital."""


@pytest.fixture(scope="module")
def written_template(tmp_path_factory: pytest.TempPathFactory) -> SyntheticTemplate:
    return write_synthetic_template(tmp_path_factory.mktemp("ceiling_template"), order=5)


@pytest.fixture(scope="module")
def small_template(written_template: SyntheticTemplate) -> Template:
    """fsaverage4 (2,562 vertices) from a synthetic directory of order 5."""
    return build_template(written_template.path, "explicit", mesh="fsaverage4")


@pytest.fixture(scope="module")
def patched_subject(
    tmp_path_factory: pytest.TempPathFactory, written_template: SyntheticTemplate
) -> SyntheticSubject:
    """10,242 vertices per surface, registered with a warp, with :data:`PATCH` at 5 mm."""
    subject = write_synthetic_subject(
        tmp_path_factory.mktemp("ceiling_subject"),
        voxel_size=10.0,
        shape=(16, 16, 16),
        subdivisions=5,
    )
    add_tier_b(subject, written_template, warp=WARP, patch=PATCH)
    return subject


def registered(subject: SyntheticSubject) -> np.ndarray:
    """Template-space direction of every native vertex."""
    return warp_directions(subject_directions(subject)[0], WARP)


def nearest_template_vertex(points: np.ndarray) -> np.ndarray:
    """By brute force: on the unit sphere the nearest vertex has the largest dot product."""
    return np.argmax(points @ template_mesh(TEMPLATE_ORDER)[0].T, axis=1)


def expected_template_region(
    subject: SyntheticSubject, hemi: Hemi, name: str, *, cortex_only: bool
) -> np.ndarray:
    points = registered(subject)
    on_template = expected_regions(template_mesh(TEMPLATE_ORDER)[0], hemi)[name]
    mask = on_template[nearest_template_vertex(points)]
    return mask & (parcel_of(points, hemi) >= 0) if cortex_only else mask


def read_hemi(
    subject: SyntheticSubject, hemi: Hemi, template: Template
) -> tuple[np.ndarray, Annotation, SphereMap]:
    thickness = read_morph(subject.path / "surf" / f"{hemi}.thickness")
    annot = read_annotation(subject.path / "label" / f"{hemi}.aparc.annot")
    sphere = read_surface(subject.path / "surf" / f"{hemi}.sphere.reg")
    return thickness, annot, build_sphere_map(sphere, template.hemis[hemi].sphere)


@pytest.mark.parametrize("hemi", HEMIS)
def test_template_regions_on_the_synthetic_subject(
    patched_subject: SyntheticSubject, small_template: Template, hemi: Hemi
) -> None:
    _, annot, smap = read_hemi(patched_subject, hemi, small_template)
    strip = medial_coordinate(registered(patched_subject), hemi) > MEDIAL_WALL_M
    for name in REGION_NAMES:
        with_annot, kind = native_region(
            name, annot=annot, smap=smap, template_hemi=small_template.hemis[hemi], regions=REGIONS
        )
        assert kind == "template" and with_annot is not None
        expected = expected_template_region(patched_subject, hemi, name, cortex_only=True)
        assert expected.sum() > 100, name
        assert np.array_equal(with_annot, expected), name
        assert not (with_annot & strip).any()

        bare, kind = native_region(
            name, annot=None, smap=smap, template_hemi=small_template.hemis[hemi], regions=REGIONS
        )
        assert kind == "template" and bare is not None
        assert np.array_equal(
            bare, expected_template_region(patched_subject, hemi, name, cortex_only=False)
        ), name
        assert np.array_equal(bare & ~strip, with_annot)
    # the search region reaches the medial strip, so the annotation has something to remove
    search, _ = native_region(
        "search", annot=None, smap=smap, template_hemi=small_template.hemis[hemi], regions=REGIONS
    )
    assert search is not None and (search & strip).sum() >= 5


@pytest.mark.parametrize("hemi", HEMIS)
def test_the_parts_divide_the_search_region(
    patched_subject: SyntheticSubject, small_template: Template, hemi: Hemi
) -> None:
    _, annot, smap = read_hemi(patched_subject, hemi, small_template)

    def region(name: RegionName) -> np.ndarray:
        mask, _ = native_region(
            name, annot=annot, smap=smap, template_hemi=small_template.hemis[hemi], regions=REGIONS
        )
        assert mask is not None
        return mask

    parts = [region(name) for name in ("medial", "inferior", "lateral")]
    assert np.array_equal(np.sum(parts, axis=0), region("search").astype(int))
    assert not (region("search") & region("control")).any()


@pytest.mark.parametrize("hemi", HEMIS)
def test_annot_regions_on_the_synthetic_subject(
    patched_subject: SyntheticSubject, small_template: Template, hemi: Hemi
) -> None:
    _, annot, smap = read_hemi(patched_subject, hemi, small_template)
    parcel = parcel_of(registered(patched_subject), hemi)
    for name, parcels in (("search", REGIONS.occipital), ("control", REGIONS.control)):
        mask, kind = native_region(
            name, annot=annot, smap=None, template_hemi=None, regions=REGIONS
        )
        assert kind == "annot" and mask is not None
        expected = np.isin(parcel, [DESIKAN_NAMES.index(p) for p in parcels])
        assert expected.sum() > 100
        assert np.array_equal(mask, expected), name
    # the template's search region holds the occipital parcels and more: the two kinds are
    # not one region, here as on the real subject (16,429 against 21,484 vertices)
    own, _ = native_region("search", annot=annot, smap=None, template_hemi=None, regions=REGIONS)
    through_template, _ = native_region(
        "search", annot=annot, smap=smap, template_hemi=small_template.hemis[hemi], regions=REGIONS
    )
    assert own is not None and through_template is not None
    assert through_template.sum() > 1.3 * own.sum()
    assert (own & ~through_template).sum() < 0.02 * own.sum()


def test_ceiling_counts_of_a_patch_by_region_and_kind(
    patched_subject: SyntheticSubject, small_template: Template
) -> None:
    hemi: Hemi = "lh"
    thickness, annot, smap = read_hemi(patched_subject, hemi, small_template)
    points = registered(patched_subject)
    patch = in_patch(points, PATCH)
    assert not thickness_above_max(thickness)
    assert np.array_equal(thickness >= 4.9, patch) and patch.sum() > 100
    # all thickness outside the patch and the medial strip is 2.0 to 3.0 mm

    def counts(name: RegionName, *, template: bool, with_annot: bool = True) -> CeilingCounts:
        mask, _ = native_region(
            name,
            annot=annot if with_annot else None,
            smap=smap if template else None,
            template_hemi=small_template.hemis[hemi] if template else None,
            regions=REGIONS,
        )
        assert mask is not None
        return ceiling_counts(thickness, mask)

    by_part = {}
    for name in REGION_NAMES:
        expected = expected_template_region(patched_subject, hemi, name, cortex_only=True)
        n_patch = int((expected & patch).sum())
        got = counts(name, template=True)
        assert got == CeilingCounts(int(expected.sum()), n_patch, n_patch, 0), name
        assert got.fraction == n_patch / expected.sum()
        by_part[name] = n_patch
    # the patch lies in the search region with all of its vertices, in each of the three parts
    assert by_part["search"] == patch.sum()
    assert by_part["search"] == by_part["medial"] + by_part["inferior"] + by_part["lateral"]
    assert min(by_part["medial"], by_part["inferior"], by_part["lateral"]) > 10
    assert by_part["control"] == 0 and counts("control", template=True).fraction == 0.0

    # the subject's own occipital parcels hold only the part of the patch inside the cap
    occipital = np.isin(
        parcel_of(points, hemi), [DESIKAN_NAMES.index(p) for p in REGIONS.occipital]
    )
    own = counts("search", template=False)
    assert own == CeilingCounts(
        int(occipital.sum()), int((occipital & patch).sum()), int((occipital & patch).sum()), 0
    )
    assert 10 < own.n_exact < by_part["search"] - 10
    assert counts("control", template=False).fraction == 0.0

    # without the annotation the medial strip, whose thickness is 0, is counted with the region
    strip = medial_coordinate(points, hemi) > MEDIAL_WALL_M
    bare = expected_template_region(patched_subject, hemi, "search", cortex_only=False)
    without = counts("search", template=True, with_annot=False)
    assert without == CeilingCounts(
        int(bare.sum()), by_part["search"], by_part["search"], int((bare & strip).sum())
    )
    assert without.n_zero >= 5 and without.fraction > counts("search", template=True).fraction


def test_the_patch_on_the_right_hemisphere_is_in_other_parts(
    patched_subject: SyntheticSubject, small_template: Template
) -> None:
    # add_tier_b gives both hemispheres the patch at one template-space direction; on the
    # right one that direction is medial of the pole, so the lateral part has none of it
    thickness, annot, smap = read_hemi(patched_subject, "rh", small_template)
    patch = in_patch(registered(patched_subject), PATCH)
    exact = {}
    for name in REGION_NAMES:
        mask, _ = native_region(
            name, annot=annot, smap=smap, template_hemi=small_template.hemis["rh"], regions=REGIONS
        )
        assert mask is not None
        expected = expected_template_region(patched_subject, "rh", name, cortex_only=True)
        n_patch = int((expected & patch).sum())
        assert ceiling_counts(thickness, mask) == CeilingCounts(
            int(expected.sum()), n_patch, n_patch, 0
        ), name
        exact[name] = n_patch
    assert exact["lateral"] == 0 and exact["control"] == 0
    assert exact["medial"] > 10 and exact["inferior"] > 10
    assert exact["search"] == exact["medial"] + exact["inferior"] == patch.sum()


# ---------------------------------------------------------------------------------------
# the real subject, where the network tests have fetched it

REAL_SUBJECT = "ds000102-sub-01"
REAL_FSAVERAGE = "ds000102-fsaverage"
REAL_WHOLE = {  # every vertex of the hemisphere; FreeSurfer 6.0.1, counted 2026-11-13
    "lh": CeilingCounts(n_region=148_332, n_exact=16, n_near=28, n_zero=5_919),
    "rh": CeilingCounts(n_region=146_947, n_exact=41, n_near=61, n_zero=5_875),
}
REAL_CORTEX = {  # what the subject's aparc.annot calls cortex
    "lh": CeilingCounts(n_region=140_305, n_exact=16, n_near=28, n_zero=5),
    "rh": CeilingCounts(n_region=138_769, n_exact=16, n_near=27, n_zero=12),
}
REAL_ANNOT = {
    "lh": {"search": 16_429, "control": 27_172},
    "rh": {"search": 16_313, "control": 26_312},
}
REAL_TEMPLATE = {  # through fsaverage6 of the fsaverage next to the subject
    "lh": {
        "search": 21_484,
        "medial": 7_111,
        "inferior": 7_326,
        "lateral": 7_047,
        "control": 27_100,
    },
    "rh": {
        "search": 21_245,
        "medial": 7_860,
        "inferior": 6_636,
        "lateral": 6_749,
        "control": 26_212,
    },
}


def real_hemi_files(subject: Path, hemi: str) -> tuple[Path, Path, Path]:
    return (
        subject / "surf" / f"{hemi}.thickness",
        subject / "label" / f"{hemi}.aparc.annot",
        subject / "surf" / f"{hemi}.sphere.reg",
    )


def test_the_real_subject_on_its_own_annotation(cached_subjects: Path) -> None:
    subject = cached_subjects / REAL_SUBJECT
    if not all(path.is_file() for hemi in HEMIS for path in real_hemi_files(subject, hemi)[:2]):
        pytest.skip("thickness and annotation of the real subject are not in the pytest cache")
    for hemi in HEMIS:
        thickness_file, annot_file, _ = real_hemi_files(subject, hemi)
        thickness, annot = read_morph(thickness_file), read_annotation(annot_file)
        assert thickness.dtype == np.float32 and thickness.max() == 5.0
        assert not thickness_above_max(thickness)
        assert ceiling_counts(thickness, everywhere(len(thickness))) == REAL_WHOLE[hemi]
        assert ceiling_counts(thickness, annot.cortex()) == REAL_CORTEX[hemi]
        for name, n_region in REAL_ANNOT[hemi].items():
            mask, kind = native_region(
                typing.cast(RegionName, name),
                annot=annot,
                smap=None,
                template_hemi=None,
                regions=REGIONS,
            )
            assert kind == "annot" and mask is not None
            counts = ceiling_counts(thickness, mask)
            # this subject has no vertex at the ceiling or at zero in either region
            assert counts == CeilingCounts(n_region, 0, 0, 0), (hemi, name)
            assert counts.fraction == 0.0


def test_the_real_subject_through_the_real_template(cached_subjects: Path) -> None:
    subject, fsaverage = cached_subjects / REAL_SUBJECT, cached_subjects / REAL_FSAVERAGE
    needed = [path for hemi in HEMIS for path in real_hemi_files(subject, hemi)]
    needed += [fsaverage / name.format(h=hemi) for name in TEMPLATE_FILES for hemi in HEMIS]
    if not all(path.is_file() for path in needed):
        pytest.skip("the real subject and the template files are not all in the pytest cache")
    check_real_template_regions(subject, fsaverage)


def check_real_template_regions(subject: Path, fsaverage: Path) -> None:
    template = build_template(fsaverage, "adjacent")
    for hemi in HEMIS:
        thickness_file, annot_file, sphere_file = real_hemi_files(subject, hemi)
        thickness, annot = read_morph(thickness_file), read_annotation(annot_file)
        smap = build_sphere_map(read_surface(sphere_file), template.hemis[hemi].sphere)
        for name, n_region in REAL_TEMPLATE[hemi].items():
            region = typing.cast(RegionName, name)
            mask, kind = native_region(
                region, annot=annot, smap=smap, template_hemi=template.hemis[hemi], regions=REGIONS
            )
            assert kind == "template" and mask is not None
            assert ceiling_counts(thickness, mask) == CeilingCounts(n_region, 0, 0, 0), (hemi, name)
            # on this subject the annotation removes no vertex: none of the non-cortex
            # vertices has its nearest template vertex in a region
            bare, _ = native_region(
                region, annot=None, smap=smap, template_hemi=template.hemis[hemi], regions=REGIONS
            )
            assert bare is not None and np.array_equal(bare, mask)
