"""What the modules of milestone 3 share: error classes, the vocabulary of the metric
tables, and the names that the packages of the milestone import from each other.

The names were fixed before the functions were written (decision D30), and stay fixed now
that they are.
"""

from __future__ import annotations

import dataclasses
import pickle
import typing
from importlib import import_module

import pytest

import surfcensus as sc
from surfcensus import records, schemas, templates
from surfcensus.metrics import MetricError
from surfcensus.metrics.ceiling import RegionKind
from surfcensus.run import METRIC_STAGES
from surfcensus.templates import (
    MESH_ORDER,
    REGION_NAMES,
    TemplateError,
    TemplateNotFound,
    icosahedron_vertices,
)

INTERFACE: dict[str, tuple[str, ...]] = {
    "surfcensus.metrics": ("MetricError",),
    "surfcensus.fsio.stats": (
        "StatsFile",
        "parse_stats",
        "read_stats",
        "aparc_frame",
        "surface_holes",
        "euler_number",
    ),
    "surfcensus.fsio.annot": ("Annotation", "read_annotation", "read_label"),
    "surfcensus.fsio.morph": ("read_morph",),
    "surfcensus.fsio.registration": ("SphereMap", "unit_sphere", "build_sphere_map", "resample"),
    "surfcensus.templates": (
        "TEMPLATE_SCHEMA",
        "TEMPLATE_FILES",
        "TEMPLATE_OPTIONAL_FILES",
        "RegionName",
        "REGION_NAMES",
        "Origin",
        "MESH_ORDER",
        "TemplateHemi",
        "Template",
        "TemplateError",
        "TemplateNotFound",
        "coarsen_faces",
        "template_candidates",
        "build_template",
        "write_template",
        "read_template",
        "find_template",
        "store_template",
        "load_template",
    ),
    "surfcensus.metrics.regional": ("regional_rows", "region_mean_thickness", "occ_ctrl_ratio"),
    "surfcensus.metrics.ceiling": ("CeilingCounts", "ceiling_counts", "native_region"),
    "surfcensus.metrics.fsaverage": (
        "ReferenceMaps",
        "HemiMap",
        "resample_hemi",
        "global_median",
        "write_maps",
        "read_maps",
        "build_reference",
        "zmap",
        "assign_strata",
        "stratum_members",
    ),
    "surfcensus.metrics.clusters": ("Cluster", "find_clusters", "native_centroids"),
    "surfcensus.metrics.scoring": (
        "subject_metrics",
        "normal_scores",
        "robust_z",
        "quantile_scores",
        "composite",
        "fallback_rank",
        "add_rank",
    ),
    "surfcensus.metrics.pipeline": (
        "run_metrics",
        "update_native",
        "merge_tables",
        "metrics_pending",
        "map_path",
        "reference_dir",
    ),
    "surfcensus.reference.load": (
        "ReferenceMaps",
        "Reference",
        "shipped_reference",
        "applicable_stratum",
        "write_reference",
        "read_reference",
        "read_index",
        "stratum_size",
    ),
    "surfcensus.reference.strata": (
        "field_bin",
        "dataset_stratum",
        "field_version_stratum",
        "version_stratum",
        "members_sha256",
        "set_reference_id",
    ),
}

FIELDS: dict[tuple[str, str], tuple[str, ...]] = {
    ("surfcensus.fsio.stats", "StatsFile"): ("measures", "columns", "rows"),
    ("surfcensus.fsio.annot", "Annotation"): ("labels", "names"),
    ("surfcensus.fsio.registration", "SphereMap"): (
        "n_source",
        "n_target",
        "corners",
        "weights",
        "fallback",
        "nearest_source",
        "nearest_target",
        "flipped",
    ),
    ("surfcensus.templates", "TemplateHemi"): (
        "sphere",
        "faces",
        "white",
        "cortex",
        "parcel",
        "area",
        "regions",
        "pole_vertex",
    ),
    ("surfcensus.templates", "Template"): (
        "mesh",
        "order",
        "template_id",
        "origin",
        "files",
        "area_source",
        "build_stamp",
        "parcel_names",
        "regions",
        "hemis",
    ),
    ("surfcensus.metrics.ceiling", "CeilingCounts"): ("n_region", "n_exact", "n_near", "n_zero"),
    ("surfcensus.metrics.fsaverage", "HemiMap"): (
        "thickness",
        "n_fallback",
        "n_missing_cortex",
        "global_median",
        "flipped",
    ),
    ("surfcensus.metrics.clusters", "Cluster"): (
        "vertices",
        "area_mm2",
        "mass",
        "excess_mass",
        "peak_z",
        "peak_vertex",
        "centroid_vertex",
    ),
    ("surfcensus.reference.load", "ReferenceMaps"): ("median", "mad", "n"),
    ("surfcensus.reference.load", "Reference"): (
        "reference_id",
        "kind",
        "mesh",
        "template_id",
        "sphere_sha256",
        "config_hash",
        "strata",
        "members",
        "metric_norms",
        "parcel_norms",
    ),
}


@pytest.mark.parametrize("module", sorted(INTERFACE))
def test_the_names_that_packages_import_from_each_other_exist(module: str) -> None:
    loaded = import_module(module)
    missing = [name for name in INTERFACE[module] if not hasattr(loaded, name)]
    assert missing == []


@pytest.mark.parametrize(("module", "name"), sorted(FIELDS))
def test_the_shared_records_are_frozen_and_have_their_fields(module: str, name: str) -> None:
    record = getattr(import_module(module), name)
    assert dataclasses.is_dataclass(record)
    assert tuple(field.name for field in dataclasses.fields(record)) == FIELDS[module, name]
    assert record.__dataclass_params__.frozen


def test_the_reference_maps_of_the_metrics_are_those_of_the_reference() -> None:
    from surfcensus.metrics import fsaverage
    from surfcensus.reference import load

    assert fsaverage.ReferenceMaps is load.ReferenceMaps


def test_metric_error_carries_reason_and_detail_and_survives_pickling() -> None:
    error = MetricError("low coverage", "lh: 41% of the template cortex has a value")
    assert str(error) == "low coverage: lh: 41% of the template cortex has a value"
    assert (error.reason, error.detail) == ("low coverage", error.args[0].split(": ", 1)[1])
    copy = pickle.loads(pickle.dumps(error))
    assert type(copy) is MetricError
    assert (copy.reason, copy.detail, str(copy)) == (error.reason, error.detail, str(error))
    assert sc.metrics.MetricError is MetricError


def test_template_errors_survive_pickling() -> None:
    error = TemplateError("changed", "0123456789ab -> ba9876543210; pass --retemplate")
    copy = pickle.loads(pickle.dumps(error))
    assert type(copy) is TemplateError and (copy.reason, copy.detail) == (
        "changed",
        "0123456789ab -> ba9876543210; pass --retemplate",
    )
    looked_in = (
        ("freesurfer_home", "$FREESURFER_HOME/subjects/fsaverage", "not set"),
        ("adjacent", "/data/subjects/fsaverage", "incomplete: surf/lh.white"),
        ("bundled", "bundled copy", "not shipped"),
    )
    missing = TemplateNotFound(looked_in)
    assert missing.looked_in == looked_in and str(missing) == "no fsaverage template found"
    copy = pickle.loads(pickle.dumps(missing))
    assert type(copy) is TemplateNotFound and copy.looked_in == looked_in


def test_mesh_names_orders_and_sizes() -> None:
    assert MESH_ORDER == {"fsaverage4": 4, "fsaverage5": 5, "fsaverage6": 6, "fsaverage": 7}
    assert [icosahedron_vertices(order) for order in (0, 4, 5, 6, 7)] == [
        12,
        2_562,
        10_242,
        40_962,
        163_842,
    ]
    with pytest.raises(ValueError, match="not negative"):
        icosahedron_vertices(-1)
    assert REGION_NAMES == ("search", "medial", "inferior", "lateral", "control")
    assert templates.TEMPLATE_SCHEMA == 1
    assert (templates.TEMPLATE_JSON, templates.TEMPLATE_NPZ) == ("template.json", "template.npz")


def test_the_metric_stages() -> None:
    assert records.METRIC_STAGES == ("regional", "fsaverage", "reference", "score")
    assert records.ALL_METRIC_STAGES == (*records.METRIC_STAGES, "native", "merge")
    assert METRIC_STAGES is records.METRIC_STAGES


def test_flags_are_written_sorted_and_only_known_ones() -> None:
    assert schemas.join_flags([]) == ""
    assert schemas.join_flags(["no_template"]) == "no_template"
    assert schemas.join_flags(["sphere_flipped", "low_coverage", "sphere_flipped"]) == (
        "low_coverage,sphere_flipped"
    )
    assert schemas.join_flags(iter(("partial_score", "no_stats"))) == "no_stats,partial_score"
    with pytest.raises(ValueError, match="no_templat, typo"):
        schemas.join_flags(["no_templat", "typo", "no_annot"])
    assert schemas.split_flags("") == ()
    assert schemas.split_flags("low_coverage,sphere_flipped") == ("low_coverage", "sphere_flipped")
    for flags in (schemas.VERTEX_FLAGS, schemas.SCORE_FLAGS):
        assert list(flags) == sorted(set(flags))
        assert schemas.split_flags(schemas.join_flags(flags)) == flags
        assert not any("," in flag or " " in flag for flag in flags)
    assert set(schemas.SCORE_FLAGS) - set(schemas.VERTEX_FLAGS) == {"no_stats", "partial_score"}


def test_every_failing_reason_of_a_hemisphere_has_a_flag() -> None:
    assert set(schemas.FLAG_OF_REASON) == {
        "degenerate sphere",
        "low coverage",
        "vertex count mismatch",
    }
    assert set(schemas.FLAG_OF_REASON.values()) <= set(schemas.VERTEX_FLAGS)
    assert all(flag == reason.replace(" ", "_") for reason, flag in schemas.FLAG_OF_REASON.items())
    assert schemas.NO_REFERENCE == "none"
    assert schemas.SCORE_KINDS == ("composite_z", "fallback_rank")
    assert schemas.REGION_KINDS == ("template", "annot", "none")
    assert typing.get_args(RegionKind) == schemas.REGION_KINDS
