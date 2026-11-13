"""Arrow schemas and keys of the parquet tables under ``tables/`` of a run directory.

Key columns come first in every table and are its sort order; the first of them is always
``subject_uid``. Strings are Arrow ``string`` and floats ``float32``. A column that may be
null is nullable in the schema; every other column must have a value.

The tables are written by the metrics phases (milestone 3) and by the render (the
intensity table, milestone 5). They are defined here from the start so that the table
helpers of ``store.py`` know every key.
"""

from __future__ import annotations

from collections.abc import Iterable

import pyarrow as pa

NO_REFERENCE = "none"
"""``reference_id`` of a row that was computed without a reference: the scores of decision
D16, and those of a subject without a z-map."""

SCORE_KINDS: tuple[str, ...] = ("composite_z", "fallback_rank")
"""Values of ``scores.score_kind``: the maximum of standardised components, or the ranking
of decision D16."""

REGION_KINDS: tuple[str, ...] = ("template", "annot", "none")
"""Values of ``vertex_summary.region_kind`` and ``scores.ceiling_region``: how the search
region was placed on the native surface."""

VERTEX_FLAGS: tuple[str, ...] = (
    "degenerate_sphere",
    "low_coverage",
    "no_annot",
    "no_sphere_reg",
    "no_template",
    "no_thickness",
    "sphere_flipped",
    "thickness_above_max",
    "vertex_count_mismatch",
)
"""What ``vertex_summary.flags`` can hold for one hemisphere, sorted."""

FLAG_OF_REASON: dict[str, str] = {
    "degenerate sphere": "degenerate_sphere",
    "low coverage": "low_coverage",
    "vertex count mismatch": "vertex_count_mismatch",
}
"""The flag a hemisphere gets when its map fails with a ``MetricError`` of that reason. The
other reasons (``missing file``, ``unreadable``) fail the subject, not the hemisphere."""

SCORE_FLAGS: tuple[str, ...] = tuple(sorted((*VERTEX_FLAGS, "no_stats", "partial_score")))
"""What ``scores.flags`` can hold: the union of the hemispheres' flags, ``no_stats`` for a
subject without stats files, and ``partial_score`` for one scored on fewer components than
configured."""


def join_flags(flags: Iterable[str]) -> str:
    """The text of a ``flags`` column: the distinct flags, sorted, joined with commas; empty
    for none. Raises ValueError for a flag that is not in :data:`SCORE_FLAGS`."""
    distinct = sorted(set(flags))
    unknown = [flag for flag in distinct if flag not in SCORE_FLAGS]
    if unknown:
        raise ValueError(f"not a flag of the metric tables: {', '.join(unknown)}")
    return ",".join(distinct)


def split_flags(text: str) -> tuple[str, ...]:
    """The flags in the text of a ``flags`` column, in its order; ``()`` for an empty text."""
    return tuple(flag for flag in text.split(",") if flag)


def _schema(*groups: tuple[pa.DataType, bool, tuple[str, ...]]) -> pa.Schema:
    return pa.schema(
        [
            pa.field(name, kind, nullable=nullable)
            for kind, nullable, names in groups
            for name in names
        ]
    )


def _required(kind: pa.DataType, *names: str) -> tuple[pa.DataType, bool, tuple[str, ...]]:
    return kind, False, names


def _nullable(kind: pa.DataType, *names: str) -> tuple[pa.DataType, bool, tuple[str, ...]]:
    return kind, True, names


_STRING, _BOOL = pa.string(), pa.bool_()
_INT8, _INT16, _INT32, _FLOAT32 = pa.int8(), pa.int16(), pa.int32(), pa.float32()

REGIONAL = _schema(
    _required(_STRING, "subject_uid", "hemi", "parcel"),
    _required(_INT32, "num_vert", "surf_area_mm2", "gray_vol_mm3"),
    _required(_FLOAT32, "thick_avg_mm", "thick_std_mm"),
)
"""One row per parcel of ``?h.aparc.stats``."""

VERTEX_SUMMARY = _schema(
    _required(_STRING, "subject_uid", "hemi"),
    _required(_INT32, "n_native"),
    _required(_FLOAT32, "thickness_max_mm"),
    _nullable(_STRING, "template_id"),
    _nullable(_INT32, "n_fallback", "n_missing_cortex"),
    _nullable(_FLOAT32, "global_median_mm"),
    _required(_STRING, "region_kind"),
    _nullable(_INT32, "search_n", "search_n_exact", "search_n_near", "search_n_zero"),
    _nullable(_INT32, "control_n", "control_n_exact", "control_n_near", "control_n_zero"),
    _nullable(_FLOAT32, "ceiling_frac", "control_ceiling_frac"),
    _nullable(_INT32, "annot_search_n", "annot_search_n_near", "annot_search_n_zero"),
    _nullable(_FLOAT32, "ceiling_frac_annot"),
    _required(_STRING, "flags"),
)
"""One row per subject and hemisphere: resampling facts and ceiling counts."""

CLUSTERS = _schema(
    _required(_STRING, "subject_uid", "reference_id", "hemi", "region"),
    _required(_INT16, "rank"),
    _required(_INT32, "n_vertices"),
    _required(_FLOAT32, "area_mm2", "mass", "excess_mass", "peak_z"),
    _required(_INT32, "peak_vertex", "centroid_vertex"),
)
"""The largest outlier clusters per subject, reference, hemisphere and region."""

CLUSTERS_NATIVE = _schema(
    _required(_STRING, "subject_uid", "reference_id", "hemi", "region"),
    _required(_INT16, "rank"),
    _required(_INT32, "native_vertex"),
    _required(_FLOAT32, "tkr_x", "tkr_y", "tkr_z"),
    _required(_STRING, "template_id"),
)
"""The clusters of ``CLUSTERS`` whose centroid has a native position in a rendered box."""

SCORES = _schema(
    _required(_STRING, "subject_uid", "reference_id"),
    _required(_BOOL, "primary"),
    _required(_STRING, "dataset", "score_kind"),
    _nullable(_STRING, "stratum", "template_id"),
    _nullable(_INT32, "n_reference"),
    _nullable(_FLOAT32, "composite"),
    _nullable(_STRING, "composite_component"),
    _required(_INT8, "n_components"),
    _required(_INT32, "rank"),
    _required(_STRING, "mass_kind"),
    _nullable(
        _FLOAT32, "mass_search", "mass_medial", "mass_inferior", "mass_lateral", "mass_control"
    ),
    _nullable(_FLOAT32, "area_search_mm2", "peak_z_search"),
    _nullable(
        _FLOAT32,
        "ceiling_frac",
        "control_ceiling_frac",
        "ceiling_exact_frac",
        "zero_frac",
        "ceiling_frac_annot",
    ),
    _required(_STRING, "ceiling_region"),
    _nullable(_FLOAT32, "occ_thick_mm", "ctrl_thick_mm", "occ_ctrl_ratio"),
    _nullable(
        _FLOAT32, "std_mass_medial", "std_mass_inferior", "std_ceiling_frac", "std_occ_ctrl_ratio"
    ),
    _nullable(_INT32, "holes_lh", "holes_rh"),
    _nullable(_FLOAT32, "trunc_frac"),
    _required(_STRING, "flags", "score_config_hash"),
)
"""One row per present subject and reference; ``dataset`` is the pseudonym."""


DEFAULT_INTENSITY_DEPTHS = 4
"""Number of depths of the default ``IntensityConfig``."""


def intensity_schema(n_depths: int) -> pa.Schema:
    """Schema of the intensity table for ``n_depths`` sampling depths.

    The medians at depth ``k`` (0-based position in ``IntensityConfig.depths``) are the
    columns ``search_median_d<k>`` and ``control_median_d<k>``. A hemisphere without
    regions has nulls in everything but the key and ``region_kind``.
    """
    medians = [f"{region}_median_d{k}" for region in ("search", "control") for k in range(n_depths)]
    return _schema(
        _required(_STRING, "subject_uid", "hemi"),
        _required(_STRING, "region_kind"),
        _nullable(_INT32, "search_n", "control_n"),
        _nullable(_FLOAT32, "outer_outlier_frac", "gap_frac"),
        _nullable(_FLOAT32, *medians),
    )


SCHEMAS: dict[str, pa.Schema] = {
    "regional": REGIONAL,
    "vertex_summary": VERTEX_SUMMARY,
    "clusters": CLUSTERS,
    "clusters_native": CLUSTERS_NATIVE,
    "scores": SCORES,
    "intensity": intensity_schema(DEFAULT_INTENSITY_DEPTHS),
}
"""Table name (the file is ``tables/<name>.parquet``) -> schema. The intensity entry is for
the default depths; a run with other depths takes its schema from :func:`intensity_schema`."""

KEYS: dict[str, tuple[str, ...]] = {
    "regional": ("subject_uid", "hemi", "parcel"),
    "vertex_summary": ("subject_uid", "hemi"),
    "clusters": ("subject_uid", "reference_id", "hemi", "region", "rank"),
    "clusters_native": ("subject_uid", "reference_id", "hemi", "region", "rank"),
    "scores": ("subject_uid", "reference_id"),
    "intensity": ("subject_uid", "hemi"),
}
"""Table name -> key columns: unique per row, and the order rows are written in."""
