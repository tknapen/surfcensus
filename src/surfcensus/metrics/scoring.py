"""The composite score: subject-level metrics on one scale, and their maximum. See
docs/metrics.md. A ranking device for selection and review, not a classifier.

Every component is oriented so that larger is more suspicious. pandas and scipy only.

The order of use: :func:`subject_metrics` joins the metric tables into one row per subject
and reference; :func:`composite` (or, when no subject has a z-map, :func:`fallback_rank`)
puts the components on one scale and takes their maximum; :func:`add_rank` orders the
subjects. The frames hold the columns of ``schemas.SCORES`` in its types (float32, object
strings with None), so every tie and every maximum is decided on the values as they are
stored and can be checked from the table alone.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Literal

import numpy as np
import pandas as pd
from jaxtyping import Float
from scipy.stats import norm, rankdata

from surfcensus.config import RegionConfig, ScoringConfig
from surfcensus.schemas import NO_REFERENCE, REGION_KINDS, SCORES, join_flags, split_flags

__all__ = [
    "CLUSTER_METRICS",
    "FALLBACK_COMPONENTS",
    "MASS_REGIONS",
    "METRIC_COLUMNS",
    "STD_COLUMNS",
    "add_rank",
    "composite",
    "fallback_rank",
    "normal_scores",
    "quantile_scores",
    "robust_z",
    "subject_metrics",
]

FALLBACK_COMPONENTS: tuple[str, ...] = ("ceiling_frac", "occ_ctrl_ratio")
"""What subjects are ranked on when no z-map can be computed (decision D16)."""

MASS_REGIONS: tuple[str, ...] = ("search", "medial", "inferior", "lateral", "control")
"""The regions clusters are found in; ``scores`` has one ``mass_<region>`` column each."""

CLUSTER_METRICS: tuple[str, ...] = (
    *(f"mass_{region}" for region in MASS_REGIONS),
    "area_search_mm2",
    "peak_z_search",
)
"""The metrics that come from a z-map and so differ between the references of one subject.
Every other metric is one number per subject."""

METRIC_COLUMNS: tuple[str, ...] = (
    "subject_uid",
    "reference_id",
    "dataset",
    "template_id",
    "mass_kind",
    *CLUSTER_METRICS,
    "ceiling_frac",
    "control_ceiling_frac",
    "ceiling_exact_frac",
    "zero_frac",
    "ceiling_frac_annot",
    "ceiling_region",
    "occ_thick_mm",
    "ctrl_thick_mm",
    "occ_ctrl_ratio",
    "holes_lh",
    "holes_rh",
    "trunc_frac",
    "flags",
)
"""The columns :func:`subject_metrics` returns, in the order of ``schemas.SCORES``."""

STD_COLUMNS: tuple[str, ...] = tuple(name for name in SCORES.names if name.startswith("std_"))
"""The standardised components that ``scores.parquet`` has a column for. :func:`composite`
and :func:`fallback_rank` always return these, null for a component that was not used."""

_TRUNCATIONS = ("trunc_lh_1", "trunc_lh_2", "trunc_rh_1", "trunc_rh_2")
_KEY = ["subject_uid", "reference_id"]
_CEILING_FRACTIONS = (
    "ceiling_frac",
    "control_ceiling_frac",
    "ceiling_exact_frac",
    "zero_frac",
    "ceiling_frac_annot",
)
_REGIONAL = ("occ_thick_mm", "ctrl_thick_mm", "occ_ctrl_ratio")
_COMPOSITE_Z, _FALLBACK_RANK = "composite_z", "fallback_rank"
_PARTIAL = "partial_score"
_GRID_POINTS = 101

_Values = Float[np.ndarray, " n"]


def _require(frame: pd.DataFrame, what: str, columns: Sequence[str]) -> None:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"{what}: no column {', '.join(missing)}")


def _floats(values: pd.Series) -> _Values:
    """float64 with NaN for every missing value, whatever the type of the column."""
    return values.to_numpy(dtype=np.float64, na_value=np.nan)


def _texts(values: object, index: pd.Index | None = None) -> pd.Series:
    """A text column as the catalog and the tables hand it out: ``object``, None for missing."""
    series = pd.Series(values, index=index, dtype=object)
    return series.where(series.notna(), None)


# --- subject metrics ------------------------------------------------------------------------


def _present(subjects: pd.DataFrame) -> pd.DataFrame:
    """The subjects that get a row, indexed and sorted by ``subject_uid``."""
    columns = ("subject_uid", "dataset", "holes_lh", "holes_rh", *_TRUNCATIONS)
    _require(subjects, "subjects", columns)
    frame = subjects
    if "present" in frame.columns:
        frame = frame[frame["present"].to_numpy(dtype=bool)]
    if frame["subject_uid"].duplicated().any():
        raise ValueError("subjects: a subject_uid twice")
    return frame.sort_values("subject_uid").set_index("subject_uid")


def _from_vertex_summary(vertex_summary: pd.DataFrame, uids: pd.Index) -> pd.DataFrame:
    """Per subject: the ceiling fractions of the larger hemisphere, the region kind behind
    ``ceiling_frac``, the template id, the vertex count of both hemispheres and the flags."""
    counts = ("search_n", "search_n_exact", "search_n_zero")
    fractions = ("ceiling_frac", "control_ceiling_frac", "ceiling_frac_annot")
    texts = ("subject_uid", "hemi", "template_id", "region_kind", "flags")
    _require(vertex_summary, "vertex_summary", (*texts, "n_native", *counts, *fractions))
    rows = vertex_summary[vertex_summary["subject_uid"].isin(uids)]
    if rows.duplicated(["subject_uid", "hemi"]).any():
        raise ValueError("vertex_summary: more than one row for a hemisphere of one subject")
    unknown = sorted(set(rows["region_kind"]) - set(REGION_KINDS))
    if unknown:
        raise ValueError(f"vertex_summary: not a region kind: {', '.join(map(str, unknown))}")

    search_n = _floats(rows["search_n"])
    with np.errstate(divide="ignore", invalid="ignore"):
        exact = np.where(search_n > 0, _floats(rows["search_n_exact"]) / search_n, np.nan)
        zero = np.where(search_n > 0, _floats(rows["search_n_zero"]) / search_n, np.nan)
    work = pd.DataFrame(
        {
            "subject_uid": rows["subject_uid"].to_numpy(dtype=object),
            "hemi": rows["hemi"].to_numpy(dtype=object),
            "template_id": rows["template_id"].to_numpy(dtype=object),
            "region_kind": rows["region_kind"].to_numpy(dtype=object),
            "flags": rows["flags"].to_numpy(dtype=object),
            "n_native": _floats(rows["n_native"]),
            "ceiling_frac": _floats(rows["ceiling_frac"]),
            "control_ceiling_frac": _floats(rows["control_ceiling_frac"]),
            "ceiling_exact_frac": exact,
            "zero_frac": zero,
            "ceiling_frac_annot": _floats(rows["ceiling_frac_annot"]),
        }
    )
    grouped = work.groupby("subject_uid")
    out = grouped[list(_CEILING_FRACTIONS)].max().reindex(uids)

    # the kind of the hemisphere whose fraction is reported; on a tie the first of REGION_KINDS
    work["no_fraction"] = np.isnan(work["ceiling_frac"].to_numpy())
    work["kind_order"] = work["region_kind"].map({kind: k for k, kind in enumerate(REGION_KINDS)})
    by_fraction = work.sort_values(
        ["subject_uid", "no_fraction", "ceiling_frac", "kind_order", "hemi"],
        ascending=[True, True, False, True, True],
    ).drop_duplicates("subject_uid")
    kinds = by_fraction.set_index("subject_uid")["region_kind"].reindex(uids)
    out["ceiling_region"] = _texts(kinds.where(kinds.notna(), "none"))

    with_template = work[work["template_id"].notna()].sort_values("hemi", kind="stable")
    template = with_template.drop_duplicates("subject_uid").set_index("subject_uid")
    out["template_id"] = _texts(template["template_id"].reindex(uids))

    both = work[work["hemi"].isin(["lh", "rh"])].groupby("subject_uid")["n_native"]
    out["n_native"] = both.sum().where(both.count() == 2).reindex(uids)

    flags: dict[object, tuple[str, ...]] = {
        uid: tuple(flag for text in of_hemis if text for flag in split_flags(text))
        for uid, of_hemis in grouped["flags"]
    }
    out["hemi_flags"] = pd.Series([flags.get(uid, ()) for uid in uids], index=uids, dtype=object)
    return out


def _from_regional(regional: pd.DataFrame, regions: RegionConfig, uids: pd.Index) -> pd.DataFrame:
    """Per subject: the area-weighted mean thickness of the occipital and of the control
    parcels, both hemispheres pooled, and their ratio. The rule of
    ``metrics.regional.occ_ctrl_ratio``, for all subjects at once."""
    columns = ("subject_uid", "hemi", "parcel", "surf_area_mm2", "thick_avg_mm")
    _require(regional, "regional", columns)
    rows = regional[regional["subject_uid"].isin(uids)]
    if rows.duplicated(["subject_uid", "hemi", "parcel"]).any():
        raise ValueError("regional: more than one row for a parcel of one hemisphere")
    uid = rows["subject_uid"].to_numpy(dtype=object)
    area, thickness = _floats(rows["surf_area_mm2"]), _floats(rows["thick_avg_mm"])
    use = np.isfinite(thickness) & (area > 0)

    def mean(parcels: tuple[str, ...]) -> pd.Series:
        chosen = use & rows["parcel"].isin(set(parcels)).to_numpy(dtype=bool)
        sums = (
            pd.DataFrame(
                {
                    "subject_uid": uid[chosen],
                    "area": area[chosen],
                    "weighted": area[chosen] * thickness[chosen],
                }
            )
            .groupby("subject_uid")
            .sum()
        )
        return (sums["weighted"] / sums["area"]).reindex(uids)

    occipital, control = mean(regions.occipital), mean(regions.control)
    out = pd.DataFrame({"occ_thick_mm": occipital, "ctrl_thick_mm": control}, index=uids)
    usable = np.isfinite(_floats(occipital)) & (_floats(control) > 0)
    out["occ_ctrl_ratio"] = (occipital / control).where(usable)
    out["has_stats"] = uids.isin(pd.unique(uid))
    return out


def _largest_clusters(
    clusters: pd.DataFrame, uids: pd.Index, mass: Literal["excess", "sum_z"]
) -> pd.DataFrame:
    """Per subject and reference (the index): the metrics of the largest cluster of each
    region, the larger of the two hemispheres. NaN where there is no cluster."""
    rows = clusters[clusters["subject_uid"].isin(uids) & (clusters["rank"] == 1)]
    unknown = sorted(set(rows["region"]) - set(MASS_REGIONS))
    if unknown:
        raise ValueError(f"clusters: not a region: {', '.join(map(str, unknown))}")
    if rows.duplicated([*_KEY, "hemi", "region"]).any():
        raise ValueError("clusters: two clusters of rank 1 in one region of one hemisphere")
    work = pd.DataFrame(
        {
            "subject_uid": rows["subject_uid"].to_numpy(dtype=object),
            "reference_id": rows["reference_id"].to_numpy(dtype=object),
            "region": rows["region"].to_numpy(dtype=object),
            "mass": _floats(rows["excess_mass" if mass == "excess" else "mass"]),
            "area_mm2": _floats(rows["area_mm2"]),
            "peak_z": _floats(rows["peak_z"]),
        }
    )
    top = work.groupby([*_KEY, "region"])[["mass", "area_mm2", "peak_z"]].max()
    out = top["mass"].unstack("region").reindex(columns=list(MASS_REGIONS))
    out.columns = [f"mass_{region}" for region in MASS_REGIONS]
    search = top[top.index.get_level_values("region") == "search"].droplevel("region")
    out["area_search_mm2"] = search["area_mm2"].reindex(out.index)
    out["peak_z_search"] = search["peak_z"].reindex(out.index)
    return out


def _scored_pairs(
    clusters: pd.DataFrame | None,
    zmaps: Mapping[str, Collection[str]] | None,
    uids: pd.Index,
    has_map: pd.Series,
) -> list[tuple[str, str]]:
    """The ``(subject_uid, reference_id)`` pairs that have a z-map, sorted."""
    if clusters is None:
        if zmaps:
            raise ValueError("zmaps name a reference, and there is no cluster table")
        return []
    present = set(uids)
    if zmaps is None:
        references = sorted(set(clusters["reference_id"]))
        pairs = {(uid, reference) for reference in references for uid in uids[has_map.to_numpy()]}
    else:
        pairs = {(uid, ref) for ref, members in zmaps.items() for uid in members if uid in present}
    if any(reference == NO_REFERENCE for _, reference in pairs):
        raise ValueError(f"{NO_REFERENCE!r} is the reference_id of a row without a reference")
    with_clusters = clusters.loc[clusters["subject_uid"].isin(uids), _KEY].drop_duplicates()
    stray = sorted(set(with_clusters.itertuples(index=False, name=None)) - pairs)
    if stray:
        uid, reference = stray[0]
        raise ValueError(
            f"clusters: {len(stray)} pairs of subject and reference have clusters and no "
            f"z-map, the first is {uid} with {reference}; the tables are not of one state "
            "of the run"
        )
    return sorted(pairs)


def subject_metrics(
    vertex_summary: pd.DataFrame,
    clusters: pd.DataFrame | None,
    regional: pd.DataFrame,
    subjects: pd.DataFrame,
    regions: RegionConfig,
    *,
    mass: Literal["excess", "sum_z"],
    zmaps: Mapping[str, Collection[str]] | None = None,
) -> pd.DataFrame:
    """One row per subject and ``reference_id`` with :data:`METRIC_COLUMNS`, the metric
    columns of ``schemas.SCORES``, sorted by that key.

    ``vertex_summary``, ``clusters`` and ``regional`` are the tables of those names;
    ``subjects`` is ``PublicCatalog.subjects()`` for the present subjects (rows whose
    ``present`` is false are left out, and so are table rows of other subjects).

    Which rows. A subject gets one row for each reference it has a z-map against, and a
    subject without any gets one row with ``schemas.NO_REFERENCE``. ``zmaps`` says who has
    one: reference id -> the ``subject_uid``s scored against it. Without it, every subject
    with a map (a ``template_id`` in ``vertex_summary``) is taken to have a z-map against
    every reference that ``clusters`` names, which is right for a reference built within the
    run, wrong for a shipped one that has no stratum for some subjects, and finds no
    reference at all in a cluster table without rows. ``clusters`` None means the run has no
    reference: every row is ``NO_REFERENCE``.

    Columns.

    * Cluster metrics (:data:`CLUSTER_METRICS`): of the largest cluster (rank 1) of the
      region, the larger of the two hemispheres, each metric on its own. 0 for a subject
      that has a z-map and no cluster, null in a row without a reference. The ``mass_*``
      columns hold ``excess_mass`` or ``mass`` according to ``mass`` (``mass_kind``).
    * Ceiling fractions: the larger of the two hemispheres. ``ceiling_exact_frac`` and
      ``zero_frac`` are ``search_n_exact`` and ``search_n_zero`` over ``search_n``.
      ``ceiling_region`` is the region kind of the hemisphere that gave ``ceiling_frac``
      (the first of ``schemas.REGION_KINDS`` on a tie), ``none`` without a row.
    * ``occ_thick_mm``, ``ctrl_thick_mm``, ``occ_ctrl_ratio``: the rule of
      ``metrics.regional.occ_ctrl_ratio``, both hemispheres pooled.
    * ``holes_lh`` and ``holes_rh`` are those of ``subjects``, where the caller may have put
      the numbers of ``aseg.stats``. ``trunc_frac`` is the sum of the four truncation counts
      over twice the native vertices of both hemispheres; null unless all six are known.
    * ``flags``: the flags of the hemispheres, and ``no_stats`` for a subject without a row
      in ``regional``.

    Raises ValueError for a table that lacks a column or has a key twice, and for a subject
    that has clusters and no z-map.
    """
    people = _present(subjects)
    uids = people.index
    vertex = _from_vertex_summary(vertex_summary, uids)
    stats = _from_regional(regional, regions, uids)
    if clusters is not None:
        columns = (*_KEY, "hemi", "region", "rank", "area_mm2", "mass", "excess_mass", "peak_z")
        _require(clusters, "clusters", columns)
    pairs = _scored_pairs(clusters, zmaps, uids, vertex["template_id"].notna())
    scored = {uid for uid, _ in pairs}
    pairs = sorted([*pairs, *((uid, NO_REFERENCE) for uid in uids if uid not in scored)])
    key = pd.MultiIndex.from_arrays(
        [[uid for uid, _ in pairs], [reference for _, reference in pairs]], names=_KEY
    )
    uid = key.get_level_values("subject_uid")
    with_reference = key.get_level_values("reference_id").to_numpy(dtype=object) != NO_REFERENCE

    def per_subject(values: pd.Series) -> pd.Series:
        return values.reindex(uid)

    def text(values: pd.Series) -> pd.Series:
        return _texts(per_subject(values).to_numpy(dtype=object))

    def float32(values: pd.Series) -> np.ndarray:
        return _floats(per_subject(values)).astype(np.float32)

    n = len(key)
    out: dict[str, object] = {
        "subject_uid": _texts(uid.to_numpy(dtype=object)),
        "reference_id": _texts(key.get_level_values("reference_id").to_numpy(dtype=object)),
        "dataset": text(people["dataset"]),
        "template_id": text(vertex["template_id"]),
        "mass_kind": _texts(np.full(n, mass, dtype=object)),
    }
    largest = None if clusters is None else _largest_clusters(clusters, uids, mass).reindex(key)
    for name in CLUSTER_METRICS:
        values = np.where(with_reference, 0.0, np.nan)
        if largest is not None:
            found = _floats(largest[name])
            values = np.where(with_reference & ~np.isnan(found), found, values)
        out[name] = values.astype(np.float32)
    for name in _CEILING_FRACTIONS:
        out[name] = float32(vertex[name])
    out["ceiling_region"] = text(vertex["ceiling_region"])
    for name in _REGIONAL:
        out[name] = float32(stats[name])
    for name in ("holes_lh", "holes_rh"):
        out[name] = pd.array(per_subject(people[name]).to_numpy(), dtype="Int32")
    truncated = np.sum([_floats(people[name]) for name in _TRUNCATIONS], axis=0)
    n_native = _floats(vertex["n_native"])
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = np.where(n_native > 0, truncated / (2.0 * n_native), np.nan)
    out["trunc_frac"] = float32(pd.Series(fraction, index=uids))
    flags = pd.Series(
        [
            join_flags([*hemi_flags, *(() if has_stats else ("no_stats",))])
            for hemi_flags, has_stats in zip(vertex["hemi_flags"], stats["has_stats"], strict=True)
        ],
        index=uids,
        dtype=object,
    )
    out["flags"] = text(flags)
    return pd.DataFrame(out, columns=list(METRIC_COLUMNS))


# --- one scale ------------------------------------------------------------------------------


def _normal_scores(x: _Values) -> _Values:
    out = np.full(len(x), np.nan)
    has = ~np.isnan(x)
    n = int(has.sum())
    if n:
        out[has] = norm.ppf((rankdata(x[has], method="min") - 0.5) / n)
    return out


def _robust_z(x: _Values) -> _Values:
    has = ~np.isnan(x)
    values = x[has]
    if not values.size:
        return np.full(len(x), np.nan)
    with np.errstate(invalid="ignore", over="ignore"):
        median = np.median(values)
        deviation = values - median
        largest = np.max(np.abs(deviation))
        low, high = np.percentile(values, [25.0, 75.0])
        candidates = (
            1.4826 * np.median(np.abs(deviation)),
            (high - low) / 1.349,
            np.std(values, ddof=1) if values.size > 1 else np.nan,
        )
        scale = next((c for c in candidates if np.isfinite(c) and c > 1e-6 * largest), 1.0)
        out = np.full(len(x), np.nan)
        out[has] = deviation / scale
    return out


def _quantile_scores(x: _Values, grid: Sequence[float]) -> _Values:
    points = np.asarray(grid, dtype=np.float64)
    if points.shape != (_GRID_POINTS,):
        raise ValueError(f"a percentile grid has {_GRID_POINTS} values, got {points.size}")
    if not np.isfinite(points).all() or (np.diff(points) < 0).any():
        raise ValueError("a percentile grid holds finite values that do not decrease")
    out = np.full(len(x), np.nan)
    has = ~np.isnan(x)
    values = x[has]
    # the first grid point that is not below the value: equal points share the lowest
    # percentile, as tied values share the lowest rank in normal_scores
    upper = np.searchsorted(points, values, side="left")
    inside = (upper > 0) & (upper < _GRID_POINTS)
    percentile = np.where(upper == 0, 0.0, 100.0)
    below, above = points[upper[inside] - 1], points[upper[inside]]
    percentile[inside] = upper[inside] - 1 + (values[inside] - below) / (above - below)
    out[has] = norm.ppf(np.clip(percentile, 0.5, 99.5) / 100.0)
    return out


def _percentile_rank(x: _Values) -> _Values:
    """``(r - 1) / (n - 1)`` with ``r`` the minimum rank among the ``n`` values; 0 for one."""
    out = np.full(len(x), np.nan)
    has = ~np.isnan(x)
    n = int(has.sum())
    if n == 1:
        out[has] = 0.0
    elif n:
        out[has] = (rankdata(x[has], method="min") - 1.0) / (n - 1.0)
    return out


def _as_series(values: pd.Series, scores: _Values) -> pd.Series:
    return pd.Series(scores, index=values.index, name=values.name, dtype=np.float64)


def normal_scores(values: pd.Series) -> pd.Series:
    """Rank-based normal scores: ``norm.ppf((r - 0.5) / n)`` with ``r`` the minimum rank of
    the value among the ``n`` values that are not NaN (1 = smallest; tied values share the
    lowest rank of the tie). NaN stays NaN; one value gives 0.

    Bounded, and without a scale: 26 zeros and the values 10, 50, 200, 1000 give -2.13 for
    the zeros and 1.19, 1.38, 1.64, 2.13.
    """
    return _as_series(values, _normal_scores(_floats(values)))


def robust_z(values: pd.Series) -> pd.Series:
    """``(x - median) / scale``, with the first usable scale of ``1.4826 * MAD``,
    ``IQR / 1.349``, the standard deviation (ddof 1) and 1. Usable: finite and above
    ``1e-6 * max(|x - median|)``. NaN stays NaN.

    On a metric that is 0 for most subjects the scale is the standard deviation, which the
    outliers inflate: 26 zeros and the values 10, 50, 200, 1000 give 0.05, 0.27, 1.08, 5.41.
    """
    return _as_series(values, _robust_z(_floats(values)))


def quantile_scores(values: pd.Series, grid: Sequence[float]) -> pd.Series:
    """Normal scores against a reference population: ``grid`` holds its 0th to 100th
    percentiles of the component (101 values that do not decrease); the value's percentile
    is interpolated in it, clipped to [0.5, 99.5] and passed to ``norm.ppf``.

    A value equal to several grid points has the lowest of their percentiles, as tied values
    share the lowest rank in :func:`normal_scores`; a value below the grid has percentile 0
    and one above it 100. NaN stays NaN. Raises ValueError for another grid.
    """
    return _as_series(values, _quantile_scores(_floats(values), grid))


def _within(
    values: _Values, labels: np.ndarray | None, score: Callable[[_Values], _Values]
) -> _Values:
    """``score`` over all values, or over the values of each label; a value without a label
    (None) keeps its score among all."""
    out = score(values)
    if labels is None:
        return out
    named = pd.Series(labels, dtype=object).dropna()
    for positions in named.groupby(named).indices.values():
        rows = named.index.to_numpy()[positions]
        out[rows] = score(values[rows])
    return out


def _one_per_subject(frame: pd.DataFrame, name: str) -> tuple[np.ndarray, np.ndarray]:
    """``(first, position)``: the row of each subject's first occurrence, and for every row
    the position of its subject among those. Raises when a subject-level metric differs
    between the rows of one subject."""
    position, _ = pd.factorize(frame["subject_uid"])
    _, first = np.unique(position, return_index=True)
    values = _floats(frame[name])
    of_first = values[first][position]
    if not ((values == of_first) | (np.isnan(values) & np.isnan(of_first))).all():
        raise ValueError(f"{name} differs between the rows of one subject")
    return first, position


def _finish(frame: pd.DataFrame, scores: Mapping[str, _Values], kind: str) -> pd.DataFrame:
    """``frame`` with the ``std_*`` columns, the composite and what goes with it. The maximum
    and its component are taken on the float32 values that are stored."""
    out = frame.copy()
    n = len(out)
    names = list(scores)
    matrix = np.full((n, len(names)), np.nan, dtype=np.float32)
    for column, name in enumerate(names):
        matrix[:, column] = scores[name].astype(np.float32)
    for name in STD_COLUMNS:
        out[name] = np.full(n, np.nan, dtype=np.float32)
    for column, name in enumerate(names):
        out[f"std_{name}"] = matrix[:, column]
    has = ~np.isnan(matrix)
    count = has.sum(axis=1)
    best = np.where(has, matrix, -np.inf).argmax(axis=1) if names else np.zeros(n, dtype=np.intp)
    scored = count > 0
    value = np.full(n, np.nan, dtype=np.float32)
    if names:
        value[scored] = matrix[np.arange(n), best][scored]
    out["composite"] = value
    component = np.array([*names, None], dtype=object)[np.where(scored, best, len(names))]
    out["composite_component"] = _texts(component, out.index)
    out["n_components"] = count.astype(np.int8)
    out["score_kind"] = _texts(np.full(n, kind, dtype=object), out.index)
    out["flags"] = _texts(
        [
            join_flags([*split_flags(text or ""), *((_PARTIAL,) if partial else ())])
            for text, partial in zip(out["flags"], count < len(names), strict=True)
        ],
        out.index,
    )
    return out


def composite(
    metrics: pd.DataFrame,
    strata: pd.Series,
    config: ScoringConfig,
    *,
    norms: Mapping[str, Sequence[float]] | None = None,
) -> pd.DataFrame:
    """``metrics`` with ``std_<component>`` for each component of ``config``, ``composite``
    (the maximum over the components that are not NaN), ``composite_component`` (the first
    in config order that attains it), ``n_components`` and ``score_kind`` =
    ``composite_z``.

    ``metrics`` are rows of :func:`subject_metrics`: pass all rows of the run in one call,
    because the population is taken from them. Each component is standardised over
    ``config.population``: the whole run, or the subject's stratum (``strata``:
    ``subject_uid`` -> stratum name or None; a subject without a stratum is scored among all
    subjects). A cluster metric (:data:`CLUSTER_METRICS`) is standardised among the rows of
    one ``reference_id``. Every other component is one number per subject, and is
    standardised among the subjects, each counted once, whatever their rows: a subject
    without a z-map is scored on its ceiling fraction and its regional ratio against
    everyone else, not against the other subjects without a z-map.

    With ``norms`` (component -> the 101 percentiles of a reference population) the scores
    are :func:`quantile_scores` and no population of the run is used; ``norms`` must hold
    every component.

    A row with fewer components than configured gets the flag ``partial_score``: the maximum
    over fewer scores is lower on average. A row without any has no composite.

    Raises ValueError for a component that is not a numeric column of ``metrics``, a
    component named twice, a ``(subject_uid, reference_id)`` twice, a subject twice in
    ``strata``, and a component that ``norms`` lacks.
    """
    components = list(config.components)
    _require(metrics, "metrics", (*_KEY, "flags", *components))
    if len(set(components)) != len(components):
        raise ValueError("a component is named twice")
    if metrics.duplicated(_KEY).any():
        raise ValueError("metrics: a subject twice for one reference")
    if norms is not None:
        lacking = [name for name in components if name not in norms]
        if lacking:
            raise ValueError(f"norms hold no percentiles for {', '.join(lacking)}")
    stratum = None
    if norms is None and config.population == "stratum":
        if strata.index.duplicated().any():
            raise ValueError("strata: a subject twice")
        stratum = strata.reindex(metrics["subject_uid"]).to_numpy(dtype=object)
    plain = _normal_scores if config.standardize == "normal_score" else _robust_z
    reference = metrics["reference_id"].to_numpy(dtype=object)

    scores: dict[str, _Values] = {}
    for name in components:
        try:
            values = _floats(metrics[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} is not a numeric column of the metrics") from exc
        if norms is not None:
            scores[name] = _quantile_scores(values, norms[name])
        elif name in CLUSTER_METRICS:
            out = np.full(len(values), np.nan)
            for rows in pd.Series(reference, dtype=object).groupby(reference).indices.values():
                labels = None if stratum is None else stratum[rows]
                out[rows] = _within(values[rows], labels, plain)
            scores[name] = out
        else:
            first, position = _one_per_subject(metrics, name)
            labels = None if stratum is None else stratum[first]
            scores[name] = _within(values[first], labels, plain)[position]
    return _finish(metrics, scores, _COMPOSITE_Z)


def fallback_rank(metrics: pd.DataFrame) -> pd.DataFrame:
    """The ranking of decision D16: per component of :data:`FALLBACK_COMPONENTS` the
    percentile ``(r - 1) / (n - 1)`` with ``r`` the minimum rank among the ``n`` subjects
    with a value (0 when ``n`` is 1); ``composite`` is the larger of the two,
    ``score_kind`` is ``fallback_rank`` and ``reference_id`` is ``schemas.NO_REFERENCE``.

    The percentiles are in ``std_ceiling_frac`` and ``std_occ_ctrl_ratio``. With no vertex
    at the ceiling anywhere, the common case, every ceiling percentile is 0 and the order
    follows the ratio. A subject with one of the two gets the flag ``partial_score``.

    ``metrics`` are rows of :func:`subject_metrics`, one per subject (ValueError otherwise).
    """
    _require(metrics, "metrics", ("subject_uid", "flags", *FALLBACK_COMPONENTS))
    if metrics["subject_uid"].duplicated().any():
        raise ValueError("metrics: a subject twice; the ranking of D16 has one row per subject")
    scores = {name: _percentile_rank(_floats(metrics[name])) for name in FALLBACK_COMPONENTS}
    out = _finish(metrics, scores, _FALLBACK_RANK)
    out["reference_id"] = _texts(np.full(len(out), NO_REFERENCE, dtype=object), out.index)
    return out


def add_rank(scores: pd.DataFrame) -> pd.DataFrame:
    """``scores`` with ``rank`` within each ``reference_id``: 1 is the highest composite;
    ties go by the mean of the ``std_*`` scores that are not NaN, descending, then by
    ``subject_uid``; subjects without a composite come after all others.

    The composite is compared as float32, the type it is stored in. The ranks of one
    reference are 1 to the number of its rows, each once.
    """
    _require(scores, "scores", (*_KEY, "composite"))
    std = [str(name) for name in scores.columns if str(name).startswith("std_")]
    n = len(scores)
    values = _floats(scores["composite"]).astype(np.float32)
    matrix = np.full((n, len(std)), np.nan)
    for column, name in enumerate(std):
        matrix[:, column] = _floats(scores[name])
    has = ~np.isnan(matrix)
    count = has.sum(axis=1)
    mean = np.full(n, np.nan)
    mean[count > 0] = np.where(has, matrix, 0.0).sum(axis=1)[count > 0] / count[count > 0]
    order = pd.DataFrame(
        {
            "reference_id": scores["reference_id"].to_numpy(dtype=object),
            "unscored": np.isnan(values),
            "composite": values,
            "mean": mean,
            "subject_uid": scores["subject_uid"].to_numpy(dtype=object),
        }
    ).sort_values(
        ["reference_id", "unscored", "composite", "mean", "subject_uid"],
        ascending=[True, True, False, False, True],
        na_position="last",
    )
    rank = np.empty(n, dtype=np.int32)
    rank[order.index.to_numpy()] = order.groupby("reference_id").cumcount().to_numpy() + 1
    out = scores.copy()
    out["rank"] = rank
    return out
