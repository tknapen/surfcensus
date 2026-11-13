"""metrics/scoring: subject metrics from the tables, one scale for the components, the
composite, the ranking of decision D16 and the rank.

Tables are built here from rows and passed through ``store.write_table`` and
``store.read_table``, so the frames have the types a run hands out. The two cohort tests at
the end need tables for 32 and for 5 synthetic subjects. The stages that make them in a run
belong to other packages, so a stand-in below computes them from the cohort's files with
numpy and scipy: nearest-neighbour maps on a 2,562-vertex mesh, a per-vertex median and MAD,
connected components. It follows docs/metrics.md in outline only, and what it tests is the
scoring of tables that have the structure of a real cohort.
"""

from __future__ import annotations

import math
import tempfile
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Literal

import nibabel.freesurfer as fs
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.stats import norm
from synth_template import (  # pyrefly: ignore[missing-import]
    DESIKAN_NAMES,
    expected_regions,
    template_mesh,
)
from synth_tierb import SyntheticCohort  # pyrefly: ignore[missing-import]

from surfcensus import schemas, store
from surfcensus.config import RegionConfig, ScoringConfig
from surfcensus.metrics.scoring import (
    CLUSTER_METRICS,
    FALLBACK_COMPONENTS,
    MASS_REGIONS,
    METRIC_COLUMNS,
    STD_COLUMNS,
    add_primary_rank,
    add_rank,
    composite,
    fallback_rank,
    normal_scores,
    quantile_scores,
    robust_z,
    subject_metrics,
)
from surfcensus.sources.base import HEMIS

REGIONS = RegionConfig()
CONFIG = ScoringConfig()
REF = "set-0123456789ab"
OTHER = "shipped-1"
NONE = schemas.NO_REFERENCE
DATASET = "D-00000001"
ZEROS_CASE = [0.0] * 26 + [10.0, 50.0, 200.0, 1000.0]
"""The case of section 3 of the contract: most subjects at zero, four outliers."""

DEFAULTS: dict[str, dict[str, object]] = {
    "vertex_summary": {
        "n_native": 1000,
        "thickness_max_mm": 3.5,
        "template_id": "aaaaaaaaaaaa",
        "n_fallback": 0,
        "n_missing_cortex": 0,
        "global_median_mm": 2.5,
        "region_kind": "template",
        "search_n": 200,
        "search_n_exact": 0,
        "search_n_near": 0,
        "search_n_zero": 0,
        "control_n": 300,
        "control_n_exact": 0,
        "control_n_near": 0,
        "control_n_zero": 0,
        "ceiling_frac": 0.0,
        "control_ceiling_frac": 0.0,
        "annot_search_n": 150,
        "annot_search_n_near": 0,
        "annot_search_n_zero": 0,
        "ceiling_frac_annot": 0.0,
        "flags": "",
    },
    "clusters": {
        "reference_id": REF,
        "hemi": "lh",
        "region": "search",
        "rank": 1,
        "n_vertices": 10,
        "area_mm2": 10.0,
        "mass": 40.0,
        "excess_mass": 10.0,
        "peak_z": 5.0,
        "peak_vertex": 0,
        "centroid_vertex": 0,
    },
    "regional": {"num_vert": 100, "surf_area_mm2": 100, "gray_vol_mm3": 250, "thick_std_mm": 0.5},
}


def table(name: str, rows: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    """Rows of a metric table as ``store.read_table`` hands them out: the defaults above
    filled in, written and read back. No rows give the empty table of the schema."""
    schema = schemas.SCHEMAS[name]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{name}.parquet"
        if rows:
            frame = pd.DataFrame(
                [{**DEFAULTS.get(name, {}), **row} for row in rows], columns=schema.names
            )
            store.write_table(path, frame, schema, sort_by=schemas.KEYS[name])
        return store.read_table(path, schema)


def subjects(*uids: str, **columns: Mapping[str, object]) -> pd.DataFrame:
    """The columns of ``PublicCatalog.subjects()`` that scoring reads, in the catalog's types.
    ``columns``: column -> {uid: value} for what differs from the default."""
    defaults: dict[str, object] = {"dataset": DATASET, "present": True}
    counts = ("holes_lh", "holes_rh", "trunc_lh_1", "trunc_lh_2", "trunc_rh_1", "trunc_rh_2")
    data: dict[str, object] = {"subject_uid": pd.Series(list(uids), dtype=object)}
    for name in ("dataset", "present", *counts):
        values = [columns.get(name, {}).get(uid, defaults.get(name)) for uid in uids]
        if name == "dataset":
            data[name] = pd.Series(values, dtype=object)
        elif name == "present":
            data[name] = pd.Series(values, dtype=bool)
        else:
            data[name] = pd.Series(pd.array(values, dtype="Int64"))
    return pd.DataFrame(data)


def both_hemis(uid: str, **values: object) -> list[dict[str, object]]:
    return [{"subject_uid": uid, "hemi": hemi, **values} for hemi in HEMIS]


def parcels(uid: str, thickness: Mapping[str, float], hemi: str = "lh") -> list[dict[str, object]]:
    return [
        {"subject_uid": uid, "hemi": hemi, "parcel": parcel, "thick_avg_mm": value}
        for parcel, value in thickness.items()
    ]


def metrics_of(
    vertex_summary: Sequence[Mapping[str, object]],
    clusters: Sequence[Mapping[str, object]] | None,
    regional: Sequence[Mapping[str, object]],
    people: pd.DataFrame,
    *,
    mass: Literal["excess", "sum_z"] = "excess",
    zmaps: Mapping[str, Collection[str]] | None = None,
) -> pd.DataFrame:
    """``subject_metrics`` on tables made of rows, indexed by subject and reference."""
    frame = subject_metrics(
        table("vertex_summary", vertex_summary),
        None if clusters is None else table("clusters", clusters),
        table("regional", regional),
        people,
        REGIONS,
        mass=mass,
        zmaps=zmaps,
    )
    return frame.set_index(["subject_uid", "reference_id"])


def components_frame(
    values: Mapping[str, Sequence[float | None]],
    *,
    uids: Sequence[str] | None = None,
    reference: str | Sequence[str] = REF,
    flags: str = "",
) -> pd.DataFrame:
    """Rows as ``subject_metrics`` returns them, with only what ``composite`` reads: the key,
    the flags and the four default components (float32, NaN unless given)."""
    n = len(next(iter(values.values())))
    names = list(uids) if uids is not None else [f"s{k:03d}" for k in range(n)]
    references = [reference] * n if isinstance(reference, str) else list(reference)
    data: dict[str, object] = {
        "subject_uid": pd.Series(names, dtype=object),
        "reference_id": pd.Series(references, dtype=object),
        "flags": pd.Series([flags] * n, dtype=object),
    }
    for name in dict.fromkeys([*CONFIG.components, *values]):
        column = values.get(name, [None] * n)
        data[name] = np.array([np.nan if v is None else v for v in column], dtype=np.float32)
    return pd.DataFrame(data)


NO_STRATA = pd.Series([], dtype=object)


def score(p: float) -> float:
    return float(norm.ppf(p))


# --- normal scores --------------------------------------------------------------------------


def test_normal_scores_of_a_metric_that_is_zero_for_most_subjects() -> None:
    scores = normal_scores(pd.Series(ZEROS_CASE))
    assert np.round(scores.to_numpy()[[0, 26, 27, 28, 29]], 2).tolist() == [
        -2.13,
        1.19,
        1.38,
        1.64,
        2.13,
    ]
    assert (scores.iloc[:26] == scores.iloc[0]).all()
    assert scores.iloc[0] == pytest.approx(score(0.5 / 30))
    assert scores.iloc[29] == pytest.approx(score(29.5 / 30))


def test_tied_values_share_the_lowest_rank_of_the_tie() -> None:
    scores = normal_scores(pd.Series([3.0, 1.0, 3.0, 2.0]))
    expected = [score(2.5 / 4), score(0.5 / 4), score(2.5 / 4), score(1.5 / 4)]
    assert scores.to_numpy() == pytest.approx(expected)
    # the average rank of the tie would be 3.5
    assert scores.iloc[0] != pytest.approx(score(3.0 / 4))


def test_normal_scores_of_one_value_of_missing_values_and_of_none() -> None:
    assert normal_scores(pd.Series([7.5])).tolist() == [0.0]
    scores = normal_scores(pd.Series([np.nan, 2.0, np.nan, 1.0]))
    assert np.isnan(scores.iloc[[0, 2]]).all()
    # n counts the values that are there: two, not four
    assert scores.iloc[[1, 3]].to_numpy() == pytest.approx([score(0.75), score(0.25)])
    assert normal_scores(pd.Series([np.nan, np.nan])).isna().all()
    empty = normal_scores(pd.Series([], dtype=np.float64))
    assert len(empty) == 0 and empty.dtype == np.float64


def test_normal_scores_keep_index_and_name_and_read_nullable_columns() -> None:
    values = pd.Series(pd.array([4, None, 1], dtype="Int32"), index=["x", "y", "z"], name="holes")
    scores = normal_scores(values)
    assert scores.index.tolist() == ["x", "y", "z"] and scores.name == "holes"
    assert scores.dtype == np.float64
    assert scores["x"] == pytest.approx(score(0.75)) and math.isnan(scores["y"])
    as_objects = pd.Series([4.0, None, 1.0], dtype=object)
    assert normal_scores(as_objects).to_numpy() == pytest.approx(scores.to_numpy(), nan_ok=True)


@settings(max_examples=200, deadline=None)
@given(
    st.lists(
        st.one_of(st.none(), st.integers(-5, 5), st.floats(-1e6, 1e6)), min_size=1, max_size=40
    )
)
def test_normal_scores_follow_the_order_of_the_values_and_nothing_else(
    values: list[float | None],
) -> None:
    x = np.array([np.nan if v is None else float(v) for v in values])
    scores = normal_scores(pd.Series(x)).to_numpy()
    has = ~np.isnan(x)
    assert (np.isnan(scores) == ~has).all()
    n = int(has.sum())
    if not n:
        return
    v, s = x[has], scores[has]
    assert ((v[:, None] < v[None, :]) == (s[:, None] < s[None, :])).all()
    assert s.min() == pytest.approx(score(0.5 / n))
    assert s.max() <= score((n - 0.5) / n) + 1e-12
    # any increasing map of the values gives the same scores (this one is exact in floats)
    stretched = np.where(x > 0, 4.0 * x, x)
    assert normal_scores(pd.Series(stretched)).to_numpy()[has].tolist() == s.tolist()


# --- robust z -------------------------------------------------------------------------------


def test_robust_z_uses_the_mad_when_it_is_usable() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])
    assert robust_z(values).to_numpy() == pytest.approx((values.to_numpy() - 3.0) / 1.4826)


def test_robust_z_falls_to_the_iqr_when_the_mad_is_zero() -> None:
    values = pd.Series([0.0] * 6 + [1.0] * 3)  # median 0, MAD 0, quartiles 0 and 1
    assert robust_z(values).to_numpy() == pytest.approx(values.to_numpy() * 1.349)


def test_robust_z_falls_to_the_sd_when_mad_and_iqr_are_zero() -> None:
    z = robust_z(pd.Series(ZEROS_CASE)).to_numpy()
    assert np.round(z[26:], 2).tolist() == [0.05, 0.27, 1.08, 5.41]
    assert z[26:] == pytest.approx(np.array(ZEROS_CASE[26:]) / np.std(ZEROS_CASE, ddof=1))
    assert (z[:26] == 0).all()


def test_robust_z_divides_by_one_when_nothing_varies() -> None:
    assert robust_z(pd.Series([2.5, 2.5, 2.5])).tolist() == [0.0, 0.0, 0.0]
    assert robust_z(pd.Series([2.5])).tolist() == [0.0]  # the SD of one value is not finite


def test_a_scale_far_below_the_largest_deviation_is_not_usable() -> None:
    # MAD 1e-9 against a deviation of 1000: below 1e-6 of it, so the IQR is used
    values = np.array([0.0, 1e-9, 2e-9, 1e3])
    low, high = np.percentile(values, [25, 75])
    z = robust_z(pd.Series(values)).to_numpy()
    assert z == pytest.approx((values - np.median(values)) * 1.349 / (high - low))
    assert z[3] == pytest.approx(1.349 * 1e3 / 250, rel=1e-6)
    # just inside the bound the MAD is the scale: MAD 0.002 against a deviation of 1000
    usable = np.array([0.0, 0.002, 0.004, 1e3 + 0.003])
    mad = np.median(np.abs(usable - np.median(usable)))
    assert 1.4826 * mad > 1e-6 * 1e3
    assert robust_z(pd.Series(usable)).to_numpy() == pytest.approx(
        (usable - np.median(usable)) / (1.4826 * mad)
    )


def test_robust_z_leaves_missing_values_missing() -> None:
    z = robust_z(pd.Series([np.nan, 1.0, 2.0, 3.0, 4.0, 100.0], name="m"))
    assert math.isnan(z.iloc[0]) and z.name == "m"
    assert z.iloc[1:].to_numpy() == pytest.approx((np.array([1, 2, 3, 4, 100]) - 3.0) / 1.4826)
    assert robust_z(pd.Series([np.nan])).isna().all()
    assert len(robust_z(pd.Series([], dtype=np.float64))) == 0


# --- quantile scores ------------------------------------------------------------------------

UNIFORM_GRID = tuple(float(k) for k in range(101))


def test_quantile_scores_against_a_grid() -> None:
    values = pd.Series([50.0, 97.5, 2.5, 20.25, np.nan], name="ceiling_frac")
    scores = quantile_scores(values, UNIFORM_GRID)
    assert scores.iloc[:4].to_numpy() == pytest.approx(
        [0.0, score(0.975), score(0.025), score(0.2025)]
    )
    assert math.isnan(scores.iloc[4]) and scores.name == "ceiling_frac"
    assert scores.iloc[1] == pytest.approx(1.96, abs=0.001)


def test_quantile_scores_are_clipped_at_half_a_percent() -> None:
    scores = quantile_scores(pd.Series([-5.0, 0.0, 0.2, 99.9, 100.0, 1e9]), UNIFORM_GRID)
    low, high = score(0.005), score(0.995)
    assert scores.to_numpy() == pytest.approx([low, low, low, high, high, high])
    assert high == pytest.approx(2.5758, abs=1e-4)


def test_quantile_scores_interpolate_between_unequal_grid_points() -> None:
    grid = tuple(float(k) ** 2 for k in range(101))  # the kth percentile is k squared
    scores = quantile_scores(pd.Series([900.0, 930.5, 961.0]), grid)
    assert scores.to_numpy() == pytest.approx([score(0.30), score(0.305), score(0.31)])


def test_a_value_on_equal_grid_points_takes_the_lowest_percentile() -> None:
    # zero up to the 60th percentile, then rising by one per percentile
    grid = tuple([0.0] * 61 + [float(k) for k in range(1, 41)])
    scores = quantile_scores(pd.Series([0.0, 1e-9, 0.5, 1.0, 40.0]), grid)
    assert scores.iloc[0] == pytest.approx(score(0.005))  # percentile 0, clipped
    assert scores.iloc[1] == pytest.approx(score(0.60), abs=1e-6)  # just above every zero
    assert scores.iloc[2:].to_numpy() == pytest.approx([score(0.605), score(0.61), score(0.995)])


def test_quantile_scores_agree_with_normal_scores_on_the_population_of_the_grid() -> None:
    values = np.random.default_rng(3).gamma(2.0, size=4001)
    grid = tuple(float(v) for v in np.percentile(values, np.arange(101)))
    inner = (values > grid[2]) & (values < grid[98])
    against_grid = quantile_scores(pd.Series(values), grid).to_numpy()[inner]
    # 40 values share a percentile step, and the grid is straight between its points
    assert against_grid == pytest.approx(
        normal_scores(pd.Series(values)).to_numpy()[inner], abs=0.05
    )


@pytest.mark.parametrize(
    "grid",
    [
        UNIFORM_GRID[:100],
        (*UNIFORM_GRID, 101.0),
        (*UNIFORM_GRID[:50], 48.0, *UNIFORM_GRID[51:]),
        (*UNIFORM_GRID[:100], float("nan")),
        (*UNIFORM_GRID[:100], float("inf")),
    ],
    ids=["100 points", "102 points", "decreasing", "nan", "infinite"],
)
def test_a_grid_that_is_not_101_rising_percentiles_is_refused(grid: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="percentile grid"):
        quantile_scores(pd.Series([1.0]), grid)


# --- subject metrics ------------------------------------------------------------------------


def test_the_names_follow_the_scores_schema() -> None:
    names = schemas.SCORES.names
    assert [name for name in names if name in METRIC_COLUMNS] == list(METRIC_COLUMNS)
    assert set(CLUSTER_METRICS) < set(METRIC_COLUMNS)
    assert [f"mass_{region}" for region in MASS_REGIONS] == [
        name for name in names if name.startswith("mass_") and name != "mass_kind"
    ]
    assert STD_COLUMNS == tuple(f"std_{name}" for name in CONFIG.components)
    assert set(FALLBACK_COMPONENTS) <= set(CONFIG.components)
    # the pipeline's
    later = {"primary", "primary_rank", "stratum", "n_reference", "score_config_hash"}
    ours = {*METRIC_COLUMNS, *STD_COLUMNS, "composite", "composite_component", "n_components"}
    assert set(names) == ours | {"score_kind", "rank"} | later


def test_mass_regions_are_the_regions_of_the_template() -> None:
    from surfcensus.templates import REGION_NAMES

    assert MASS_REGIONS == REGION_NAMES


def test_cluster_metrics_are_those_of_the_largest_cluster_of_the_larger_hemisphere() -> None:
    clusters = [
        {
            "subject_uid": "a",
            "hemi": "lh",
            "area_mm2": 30,
            "excess_mass": 4,
            "mass": 40,
            "peak_z": 6,
        },
        {
            "subject_uid": "a",
            "hemi": "rh",
            "area_mm2": 20,
            "excess_mass": 9,
            "mass": 30,
            "peak_z": 7,
        },
        # a second cluster, smaller in area and heavier: not the largest, so not reported
        {"subject_uid": "a", "hemi": "lh", "rank": 2, "area_mm2": 5, "excess_mass": 99, "mass": 99},
        {"subject_uid": "a", "hemi": "lh", "region": "medial", "excess_mass": 3, "mass": 33},
        {"subject_uid": "a", "hemi": "rh", "region": "inferior", "excess_mass": 2, "mass": 22},
        {"subject_uid": "a", "hemi": "rh", "region": "control", "excess_mass": 1, "mass": 11},
    ]
    vertex = both_hemis("a") + both_hemis("b")
    rows = metrics_of(vertex, clusters, [], subjects("a", "b"))
    a = rows.loc[("a", REF)]
    assert a["mass_kind"] == "excess"
    assert a[[f"mass_{region}" for region in MASS_REGIONS]].tolist() == [9, 3, 2, 0, 1]
    # each metric takes the larger hemisphere on its own
    assert a["area_search_mm2"] == 30 and a["peak_z_search"] == 7

    by_sum = metrics_of(vertex, clusters, [], subjects("a", "b"), mass="sum_z").loc[("a", REF)]
    assert by_sum["mass_kind"] == "sum_z"
    assert by_sum[[f"mass_{region}" for region in MASS_REGIONS]].tolist() == [40, 33, 22, 0, 11]


def test_a_z_map_without_a_cluster_gives_zero_and_no_z_map_gives_null() -> None:
    clusters = [{"subject_uid": "a"}]
    vertex = [
        *both_hemis("a"),
        *both_hemis("b"),  # a map, and no cluster
        *both_hemis("c", template_id=None, region_kind="annot"),  # no map
    ]
    rows = metrics_of(vertex, clusters, [], subjects("a", "b", "c", "d"))
    assert rows.index.tolist() == [("a", REF), ("b", REF), ("c", NONE), ("d", NONE)]
    names = list(CLUSTER_METRICS)
    assert rows.loc[("b", REF), names].tolist() == [0.0] * len(names)
    assert rows.loc[[("c", NONE), ("d", NONE)], names].isna().all().all()
    assert rows.loc[("a", REF), "mass_search"] == 10
    assert rows["template_id"].tolist() == ["aaaaaaaaaaaa", "aaaaaaaaaaaa", None, None]


def test_one_usable_hemisphere_is_a_z_map() -> None:
    vertex = [
        {"subject_uid": "a", "hemi": "lh", "template_id": None, "flags": "low_coverage"},
        {"subject_uid": "a", "hemi": "rh"},
        *both_hemis("b"),
    ]
    rows = metrics_of(vertex, [{"subject_uid": "b"}], [], subjects("a", "b"))
    assert rows.index.tolist() == [("a", REF), ("b", REF)]
    assert rows.loc[("a", REF), "mass_search"] == 0
    assert rows.loc[("a", REF), "template_id"] == "aaaaaaaaaaaa"
    assert rows.loc[("a", REF), "flags"] == "low_coverage,no_stats"


def test_without_a_cluster_table_every_row_is_without_a_reference() -> None:
    rows = metrics_of(both_hemis("a") + both_hemis("b"), None, [], subjects("a", "b"))
    assert rows.index.tolist() == [("a", NONE), ("b", NONE)]
    assert rows[list(CLUSTER_METRICS)].isna().all().all()
    assert rows["mass_kind"].tolist() == ["excess", "excess"]
    # the template is still named: the ceiling region was placed with it
    assert rows["template_id"].tolist() == ["aaaaaaaaaaaa", "aaaaaaaaaaaa"]


def test_a_subject_has_a_row_for_each_reference() -> None:
    clusters = [
        {"subject_uid": "a", "excess_mass": 4},
        {"subject_uid": "a", "reference_id": OTHER, "excess_mass": 6},
        {"subject_uid": "b", "reference_id": OTHER, "excess_mass": 2},
    ]
    rows = metrics_of(both_hemis("a") + both_hemis("b"), clusters, [], subjects("a", "b"))
    assert rows.index.tolist() == [("a", REF), ("a", OTHER), ("b", REF), ("b", OTHER)]
    assert rows["mass_search"].tolist() == [4, 6, 0, 2]


def test_zmaps_say_which_subject_is_scored_against_which_reference() -> None:
    vertex = both_hemis("a") + both_hemis("b") + both_hemis("c")
    people = subjects("a", "b", "c")
    clusters = [{"subject_uid": "a", "reference_id": OTHER}]
    # the shipped reference has a stratum for a only; b and c are scored within the set,
    # where nobody has a cluster; "gone" is no longer a subject of the run
    zmaps = {OTHER: ["a", "gone"], REF: ("a", "b", "c")}
    rows = metrics_of(vertex, clusters, [], people, zmaps=zmaps)
    assert rows.index.tolist() == [("a", REF), ("a", OTHER), ("b", REF), ("c", REF)]
    assert rows["mass_search"].tolist() == [0, 10, 0, 0]

    # without zmaps the same tables read as: everyone against the one reference with rows
    guessed = metrics_of(vertex, clusters, [], people)
    assert guessed.index.tolist() == [("a", OTHER), ("b", OTHER), ("c", OTHER)]

    # a cluster table without rows names no reference; zmaps still do
    assert metrics_of(vertex, [], [], people).index.get_level_values(1).tolist() == [NONE] * 3
    named = metrics_of(vertex, [], [], people, zmaps={REF: ["b"]})
    assert named.index.tolist() == [("a", NONE), ("b", REF), ("c", NONE)]
    assert named.loc[("b", REF), "mass_medial"] == 0 and math.isnan(
        named.loc[("a", NONE), "mass_medial"]
    )


def test_clusters_of_a_subject_without_a_z_map_are_refused() -> None:
    vertex = both_hemis("a") + both_hemis("b", template_id=None)
    people = subjects("a", "b")
    with pytest.raises(ValueError, match=f"clusters and no z-map, the first is b with {REF}"):
        metrics_of(vertex, [{"subject_uid": "b"}], [], people)
    with pytest.raises(ValueError, match=f"clusters and no z-map, the first is a with {REF}"):
        metrics_of(vertex, [{"subject_uid": "a"}], [], people, zmaps={REF: ["b"]})
    with pytest.raises(ValueError, match="no cluster table"):
        metrics_of(vertex, None, [], people, zmaps={REF: ["a"]})
    with pytest.raises(ValueError, match="without a reference"):
        metrics_of(vertex, [], [], people, zmaps={NONE: ["a"]})
    # clusters of a subject that left the run are dropped, not refused
    rows = metrics_of(vertex, [{"subject_uid": "left"}, {"subject_uid": "a"}], [], people)
    assert rows.index.tolist() == [("a", REF), ("b", NONE)]


def test_ceiling_fractions_are_those_of_the_larger_hemisphere() -> None:
    vertex = [
        {
            "subject_uid": "a",
            "hemi": "lh",
            "search_n": 200,
            "search_n_exact": 2,
            "search_n_near": 6,
            "search_n_zero": 1,
            "ceiling_frac": 7 / 200,
            "control_ceiling_frac": 0.001,
            "ceiling_frac_annot": 0.02,
        },
        {
            "subject_uid": "a",
            "hemi": "rh",
            "search_n": 100,
            "search_n_exact": 3,
            "search_n_near": 3,
            "search_n_zero": 0,
            "ceiling_frac": 3 / 100,
            "control_ceiling_frac": 0.004,
            "ceiling_frac_annot": 0.05,
        },
    ]
    a = metrics_of(vertex, None, [], subjects("a")).loc[("a", NONE)]
    assert a["ceiling_frac"] == np.float32(7 / 200)
    assert a["control_ceiling_frac"] == np.float32(0.004)
    assert a["ceiling_frac_annot"] == np.float32(0.05)
    assert a["ceiling_exact_frac"] == np.float32(3 / 100)  # rh: 3 of 100 beats 2 of 200
    assert a["zero_frac"] == np.float32(1 / 200)


def test_a_hemisphere_without_a_region_gives_no_fraction() -> None:
    empty = {
        "region_kind": "none",
        "template_id": None,
        "search_n": None,
        "search_n_exact": None,
        "search_n_near": None,
        "search_n_zero": None,
        "ceiling_frac": None,
        "control_ceiling_frac": None,
        "ceiling_frac_annot": None,
    }
    vertex = [
        *both_hemis("a", **empty),
        {"subject_uid": "b", "hemi": "lh", **empty},
        {"subject_uid": "b", "hemi": "rh", "search_n": 50, "search_n_zero": 5, "ceiling_frac": 0.1},
        # a region without a vertex: counts of 0 and no fraction
        *both_hemis("c", search_n=0, ceiling_frac=None),
    ]
    rows = metrics_of(vertex, None, [], subjects("a", "b", "c", "d"))
    fractions = ["ceiling_frac", "control_ceiling_frac", "ceiling_exact_frac", "zero_frac"]
    assert (
        rows.loc[[("a", NONE), ("d", NONE)], [*fractions, "ceiling_frac_annot"]].isna().all().all()
    )
    assert rows.loc[("b", NONE), fractions].tolist() == pytest.approx([0.1, 0.0, 0.0, 0.1])
    assert rows.loc[("c", NONE), ["ceiling_frac", "ceiling_exact_frac", "zero_frac"]].isna().all()
    assert rows["ceiling_region"].tolist() == ["none", "template", "template", "none"]


@pytest.mark.parametrize(
    ("lh", "rh", "expected"),
    [
        (("template", 0.01), ("annot", 0.02), "annot"),
        (("template", 0.02), ("annot", 0.01), "template"),
        (("annot", 0.0), ("template", 0.0), "template"),
        (("annot", 0.0), ("annot", 0.0), "annot"),
        (("none", None), ("annot", 0.0), "annot"),
        (("none", None), ("template", None), "template"),
        (("none", None), ("none", None), "none"),
    ],
)
def test_ceiling_region_is_the_kind_behind_the_reported_fraction(
    lh: tuple[str, float | None], rh: tuple[str, float | None], expected: str
) -> None:
    vertex = [
        {"subject_uid": "a", "hemi": hemi, "region_kind": kind, "ceiling_frac": fraction}
        for hemi, (kind, fraction) in zip(HEMIS, (lh, rh), strict=True)
    ]
    rows = metrics_of(vertex, None, [], subjects("a"))
    assert rows["ceiling_region"].tolist() == [expected]


def test_the_regional_ratio_pools_both_hemispheres_by_area() -> None:
    regional = [
        {
            "subject_uid": "a",
            "hemi": "lh",
            "parcel": "cuneus",
            "surf_area_mm2": 100,
            "thick_avg_mm": 2.0,
        },
        {
            "subject_uid": "a",
            "hemi": "rh",
            "parcel": "cuneus",
            "surf_area_mm2": 300,
            "thick_avg_mm": 3.0,
        },
        {
            "subject_uid": "a",
            "hemi": "lh",
            "parcel": "lingual",
            "surf_area_mm2": 100,
            "thick_avg_mm": 1.0,
        },
        {
            "subject_uid": "a",
            "hemi": "lh",
            "parcel": "superiorfrontal",
            "surf_area_mm2": 50,
            "thick_avg_mm": 2.5,
        },
        {
            "subject_uid": "a",
            "hemi": "rh",
            "parcel": "caudalmiddlefrontal",
            "surf_area_mm2": 150,
            "thick_avg_mm": 3.5,
        },
        # neither occipital nor control: no weight in either mean
        {
            "subject_uid": "a",
            "hemi": "lh",
            "parcel": "insula",
            "surf_area_mm2": 900,
            "thick_avg_mm": 4.0,
        },
        {
            "subject_uid": "a",
            "hemi": "lh",
            "parcel": "fusiform",
            "surf_area_mm2": 900,
            "thick_avg_mm": 4.0,
        },
        # a parcel without area has no weight
        {
            "subject_uid": "a",
            "hemi": "rh",
            "parcel": "lingual",
            "surf_area_mm2": 0,
            "thick_avg_mm": 9.0,
        },
    ]
    a = metrics_of(both_hemis("a"), None, regional, subjects("a")).loc[("a", NONE)]
    occipital = (100 * 2.0 + 300 * 3.0 + 100 * 1.0) / 500
    control = (50 * 2.5 + 150 * 3.5) / 200
    assert a["occ_thick_mm"] == pytest.approx(occipital)
    assert a["ctrl_thick_mm"] == pytest.approx(control)
    assert a["occ_ctrl_ratio"] == pytest.approx(occipital / control)
    assert a["flags"] == ""


def test_the_regional_columns_are_those_of_the_regional_module(
    synth_cohort: SyntheticCohort,
) -> None:
    from surfcensus.metrics.regional import occ_ctrl_ratio

    regional = cohort_regional(synth_cohort)
    people = subjects(*(subject.path.name for subject in synth_cohort.subjects))
    rows = subject_metrics(
        table("vertex_summary", []), None, regional, people, REGIONS, mass="excess"
    )
    assert len(rows) == 32
    for row in rows.itertuples():
        of_subject = regional[regional["subject_uid"] == row.subject_uid]
        expected = occ_ctrl_ratio(of_subject, REGIONS)
        got = (row.occ_thick_mm, row.ctrl_thick_mm, row.occ_ctrl_ratio)
        assert got == pytest.approx(expected, rel=1e-6, nan_ok=True)
    without = rows.set_index("subject_uid").loc[synth_cohort.no_stats[0]]
    assert math.isnan(without["occ_ctrl_ratio"]) and "no_stats" in without["flags"]
    assert rows["occ_ctrl_ratio"].notna().sum() == 31


def test_a_subject_without_stats_or_without_one_of_the_two_regions() -> None:
    regional = [
        *parcels("b", {"cuneus": 2.0}),  # no control parcel
        *parcels("c", {"superiorfrontal": 2.5}),  # no occipital parcel
        *parcels("d", {"cuneus": 2.0, "superiorfrontal": 2.5}),
    ]
    vertex = [row for uid in "abcd" for row in both_hemis(uid, flags="no_template")]
    rows = metrics_of(vertex, None, regional, subjects("a", "b", "c", "d"))
    columns = ["occ_thick_mm", "ctrl_thick_mm", "occ_ctrl_ratio"]
    assert rows.loc[("a", NONE), columns].isna().all()
    assert rows.loc[("b", NONE), columns].tolist() == pytest.approx(
        [2.0, np.nan, np.nan], nan_ok=True
    )
    assert rows.loc[("c", NONE), columns].tolist() == pytest.approx(
        [np.nan, 2.5, np.nan], nan_ok=True
    )
    assert rows.loc[("d", NONE), columns].tolist() == pytest.approx([2.0, 2.5, 0.8])
    # no_stats is for a subject without any row, not for one that lacks a parcel
    assert rows["flags"].tolist() == [
        "no_stats,no_template",
        "no_template",
        "no_template",
        "no_template",
    ]


def test_holes_and_the_truncated_fraction() -> None:
    people = subjects(
        "a",
        "b",
        "c",
        "d",
        holes_lh={"a": 41, "b": None, "c": 0, "d": 1},
        holes_rh={"a": 26, "b": None, "c": 0, "d": 1},
        trunc_lh_1={"a": 39, "b": 1, "c": 1, "d": 1},
        trunc_lh_2={"a": 225, "b": 1, "c": 1, "d": 1},
        trunc_rh_1={"a": 93, "b": 1, "c": 1, "d": 1},
        trunc_rh_2={"a": 434, "b": None, "c": 1, "d": 1},
    )
    vertex = [
        {"subject_uid": "a", "hemi": "lh", "n_native": 148_332},
        {"subject_uid": "a", "hemi": "rh", "n_native": 146_947},
        *both_hemis("b"),
        {"subject_uid": "c", "hemi": "lh"},  # one hemisphere only: no denominator
    ]
    rows = metrics_of(vertex, None, [], people)
    assert rows["holes_lh"].tolist() == [41, pd.NA, 0, 1]
    assert rows["holes_rh"].dtype == pd.Int32Dtype()
    expected = (39 + 225 + 93 + 434) / (2 * (148_332 + 146_947))
    assert rows.loc[("a", NONE), "trunc_frac"] == pytest.approx(expected, rel=1e-6)
    assert rows.loc[[("b", NONE), ("c", NONE), ("d", NONE)], "trunc_frac"].isna().all()


def test_flags_are_the_union_of_the_hemispheres_sorted() -> None:
    vertex = [
        {"subject_uid": "a", "hemi": "lh", "flags": "sphere_flipped,thickness_above_max"},
        {"subject_uid": "a", "hemi": "rh", "flags": "low_coverage,sphere_flipped"},
        *both_hemis("b"),
    ]
    regional = parcels("b", {"cuneus": 2.0, "superiorfrontal": 2.5})
    rows = metrics_of(vertex, None, regional, subjects("a", "b"))
    assert rows["flags"].tolist() == [
        "low_coverage,no_stats,sphere_flipped,thickness_above_max",
        "",
    ]
    bad = [{"subject_uid": "a", "hemi": "lh", "flags": "not_a_flag"}]
    with pytest.raises(ValueError, match="not_a_flag"):
        metrics_of(bad, None, [], subjects("a"))


def test_rows_are_for_the_present_subjects_in_key_order_with_the_types_of_the_table() -> None:
    people = subjects("c", "a", "b", present={"b": False}, dataset={"c": "D-00000002"})
    vertex = both_hemis("a") + both_hemis("b") + both_hemis("c") + both_hemis("unknown")
    clusters = [{"subject_uid": "c"}, {"subject_uid": "b"}, {"subject_uid": "unknown"}]
    regional = parcels("b", {"cuneus": 2.0}) + parcels("unknown", {"cuneus": 2.0})
    frame = subject_metrics(
        table("vertex_summary", vertex),
        table("clusters", clusters),
        table("regional", regional),
        people,
        REGIONS,
        mass="excess",
    )
    assert frame.columns.tolist() == list(METRIC_COLUMNS)
    assert frame["subject_uid"].tolist() == ["a", "c"]
    assert frame["dataset"].tolist() == [DATASET, "D-00000002"]
    assert isinstance(frame.index, pd.RangeIndex)
    for name in METRIC_COLUMNS:
        kind = schemas.SCORES.field(name).type
        if kind == "string":
            assert frame[name].dtype == object, name
        elif kind == "float32":
            assert frame[name].dtype == np.float32, name
        else:
            assert frame[name].dtype == pd.Int32Dtype(), name

    nobody = subject_metrics(
        table("vertex_summary", vertex),
        None,
        table("regional", []),
        subjects(),
        REGIONS,
        mass="excess",
    )
    assert len(nobody) == 0 and nobody.columns.tolist() == list(METRIC_COLUMNS)


def test_a_table_without_a_column_or_with_a_key_twice_is_refused() -> None:
    vertex, people = table("vertex_summary", both_hemis("a")), subjects("a")
    regional, clusters = table("regional", []), table("clusters", [{"subject_uid": "a"}])

    def call(
        v: pd.DataFrame = vertex,
        c: pd.DataFrame = clusters,
        r: pd.DataFrame = regional,
        s: pd.DataFrame = people,
    ) -> pd.DataFrame:
        return subject_metrics(v, c, r, s, REGIONS, mass="excess")

    assert len(call()) == 1
    with pytest.raises(ValueError, match="vertex_summary: no column ceiling_frac"):
        call(v=vertex.drop(columns="ceiling_frac"))
    with pytest.raises(ValueError, match="clusters: no column excess_mass, peak_z"):
        call(c=clusters.drop(columns=["excess_mass", "peak_z"]))
    with pytest.raises(ValueError, match="regional: no column surf_area_mm2"):
        call(r=regional.drop(columns="surf_area_mm2"))
    with pytest.raises(ValueError, match="subjects: no column trunc_rh_2"):
        call(s=people.drop(columns="trunc_rh_2"))
    with pytest.raises(ValueError, match="subjects: a subject_uid twice"):
        call(s=pd.concat([people, people]))
    with pytest.raises(ValueError, match="more than one row for a hemisphere"):
        call(v=pd.concat([vertex, vertex]))
    with pytest.raises(ValueError, match="two clusters of rank 1"):
        call(c=pd.concat([clusters, clusters]))
    twice = table("regional", parcels("a", {"cuneus": 2.0}))
    with pytest.raises(ValueError, match="more than one row for a parcel"):
        call(r=pd.concat([twice, twice]))
    with pytest.raises(ValueError, match="not a region: ventral"):
        call(c=clusters.assign(region="ventral"))
    with pytest.raises(ValueError, match="not a region kind: guess"):
        call(v=vertex.assign(region_kind="guess"))


# --- composite ------------------------------------------------------------------------------


def test_the_composite_is_the_largest_score_and_names_its_component() -> None:
    frame = components_frame(
        {
            "mass_medial": [0, 0, 0, 9],
            "mass_inferior": [0, 5, 0, 0],
            "ceiling_frac": [0, 0.01, 0.02, 0.03],
            "occ_ctrl_ratio": [0.9, 0.7, 0.8, 0.6],
        }
    )
    scored = composite(frame, NO_STRATA, CONFIG)
    quarter = [score((r - 0.5) / 4) for r in (1, 2, 3, 4)]
    assert scored["std_mass_medial"].tolist() == pytest.approx([quarter[0]] * 3 + [quarter[3]])
    assert scored["std_ceiling_frac"].tolist() == pytest.approx(quarter)
    assert scored["std_occ_ctrl_ratio"].tolist() == pytest.approx(
        [quarter[3], quarter[1], quarter[2], quarter[0]]
    )
    assert scored["composite"].tolist() == pytest.approx(
        [quarter[3]] * 2 + [quarter[2], quarter[3]]
    )
    # s003 is highest on mass_medial and on ceiling_frac: the first in config order is named
    assert scored["composite_component"].tolist() == [
        "occ_ctrl_ratio",
        "mass_inferior",
        "ceiling_frac",
        "mass_medial",
    ]
    assert scored["n_components"].tolist() == [4, 4, 4, 4]
    assert set(scored["score_kind"]) == {"composite_z"} and set(scored["flags"]) == {""}
    for name in ("composite", *STD_COLUMNS):
        assert scored[name].dtype == np.float32
    assert scored["n_components"].dtype == np.int8
    # the composite is the maximum of the scores as they are stored
    assert (scored["composite"] == scored[list(STD_COLUMNS)].max(axis=1)).all()


def test_the_order_of_the_components_decides_a_tie() -> None:
    frame = components_frame({"mass_medial": [0, 1], "ceiling_frac": [0, 1]})
    config = ScoringConfig(components=("ceiling_frac", "mass_medial"))
    assert (
        composite(frame, NO_STRATA, config)["composite_component"].tolist() == ["ceiling_frac"] * 2
    )
    config = ScoringConfig(components=("mass_medial", "ceiling_frac"))
    assert (
        composite(frame, NO_STRATA, config)["composite_component"].tolist() == ["mass_medial"] * 2
    )


def test_a_subject_with_fewer_components_is_flagged_partial() -> None:
    frame = components_frame(
        {
            "mass_medial": [1, 2, None, None],
            "mass_inferior": [1, 2, None, None],
            "ceiling_frac": [0.1, 0.2, 0.3, None],
            "occ_ctrl_ratio": [0.8, 0.9, 0.7, None],
        },
        flags="no_sphere_reg",
    )
    scored = composite(frame, NO_STRATA, CONFIG)
    assert scored["n_components"].tolist() == [4, 4, 2, 0]
    assert scored["flags"].tolist() == [
        "no_sphere_reg",
        "no_sphere_reg",
        "no_sphere_reg,partial_score",
        "no_sphere_reg,partial_score",
    ]
    # scored among the three that have the component
    assert scored.loc[2, "composite"] == pytest.approx(score(2.5 / 3))
    assert scored.loc[2, "composite_component"] == "ceiling_frac"
    assert math.isnan(scored.loc[3, "composite"]) and scored.loc[3, "composite_component"] is None
    assert scored.loc[3, list(STD_COLUMNS)].isna().all()
    # the frame given is not changed
    assert frame["flags"].tolist() == ["no_sphere_reg"] * 4 and "composite" not in frame.columns


def planted_datasets() -> tuple[pd.DataFrame, pd.Series]:
    """Dataset A: 20 clean subjects, one of them mildly odd. Dataset B: 20 subjects that are
    all bad, with ceiling fractions and ratios above everything in A. Every subject has a
    z-map and no cluster."""
    uids = [f"a{k:02d}" for k in range(20)] + [f"b{k:02d}" for k in range(20)]
    ceiling = [0.0] * 19 + [0.002] + [0.02 + 0.001 * k for k in range(20)]
    ratio = [0.78 + 0.002 * k for k in range(20)] + [1.0 + 0.005 * k for k in range(20)]
    frame = components_frame(
        {
            "mass_medial": [0.0] * 40,
            "mass_inferior": [0.0] * 40,
            "ceiling_frac": ceiling,
            "occ_ctrl_ratio": ratio,
        },
        uids=uids,
    )
    strata = pd.Series(["dataset:D-A"] * 20 + ["dataset:D-B"] * 20, index=uids, dtype=object)
    return frame, strata


def test_a_dataset_whose_every_subject_is_bad_ranks_high_over_the_run() -> None:
    frame, strata = planted_datasets()
    over_run = add_rank(composite(frame, strata, CONFIG)).set_index("subject_uid")
    bad = [uid for uid in over_run.index if uid.startswith("b")]
    assert sorted(over_run.loc[bad, "rank"]) == list(range(1, 21))
    assert over_run.loc["b10", "composite"] > 0.7  # a typical subject of the bad dataset
    assert over_run.loc["a19", "rank"] == 21  # the mildly odd one, behind all of them

    config = ScoringConfig(population="stratum")
    in_stratum = add_rank(composite(frame, strata, config)).set_index("subject_uid")
    # standardised inside its dataset the typical bad subject looks like nothing
    assert abs(in_stratum.loc["b10", "composite"]) < 0.1
    assert in_stratum.loc["a19", "composite"] == pytest.approx(score(19.5 / 20))
    assert in_stratum.loc["a19", "rank"] < in_stratum.loc["b10", "rank"]
    assert in_stratum.loc[bad, "rank"].mean() > 15
    # within a stratum the scores are those of the stratum alone
    alone = composite(frame.iloc[20:], strata, CONFIG).set_index("subject_uid")
    assert in_stratum.loc[bad, "composite"].tolist() == alone.loc[bad, "composite"].tolist()


def test_a_subject_without_a_stratum_is_scored_among_all() -> None:
    frame, strata = planted_datasets()
    config = ScoringConfig(population="stratum")
    over_run = composite(frame, strata, CONFIG).set_index("subject_uid")
    for partial in (
        strata.drop(["b10", "a03"]),
        strata.where(~strata.index.isin(["b10", "a03"]), None),
    ):
        scored = composite(frame, partial, config).set_index("subject_uid")
        for uid in ("b10", "a03"):
            assert scored.loc[uid, "std_ceiling_frac"] == over_run.loc[uid, "std_ceiling_frac"]
            assert scored.loc[uid, "std_occ_ctrl_ratio"] == over_run.loc[uid, "std_occ_ctrl_ratio"]
        # and the strata are scored without them: 19 subjects each
        assert scored.loc["b19", "std_occ_ctrl_ratio"] == pytest.approx(score(18.5 / 19))
    # population "run" does not read the strata at all
    assert composite(frame, NO_STRATA, CONFIG).equals(composite(frame, strata, CONFIG))


def two_references() -> pd.DataFrame:
    """Subjects a to d with a row for each of two references, e without a z-map."""
    return components_frame(
        {
            "mass_medial": [0, 0, 5, 9, 9, 5, 0, 0, None],
            "mass_inferior": [1, 2, 3, 4, 4, 4, 4, 4, None],
            "ceiling_frac": [0.0, 0.1, 0.2, 0.3, 0.0, 0.1, 0.2, 0.3, 0.4],
            "occ_ctrl_ratio": [0.9, 0.8, 0.7, 0.6, 0.9, 0.8, 0.7, 0.6, None],
        },
        uids=["a", "b", "c", "d", "a", "b", "c", "d", "e"],
        reference=[REF] * 4 + [OTHER] * 4 + [NONE],
    )


def test_a_subject_level_component_counts_each_subject_once() -> None:
    scored = composite(two_references(), NO_STRATA, CONFIG)
    fifth = [score((r - 0.5) / 5) for r in (1, 2, 3, 4, 5)]
    # five subjects, not nine rows; e is scored against everyone, not alone
    assert scored["std_ceiling_frac"].tolist() == pytest.approx([*fifth[:4], *fifth[:4], fifth[4]])
    quarter = [score((r - 0.5) / 4) for r in (4, 3, 2, 1)]
    assert scored["std_occ_ctrl_ratio"].tolist() == pytest.approx(
        [*quarter, *quarter, np.nan], nan_ok=True
    )
    e = scored.iloc[8]
    assert e["composite"] == pytest.approx(fifth[4]) and e["composite_component"] == "ceiling_frac"
    assert e["n_components"] == 1 and e["flags"] == "partial_score"


def test_a_cluster_metric_is_standardised_within_its_reference() -> None:
    scored = composite(two_references(), NO_STRATA, CONFIG)
    low, mid, high = score(0.5 / 4), score(2.5 / 4), score(3.5 / 4)
    assert scored["std_mass_medial"].tolist() == pytest.approx(
        [low, low, mid, high, high, mid, low, low, np.nan], nan_ok=True
    )
    # in the second reference all four are tied
    assert scored["std_mass_inferior"].tolist()[4:8] == pytest.approx([low] * 4)
    assert scored["std_mass_inferior"].tolist()[:4] == pytest.approx(
        [score((r - 0.5) / 4) for r in (1, 2, 3, 4)]
    )


def test_a_cluster_metric_in_a_stratum_is_scored_within_reference_and_stratum() -> None:
    frame = two_references()
    strata = pd.Series({"a": "s1", "b": "s1", "c": "s2", "d": "s2", "e": None}, dtype=object)
    scored = composite(frame, strata, ScoringConfig(population="stratum"))
    low, high = score(0.25), score(0.75)
    assert scored["std_mass_medial"].tolist() == pytest.approx(
        [low, low, low, high, high, low, low, low, np.nan], nan_ok=True
    )
    # e has no stratum: its ceiling fraction is scored among all five
    assert scored["std_ceiling_frac"].tolist() == pytest.approx(
        [low, high, low, high, low, high, low, high, score(4.5 / 5)]
    )


def test_robust_z_is_the_other_way_to_standardise() -> None:
    frame = components_frame(
        {"mass_medial": ZEROS_CASE, "ceiling_frac": [float(k) for k in range(30)]}
    )
    config = ScoringConfig(components=("mass_medial", "ceiling_frac"), standardize="robust_z")
    scored = composite(frame, NO_STRATA, config)
    assert np.round(scored["std_mass_medial"].to_numpy()[26:], 2).tolist() == pytest.approx(
        [0.05, 0.27, 1.08, 5.41]
    )
    expected = robust_z(pd.Series(np.arange(30.0))).to_numpy().astype(np.float32)
    assert scored["std_ceiling_frac"].tolist() == expected.tolist()
    # a real cluster of mass 50 loses against a middling ceiling fraction
    assert scored.loc[27, "composite_component"] == "ceiling_frac"
    by_rank = composite(frame, NO_STRATA, ScoringConfig(components=config.components))
    assert by_rank.loc[27, "std_mass_medial"] == pytest.approx(1.38, abs=0.005)


def test_with_norms_the_scores_are_percentiles_of_the_reference_population() -> None:
    frame = components_frame(
        {
            "mass_medial": [0, 50, None],
            "mass_inferior": [10, 20, None],
            "ceiling_frac": [2.5, 97.5, 50],
            "occ_ctrl_ratio": [30, 60, None],
        }
    )
    norms = {name: UNIFORM_GRID for name in CONFIG.components}
    scored = composite(frame, NO_STRATA, CONFIG, norms=norms)
    assert scored["std_ceiling_frac"].tolist() == pytest.approx([score(0.025), score(0.975), 0.0])
    assert scored["std_mass_medial"].tolist() == pytest.approx(
        [score(0.005), 0.0, np.nan], nan_ok=True
    )
    assert scored["composite"].tolist() == pytest.approx([score(0.30), score(0.975), 0.0])
    assert scored["flags"].tolist() == ["", "", "partial_score"]
    # nothing of the run enters: one row alone scores the same
    alone = composite(frame.iloc[[1]], NO_STRATA, ScoringConfig(population="stratum"), norms=norms)
    assert alone["composite"].tolist() == scored["composite"].tolist()[1:2]
    with pytest.raises(ValueError, match="no percentiles for mass_inferior, occ_ctrl_ratio"):
        composite(
            frame,
            NO_STRATA,
            CONFIG,
            norms={"mass_medial": UNIFORM_GRID, "ceiling_frac": UNIFORM_GRID},
        )


def test_another_component_gets_its_own_column() -> None:
    frame = components_frame({"mass_lateral": [0, 3, 1], "ceiling_frac": [0.1, 0.0, 0.2]})
    config = ScoringConfig(components=("mass_lateral", "ceiling_frac"))
    scored = composite(frame, NO_STRATA, config)
    third = [score((r - 0.5) / 3) for r in (1, 2, 3)]
    assert scored["std_mass_lateral"].tolist() == pytest.approx([third[0], third[2], third[1]])
    assert scored["composite_component"].tolist() == [
        "ceiling_frac",
        "mass_lateral",
        "ceiling_frac",
    ]
    assert scored["n_components"].tolist() == [2, 2, 2] and set(scored["flags"]) == {""}
    # the columns of the table are there, empty for what was not asked for
    assert scored[["std_mass_medial", "std_mass_inferior", "std_occ_ctrl_ratio"]].isna().all().all()


def test_composite_keeps_the_index_and_the_rows_of_the_frame() -> None:
    frame, _ = planted_datasets()
    shuffled = frame.sample(frac=1.0, random_state=4)
    scored = composite(shuffled, NO_STRATA, CONFIG)
    assert scored.index.tolist() == shuffled.index.tolist()
    assert (
        scored.sort_index()["composite"].tolist()
        == composite(frame, NO_STRATA, CONFIG)["composite"].tolist()
    )


def test_composite_refuses_what_it_cannot_score() -> None:
    frame = components_frame({"ceiling_frac": [0.1, 0.2]})
    with pytest.raises(ValueError, match="metrics: no column mass_lateral"):
        composite(frame, NO_STRATA, ScoringConfig(components=("ceiling_frac", "mass_lateral")))
    with pytest.raises(ValueError, match="named twice"):
        composite(frame, NO_STRATA, ScoringConfig(components=("ceiling_frac", "ceiling_frac")))
    with pytest.raises(ValueError, match="flags is not a numeric column"):
        composite(frame.assign(flags="x"), NO_STRATA, ScoringConfig(components=("flags",)))
    with pytest.raises(ValueError, match="a subject twice for one reference"):
        composite(pd.concat([frame, frame]), NO_STRATA, CONFIG)
    twice = pd.Series(["s", "t"], index=["s000", "s000"], dtype=object)
    with pytest.raises(ValueError, match="strata: a subject twice"):
        composite(frame, twice, ScoringConfig(population="stratum"))
    differing = components_frame(
        {"ceiling_frac": [0.1, 0.2]}, uids=["a", "a"], reference=[REF, OTHER]
    )
    with pytest.raises(ValueError, match="ceiling_frac differs between the rows of one subject"):
        composite(differing, NO_STRATA, CONFIG)


def test_ties_are_decided_on_the_stored_float32_scores() -> None:
    # two scores that differ in float64 and not in float32
    grid = UNIFORM_GRID
    frame = components_frame({"mass_medial": [84.0], "ceiling_frac": [84.0]})
    frame["ceiling_frac"] = frame["ceiling_frac"].astype(np.float64) + 1e-9
    assert score(0.84 + 1e-11) > score(0.84)
    scored = composite(frame, NO_STRATA, CONFIG, norms={name: grid for name in CONFIG.components})
    assert scored.loc[0, "std_mass_medial"] == scored.loc[0, "std_ceiling_frac"]
    assert scored.loc[0, "composite_component"] == "mass_medial"


def test_a_run_without_subjects_scores_to_empty_frames() -> None:
    nobody = subject_metrics(
        table("vertex_summary", []),
        table("clusters", []),
        table("regional", []),
        subjects(),
        REGIONS,
        mass="excess",
    )
    for scored in (composite(nobody, NO_STRATA, CONFIG), fallback_rank(nobody)):
        ranked = add_rank(scored)
        assert len(ranked) == 0
        assert set(schemas.SCORES.names) - set(ranked.columns) == {
            "primary",
            "primary_rank",
            "stratum",
            "n_reference",
            "score_config_hash",
        }
        assert ranked["composite"].dtype == np.float32 and ranked["n_components"].dtype == np.int8


# --- the ranking of decision D16 ------------------------------------------------------------


def test_fallback_rank_on_five_subjects() -> None:
    frame = components_frame(
        {"ceiling_frac": [0.0, 0.0, 0.01, 0.0, 0.03], "occ_ctrl_ratio": [0.8, 0.9, 0.7, 1.0, 0.75]},
        reference=REF,
    )
    ranked = fallback_rank(frame)
    assert ranked["std_ceiling_frac"].tolist() == pytest.approx([0, 0, 0.75, 0, 1])
    assert ranked["std_occ_ctrl_ratio"].tolist() == pytest.approx([0.5, 0.75, 0, 1, 0.25])
    assert ranked["composite"].tolist() == pytest.approx([0.5, 0.75, 0.75, 1, 1])
    assert ranked["composite_component"].tolist() == [
        "occ_ctrl_ratio",
        "occ_ctrl_ratio",
        "ceiling_frac",
        "occ_ctrl_ratio",
        "ceiling_frac",
    ]
    assert set(ranked["score_kind"]) == {"fallback_rank"}
    assert set(ranked["reference_id"]) == {NONE}
    assert ranked["n_components"].tolist() == [2] * 5 and set(ranked["flags"]) == {""}
    assert ranked[["std_mass_medial", "std_mass_inferior"]].isna().all().all()
    # ties by the mean of the two percentiles, then by subject
    assert add_rank(ranked)["rank"].tolist() == [5, 3, 4, 2, 1]


def test_twelve_subjects_without_a_ceiling_vertex_get_twelve_ranks() -> None:
    ratios = [0.80, 0.91, 0.77, 0.85, 0.79, 0.88, 0.83, 0.95, 0.81, 0.87, 0.76, 0.90]
    frame = components_frame({"ceiling_frac": [0.0] * 12, "occ_ctrl_ratio": ratios}, reference=NONE)
    ranked = add_rank(fallback_rank(frame))
    # with average ranks every ceiling percentile would be 0.5 and seven composites remain
    assert (ranked["std_ceiling_frac"] == 0).all()
    assert ranked["composite"].nunique() == 12
    assert ranked["composite"].tolist() == pytest.approx(
        [(sorted(ratios).index(r)) / 11 for r in ratios]
    )
    assert sorted(ranked["rank"]) == list(range(1, 13))
    assert ranked.sort_values("rank")["occ_ctrl_ratio"].tolist() == sorted(
        np.float32(ratios).tolist(), reverse=True
    )
    assert set(ranked["composite_component"]) == {"ceiling_frac", "occ_ctrl_ratio"}
    # only the subject with the lowest ratio ties the two percentiles, at 0
    assert ranked.loc[10, "composite_component"] == "ceiling_frac"


def test_fallback_rank_with_one_subject_and_with_missing_components() -> None:
    one = fallback_rank(components_frame({"ceiling_frac": [0.02], "occ_ctrl_ratio": [0.8]}))
    assert one["composite"].tolist() == [0.0] and one["n_components"].tolist() == [2]
    frame = components_frame(
        {"ceiling_frac": [0.0, 0.01, None, None], "occ_ctrl_ratio": [0.8, None, 0.7, None]},
        flags="no_template",
    )
    ranked = fallback_rank(frame)
    assert ranked["std_ceiling_frac"].tolist() == pytest.approx([0, 1, np.nan, np.nan], nan_ok=True)
    assert ranked["std_occ_ctrl_ratio"].tolist() == pytest.approx(
        [1, np.nan, 0, np.nan], nan_ok=True
    )
    assert ranked["composite"].tolist() == pytest.approx([1, 1, 0, np.nan], nan_ok=True)
    assert ranked["n_components"].tolist() == [2, 1, 1, 0]
    assert ranked["flags"].tolist() == [
        "no_template",
        "no_template,partial_score",
        "no_template,partial_score",
        "no_template,partial_score",
    ]
    assert ranked["composite_component"].tolist() == [
        "occ_ctrl_ratio",
        "ceiling_frac",
        "occ_ctrl_ratio",
        None,
    ]
    with pytest.raises(ValueError, match="a subject twice"):
        fallback_rank(pd.concat([frame, frame]))
    with pytest.raises(ValueError, match="metrics: no column occ_ctrl_ratio"):
        fallback_rank(frame.drop(columns="occ_ctrl_ratio"))


# --- rank -----------------------------------------------------------------------------------


def rank_frame(
    rows: Sequence[tuple[str, str, float | None, float | None, float | None]],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_uid": pd.Series([row[0] for row in rows], dtype=object),
            "reference_id": pd.Series([row[1] for row in rows], dtype=object),
            "composite": np.array(
                [np.nan if row[2] is None else row[2] for row in rows], np.float32
            ),
            "std_ceiling_frac": np.array(
                [np.nan if row[3] is None else row[3] for row in rows], np.float32
            ),
            "std_occ_ctrl_ratio": np.array(
                [np.nan if row[4] is None else row[4] for row in rows], np.float32
            ),
        }
    )


def test_rank_one_is_the_highest_composite_within_a_reference() -> None:
    frame = rank_frame(
        [
            ("a", REF, 0.5, 0.5, 0.1),
            ("b", REF, 2.0, 2.0, 0.0),
            ("c", REF, -1.0, -1.0, -2.0),
            ("a", OTHER, 0.1, 0.1, 0.0),
            ("b", OTHER, -0.3, -0.3, -1.0),
            ("d", NONE, 1.0, 1.0, None),
        ]
    )
    ranked = add_rank(frame)
    assert ranked["rank"].tolist() == [2, 1, 3, 1, 2, 1]
    assert ranked["rank"].dtype == np.int32
    assert ranked.drop(columns="rank").equals(frame) and "rank" not in frame.columns
    # an earlier rank is replaced, and the order of the rows does not matter
    again = add_rank(ranked.assign(rank=0).iloc[::-1])
    assert again.sort_index()["rank"].tolist() == [2, 1, 3, 1, 2, 1]


def test_rank_ties_go_by_the_mean_score_then_by_subject() -> None:
    frame = rank_frame(
        [
            ("e", REF, 1.0, 1.0, 0.2),
            ("d", REF, 1.0, 1.0, 0.6),  # the higher mean of the tie
            ("c", REF, 1.0, 1.0, None),  # mean over one score: 1.0, the highest
            ("b", REF, 1.0, 0.2, 1.0),  # the same mean as e: by subject
            ("a", REF, 0.9, 0.9, 0.9),  # a higher mean does not beat a higher composite
        ]
    )
    assert add_rank(frame).set_index("subject_uid")["rank"].to_dict() == {
        "c": 1,
        "d": 2,
        "b": 3,
        "e": 4,
        "a": 5,
    }


def test_subjects_without_a_composite_come_last() -> None:
    frame = rank_frame(
        [
            ("z", REF, None, None, None),
            ("y", REF, -3.0, -3.0, -4.0),
            ("x", REF, None, None, None),
            ("w", REF, 0.0, 0.0, None),
        ]
    )
    assert add_rank(frame)["rank"].tolist() == [4, 2, 3, 1]
    empty = add_rank(rank_frame([]))
    assert len(empty) == 0 and empty["rank"].dtype == np.int32
    with pytest.raises(ValueError, match="scores: no column composite"):
        add_rank(frame.drop(columns="composite"))


def test_the_composite_is_compared_as_it_is_stored() -> None:
    frame = rank_frame([("b", REF, 1.0, 0.0, None), ("a", REF, 1.0, 0.5, None)])
    frame["composite"] = np.array([1.0 + 1e-9, 1.0])  # equal as float32
    assert add_rank(frame)["rank"].tolist() == [2, 1]
    frame["composite"] = np.array([1.0 + 1e-6, 1.0])  # not equal as float32
    assert add_rank(frame)["rank"].tolist() == [1, 2]


def test_primary_rank_is_one_order_over_the_primary_rows() -> None:
    """``rank`` is within a reference, so a subject without a z-map is rank 1 among the
    rows without a reference, beside the rank 1 of the scored subjects. ``primary_rank``
    numbers the primary rows once: those with a reference first."""
    frame = add_rank(
        rank_frame(
            [
                ("a", REF, 0.5, 0.5, 0.1),
                ("b", REF, 2.0, 2.0, 0.0),
                ("c", NONE, 3.0, 3.0, None),  # no thickness: scored on what is left
                ("a", OTHER, 0.1, 0.1, 0.0),
                ("b", OTHER, -0.3, -0.3, -1.0),
                ("d", NONE, 1.0, 1.0, None),
                ("e", OTHER, 0.0, 0.0, 0.0),  # primary under another reference than a and b
            ]
        )
    )
    frame["primary"] = [True, True, True, False, False, True, True]
    assert frame["rank"].tolist() == [2, 1, 1, 1, 3, 2, 2]
    assert frame.loc[frame["primary"], "rank"].tolist().count(1) == 2
    ranked = add_primary_rank(frame)
    assert ranked["primary_rank"].dtype == pd.Int32Dtype()
    # b (1), then a and e (both 2, by subject), then the two without a reference
    assert ranked["primary_rank"].tolist() == [2, 1, 4, pd.NA, pd.NA, 5, 3]
    assert ranked.drop(columns="primary_rank").equals(frame)
    assert "primary_rank" not in frame.columns
    # the order of the rows does not matter, and an earlier column is replaced
    again = add_primary_rank(ranked.assign(primary_rank=0).iloc[::-1].reset_index(drop=True))
    assert again["primary_rank"].tolist() == [2, 1, 4, pd.NA, pd.NA, 5, 3][::-1]
    empty = add_primary_rank(add_rank(rank_frame([])).assign(primary=True))
    assert len(empty) == 0 and empty["primary_rank"].dtype == pd.Int32Dtype()
    with pytest.raises(ValueError, match="scores: no column primary"):
        add_primary_rank(frame.drop(columns="primary"))


# --- cohorts --------------------------------------------------------------------------------

MESH_ORDER = 4
CLUSTER_Z = 3.0
MAD_FLOOR_MM = 0.05


def cohort_regional(cohort: SyntheticCohort) -> pd.DataFrame:
    """The parcel rows of the cohort's stats files, read as text."""
    rows = []
    for subject in cohort.subjects:
        for hemi in HEMIS:
            path = subject.path / "stats" / f"{hemi}.aparc.stats"
            if not path.is_file():
                continue
            for line in path.read_text().splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                name, num_vert, area, volume, thickness, std = line.split()[:6]
                rows.append(
                    {
                        "subject_uid": subject.path.name,
                        "hemi": hemi,
                        "parcel": name,
                        "num_vert": int(num_vert),
                        "surf_area_mm2": int(area),
                        "gray_vol_mm3": int(volume),
                        "thick_avg_mm": float(thickness),
                        "thick_std_mm": float(std),
                    }
                )
    return table("regional", rows)


def _ceiling(thickness: np.ndarray, region: np.ndarray, prefix: str) -> dict[str, object]:
    values = thickness[region]
    return {
        f"{prefix}_n": len(values),
        f"{prefix}_n_exact": int((values >= 5.0).sum()),
        f"{prefix}_n_near": int((values >= 4.9).sum()),
        f"{prefix}_n_zero": int((values == 0).sum()),
    }


def cohort_tables(
    cohort: SyntheticCohort,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    """``(vertex_summary, clusters, regional)`` for the cohort, with the subject's name as
    its ``subject_uid``. The stand-in of the module docstring: clusters only when at least
    30 subjects have a map, as decision D16 has it."""
    directions, faces = template_mesh(MESH_ORDER)
    n = len(directions)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    adjacency = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
    occipital = [DESIKAN_NAMES.index(name) for name in REGIONS.occipital]
    summary: list[dict[str, object]] = []
    maps: dict[tuple[str, str], np.ndarray] = {}
    for subject in cohort.subjects:
        uid = subject.path.name
        for hemi in HEMIS:
            thickness = fs.read_morph_data(str(subject.path / "surf" / f"{hemi}.thickness"))
            labels, _, _ = fs.read_annot(str(subject.path / "label" / f"{hemi}.aparc.annot"))
            cortex = labels >= 0
            by_annot = _ceiling(thickness, cortex & np.isin(labels, occipital), "annot_search")
            row: dict[str, object] = {
                "subject_uid": uid,
                "hemi": hemi,
                "n_native": len(thickness),
                "thickness_max_mm": float(thickness.max()),
                "annot_search_n": by_annot["annot_search_n"],
                "annot_search_n_near": by_annot["annot_search_n_near"],
                "annot_search_n_zero": by_annot["annot_search_n_zero"],
            }
            row["ceiling_frac_annot"] = (
                by_annot["annot_search_n_near"] + by_annot["annot_search_n_zero"]
            ) / by_annot["annot_search_n"]
            sphere = subject.path / "surf" / f"{hemi}.sphere.reg"
            if sphere.is_file():
                registered = fs.read_geometry(str(sphere))[0] / 100.0
                native = expected_regions(registered, hemi)
                counts = {
                    **_ceiling(thickness, cortex & native["search"], "search"),
                    **_ceiling(thickness, cortex & native["control"], "control"),
                }
                nearest = cKDTree(registered).query(directions)[1]
                resampled = thickness[nearest].astype(np.float64)
                resampled[(resampled <= 0) | ~cortex[nearest]] = np.nan
                maps[uid, hemi] = resampled
                row.update(region_kind="template", flags="", **counts)
            else:
                counts = {
                    "search_n": by_annot["annot_search_n"],
                    "search_n_exact": int(
                        (thickness[cortex & np.isin(labels, occipital)] >= 5).sum()
                    ),
                    "search_n_near": by_annot["annot_search_n_near"],
                    "search_n_zero": by_annot["annot_search_n_zero"],
                    "control_n": None,
                    "control_n_exact": None,
                    "control_n_near": None,
                    "control_n_zero": None,
                }
                row.update(
                    region_kind="annot",
                    flags="no_sphere_reg",
                    template_id=None,
                    n_fallback=None,
                    n_missing_cortex=None,
                    global_median_mm=None,
                    **counts,
                )
            row["ceiling_frac"] = (row["search_n_near"] + row["search_n_zero"]) / row["search_n"]
            row["control_ceiling_frac"] = (
                None
                if row["control_n"] is None
                else (row["control_n_near"] + row["control_n_zero"]) / row["control_n"]
            )
            summary.append(row)

    clusters: list[dict[str, object]] | None = None
    with_map = sorted({uid for uid, _ in maps})
    if len(with_map) >= 30:
        clusters = []
        for hemi in HEMIS:
            regions = expected_regions(directions, hemi)
            stack = np.stack([maps[uid, hemi] for uid in with_map])
            outside = ~regions["search"]
            stack = stack - np.nanmedian(stack[:, outside], axis=1, keepdims=True)
            stack = np.where(np.isfinite(stack).any(axis=0), stack, 0.0)  # the medial strip
            median = np.nanmedian(stack, axis=0)
            mad = np.maximum(np.nanmedian(np.abs(stack - median), axis=0), MAD_FLOOR_MM)
            z = (stack - median) / (1.4826 * mad)
            for uid, z_map in zip(with_map, z, strict=True):
                for region in MASS_REGIONS:
                    members = np.flatnonzero(regions[region] & (np.nan_to_num(z_map) > CLUSTER_Z))
                    if not len(members):
                        continue
                    _, label = connected_components(adjacency[members][:, members], directed=False)
                    found = [members[label == k] for k in range(label.max() + 1)]
                    found.sort(key=lambda vertices: (-len(vertices), vertices[0]))
                    for rank, vertices in enumerate(found[:5], start=1):
                        clusters.append(
                            {
                                "subject_uid": uid,
                                "hemi": hemi,
                                "region": region,
                                "rank": rank,
                                "n_vertices": len(vertices),
                                "area_mm2": float(len(vertices)),
                                "mass": float(z_map[vertices].sum()),
                                "excess_mass": float((z_map[vertices] - CLUSTER_Z).sum()),
                                "peak_z": float(z_map[vertices].max()),
                                "peak_vertex": int(vertices[z_map[vertices].argmax()]),
                                "centroid_vertex": int(vertices[0]),
                            }
                        )
    return (
        table("vertex_summary", summary),
        None if clusters is None else table("clusters", clusters),
        cohort_regional(cohort),
    )


def cohort_subjects(cohort: SyntheticCohort) -> pd.DataFrame:
    names = [subject.path.name for subject in cohort.subjects]
    numbers = {name: 10 + k for k, name in enumerate(names)}
    return subjects(
        *names,
        **{name: numbers for name in ("holes_lh", "holes_rh")},
        **{name: numbers for name in ("trunc_lh_1", "trunc_lh_2", "trunc_rh_1", "trunc_rh_2")},
    )


def test_the_three_bad_subjects_of_the_cohort_rank_one_to_three(
    synth_cohort: SyntheticCohort, tmp_path: Path
) -> None:
    vertex_summary, clusters, regional = cohort_tables(synth_cohort)
    assert clusters is not None
    people = cohort_subjects(synth_cohort)
    metrics = subject_metrics(vertex_summary, clusters, regional, people, REGIONS, mass="excess")
    assert len(metrics) == 32 and metrics["subject_uid"].is_unique
    scores = add_rank(composite(metrics, NO_STRATA, CONFIG)).set_index("subject_uid")

    bad = sorted(synth_cohort.bad)
    assert bad == ["sub-004", "sub-013", "sub-022"]
    within_set = scores[scores["reference_id"] != NONE].sort_values("rank")
    assert sorted(within_set.index[:3]) == bad
    assert sorted(scores.loc[bad, "rank"]) == [1, 2, 3]
    # each of them leads on the ceiling fraction, on the ratio, and on a cluster mass
    others = scores.drop(index=bad)
    for name in ("ceiling_frac", "occ_ctrl_ratio", "mass_medial"):
        assert scores.loc[bad, name].min() > others[name].max(), name
    assert scores.loc["sub-013", "mass_inferior"] == scores["mass_inferior"].max()
    assert (others["ceiling_frac"] == 0).all()
    assert scores.loc[bad, "composite"].min() > others["composite"].max()

    (no_sphere,), (no_stats,) = synth_cohort.no_sphere_reg, synth_cohort.no_stats
    assert scores.loc[no_sphere, "reference_id"] == NONE
    assert scores.loc[no_sphere, "flags"] == "no_sphere_reg,partial_score"
    assert scores.loc[no_sphere, "n_components"] == 2
    assert scores.loc[no_sphere, "ceiling_region"] == "annot"
    assert scores.loc[no_sphere, "rank"] == 1  # alone under its reference_id
    assert scores.loc[no_stats, "flags"] == "no_stats,partial_score"
    assert scores.loc[no_stats, "n_components"] == 3
    complete = scores.drop(index=[no_sphere, no_stats])
    assert (complete["n_components"] == 4).all() and (complete["flags"] == "").all()
    assert complete["reference_id"].nunique() == 1
    assert sorted(scores.loc[scores["reference_id"] != NONE, "rank"]) == list(range(1, 32))
    # the subject without a z-map is scored on its two components against all 32 and 31
    assert scores.loc[no_sphere, "composite"] == max(
        scores.loc[no_sphere, "std_ceiling_frac"], scores.loc[no_sphere, "std_occ_ctrl_ratio"]
    )
    assert scores.loc[no_sphere, "std_ceiling_frac"] == pytest.approx(score(0.5 / 32))

    # with the columns the pipeline adds, the frame is a scores table
    full = add_primary_rank(scores.reset_index().assign(primary=True)).assign(
        stratum=None, n_reference=31, score_config_hash="0" * 12
    )
    # one order over the 32: the 31 with a reference by their rank, then the one without
    assert sorted(full["primary_rank"]) == list(range(1, 33))
    by_uid = full.set_index("subject_uid")
    assert by_uid.loc[no_sphere, "primary_rank"] == 32 and by_uid.loc[no_sphere, "rank"] == 1
    referenced = by_uid.drop(index=no_sphere)
    assert referenced["primary_rank"].tolist() == referenced["rank"].tolist()
    path = tmp_path / "scores.parquet"
    store.write_table(path, full, schemas.SCORES, sort_by=schemas.KEYS["scores"])
    back = store.read_table(path, schemas.SCORES).set_index("subject_uid")
    assert back.loc[scores.index, "rank"].tolist() == scores["rank"].tolist()
    assert back.loc[scores.index, "composite"].tolist() == scores["composite"].tolist()
    # and scoring what was read back gives the same ranks: nothing depends on more digits
    again = add_rank(composite(back.reset_index()[list(METRIC_COLUMNS)], NO_STRATA, CONFIG))
    assert (
        again.set_index("subject_uid").loc[scores.index, "rank"].tolist() == scores["rank"].tolist()
    )


def test_five_subjects_are_ranked_as_decision_d16_says(small_cohort: SyntheticCohort) -> None:
    vertex_summary, clusters, regional = cohort_tables(small_cohort)
    assert clusters is None  # too few for a reference of their own
    metrics = subject_metrics(
        vertex_summary, None, regional, cohort_subjects(small_cohort), REGIONS, mass="excess"
    )
    assert set(metrics["reference_id"]) == {NONE}
    assert metrics[list(CLUSTER_METRICS)].isna().all().all()
    scores = add_rank(fallback_rank(metrics)).set_index("subject_uid")
    (bad,) = small_cohort.bad
    assert scores.loc[bad, "rank"] == 1 and scores.loc[bad, "composite"] == 1.0
    assert scores.loc[bad, "composite_component"] == "ceiling_frac"
    assert sorted(scores["rank"]) == [1, 2, 3, 4, 5]
    assert set(scores["score_kind"]) == {"fallback_rank"} and set(scores["flags"]) == {""}
    others = scores.drop(index=bad)
    # nobody else has a vertex at the ceiling, so they follow the ratio
    assert (others["std_ceiling_frac"] == 0).all() and others["composite"].is_unique
    by_ratio = others.sort_values("occ_ctrl_ratio", ascending=False).index.tolist()
    assert others.sort_values("rank").index.tolist() == by_ratio
