"""metrics/regional: the rows of ``regional.parquet`` and the occipital-to-control ratio.

Stats texts are written here in the layout of the real files; the cohort tests read the
files that ``tests/synth_tierb.py`` derives from each subject's own thickness and
annotation. The test at the end reads the real subject where the network tests have put it
in the pytest cache, and is skipped elsewhere.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import nibabel.freesurfer as fs
import numpy as np
import pandas as pd
import pytest
from synth_template import mesh_vertex_areas  # pyrefly: ignore[missing-import]
from synth_tierb import SyntheticCohort  # pyrefly: ignore[missing-import]

from surfcensus import schemas, store
from surfcensus.config import RegionConfig
from surfcensus.fsio.stats import StatsFile, parse_stats, read_stats
from surfcensus.metrics.regional import occ_ctrl_ratio, region_mean_thickness, regional_rows
from surfcensus.sources.base import HEMIS, Hemi

UID = "0123456789abcdef"
HEADER = (
    "# Measure Cortex, NumVert, Number of Vertices, 140304, unitless\n"
    "# ColHeaders StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv "
    "FoldInd CurvInd\n"
)
REGIONS = RegionConfig()

Parcel = tuple[str, object, object, object, object, object]
"""Name, NumVert, SurfArea, GrayVol, ThickAvg, ThickStd; written into the file as given."""

FOUR: tuple[Parcel, ...] = (
    ("cuneus", 2071, 1303, 2497, "1.820", "0.375"),
    ("bankssts", 1338, 953, 2385, "2.610", "0.429"),
    ("superiorfrontal", 10123, 6880, 21402, "2.704", "0.540"),
    ("lingual", 4880, 3130, 6882, "2.004", "0.525"),
)


def stats_of(parcels: Sequence[Parcel]) -> StatsFile:
    rows = "".join(
        f"{name:<40}{num_vert!s:>5}  {area!s:>5}  {volume!s:>5}  {thick!s:>5} {std!s:>5}"
        "     0.096     0.018        7     1.0\n"
        for name, num_vert, area, volume, thick, std in parcels
    )
    return parse_stats(HEADER + rows)


def table(rows: Mapping[tuple[str, str], tuple[float, float]], uid: str = UID) -> pd.DataFrame:
    """Rows of the regional table from ``(hemi, parcel) -> (area, thickness)``."""
    frame = pd.DataFrame(
        {
            "subject_uid": pd.Series([uid] * len(rows), dtype=object),
            "hemi": pd.Series([hemi for hemi, _ in rows], dtype=object),
            "parcel": pd.Series([parcel for _, parcel in rows], dtype=object),
            # contrary to the areas, so a mean weighted by vertex count gives another number
            "num_vert": np.arange(len(rows), 0, -1, dtype=np.int32) * 1000,
            "surf_area_mm2": np.array([area for area, _ in rows.values()], dtype=np.int32),
            "gray_vol_mm3": np.full(len(rows), 7, dtype=np.int32),
            "thick_avg_mm": np.array([thick for _, thick in rows.values()], dtype=np.float32),
            "thick_std_mm": np.full(len(rows), 0.5, dtype=np.float32),
        }
    )
    return frame


# ---------------------------------------------------------------------------------------
# regional_rows


def test_the_rows_have_the_columns_and_types_of_the_table() -> None:
    rows = regional_rows(UID, "lh", stats_of(FOUR))
    assert list(rows.columns) == schemas.REGIONAL.names
    assert {name: str(dtype) for name, dtype in rows.dtypes.items()} == {
        "subject_uid": "object",
        "hemi": "object",
        "parcel": "object",
        "num_vert": "int32",
        "surf_area_mm2": "int32",
        "gray_vol_mm3": "int32",
        "thick_avg_mm": "float32",
        "thick_std_mm": "float32",
    }
    assert isinstance(rows.index, pd.RangeIndex)
    assert set(rows["subject_uid"]) == {UID}
    assert set(rows["hemi"]) == {"lh"}
    assert all(type(value) is str for value in rows["parcel"])


def test_every_parcel_of_the_file_is_a_row_in_parcel_order() -> None:
    rows = regional_rows(UID, "rh", stats_of(FOUR))
    assert list(rows["parcel"]) == ["bankssts", "cuneus", "lingual", "superiorfrontal"]
    assert list(rows["num_vert"]) == [1338, 2071, 4880, 10123]
    assert list(rows["surf_area_mm2"]) == [953, 1303, 3130, 6880]
    assert list(rows["gray_vol_mm3"]) == [2385, 2497, 6882, 21402]
    np.testing.assert_array_equal(
        rows["thick_avg_mm"].to_numpy(), np.array([2.610, 1.820, 2.004, 2.704], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        rows["thick_std_mm"].to_numpy(), np.array([0.429, 0.375, 0.525, 0.540], dtype=np.float32)
    )
    assert set(rows["hemi"]) == {"rh"}


def test_the_rows_are_written_and_read_back_as_they_are(tmp_path: Path) -> None:
    rows = pd.concat(
        [regional_rows(UID, hemi, stats_of(FOUR)) for hemi in ("lh", "rh")], ignore_index=True
    )
    path = tmp_path / "regional.parquet"
    store.write_table(path, rows, schemas.REGIONAL, sort_by=schemas.KEYS["regional"])
    pd.testing.assert_frame_equal(store.read_table(path, schemas.REGIONAL), rows)


def test_a_file_without_rows_gives_an_empty_frame_that_the_table_takes(tmp_path: Path) -> None:
    rows = regional_rows(UID, "lh", parse_stats(HEADER))
    assert len(rows) == 0
    assert list(rows.columns) == schemas.REGIONAL.names
    assert str(rows["num_vert"].dtype) == "int32"
    assert str(rows["thick_avg_mm"].dtype) == "float32"
    path = tmp_path / "regional.parquet"
    store.write_table(path, rows, schemas.REGIONAL, sort_by=schemas.KEYS["regional"])
    assert len(store.read_table(path, schemas.REGIONAL)) == 0


@pytest.mark.parametrize(
    ("thick", "std"),
    [("nan", "0.5"), ("-nan", "nan"), ("2.5", "nan"), ("inf", "0.5"), ("1e39", "0.5")],
)
def test_a_parcel_without_a_finite_thickness_is_left_out_and_said(
    thick: str, std: str, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    parcels = (*FOUR, ("frontalpole", 0, 0, 0, thick, std), ("insula", 3, 2, 5, "nan", "nan"))
    with caplog.at_level(logging.WARNING, logger="surfcensus"):
        rows = regional_rows(UID, "lh", stats_of(parcels))
    assert list(rows["parcel"]) == ["bankssts", "cuneus", "lingual", "superiorfrontal"]
    assert list(rows["num_vert"]) == [1338, 2071, 4880, 10123]
    assert np.isfinite(rows["thick_avg_mm"]).all()
    messages = [record.getMessage() for record in caplog.records]
    assert messages == [
        f"{UID} lh: 2 of 6 parcels have no finite thickness in the stats file and are left out"
    ]
    # what is left fits a table without missing values
    path = tmp_path / "regional.parquet"
    store.write_table(path, rows, schemas.REGIONAL, sort_by=schemas.KEYS["regional"])


def test_no_warning_for_a_file_that_is_whole(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger="surfcensus"):
        regional_rows(UID, "lh", stats_of(FOUR))
    assert caplog.records == []


@pytest.mark.parametrize("column", ["NumVert", "SurfArea", "GrayVol"])
def test_a_count_that_does_not_fit_the_table_is_refused(column: str) -> None:
    def one(value: int) -> StatsFile:
        counts = {"NumVert": 2071, "SurfArea": 1303, "GrayVol": 2497} | {column: value}
        return stats_of([("cuneus", *counts.values(), "1.820", "0.375")])

    rows = regional_rows(UID, "lh", one(2**31 - 1))
    table_column = {"NumVert": "num_vert", "SurfArea": "surf_area_mm2", "GrayVol": "gray_vol_mm3"}
    assert int(rows[table_column[column]].iloc[0]) == 2**31 - 1
    assert str(rows[table_column[column]].dtype) == "int32"
    with pytest.raises(ValueError, match=f"column {column}: a value does not fit 32 bits"):
        regional_rows(UID, "lh", one(2**31))
    with pytest.raises(ValueError, match=f"column {column}: a value does not fit 32 bits"):
        regional_rows(UID, "lh", one(-(2**31) - 1))


def test_a_file_that_is_no_parcel_table_is_refused() -> None:
    aseg = parse_stats("# ColHeaders Index SegId StructName\n1 4 Left-Lateral-Ventricle\n")
    with pytest.raises(ValueError, match="not a parcel table: no column NumVert"):
        regional_rows(UID, "lh", aseg)
    with pytest.raises(ValueError, match="data row 1, column NumVert: not an integer"):
        regional_rows(UID, "lh", stats_of([("cuneus", "many", 1303, 2497, "1.820", "0.375")]))


# ---------------------------------------------------------------------------------------
# region_mean_thickness


def test_the_mean_is_weighted_by_area() -> None:
    rows = table({("lh", "cuneus"): (100, 2.0), ("lh", "lingual"): (300, 3.0)})
    assert region_mean_thickness(rows, ["cuneus", "lingual"]) == pytest.approx(2.75, abs=1e-6)
    # not the plain mean (2.5) and not the mean weighted by vertex count (2.333)
    assert region_mean_thickness(rows, ["cuneus"]) == pytest.approx(2.0, abs=1e-6)
    assert region_mean_thickness(rows, ("lingual",)) == pytest.approx(3.0, abs=1e-6)
    assert type(region_mean_thickness(rows, ["cuneus"])) is float


def test_the_hemispheres_are_pooled_not_averaged() -> None:
    rows = table(
        {
            ("lh", "cuneus"): (100, 2.0),
            ("lh", "lingual"): (100, 2.0),
            ("rh", "cuneus"): (600, 3.0),
        }
    )
    # (200 * 2 + 600 * 3) / 800; the mean of the two hemispheres' means would be 2.5
    assert region_mean_thickness(rows, ["cuneus", "lingual"]) == pytest.approx(2.75, abs=1e-6)


def test_one_hemisphere_alone_gives_its_own_mean() -> None:
    both = table(
        {
            ("lh", "cuneus"): (100, 2.0),
            ("lh", "lingual"): (300, 3.0),
            ("rh", "cuneus"): (500, 4.0),
            ("rh", "lingual"): (100, 1.0),
        }
    )
    lh, rh = both[both["hemi"] == "lh"], both[both["hemi"] == "rh"]
    assert region_mean_thickness(lh, ["cuneus", "lingual"]) == pytest.approx(2.75, abs=1e-6)
    assert region_mean_thickness(rh, ["cuneus", "lingual"]) == pytest.approx(3.5, abs=1e-6)
    assert region_mean_thickness(both, ["cuneus", "lingual"]) == pytest.approx(3.2, abs=1e-6)


def test_parcels_that_the_rows_lack_are_left_out() -> None:
    rows = table({("lh", "cuneus"): (100, 2.0), ("lh", "lingual"): (300, 3.0)})
    assert region_mean_thickness(rows, ["cuneus", "pericalcarine"]) == pytest.approx(2.0, abs=1e-6)
    assert math.isnan(region_mean_thickness(rows, ["pericalcarine", "lateraloccipital"]))
    assert math.isnan(region_mean_thickness(rows, []))
    assert math.isnan(region_mean_thickness(rows.iloc[:0], ["cuneus"]))


def test_rows_of_other_parcels_do_not_count() -> None:
    rows = table(
        {
            ("lh", "cuneus"): (100, 2.0),
            ("lh", "insula"): (5000, 4.0),
            ("rh", "superiorfrontal"): (9000, 1.0),
        }
    )
    assert region_mean_thickness(rows, ["cuneus"]) == pytest.approx(2.0, abs=1e-6)


def test_a_parcel_named_twice_counts_once() -> None:
    rows = table({("lh", "cuneus"): (100, 2.0), ("lh", "lingual"): (300, 3.0)})
    assert region_mean_thickness(rows, ["cuneus", "cuneus", "lingual"]) == pytest.approx(
        2.75, abs=1e-6
    )


def test_a_row_without_area_or_thickness_has_no_weight() -> None:
    rows = table(
        {
            ("lh", "cuneus"): (100, 2.0),
            ("lh", "lingual"): (0, 9.0),
            ("rh", "cuneus"): (300, float("nan")),
            ("rh", "lingual"): (-50, 9.0),
        }
    )
    assert region_mean_thickness(rows, ["cuneus", "lingual"]) == pytest.approx(2.0, abs=1e-6)
    assert math.isnan(region_mean_thickness(rows[rows["hemi"] == "rh"], ["cuneus", "lingual"]))
    assert math.isnan(region_mean_thickness(rows, ["lingual"]))


def test_the_key_columns_are_not_needed_and_nullable_columns_are_taken() -> None:
    rows = table({("lh", "cuneus"): (100, 2.0), ("rh", "cuneus"): (300, 3.0)})
    bare = rows[["parcel", "surf_area_mm2", "thick_avg_mm"]]
    assert region_mean_thickness(bare, ["cuneus"]) == pytest.approx(2.75, abs=1e-6)
    nullable = rows.astype({"surf_area_mm2": "Int32", "thick_avg_mm": "Float32"})
    nullable.loc[1, "surf_area_mm2"] = pd.NA
    assert region_mean_thickness(nullable, ["cuneus"]) == pytest.approx(2.0, abs=1e-6)
    nullable.loc[0, "thick_avg_mm"] = pd.NA
    assert math.isnan(region_mean_thickness(nullable, ["cuneus"]))


@pytest.mark.parametrize("column", ["parcel", "surf_area_mm2", "thick_avg_mm"])
def test_rows_without_a_needed_column_are_refused(column: str) -> None:
    rows = table({("lh", "cuneus"): (100, 2.0)}).drop(columns=column)
    with pytest.raises(ValueError, match=f"no column {column}$"):
        region_mean_thickness(rows, ["cuneus"])


def test_rows_of_two_subjects_are_refused() -> None:
    one = table({("lh", "cuneus"): (100, 2.0)})
    other = table({("lh", "cuneus"): (300, 3.0)}, uid="fedcba9876543210")
    with pytest.raises(ValueError, match="more than one subject"):
        region_mean_thickness(pd.concat([one, other], ignore_index=True), ["cuneus"])
    with pytest.raises(ValueError, match="more than one subject"):
        occ_ctrl_ratio(pd.concat([one, other], ignore_index=True), REGIONS)


def test_two_rows_for_one_parcel_of_a_hemisphere_are_refused() -> None:
    one = table({("lh", "cuneus"): (100, 2.0), ("rh", "cuneus"): (100, 2.0)})
    assert region_mean_thickness(one, ["cuneus"]) == pytest.approx(2.0, abs=1e-6)
    twice = pd.concat([one, one.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="more than one row for a parcel of one hemisphere"):
        region_mean_thickness(twice, ["cuneus"])


# ---------------------------------------------------------------------------------------
# occ_ctrl_ratio


def both_regions() -> dict[tuple[str, str], tuple[float, float]]:
    return {
        ("lh", "cuneus"): (100, 2.0),
        ("lh", "lateraloccipital"): (300, 3.0),
        ("rh", "pericalcarine"): (200, 1.0),
        ("lh", "superiorfrontal"): (400, 2.5),
        ("rh", "caudalmiddlefrontal"): (100, 3.0),
        ("rh", "rostralmiddlefrontal"): (500, 2.0),
        # in the search region, and in neither list of the ratio
        ("lh", "precuneus"): (5000, 9.0),
        ("lh", "fusiform"): (5000, 9.0),
        ("lh", "insula"): (5000, 9.0),
    }


def test_the_ratio_of_the_occipital_to_the_control_parcels() -> None:
    occipital, control, ratio = occ_ctrl_ratio(table(both_regions()), REGIONS)
    assert occipital == pytest.approx((200 + 900 + 200) / 600, abs=1e-6)
    assert control == pytest.approx((1000 + 300 + 1000) / 1000, abs=1e-6)
    assert ratio == pytest.approx((1300 / 600) / 2.3, abs=1e-6)
    assert all(type(value) is float for value in (occipital, control, ratio))


def test_the_ratio_follows_the_region_config() -> None:
    rows = table(both_regions())
    regions = RegionConfig(occipital=("cuneus",), control=("superiorfrontal", "insula"))
    occipital, control, ratio = occ_ctrl_ratio(rows, regions)
    assert occipital == pytest.approx(2.0, abs=1e-6)
    assert control == pytest.approx((1000 + 45000) / 5400, abs=1e-5)
    assert ratio == pytest.approx(2.0 * 5400 / 46000, abs=1e-6)
    # the other lists of the config define the search region and do not enter the ratio
    other = RegionConfig(posterior=("insula",), medial=("insula",), inferior=("lingual",))
    assert occ_ctrl_ratio(rows, other) == occ_ctrl_ratio(rows, REGIONS)


def test_no_ratio_without_one_of_the_regions() -> None:
    rows = table(both_regions())
    control_parcels = ("superiorfrontal", "caudalmiddlefrontal", "rostralmiddlefrontal")
    occipital, control, ratio = occ_ctrl_ratio(rows[~rows["parcel"].isin(control_parcels)], REGIONS)
    assert occipital == pytest.approx(1300 / 600, abs=1e-6)
    assert math.isnan(control) and math.isnan(ratio)
    occipital, control, ratio = occ_ctrl_ratio(rows[rows["parcel"].isin(control_parcels)], REGIONS)
    assert math.isnan(occipital) and math.isnan(ratio)
    assert control == pytest.approx(2.3, abs=1e-6)
    assert all(math.isnan(value) for value in occ_ctrl_ratio(rows.iloc[:0], REGIONS))


def test_no_ratio_over_a_control_thickness_of_zero() -> None:
    rows = table({("lh", "cuneus"): (100, 2.0), ("lh", "superiorfrontal"): (400, 0.0)})
    occipital, control, ratio = occ_ctrl_ratio(rows, REGIONS)
    assert (occipital, control) == (2.0, 0.0)
    assert math.isnan(ratio)


def test_the_ratio_of_one_hemisphere() -> None:
    rows = table(both_regions())
    occipital, control, ratio = occ_ctrl_ratio(rows[rows["hemi"] == "lh"], REGIONS)
    assert occipital == pytest.approx(1100 / 400, abs=1e-6)
    assert control == pytest.approx(2.5, abs=1e-6)
    assert ratio == pytest.approx(1.1, abs=1e-6)


def test_from_stats_text_to_ratio() -> None:
    lh = regional_rows(UID, "lh", stats_of(FOUR))
    rh = regional_rows(
        UID,
        "rh",
        stats_of(
            [
                ("pericalcarine", 2000, 1500, 2600, "1.500", "0.400"),
                ("rostralmiddlefrontal", 8000, 5500, 13000, "2.300", "0.500"),
            ]
        ),
    )
    occipital, control, ratio = occ_ctrl_ratio(pd.concat([lh, rh], ignore_index=True), REGIONS)
    expected_occipital = (1303 * 1.820 + 3130 * 2.004 + 1500 * 1.5) / (1303 + 3130 + 1500)
    expected_control = (6880 * 2.704 + 5500 * 2.3) / (6880 + 5500)
    assert occipital == pytest.approx(expected_occipital, abs=1e-6)
    assert control == pytest.approx(expected_control, abs=1e-6)
    assert ratio == pytest.approx(expected_occipital / expected_control, abs=1e-6)


# ---------------------------------------------------------------------------------------
# the synthetic cohort


def subject_rows(subject_dir: Path, hemis: Sequence[Hemi] = HEMIS) -> pd.DataFrame:
    return pd.concat(
        [
            regional_rows(UID, hemi, read_stats(subject_dir / "stats" / f"{hemi}.aparc.stats"))
            for hemi in hemis
        ],
        ignore_index=True,
    )


def surface_mean(subject_dir: Path, parcels: Sequence[str]) -> float:
    """The area-weighted mean of the parcels' mean thickness, from the surfaces themselves."""
    weighted, total = 0.0, 0.0
    for hemi in ("lh", "rh"):
        thickness = fs.read_morph_data(str(subject_dir / "surf" / f"{hemi}.thickness"))
        labels, _, names = fs.read_annot(str(subject_dir / "label" / f"{hemi}.aparc.annot"))
        white, faces = fs.read_geometry(str(subject_dir / "surf" / f"{hemi}.white"))
        area = mesh_vertex_areas(white, faces)
        for index, name in enumerate(names):
            members = labels == index
            if name.decode() in parcels and members.any():
                weighted += float(area[members].sum()) * float(thickness[members].mean())
                total += float(area[members].sum())
    return weighted / total


def test_the_cohort_rows_follow_the_surfaces(small_cohort: SyntheticCohort) -> None:
    for subject in small_cohort.subjects:
        rows = subject_rows(subject.path)
        assert len(rows) == 22  # 11 parcels have vertices on the synthetic annotation
        assert not rows.duplicated(["hemi", "parcel"]).any()
        labels, _, names = fs.read_annot(str(subject.path / "label" / "lh.aparc.annot"))
        thickness = fs.read_morph_data(str(subject.path / "surf" / "lh.thickness"))
        lh = rows[rows["hemi"] == "lh"].set_index("parcel")
        for index, name in enumerate(names):
            members = labels == index
            if name.decode() in lh.index:
                assert int(lh.loc[name.decode(), "num_vert"]) == int(members.sum())
                assert float(lh.loc[name.decode(), "thick_avg_mm"]) == pytest.approx(
                    float(thickness[members].mean()), abs=6e-4
                )
        occipital, control, ratio = occ_ctrl_ratio(rows, REGIONS)
        # the file rounds areas to 1 mm2 and thickness to 0.001 mm
        assert occipital == pytest.approx(surface_mean(subject.path, REGIONS.occipital), abs=2e-3)
        assert control == pytest.approx(surface_mean(subject.path, REGIONS.control), abs=2e-3)
        assert ratio == pytest.approx(occipital / control, abs=1e-12)


def test_the_subjects_with_a_patch_have_the_highest_ratios(synth_cohort: SyntheticCohort) -> None:
    ratios = {
        subject.path.name: occ_ctrl_ratio(subject_rows(subject.path), REGIONS)[2]
        for subject in synth_cohort.subjects
        if subject.path.name not in synth_cohort.no_stats
    }
    assert len(ratios) == 31
    assert all(math.isfinite(ratio) for ratio in ratios.values())
    ranked = sorted(ratios, key=lambda name: -ratios[name])
    assert set(ranked[:3]) == set(synth_cohort.bad) == {"sub-004", "sub-013", "sub-022"}
    # a 5 mm patch in a search-region parcel raises the ratio by more than a tenth; the
    # other subjects differ by noise
    assert min(ratios[name] for name in ranked[:3]) > 1.1
    assert max(ratios[name] for name in ranked[3:]) < 1.05


def test_a_subject_with_one_stats_file_is_scored_on_that_hemisphere(
    small_cohort: SyntheticCohort,
) -> None:
    subject = small_cohort.subjects[0].path
    lh, both = subject_rows(subject, ["lh"]), subject_rows(subject)
    assert set(lh["hemi"]) == {"lh"}
    ratio = occ_ctrl_ratio(lh, REGIONS)
    assert all(math.isfinite(value) for value in ratio)
    assert ratio == occ_ctrl_ratio(both[both["hemi"] == "lh"], REGIONS)
    assert ratio != occ_ctrl_ratio(both, REGIONS)


# ---------------------------------------------------------------------------------------
# the real subject, where the network tests have fetched it


def test_the_real_subject(cached_subjects: Path, tmp_path: Path) -> None:
    stats = cached_subjects / "ds000102-sub-01" / "stats"
    if not all((stats / f"{hemi}.aparc.stats").is_file() for hemi in ("lh", "rh")):
        pytest.skip("the stats files are not in the pytest cache (the network tests fetch them)")
    rows = subject_rows(stats.parent)
    assert rows.groupby("hemi").size().to_dict() == {"lh": 34, "rh": 34}
    lh = rows[rows["hemi"] == "lh"].set_index("parcel")
    assert lh.loc["cuneus"].to_dict() == {
        "subject_uid": UID,
        "hemi": "lh",
        "num_vert": 2071,
        "surf_area_mm2": 1303,
        "gray_vol_mm3": 2497,
        "thick_avg_mm": np.float32(1.820),
        "thick_std_mm": np.float32(0.375),
    }
    occipital, control, ratio = occ_ctrl_ratio(rows, REGIONS)
    assert ratio == pytest.approx(0.811, abs=0.001)
    assert occipital == pytest.approx(1.947, abs=0.001)
    assert control == pytest.approx(2.402, abs=0.001)
    path = tmp_path / "regional.parquet"
    store.write_table(path, rows, schemas.REGIONAL, sort_by=schemas.KEYS["regional"])
    back = store.read_table(path, schemas.REGIONAL)
    assert occ_ctrl_ratio(back, REGIONS) == (occipital, control, ratio)
