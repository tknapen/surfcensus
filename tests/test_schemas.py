"""Tests of surfcensus/schemas.py: the parquet tables of a run directory."""

from __future__ import annotations

import io

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from surfcensus.config import IntensityConfig, ScoringConfig
from surfcensus.schemas import DEFAULT_INTENSITY_DEPTHS, KEYS, SCHEMAS, intensity_schema

TABLES = ("regional", "vertex_summary", "clusters", "clusters_native", "scores", "intensity")
ALLOWED_TYPES = {pa.string(), pa.bool_(), pa.int8(), pa.int16(), pa.int32(), pa.float32()}


def nullable(table: str) -> set[str]:
    return {field.name for field in SCHEMAS[table] if field.nullable}


def test_every_table_has_a_schema_and_a_key() -> None:
    assert tuple(SCHEMAS) == TABLES
    assert tuple(KEYS) == TABLES


@pytest.mark.parametrize("table", TABLES)
def test_key_columns_come_first_and_always_have_a_value(table: str) -> None:
    schema, key = SCHEMAS[table], KEYS[table]
    assert tuple(schema.names[: len(key)]) == key
    assert key[0] == "subject_uid"
    assert len(set(schema.names)) == len(schema.names)
    for name in key:
        assert not schema.field(name).nullable, name
    assert {field.type for field in schema} <= ALLOWED_TYPES
    assert schema.metadata is None


def test_keys_are_those_of_the_specification() -> None:
    assert KEYS == {
        "regional": ("subject_uid", "hemi", "parcel"),
        "vertex_summary": ("subject_uid", "hemi"),
        "clusters": ("subject_uid", "reference_id", "hemi", "region", "rank"),
        "clusters_native": ("subject_uid", "reference_id", "hemi", "region", "rank"),
        "scores": ("subject_uid", "reference_id"),
        "intensity": ("subject_uid", "hemi"),
    }


def test_regional_columns() -> None:
    schema = SCHEMAS["regional"]
    assert schema.names == [
        "subject_uid",
        "hemi",
        "parcel",
        "num_vert",
        "surf_area_mm2",
        "gray_vol_mm3",
        "thick_avg_mm",
        "thick_std_mm",
    ]
    assert [schema.field(n).type for n in schema.names[3:6]] == [pa.int32()] * 3
    assert [schema.field(n).type for n in schema.names[6:]] == [pa.float32()] * 2
    assert not nullable("regional")


def test_vertex_summary_columns() -> None:
    schema = SCHEMAS["vertex_summary"]
    assert schema.names == [
        "subject_uid",
        "hemi",
        "n_native",
        "thickness_max_mm",
        "template_id",
        "n_fallback",
        "n_missing_cortex",
        "global_median_mm",
        "region_kind",
        "search_n",
        "search_n_exact",
        "search_n_near",
        "search_n_zero",
        "control_n",
        "control_n_exact",
        "control_n_near",
        "control_n_zero",
        "ceiling_frac",
        "control_ceiling_frac",
        "annot_search_n",
        "annot_search_n_near",
        "annot_search_n_zero",
        "ceiling_frac_annot",
        "flags",
    ]
    # what a subject without a template, a map or an annotation cannot have
    assert set(schema.names) - nullable("vertex_summary") == {
        "subject_uid",
        "hemi",
        "n_native",
        "thickness_max_mm",
        "region_kind",
        "flags",
    }
    assert schema.field("n_native").type == pa.int32()
    assert schema.field("ceiling_frac").type == pa.float32()
    assert schema.field("template_id").type == pa.string()


def test_cluster_columns() -> None:
    clusters, native = SCHEMAS["clusters"], SCHEMAS["clusters_native"]
    assert clusters.names == [
        "subject_uid",
        "reference_id",
        "hemi",
        "region",
        "rank",
        "n_vertices",
        "area_mm2",
        "mass",
        "excess_mass",
        "peak_z",
        "peak_vertex",
        "centroid_vertex",
    ]
    assert native.names == [
        "subject_uid",
        "reference_id",
        "hemi",
        "region",
        "rank",
        "native_vertex",
        "tkr_x",
        "tkr_y",
        "tkr_z",
        "template_id",
    ]
    assert clusters.field("rank").type == native.field("rank").type == pa.int16()
    assert not nullable("clusters") and not nullable("clusters_native")
    # the two tables join on their common key
    assert [clusters.field(n).type for n in KEYS["clusters"]] == [
        native.field(n).type for n in KEYS["clusters_native"]
    ]


def test_scores_columns() -> None:
    schema = SCHEMAS["scores"]
    assert schema.field("primary").type == pa.bool_()
    assert schema.field("n_components").type == pa.int8()
    assert schema.field("rank").type == pa.int32()
    assert set(schema.names) - nullable("scores") == {
        "subject_uid",
        "reference_id",
        "primary",
        "dataset",
        "score_kind",
        "n_components",
        "rank",
        "mass_kind",
        "ceiling_region",
        "flags",
        "score_config_hash",
    }
    assert schema.names == [
        "subject_uid",
        "reference_id",
        "primary",
        "dataset",
        "score_kind",
        "stratum",
        "template_id",
        "n_reference",
        "composite",
        "composite_component",
        "n_components",
        "rank",
        "mass_kind",
        "mass_search",
        "mass_medial",
        "mass_inferior",
        "mass_lateral",
        "mass_control",
        "area_search_mm2",
        "peak_z_search",
        "ceiling_frac",
        "control_ceiling_frac",
        "ceiling_exact_frac",
        "zero_frac",
        "ceiling_frac_annot",
        "ceiling_region",
        "occ_thick_mm",
        "ctrl_thick_mm",
        "occ_ctrl_ratio",
        "std_mass_medial",
        "std_mass_inferior",
        "std_ceiling_frac",
        "std_occ_ctrl_ratio",
        "holes_lh",
        "holes_rh",
        "trunc_frac",
        "flags",
        "score_config_hash",
    ]
    assert schema.field("holes_lh").type == schema.field("n_reference").type == pa.int32()
    # every component of the composite has its raw and its standardised column
    for component in ScoringConfig().components:
        assert schema.field(component).type == pa.float32()
        assert schema.field(f"std_{component}").type == pa.float32()


def test_the_intensity_schema_follows_the_number_of_depths() -> None:
    assert DEFAULT_INTENSITY_DEPTHS == len(IntensityConfig().depths)
    assert SCHEMAS["intensity"] == intensity_schema(DEFAULT_INTENSITY_DEPTHS)
    two = intensity_schema(2)
    assert two.names == [
        "subject_uid",
        "hemi",
        "region_kind",
        "search_n",
        "control_n",
        "outer_outlier_frac",
        "gap_frac",
        "search_median_d0",
        "search_median_d1",
        "control_median_d0",
        "control_median_d1",
    ]
    assert {f.name for f in two if not f.nullable} == {"subject_uid", "hemi", "region_kind"}
    assert len(SCHEMAS["intensity"].names) == 7 + 2 * DEFAULT_INTENSITY_DEPTHS


@pytest.mark.parametrize("table", TABLES)
def test_an_empty_table_of_each_schema_survives_parquet(table: str) -> None:
    schema = SCHEMAS[table]
    empty = schema.empty_table()
    buffer = io.BytesIO()
    pq.write_table(empty, buffer, compression="zstd")
    back = pq.read_table(io.BytesIO(buffer.getvalue()))
    assert back.schema.equals(schema, check_metadata=False)
    assert back.num_rows == 0
    assert list(empty.to_pandas().columns) == schema.names


def test_a_frame_casts_to_its_schema_and_a_null_key_does_not() -> None:
    schema = SCHEMAS["clusters_native"]
    frame = pd.DataFrame(
        {
            "subject_uid": ["0123456789abcdef"],
            "reference_id": ["set"],
            "hemi": ["lh"],
            "region": ["medial"],
            "rank": [1],
            "native_vertex": [12345],
            "tkr_x": [-8.5],
            "tkr_y": [-80.25],
            "tkr_z": [4.0],
            "template_id": ["0123456789ab"],
        }
    )
    table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
    assert table.schema.equals(schema, check_metadata=False)
    assert table.column("tkr_y").to_pylist() == [-80.25]
    frame.loc[0, "hemi"] = None
    with pytest.raises((pa.ArrowInvalid, ValueError)):
        pa.Table.from_pandas(frame, schema=schema, preserve_index=False).validate(full=True)
