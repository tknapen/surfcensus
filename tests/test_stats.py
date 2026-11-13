"""fsio/stats: the measures and the table of ``?h.aparc.stats`` and ``aseg.stats``.

The header texts below have the lines of the stats files of two real subjects (ds000102
sub-01, FreeSurfer 6.0.1; ds003020 sub-UTS01, FreeSurfer 7.3.2; both under fMRIPrep), with
the host, the user, the subjects directory and the subject name replaced by canaries. The
tests at the end read the real files where the network tests have put them in the pytest
cache, and are skipped elsewhere.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from synth import write_synthetic_subject  # pyrefly: ignore[missing-import]
from synth_tierb import CANARIES, add_tier_a  # pyrefly: ignore[missing-import]

from surfcensus.fsio.stats import (
    StatsFile,
    aparc_frame,
    euler_number,
    parse_stats,
    read_stats,
    surface_holes,
)

SUBJECTS_DIR = "/data/ds999999-fmriprep/sourcedata/freesurfer"
PRIVATE = ("canary-host", "canary-user", "ds999999", "/data/", "sub-canary", "cmdline-canary")
"""What the headers below carry where the real ones carry a host, a user, a path with the
accession, a subject name and a command line."""

_MACHINE = """\
# sysname  Linux
# hostname canary-host
# machine  x86_64
# user     canary-user
#
# SUBJECTS_DIR /data/ds999999-fmriprep/sourcedata/freesurfer
# anatomy_type surface
# subjectname sub-canary
# hemi lh
# AnnotationFile ../label/lh.aparc.annot
# AnnotationFileTimeStamp 2022/02/10 10:17:18
"""

_TABLE_COLUMNS = """\
# NTableCols 10
# TableCol  1 ColHeader StructName
# TableCol  1 FieldName Structure Name
# TableCol  1 Units     NA
# TableCol  2 ColHeader NumVert
# TableCol  2 FieldName Number of Vertices
# TableCol  2 Units     unitless
# TableCol  5 ColHeader ThickAvg
# TableCol  5 FieldName Average Thickness
# TableCol  5 Units     mm
# ColHeaders StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd
"""

FS6_APARC_HEADER = (
    """\
# Table of FreeSurfer cortical parcellation anatomical statistics
#
# CreationTime 2022/02/10-10:37:09-GMT
# generating_program mris_anatomical_stats
# cvs_version $Id: mris_anatomical_stats.c,v 1.79 2016/03/14 15:15:34 greve Exp $
# mrisurf.c-cvs_version $Id: mrisurf.c,v 1.781.2.6 2016/12/27 16:47:14 zkaufman Exp $
# cmdline cmdline-canary -th3 -mgz -cortex ../label/lh.cortex.label -f ../stats/lh.aparc.stats -b -a ../label/lh.aparc.annot -c ../label/aparc.annot.ctab sub-canary lh white
"""  # noqa: E501
    + _MACHINE
    + """\
# Measure Cortex, NumVert, Number of Vertices, 140304, unitless
# Measure Cortex, WhiteSurfArea, White Surface Total Area, 91950.9, mm^2
# Measure Cortex, MeanThickness, Mean Thickness, 2.39047, mm
# Measure BrainSeg, BrainSegVol, Brain Segmentation Volume, 1158358.000000, mm^3
# Measure BrainSegNotVent, BrainSegVolNotVent, Brain Segmentation Volume Without Ventricles, 1132855.000000, mm^3
# Measure BrainSegNotVentSurf, BrainSegVolNotVentSurf, Brain Segmentation Volume Without Ventricles from Surf, 1131601.279922, mm^3
# Measure Cortex, CortexVol Total cortical gray matter volume, 484608.441133, mm^3
# Measure SupraTentorial, SupraTentorialVol, Supratentorial volume, 1017319.279922, mm^3
# Measure SupraTentorialNotVent, SupraTentorialVolNotVent, Supratentorial volume, 997485.279922, mm^3
# Measure EstimatedTotalIntraCranialVol, eTIV, Estimated Total Intracranial Volume, 1519867.314105, mm^3
"""  # noqa: E501
    + _TABLE_COLUMNS
)
"""The header of a FreeSurfer 6.0.1 ``lh.aparc.stats``: ten measures, the seventh without
the comma after its name."""

FS7_APARC_HEADER = (
    """\
# Table of FreeSurfer cortical parcellation anatomical statistics
#
# CreationTime 2023/04/22-05:38:01-GMT
# generating_program mris_anatomical_stats
# cvs_version 7.3.2
# mrisurf.c-cvs_version 7.3.2
# cmdline cmdline-canary -th3 -mgz -noglobal -cortex ../label/lh.cortex.label -f ../stats/lh.aparc.stats -b -a ../label/lh.aparc.annot -c ../label/aparc.annot.ctab sub-canary lh white
"""  # noqa: E501
    + _MACHINE
    + """\
# Measure Cortex, NumVert, Number of Vertices, 123304, unitless
# Measure Cortex, WhiteSurfArea, White Surface Total Area, 85444.1, mm^2
# Measure Cortex, MeanThickness, Mean Thickness, 2.52549, mm
"""
    + _TABLE_COLUMNS
)
"""The header of a FreeSurfer 7.3.2 ``lh.aparc.stats`` (``-noglobal``): three measures."""

APARC_ROWS = """\
bankssts                                 1338    953   2385  2.610 0.429     0.096     0.018        7     1.0
caudalanteriorcingulate                  1094    712   2049  2.481 0.724     0.119     0.013       14     0.6
caudalmiddlefrontal                      4717   3013   7436  2.358 0.466     0.104     0.021       41     3.9
cuneus                                   2071   1303   2497  1.820 0.375     0.138     0.034       30     2.9
"""  # noqa: E501
"""The first four data rows of the real 6.0.1 file."""

ASEG = """\
# Title Segmentation Statistics
#
# generating_program mri_segstats
# cvs_version $Id: mri_segstats.c,v 1.121 2016/05/31 17:27:11 greve Exp $
# cmdline cmdline-canary --seg mri/aseg.mgz --sum stats/aseg.stats --pv mri/norm.mgz --empty --brainmask mri/brainmask.mgz --brain-vol-from-seg --excludeid 0 --excl-ctxgmwm --supratent --subcortgray --in mri/norm.mgz --in-intensity-name norm --in-intensity-units MR --etiv --surf-wm-vol --surf-ctx-vol --totalgray --euler --ctab /opt/freesurfer/ASegStatsLUT.txt --subject sub-canary
# sysname  Linux
# hostname canary-host
# machine  x86_64
# user     canary-user
# anatomy_type volume
#
# SUBJECTS_DIR /data/ds999999-fmriprep/sourcedata/freesurfer
# subjectname sub-canary
# Measure BrainSeg, BrainSegVol, Brain Segmentation Volume, 1158358.000000, mm^3
# Measure Cortex, CortexVol, Total cortical gray matter volume, 484608.441133, mm^3
# Measure BrainSegVol-to-eTIV, BrainSegVol-to-eTIV, Ratio of BrainSegVol to eTIV, 0.762144, unitless
# Measure lhSurfaceHoles, lhSurfaceHoles, Number of defect holes in lh surfaces prior to fixing, 41, unitless
# Measure rhSurfaceHoles, rhSurfaceHoles, Number of defect holes in rh surfaces prior to fixing, 26, unitless
# Measure SurfaceHoles, SurfaceHoles, Total number of defect holes in surfaces prior to fixing, 67, unitless
# Measure EstimatedTotalIntraCranialVol, eTIV, Estimated Total Intracranial Volume, 1519867.314105, mm^3
# SegVolFile mri/aseg.mgz
# SegVolFileTimeStamp  2022/02/10 10:13:34
# ColorTable /opt/freesurfer/ASegStatsLUT.txt
# InVolFile  mri/norm.mgz
# Excluding Cortical Gray and White Matter
# ExcludeSegId 0 2 3 41 42
# VoxelVolume_mm3 1
# TableCol  1 ColHeader Index
# TableCol  1 FieldName Index
# TableCol  1 Units     NA
# NRows 45
# NTableCols 10
# ColHeaders  Index SegId NVoxels Volume_mm3 StructName normMean normStdDev normMin normMax normRange
  1   4      8830     9198.1  Left-Lateral-Ventricle            20.8310    13.9629     0.0000    83.0000    83.0000
  2   5       490      515.3  Left-Inf-Lat-Vent                 34.1041    15.8871     0.0000    74.0000    74.0000
"""  # noqa: E501
"""A FreeSurfer 6.0.1 ``aseg.stats``, with part of its measures and two of its 45 rows."""

APARC_COLUMNS = (
    "StructName",
    "NumVert",
    "SurfArea",
    "GrayVol",
    "ThickAvg",
    "ThickStd",
    "MeanCurv",
    "GausCurv",
    "FoldInd",
    "CurvInd",
)
MINIMAL_HEADER = "# ColHeaders " + " ".join(APARC_COLUMNS) + "\n"


def parcel(
    name: str, num_vert: object, area: object, volume: object, thick: object, std: object
) -> str:
    """One data row; the numbers are written as they are given, so a test can plant any text."""
    return f"{name:<40}{num_vert!s:>5}  {area!s:>5}  {volume!s:>5}  {thick!s:>5} {std!s:>5}     0.096     0.018        7     1.0\n"  # noqa: E501


# ---------------------------------------------------------------------------------------
# measures


def test_the_measures_of_a_freesurfer_6_header_with_its_malformed_line() -> None:
    stats = parse_stats(FS6_APARC_HEADER + APARC_ROWS)
    assert stats.measures == {
        "NumVert": (140304.0, "unitless"),
        "WhiteSurfArea": (91950.9, "mm^2"),
        "MeanThickness": (2.39047, "mm"),
        "BrainSegVol": (1158358.0, "mm^3"),
        "BrainSegVolNotVent": (1132855.0, "mm^3"),
        "BrainSegVolNotVentSurf": (1131601.279922, "mm^3"),
        "CortexVol": (484608.441133, "mm^3"),
        "SupraTentorialVol": (1017319.279922, "mm^3"),
        "SupraTentorialVolNotVent": (997485.279922, "mm^3"),
        "eTIV": (1519867.314105, "mm^3"),
    }


def test_the_measures_of_a_freesurfer_7_header() -> None:
    stats = parse_stats(FS7_APARC_HEADER + APARC_ROWS)
    assert stats.measures == {
        "NumVert": (123304.0, "unitless"),
        "WhiteSurfArea": (85444.1, "mm^2"),
        "MeanThickness": (2.52549, "mm"),
    }


def test_the_malformed_line_gives_the_name_before_the_description() -> None:
    line = "# Measure Cortex, CortexVol Total cortical gray matter volume, 484608.441133, mm^3\n"
    assert parse_stats(line).measures == {"CortexVol": (484608.441133, "mm^3")}


def test_the_name_is_the_second_field_and_not_the_first() -> None:
    # the two differ in most lines of a real file: "Cortex, NumVert", "BrainSeg, BrainSegVol"
    line = "# Measure EstimatedTotalIntraCranialVol, eTIV, Estimated Total Intracranial Volume, 1519867.314105, mm^3\n"  # noqa: E501
    assert list(parse_stats(line).measures) == ["eTIV"]


def test_a_description_with_commas_does_not_move_value_and_unit() -> None:
    line = "# Measure Mask, MaskVol, Mask Volume, in voxels, of the brain, 1372418.000000, mm^3\n"
    assert parse_stats(line).measures == {"MaskVol": (1372418.0, "mm^3")}


@pytest.mark.parametrize(
    ("number", "value"),
    [("41", 41.0), ("-3", -3.0), ("+2.5", 2.5), (".5", 0.5), ("7.", 7.0), ("1.5e-3", 0.0015)],
)
def test_the_forms_of_a_value(number: str, value: float) -> None:
    stats = parse_stats(f"# Measure A, Name, some text, {number}, unitless\n")
    assert stats.measures == {"Name": (value, "unitless")}


@pytest.mark.parametrize(
    "line",
    [
        "# Measure Cortex, NumVert, Number of Vertices, 140304",  # no unit: the value is last
        "# Measure Cortex, NumVert, Number of Vertices, many, unitless",
        "# Measure NumVert, 140304, unitless",  # name and value would be one field
        "# Measure Cortex, NumVert, Number of Vertices, 140304, ",
        "# Measure Cortex, NumVert, Number of Vertices, 140304, per unit",
        "# Measure Cortex, , Number of Vertices, 140304, unitless",
        "# Measure Cortex, NumVert, Number of Vertices, nan, unitless",
        "# Measure Cortex, NumVert, Number of Vertices, inf, unitless",
        "# Measure Cortex, NumVert, Number of Vertices, 1e999, unitless",
        "# Measure Cortex, NumVert, Number of Vertices, 140_304, unitless",
        "# Measure Cortex, NumVert, Number of Vertices, 0x10, unitless",
        "# Measure Cortex, NumVert, Number of Vertices, ١٤٠, unitless",  # Arabic-Indic digits
        "# Measure Cortex, NumVert, Number of Vertices, 140 304, unitless",
        "# Measure",
        "# Measure ",
        "# Measurement Cortex, NumVert, Number of Vertices, 140304, unitless",
        "# NoMeasure Cortex, NumVert, Number of Vertices, 140304, unitless",
        "# TableCol  2 Measure Cortex, NumVert, Number of Vertices, 140304, unitless",
    ],
)
def test_a_line_that_is_not_a_measure_is_skipped(line: str) -> None:
    good = "# Measure Cortex, MeanThickness, Mean Thickness, 2.39047, mm"
    stats = parse_stats(f"{line}\n{good}\n")
    assert stats.measures == {"MeanThickness": (2.39047, "mm")}
    assert (stats.columns, stats.rows) == ((), ())


@pytest.mark.parametrize(
    "line",
    [
        "#Measure Cortex, NumVert, Number of Vertices, 140304, unitless",
        "#  Measure\tCortex,NumVert,Number of Vertices,140304,unitless",
        "   # Measure Cortex ,  NumVert , Number of Vertices ,  140304  ,  unitless   ",
        "## Measure Cortex, NumVert, Number of Vertices, 140304, unitless",
    ],
)
def test_white_space_around_the_parts_of_a_measure_line_does_not_matter(line: str) -> None:
    assert parse_stats(line + "\n").measures == {"NumVert": (140304.0, "unitless")}


def test_of_two_measures_with_one_name_the_first_is_kept() -> None:
    text = (
        "# Measure Cortex, CortexVol, Total cortical gray matter volume, 1.0, mm^3\n"
        "# Measure Cortex, CortexVol Total cortical gray matter volume, 2.0, mm^3\n"
    )
    assert parse_stats(text).measures == {"CortexVol": (1.0, "mm^3")}


# ---------------------------------------------------------------------------------------
# the table


def test_columns_and_rows_of_a_parcel_table() -> None:
    for header in (FS6_APARC_HEADER, FS7_APARC_HEADER):
        stats = parse_stats(header + APARC_ROWS)
        assert stats.columns == APARC_COLUMNS
        assert len(stats.rows) == 4
        assert stats.rows[0] == (
            "bankssts",
            "1338",
            "953",
            "2385",
            "2.610",
            "0.429",
            "0.096",
            "0.018",
            "7",
            "1.0",
        )
        assert [row[0] for row in stats.rows] == [
            "bankssts",
            "caudalanteriorcingulate",
            "caudalmiddlefrontal",
            "cuneus",
        ]


def test_columns_and_rows_of_a_segmentation_table() -> None:
    # the ColHeaders line of aseg.stats has two spaces after the keyword and at its end, and
    # the rows start and end with white space
    stats = parse_stats(ASEG)
    assert stats.columns == (
        "Index",
        "SegId",
        "NVoxels",
        "Volume_mm3",
        "StructName",
        "normMean",
        "normStdDev",
        "normMin",
        "normMax",
        "normRange",
    )
    assert stats.rows == (
        (
            "1",
            "4",
            "8830",
            "9198.1",
            "Left-Lateral-Ventricle",
            "20.8310",
            "13.9629",
            "0.0000",
            "83.0000",
            "83.0000",
        ),
        (
            "2",
            "5",
            "490",
            "515.3",
            "Left-Inf-Lat-Vent",
            "34.1041",
            "15.8871",
            "0.0000",
            "74.0000",
            "74.0000",
        ),
    )
    assert len(stats.measures) == 7
    assert stats.measures["BrainSegVol-to-eTIV"] == (0.762144, "unitless")


def test_the_table_column_lines_are_not_the_header() -> None:
    # "# TableCol  2 ColHeader NumVert" holds the word; only "# ColHeaders" names the columns
    text = "# TableCol  1 ColHeader StructName\n# TableCol  2 ColHeader NumVert\n"
    assert parse_stats(text).columns == ()
    assert parse_stats(text + "# ColHeaders A B\nx 1\n").columns == ("A", "B")


def test_an_empty_text_and_a_header_alone() -> None:
    assert parse_stats("") == StatsFile(measures={}, columns=(), rows=())
    assert parse_stats("\n\n  \n") == StatsFile(measures={}, columns=(), rows=())
    stats = parse_stats(FS6_APARC_HEADER)
    assert (len(stats.measures), stats.columns, stats.rows) == (10, APARC_COLUMNS, ())


def test_blank_lines_between_rows_are_skipped() -> None:
    rows = APARC_ROWS.splitlines()
    spaced = MINIMAL_HEADER + "\n" + rows[0] + "\n\n   \n" + rows[1] + "\n\n"
    assert parse_stats(spaced) == parse_stats(MINIMAL_HEADER + rows[0] + "\n" + rows[1] + "\n")


@pytest.mark.parametrize("n_fields", [9, 11])
def test_a_row_with_another_field_count_is_refused_with_its_line_number(n_fields: int) -> None:
    rows = APARC_ROWS.splitlines()
    fields = rows[2].split()
    bad = " ".join(fields[:9] if n_fields == 9 else [*fields, "ds999999"])
    text = FS6_APARC_HEADER + "\n".join([rows[0], rows[1], bad, rows[3]]) + "\n"
    line = FS6_APARC_HEADER.count("\n") + 3
    with pytest.raises(ValueError, match=rf"^line {line}: {n_fields} fields") as caught:
        parse_stats(text)
    # the message names the place and never quotes the file
    assert "10 in the ColHeaders line" in str(caught.value)
    assert "caudalmiddlefrontal" not in str(caught.value)
    assert "ds999999" not in str(caught.value)


def test_a_structure_name_with_a_space_is_a_wrong_field_count() -> None:
    text = MINIMAL_HEADER + parcel("lateral occipital", 7544, 4719, 9900, "1.976", "0.405")
    with pytest.raises(ValueError, match=r"^line 2: 11 fields"):
        parse_stats(text)


def test_a_row_without_a_header_is_refused() -> None:
    # a file that is not a stats file must not come back as rows of its text
    with pytest.raises(ValueError, match=r"^line 3: a data row without a ColHeaders line"):
        parse_stats("# Title\n\n/data/ds999999-fmriprep/sourcedata/freesurfer/sub-01\n")
    rows = APARC_ROWS.splitlines()
    with pytest.raises(ValueError, match=r"^line 1: a data row without a ColHeaders line"):
        parse_stats(rows[0] + "\n" + MINIMAL_HEADER + rows[1] + "\n")


def test_a_second_header_is_refused() -> None:
    text = MINIMAL_HEADER + APARC_ROWS + "# ColHeaders StructName NumVert\nbankssts 1\n"
    with pytest.raises(ValueError, match=r"^line 6: a second ColHeaders line$"):
        parse_stats(text)


@pytest.mark.parametrize("newline", ["\r\n", "\r"])
def test_other_line_ends_give_the_same_result_and_the_same_line_numbers(newline: str) -> None:
    text = FS6_APARC_HEADER + APARC_ROWS
    assert parse_stats(text.replace("\n", newline)) == parse_stats(text)
    bad = (MINIMAL_HEADER + APARC_ROWS + "cuneus 1 2\n").replace("\n", newline)
    with pytest.raises(ValueError, match=r"^line 6: 3 fields"):
        parse_stats(bad)


def test_a_byte_order_mark_is_dropped() -> None:
    text = FS7_APARC_HEADER + APARC_ROWS
    assert parse_stats("﻿" + text) == parse_stats(text)


# ---------------------------------------------------------------------------------------
# nothing private is kept


@pytest.mark.parametrize(
    "text",
    [FS6_APARC_HEADER + APARC_ROWS, FS7_APARC_HEADER + APARC_ROWS, ASEG],
    ids=["aparc-fs6", "aparc-fs7", "aseg"],
)
def test_no_private_header_value_is_in_the_returned_object(text: str) -> None:
    for word in PRIVATE:  # the test texts do carry them
        assert word in text
    stats = parse_stats(text)
    kept = repr(stats)
    for word in PRIVATE:
        assert word not in kept
    # all that is kept: measure names with a number and a unit, column names, row fields
    assert set(stats.measures) <= {
        line.split(",")[1].split()[0] for line in text.splitlines() if line.startswith("# Measure")
    }
    for other in ("mris_anatomical_stats", "mri_segstats", "aparc.annot", "ASegStatsLUT", "2022"):
        assert other not in kept


HEADER_KEYS = ("hostname", "user", "SUBJECTS_DIR", "subjectname", "cmdline", "sysname", "machine")
ONE_LINE = st.text(
    alphabet=st.characters(blacklist_categories=["Cs"], blacklist_characters="\n\r"), max_size=60
)


@given(key=st.sampled_from(HEADER_KEYS), value=ONE_LINE, where=st.integers(0, 40))
def test_whatever_a_header_line_holds_none_of_it_is_kept(key: str, value: str, where: int) -> None:
    lines = (FS6_APARC_HEADER + APARC_ROWS).splitlines()
    lines.insert(where, f"# {key} SECRET{value}")
    stats = parse_stats("\n".join(lines) + "\n")
    assert "SECRET" not in repr(stats)
    assert len(stats.measures) == 10
    assert len(stats.rows) == 4


LINES = st.one_of(
    ONE_LINE,
    st.sampled_from(
        [
            MINIMAL_HEADER.strip(),
            "# ColHeaders A B",
            "# Measure Cortex, NumVert, Number of Vertices, 140304, unitless",
            "# Measure Cortex, CortexVol Total cortical gray matter volume, 484608.4, mm^3",
            "# hostname canary-host",
            "x 1",
            APARC_ROWS.splitlines()[0],
            "",
        ]
    ),
    st.builds("# Measure {}".format, ONE_LINE),
    st.builds("# ColHeaders {}".format, ONE_LINE),
)


@given(st.lists(LINES, max_size=12).map("\n".join))
def test_any_text_parses_or_is_refused_by_line_number(text: str) -> None:
    try:
        stats = parse_stats(text)
    except ValueError as exc:
        assert re.fullmatch(r"line \d+: [A-Za-z0-9 ,]+", str(exc))
        return
    assert all(len(row) == len(stats.columns) for row in stats.rows)
    for value, unit in stats.measures.values():
        assert math.isfinite(value)
        assert unit and not any(character.isspace() for character in unit)


# ---------------------------------------------------------------------------------------
# read_stats


def test_read_stats_is_parse_stats_of_the_file(tmp_path: Path) -> None:
    path = tmp_path / "lh.aparc.stats"
    path.write_text(FS6_APARC_HEADER + APARC_ROWS)
    assert read_stats(path) == parse_stats(FS6_APARC_HEADER + APARC_ROWS)
    assert read_stats(str(path)) == read_stats(path)


def test_read_stats_replaces_bytes_that_are_not_utf8(tmp_path: Path) -> None:
    text = FS6_APARC_HEADER + APARC_ROWS
    data = text.encode().replace(b"# user     canary-user", b"# user     Andr\xe9 \xff\xfe")
    assert data != text.encode()
    path = tmp_path / "lh.aparc.stats"
    path.write_bytes(data)
    assert read_stats(path) == parse_stats(text)
    # in a data row the byte is replaced and the row is kept
    path.write_bytes((MINIMAL_HEADER + APARC_ROWS).encode().replace(b"cuneus", b"cune\xfcs"))
    assert read_stats(path).rows[3][0] == "cune�s"


def test_read_stats_keeps_line_ends_and_drops_a_byte_order_mark(tmp_path: Path) -> None:
    text = FS7_APARC_HEADER + APARC_ROWS
    path = tmp_path / "lh.aparc.stats"
    path.write_bytes(b"\xef\xbb\xbf" + text.replace("\n", "\r\n").encode())
    assert read_stats(path) == parse_stats(text)


def test_read_stats_of_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_stats(tmp_path / "stats" / "lh.aparc.stats")


# ---------------------------------------------------------------------------------------
# aparc_frame


def test_the_parcel_frame_has_the_six_columns_with_their_types() -> None:
    frame = aparc_frame(parse_stats(FS6_APARC_HEADER + APARC_ROWS))
    assert list(frame.columns) == [
        "StructName",
        "NumVert",
        "SurfArea",
        "GrayVol",
        "ThickAvg",
        "ThickStd",
    ]
    assert frame["StructName"].dtype == object
    assert all(type(name) is str for name in frame["StructName"])
    assert [str(frame[name].dtype) for name in ("NumVert", "SurfArea", "GrayVol")] == ["int64"] * 3
    assert [str(frame[name].dtype) for name in ("ThickAvg", "ThickStd")] == ["float64"] * 2
    assert isinstance(frame.index, pd.RangeIndex)
    expected = pd.DataFrame(
        {
            "StructName": pd.Series(
                ["bankssts", "caudalanteriorcingulate", "caudalmiddlefrontal", "cuneus"],
                dtype=object,
            ),
            "NumVert": [1338, 1094, 4717, 2071],
            "SurfArea": [953, 712, 3013, 1303],
            "GrayVol": [2385, 2049, 7436, 2497],
            "ThickAvg": [2.610, 2.481, 2.358, 1.820],
            "ThickStd": [0.429, 0.724, 0.466, 0.375],
        }
    )
    pd.testing.assert_frame_equal(frame, expected)


def test_the_parcel_frame_keeps_the_order_of_the_file() -> None:
    rows = APARC_ROWS.splitlines(keepends=True)
    frame = aparc_frame(parse_stats(MINIMAL_HEADER + "".join(reversed(rows))))
    assert list(frame["StructName"]) == [
        "cuneus",
        "caudalmiddlefrontal",
        "caudalanteriorcingulate",
        "bankssts",
    ]
    assert list(frame["NumVert"]) == [2071, 4717, 1094, 1338]


def test_the_parcel_frame_finds_columns_by_name_not_by_position() -> None:
    text = (
        "# ColHeaders ThickStd GrayVol StructName ThickAvg Extra SurfArea NumVert\n"
        "0.429 2385 bankssts 2.610 x 953 1338\n"
    )
    frame = aparc_frame(parse_stats(text))
    assert frame.iloc[0].to_dict() == {
        "StructName": "bankssts",
        "NumVert": 1338,
        "SurfArea": 953,
        "GrayVol": 2385,
        "ThickAvg": 2.610,
        "ThickStd": 0.429,
    }


def test_a_header_without_rows_gives_an_empty_frame_of_the_same_types() -> None:
    frame = aparc_frame(parse_stats(FS7_APARC_HEADER))
    assert len(frame) == 0
    assert [str(dtype) for dtype in frame.dtypes] == [
        "object",
        "int64",
        "int64",
        "int64",
        "float64",
        "float64",
    ]


@pytest.mark.parametrize("column", ["StructName", "NumVert", "SurfArea", "GrayVol", "ThickAvg"])
def test_a_missing_column_is_named(column: str) -> None:
    columns = [name for name in APARC_COLUMNS if name != column]
    stats = parse_stats("# ColHeaders " + " ".join(columns) + "\n")
    with pytest.raises(ValueError, match=rf"no column {column}$"):
        aparc_frame(stats)


def test_a_segmentation_table_and_an_empty_file_are_not_parcel_tables() -> None:
    with pytest.raises(
        ValueError, match="no column NumVert, SurfArea, GrayVol, ThickAvg, ThickStd"
    ):
        aparc_frame(parse_stats(ASEG))
    with pytest.raises(ValueError, match="no column StructName, NumVert"):
        aparc_frame(parse_stats(""))


def test_a_column_named_twice_is_refused() -> None:
    stats = parse_stats(MINIMAL_HEADER.strip() + " ThickAvg\n")
    with pytest.raises(ValueError, match="more than one column ThickAvg"):
        aparc_frame(stats)


@pytest.mark.parametrize(
    ("column", "text", "kind"),
    [
        ("NumVert", "12.5", "an integer"),
        ("NumVert", "many", "an integer"),
        ("SurfArea", "nan", "an integer"),
        ("GrayVol", "1_000", "an integer"),
        ("GrayVol", "9223372036854775808", "an integer"),  # 2**63
        ("NumVert", "1e30", "an integer"),
        ("ThickAvg", "thick", "a number"),
        ("ThickStd", "0,429", "a number"),
        ("ThickAvg", "2.6.1", "a number"),
    ],
)
def test_a_value_of_the_wrong_kind_is_refused_with_row_and_column(
    column: str, text: str, kind: str
) -> None:
    values: dict[str, object] = {
        "NumVert": 1094,
        "SurfArea": 712,
        "GrayVol": 2049,
        "ThickAvg": "2.481",
        "ThickStd": "0.724",
    }
    values[column] = text
    rows = APARC_ROWS.splitlines(keepends=True)[0] + parcel(
        "caudalanteriorcingulate", *values.values()
    )
    with pytest.raises(ValueError, match=rf"^data row 2, column {column}: not {kind}$") as caught:
        aparc_frame(parse_stats(MINIMAL_HEADER + rows))
    assert text not in str(caught.value)


def test_a_count_written_as_a_decimal_with_an_integral_value_is_read() -> None:
    text = MINIMAL_HEADER + parcel("bankssts", "1338.0", "9.53e2", "+2385", "2.610", "0.429")
    frame = aparc_frame(parse_stats(text))
    assert frame.iloc[0].to_dict() == {
        "StructName": "bankssts",
        "NumVert": 1338,
        "SurfArea": 953,
        "GrayVol": 2385,
        "ThickAvg": 2.610,
        "ThickStd": 0.429,
    }
    assert str(frame["NumVert"].dtype) == "int64"


@pytest.mark.parametrize("text", ["nan", "-nan", "NaN", "inf", "-inf"])
def test_a_thickness_that_is_not_finite_is_kept_as_it_is_written(text: str) -> None:
    rows = APARC_ROWS + parcel("frontalpole", 0, 0, 0, text, text)
    frame = aparc_frame(parse_stats(MINIMAL_HEADER + rows))
    assert len(frame) == 5
    value = frame["ThickAvg"].iloc[4]
    assert math.isnan(value) if "nan" in text.lower() else value == float(text)
    assert np.isfinite(frame["ThickAvg"].iloc[:4]).all()


def test_a_structure_named_twice_is_refused() -> None:
    rows = APARC_ROWS + APARC_ROWS.splitlines(keepends=True)[1]
    with pytest.raises(ValueError, match=r"^data rows 2 and 5 have one structure name$"):
        aparc_frame(parse_stats(MINIMAL_HEADER + rows))


# ---------------------------------------------------------------------------------------
# holes and the Euler number


def test_surface_holes_of_a_segmentation_file() -> None:
    assert surface_holes(parse_stats(ASEG)) == (41, 26)
    holes = surface_holes(parse_stats(ASEG))
    assert holes is not None and all(type(count) is int for count in holes)


def test_left_and_right_are_not_exchanged_and_the_total_is_not_used() -> None:
    text = (
        "# Measure SurfaceHoles, SurfaceHoles, Total number of defect holes, 9, unitless\n"
        "# Measure rhSurfaceHoles, rhSurfaceHoles, Number of defect holes in rh, 7, unitless\n"
        "# Measure lhSurfaceHoles, lhSurfaceHoles, Number of defect holes in lh, 2, unitless\n"
    )
    assert surface_holes(parse_stats(text)) == (2, 7)


def test_no_holes_where_the_file_has_none() -> None:
    assert surface_holes(parse_stats(FS6_APARC_HEADER + APARC_ROWS)) is None
    assert surface_holes(parse_stats("")) is None
    # a segmentation file made without --euler: the other measures, and no hole counts
    without = "".join(line for line in ASEG.splitlines(keepends=True) if "SurfaceHoles" not in line)
    assert len(parse_stats(without).measures) == 4
    assert surface_holes(parse_stats(without)) is None


@pytest.mark.parametrize("missing", ["lhSurfaceHoles", "rhSurfaceHoles"])
def test_no_holes_when_one_hemisphere_has_no_count(missing: str) -> None:
    text = "".join(line for line in ASEG.splitlines(keepends=True) if f" {missing}," not in line)
    assert len(parse_stats(text).measures) == 6
    assert surface_holes(parse_stats(text)) is None


@pytest.mark.parametrize("value", ["41.5", "-1", "1e-3"])
def test_no_holes_when_a_count_is_not_a_count(value: str) -> None:
    text = ASEG.replace("prior to fixing, 26, unitless", f"prior to fixing, {value}, unitless")
    assert parse_stats(text).measures["rhSurfaceHoles"] == (float(value), "unitless")
    assert surface_holes(parse_stats(text)) is None


def test_zero_holes_are_holes() -> None:
    text = ASEG.replace("fixing, 41, unitless", "fixing, 0, unitless")
    assert surface_holes(parse_stats(text)) == (0, 26)


def test_surface_holes_of_a_file_built_by_hand() -> None:
    stats = StatsFile(
        measures={
            "lhSurfaceHoles": (3.0, "unitless"),
            "rhSurfaceHoles": (float("nan"), "unitless"),
        },
        columns=(),
        rows=(),
    )
    assert surface_holes(stats) is None


@pytest.mark.parametrize(("holes", "euler"), [(0, 2), (1, 0), (41, -80), (26, -50)])
def test_the_euler_number(holes: int, euler: int) -> None:
    assert euler_number(holes) == euler
    # what mri_segstats computes the count from
    assert 1 - euler // 2 == holes


def test_no_euler_number_of_a_negative_count() -> None:
    with pytest.raises(ValueError, match="-1 holes"):
        euler_number(-1)


# ---------------------------------------------------------------------------------------
# the files of the synthetic subjects


@pytest.fixture(scope="module", params=["6.0.1", "7.3.2"])
def tier_a_stats(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """``stats/`` of a small synthetic subject as ``add_tier_a`` writes it without Tier B:
    34 parcels per hemisphere, holes (3, 5)."""
    subject = write_synthetic_subject(
        tmp_path_factory.mktemp("stats"), shape=(32, 32, 32), subdivisions=1
    )
    fs7 = request.param.startswith("7")
    add_tier_a(
        subject, fs_version=request.param, log_style="fs7-fmriprep" if fs7 else "fs6-fmriprep"
    )
    return subject.path / "stats"


def test_the_stats_files_of_a_synthetic_subject(tier_a_stats: Path) -> None:
    fs6 = "$Id" in (tier_a_stats / "aseg.stats").read_text()
    for hemi in ("lh", "rh"):
        path = tier_a_stats / f"{hemi}.aparc.stats"
        for canary in CANARIES:
            assert canary in path.read_text()
        stats = read_stats(path)
        for canary in CANARIES:
            assert canary not in repr(stats)
        assert ("CortexVol" in stats.measures) == fs6
        assert len(stats.measures) == (6 if fs6 else 3)
        frame = aparc_frame(stats)
        assert len(frame) == 34
        assert frame["StructName"].is_unique
        assert "unknown" not in set(frame["StructName"])
        assert stats.measures["NumVert"] == (float(frame["NumVert"].sum()), "unitless")
        assert (frame["ThickAvg"] == 2.5).all()
    aseg = read_stats(tier_a_stats / "aseg.stats")
    for canary in CANARIES:
        assert canary in (tier_a_stats / "aseg.stats").read_text()
        assert canary not in repr(aseg)
    assert surface_holes(aseg) == (3, 5)
    assert [row[4] for row in aseg.rows] == ["Left-Lateral-Ventricle", "Right-Lateral-Ventricle"]


# ---------------------------------------------------------------------------------------
# the real files, where the network tests have fetched them


def _real_stats(cached_subjects: Path, subject: str) -> Path:
    stats = cached_subjects / subject / "stats"
    names = ("lh.aparc.stats", "rh.aparc.stats", "aseg.stats")
    if not all((stats / name).is_file() for name in names):
        pytest.skip("the stats files are not in the pytest cache (the network tests fetch them)")
    return stats


def _header_values(text: str) -> list[str]:
    """What the header of a real file holds after the keys that name a machine or a dataset."""
    values = []
    for line in text.splitlines():
        key, _, value = line.removeprefix("# ").partition(" ")
        if line.startswith("# ") and key in ("hostname", "user", "SUBJECTS_DIR", "subjectname"):
            values.append(value.strip())
    assert len(values) == 4
    return values


def test_the_stats_files_of_the_real_freesurfer_6_subject(cached_subjects: Path) -> None:
    stats_dir = _real_stats(cached_subjects, "ds000102-sub-01")
    lh = read_stats(stats_dir / "lh.aparc.stats")
    assert lh.measures["MeanThickness"] == (2.39047, "mm")
    assert lh.measures["NumVert"] == (140304.0, "unitless")
    assert lh.measures["CortexVol"] == (484608.441133, "mm^3")
    assert len(lh.measures) == 10
    for hemi in ("lh", "rh"):
        stats = read_stats(stats_dir / f"{hemi}.aparc.stats")
        frame = aparc_frame(stats)
        assert len(frame) == 34
        assert stats.columns == APARC_COLUMNS
        # as counted in the two files; the NumVert measures are 140,304 and 138,761
        assert int(frame["NumVert"].sum()) == {"lh": 140305, "rh": 138769}[hemi]
    aseg = read_stats(stats_dir / "aseg.stats")
    assert surface_holes(aseg) == (41, 26)
    assert aseg.measures["eTIV"] == (1519867.314105, "mm^3")
    assert len(aseg.rows) == 45
    assert parse_stats(FS6_APARC_HEADER).measures == lh.measures


def test_the_stats_files_of_the_real_freesurfer_7_subject(cached_subjects: Path) -> None:
    stats_dir = _real_stats(cached_subjects, "ds003020-sub-UTS01")
    lh = read_stats(stats_dir / "lh.aparc.stats")
    assert lh.measures == parse_stats(FS7_APARC_HEADER).measures
    for hemi in ("lh", "rh"):
        assert len(aparc_frame(read_stats(stats_dir / f"{hemi}.aparc.stats"))) == 34
    aseg = read_stats(stats_dir / "aseg.stats")
    assert surface_holes(aseg) == (4, 15)
    assert len(aseg.rows) == 45


@pytest.mark.parametrize("subject", ["ds000102-sub-01", "ds003020-sub-UTS01"])
def test_nothing_of_a_real_header_is_kept(cached_subjects: Path, subject: str) -> None:
    stats_dir = _real_stats(cached_subjects, subject)
    for name in ("lh.aparc.stats", "rh.aparc.stats", "aseg.stats"):
        path = stats_dir / name
        kept = repr(read_stats(path))
        for value in _header_values(path.read_text()):
            assert value not in kept
        assert not re.search(r"ds\d{6}", kept)
        assert "/" not in kept
