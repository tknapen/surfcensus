"""FreeSurfer stats files: ``stats/?h.aparc.stats`` and ``stats/aseg.stats``.

Only the ``# Measure`` lines, the ``# ColHeaders`` line and the data rows are read. The
header also carries ``SUBJECTS_DIR``, ``hostname``, ``user``, ``cmdline`` and
``subjectname``; none of them is parsed or kept, so nothing returned here names a dataset.
The text of an error names a line or a row by its number and never quotes the file.

What the rules rest on:

* the six stats files of two subjects of the OpenNeuroDerivatives bucket, read on
  2026-11-13: ``ds000102`` sub-01 (FreeSurfer 6.0.1) and ``ds003020`` sub-UTS01 (7.3.2).
  Each ``?h.aparc.stats`` has 34 data rows of 10 fields; the 6.0.1 files have 10 measure
  lines, one of them malformed (below), the 7.3.2 files 3. Both ``aseg.stats`` files have
  ``lhSurfaceHoles`` and ``rhSurfaceHoles``;
* ``mris_anatomical_stats`` and ``mri_segstats`` of FreeSurfer v6.0.0 and v7.4.1, and
  ``mri_segstats`` of v5.3.0, read on GitHub the same day. A parcel row is printed as
  ``%-40s%5d  %5.0f  %5.0f  %5.3f %5.3f ...``, so vertex count, area and volume are
  integers in the file. A parcel without vertices gets no row, and neither do ``unknown``
  and ``corpuscallosum``, so a file can hold fewer than 34 rows. In all three versions
  ``mri_segstats`` writes the hole counts only when it is called with ``--euler``, as
  ``1 - euler / 2``.

No stats file of another FreeSurfer version was looked at.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "StatsFile",
    "aparc_frame",
    "euler_number",
    "parse_stats",
    "read_stats",
    "surface_holes",
]

_BOM = "﻿"
# ASCII digits only: \d alone also matches the decimal digits of other scripts
_NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?", re.ASCII)
_INTEGER = re.compile(r"[+-]?\d+", re.ASCII)
# what printf writes for a value that is not a number
_NOT_FINITE = re.compile(r"[+-]?(?:nan|inf|infinity)", re.IGNORECASE)

_APARC_INT = ("NumVert", "SurfArea", "GrayVol")
_APARC_FLOAT = ("ThickAvg", "ThickStd")
_INT64_MAX = 2**63 - 1


@dataclass(frozen=True)
class StatsFile:
    measures: dict[str, tuple[float, str]]
    """Measure name -> (value, unit): ``"MeanThickness" -> (2.39047, "mm")``,
    ``"lhSurfaceHoles" -> (41.0, "unitless")``."""
    columns: tuple[str, ...]
    """From the ``# ColHeaders`` line."""
    rows: tuple[tuple[str, ...], ...]
    """The data rows, each split on white space, as strings."""


def _measure(rest: str) -> tuple[str, float, str] | None:
    """``(name, value, unit)`` of the text after ``# Measure``, or None for a line that
    does not have the form."""
    fields = [field.strip() for field in rest.split(",")]
    if len(fields) < 4:  # name and value must be different fields
        return None
    name, number, unit = fields[1].split(), fields[-2], fields[-1]
    if not name or len(unit.split()) != 1 or not _NUMBER.fullmatch(number):
        return None
    value = float(number)
    if not math.isfinite(value):  # 1e999
        return None
    return name[0], value, unit


def parse_stats(text: str) -> StatsFile:
    """Parse the text of a stats file.

    A measure line is ``# Measure <a>, <b>..., <value>, <unit>``, split on commas: the unit
    is the last field, the value the one before it, and the name the first
    whitespace-separated token of the second field. FreeSurfer 6 writes one line without the
    comma after the name (``# Measure Cortex, CortexVol Total cortical gray matter volume,
    484608.441133, mm^3``), which this rule reads as ``CortexVol``. A line that does not end
    in a number and a unit is skipped: fewer than four fields, a value that is not a finite
    decimal number, or a unit that is not one word. Of two lines with one name the first is
    kept.

    Every other comment line is ignored, and so are blank lines. A line that is not a
    comment is a data row, split on white space.

    Raises ValueError, with the line number, for a data row whose field count differs from
    the header's, for a data row above the ``# ColHeaders`` line or in a file without one,
    and for a second ``# ColHeaders`` line.
    """
    measures: dict[str, tuple[float, str]] = {}
    columns: tuple[str, ...] | None = None
    rows: list[tuple[str, ...]] = []
    lines = text.removeprefix(_BOM).replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for number, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            keyword, rest = (*line.lstrip("#").split(None, 1), "", "")[:2]
            if keyword == "Measure":
                measure = _measure(rest)
                if measure is not None:
                    name, value, unit = measure
                    measures.setdefault(name, (value, unit))
            elif keyword == "ColHeaders":
                if columns is not None:
                    raise ValueError(f"line {number}: a second ColHeaders line")
                columns = tuple(rest.split())
            continue
        fields = tuple(line.split())
        if columns is None:
            raise ValueError(f"line {number}: a data row without a ColHeaders line above it")
        if len(fields) != len(columns):
            raise ValueError(
                f"line {number}: {len(fields)} fields in a data row, "
                f"{len(columns)} in the ColHeaders line"
            )
        rows.append(fields)
    return StatsFile(measures=measures, columns=columns or (), rows=tuple(rows))


def read_stats(path: Path | str) -> StatsFile:
    """:func:`parse_stats` of a file, read as UTF-8 with undecodable bytes replaced."""
    return parse_stats(Path(path).read_bytes().decode("utf-8", errors="replace"))


def _integer(text: str) -> int | None:
    """An integer, also when written as a decimal with an integral value (``953.0``)."""
    if _INTEGER.fullmatch(text):
        value = int(text)
    elif _NUMBER.fullmatch(text) and float(text).is_integer():
        value = int(float(text))
    else:
        return None
    return value if abs(value) <= _INT64_MAX else None


def _decimal(text: str) -> float | None:
    if _NUMBER.fullmatch(text):
        return float(text)
    if _NOT_FINITE.fullmatch(text):
        return float(text)
    return None


def aparc_frame(stats: StatsFile) -> pd.DataFrame:
    """The parcel table of an ``aparc.stats`` file.

    Columns ``StructName`` (str), ``NumVert``, ``SurfArea``, ``GrayVol`` (int64),
    ``ThickAvg``, ``ThickStd`` (float64), one row per parcel in file order. A thickness
    written as ``nan`` or ``inf`` is kept as that.

    Raises ValueError when one of these columns is missing or named twice, when a value is
    not a number of its column's kind (the message gives the data row, counted from 1, and
    the column), and when a structure name appears in two rows.
    """
    needed = ("StructName", *_APARC_INT, *_APARC_FLOAT)
    missing = [name for name in needed if name not in stats.columns]
    if missing:
        raise ValueError(f"not a parcel table: no column {', '.join(missing)}")
    twice = [name for name in needed if stats.columns.count(name) > 1]
    if twice:
        raise ValueError(f"not a parcel table: more than one column {', '.join(twice)}")
    position = {name: stats.columns.index(name) for name in needed}

    names: list[str] = []
    seen: dict[str, int] = {}
    integers: dict[str, list[int]] = {name: [] for name in _APARC_INT}
    decimals: dict[str, list[float]] = {name: [] for name in _APARC_FLOAT}
    for number, row in enumerate(stats.rows, start=1):
        name = row[position["StructName"]]
        if name in seen:
            raise ValueError(f"data rows {seen[name]} and {number} have one structure name")
        seen[name] = number
        names.append(name)
        for column, values in integers.items():
            integer = _integer(row[position[column]])
            if integer is None:
                raise ValueError(f"data row {number}, column {column}: not an integer")
            values.append(integer)
        for column, values in decimals.items():
            decimal = _decimal(row[position[column]])
            if decimal is None:
                raise ValueError(f"data row {number}, column {column}: not a number")
            values.append(decimal)
    return pd.DataFrame(
        {
            "StructName": pd.Series(names, dtype=object),
            **{name: np.array(values, dtype=np.int64) for name, values in integers.items()},
            **{name: np.array(values, dtype=np.float64) for name, values in decimals.items()},
        }
    )


def surface_holes(aseg: StatsFile) -> tuple[int, int] | None:
    """``(lhSurfaceHoles, rhSurfaceHoles)`` of an ``aseg.stats`` file: the holes of the two
    surfaces before topology fixing.

    None unless both measures are there and each is a whole number that is not negative.
    ``mri_segstats`` writes them only when it is called with ``--euler``; where they are
    absent the caller uses the counts of the recon-all log.
    """
    holes: list[int] = []
    for name in ("lhSurfaceHoles", "rhSurfaceHoles"):
        measure = aseg.measures.get(name)
        if measure is None:
            return None
        value = float(measure[0])
        if not value.is_integer() or value < 0:
            return None
        holes.append(int(value))
    return holes[0], holes[1]


def euler_number(holes: int) -> int:
    """``2 - 2 * holes``: the Euler number of the surface before topology fixing.

    ``mri_segstats`` writes the hole count as ``1 - euler / 2``. ValueError for a negative
    count.
    """
    if holes < 0:
        raise ValueError(f"a surface cannot have {holes} holes")
    return 2 - 2 * holes
