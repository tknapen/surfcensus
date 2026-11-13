"""FreeSurfer stats files: ``stats/?h.aparc.stats`` and ``stats/aseg.stats``.

Only the ``# Measure`` lines, the ``# ColHeaders`` line and the data rows are read. The
header also carries ``SUBJECTS_DIR``, ``hostname``, ``user``, ``cmdline`` and
``subjectname``; none of them is parsed or kept, so nothing returned here names a dataset.

Interface of milestone 3; the functions are written by package 3.A.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_OWNER = "package 3.A (stats and regional metrics)"


@dataclass(frozen=True)
class StatsFile:
    measures: dict[str, tuple[float, str]]
    """Measure name -> (value, unit): ``"MeanThickness" -> (2.39047, "mm")``,
    ``"lhSurfaceHoles" -> (41.0, "unitless")``."""
    columns: tuple[str, ...]
    """From the ``# ColHeaders`` line."""
    rows: tuple[tuple[str, ...], ...]
    """The data rows, each split on white space, as strings."""


def parse_stats(text: str) -> StatsFile:
    """Parse the text of a stats file.

    A measure line is ``# Measure <a>, <b>..., <value>, <unit>``, split on commas: the unit
    is the last field, the value the one before it, and the name the first
    whitespace-separated token of the second field. FreeSurfer 6 writes one line without the
    comma after the name (``# Measure Cortex, CortexVol Total cortical gray matter volume,
    484608.441133, mm^3``), which this rule reads as ``CortexVol``. A line that does not end
    in a number and a unit is skipped.

    Raises ValueError, with the line number, for a data row whose field count differs from
    the header's.
    """
    raise NotImplementedError(f"fsio.stats.parse_stats: {_OWNER}")


def read_stats(path: Path | str) -> StatsFile:
    """:func:`parse_stats` of a file, read as UTF-8 with undecodable bytes replaced."""
    raise NotImplementedError(f"fsio.stats.read_stats: {_OWNER}")


def aparc_frame(stats: StatsFile) -> pd.DataFrame:
    """The parcel table of an ``aparc.stats`` file.

    Columns ``StructName`` (str), ``NumVert``, ``SurfArea``, ``GrayVol`` (int64),
    ``ThickAvg``, ``ThickStd`` (float64), one row per parcel in file order. Raises
    ValueError when one of these columns is missing.
    """
    raise NotImplementedError(f"fsio.stats.aparc_frame: {_OWNER}")


def surface_holes(aseg: StatsFile) -> tuple[int, int] | None:
    """``(lhSurfaceHoles, rhSurfaceHoles)`` of an ``aseg.stats`` file; None when it has
    neither measure."""
    raise NotImplementedError(f"fsio.stats.surface_holes: {_OWNER}")


def euler_number(holes: int) -> int:
    """``2 - 2 * holes``: the Euler number of the surface before topology fixing."""
    raise NotImplementedError(f"fsio.stats.euler_number: {_OWNER}")
