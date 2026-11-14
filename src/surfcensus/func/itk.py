"""ITK text transforms as RAS point maps. Work package E1.A.

A ``mode-image`` text transform of fMRIPrep maps a point of the reference (fixed) space to the
moving space: ``from-scanner_to-T1w`` maps a point of T1w to the run's scanner space,
``from-T1w_to-fsnative`` a point of fsnative to T1w. The file stores
``y = M (x - c) + t + c`` in LPS (``Parameters``: the 9 numbers of ``M`` row by row, then
``t``; ``FixedParameters``: the centre ``c``), which is turned into RAS. The convention is that
of nitransforms 24.1.1 (``io/itk.py``, ``ITKLinearTransform.to_ras``) as the review of stage 1
read it.

Read again in the wheel of nitransforms 24.1.1 from PyPI (2026-11-14): ``io/itk.py`` line 16
(``LPS = np.diag([-1, -1, 1, 1])``), lines 84 to 91 (``to_ras`` returns
``LPS . C(+c) . M . C(-c) . LPS``, with ``M`` the 4x4 of ``Parameters`` and ``C`` a shift by
the centre), lines 187 to 210 (``from_string``: the third non-blank line after the banner holds
``Parameters``, the fourth ``FixedParameters``; a second transform in the text is an error);
``linear.py`` lines 36 to 40 (the matrix maps coordinates from the reference space into the
moving space). nitransforms reads both lines as float32; this reader keeps float64, so the two
agree exactly for numbers that float32 holds and within float32 rounding otherwise.

The arithmetic is in Python floats, without numpy, so that the matrix does not depend on the
linear-algebra library.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from surfcensus.records import Matrix4, to_matrix4

LPS: Matrix4 = (
    (-1.0, 0.0, 0.0, 0.0),
    (0.0, -1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)
"""RAS to LPS and back."""

TRANSFORM_KINDS: tuple[str, ...] = ("AffineTransform", "MatrixOffsetTransformBase")
"""The ``Transform:`` types read, by prefix (``AffineTransform_double_3_3``, ...)."""

BANNER = "#Insight Transform File"
"""The first line of an ITK text transform file (``#Insight Transform File V1.0``)."""
MAX_BYTES = 1024 * 1024
"""A larger file is not read: one affine transform takes a few hundred bytes."""

_NUMBER = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]{1,4})?")
_KIND = re.compile(r"([A-Za-z]+)_(?:double|float)_3_3")
_SIGNS = (-1.0, -1.0, 1.0)


class ItkError(ValueError):
    """A text that is not one 3D affine ITK transform. The message names no path."""


def _numbers(line: str, what: str, count: int) -> list[float]:
    tokens = line.split(":", 1)[1].split()
    if len(tokens) != count:
        raise ItkError(f"{what} holds {len(tokens)} numbers, not {count}")
    values = []
    for token in tokens:
        if len(token) > 64 or not _NUMBER.fullmatch(token):
            raise ItkError(f"{what} holds something that is not a number")
        value = float(token)
        if not math.isfinite(value):
            raise ItkError(f"{what} holds a number that is not finite")
        values.append(value)
    return values


def read_itk(text: str) -> Matrix4:
    """The RAS point map of the single affine transform in ``text``. Raises :class:`ItkError`
    for none, more than one, or one that is not affine.

    The text must start with the banner (:data:`BANNER`) and hold one ``Transform:`` line of
    a kind of :data:`TRANSFORM_KINDS` in three dimensions, followed by one ``Parameters:``
    line of 12 finite numbers and one ``FixedParameters:`` line of 3; comment lines and other
    lines are ignored. The point map is ``x -> F (M (F x - c) + t + c)`` with ``F`` the flip
    :data:`LPS`."""
    if not isinstance(text, str):
        raise ItkError("an ITK transform is text")
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    if not lines or not lines[0].startswith(BANNER):
        raise ItkError("not an ITK text transform: the first line is not its banner")
    kinds = [line for line in lines if line.startswith("Transform:")]
    if not kinds:
        raise ItkError("no transform")
    if len(kinds) > 1:
        raise ItkError(f"{len(kinds)} transforms, not one")
    kind = kinds[0].split(":", 1)[1].strip()
    matched = _KIND.fullmatch(kind)
    if matched is None or not matched.group(1).startswith(TRANSFORM_KINDS):
        raise ItkError("the transform is not a 3D affine one")
    params = [line for line in lines if line.startswith("Parameters:")]
    fixed = [line for line in lines if line.startswith("FixedParameters:")]
    if len(params) != 1 or len(fixed) != 1:
        raise ItkError("the transform needs one Parameters and one FixedParameters line")
    p = _numbers(params[0], "Parameters", 12)
    c = _numbers(fixed[0], "FixedParameters", 3)
    a = [p[0:3], p[3:6], p[6:9]]
    t = p[9:12]
    # LPS: y = A x + (t + c - A c); RAS: flip the first two axes on both sides
    offset = [t[i] + c[i] - (a[i][0] * c[0] + a[i][1] * c[1] + a[i][2] * c[2]) for i in range(3)]
    rows = []
    for i in range(3):
        row = [_SIGNS[i] * _SIGNS[j] * a[i][j] for j in range(3)]
        row.append(_SIGNS[i] * offset[i])
        rows.append([x + 0.0 for x in row])  # no negative zero
    rows.append([0.0, 0.0, 0.0, 1.0])
    if not all(math.isfinite(x) for row in rows for x in row):
        raise ItkError("the transform's numbers overflow")
    return to_matrix4(rows)


def read_itk_file(path: Path) -> Matrix4:
    """:func:`read_itk` of a file, read as UTF-8; the error names the file, not its path.

    ``ItkError`` for a file that is not one affine transform (also one that is not UTF-8 or
    larger than :data:`MAX_BYTES`); ``OSError`` for one that cannot be read, its message the
    file's name and the system's reason, without the path."""
    name = Path(path).name
    try:
        with open(path, "rb") as handle:
            data = handle.read(MAX_BYTES + 1)
    except OSError as error:
        raise OSError(error.errno, f"{name}: {error.strerror or 'cannot be read'}") from None
    if len(data) > MAX_BYTES:
        raise ItkError(f"{name}: too large for one ITK transform")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise ItkError(f"{name}: not UTF-8 text") from None
    try:
        return read_itk(text)
    except ItkError as error:
        raise ItkError(f"{name}: {error}") from None
