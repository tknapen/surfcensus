"""ITK text transforms as RAS point maps. Work package E1.A.

A ``mode-image`` text transform of fMRIPrep maps a point of the reference (fixed) space to the
moving space: ``from-scanner_to-T1w`` maps a point of T1w to the run's scanner space,
``from-T1w_to-fsnative`` a point of fsnative to T1w. The file stores
``y = M (x - c) + t + c`` in LPS (``Parameters``: the 9 numbers of ``M`` row by row, then
``t``; ``FixedParameters``: the centre ``c``), which is turned into RAS. The convention is that
of nitransforms 24.1.1 (``io/itk.py``, ``ITKLinearTransform.to_ras``) as the review of stage 1
read it.
"""

from __future__ import annotations

from pathlib import Path

from surfcensus.records import Matrix4

LPS: Matrix4 = (
    (-1.0, 0.0, 0.0, 0.0),
    (0.0, -1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)
"""RAS to LPS and back."""

TRANSFORM_KINDS: tuple[str, ...] = ("AffineTransform", "MatrixOffsetTransformBase")
"""The ``Transform:`` types read, by prefix (``AffineTransform_double_3_3``, ...)."""


class ItkError(ValueError):
    """A text that is not one 3D affine ITK transform. The message names no path."""


def read_itk(text: str) -> Matrix4:
    """The RAS point map of the single affine transform in ``text``. Raises :class:`ItkError`
    for none, more than one, or one that is not affine."""
    raise NotImplementedError("work package E1.A: func/itk.py")


def read_itk_file(path: Path) -> Matrix4:
    """:func:`read_itk` of a file, read as UTF-8; the error names the file, not its path."""
    raise NotImplementedError("work package E1.A: func/itk.py")
