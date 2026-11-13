"""FreeSurfer annotations (``label/?h.aparc.annot``) and label files (``label/?h.cortex.label``).

Interface of milestone 3; the functions are written by package 3.B.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Bool, Int

_OWNER = "package 3.B (annotation and morph data)"

NON_CORTEX_NAMES: tuple[str, ...] = ("unknown", "corpuscallosum")
"""Parcels of the Desikan annotation that are not cortex."""


@dataclass(frozen=True)
class Annotation:
    labels: Int[np.ndarray, " n"]
    """int32 index into ``names`` per vertex; -1 for a vertex without a label."""
    names: tuple[str, ...]
    """Parcel names, decoded as UTF-8, in the order of the file's colour table."""

    def mask(self, *names: str) -> Bool[np.ndarray, " n"]:
        """Vertices labelled with one of ``names``. Raises KeyError naming every name that
        the file lacks."""
        raise NotImplementedError(f"fsio.annot.Annotation.mask: {_OWNER}")

    def cortex(self) -> Bool[np.ndarray, " n"]:
        """Vertices with a label that is not one of :data:`NON_CORTEX_NAMES`."""
        raise NotImplementedError(f"fsio.annot.Annotation.cortex: {_OWNER}")


def read_annotation(path: Path | str) -> Annotation:
    """Read an annotation with nibabel. Raises ValueError when it has no colour table."""
    raise NotImplementedError(f"fsio.annot.read_annotation: {_OWNER}")


def read_label(path: Path | str) -> Int[np.ndarray, " k"]:
    """The vertex indices of a ``.label`` file, int64, sorted ascending (the file's own
    order is arbitrary: ``fsaverage/label/lh.cortex.label`` is not sorted)."""
    raise NotImplementedError(f"fsio.annot.read_label: {_OWNER}")
