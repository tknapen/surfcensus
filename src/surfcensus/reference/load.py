"""A normative reference in memory and on disk: per-vertex median and MAD maps per stratum.

On disk a reference is a directory (``tables/reference/<reference_id>/`` of a run, and
``surfcensus/data/reference/`` for the one shipped with the package):

* ``index.json``: ``reference_id``, ``kind``, ``mesh``, ``template_id``, ``sphere_sha256``,
  ``config_hash``, ``metric_norms``, ``parcel_norms``, and ``strata``: name ->
  ``{"file": "stratum_000.npz", "n": ..., "members_sha256": ...}``;
* one ``stratum_<nnn>.npz`` per stratum with ``{lh,rh}_median`` and ``{lh,rh}_mad``
  (``<f4``) and ``{lh,rh}_n`` (``<i4``);
* ``members.parquet`` with the columns ``stratum`` and ``subject_uid``.

Stratum names and reference ids are made by ``reference.strata``.

Interface of milestone 3. The types are final; the functions are written by package 3.E.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from jaxtyping import Float, Int

from surfcensus.sources.base import Hemi

_OWNER = "package 3.E (reference files)"

INDEX_JSON = "index.json"
MEMBERS_PARQUET = "members.parquet"

ReferenceKind = Literal["shipped", "set", "rated"]


@dataclass(frozen=True)
class ReferenceMaps:
    """The reference of one stratum on one hemisphere."""

    median: Float[np.ndarray, " v"]
    """float32, of thickness minus the subject's global median; NaN where there is no
    reference."""
    mad: Float[np.ndarray, " v"]
    """float32, smoothed on the mesh and floored."""
    n: Int[np.ndarray, " v"]
    """int32, subjects with a value at the vertex."""


@dataclass(frozen=True)
class Reference:
    reference_id: str
    """``shipped-<version>``, ``set-<12 hex>``, later ``rated-<12 hex>``."""
    kind: ReferenceKind
    mesh: str
    template_id: str | None
    """None for a shipped reference."""
    sphere_sha256: Mapping[Hemi, str]
    """Of the template's source ``?h.sphere.reg``. A reference applies to a run only when
    these equal the run template's."""
    config_hash: str
    strata: Mapping[str, Mapping[Hemi, ReferenceMaps]]
    members: Mapping[str, tuple[str, ...]]
    """Stratum -> sorted ``subject_uid``s; empty for a shipped reference."""
    metric_norms: Mapping[str, Mapping[str, tuple[float, ...]]]
    """Stratum -> score component -> its 101 percentiles (0th to 100th) in the reference
    population; shipped and rated references only."""
    parcel_norms: Mapping[str, Mapping[str, tuple[float, float]]]
    """Stratum -> parcel -> (median, MAD) of ``ThickAvg`` minus the subject's mean
    thickness; shipped and rated references only."""


def shipped_reference() -> Reference | None:
    """The reference in ``surfcensus/data/reference/``; None while it has no ``index.json``."""
    raise NotImplementedError(f"reference.load.shipped_reference: {_OWNER}")


def applicable_stratum(
    ref: Reference, field_strength: float | None, fs_major: int | None
) -> str | None:
    """The stratum of ``ref`` that a subject is compared with: the first that the reference
    has of ``field:<T>|fs:<major>``, ``fs:<major>``, ``all`` (names from
    ``reference.strata``); None when it has none of them."""
    raise NotImplementedError(f"reference.load.applicable_stratum: {_OWNER}")


def write_reference(directory: Path, ref: Reference) -> None:
    """Write ``ref`` into ``directory`` in the layout of the module docstring."""
    raise NotImplementedError(f"reference.load.write_reference: {_OWNER}")


def read_reference(directory: Path) -> Reference:
    """Read a reference written by :func:`write_reference`; equal to the one written."""
    raise NotImplementedError(f"reference.load.read_reference: {_OWNER}")
