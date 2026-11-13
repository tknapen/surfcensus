"""The ceiling fraction on native surfaces: how much of a region sits at the thickness
maximum or at zero. See docs/metrics.md, "The 5 mm ceiling".

Interface of milestone 3. ``CeilingCounts`` and ``RegionKind`` are final; the functions are
written by package 3.G.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from jaxtyping import Bool, Float

from surfcensus.config import RegionConfig
from surfcensus.fsio.annot import Annotation
from surfcensus.fsio.registration import SphereMap
from surfcensus.templates import RegionName, TemplateHemi

_OWNER = "package 3.G (ceiling)"

RegionKind = Literal["template", "annot", "none"]
"""How a region was placed on a native surface: through the template and the sphere map,
from the subject's own annotation, or not at all."""


@dataclass(frozen=True)
class CeilingCounts:
    n_region: int
    n_exact: int
    """Vertices at or above the maximum: both searches of the thickness measurement were
    truncated."""
    n_near: int
    """Vertices at or above the maximum minus the margin; includes ``n_exact``."""
    n_zero: int
    """Vertices of the region with thickness 0 (the region lies inside the cortex mask)."""

    @property
    def fraction(self) -> float:
        """``(n_near + n_zero) / n_region``; NaN for an empty region."""
        raise NotImplementedError(f"metrics.ceiling.CeilingCounts.fraction: {_OWNER}")


def ceiling_counts(
    thickness: Float[np.ndarray, " n"],
    region: Bool[np.ndarray, " n"],
    *,
    max_mm: float = 5.0,
    margin_mm: float = 0.1,
) -> CeilingCounts:
    """Counts of ``thickness`` values in ``region`` at the ceiling and at zero."""
    raise NotImplementedError(f"metrics.ceiling.ceiling_counts: {_OWNER}")


def native_region(
    name: RegionName,
    *,
    annot: Annotation | None,
    smap: SphereMap | None,
    template_hemi: TemplateHemi | None,
    regions: RegionConfig,
) -> tuple[Bool[np.ndarray, " n"] | None, RegionKind]:
    """A region of the template on the native surface of one hemisphere.

    With a template and a sphere map (kind ``template``): the native vertices whose nearest
    template vertex is in the region and which, when an annotation is given, it calls
    cortex. Without them (kind ``annot``): the subject's own parcels, ``regions.occipital``
    for ``search`` and ``regions.control`` for ``control``. With neither, or for a region
    that the annotation cannot give: ``(None, "none")``.
    """
    raise NotImplementedError(f"metrics.ceiling.native_region: {_OWNER}")
