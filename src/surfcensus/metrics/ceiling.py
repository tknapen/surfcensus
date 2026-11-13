"""The ceiling fraction on native surfaces: how much of a region sits at the thickness
maximum or at zero. See docs/metrics.md, "The 5 mm ceiling".

FreeSurfer clamps cortical thickness at a maximum (5 mm unless recon-all was told
otherwise), so a pile-up at the maximum is a sign that the pial surface went past the
cortex. The fraction is counted on the subject's own vertices, before any resampling: an
interpolated value at the ceiling would need every corner at the ceiling.

What the fraction cannot see. The stored thickness is the mean of two distances, white to
pial and pial to white, each clamped at the maximum. A vertex therefore reads the maximum
only when both searches were cut short. With one cut short and the other measuring ``d``
it reads ``(max + d) / 2``, which lies below ``max - margin`` whenever
``d < max - 2 * margin`` (below 4.9 mm whenever ``d < 4.8``, with the defaults). So
``n_near`` is a lower bound on the vertices with a truncated search, and no threshold on
the stored value recovers the rest. The complement is the pair of truncation counts that
the thickness measurement prints into the recon-all log, for the whole hemisphere
(``subjects.trunc_*``). On ``ds000102`` sub-01 ``lh`` (FreeSurfer 6.0.1) the log says 39
and 225 truncations, while 16 vertices of the hemisphere are at exactly 5.0 and 28 at 4.9
or more (counted 2026-11-13 with :func:`ceiling_counts` over all vertices).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from jaxtyping import Bool, Float

from surfcensus.config import RegionConfig
from surfcensus.fsio.annot import Annotation
from surfcensus.fsio.registration import SphereMap
from surfcensus.log import get_logger
from surfcensus.metrics import MetricError
from surfcensus.templates import RegionName, TemplateHemi

RegionKind = Literal["template", "annot", "none"]
"""How a region was placed on a native surface: through the template and the sphere map,
from the subject's own annotation, or not at all."""

ABOVE_MAX_TOLERANCE_MM = 0.01
"""A hemisphere whose largest thickness exceeds the maximum by more than this was measured
with another maximum (:func:`thickness_above_max`)."""


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
        if self.n_region == 0:
            return float("nan")
        return (self.n_near + self.n_zero) / self.n_region


def ceiling_counts(
    thickness: Float[np.ndarray, " n"],
    region: Bool[np.ndarray, " n"],
    *,
    max_mm: float = 5.0,
    margin_mm: float = 0.1,
) -> CeilingCounts:
    """Counts of ``thickness`` values in ``region`` at the ceiling and at zero.

    ``n_exact`` counts values of at least ``max_mm``, ``n_near`` values of at least
    ``max_mm - margin_mm`` and ``n_zero`` values equal to 0. The two limits are compared in
    the type of ``thickness``: a float32 file value that is the nearest float32 to 4.9
    counts as 4.9, although it differs from the float64 4.9 in the eighth digit.

    A NaN is a vertex of the region and counts as nothing else; an infinite value is above
    any maximum. ``n_near`` is a lower bound on the vertices whose thickness search was
    truncated, see the module docstring.

    Raises ValueError when the two arrays differ in length, and unless
    ``0 <= margin_mm < max_mm``: with a margin as large as the maximum, a vertex at zero
    would be counted as near the ceiling as well.
    """
    if len(thickness) != len(region):
        raise ValueError(
            f"{len(thickness)} thickness values for a region of {len(region)} vertices"
        )
    if not 0 <= margin_mm < max_mm:
        raise ValueError(
            f"the margin must be at least 0 and below the maximum, got a margin of "
            f"{margin_mm} mm and a maximum of {max_mm} mm"
        )
    values = thickness[region]
    kind = values.dtype.type
    exact = values >= kind(max_mm)
    near = values >= kind(max_mm - margin_mm)
    return CeilingCounts(
        n_region=int(values.size),
        n_exact=int(np.count_nonzero(exact)),
        n_near=int(np.count_nonzero(near)),
        n_zero=int(np.count_nonzero(values == 0)),
    )


def thickness_above_max(thickness: Float[np.ndarray, " n"], *, max_mm: float = 5.0) -> bool:
    """Whether the largest thickness exceeds ``max_mm`` by more than
    :data:`ABOVE_MAX_TOLERANCE_MM`: recon-all then ran with another maximum, and the counts
    at ``max_mm`` do not mark truncated searches. Such a hemisphere gets the flag
    ``thickness_above_max``; its counts are still reported.

    The limit is compared in the type of ``thickness``, like those of
    :func:`ceiling_counts`. NaN values are ignored; an array without a number gives False.
    """
    values = thickness[~np.isnan(thickness)]
    if values.size == 0:
        return False
    return bool(values.max() > values.dtype.type(max_mm + ABOVE_MAX_TOLERANCE_MM))


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

    The two kinds are different regions. The ``template`` search region also holds the
    part of the posterior parcels (precuneus and fusiform by default) that lies within
    the template's cut in front of the pole, which only the template can cut; the ``annot``
    one is the occipital parcels alone. The annotation cannot give ``medial``,
    ``inferior`` and ``lateral``, which need the same cut. It also cannot give a region
    when its colour table lacks one of the region's parcels: the vertices of the remaining
    parcels would be another region under the same name, so there is none, and a warning
    names the parcels. A parcel that the table has and no vertex carries is not lacking.

    ``regions`` is read for the kind ``annot`` only. The masks of ``template_hemi`` were
    cut with the config the template was built with.

    Raises ``MetricError("vertex count mismatch", ...)`` when the annotation and the source
    sphere of the map differ in length, and ValueError when the map was not built onto the
    mesh of ``template_hemi``.
    """
    if smap is not None and template_hemi is not None:
        on_template = template_hemi.regions[name]
        if len(on_template) != smap.n_target:
            raise ValueError(
                f"the sphere map has {smap.n_target} target vertices and the template mesh "
                f"{len(on_template)}"
            )
        mask = on_template[smap.nearest_target]
        if annot is not None:
            if len(annot.labels) != smap.n_source:
                raise MetricError(
                    "vertex count mismatch",
                    f"the annotation has {len(annot.labels)} vertices and the sphere "
                    f"{smap.n_source}",
                )
            mask = mask & annot.cortex()
        return mask, "template"
    if annot is None:
        return None, "none"
    if name == "search":
        parcels = regions.occipital
    elif name == "control":
        parcels = regions.control
    else:
        return None, "none"
    missing = [parcel for parcel in parcels if parcel not in annot.names]
    if missing:
        get_logger().warning(
            "The annotation has no parcel %s: no %s region from it.", ", ".join(missing), name
        )
        return None, "none"
    return annot.mask(*parcels), "annot"
