"""A synthetic ``fsaverage`` directory with every file the template is built from, written
with nibabel, and the closed-form rules it is laid out by.

The mesh is ``synth.icosphere(order)``, whose vertex order nests like that of the real
fsaverage: the first ``10 * 4**k + 2`` vertices are the mesh of order ``k``, and every later
vertex is the midpoint of two earlier ones. So a directory of order 7 is a valid input of the
default code path, which coarsens it to fsaverage6.

A point of the template is named by its direction ``u`` on the unit sphere, which is also
its anatomical direction: ``-y`` is posterior, ``+z`` superior, and the medial side is ``+x``
on the left hemisphere and ``-x`` on the right (as on the real fsaverage spheres, where the
occipital parcels lie around ``-y`` and the medial wall towards ``+x`` on ``lh``). The white
surface is the white ellipsoid of the synthetic subject, ``centre + u * WHITE_SEMI_AXES``,
so its most posterior point, the occipital pole, is the vertex at ``u = (0, -1, 0)``.

Parcels, by rule (:func:`parcel_of`), with ``m`` the medial coordinate and angles in degrees:

* medial strip, no label and not cortex: ``m > MEDIAL_WALL_M``;
* posterior cap, within ``POSTERIOR_CAP_DEG`` of ``-y``, in four sectors by the azimuth
  ``atan2(z, m)``: pericalcarine around the medial direction, cuneus above, lingual below,
  lateraloccipital on the lateral side;
* a ring in front of the cap up to ``y < RING_FRONT_Y``: precuneus above (``z >= 0``),
  fusiform below. It reaches farther forward than the default cut of the search region
  (45 mm from the pole is ``y = -0.224``), so the cut divides both parcels;
* frontal cap, ``y > FRONTAL_CAP_Y``, in three sectors: superiorfrontal,
  rostralmiddlefrontal, caudalmiddlefrontal;
* the band between ring and frontal cap: superiorparietal above, inferiortemporal below.

The annotation carries the 36 Desikan names in the order of a real ``aparc.annot`` (read
from ``ds000102`` sub-01 on 2026-11-13) with colours made by rule; 11 of them have vertices.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import nibabel as nib
import nibabel.freesurfer as fs
import numpy as np
from synth import WHITE_SEMI_AXES, icosphere  # pyrefly: ignore[missing-import]
from synth_tree import write_label  # pyrefly: ignore[missing-import]

from surfcensus.config import RegionConfig
from surfcensus.sources.base import HEMIS

__all__ = [
    "AVG_AREA_FACTOR_CAP",
    "AVG_AREA_FACTOR_REST",
    "DESIKAN_NAMES",
    "REGION_NAMES",
    "SPHERE_RADIUS",
    "TEMPLATE_BUILD_STAMP",
    "WHITE_CENTRE",
    "SyntheticTemplate",
    "desikan_colours",
    "direction",
    "expected_regions",
    "medial_coordinate",
    "mesh_vertex_areas",
    "parcel_of",
    "template_mesh",
    "template_white",
    "write_label",
    "write_synthetic_template",
]

DESIKAN_NAMES: tuple[str, ...] = (
    "unknown",
    "bankssts",
    "caudalanteriorcingulate",
    "caudalmiddlefrontal",
    "corpuscallosum",
    "cuneus",
    "entorhinal",
    "fusiform",
    "inferiorparietal",
    "inferiortemporal",
    "isthmuscingulate",
    "lateraloccipital",
    "lateralorbitofrontal",
    "lingual",
    "medialorbitofrontal",
    "middletemporal",
    "parahippocampal",
    "paracentral",
    "parsopercularis",
    "parsorbitalis",
    "parstriangularis",
    "pericalcarine",
    "postcentral",
    "posteriorcingulate",
    "precentral",
    "precuneus",
    "rostralanteriorcingulate",
    "rostralmiddlefrontal",
    "superiorfrontal",
    "superiorparietal",
    "superiortemporal",
    "supramarginal",
    "frontalpole",
    "temporalpole",
    "transversetemporal",
    "insula",
)
REGION_NAMES = ("search", "medial", "inferior", "lateral", "control")

SPHERE_RADIUS = 100.0
MEDIAL_WALL_M = 0.85
POSTERIOR_CAP_DEG = 40.0
RING_FRONT_Y = 0.1
FRONTAL_CAP_Y = 0.6
WHITE_CENTRE: dict[str, tuple[float, float, float]] = {
    "lh": (-34.0, -10.0, 5.0),
    "rh": (26.0, -12.0, 5.0),
}
"""Centre of each hemisphere's white ellipsoid, as in the default synthetic subject."""
AVG_AREA_FACTOR_CAP = 1.5
AVG_AREA_FACTOR_REST = 1.2
"""``surf/?h.white.avg.area.mgh`` holds the vertex areas of the white mesh times 1.5 inside
the posterior cap and times 1.2 elsewhere, so the two area sources can be told apart."""
TEMPLATE_BUILD_STAMP = "freesurfer-x86_64-synthetic-linux-gnu-dev0-20261113"
"""Like the stamp of the real fsaverage (``freesurfer-x86_64-redhat-linux-gnu-dev4-20090216``),
it holds no version number."""
_CREATE_STAMP = "created by surfcensus tests/synth_template.py"
_MEDIAL_SIGN = {"lh": 1.0, "rh": -1.0}


@dataclass(frozen=True)
class SyntheticTemplate:
    path: Path
    order: int
    masks: Mapping[str, Mapping[str, np.ndarray]]
    """Hemisphere -> name -> boolean mask over the vertices of the written mesh. Names:
    ``cortex``, ``medial_wall``, ``posterior_cap``, every Desikan name that has vertices,
    and the five regions of :data:`REGION_NAMES` under the default ``RegionConfig``. The
    masks of a coarser mesh are the first entries of these."""
    n_vertices: int
    pole_vertex: int
    """The vertex at ``(0, -1, 0)``: the most posterior white vertex of both hemispheres. It
    is among the first 42, so it is a vertex of every mesh from order 1 on."""
    with_avg_area: bool


@lru_cache(maxsize=4)
def _mesh(order: int) -> tuple[np.ndarray, np.ndarray]:
    directions, faces = icosphere(order)
    directions.setflags(write=False)
    faces = faces.astype(np.int32)
    faces.setflags(write=False)
    return directions, faces


def template_mesh(order: int) -> tuple[np.ndarray, np.ndarray]:
    """``(directions, faces)`` of the template mesh of ``order``: unit vectors (n, 3) float64
    and outward-wound faces (m, 3) int32. Read-only arrays, shared between calls."""
    return _mesh(order)


def medial_coordinate(directions: np.ndarray, hemi: str) -> np.ndarray:
    """``m``: +1 at the most medial point of the hemisphere, -1 at the most lateral."""
    return _MEDIAL_SIGN[hemi] * np.asarray(directions, dtype=float)[..., 0]


def direction(
    hemi: str, *, from_pole_deg: float, azimuth_deg: float = 0.0
) -> tuple[float, float, float]:
    """The unit vector ``from_pole_deg`` away from the occipital pole ``(0, -1, 0)``, towards
    ``azimuth_deg``: 0 is medial, 90 superior, 180 lateral, -90 inferior. With
    ``from_pole_deg`` below ``POSTERIOR_CAP_DEG`` these are the pericalcarine, cuneus,
    lateraloccipital and lingual sectors."""
    polar, azimuth = np.deg2rad(from_pole_deg), np.deg2rad(azimuth_deg)
    m = np.sin(polar) * np.cos(azimuth)
    return (
        float(_MEDIAL_SIGN[hemi] * m),
        float(-np.cos(polar)),
        float(np.sin(polar) * np.sin(azimuth)),
    )


def parcel_of(directions: np.ndarray, hemi: str) -> np.ndarray:
    """Index into :data:`DESIKAN_NAMES` of the parcel at each direction, int32; -1 on the
    medial strip. The rule is in the module docstring."""
    u = np.asarray(directions, dtype=float)
    m, y, z = medial_coordinate(u, hemi), u[..., 1], u[..., 2]
    azimuth = np.rad2deg(np.arctan2(z, m))
    index = DESIKAN_NAMES.index

    cap = np.select(
        [
            np.abs(azimuth) <= 45.0,
            (azimuth > 45.0) & (azimuth <= 135.0),
            (azimuth < -45.0) & (azimuth >= -135.0),
        ],
        [index("pericalcarine"), index("cuneus"), index("lingual")],
        default=index("lateraloccipital"),
    )
    frontal = np.select(
        [(azimuth >= -60.0) & (azimuth < 60.0), azimuth >= 60.0],
        [index("superiorfrontal"), index("rostralmiddlefrontal")],
        default=index("caudalmiddlefrontal"),
    )
    upper = z >= 0
    parcel = np.select(
        [
            y < -np.cos(np.deg2rad(POSTERIOR_CAP_DEG)),
            y < RING_FRONT_Y,
            y > FRONTAL_CAP_Y,
        ],
        [
            cap,
            np.where(upper, index("precuneus"), index("fusiform")),
            frontal,
        ],
        default=np.where(upper, index("superiorparietal"), index("inferiortemporal")),
    )
    return np.where(m > MEDIAL_WALL_M, -1, parcel).astype(np.int32)


def template_white(directions: np.ndarray, hemi: str) -> np.ndarray:
    """The white surface position of each direction, in the template's surface RAS (float64)."""
    return np.array(WHITE_CENTRE[hemi]) + np.asarray(directions, dtype=float) * np.array(
        WHITE_SEMI_AXES
    )


def mesh_vertex_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Per vertex, a third of the areas of the faces around it (float64)."""
    a, b, c = (vertices[faces[:, k]] for k in range(3))
    face_area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    area = np.zeros(len(vertices))
    for corner in range(3):
        np.add.at(area, faces[:, corner], face_area / 3.0)
    return area


def expected_regions(
    directions: np.ndarray, hemi: str, regions: RegionConfig | None = None
) -> dict[str, np.ndarray]:
    """The five region masks at ``directions`` by the rule of docs/metrics.md, worked out
    from the fixture's own closed form and not through the package:

    ``search`` is the occipital parcels plus the part of the posterior parcels whose white
    ``y`` lies within ``cut_mm`` of the pole's; ``medial`` and ``inferior`` are ``search``
    restricted to those parcel lists, ``lateral`` is the rest of ``search``, and ``control``
    is the control parcels. All lie in cortex, which here is every vertex with a parcel.

    The white surface is stored as float32, so a vertex closer to the cut than rounding
    could fall on either side. ``tests/test_synth_template.py`` checks that with the default
    config no vertex of the meshes used lies within a micrometre of it.
    """
    regions = regions or RegionConfig()
    parcel = parcel_of(directions, hemi)

    def named(names: tuple[str, ...]) -> np.ndarray:
        return np.isin(parcel, [DESIKAN_NAMES.index(name) for name in names])

    white_y = template_white(directions, hemi)[..., 1]
    pole_y = WHITE_CENTRE[hemi][1] - WHITE_SEMI_AXES[1]
    search = named(regions.occipital) | (
        named(regions.posterior) & (white_y <= pole_y + regions.cut_mm)
    )
    medial = search & named(regions.medial)
    inferior = search & named(regions.inferior)
    return {
        "search": search,
        "medial": medial,
        "inferior": inferior,
        "lateral": search & ~medial & ~inferior,
        "control": named(regions.control),
    }


def desikan_colours() -> np.ndarray:
    """One distinct RGBT row per Desikan name, made by rule (not FreeSurfer's colours); none
    is black, which an annotation file cannot tell from "no label"."""
    index = np.arange(len(DESIKAN_NAMES))
    return np.column_stack(
        [10 + 6 * index, 250 - 5 * index, (37 * index) % 256, np.zeros_like(index)]
    ).astype(np.int32)


def write_synthetic_template(
    root: Path, *, name: str = "fsaverage", order: int = 7, with_avg_area: bool = True
) -> SyntheticTemplate:
    """Write ``root/name`` with the eight required template files, ``scripts/build-stamp.txt``
    and, unless ``with_avg_area`` is False, ``surf/?h.white.avg.area.mgh``.

    ``surf/?h.sphere.reg`` is the mesh at radius 100 and ``surf/?h.white`` the white
    ellipsoid, both with the same faces. ``label/?h.aparc.annot`` follows :func:`parcel_of`.
    ``label/?h.cortex.label`` lists every vertex with a parcel, in descending order: the real
    file is not sorted either. The average-area file has the shape and type of the real one,
    ``(n, 1, 1)`` float32. The bytes written depend on ``order`` and ``with_avg_area`` only.
    """
    path = Path(root) / name
    for folder in ("surf", "label", "scripts"):
        (path / folder).mkdir(parents=True, exist_ok=True)
    directions, faces = template_mesh(order)
    cap = directions[:, 1] < -np.cos(np.deg2rad(POSTERIOR_CAP_DEG))
    masks: dict[str, dict[str, np.ndarray]] = {}
    for hemi in HEMIS:
        white = template_white(directions, hemi)
        parcel = parcel_of(directions, hemi)
        cortex = parcel >= 0
        fs.write_geometry(
            str(path / "surf" / f"{hemi}.sphere.reg"),
            directions * SPHERE_RADIUS,
            faces,
            create_stamp=_CREATE_STAMP,
        )
        fs.write_geometry(
            str(path / "surf" / f"{hemi}.white"), white, faces, create_stamp=_CREATE_STAMP
        )
        fs.write_annot(
            str(path / "label" / f"{hemi}.aparc.annot"),
            parcel,
            desikan_colours(),
            list(DESIKAN_NAMES),
            fill_ctab=True,
        )
        listed = np.flatnonzero(cortex)[::-1]
        write_label(path / "label" / f"{hemi}.cortex.label", listed, white[listed])
        if with_avg_area:
            factor = np.where(cap, AVG_AREA_FACTOR_CAP, AVG_AREA_FACTOR_REST)
            area = (mesh_vertex_areas(white, faces) * factor).astype(np.float32)
            nib.save(
                nib.MGHImage(area.reshape(-1, 1, 1), np.eye(4)),
                str(path / "surf" / f"{hemi}.white.avg.area.mgh"),
            )
        hemi_masks = {"cortex": cortex, "medial_wall": ~cortex, "posterior_cap": cap.copy()}
        for index in np.unique(parcel[cortex]):
            hemi_masks[DESIKAN_NAMES[index]] = parcel == index
        hemi_masks.update(expected_regions(directions, hemi))
        for mask in hemi_masks.values():
            mask.setflags(write=False)
        masks[hemi] = hemi_masks
    (path / "scripts" / "build-stamp.txt").write_text(TEMPLATE_BUILD_STAMP + "\n")
    pole = int(np.argmin(directions[:, 1]))
    return SyntheticTemplate(
        path=path,
        order=order,
        masks=masks,
        n_vertices=len(directions),
        pole_vertex=pole,
        with_avg_area=with_avg_area,
    )
