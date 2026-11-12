"""FreeSurfer surface meshes (``surf/?h.white``, ``surf/?h.pial``) in tkr RAS."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel.freesurfer as fs
import numpy as np
from jaxtyping import Float, Int


@dataclass(frozen=True)
class Surface:
    vertices: Float[np.ndarray, "n 3"]
    """tkr RAS (surface RAS) in mm, float64."""
    faces: Int[np.ndarray, "m 3"]
    """Vertex indices, wound counter-clockwise seen from outside."""


def read_surface(path: Path | str) -> Surface:
    """Read a FreeSurfer geometry file.

    Coordinates are returned as stored, which is tkr RAS, the space of ``vox2ras_tkr``. The
    ``c_ras`` in the file footer (the offset to scanner RAS) is not applied.
    """
    vertices, faces = fs.read_geometry(str(path))
    return Surface(
        vertices=np.ascontiguousarray(vertices, dtype=np.float64),
        faces=np.ascontiguousarray(faces, dtype=np.int64),
    )


def vertex_normals(surface: Surface) -> Float[np.ndarray, "n 3"]:
    """Area-weighted unit vertex normals, pointing outward for FreeSurfer's face winding.

    A vertex with no faces, or whose faces all have zero area, gets a zero vector.
    """
    v, f = surface.vertices, surface.faces
    # the cross product has length twice the face area, so summing it weights by area
    face_normals = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    normals = np.zeros_like(v)
    for corner in range(3):
        np.add.at(normals, f[:, corner], face_normals)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return np.divide(normals, length, out=np.zeros_like(normals), where=length > 0)
