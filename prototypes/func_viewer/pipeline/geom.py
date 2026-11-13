"""Geometry shared by the prototype: ITK text transforms, resampling, surfaces in T1w space.

Conventions that the checks in verify_geometry.py test:
* An ITK affine text transform holds a 3x3 matrix (row-major) and a translation, in LPS mm.
  Applied to a point of the FIXED (reference, output) image it gives the point of the MOVING
  (input) image: p_moving = A @ p_fixed + t (the files here have a zero centre).
* fMRIPrep names a file after the direction in which it moves IMAGES: from-scanner_to-T1w
  resamples a scanner-space image onto the T1w grid, so as a point map it goes T1w -> scanner.
* RAS = diag(-1, -1, 1) @ LPS, so M_ras = F @ M_lps @ F.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

FLIP = np.diag([-1.0, -1.0, 1.0, 1.0])


def read_itk_ras(path):
    """4x4 point map fixed -> moving in RAS mm, from an ITK affine text transform."""
    params = fixed = None
    kind = ""
    for line in Path(path).read_text().splitlines():
        if line.startswith("Transform:"):
            kind = line.split(":", 1)[1].strip()
        elif line.startswith("Parameters:"):
            params = np.array(line.split(":", 1)[1].split(), dtype=float)
        elif line.startswith("FixedParameters:"):
            fixed = np.array(line.split(":", 1)[1].split(), dtype=float)
    affine_kinds = ("AffineTransform", "MatrixOffsetTransformBase")
    if params is None or params.size != 12 or not kind.startswith(affine_kinds):
        raise ValueError(f"not a 3D affine ITK text transform: {kind!r}")
    a, t = params[:9].reshape(3, 3), params[9:]
    c = fixed if fixed is not None and fixed.size == 3 else np.zeros(3)
    m = np.eye(4)
    m[:3, :3] = a
    m[:3, 3] = t + c - a @ c  # p' = A (p - c) + c + t
    return FLIP @ m @ FLIP


_INSIDE = {}


def resample(vol, aff_in, shape_out, aff_out, out_to_in_ras=None, order=3):
    """`vol` (on aff_in) sampled at the voxel centres of the grid (shape_out, aff_out).

    out_to_in_ras maps a world point of the output space to the world point of the input space
    (identity when None). One interpolation, B-spline of the given order. A point inside the
    outermost voxels of the input (up to half a voxel beyond their centres) takes the edge
    value; a point outside the input's field of view is 0.
    """
    m = np.eye(4) if out_to_in_ras is None else out_to_in_ras
    vox = np.linalg.inv(aff_in) @ m @ aff_out  # output voxel -> input voxel
    out = ndimage.affine_transform(
        vol,
        vox[:3, :3],
        offset=vox[:3, 3],
        output_shape=tuple(shape_out),
        order=order,
        mode="nearest",
        output=np.float32,
    )
    key = (vol.shape, tuple(shape_out), vox.tobytes())
    if key not in _INSIDE:  # nearest-voxel lookup of a volume of ones: 1 inside the field of view
        _INSIDE[key] = ndimage.affine_transform(
            np.ones(vol.shape, np.uint8),
            vox[:3, :3],
            offset=vox[:3, 3],
            output_shape=tuple(shape_out),
            order=0,
            mode="constant",
            cval=0,
        ).astype(bool)
    out[~_INSIDE[key]] = 0
    return out


def surfaces_in_t1w(fs_dir, t1w_to_fsnative_itk):
    """White and pial surfaces of both hemispheres in the world (RAS) space of fMRIPrep's T1w.

    tkr RAS -> scanner RAS of T1.mgz (vox2ras @ inv(vox2ras_tkr)), then fsnative -> T1w.
    Moving POINTS from fsnative to T1w needs the point map of the from-T1w_to-fsnative file
    (fixed = fsnative, moving = T1w), which is the inverse of the from-fsnative_to-T1w one.
    """
    fs_dir = Path(fs_dir)
    t1 = nib.load(fs_dir / "mri" / "T1.mgz")
    tkr_to_scanner = t1.affine @ np.linalg.inv(t1.header.get_vox2ras_tkr())
    fsnative_to_t1w = read_itk_ras(t1w_to_fsnative_itk)
    m = fsnative_to_t1w @ tkr_to_scanner
    out = {}
    for hemi in ("lh", "rh"):
        for name in ("white", "pial"):
            v, f = nib.freesurfer.read_geometry(str(fs_dir / "surf" / f"{hemi}.{name}"))
            out[hemi, name] = (v @ m[:3, :3].T + m[:3, 3], f.astype(np.intp))
    return out, m


def vertex_normals(v, f):
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    out = np.zeros_like(v)
    for k in range(3):
        np.add.at(out, f[:, k], n)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def sample(vol, aff, pts, order=1):
    ijk = (pts - aff[:3, 3]) @ np.linalg.inv(aff[:3, :3]).T
    return ndimage.map_coordinates(vol, ijk.T, order=order, mode="constant", cval=np.nan)
