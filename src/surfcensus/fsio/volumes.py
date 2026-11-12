"""FreeSurfer volumes (``mri/T1.mgz``, ``mri/brainmask.mgz``) and the display intensity window."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from jaxtyping import Float, Shaped
from nibabel.freesurfer.mghformat import MGHImage
from scipy import ndimage


@dataclass(frozen=True)
class Volume:
    data: Shaped[np.ndarray, "i j k"]
    """Voxel values in the stored dtype. An integer index is the centre of a voxel."""
    vox2ras_tkr: Float[np.ndarray, "4 4"]
    """Voxel indices to tkr RAS (the space of the surfaces), float64."""
    vox2ras: Float[np.ndarray, "4 4"]
    """Voxel indices to scanner RAS, float64."""
    voxel_size: tuple[float, float, float]
    """mm along voxel axes 0, 1, 2."""

    @property
    def shape(self) -> tuple[int, int, int]:
        i, j, k = self.data.shape
        return (i, j, k)


def read_volume(path: Path | str) -> Volume:
    """Read a 3D ``.mgz`` or ``.mgh`` volume; a single-frame 4D volume is squeezed to 3D."""
    img = nib.load(str(path))
    if not isinstance(img, MGHImage):
        raise ValueError(f"not an MGH or MGZ volume: {path}")
    data = np.asanyarray(img.dataobj)
    data = data.astype(data.dtype.newbyteorder("="), copy=False)  # MGH stores big-endian
    if data.ndim == 4 and data.shape[3] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"expected a 3D volume, got shape {data.shape}: {path}")
    header = img.header
    dx, dy, dz = (float(z) for z in header.get_zooms()[:3])
    return Volume(
        data=data,
        vox2ras_tkr=np.asarray(header.get_vox2ras_tkr(), dtype=np.float64),
        vox2ras=np.asarray(header.get_vox2ras(), dtype=np.float64),
        voxel_size=(dx, dy, dz),
    )


def intensity_window(
    t1: Volume,
    brainmask: Volume | None,
    percentiles: tuple[float, float] = (1.0, 99.5),
    dilation: int = 3,
) -> tuple[float, float]:
    """Display window ``(lo, hi)`` with ``hi > lo``: percentiles of T1 inside the brain.

    With a brainmask, the voxels are those of ``brainmask > 0`` after ``dilation`` iterations
    of binary dilation (6-connected). Without one, or if it selects nothing, the non-zero T1
    voxels. An all-zero T1 gives ``(0.0, 1.0)``.
    """
    values = np.empty(0)
    if brainmask is not None:
        if brainmask.shape != t1.shape:
            raise ValueError(f"brainmask shape {brainmask.shape} differs from T1 {t1.shape}")
        mask = brainmask.data > 0
        if dilation > 0:  # scipy reads iterations < 1 as "repeat until nothing changes"
            mask = ndimage.binary_dilation(mask, iterations=dilation)
        values = t1.data[mask]
    if values.size == 0:
        values = t1.data[t1.data != 0]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (0.0, 1.0)
    lo, hi = (float(p) for p in np.percentile(values, percentiles))
    if not hi > lo:
        hi = lo + 1.0
    return (lo, hi)
