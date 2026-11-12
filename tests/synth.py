"""A synthetic FreeSurfer subject with analytically known geometry, written with nibabel.

Each hemisphere is an ellipsoid; white and pial share their faces. ``T1.mgz`` is laid out
like a FreeSurfer conformed volume (LIA) with a non-zero ``c_ras``, so scanner RAS and tkr
RAS differ, and its intensities follow from the same ellipsoids evaluated at voxel centres.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import nibabel.freesurfer as fs
import numpy as np

WHITE_SEMI_AXES = (26.0, 58.0, 42.0)
PIAL_SEMI_AXES = (28.5, 60.5, 44.5)
HEAD_SEMI_AXES = (75.0, 78.0, 70.0)
T1_WHITE, T1_GRAY, T1_HEAD = 110, 70, 25


@dataclass(frozen=True)
class SyntheticSubject:
    path: Path
    voxel_size: float
    shape: tuple[int, int, int]
    c_ras: tuple[float, float, float]
    midline_x: float
    lh_centre: tuple[float, float, float]
    """Ellipsoid centre in tkr RAS; likewise ``rh_centre``."""
    rh_centre: tuple[float, float, float]
    white_semi_axes: tuple[float, float, float]
    """Along tkr x, y, z, in mm; likewise ``pial_semi_axes``."""
    pial_semi_axes: tuple[float, float, float]
    n_vertices: int
    """Per surface."""


def icosphere(subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    """Unit sphere mesh: (vertices float64 (n, 3), faces int (m, 3)), normals outward.

    ``n = 10 * 4**subdivisions + 2``. From one subdivision on, the six axis points
    ``(+-1, 0, 0)``, ``(0, +-1, 0)``, ``(0, 0, +-1)`` are vertices.
    """
    phi = (1 + 5**0.5) / 2
    v = np.array(
        [
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
        ]
    )  # fmt: skip
    f = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ]
    )  # fmt: skip
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    for _ in range(subdivisions):
        edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
        unique, inverse = np.unique(edges, axis=0, return_inverse=True)
        mid = v[unique[:, 0]] + v[unique[:, 1]]
        mid /= np.linalg.norm(mid, axis=1, keepdims=True)
        ab, bc, ca = len(v) + inverse.reshape(3, -1)
        a, b, c = f.T
        f = np.concatenate(
            [
                np.column_stack([a, ab, ca]),
                np.column_stack([b, bc, ab]),
                np.column_stack([c, ca, bc]),
                np.column_stack([ab, bc, ca]),
            ]
        )
        v = np.concatenate([v, mid])
    return v, f


def conformed_vox2ras_tkr(voxel_size: float, shape: tuple[int, int, int]) -> np.ndarray:
    """The tkr affine FreeSurfer gives an LIA volume, as stored (float32 precision)."""
    d, (n0, n1, n2) = voxel_size, shape
    exact = [[-d, 0, 0, d * n0 / 2], [0, 0, d, -d * n2 / 2], [0, -d, 0, d * n1 / 2], [0, 0, 0, 1]]
    return np.array(exact, dtype=np.float32).astype(np.float64)


def snapshot_tree(root: Path) -> dict[str, tuple[int, int]]:
    """Relative path -> (size, mtime in ns) of everything under ``root``, links not followed."""
    return {
        p.relative_to(root).as_posix(): (p.lstat().st_size, p.lstat().st_mtime_ns)
        for p in sorted(root.rglob("*"))
    }


def file_modes_apply(scratch: Path) -> bool:
    """Whether a file of mode 000 in ``scratch`` is unreadable; it is not for root."""
    probe = scratch / "mode-probe"
    probe.write_text("x")
    probe.chmod(0)
    try:
        probe.read_text()
    except PermissionError:
        return True
    finally:
        probe.unlink()
    return False


def write_synthetic_subject(
    root: Path,
    *,
    name: str = "sub-synth",
    voxel_size: float = 1.0,
    shape: tuple[int, int, int] = (160, 160, 160),
    c_ras: tuple[float, float, float] = (3.0, -7.0, 11.0),
    midline_x: float = -4.0,
    subdivisions: int = 5,
    pial_variant: str = "pial",
    with_brainmask: bool = True,
) -> SyntheticSubject:
    """Write ``root/name`` with ``surf/?h.white``, ``surf/?h.<pial_variant>``, ``mri/T1.mgz``
    and, optionally, ``mri/brainmask.mgz``."""
    subject = SyntheticSubject(
        path=Path(root) / name,
        voxel_size=voxel_size,
        shape=shape,
        c_ras=c_ras,
        midline_x=midline_x,
        lh_centre=(midline_x - 30.0, -10.0, 5.0),
        rh_centre=(midline_x + 30.0, -12.0, 5.0),
        white_semi_axes=WHITE_SEMI_AXES,
        pial_semi_axes=PIAL_SEMI_AXES,
        n_vertices=10 * 4**subdivisions + 2,
    )
    (subject.path / "surf").mkdir(parents=True, exist_ok=True)
    (subject.path / "mri").mkdir(exist_ok=True)

    unit, faces = icosphere(subdivisions)
    volume_info = {  # the footer recon-all writes; c_ras must not move the vertices
        "head": np.array([2, 0, 20], dtype=np.int32),
        "valid": "1  # volume info valid",
        "filename": "../mri/filled-pretess255.mgz",
        "volume": np.array(shape),
        "voxelsize": np.full(3, voxel_size),
        "xras": np.array([-1.0, 0.0, 0.0]),
        "yras": np.array([0.0, 0.0, -1.0]),
        "zras": np.array([0.0, 1.0, 0.0]),
        "cras": np.array(c_ras),
    }
    for hemi, centre in (("lh", subject.lh_centre), ("rh", subject.rh_centre)):
        for surface, semi in (("white", WHITE_SEMI_AXES), (pial_variant, PIAL_SEMI_AXES)):
            fs.write_geometry(
                str(subject.path / "surf" / f"{hemi}.{surface}"),
                np.array(centre) + unit * np.array(semi),
                faces,
                create_stamp="created by surfcensus tests/synth.py",
                volume_info=volume_info,
            )

    # scanner RAS = tkr RAS + c_ras; the header derives c_ras (Pxyz_c) from this affine
    d = float(voxel_size)
    vox2ras = np.array(
        [
            [-d, 0, 0, d * shape[0] / 2 + c_ras[0]],
            [0, 0, d, -d * shape[2] / 2 + c_ras[1]],
            [0, -d, 0, d * shape[1] / 2 + c_ras[2]],
            [0, 0, 0, 1],
        ]
    )
    # voxel centres in tkr RAS, from the affine as it will be read back: voxel axis 0 runs
    # along x, axis 1 along z, axis 2 along y
    tkr = nib.MGHImage(np.zeros(shape, dtype=np.uint8), vox2ras).header.get_vox2ras_tkr()
    tkr = tkr.astype(float)
    x = tkr[0, 0] * np.arange(shape[0], dtype=float) + tkr[0, 3]
    z = tkr[2, 1] * np.arange(shape[1], dtype=float) + tkr[2, 3]
    y = tkr[1, 2] * np.arange(shape[2], dtype=float) + tkr[1, 3]

    def inside(centre: tuple[float, float, float], semi: tuple[float, float, float]) -> np.ndarray:
        return (
            ((x - centre[0]) / semi[0])[:, None, None] ** 2
            + ((z - centre[2]) / semi[2])[None, :, None] ** 2
            + ((y - centre[1]) / semi[1])[None, None, :] ** 2
        ) <= 1.0

    pial = inside(subject.lh_centre, PIAL_SEMI_AXES) | inside(subject.rh_centre, PIAL_SEMI_AXES)
    white = inside(subject.lh_centre, WHITE_SEMI_AXES) | inside(subject.rh_centre, WHITE_SEMI_AXES)
    t1 = np.zeros(shape, dtype=np.uint8)
    t1[inside((midline_x, -11.0, 5.0), HEAD_SEMI_AXES)] = T1_HEAD
    t1[pial] = T1_GRAY
    t1[white] = T1_WHITE
    nib.save(nib.MGHImage(t1, vox2ras), str(subject.path / "mri" / "T1.mgz"))
    if with_brainmask:
        brainmask = np.where(pial, t1, 0).astype(np.uint8)
        nib.save(nib.MGHImage(brainmask, vox2ras), str(subject.path / "mri" / "brainmask.mgz"))
    return subject
