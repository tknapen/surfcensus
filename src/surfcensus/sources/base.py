"""File roles shared by every source. The Source protocol follows in milestone 2."""

from __future__ import annotations

from enum import StrEnum


class FileRole(StrEnum):
    """A file inside a FreeSurfer subject directory, named by what it is used for.

    The values are the role names of the ``files`` table in docs/architecture.md.
    """

    T1 = "t1"
    BRAINMASK = "brainmask"
    LH_WHITE = "lh_white"
    RH_WHITE = "rh_white"
    LH_PIAL = "lh_pial"
    RH_PIAL = "rh_pial"
    LH_THICKNESS = "lh_thickness"
    RH_THICKNESS = "rh_thickness"
    LH_SPHERE_REG = "lh_sphere_reg"
    RH_SPHERE_REG = "rh_sphere_reg"
    LH_ANNOT = "lh_annot"
    RH_ANNOT = "rh_annot"
    LH_STATS = "lh_stats"
    RH_STATS = "rh_stats"
    ASEG_STATS = "aseg_stats"
    TALAIRACH = "talairach"
    RECON_LOG = "recon_log"
    BUILD_STAMP = "build_stamp"


RENDER_ROLES: tuple[FileRole, ...] = (
    FileRole.T1,
    FileRole.LH_WHITE,
    FileRole.RH_WHITE,
    FileRole.LH_PIAL,
    FileRole.RH_PIAL,
)
"""Roles a render cannot do without. ``brainmask`` is used when present."""
