"""Path patterns that identify a FreeSurfer subject directory and the files inside it.

Read-only: nothing here opens, creates or changes a file.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from surfcensus.sources.base import RENDER_ROLES, FileRole


def _pial(hemi: str) -> tuple[str, ...]:
    # FreeSurfer 7 may write ?h.pial as a link to ?h.pial.T1 or ?h.pial.T2, and a copy of the
    # directory can lose the link, so the variants are tried after the plain name
    return tuple(f"surf/{hemi}.pial{suffix}" for suffix in ("", ".T2", ".FLAIR", ".T1"))


ROLE_CANDIDATES: dict[FileRole, tuple[str, ...]] = {
    FileRole.T1: ("mri/T1.mgz",),
    FileRole.BRAINMASK: ("mri/brainmask.mgz",),
    FileRole.LH_WHITE: ("surf/lh.white",),
    FileRole.RH_WHITE: ("surf/rh.white",),
    FileRole.LH_PIAL: _pial("lh"),
    FileRole.RH_PIAL: _pial("rh"),
    FileRole.LH_THICKNESS: ("surf/lh.thickness",),
    FileRole.RH_THICKNESS: ("surf/rh.thickness",),
    FileRole.LH_SPHERE_REG: ("surf/lh.sphere.reg",),
    FileRole.RH_SPHERE_REG: ("surf/rh.sphere.reg",),
    FileRole.LH_ANNOT: ("label/lh.aparc.annot",),
    FileRole.RH_ANNOT: ("label/rh.aparc.annot",),
    FileRole.LH_STATS: ("stats/lh.aparc.stats",),
    FileRole.RH_STATS: ("stats/rh.aparc.stats",),
    FileRole.ASEG_STATS: ("stats/aseg.stats",),
    FileRole.TALAIRACH: ("mri/transforms/talairach.xfm",),
    FileRole.RECON_LOG: ("scripts/recon-all.log",),
    FileRole.BUILD_STAMP: ("scripts/build-stamp.txt",),
}
"""Paths relative to the subject directory for each role, in order of preference."""


def resolve_subject_files(subject_dir: Path | str) -> dict[FileRole, Path]:
    """The first existing candidate of each role; roles with none are absent.

    A symlink counts if its target exists, a dangling one does not. The returned path is the
    candidate itself, not the link target; ``Path.resolve()`` tells which variant a link names.
    """
    root = Path(subject_dir)
    files: dict[FileRole, Path] = {}
    for role, candidates in ROLE_CANDIDATES.items():
        for candidate in candidates:
            path = root / candidate
            if path.is_file():
                files[role] = path
                break
    return files


def missing_roles(
    files: Mapping[FileRole, Path], required: Iterable[FileRole] = RENDER_ROLES
) -> list[FileRole]:
    """Required roles that have no file, in the order of ``required``."""
    return [role for role in required if role not in files]


def is_freesurfer_subject_dir(path: Path | str) -> bool:
    """A directory with ``mri/T1.mgz`` and a left pial surface that is not an fsaverage template.

    Templates are recognized by name, of the directory as given and of what it links to
    (``SUBJECTS_DIR/fsaverage`` is usually a link into the FreeSurfer installation).
    """
    path = Path(path)
    if not path.is_dir():
        return False
    if any(name.startswith("fsaverage") for name in (path.name, path.resolve().name)):
        return False
    return (path / "mri" / "T1.mgz").is_file() and any(
        (path / candidate).is_file() for candidate in ROLE_CANDIDATES[FileRole.LH_PIAL]
    )
