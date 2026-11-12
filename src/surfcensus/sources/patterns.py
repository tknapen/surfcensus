"""Path patterns that identify a FreeSurfer subject directory and the files inside it.

Read-only: nothing here opens, creates or changes a file.
"""

from __future__ import annotations

import errno
import stat
from collections.abc import Iterable, Mapping
from fnmatch import fnmatchcase
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
    FileRole.RECON_LOG_LH: ("scripts/recon-all-lh.log",),
    FileRole.RECON_LOG_RH: ("scripts/recon-all-rh.log",),
}
"""Paths relative to the subject directory for each role, in order of preference."""

TEMPLATE_FILES: tuple[str, ...] = (
    "surf/{h}.sphere.reg",
    "surf/{h}.white",
    "label/{h}.cortex.label",
    "label/{h}.aparc.annot",
)
"""What an fsaverage directory must hold to serve as the template, for each hemisphere ``h``,
relative to that directory."""

TEMPLATE_OPTIONAL_FILES: tuple[str, ...] = ("surf/{h}.white.avg.area.mgh",)
"""Used when present."""

TEMPLATE_NAME_PATTERNS: tuple[str, ...] = (
    "fsaverage*",
    "cvs_avg35*",
    "V1_average",
    "?h.EC_average",
)
"""Directory names of the template subjects that a FreeSurfer subjects directory can hold.
``fsaverage`` and ``fsaverage5`` were seen next to the subjects of ds000102-fmriprep; the
other names are from memory of a FreeSurfer installation and have not been checked."""

# errors of stat() that say nothing is there: no such entry, a parent that is a file, a cycle
# of links, a name longer than the file system stores
_ABSENT = (errno.ENOENT, errno.ENOTDIR, errno.ELOOP, errno.ENAMETOOLONG)


def _mode(path: Path) -> int | None:
    """``st_mode`` of ``path`` after following links, or None if nothing is there.

    Any other failure of ``stat()`` (a parent that may not be searched, an I/O error) is
    raised: whether the file exists is then unknown. ``Path.is_file()`` is not used because its
    answer to that depends on the Python version (it raises up to 3.13, and is False from 3.14).
    """
    try:
        return path.stat().st_mode
    except OSError as exc:
        if exc.errno in _ABSENT:
            return None
        raise
    except ValueError:  # a name the operating system does not take, such as one with a NUL
        return None


def is_file(path: Path | str) -> bool:
    """Whether ``path`` is a regular file or a link to one; a dangling link is not.

    Raises OSError if the path cannot be examined, so that an unreadable file is never taken
    for a missing one.
    """
    mode = _mode(Path(path))
    return mode is not None and stat.S_ISREG(mode)


def is_template_name(name: str) -> bool:
    """Whether a directory of this name is a template subject, not a person."""
    return any(fnmatchcase(name, pattern) for pattern in TEMPLATE_NAME_PATTERNS)


def is_subject_candidate(path: Path | str) -> bool:
    """A directory with subdirectories ``mri/`` and ``surf/`` that is not a template.

    This is what a source lists as a subject (docs/architecture.md). A candidate that fails
    :func:`is_freesurfer_subject_dir` is an incomplete subject, such as a recon-all run that
    crashed, and is inventoried as one. Links are followed; templates are recognized by the
    name of the directory as given and of what it links to.

    Raises OSError when the directory cannot be examined.
    """
    path = Path(path)
    mode = _mode(path)
    if mode is None or not stat.S_ISDIR(mode):
        return False
    if is_template_name(path.name) or is_template_name(path.resolve().name):
        return False
    for name in ("mri", "surf"):
        mode = _mode(path / name)
        if mode is None or not stat.S_ISDIR(mode):
            return False
    return True


def resolve_subject_files(subject_dir: Path | str) -> dict[FileRole, Path]:
    """The first existing candidate of each role; roles with none are absent.

    A symlink counts if its target exists, a dangling one does not. The returned path is the
    candidate itself, not the link target; ``Path.resolve()`` tells which variant a link names.
    Raises OSError if a candidate cannot be examined (see :func:`is_file`).
    """
    root = Path(subject_dir)
    files: dict[FileRole, Path] = {}
    for role, candidates in ROLE_CANDIDATES.items():
        for candidate in candidates:
            path = root / candidate
            if is_file(path):
                files[role] = path
                break
    return files


def variant_name(role: FileRole, path: Path | str) -> str:
    """The name FreeSurfer gives the file that ``path`` is read as, among the role's candidates.

    A link inside its own directory names its target when that is a candidate of the role
    (FreeSurfer 7: ``lh.pial -> lh.pial.T1`` reads as ``lh.pial.T1``). A link into another
    directory, or a file under a name of someone's choosing, reads under the candidate's own
    name, so that no foreign name is recorded. This is the rule by which ``render.json``
    names its inputs (``render/api.py``); a source records the same name as
    ``FileRef.variant``.
    """
    path = Path(path)
    names = [Path(candidate).name for candidate in ROLE_CANDIDATES[role]]
    target = path.resolve()
    if target.parent == path.parent.resolve() and target.name in names:
        return target.name
    return path.name if path.name in names else names[0]


def missing_roles(
    files: Mapping[FileRole, Path], required: Iterable[FileRole] = RENDER_ROLES
) -> list[FileRole]:
    """Required roles that have no file, in the order of ``required``."""
    return [role for role in required if role not in files]


def is_freesurfer_subject_dir(path: Path | str) -> bool:
    """A directory with ``mri/T1.mgz`` and a left pial surface that is not an fsaverage template.

    Templates are recognized by name, of the directory as given and of what it links to
    (``SUBJECTS_DIR/fsaverage`` is usually a link into the FreeSurfer installation).

    Raises OSError for a directory that cannot be looked into: it may or may not hold a
    subject. A scan over many directories has to catch that for each of them and record the
    directory as unreadable, so that it is neither dropped unseen nor stops the scan.
    """
    path = Path(path)
    mode = _mode(path)
    if mode is None or not stat.S_ISDIR(mode):
        return False
    if any(name.startswith("fsaverage") for name in (path.name, path.resolve().name)):
        return False
    return is_file(path / "mri" / "T1.mgz") and any(
        is_file(path / candidate) for candidate in ROLE_CANDIDATES[FileRole.LH_PIAL]
    )
