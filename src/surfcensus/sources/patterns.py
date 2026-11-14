"""Path patterns that identify a FreeSurfer subject directory and the files inside it.

Read-only: nothing here opens, creates or changes a file.
"""

from __future__ import annotations

import errno
import os
import re
import stat
from collections.abc import Iterable, Mapping
from fnmatch import fnmatchcase
from pathlib import Path

from surfcensus.sources.base import RENDER_ROLES, UNFETCHED, FileRole, Unfetched


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


# --- git-annex placeholders ----------------------------------------------------------------------
#
# Read in git-annex 10.20261026 (doc/internals/key_format.mdwn, Annex/Link.hs) and checked
# against a clone made with DataLad 1.8.0 on 2026-11-14: a file under annex whose content is
# not there is, in a locked branch, a link into .git/annex/objects/<hash>/<hash>/<key>/<key>
# with no target, and, in an unlocked or adjusted branch, a pointer file whose first line is
# "/annex/objects/<key>". git-annex takes a file of at most 32768 bytes whose first line
# holds "/annex/objects" (isLinkToAnnex) for a pointer, and every further line must hold
# "/annex/" (validPointerLineTag).

ANNEX_OBJECTS = "/annex/objects/"
"""What the target of an annex link and the first line of a pointer file contain."""

POINTER_MAX_BYTES = 32768
"""The largest pointer file git-annex takes (``maxValidPointerSz``). A file of a role that is
larger is never read to see whether it is one."""

POINTER_PROBE_BYTES = 1024
"""How much of a file of at most :data:`POINTER_MAX_BYTES` is read to tell a pointer file:
its first line is the path of a key, which is far shorter."""

_KEY = re.compile(r"[A-Z0-9]+(?:-[a-zA-Z][0-9]+)*--[^/\n]+")
_KEY_SIZE = re.compile(r"-s([0-9]{1,20})(?=-)")


def annex_key_size(key: str) -> int | None:
    """The ``-s`` field of a git-annex key (``BACKEND[-sNNNN][-mNNNN][-SNNNN-CNNNN]--NAME``),
    the size of the content in bytes; None when the key has none."""
    head = key.split("--", 1)[0] + "-"
    found = _KEY_SIZE.search(head)
    return int(found.group(1)) if found else None


def _key_of(target: str) -> str | None:
    """The key that a link target or a pointer line names, or None when it names none."""
    if ANNEX_OBJECTS not in target.replace("\\", "/"):
        return None
    name = target.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return name if _KEY.fullmatch(name) else None


def annex_link(path: Path | str) -> tuple[Path, str] | None:
    """For a link at ``path`` that leads, through any further links, into a git-annex object
    store and whose target is not there: the last file of the chain outside the store (what
    ``datalad get`` is asked for) and the key. None for anything else: no link, a link whose
    target exists, a link elsewhere.

    Reads links only (``lstat``, ``readlink``) and opens no file. Raises OSError when a link
    cannot be read for another reason than its absence.
    """
    current = Path(path)
    for _ in range(40):  # the limit of links that the kernel follows
        try:
            if not stat.S_ISLNK(current.lstat().st_mode):
                return None
            target = os.readlink(current)
        except OSError as exc:
            if exc.errno in _ABSENT:
                return None
            raise
        key = _key_of(target)
        if key is not None:
            # into the store: a placeholder only while the content is not there
            return None if _mode(current) is not None else (current, key)
        current = Path(os.path.normpath(current.parent / target))
    return None


def annex_pointer(head: bytes, size: int) -> str | None:
    """The key of a pointer file, from its first bytes ``head`` (up to
    :data:`POINTER_PROBE_BYTES`) and its size; None for any other file. git-annex's own
    rule (``parseLinkTargetOrPointer'``): at most :data:`POINTER_MAX_BYTES`, a first line
    that names a key under ``/annex/objects``, and ``/annex/`` on every further line."""
    if size > POINTER_MAX_BYTES or not head.startswith(b"/"):
        return None
    text = head.decode("utf-8", errors="replace")
    first, _, rest = text.partition("\n")
    key = _key_of(first.removesuffix("\r"))
    if key is None:
        return None
    lines = rest.split("\n")
    if len(head) < size:
        lines = lines[:-1]  # the last line read may be cut short
    if any(line and "/annex/" not in line for line in lines):
        return None
    return key


def find_unfetched(
    subject_dir: Path | str, present: Mapping[FileRole, Path]
) -> dict[FileRole, Unfetched]:
    """The git-annex links without content among the candidates of the roles that
    ``present`` (:func:`resolve_subject_files`) lacks: for each such role the first
    candidate that is one. Opens no file. Raises OSError like :func:`is_file`."""
    root = Path(subject_dir)
    found: dict[FileRole, Unfetched] = {}
    for role, candidates in ROLE_CANDIDATES.items():
        if role in present:
            continue
        for candidate in candidates:
            link = annex_link(root / candidate)
            if link is None:
                continue
            last, key = link
            fetch = candidate
            try:
                inside = last.parent.resolve() == (root / candidate).parent.resolve()
            except (OSError, RuntimeError):
                inside = False
            names = {Path(name).name for name in candidates}
            if inside and last.name in names:
                fetch = (Path(candidate).parent / last.name).as_posix()
            found[role] = Unfetched(role, candidate, fetch, annex_key_size(key), "link")
            break
    return found


def size_words(size: int) -> str:
    """A size in bytes for a message: ``32.4 MB``, or ``12 kB`` below 0.1 MB."""
    return f"{size / 1e6:.1f} MB" if size >= 100_000 else f"{max(1, round(size / 1e3))} kB"


def unfetched_problem(unfetched: Iterable[Unfetched], *, sizes: bool = True) -> str:
    """The ``problem`` of a subject whose files ``unfetched`` have no content, one sentence:
    ``not fetched: 5 files are git-annex links whose content is not here: mri/T1.mgz,
    surf/lh.white, ...; 32.4 MB``. It names what a fetch is asked for, relative to the
    subject directory, and, with ``sizes``, their size from the keys (``at least`` when a key
    has none). :func:`unfetched_files` reads the names back."""
    items = sorted(unfetched, key=lambda item: list(FileRole).index(item.role))
    if not items:
        raise ValueError("no file without content")
    kinds = {item.kind for item in items}
    n = len(items)
    if kinds == {"link"}:
        what = "a git-annex link" if n == 1 else "git-annex links"
    elif kinds == {"pointer"}:
        what = "a git-annex pointer file" if n == 1 else "git-annex pointer files"
    else:
        what = "git-annex links and pointer files"
    names = ", ".join(dict.fromkeys(item.fetch for item in items))
    known = [item.size for item in items if item.size is not None]
    amount = ""
    if sizes and known:
        amount = f"; {'at least ' if len(known) < n else ''}{size_words(sum(known))}"
    verb = "file is" if n == 1 else "files are"
    return f"{UNFETCHED}: {n} {verb} {what} whose content is not here: {names}{amount}"


_UNFETCHED_NAMES = re.compile(r" is not here: ([^;]*?)(?:; (at least )?([0-9.]+) (MB|kB))?$")


def unfetched_files(problem: object) -> tuple[list[str], int | None] | None:
    """What :func:`unfetched_problem` wrote: the names relative to the subject directory and
    the size in bytes as written (rounded; None when it gives none). None for any other
    ``problem``, also for None."""
    if not isinstance(problem, str) or not problem.startswith(f"{UNFETCHED}: "):
        return None
    found = _UNFETCHED_NAMES.search(problem)
    if found is None:
        return None
    names = [name for name in found.group(1).split(", ") if name]
    size = None
    if found.group(3) is not None:
        size = round(float(found.group(3)) * (1e6 if found.group(4) == "MB" else 1e3))
    return names, size


def unfetched_said(problem: object, *, names: int = 5) -> str | None:
    """A short form of a ``not fetched`` problem for a line that names several subjects:
    ``13 files not fetched (git-annex): mri/T1.mgz, mri/brainmask.mgz, ... and 8 more; 15
    kB``, with at most ``names`` names. None for any other problem."""
    found = unfetched_files(problem)
    if found is None:
        return None
    files, size = found
    shown = ", ".join(files[:names])
    if len(files) > names:
        shown += f" and {len(files) - names} more"
    amount = (
        ""
        if size is None
        else f"; {'at least ' if 'at least' in str(problem) else ''}{size_words(size)}"
    )
    what = "1 file" if len(files) == 1 else f"{len(files)} files"
    return f"{what} not fetched (git-annex): {shown}{amount}"


def is_unfetched_problem(problem: object) -> bool:
    """Whether a subject's ``problem`` says that files have no content (:data:`UNFETCHED`),
    which is no failure, rather than that its directory could not be listed."""
    return isinstance(problem, str) and problem.startswith(f"{UNFETCHED}: ")


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
