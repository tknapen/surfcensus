"""Files whose content is not there: the placeholders of a git-annex repository and of a
DataLad dataset, made from the synthetic trees of ``synth_tree.py``.

Nothing here runs git, git-annex or DataLad. The shapes are those of the documentation and
of a real clone, checked on 2026-11-14 with git-annex 10.20261026 and DataLad 1.8.0
(installed from PyPI into a scratch environment, never into the repository's):

* A file under annex in a locked (the default) branch is a symbolic link into
  ``.git/annex/objects/<two hash directories>/<key>/<key>`` of its repository, relative
  to the link's directory: ``mri/T1.mgz ->
  ../../../.git/annex/objects/7x/8x/MD5E-s300000--e3d9e7019c7c5bd4cfd056ce54406f4f.mgz/
  MD5E-s300000--e3d9e7019c7c5bd4cfd056ce54406f4f.mgz`` in a subdataset ``derivatives``
  of a fresh ``datalad clone``. Until the content is fetched the target does not exist.
* A key is ``BACKEND[-sNNNN][-mNNNN][-SNNNN-CNNNN]--NAME`` (git-annex,
  ``doc/internals/key_format.mdwn``); DataLad's default backend is ``MD5E``, whose name
  is the MD5 of the content followed by the file's extension.
* In an unlocked or adjusted branch (``git annex adjust --unlock``, and on file systems
  without symbolic links) a file whose content is not there is a pointer file: one line
  ``/annex/objects/<key>`` and a newline (66 bytes for that ``T1.mgz``).
* A subdataset that is not installed is an empty directory, registered in the
  ``.gitmodules`` of the dataset that holds it.

What is from the clone and not from this module's own reasoning: the two hash directories
(``hashdirmixed``, git-annex ``Annex/DirHashes.hs`` and ``Utility/MD5.hs``), which
:func:`hash_dir_mixed` computes and which agree with ``git annex examinekey`` for the two
keys of the example. The keys of :func:`annex_key` equal, name and all, those of a
DataLad 1.8.0 clone of the light subjects of ``synth_tree.py`` for every file of a subject
(``T1.mgz``, ``brainmask.mgz``, the surfaces, ``lh.pial.T1``, the stats files, the logs and
the build stamp), and so do the link targets that :func:`unfetch` writes.
"""

from __future__ import annotations

import hashlib
import os
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Kind = Literal["link", "pointer"]

_CHARS = "0123456789zqjxkmvwgpfZQJXKMVWGPF"


def hash_dir_mixed(key: str) -> str:
    """The two directories under ``.git/annex/objects`` of a key, with a trailing slash."""
    word = struct.unpack("<I", hashlib.md5(key.encode()).digest()[:4])[0]
    chars = [_CHARS[(word >> (6 * x)) & 31] for x in range(8)]
    swapped = "".join(chars[i + 1] + chars[i] for i in range(0, 8, 2))
    return f"{swapped[:2]}/{swapped[2:4]}/"


def annex_key(data: bytes, name: str) -> str:
    """The ``MD5E`` key of ``data`` stored under the file name ``name``: up to two
    extensions from the end, each of at most four letters or digits, stopping at the first
    that is not (``lh.pial.T1`` keeps ``.pial.T1``, ``lh.sphere.reg`` ``.reg``,
    ``lh.aparc.stats`` none)."""
    parts = name.split(".")[1:]
    kept: list[str] = []
    for part in reversed(parts[-2:] if len(parts) > 1 else parts):
        if not (0 < len(part) <= 4 and part.isalnum()):
            break
        kept.insert(0, part)
    extension = "".join(f".{part}" for part in kept)
    return f"MD5E-s{len(data)}--{hashlib.md5(data).hexdigest()}{extension}"


def make_repository(root: Path, *, datalad: bool = True) -> Path:
    """Mark ``root`` as the work tree of a git-annex repository (``.git/annex/objects``),
    and with ``datalad`` as a DataLad dataset (``.datalad/config``). No git command runs:
    the directories are what a reader of the tree sees."""
    root = Path(root)
    (root / ".git" / "annex" / "objects").mkdir(parents=True, exist_ok=True)
    (root / ".git" / "HEAD").write_text("ref: refs/heads/master\n")
    if datalad:
        (root / ".datalad").mkdir(exist_ok=True)
        (root / ".datalad" / "config").write_text('[datalad "dataset"]\n\tid = synthetic\n')
    return root


@dataclass(frozen=True)
class Placeholder:
    path: Path
    """The file in the work tree."""
    key: str
    data: bytes
    """The content that a fetch puts back."""
    kind: Kind
    object_path: Path
    """Where the content goes under ``.git/annex/objects``."""


def unfetch(repository: Path, files: Iterable[Path], kind: Kind = "link") -> list[Placeholder]:
    """Replace each of ``files`` (regular files in the work tree of ``repository``) by what
    a clone holds before ``datalad get``: a link to an object that is not there, or a
    pointer file. Returns what :func:`fetch` needs to put the content back."""
    repository = Path(repository)
    made = []
    for path in files:
        path = Path(path)
        data = path.read_bytes()
        key = annex_key(data, path.name)
        relative = Path(".git", "annex", "objects", hash_dir_mixed(key), key, key)
        path.unlink()
        if kind == "link":
            up = os.path.relpath(repository, path.parent)
            path.symlink_to(Path(up) / relative)
        else:
            path.write_bytes(f"/annex/objects/{key}\n".encode())
        made.append(Placeholder(path, key, data, kind, repository / relative))
    return made


def fetch(placeholders: Iterable[Placeholder]) -> None:
    """What ``datalad get`` does to them: the content comes into the object store and the
    link resolves, or the pointer file is replaced by the content."""
    for placeholder in placeholders:
        if placeholder.kind == "link":
            placeholder.object_path.parent.mkdir(parents=True, exist_ok=True)
            placeholder.object_path.write_bytes(placeholder.data)
        else:
            placeholder.path.write_bytes(placeholder.data)


def uninstalled_subdataset(superdataset: Path, relative: str) -> Path:
    """An empty directory at ``superdataset/relative`` that the superdataset's
    ``.gitmodules`` registers, as a fresh ``datalad clone`` leaves a subdataset."""
    superdataset = Path(superdataset)
    folder = superdataset / relative
    folder.mkdir(parents=True, exist_ok=True)
    with (superdataset / ".gitmodules").open("a") as stream:
        stream.write(
            f'[submodule "{relative}"]\n\tpath = {relative}\n\turl = ./{relative}\n'
            "\tdatalad-id = synthetic\n"
        )
    return folder


def snapshot_links(root: Path) -> dict[str, tuple[str, int, int, str | None]]:
    """Every entry under ``root``, links not followed: kind, size, mtime in ns, and the
    target of a link."""
    found = {}
    for path in sorted(Path(root).rglob("*")):
        info = path.lstat()
        link = os.readlink(path) if path.is_symlink() else None
        kind = "link" if link is not None else "dir" if path.is_dir() else "file"
        found[path.relative_to(root).as_posix()] = (kind, info.st_size, info.st_mtime_ns, link)
    return found
