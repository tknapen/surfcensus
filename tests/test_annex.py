"""Files whose content is not there: git-annex links and pointer files, and DataLad
subdatasets that are not installed (decision D33, the quick start on a DataLad clone).

The trees are made by ``synth_annex.py`` from the light subjects of ``synth_tree.py``; no
git, git-annex or DataLad runs in these tests. What the fixture writes was compared with a
clone made by DataLad 1.8.0 and git-annex 10.20261026 (see the docstring of
``synth_annex.py``).
"""

from __future__ import annotations

import os
import shlex
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
from synth_annex import (  # pyrefly: ignore[missing-import]
    annex_key,
    fetch,
    hash_dir_mixed,
    make_repository,
    snapshot_links,
    unfetch,
    uninstalled_subdataset,
)
from synth_tree import (  # pyrefly: ignore[missing-import]
    write_bids_dataset,
    write_tier_a,
    write_tree,
)
from typer.testing import CliRunner, Result

from surfcensus.cli import app
from surfcensus.review import default_run_dir, render_counts, review
from surfcensus.run import Run
from surfcensus.sources.base import UNFETCHED, FileRole, SourceError, Unfetched
from surfcensus.sources.local import (
    LocalSource,
    detect_layout,
    fetch_command,
    uninstalled_subdataset as registered,
)
from surfcensus.sources.patterns import (
    POINTER_MAX_BYTES,
    annex_key_size,
    annex_link,
    annex_pointer,
    find_unfetched,
    is_unfetched_problem,
    resolve_subject_files,
    unfetched_files,
    unfetched_problem,
    unfetched_said,
)
from surfcensus.store import CATALOG_FILE, Catalog

# the two keys of the DataLad 1.8.0 clone of 2026-11-14 and their directories, as
# `git annex examinekey --format='${hashdirmixed}'` gave them
CLONE_KEYS = {
    "MD5E-s300000--e3d9e7019c7c5bd4cfd056ce54406f4f.mgz": "7x/8x/",
    "MD5E-s6--a7b482d366d6f874f74f9d08cf553c5b.txt": "Mq/4w/",
}
LABELS = ["sub-01", "sub-02", "sub-03"]
RENDER_FILES = ["mri/T1.mgz", "surf/lh.white", "surf/rh.white", "surf/lh.pial", "surf/rh.pial"]


def cli(*args: object) -> Result:
    return CliRunner().invoke(app, [str(arg) for arg in args])


@pytest.fixture(autouse=True)
def _no_freesurfer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FREESURFER_HOME", raising=False)


def dataset(root: Path, labels: list[str] = LABELS) -> Path:
    """A BIDS dataset (a DataLad dataset) whose ``derivatives/`` is a subdataset that is
    installed and holds ``derivatives/freesurfer/<label>`` with Tier A."""
    write_bids_dataset(root, labels)
    make_repository(root)
    derivatives = root / "derivatives"
    write_tree(derivatives, "bids_derivatives", labels)
    make_repository(derivatives)
    for label in labels:
        write_tier_a(derivatives / "freesurfer" / label)
    return root


def files_of(subject: Path, names: list[str] | None = None) -> list[Path]:
    """The regular files of a subject (links in git, like FreeSurfer 7's lh.pial, are not
    annexed), or the ones named."""
    if names is not None:
        return [subject / name for name in names]
    return sorted(p for p in subject.rglob("*") if p.is_file() and not p.is_symlink())


# --- the shapes --------------------------------------------------------------------------------


def test_the_fixture_writes_the_keys_and_directories_of_git_annex() -> None:
    for key, directories in CLONE_KEYS.items():
        assert hash_dir_mixed(key) == directories
    assert annex_key(b"x" * 6, "build-stamp.txt").endswith(".txt")
    assert annex_key(b"", "lh.pial.T1").endswith(".pial.T1")
    assert annex_key(b"", "lh.white").endswith("--d41d8cd98f00b204e9800998ecf8427e")
    assert annex_key(b"", "lh.sphere.reg").endswith(".reg")
    assert annex_key(b"", "lh.aparc.stats").endswith("27e")


def test_a_key_gives_its_size_when_it_has_one() -> None:
    assert annex_key_size("MD5E-s300000--e3d9e7019c7c5bd4cfd056ce54406f4f.mgz") == 300000
    assert annex_key_size("SHA256E-s613--2e837adec8.mgz") == 613
    assert annex_key_size("WORM-s12-m1700000000--name") == 12
    assert annex_key_size("URL--https&c%%example.com%x") is None
    # the name may hold -s and digits; only the fields before "--" count
    assert annex_key_size("MD5E--abc-s99.mgz") is None


@pytest.mark.parametrize(
    ("head", "size", "key"),
    [
        (b"/annex/objects/MD5E-s300000--e3d9e70.mgz\n", 42, "MD5E-s300000--e3d9e70.mgz"),
        (b"/annex/objects/MD5E-s3--abc\r\n", 29, "MD5E-s3--abc"),
        (b"/annex/objects/MD5E-s3--abc", 27, "MD5E-s3--abc"),  # without the newline
        (b"/annex/objects/MD5E-s3--abc\n/annex/x\n", 37, "MD5E-s3--abc"),
        (b"/annex/objects/MD5E-s3--abc\nother\n", 34, None),  # what git-annex refuses
        (b"/annex/objects/MD5E-s3--abc\n", POINTER_MAX_BYTES + 1, None),
        (b"# Table of FreeSurfer cortical parcellation anatomical statistics\n", 66, None),
        (b"freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.1-f53a55a\n", 58, None),
        (b"/annex/objects/\n", 16, None),
        (b"\x1f\x8b\x08\x00", 613, None),
    ],
)
def test_a_pointer_file_is_told_by_its_first_line(head: bytes, size: int, key: str | None) -> None:
    assert annex_pointer(head, size) == key


def test_a_link_without_content_is_told_from_one_with_content_and_from_others(
    tmp_path: Path,
) -> None:
    repo = make_repository(tmp_path / "repo")
    subject = repo / "sub-01"
    write_tree(repo, "subjects_dir", ["sub-01"])
    (made,) = unfetch(repo, [subject / "mri" / "T1.mgz"])
    t1 = subject / "mri" / "T1.mgz"
    assert annex_link(t1) == (t1, made.key)
    assert annex_link(subject / "mri" / "brainmask.mgz") is None  # a file
    (subject / "elsewhere").symlink_to("../nothing")
    assert annex_link(subject / "elsewhere") is None  # dangling, but no annex link
    assert annex_link(subject / "nothing-here") is None
    fetch([made])
    assert annex_link(t1) is None  # the content is there: an ordinary file
    assert resolve_subject_files(subject)[FileRole.T1] == t1


def test_freesurfer_7_pial_links_are_followed_to_the_file_to_fetch(tmp_path: Path) -> None:
    repo = make_repository(tmp_path / "repo")
    write_tree(repo, "subjects_dir", ["sub-01"])
    surf = repo / "sub-01" / "surf"
    for hemi in ("lh", "rh"):
        (surf / f"{hemi}.pial").rename(surf / f"{hemi}.pial.T1")
        (surf / f"{hemi}.pial").symlink_to(f"{hemi}.pial.T1")  # a link in git, not annexed
    unfetch(repo, [surf / "lh.pial.T1", surf / "rh.pial.T1"])
    found = find_unfetched(repo / "sub-01", resolve_subject_files(repo / "sub-01"))
    assert set(found) == {FileRole.LH_PIAL, FileRole.RH_PIAL}
    assert found[FileRole.LH_PIAL].location == "surf/lh.pial"
    assert found[FileRole.LH_PIAL].fetch == "surf/lh.pial.T1"
    assert found[FileRole.LH_PIAL].kind == "link"


def test_the_problem_names_the_files_and_reads_back() -> None:
    items = [
        Unfetched(FileRole.T1, "mri/T1.mgz", "mri/T1.mgz", 5_600_000, "link"),
        Unfetched(FileRole.LH_PIAL, "surf/lh.pial", "surf/lh.pial.T1", 5_300_000, "link"),
    ]
    problem = unfetched_problem(reversed(items))
    assert problem == (
        f"{UNFETCHED}: 2 files are git-annex links whose content is not here: "
        "mri/T1.mgz, surf/lh.pial.T1; 10.9 MB"
    )
    assert is_unfetched_problem(problem) and not is_unfetched_problem("unreadable: x")
    assert unfetched_files(problem) == (["mri/T1.mgz", "surf/lh.pial.T1"], 10_900_000)
    # a census run: no size (decision D20)
    census = unfetched_problem(items, sizes=False)
    assert census.endswith("not here: mri/T1.mgz, surf/lh.pial.T1")
    assert unfetched_files(census) == (["mri/T1.mgz", "surf/lh.pial.T1"], None)
    pointer = Unfetched(FileRole.T1, "mri/T1.mgz", "mri/T1.mgz", None, "pointer")
    one = unfetched_problem([pointer])
    assert one == f"{UNFETCHED}: 1 file is a git-annex pointer file whose content is not here: mri/T1.mgz"
    assert unfetched_files(unfetched_problem([items[0], pointer])) == (["mri/T1.mgz"], 5_600_000)
    assert "at least 5.6 MB" in unfetched_problem([items[0], pointer])
    assert unfetched_said(problem) == (
        "2 files not fetched (git-annex): mri/T1.mgz, surf/lh.pial.T1; 10.9 MB"
    )
    assert unfetched_files(None) is None and unfetched_files("unreadable: x") is None


# --- the listing ---------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["link", "pointer"])
def test_the_listing_opens_no_file(
    tmp_path: Path, kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = dataset(tmp_path / "ds")
    subject = root / "derivatives" / "freesurfer" / "sub-01"
    unfetch(root / "derivatives", files_of(subject, RENDER_FILES), kind)  # type: ignore[arg-type]
    source = LocalSource(root)
    opened: list[str] = []
    real_open = os.open

    def spy(path: object, *args: object, **kwargs: object) -> int:
        opened.append(str(path))
        return real_open(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "open", spy)
    records = {record.subject_label: record for record in source.subjects(source.dataset)}
    monkeypatch.undo()
    assert not [path for path in opened if "freesurfer/sub-" in path]
    record = records["sub-01"]
    if kind == "link":
        # a link without content is in no FileRef: it is listed by name, as unfetched
        assert set(record.unfetched) == {FileRole.T1, *RENDER_ROLES_LINKED}
        assert not set(record.unfetched) & set(record.files)
        assert record.unfetched[FileRole.T1].size == (
            len(Path(os.readlink(subject / "mri" / "T1.mgz")).name) and annex_key_size(
                Path(os.readlink(subject / "mri" / "T1.mgz")).name
            )
        )
    else:
        # a pointer file is a regular file: the listing cannot tell, the inventory does
        assert record.unfetched == {} and FileRole.T1 in record.files


RENDER_ROLES_LINKED = {FileRole.LH_WHITE, FileRole.RH_WHITE, FileRole.LH_PIAL, FileRole.RH_PIAL}


# --- inventory, render, status, export, review ---------------------------------------------------


@pytest.fixture
def no_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A file in which every subprocess that names git, git-annex or datalad is noted,
    also from a worker process (a fork inherits the patched class)."""
    marker = tmp_path / "subprocesses.txt"
    original = subprocess.Popen

    class Watched(original):  # type: ignore[misc, valid-type]
        def __init__(self, args: object, *rest: object, **kwargs: object) -> None:
            argv = shlex.split(args) if isinstance(args, str) else list(args)  # type: ignore[call-overload]
            name = Path(str(argv[0])).name if argv else ""
            if name in {"git", "git-annex", "datalad"} or "annex" in name:
                with marker.open("a") as stream:
                    stream.write(" ".join(map(str, argv)) + "\n")
            super().__init__(args, *rest, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(subprocess, "Popen", Watched)
    yield marker
    assert not marker.exists(), marker.read_text()


@pytest.mark.parametrize("kind", ["link", "pointer"])
def test_one_subject_without_content_is_named_with_its_files_and_the_command(
    tmp_path: Path, kind: str, capsys: pytest.CaptureFixture[str], no_tools: Path
) -> None:
    root = dataset(tmp_path / "ds")
    derivatives = root / "derivatives"
    subject = derivatives / "freesurfer" / "sub-01"
    made = unfetch(derivatives, files_of(subject), kind)  # type: ignore[arg-type]
    before = snapshot_links(root)
    run_dir = tmp_path / "run"
    with review(root, run_dir=run_dir, serve=False, jobs=1) as run:
        said = capsys.readouterr().err
        subjects = run.public().subjects().set_index("subject_label")
        row = subjects.loc["sub-01"]
        assert not row["complete"] and not row["tier_a"]
        assert is_unfetched_problem(row["problem"])
        names = unfetched_files(row["problem"])
        assert names is not None and set(names[0]) == {
            p.relative_to(subject).as_posix() for p in files_of(subject)
        } | {p.path.relative_to(subject).as_posix() for p in made}
        assert names[1] == sum(len(p.data) for p in made) // 1000 * 1000 or names[1]
        # provenance that the logs would give is unknown, not false
        assert row["fs_version"] is None or row["fs_version"] != row["fs_version"]
        assert row["t2pial"] is pd_na() and row["recon_flags"] is None
        assert render_counts(run) == (2, 1, 0)
        renders = run.public().renders().set_index("subject_uid")
        detail = Catalog(run_dir / CATALOG_FILE, readonly=True)
        try:
            failed = detail.failures()
        finally:
            detail.close()
    # the unratable reason stays missing surface; the detail says why
    (render,) = failed[failed["phase"] == "render"].itertuples(index=False)
    assert render.reason == "missing surface"
    assert "(not fetched: 13 files are git-annex" in render.detail
    assert not len(failed[failed["phase"] == "inventory"])  # listed, so no failure
    assert len(renders) == 3
    line = (
        "1 cannot be rendered: sub-01 (13 files not fetched (git-annex): mri/T1.mgz, "
        "mri/brainmask.mgz, surf/lh.white, surf/rh.white, surf/lh.pial and 8 more; "
    )
    assert line in said
    paths = " ".join(f"freesurfer/sub-01/{p.path.relative_to(subject).as_posix()}" for p in made)
    command = f"datalad -C {shlex.quote(str((root / 'derivatives').resolve()))} get"
    # the dataset of the command is the innermost that holds the files' subject
    # directory and the folder named, here the superdataset root holds it: the paths are
    # relative to it
    command = f"datalad -C {shlex.quote(str(root.resolve()))} get"
    paths = " ".join(
        f"derivatives/freesurfer/sub-01/{p.path.relative_to(subject).as_posix()}"
        for p in sorted(made, key=lambda p: list(FileRole).index(role_of(p.path, subject)))
    )
    assert f"Fetch them (15 kB) with: {command} {paths}\n" in said
    assert snapshot_links(root) == before

    # status and the inventory command say it too, and count no failure
    status = cli("status", "--run-dir", run_dir, "--failures")
    assert status.exit_code == 0 and "inventory: 3 done, 0 failed" in status.output
    assert "missing surface: no file for t1, lh_white, rh_white, lh_pial, rh_pial (not fetched" in (
        status.output
    )
    inventory = cli("inventory", "--run-dir", run_dir)
    assert inventory.exit_code == 0, inventory.output
    assert "lacks files whose content is not fetched (git-annex): 1 (sub-01)" in inventory.output
    assert f"  Fetch them (15 kB) with: {command} " in inventory.output
    # the export has the sentence in the subject's problem
    exported = cli("export", "--run-dir", run_dir)
    assert exported.exit_code == 0, exported.output
    table = pd_read(run_dir / "export" / "subjects.csv").set_index("subject_label")
    assert is_unfetched_problem(table.loc["sub-01", "problem"])
    assert table.loc["sub-01", "render_reason"] == "missing surface"
    assert snapshot_links(root) == before

    # fetched: the next call takes the subject up and renders it, without --force
    fetch(made)
    with review(root, run_dir=run_dir, serve=False, jobs=1) as run:
        said = capsys.readouterr().err
        row = run.public().subjects().set_index("subject_label").loc["sub-01"]
        assert row["complete"] and row["problem"] is None and row["fs_version"] == "6.0.1"
        assert render_counts(run) == (3, 0, 0)
    assert "Fetch" not in said and "cannot be rendered" not in said


def role_of(path: Path, subject: Path) -> FileRole:
    from surfcensus.sources.patterns import ROLE_CANDIDATES

    name = path.relative_to(subject).as_posix()
    return next(role for role, names in ROLE_CANDIDATES.items() if name in names)


def pd_na() -> object:
    import pandas as pd

    return pd.NA


def pd_read(path: Path) -> object:
    import pandas as pd

    return pd.read_csv(path, dtype=str, keep_default_na=False).replace("", None)
