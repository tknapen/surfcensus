"""Tests of run.py: creating and opening a run, its sources, and the phases through ``Run``.

The phases themselves are tested where they live (``test_inventory.py``,
``test_render_batch.py``); here they are reached the way a user reaches them.
"""

from __future__ import annotations

import os
import pickle
import shutil
import sqlite3
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from synth import snapshot_tree  # pyrefly: ignore[missing-import]
from synth_tree import (  # pyrefly: ignore[missing-import]
    write_linked_tree,
    write_tier_a,
    write_tree,
)

import surfcensus as sc
from surfcensus import privacy, store
from surfcensus.config import (
    CONFIG_FILE,
    CacheConfig,
    ConfigError,
    SourceConfig,
    load_config,
    new_local_config,
    run_key,
    with_changes,
)
from surfcensus.privacy import HashKey
from surfcensus.records import RunContext
from surfcensus.render import RenderConfig, RenderError, api, encode, render_subject
from surfcensus.run import METRIC_STAGES, Run, from_config
from surfcensus.sources.base import TIER_C, FileRole, SourceError
from surfcensus.sources.local import LocalSource, local
from surfcensus.sources.patterns import resolve_subject_files
from surfcensus.store import CATALOG_FILE, Catalog, PublicCatalog

SMALL = RenderConfig(panel_px=32)
LABELS = ("sub-01", "sub-02", "sub-03")
FS_ROOT = "sourcedata/freesurfer"
DATASET_KEY = HashKey(bytes(range(1, 33)), "dataset-key")
SECOND = 1_000_000_000


def tree_bytes(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def touch(path: Path) -> None:
    """Another modification time, the same content."""
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + SECOND))


def uids(run: Run) -> dict[str, str]:
    """Label -> ``subject_uid`` of a local run."""
    subjects = run.public().subjects()
    return dict(zip(subjects["subject_label"], subjects["subject_uid"], strict=True))


def renders(run: Run) -> dict[str, dict[str, object]]:
    """The public ``renders`` frame by ``subject_uid``."""
    frame = run.public().renders()
    return {row["subject_uid"]: row for row in frame.to_dict("records")}


@pytest.fixture
def study(tmp_path: Path) -> Path:
    """An fMRIPrep output with three light subjects; ``sub-03`` lacks ``surf/lh.pial``, and
    ``sub-01`` has a build stamp and logs."""
    tree = write_tree(tmp_path / "study", "fmriprep", LABELS, incomplete=["sub-03"])
    write_tier_a(tree / FS_ROOT / "sub-01")
    return tree


@pytest.fixture
def new_run(tmp_path: Path) -> Iterator[Callable[..., Run]]:
    """``new_run(tree, **config)``: a local run on ``tree`` in ``tmp_path/run`` with small
    tiles, closed after the test."""
    made: list[Run] = []

    def make(tree: Path, *, run_dir: Path | None = None, **overrides: object) -> Run:
        overrides.setdefault("render", SMALL)
        run = Run(
            run_dir or tmp_path / "run", local(tree), config=new_local_config(tree, **overrides)
        )
        made.append(run)
        return run

    yield make
    for run in made:
        run.close()


@pytest.fixture
def rendered(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """The media directories that ``render_subject`` is called for, from now on."""
    calls: list[Path] = []
    original = api.render_subject

    def recording(files: object, out: Path, *args: object, **kwargs: object) -> object:
        calls.append(Path(out))
        return original(files, out, *args, **kwargs)

    monkeypatch.setattr(api, "render_subject", recording)
    return calls


class Reads(list[FileRole]):
    def __init__(self) -> None:
        super().__init__()
        self.heads: list[FileRole] = []


@pytest.fixture
def read(monkeypatch: pytest.MonkeyPatch) -> Reads:
    """The roles read whole through ``LocalSource.read_bytes``, from now on; the reads of a
    byte range, the heads by which the inventory tells git-annex pointer files, are in
    ``heads``."""
    roles = Reads()
    heads = roles.heads
    original = LocalSource.read_bytes

    def recording(self: LocalSource, subject: object, role: FileRole, **kwargs: object) -> bytes:
        if kwargs.get("byte_range") is None:
            roles.append(role)
        else:
            heads.append(role)
        return original(self, subject, role, **kwargs)

    monkeypatch.setattr(LocalSource, "read_bytes", recording)
    return roles


# ---------------------------------------------------------------------------------------
# creating and opening


def test_creating_a_run_writes_its_configuration_and_its_catalog(
    study: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "new" / "run"
    with sc.Run(run_dir, sc.sources.local(study)) as run:
        assert isinstance(run, Run) and run.run_dir == run_dir
        config = run.config
        assert config == load_config(run_dir)
        assert config.mode == "local" and config.render_video is False
        assert config.sources == (
            SourceConfig(kind="local", root=str(study.resolve()), identity=str(study.resolve())),
        )
    assert {p.name for p in run_dir.iterdir()} == {CONFIG_FILE, CATALOG_FILE}
    with Catalog(run_dir / CATALOG_FILE) as catalog:
        assert catalog.meta("mode") == "local"
        assert catalog.meta("key_fingerprint") == config.identity.key_fingerprint
        secret = catalog.run_secret()
    text = (run_dir / CONFIG_FILE).read_text()
    assert config.identity.local_salt is not None and config.identity.local_salt in text
    assert secret.hex() not in text


def test_opening_a_run_changes_nothing(study: Path, new_run: Callable[..., Run]) -> None:
    run = new_run(study)
    run.inventory()
    run.render()
    run.close()
    stored = (run.run_dir / CONFIG_FILE).read_bytes()
    media = tree_bytes(run.run_dir / "media")

    with Run(run.run_dir) as again:
        assert again.config == run.config
        assert (again.inventory().done, again.render().done) == (0, 0)
    with Run(run.run_dir, local(study)) as again:  # the same source, given again
        assert again.config == run.config
    assert (run.run_dir / CONFIG_FILE).read_bytes() == stored
    assert tree_bytes(run.run_dir / "media") == media


def test_a_run_needs_a_source_to_be_created_and_an_empty_place(study: Path, tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="holds no run; give a source") as raised:
        Run(tmp_path / "run")
    assert raised.value.reason == "missing"
    assert not (tmp_path / "run").exists()

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "notes.txt").write_text("mine\n")
    with pytest.raises(ConfigError, match="is not empty") as raised:
        Run(occupied, local(study))
    assert raised.value.reason == "invalid"
    assert [p.name for p in occupied.iterdir()] == ["notes.txt"]

    with pytest.raises(ConfigError, match="no run to relocate"):
        Run(tmp_path / "run", local(study), relocate=True)
    with pytest.raises(ConfigError, match="draws its own salt") as raised:
        Run(tmp_path / "run", local(study), config=new_local_config(study), key=DATASET_KEY)
    assert raised.value.reason == "key"
    assert not (tmp_path / "run").exists()

    (tmp_path / "empty").mkdir()
    Run(tmp_path / "empty", local(study)).close()  # an empty directory will do
    assert (tmp_path / "empty" / CONFIG_FILE).is_file()


@pytest.mark.parametrize("place", ["sourcedata/freesurfer/run", "sourcedata/freesurfer/sub-01/run"])
def test_a_run_directory_inside_the_subjects_directory_is_refused(study: Path, place: str) -> None:
    before = snapshot_tree(study)
    with pytest.raises(ConfigError, match="lies inside the subjects directory") as raised:
        Run(study / place, local(study))
    assert raised.value.reason == "invalid"
    assert snapshot_tree(study) == before
    # beside the subjects directory, inside the fMRIPrep output, is a place like any other
    Run(study / "derived" / "run", local(study)).close()


@pytest.mark.parametrize("mode", ["local", "census"])
def test_an_output_in_the_subjects_directory_is_refused(
    study: Path, tmp_path: Path, mode: str
) -> None:
    run_dir = tmp_path / "run"
    key = DATASET_KEY if mode == "census" else None
    Run(run_dir, local(study), key=key).close()
    subjects_dir = study / FS_ROOT
    link = tmp_path / "link"
    link.symlink_to(subjects_dir / "sub-02")
    with Run(run_dir) as run:  # a census run is opened without its key
        for out in (
            subjects_dir,
            subjects_dir / "sub-01",
            subjects_dir / "sub-01" / "surf" / "tables",
            subjects_dir / "new",
            link,
            link / "tables",
        ):
            with pytest.raises(ConfigError, match="lies inside the subjects directory") as raised:
                run.refuse_output_in_sources(out)
            assert raised.value.reason == "invalid"
        # beside the subjects directory, inside the fMRIPrep output, is a place like any other
        for out in (study / "derived", study, tmp_path / "out", run_dir / "export"):
            run.refuse_output_in_sources(out)

    # the data is not mounted: the root is all the run knows of where the subjects were
    study.rename(tmp_path / "moved")
    with Run(run_dir) as run:
        with pytest.raises(ConfigError, match="lies inside the subjects directory"):
            run.refuse_output_in_sources(study / "tables")
        run.refuse_output_in_sources(tmp_path / "out")


def test_a_run_directory_in_what_a_subject_links_to_is_refused(tmp_path: Path) -> None:
    """A subject directory that is a link leads to a directory that is read as much as the
    subjects directory is. The refusal comes before ``config.yaml``: nothing is left."""
    store = tmp_path / "store"
    root = write_linked_tree(tmp_path / "study", store, ["sub-01", "sub-02"], with_template=True)
    before = snapshot_tree(tmp_path)
    for place in (
        store / "sub-01" / "run",  # named at the target
        root / "sub-02" / "run",  # named through the link
        store / "fsaverage" / "run",  # the template next to the subjects is read too
    ):
        with pytest.raises(ConfigError, match="which is only read") as raised:
            Run(place, local(root))
        assert raised.value.reason == "invalid"
        with pytest.raises(ConfigError, match="which is only read"):
            Run(place, local(root / "sub-01"))  # one subject directory as the source
    assert snapshot_tree(tmp_path) == before
    # beside what the links lead to is a place like any other
    with Run(store / "run", local(root)) as run:
        assert run.inventory().done == 2
        for out in (
            store / "sub-01",
            store / "sub-02" / "surf" / "tables",
            root / "sub-01" / "tables",
            store / "fsaverage",
        ):
            with pytest.raises(ConfigError, match="which is only read"):
                run.refuse_output_in_sources(out)
        run.refuse_output_in_sources(store / "tables")
        run.refuse_output_in_sources(tmp_path / "out")
    # a run that was moved into such a directory is refused when its source is next built
    shutil.move(store / "run", store / "sub-01" / "run")
    with Run(store / "sub-01" / "run") as run:
        with pytest.raises(ConfigError, match=r"the run directory .* lies inside the subject"):
            run.inventory()


def test_an_output_in_an_fsaverage_directory_of_the_call_is_refused(
    study: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The template a run was scored with is kept by content, not by path. What a call can
    know is the directory it was given and that of a FreeSurfer install."""
    home = tmp_path / "freesurfer"
    given = tmp_path / "templates" / "fsaverage"
    with Run(tmp_path / "run", local(study)) as run:
        for out in (given, given / "tables", home / "subjects" / "fsaverage" / "x"):
            run.refuse_output_in_sources(out)
        monkeypatch.setenv("FREESURFER_HOME", str(home))
        with pytest.raises(ConfigError, match="lies inside the fsaverage directory"):
            run.refuse_output_in_sources(home / "subjects" / "fsaverage" / "x")
        run.refuse_output_in_sources(home / "subjects" / "x")
        metrics = run.config.metrics.model_dump() | {"fsaverage_dir": str(given)}
        run.config = with_changes(run.config, metrics=metrics)
        for out in (given, given / "tables"):
            with pytest.raises(ConfigError, match="lies inside the fsaverage directory"):
                run.refuse_output_in_sources(out)
        run.refuse_output_in_sources(given.parent)
    # a new run in the install's template is refused before it exists
    with pytest.raises(ConfigError, match="lies inside the fsaverage directory"):
        Run(home / "subjects" / "fsaverage" / "run", local(study))
    assert not home.exists()


def test_one_subject_directory_is_a_run_of_that_subject(tmp_path: Path) -> None:
    tree = write_tree(tmp_path / "subjects", "subjects_dir", LABELS)
    with pytest.raises(ConfigError, match="lies inside the subjects directory"):
        Run(tree / "run", local(tree / "sub-02"))  # the subject's parent is what is read
    with Run(tmp_path / "run", local(tree / "sub-02"), config=None) as run:
        assert run.config.sources[0].root == str(tree.resolve())
        assert run.config.sources[0].subjects == ("sub-02",)
        assert run.inventory().done == 1
    with Run(tmp_path / "run") as run:
        assert list(uids(run)) == ["sub-02"]
        assert run.inventory().skipped == 1


def test_a_config_given_on_opening_replaces_the_stored_one_of_the_same_run(
    study: Path, new_run: Callable[..., Run], tmp_path: Path
) -> None:
    run = new_run(study)
    run.close()
    changed = with_changes(run.config, render_video=True, cache=CacheConfig(batch_subjects=2))
    with Run(run.run_dir, config=changed) as again:
        assert again.config == changed
    assert load_config(run.run_dir) == changed

    with pytest.raises(ConfigError, match="is not of the run") as raised:
        Run(run.run_dir, config=new_local_config(study))  # another salt
    assert raised.value.reason == "invalid"
    assert load_config(run.run_dir) == changed

    with pytest.raises(ConfigError, match="keyed with its own salt") as raised:
        Run(run.run_dir, key=DATASET_KEY)
    assert raised.value.reason == "key"

    other = tmp_path / "other"
    Run(other, local(study)).close()
    shutil.copyfile(other / CATALOG_FILE, run.run_dir / CATALOG_FILE)
    with pytest.raises(ConfigError, match="are of different runs"):
        Run(run.run_dir)


def test_a_catalog_that_was_never_made_is_made_on_opening(
    study: Path, new_run: Callable[..., Run]
) -> None:
    run = new_run(study)
    run.close()
    (run.run_dir / CATALOG_FILE).unlink()  # as after a crash between the two files
    with Run(run.run_dir) as again:
        assert again.inventory().done == 3


def _tables_without_a_run(path: Path) -> None:
    conn = sqlite3.connect(path, isolation_level=None)
    store._migrate(conn)
    conn.close()


UNFINISHED = {"an empty file": Path.touch, "tables without a row": _tables_without_a_run}


@pytest.mark.parametrize("left", list(UNFINISHED))
def test_a_run_whose_creation_was_killed_in_its_catalog_opens(
    left: str, study: Path, new_run: Callable[..., Run]
) -> None:
    """A catalog in which no run was made yet: what a kill leaves on a file system without
    hard links, and what it left before the catalog was built under another name. Every
    command then ended with "it was not made by surfcensus", for good."""
    run = new_run(study)
    run.close()
    stored = (run.run_dir / CONFIG_FILE).read_bytes()
    path = run.run_dir / CATALOG_FILE
    path.unlink()
    UNFINISHED[left](path)
    with Run(run.run_dir) as again:
        with Catalog(path, readonly=True) as catalog:
            assert catalog.meta("key_fingerprint") == again.config.identity.key_fingerprint
            assert len(catalog.run_secret()) == 32
        assert again.inventory().done == 3
    assert (run.run_dir / CONFIG_FILE).read_bytes() == stored
    assert {p.name for p in run.run_dir.iterdir()} == {CONFIG_FILE, CATALOG_FILE}
    with Run(run.run_dir) as third:
        assert third.inventory().skipped == 3


@pytest.mark.parametrize("left", ["no file", *UNFINISHED])
def test_the_unfinished_catalog_of_a_census_run_is_made_with_the_key_only(
    left: str, tmp_path: Path
) -> None:
    run_dir = tmp_path / "census"
    Run(run_dir, key=DATASET_KEY).close()
    path = run_dir / CATALOG_FILE
    path.unlink()
    if left != "no file":
        UNFINISHED[left](path)
    before = path.read_bytes() if path.exists() else None
    with pytest.raises(ConfigError, match="making it needs the key") as raised:
        Run(run_dir)
    assert raised.value.reason == "key"
    with pytest.raises(ConfigError) as raised:
        Run(run_dir, key=HashKey(bytes(range(2, 34)), "dataset-key"))
    assert raised.value.reason == "key"
    assert (path.read_bytes() if path.exists() else None) == before
    with Run(run_dir, key=DATASET_KEY) as run:
        assert run.config.mode == "census"
    with Run(run_dir) as run, Catalog(path, readonly=True) as catalog:  # whole: no key needed
        assert catalog.meta("key_fingerprint") == DATASET_KEY.fingerprint
        assert catalog.query("PRAGMA journal_mode")[0][0] == "wal"


def test_a_database_of_something_else_beside_config_yaml_is_not_filled(
    study: Path, new_run: Callable[..., Run]
) -> None:
    run = new_run(study)
    run.close()
    path = run.run_dir / CATALOG_FILE
    path.unlink()
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("CREATE TABLE notes (text)")
    conn.execute("INSERT INTO notes VALUES ('mine')")
    conn.close()
    before = path.read_bytes()
    with pytest.raises(ConfigError, match="it was not made by surfcensus"):
        Run(run.run_dir)
    assert path.read_bytes() == before


KILLED_RUN = """
import os, signal, sys
from surfcensus import store
from surfcensus.run import Run
from surfcensus.sources.local import LocalSource


def killed(*args, **kwargs):
    os.kill(os.getpid(), signal.SIGKILL)


def no_hard_links(src, dst):
    raise PermissionError(1, "Operation not permitted")


if sys.argv[3] == "before the catalog has its name":
    store._publish = killed
else:  # while the run is made in the file itself, where there are no hard links
    link, fill = os.link, store._fill

    def linking(src, dst):
        if os.path.basename(dst) == store.CATALOG_FILE:
            no_hard_links(src, dst)
        link(src, dst)

    def fill_in_place(path, **kwargs):
        if path.name == store.CATALOG_FILE:
            store._now = killed
        fill(path, **kwargs)

    os.link, store._fill = linking, fill_in_place
Run(sys.argv[2], LocalSource(sys.argv[1]))
"""


@pytest.mark.skipif(sys.platform == "win32", reason="needs SIGKILL")
@pytest.mark.parametrize(
    "where", ["before the catalog has its name", "in the transaction that makes the run"]
)
def test_a_process_killed_while_it_creates_a_run_leaves_one_that_opens(
    where: str, study: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    child = subprocess.run(
        [sys.executable, "-c", KILLED_RUN, str(study), str(run_dir), where],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert child.returncode == -9, child.stderr
    left = {p.name for p in run_dir.iterdir() if not p.name.startswith(".")}
    if where == "before the catalog has its name":
        assert left == {CONFIG_FILE}
    else:
        assert left - {f"{CATALOG_FILE}-journal"} == {CONFIG_FILE, CATALOG_FILE}
        assert (run_dir / CATALOG_FILE).stat().st_size == 0
    stored = (run_dir / CONFIG_FILE).read_bytes()
    with Run(run_dir) as run:
        assert run.config.sources[0].root == str(study.resolve())
        assert run.inventory().done == 3
    assert (run_dir / CONFIG_FILE).read_bytes() == stored
    with Run(run_dir) as run:
        assert run.inventory().skipped == 3


def test_a_run_that_is_opened_while_it_is_created_is_one_run(
    study: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Another process opens the run when ``config.yaml`` is there and the catalog is not
    yet. It used to find the catalog half made and say "it was not made by surfcensus"."""
    run_dir = tmp_path / "run"
    path = run_dir / CATALOG_FILE
    publish = store._publish
    seen: list[tuple[str | None, bytes]] = []

    def opened_meanwhile(tmp: Path, target: Path, **kwargs: object) -> bool:
        if seen:  # the other process, publishing its own
            return publish(tmp, target, **kwargs)
        seen.append((None, b""))
        assert (run_dir / CONFIG_FILE).is_file() and not path.exists()
        with Run(run_dir) as other:  # no catalog yet: it makes the catalog, with the same salt
            assert other.public().subjects().empty
        with Catalog(path, readonly=True) as catalog:
            seen[0] = (catalog.meta("run_id"), catalog.run_secret())
        return publish(tmp, target, **kwargs)

    monkeypatch.setattr(store, "_publish", opened_meanwhile)
    with Run(run_dir, local(study)) as run:
        monkeypatch.undo()
        with Catalog(path, readonly=True) as catalog:
            assert seen == [(catalog.meta("run_id"), catalog.run_secret())]
        assert run.inventory().done == 3
    assert {p.name for p in run_dir.iterdir()} == {CONFIG_FILE, CATALOG_FILE}


def test_of_two_creations_at_once_the_second_leaves_the_run_of_the_first(
    study: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both processes found the directory empty, and each drew a salt. The catalog is made
    with the salt of the first; the second must not put its own into ``config.yaml``."""
    run_dir = tmp_path / "run"
    Run(run_dir, local(study)).close()  # the first to write
    stored = (run_dir / CONFIG_FILE).read_bytes()
    catalog = (run_dir / CATALOG_FILE).read_bytes()

    second = local(study)
    is_file = Path.is_file
    with monkeypatch.context() as early:  # what the second saw when it looked: nothing yet
        early.setattr(Path, "is_file", lambda path: path.name != CONFIG_FILE and is_file(path))
        early.setattr(os, "listdir", lambda path: [])
        with pytest.raises(ConfigError, match="already holds a run") as raised:
            Run(run_dir, second)
    assert raised.value.reason == "invalid"
    assert (run_dir / CONFIG_FILE).read_bytes() == stored
    assert (run_dir / CATALOG_FILE).read_bytes() == catalog
    assert {p.name for p in run_dir.iterdir()} == {CONFIG_FILE, CATALOG_FILE}
    with Run(run_dir) as run:
        assert run.inventory().done == 3


def test_a_place_where_no_run_can_be_made_is_a_configuration_error(
    study: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if sys.platform == "linux":  # no directory can be made in /proc
        with pytest.raises(ConfigError, match="no run can be made in /proc/") as raised:
            Run("/proc/surfcensus-nowhere/run", local(study))
        assert raised.value.reason == "invalid"
        assert isinstance(raised.value.__cause__, OSError)

    def full(*args: object, **kwargs: object) -> Catalog:
        raise sqlite3.OperationalError("database or disk is full")

    run_dir = tmp_path / "run"
    with monkeypatch.context() as patched:
        patched.setattr(Catalog, "create", full)
        with pytest.raises(ConfigError, match="database or disk is full") as raised:
            Run(run_dir, local(study))
    assert raised.value.reason == "invalid" and str(run_dir) in raised.value.detail
    # what is left is a run whose catalog was never made, which the next call makes
    assert [p.name for p in run_dir.iterdir()] == [CONFIG_FILE]
    with Run(run_dir) as run:
        assert run.inventory().done == 3


# ---------------------------------------------------------------------------------------
# inventory and render through the run


def test_a_run_is_inventoried_and_rendered(
    study: Path, new_run: Callable[..., Run], tmp_path: Path
) -> None:
    before = snapshot_tree(study)
    run = new_run(study)

    report = run.inventory()
    assert (report.phase, report.done, report.skipped, report.failed) == ("inventory", 3, 0, 0)
    uid = uids(run)
    assert set(uid) == set(LABELS) and len(set(uid.values())) == 3
    subjects = run.public().subjects().set_index("subject_label")
    assert subjects["complete"].to_dict() == {"sub-01": True, "sub-02": True, "sub-03": False}
    assert subjects.loc["sub-01", "fs_version"] == "6.0.1"
    assert subjects.loc["sub-02", "fs_version"] is None

    report = run.render()
    assert (report.phase, report.done, report.skipped, report.failed) == ("render", 2, 0, 1)
    assert report.failures == ((uid["sub-03"], "missing surface"),)
    rows = renders(run)
    assert {label: rows[uid[label]]["status"] for label in LABELS} == {
        "sub-01": "done",
        "sub-02": "done",
        "sub-03": "failed",
    }
    assert rows[uid["sub-03"]]["detail"] == "no file for lh_pial"
    assert not run.media_dir(uid["sub-03"]).exists()

    # the files are those of a direct render of the subject
    for label in ("sub-01", "sub-02"):
        direct = tmp_path / "direct" / label
        render_subject(resolve_subject_files(study / FS_ROOT / label), direct, SMALL, video=False)
        assert run.media_dir(uid[label]) == run.run_dir / "media" / uid[label]
        assert tree_bytes(run.media_dir(uid[label])) == tree_bytes(direct)
    assert snapshot_tree(study) == before


def test_a_second_inventory_reads_no_file_and_a_second_render_renders_nothing(
    study: Path, new_run: Callable[..., Run], read: Reads, rendered: list[Path]
) -> None:
    run = new_run(study)
    run.inventory()
    run.render()
    assert sorted(read) == sorted(
        [FileRole.BUILD_STAMP, FileRole.RECON_LOG, FileRole.RECON_LOG_LH, FileRole.RECON_LOG_RH]
    )
    assert len(rendered) == 2
    assert read.heads and all(role not in read for role in read.heads)
    del read[:], rendered[:], read.heads[:]

    first = (run.inventory(), run.render())
    assert (first[0].done, first[0].skipped) == (0, 3)
    assert (first[1].done, first[1].skipped, first[1].failed) == (0, 2, 1)
    assert read == [] and read.heads == [] and rendered == []
    # the same again, with the same failed count: nothing was retried and nothing forgotten
    second = run.render()
    assert (second.done, second.skipped, second.failed, second.failures) == (
        0,
        2,
        1,
        first[1].failures,
    )


def test_a_touched_file_is_inventoried_again_alone_and_keeps_its_render_key(
    study: Path, new_run: Callable[..., Run], rendered: list[Path]
) -> None:
    run = new_run(study)
    run.inventory()
    run.render()
    uid = uids(run)
    keys = {u: row["render_key"] for u, row in renders(run).items()}
    media = tree_bytes(run.run_dir / "media")
    del rendered[:]

    touch(study / FS_ROOT / "sub-02" / "mri" / "T1.mgz")
    report = run.inventory()
    assert (report.done, report.skipped) == (1, 2)
    report = run.render()
    assert (report.done, report.skipped) == (1, 1)
    assert rendered == [run.media_dir(uid["sub-02"])]
    assert {u: row["render_key"] for u, row in renders(run).items()} == keys
    assert tree_bytes(run.run_dir / "media") == media


def test_a_subject_that_left_the_source_is_marked_absent_and_keeps_its_render(
    study: Path, new_run: Callable[..., Run]
) -> None:
    run = new_run(study)
    run.inventory()
    run.render()
    uid = uids(run)

    shutil.rmtree(study / FS_ROOT / "sub-02")
    report = run.inventory()
    assert (report.done, report.skipped, report.failed) == (0, 2, 0)
    assert any("no longer listed" in note for note in report.notes)
    subjects = run.public().subjects().set_index("subject_label")
    assert subjects["present"].to_dict() == {"sub-01": True, "sub-02": False, "sub-03": True}
    report = run.render()
    assert (report.done, report.skipped, report.failed) == (0, 1, 1)
    assert renders(run)[uid["sub-02"]]["status"] == "done"
    assert (run.media_dir(uid["sub-02"]) / "render.json").is_file()


def test_data_found_in_the_layout_of_another_pipeline_is_not_counted_twice(
    tmp_path: Path, rendered: list[Path]
) -> None:
    """The pipeline is part of every ``subject_uid``, so subjects that are found in the
    layout of another pipeline are new subjects. The ones the run held before are no
    longer listed, and are not present either."""
    labels = ["sub-01", "sub-02"]
    tree = write_tree(tmp_path / "study", "subjects_dir", labels)
    run_dir = tmp_path / "run"
    with Run(run_dir, local(tree), config=new_local_config(tree, render=SMALL)) as run:
        run.inventory()
        assert run.render().done == 2
        before = uids(run)
    (tree / FS_ROOT).mkdir(parents=True)  # where fMRIPrep keeps its subjects
    for label in labels:
        shutil.move(tree / label, tree / FS_ROOT / label)

    with Run(run_dir) as run:
        report = run.inventory()
        assert (report.done, report.skipped, report.failed) == (2, 0, 0)
        (note,) = report.notes
        assert "now found as pipeline fmriprep, not freesurfer" in note
        assert "the 2 subject(s) it had as freesurfer are no longer listed" in note
        subjects = run.public().subjects()
        present = subjects[subjects["present"]]
        assert sorted(present["subject_label"]) == labels
        assert set(present["subject_uid"]).isdisjoint(before.values())
        assert set(subjects[~subjects["present"]]["subject_uid"]) == set(before.values())
        datasets = run.public().datasets()
        assert datasets["pseudonym"].nunique() == 1
        listed = dict(zip(datasets["pipeline"], datasets["n_subjects_listed"], strict=True))
        assert listed == {"fmriprep": 2, "freesurfer": 0}
        # the renders of the subjects that left stay; the ones listed now are pending
        assert {uid: row["status"] for uid, row in renders(run).items()} == dict.fromkeys(
            before.values(), "done"
        )
        again = run.inventory()
        assert (again.done, again.skipped, again.notes) == (0, 2, ())
        assert run.render().done == 2 and len(renders(run)) == 4

    # the data then moves: the run can still follow it, and the earlier row goes along
    moved = tmp_path / "moved"
    shutil.move(tree, moved)
    del rendered[:]
    with Run(run_dir, local(moved), relocate=True) as run:
        report = run.inventory()
        assert (report.done, report.skipped) == (0, 2)
        assert run.render().skipped == 2 and rendered == []
    with Catalog(run_dir / CATALOG_FILE, readonly=True) as catalog:
        rows = catalog.query("SELECT root, pipeline, fs_root FROM datasets ORDER BY pipeline")
    assert [tuple(row) for row in rows] == [
        (str(moved.resolve()), "fmriprep", FS_ROOT),
        (str(moved.resolve()), "freesurfer", "."),
    ]


def test_subjects_that_moved_into_another_layout_of_the_same_pipeline_are_found_there(
    tmp_path: Path, rendered: list[Path]
) -> None:
    """Same root, same pipeline, same files: only the place of the subjects under the root
    differs, which no hash covers. Skipped, they would keep the old place, and every later
    lease would find their files gone."""
    labels = ["sub-01", "sub-02"]
    tree = write_tree(tmp_path / "study", "bids_derivatives", labels)
    run_dir = tmp_path / "run"
    with Run(run_dir, local(tree), config=new_local_config(tree, render=SMALL)) as run:
        run.inventory()
        assert run.render().done == 2
        uid = uids(run)
    for label in labels:
        shutil.move(tree / "freesurfer" / label, tree / label)
    shutil.rmtree(tree / "freesurfer")
    del rendered[:]

    with Run(run_dir) as run:
        report = run.inventory()
        assert (report.done, report.skipped, report.notes) == (2, 0, ())
        assert uids(run) == uid and run.public().subjects()["present"].all()
        (layout,) = run.public().datasets()["layout"]
        assert layout == "subjects_dir"
        with run.context().lease(uid["sub-01"], [FileRole.T1]) as paths:
            assert paths[FileRole.T1] == tree.resolve() / "sub-01" / "mri" / "T1.mgz"
        assert run.render().skipped == 2 and rendered == []  # the files are the same
        report = run.render(force=True)  # and can be read where they are now
        assert (report.done, report.failed) == (2, 0)
        assert run.inventory().skipped == 2


def test_one_process_and_two_render_the_same_files(
    study: Path, new_run: Callable[..., Run], tmp_path: Path
) -> None:
    one = new_run(study)
    two = Run(tmp_path / "run2", local(study), config=one.config)  # the same salt: the same uids
    try:
        for run, jobs in ((one, 1), (two, 2)):
            run.inventory()
            assert run.render(jobs=jobs).done == 2
        assert tree_bytes(one.run_dir / "media") == tree_bytes(two.run_dir / "media")
        assert set(renders(one)) == set(renders(two))
    finally:
        two.close()


def test_a_failing_subject_is_recorded_scrubbed_and_tried_again_only_on_request(
    study: Path, new_run: Callable[..., Run], rendered: list[Path], tmp_path: Path
) -> None:
    (study / FS_ROOT / "sub-01" / "mri" / "T1.mgz").write_bytes(b"")
    run = new_run(study)
    run.inventory()
    uid = uids(run)

    report = run.render()
    assert (report.done, report.failed) == (1, 2)  # sub-02 is rendered all the same
    assert dict(report.failures) == {
        uid["sub-01"]: "render failure",
        uid["sub-03"]: "missing surface",
    }
    row = renders(run)[uid["sub-01"]]
    assert row["status"] == "failed" and "T1.mgz" in row["detail"]
    assert str(tmp_path) not in row["detail"] and "/" not in row["detail"]
    assert len(rendered) == 2
    del rendered[:]

    again = run.render()
    assert (again.done, again.skipped, again.failed) == (0, 1, 2)
    assert again.failures == report.failures and rendered == []
    retried = run.render(retry_failed=True)
    assert (retried.done, retried.skipped, retried.failed) == (0, 1, 2)
    assert rendered == [run.media_dir(uid["sub-01"])]


def test_without_an_encoder_a_video_stops_the_render_and_nothing_is_recorded(
    study: Path, new_run: Callable[..., Run], monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_ffmpeg() -> str:
        raise RuntimeError("no ffmpeg binary was found")

    monkeypatch.setattr(encode, "ffmpeg_exe", no_ffmpeg)
    run = new_run(study)
    run.inventory()
    with pytest.raises(ConfigError, match="the video cannot be encoded") as raised:
        run.render(video=True)
    assert raised.value.reason == "missing"
    assert renders(run) == {} and not (run.run_dir / "media").exists()
    assert run.render().done == 2  # the run itself asks for no video


def test_an_interrupted_render_keeps_its_batches_and_the_next_does_the_rest(
    tmp_path: Path, new_run: Callable[..., Run], monkeypatch: pytest.MonkeyPatch
) -> None:
    tree = write_tree(tmp_path / "study", "subjects_dir", LABELS)
    run = new_run(tree, cache=CacheConfig(batch_subjects=1))
    run.inventory()
    original = api.render_subject
    calls = 0

    def interrupted(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return original(*args, **kwargs)

    monkeypatch.setattr(api, "render_subject", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run.render()
    assert [row["status"] for row in renders(run).values()] == ["done"]

    report = run.render()
    assert (report.done, report.skipped, report.failed) == (2, 1, 0)
    assert len(renders(run)) == 3


def test_subjects_are_named_by_uid_or_in_a_local_run_by_label_and_datasets_by_pseudonym(
    study: Path, new_run: Callable[..., Run]
) -> None:
    run = new_run(study)
    run.inventory()
    uid = uids(run)
    (pseudonym,) = run.public().datasets()["pseudonym"]

    assert run.render(subjects=["sub-02"]).done == 1
    assert run.render(subjects=[uid["sub-01"], "sub-02"], datasets=[pseudonym]).done == 1
    assert set(renders(run)) == set(uid.values())  # sub-03 has its missing surface row
    with pytest.raises(ConfigError, match="the run has no subject sub-99"):
        run.render(subjects=["sub-99"])
    with pytest.raises(ConfigError, match="no dataset named D-00000000") as raised:
        run.render(datasets=["D-00000000"])
    assert raised.value.reason == "invalid"
    with pytest.raises(ConfigError, match="no dataset named ds000102"):
        run.inventory(datasets=["ds000102"])
    assert run.inventory(datasets=[pseudonym]).skipped == 3


def test_a_label_of_two_datasets_is_ambiguous(tmp_path: Path, rendered: list[Path]) -> None:
    first = write_tree(tmp_path / "first", "subjects_dir", ["sub-01"])
    second = write_tree(tmp_path / "second", "subjects_dir", ["sub-01", "sub-02"])
    config = new_local_config(first, render=SMALL)
    with Run(tmp_path / "run", [local(first), local(second)], config=config) as run:
        assert run.inventory().done == 3
        with pytest.raises(ConfigError, match="sub-01 names subjects of 2 datasets") as raised:
            run.render(subjects=["sub-01"])
        assert raised.value.reason == "invalid"
        assert "use a subject_uid" in raised.value.detail
        assert renders(run) == {} and rendered == []

        # a label that one dataset has, and a subject_uid, name one subject each
        subjects = run.public().subjects()
        one, other = subjects.loc[subjects["subject_label"] == "sub-01", "subject_uid"]
        assert run.render(subjects=["sub-02"]).done == 1
        assert run.render(subjects=[one]).done == 1
        assert other not in renders(run) and len(rendered) == 2


def test_metrics_of_light_subjects_give_every_subject_a_row_without_a_score(
    study: Path, new_run: Callable[..., Run], caplog: pytest.LogCaptureFixture
) -> None:
    """The light subjects have no thickness and no template, and the stats file of
    ``sub-01`` is a few lines that are no parcel table: the metrics run, record that file
    as unreadable and give every subject a row without a score. ``tests/test_pipeline.py``
    has the runs that compute something."""
    run = new_run(study)
    run.inventory()
    assert METRIC_STAGES == ("regional", "fsaverage", "reference", "score")
    with caplog.at_level("WARNING", logger="surfcensus"):
        report = run.metrics()
    assert [r.getMessage().splitlines()[0] for r in caplog.records] == [
        "No fsaverage template found. Looked in:"
    ]
    assert report.phase == "metrics"
    assert report.failures == ((uids(run)["sub-01"], "regional: unreadable"),)
    assert any("No fsaverage template" in note for note in report.notes)
    scores = run.scores()
    assert sorted(scores["subject_uid"]) == sorted(run.public().subject_uids())
    assert scores["composite"].isna().all() and set(scores["score_kind"]) == {"fallback_rank"}
    assert run.clusters().empty
    # the failed unit is not tried again, and is still counted
    again = run.metrics()
    assert (again.done, again.failed) == (0, 1)
    with pytest.raises(ConfigError, match="no metric stage nonsense"):
        run.metrics(stages=["nonsense"])


# ---------------------------------------------------------------------------------------
# the context of the phases


def test_the_context_is_the_public_side_of_the_run(
    study: Path, new_run: Callable[..., Run], tmp_path: Path
) -> None:
    run = new_run(study)
    run.inventory()
    uid = uids(run)
    ctx = run.context()

    assert isinstance(ctx, RunContext) and isinstance(ctx.public, PublicCatalog)
    assert isinstance(run.public(), PublicCatalog)
    assert (ctx.run_dir, ctx.config) == (run.run_dir, run.config)
    assert not any(isinstance(value, Run | Catalog) for value in vars(ctx).values())

    with ctx.lease(uid["sub-01"], TIER_C) as paths:
        assert set(paths) == set(TIER_C)
        assert paths[FileRole.T1] == study.resolve() / FS_ROOT / "sub-01" / "mri" / "T1.mgz"
    with ctx.lease(uid["sub-03"], [FileRole.LH_PIAL, FileRole.RH_PIAL]) as paths:
        assert set(paths) == {FileRole.RH_PIAL}
    with pytest.raises(KeyError):
        ctx.lease("0" * 16, TIER_C)

    with Catalog(run.run_dir / CATALOG_FILE, readonly=True) as catalog:
        secret = catalog.run_secret()
    digests = {"t1": "0" * 64}
    assert ctx.render_key("abc", digests) == privacy.render_key(secret, "abc", digests)

    message = f"Empty file: '{study.resolve()}/{FS_ROOT}/sub-01/mri/T1.mgz' of sub-01"
    assert ctx.scrub(message) == "Empty file: 'T1.mgz' of sub-01"
    assert ctx.scrub(f"cannot list {study.resolve()}") == "cannot list <root>"


def test_a_run_whose_data_is_gone_opens_and_stops_a_phase_before_anything_is_recorded(
    study: Path, new_run: Callable[..., Run], tmp_path: Path
) -> None:
    run = new_run(study)
    run.inventory()
    run.close()
    shutil.move(study, tmp_path / "elsewhere")

    with Run(run.run_dir) as again:
        assert len(again.public().subjects()) == 3  # what the catalog holds can be read
        for phase in (again.inventory, again.render):
            with pytest.raises(SourceError, match="is not a directory") as raised:
                phase()
            assert raised.value.reason == "not found"
        assert renders(again) == {}
        assert again.public().subjects()["present"].all()


def test_a_render_with_nothing_to_do_needs_no_source(
    study: Path, new_run: Callable[..., Run], tmp_path: Path, rendered: list[Path]
) -> None:
    run = new_run(study)
    run.inventory()
    assert run.render().done == 2
    run.close()
    shutil.move(study, tmp_path / "unmounted")
    del rendered[:]

    with Run(run.run_dir) as again:
        before = renders(again)
        report = again.render()
        assert (report.done, report.skipped) == (0, 2) and rendered == []
        assert dict(report.failures) == {uids(again)["sub-03"]: "missing surface"}
        # work that reads the source still stops before anything is recorded
        for asked in ({"force": True}, {"subjects": ["sub-01"], "video": True}):
            with pytest.raises(SourceError, match="is not a directory"):
                again.render(**asked)
        assert renders(again) == before


# ---------------------------------------------------------------------------------------
# another root, and data that moved


def test_another_root_is_refused_and_relocate_keeps_every_uid_and_render_key(
    study: Path, new_run: Callable[..., Run], tmp_path: Path, rendered: list[Path]
) -> None:
    run = new_run(study)
    run.inventory()
    run.render()
    uid, rows = uids(run), renders(run)
    run.close()
    stored = (run.run_dir / CONFIG_FILE).read_bytes()
    moved = tmp_path / "archive" / "study-2"
    moved.parent.mkdir()
    shutil.move(study, moved)
    del rendered[:]

    with pytest.raises(ConfigError) as raised:
        Run(run.run_dir, local(moved))
    assert raised.value.reason == "other root"
    assert raised.value.detail == (
        f"this run directory belongs to {study.resolve()}; pass --run-dir for a new run, or "
        "--relocate if the data moved"
    )
    assert (run.run_dir / CONFIG_FILE).read_bytes() == stored

    with Run(run.run_dir, local(moved), relocate=True) as again:
        (source,) = again.config.sources
        assert (source.root, source.identity) == (str(moved.resolve()), str(study.resolve()))
        assert uids(again) == uid and renders(again) == rows
        # a move keeps the modification times: nothing is read again, nothing rendered again
        report = again.inventory()
        assert (report.done, report.skipped) == (0, 3)
        assert again.render().skipped == 2 and rendered == []
        assert uids(again) == uid and renders(again) == rows
        with again.context().lease(uid["sub-01"], [FileRole.T1]) as paths:
            assert paths[FileRole.T1] == moved.resolve() / FS_ROOT / "sub-01" / "mri" / "T1.mgz"
    with Run(run.run_dir) as again:  # and the run now opens from where the data is
        assert again.inventory().skipped == 3

    # a copy that lost the modification times is read and rendered again, to the same keys
    copy = tmp_path / "copy"
    shutil.copytree(moved, copy, copy_function=shutil.copyfile)
    for path in copy.rglob("*"):
        if path.is_file():
            touch(path)
    with Run(run.run_dir, local(copy), relocate=True) as again:
        assert again.inventory().done == 3
        assert again.render().done == 2
        assert uids(again) == uid
        assert {u: row["render_key"] for u, row in renders(again).items()} == {
            u: row["render_key"] for u, row in rows.items()
        }


def test_relocating_to_a_layout_with_another_fs_root_keeps_renders_usable(
    study: Path, new_run: Callable[..., Run], tmp_path: Path, rendered: list[Path]
) -> None:
    run = new_run(study)
    run.inventory()
    assert run.render().done == 2
    uid, rows = uids(run), renders(run)
    run.close()
    # the same subjects, now in the older fMRIPrep layout: freesurfer/ beside fmriprep/
    moved = tmp_path / "moved"
    (moved / "fmriprep").mkdir(parents=True)
    shutil.move(study / FS_ROOT, moved / "freesurfer")
    shutil.move(study / "dataset_description.json", moved / "fmriprep")
    del rendered[:]

    with Run(run.run_dir, local(moved), relocate=True) as again:
        report = again.inventory()
        assert (report.done, report.skipped) == (0, 3) and uids(again) == uid
        assert again.render().skipped == 2 and rendered == []
        subjects_dir = moved.resolve() / "freesurfer"
        with again.context().lease(uid["sub-01"], [FileRole.T1]) as paths:
            assert paths[FileRole.T1] == subjects_dir / "sub-01" / "mri" / "T1.mgz"
        report = again.render(force=True)  # reads every file, from the new place
        assert (report.done, report.failed) == (2, 1)
        assert report.failures == ((uid["sub-03"], "missing surface"),)
        after = renders(again)
        for label in ("sub-01", "sub-02"):
            assert after[uid[label]]["status"] == "done"
            assert after[uid[label]]["render_key"] == rows[uid[label]]["render_key"]
    with Run(run.run_dir) as again:
        assert again.inventory().skipped == 3 and again.render().skipped == 2


def test_relocating_keeps_the_subject_set_and_refuses_data_of_another_pipeline(
    study: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    with Run(run_dir, local(study, subjects=["sub-0[12]"], field_strength=7.0)) as run:
        assert run.inventory().done == 2
        uid = uids(run)
    moved = tmp_path / "moved"
    shutil.move(study, moved)
    with Run(run_dir, local(moved), relocate=True) as run:  # the globs are not given again
        (source,) = run.config.sources
        assert (source.subjects, source.field_strength) == (("sub-0[12]",), 7.0)
        assert run.inventory().skipped == 2 and uids(run) == uid

    stored = (run_dir / CONFIG_FILE).read_bytes()
    plain = write_tree(tmp_path / "plain", "subjects_dir", LABELS)
    with pytest.raises(ConfigError, match="would get other identifiers") as raised:
        Run(run_dir, local(plain), relocate=True)
    assert raised.value.reason == "invalid"
    assert (run_dir / CONFIG_FILE).read_bytes() == stored
    with pytest.raises(ConfigError, match="give the source at its new place"):
        Run(run_dir, relocate=True)


def test_a_source_given_on_opening_replaces_the_stored_subject_set(
    study: Path, new_run: Callable[..., Run]
) -> None:
    run = new_run(study)
    run.inventory()
    uid = uids(run)
    run.close()

    with Run(run.run_dir, local(study, subjects=["sub-01"], field_strength=3.0)) as again:
        (source,) = again.config.sources
        assert (source.subjects, source.field_strength) == (("sub-01",), 3.0)
        again.inventory()
        subjects = again.public().subjects().set_index("subject_label")
        assert subjects["present"].to_dict() == {"sub-01": True, "sub-02": False, "sub-03": False}
        assert subjects.loc["sub-01", "field_strength"] == 3.0
        assert uids(again) == uid
    assert load_config(run.run_dir).sources[0].subjects == ("sub-01",)


def test_with_two_local_sources_relocate_names_the_dataset_that_moved(tmp_path: Path) -> None:
    first = write_tree(tmp_path / "first", "subjects_dir", ["sub-01"])
    second = write_tree(tmp_path / "second", "subjects_dir", ["sub-01", "sub-02"])
    run_dir = tmp_path / "run"
    with Run(run_dir, [local(first), local(second)]) as run:
        assert run.inventory().done == 3
        datasets = run.public().datasets()
        by_count = dict(zip(datasets["n_subjects_listed"], datasets["pseudonym"], strict=True))
        subjects = run.public().subjects()
        before = set(subjects["subject_uid"])
        assert run.render(datasets=[by_count[2]], subjects=None).done == 2
    with pytest.raises(ConfigError, match="the same source was given twice"):
        Run(tmp_path / "twice", [local(first), local(first)])

    moved = tmp_path / "second-moved"
    shutil.move(second, moved)
    with pytest.raises(ConfigError, match="has 2 local sources") as raised:
        Run(run_dir, local(moved), relocate=True)
    assert raised.value.reason == "invalid" and by_count[1] in raised.value.detail
    with pytest.raises(ConfigError, match="no local dataset D-00000000"):
        Run(run_dir, local(moved), relocate="D-00000000")
    with Run(run_dir, local(moved), relocate=by_count[2]) as run:
        assert [s.root for s in run.config.sources] == [str(first.resolve()), str(moved.resolve())]
        report = run.inventory()
        assert (report.done, report.skipped) == (0, 3)
        assert set(run.public().subjects()["subject_uid"]) == before
        report = run.render()  # the two that moved are found rendered, the third is new
        assert (report.done, report.skipped) == (1, 2)


def test_from_config_takes_root_and_identity_from_the_caller_when_given(study: Path) -> None:
    config = new_local_config(study)
    stored = SourceConfig(kind="local", root="/data/that/moved", identity="/first/place")
    source = from_config(stored, config, Path("unused"), root=str(study))
    assert source.config() == SourceConfig(
        kind="local", root=str(study.resolve()), identity="/first/place"
    )
    by_pseudonym = SourceConfig(kind="local", dataset="D-0badcafe", subjects=("sub-01",))
    source = from_config(by_pseudonym, config, Path("unused"), root=str(study), identity="x")
    assert source.config() == SourceConfig(
        kind="local", root=str(study.resolve()), identity="x", subjects=("sub-01",)
    )
    with pytest.raises(ConfigError, match="needs a root"):
        from_config(by_pseudonym, config, Path("unused"))


# ---------------------------------------------------------------------------------------
# a census run


def test_a_census_run_is_created_with_the_key_and_keeps_paths_and_labels_to_its_catalog(
    study: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "census"
    with pytest.raises(ConfigError, match="needs the dataset key") as raised:
        Run(run_dir, local(study), key=privacy.new_local_salt())
    assert raised.value.reason == "key" and not run_dir.exists()

    pseudonym = privacy.dataset_pseudonym(DATASET_KEY, "local", str(study.resolve()))
    with Run(run_dir, local(study), key=DATASET_KEY) as run:
        assert run.config.mode == "census"
        assert run.config.sources == (SourceConfig(kind="local", dataset=pseudonym),)
        assert run.config.identity.key_fingerprint == DATASET_KEY.fingerprint
        assert run.inventory().done == 3
        subjects = run.public().subjects()
        assert subjects["subject_label"].isna().all() and subjects["participant"].isna().all()
        assert set(subjects["dataset"]) == {pseudonym}
        expected = privacy.subject_uid(
            DATASET_KEY, "local", str(study.resolve()), "fmriprep", "sub-01"
        )
        assert expected in set(subjects["subject_uid"])
    text = (run_dir / CONFIG_FILE).read_text()
    assert "study" not in text and str(tmp_path) not in text
    assert DATASET_KEY.secret.hex() not in text

    # every command but the inventory opens the run without the key
    with Run(run_dir) as run:
        report = run.render(subjects=[expected])
        assert (report.done, report.failed) == (1, 1)
        assert "detail" not in run.public().renders().columns
        with pytest.raises(ConfigError, match="needs the dataset key") as raised:
            run.inventory()
        assert raised.value.reason == "key"
    other = HashKey(bytes(range(2, 34)), "dataset-key")
    with Run(run_dir, key=other) as run, pytest.raises(ConfigError) as raised:
        run.inventory()
    assert raised.value.reason == "key"

    # data that moved: the root is replaced in the catalog, and config.yaml stays as it is
    moved = tmp_path / "moved"
    shutil.move(study, moved)
    with pytest.raises(ConfigError) as raised:
        Run(run_dir, local(moved), key=DATASET_KEY)
    assert raised.value.reason == "other root"
    assert pseudonym in raised.value.detail and str(tmp_path) not in raised.value.detail
    with Run(run_dir, local(moved), key=DATASET_KEY, relocate=True) as run:
        assert run.inventory().skipped == 3
        assert expected in set(run.public().subjects()["subject_uid"])
    assert (run_dir / CONFIG_FILE).read_text() == text
    with Run(run_dir, key=DATASET_KEY) as run:
        assert run.inventory().skipped == 3


def test_a_census_run_can_start_without_a_source(tmp_path: Path) -> None:
    with Run(tmp_path / "census", key=DATASET_KEY) as run:
        assert run.config.sources == () and run.config.app.raters_per_subject == 2
        report = run.inventory()
        assert (report.done, report.skipped, report.failed) == (0, 0, 0)


# ---------------------------------------------------------------------------------------
# errors that cross a process boundary


@pytest.mark.parametrize("error", [ConfigError, SourceError, RenderError])
def test_the_errors_of_the_package_survive_pickling(error: type[Exception]) -> None:
    # a worker process sends its exception to the parent pickled
    original = error("changed", "surf/lh.pial changed after it was listed")
    copy = pickle.loads(pickle.dumps(original))
    assert type(copy) is error and str(copy) == str(original)
    assert (copy.reason, copy.detail) == ("changed", "surf/lh.pial changed after it was listed")


def test_the_key_of_a_local_run_is_its_salt(study: Path, new_run: Callable[..., Run]) -> None:
    run = new_run(study)
    run.inventory()
    key = run_key(run.config)
    root = str(study.resolve())
    assert uids(run) == {
        label: privacy.subject_uid(key, "local", root, "fmriprep", label) for label in LABELS
    }
    assert list(run.public().datasets()["pseudonym"]) == [
        privacy.dataset_pseudonym(key, "local", root)
    ]
