"""Tests of render/batch.py: rendering the subjects of a run and recording the outcomes.

The run is put together by hand: a ``Catalog`` filled from a ``LocalSource`` on a synthetic
tree, and a ``RunContext`` over its public side. ``Run`` and the inventory phase are not
used, so these tests need neither.
"""

from __future__ import annotations

import dataclasses
import errno
import hashlib
import math
import os
import shutil
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from concurrent.futures.process import BrokenProcessPool
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from synth import snapshot_tree  # pyrefly: ignore[missing-import]
from synth_tree import write_tree  # pyrefly: ignore[missing-import]

from surfcensus import privacy, store
from surfcensus.batches import WorkerError
from surfcensus.config import (
    CacheConfig,
    ConfigError,
    IdentityConfig,
    IntensityConfig,
    RunConfig,
    SelectConfig,
    SourceConfig,
    new_census_config,
    new_local_config,
    run_key,
)
from surfcensus.inventory.provenance import subject_provenance
from surfcensus.privacy import HashKey
from surfcensus.records import PhaseReport, RenderResult, RunContext
from surfcensus.render import api, batch, encode
from surfcensus.render.batch import (
    render_check_sample,
    render_failures,
    render_status,
    run_render,
)
from surfcensus.render.model import RenderConfig, RenderInfo
from surfcensus.schemas import KEYS, SCHEMAS, intensity_schema
from surfcensus.sources.base import (
    TIER_C,
    FileRole,
    SourceError,
    SubjectRecord,
    input_signature,
)
from surfcensus.sources.local import LocalSource
from surfcensus.sources.patterns import resolve_subject_files
from surfcensus.store import CATALOG_FILE, Catalog

# 32 px tiles; the default box on a light subject (10 mm voxels) gives 7, 6 and 7 slices
CONFIG = RenderConfig(panel_px=32)
DATASET_KEY = HashKey(bytes(range(1, 33)), "dataset-key")
SECOND = 1_000_000_000


@dataclass
class Bench:
    """A run directory with its catalog and sources, as ``Run`` will hold them."""

    run_dir: Path
    config: RunConfig
    key: HashKey
    catalog: Catalog
    sources: tuple[LocalSource, ...]
    leased: list[str] = field(default_factory=list)
    """Every ``subject_uid`` that was leased, in order."""

    def inventory(self) -> None:
        """List the sources again and store what they hold, as the inventory phase does."""
        for source in self.sources:
            dataset = source.dataset
            pseudonym = privacy.dataset_pseudonym(self.key, dataset.source, dataset.identity)
            records = list(source.subjects(dataset))
            dataset_id = self.catalog.upsert_dataset(dataset, pseudonym, len(records))
            for record in records:
                self.store(dataset_id, record)
            self.catalog.mark_absent(dataset_id, [record.subject_label for record in records])

    def store(self, dataset_id: int, record: SubjectRecord) -> str:
        dataset = record.dataset
        uid = privacy.subject_uid(
            self.key, dataset.source, dataset.identity, dataset.pipeline, record.subject_label
        )
        provenance = subject_provenance({}, record.files, record.subject_label)
        self.catalog.upsert_subject(
            uid, dataset_id, record, provenance, input_signature(record.files)
        )
        return uid

    def uid(self, label: str, source: int = 0) -> str:
        dataset = self.sources[source].dataset
        return privacy.subject_uid(
            self.key, dataset.source, dataset.identity, dataset.pipeline, label
        )

    def subject_dir(self, label: str, source: int = 0) -> Path:
        return self.sources[source].subjects_dir / label

    def pseudonym(self, source: int = 0) -> str:
        dataset = self.sources[source].dataset
        return privacy.dataset_pseudonym(self.key, dataset.source, dataset.identity)

    def lease(
        self, uid: str, roles: Collection[FileRole]
    ) -> AbstractContextManager[Mapping[FileRole, Path]]:
        self.leased.append(uid)
        record = self.catalog.subject_record(uid)
        source = next(s for s in self.sources if s.dataset.root == record.dataset.root)
        return source.lease(record, roles)

    def ctx(self, **changes: object) -> RunContext:
        """The context of the run; ``changes`` are fields of its configuration."""
        secret = self.catalog.run_secret()
        pseudonyms, roots = self.catalog.scrub_terms()
        census = self.config.mode == "census"
        config = RunConfig.model_validate(self.config.model_dump() | changes)
        return RunContext(
            run_dir=self.run_dir,
            config=config,
            public=self.catalog.public(),
            lease=self.lease,
            render_key=lambda tile_hash, digests: privacy.render_key(secret, tile_hash, digests),
            scrub=lambda text: privacy.scrub(
                text, pseudonyms=pseudonyms if census else None, roots=roots
            ),
        )

    def rows(self) -> dict[str, dict[str, object]]:
        """The ``renders`` table as the private catalog holds it, by ``subject_uid``."""
        return {
            row["subject_uid"]: dict(row) for row in self.catalog.query("SELECT * FROM renders")
        }

    def media(self, uid: str) -> Path:
        return self.run_dir / "media" / uid

    def expected_key(self, uid: str, config: RenderConfig = CONFIG) -> str:
        """The render key from the content of the files, computed without the renderer."""
        record = self.catalog.subject_record(uid)
        folder = Path(record.dataset.root, record.fs_dir)
        digests = {
            role.value: hashlib.sha256((folder / ref.location).read_bytes()).hexdigest()
            for role, ref in record.files.items()
            if role in TIER_C
        }
        return privacy.render_key(self.catalog.run_secret(), config.tile_hash(), digests)


@pytest.fixture
def new_bench(tmp_path: Path) -> Iterator[Callable[..., Bench]]:
    """``new_bench(labels, incomplete=..., census=..., trees=..., **config)``: a run on a
    tree of light subjects under ``tmp_path/study`` (or on ``trees``), inventoried."""
    made: list[Bench] = []

    def make(
        labels: Sequence[str] = ("sub-01", "sub-02"),
        *,
        incomplete: Collection[str] = (),
        census: bool = False,
        trees: Sequence[Path] | None = None,
        run_dir: Path | None = None,
        **overrides: object,
    ) -> Bench:
        if trees is None:
            trees = [write_tree(tmp_path / "study", "subjects_dir", labels, incomplete=incomplete)]
        sources = tuple(LocalSource(tree) for tree in trees)
        overrides.setdefault("render", CONFIG)
        if census:
            key = DATASET_KEY
            stored = tuple(
                SourceConfig(
                    kind="local",
                    dataset=privacy.dataset_pseudonym(key, "local", source.dataset.identity),
                )
                for source in sources
            )
            config = new_census_config(key, sources=stored, **overrides)
        else:
            overrides["sources"] = tuple(source.config() for source in sources)
            config = new_local_config(trees[0], **overrides)
            key = run_key(config)
        run_dir = run_dir or tmp_path / "run"
        catalog = Catalog.create(run_dir / CATALOG_FILE, mode=config.mode, key=key)
        bench = Bench(run_dir, config, key, catalog, sources)
        made.append(bench)
        bench.inventory()
        return bench

    yield make
    for bench in made:
        bench.catalog.close()


@pytest.fixture
def rendered(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, bool, bool]]:
    """Every call of ``render_subject`` in this process: (subject_uid, video, force)."""
    calls: list[tuple[str, bool, bool]] = []
    original = api.render_subject

    def spy(
        files: Mapping[FileRole, Path],
        out_dir: Path | str,
        config: RenderConfig | None = None,
        *,
        video: bool = True,
        force: bool = False,
    ) -> RenderInfo:
        calls.append((Path(out_dir).name, video, force))
        return original(files, out_dir, config, video=video, force=force)

    monkeypatch.setattr(api, "render_subject", spy)
    return calls


def uids_of(calls: Sequence[tuple[str, bool, bool]]) -> list[str]:
    return [uid for uid, _, _ in calls]


def tree_bytes(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def touch(path: Path) -> None:
    """Another modification time, the same content."""
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + SECOND))


def alter_t1(subject: Path) -> None:
    """Another T1 of the same size and geometry: one corner voxel differs."""
    path = subject / "mri" / "T1.mgz"
    image = nib.load(path)
    data = np.asanyarray(image.dataobj).copy()
    data[0, 0, 0] += 9
    nib.save(nib.MGHImage(data, image.affine), str(path))


def ras_t1(subject: Path) -> None:
    """Voxel axes along R, A, S instead of FreeSurfer's L, I, A: a wrong orientation."""
    path = subject / "mri" / "T1.mgz"
    affine = np.diag([10.0, 10.0, 10.0, 1.0])
    affine[:3, 3] = -80.0
    nib.save(nib.MGHImage(np.asanyarray(nib.load(path).dataobj), affine), str(path))


def select(bench: Bench, uids: Sequence[str], reason: str = "all") -> None:
    rows = pd.DataFrame(
        {
            "subject_uid": list(uids),
            "reason": reason,
            "stratum": "all",
            "inclusion_prob": 1.0,
            "seed": 0,
        }
    )
    bench.catalog.public().replace_selection(rows, [reason])


def failed(report: PhaseReport) -> dict[str, str]:
    assert report.failed == len(report.failures)
    assert list(report.failures) == sorted(report.failures)
    return dict(report.failures)


# ---------------------------------------------------------------- what a render records


def test_a_run_is_rendered_and_every_outcome_is_recorded(
    new_bench: Callable[..., Bench], tmp_path: Path
) -> None:
    bench = new_bench(["sub-01", "sub-02", "sub-03"], incomplete=["sub-03"])
    alter_t1(bench.subject_dir("sub-02"))
    bench.inventory()
    source = snapshot_tree(tmp_path / "study")
    one, two, three = (bench.uid(label) for label in ("sub-01", "sub-02", "sub-03"))

    report = run_render(bench.ctx())

    assert (report.phase, report.done, report.skipped) == ("render", 2, 0)
    assert failed(report) == {three: "missing surface"}
    assert report.notes == () and report.seconds > 0
    assert sorted(bench.leased) == sorted([one, two])
    assert sorted(p.name for p in (bench.run_dir / "media").iterdir()) == sorted([one, two])
    assert snapshot_tree(tmp_path / "study") == source

    rows = bench.rows()
    assert set(rows) == {one, two, three}
    for label, uid in (("sub-01", one), ("sub-02", two)):
        # the same bytes as a render of the subject directory without a run
        direct = tmp_path / "direct" / label
        files = resolve_subject_files(bench.subject_dir(label))
        info = api.render_subject(files, direct, CONFIG, video=False)
        assert tree_bytes(bench.media(uid)) == tree_bytes(direct)
        assert batch.media_dir(bench.run_dir, uid) == bench.media(uid)
        assert not (bench.media(uid) / "anim.mp4").exists()  # a new local run has no video

        row = rows[uid]
        assert (row["status"], row["reason"], row["detail"]) == ("done", None, None)
        assert row["config_hash"] == CONFIG.config_hash() == info.config_hash
        assert row["tile_hash"] == CONFIG.tile_hash()
        assert row["render_key"] == bench.expected_key(uid)
        assert row["video"] == 0 and row["template_id"] is None
        assert row["media_path"] == f"media/{uid}"
        assert row["n_frames"] == len(info.modes["sync"].frames) > 1
        # the digests of render.json are in the catalog, for the roles the render read
        stored = bench.catalog.query(
            "SELECT role, sha256 FROM files WHERE subject_uid = ? AND sha256 IS NOT NULL", (uid,)
        )
        assert {r["role"]: r["sha256"] for r in stored} == {
            role: entry.sha256 for role, entry in info.inputs.items()
        }
        assert set(info.inputs) == {role.value for role in TIER_C}
    assert rows[one]["render_key"] != rows[two]["render_key"]  # the content differs


def test_the_report_and_the_public_frames_carry_no_signature(
    new_bench: Callable[..., Bench],
) -> None:
    bench = new_bench(["sub-01", "sub-02"], incomplete=["sub-02"])
    report = run_render(bench.ctx())
    signatures = {row["input_signature"] for row in bench.rows().values()}
    assert len(signatures) == 2 and all(len(s) == 16 for s in signatures)
    shown = repr(report) + bench.catalog.public().renders().to_csv()
    assert not any(signature in shown for signature in signatures)


def test_incomplete_subjects_get_a_missing_surface_row_and_are_not_leased(
    new_bench: Callable[..., Bench], tmp_path: Path
) -> None:
    bench = new_bench(["sub-01", "sub-02", "sub-03"], incomplete=["sub-02", "sub-03"])
    (bench.subject_dir("sub-03") / "surf" / "rh.white").unlink()
    bench.inventory()
    # a directory that could not be listed: no files, and the reason the source gave
    unreadable = bench.store(
        1,
        SubjectRecord(
            bench.sources[0].dataset,
            "sub-04",
            "sub-04",
            {},
            f"unreadable: Permission denied: '{bench.subject_dir('sub-04')}'",
        ),
    )
    one, two, three = (bench.uid(label) for label in ("sub-01", "sub-02", "sub-03"))
    select(bench, [one])  # the incomplete ones are not selected, and are recorded all the same
    source = snapshot_tree(tmp_path / "study")

    report = run_render(bench.ctx())

    assert report.done == 1
    assert failed(report) == dict.fromkeys([two, three, unreadable], "missing surface")
    assert bench.leased == [one]
    assert snapshot_tree(tmp_path / "study") == source
    rows = bench.rows()
    assert rows[two]["detail"] == "no file for lh_pial"
    assert rows[three]["detail"] == "no file for rh_white, lh_pial"
    assert rows[unreadable]["detail"] == (
        "no file for t1, lh_white, rh_white, lh_pial, rh_pial "
        "(unreadable: Permission denied: 'sub-04')"
    )
    for uid in (two, three, unreadable):
        row = rows[uid]
        assert (row["status"], row["reason"]) == ("failed", "missing surface")
        assert row["config_hash"] == CONFIG.config_hash()
        assert (row["tile_hash"], row["render_key"], row["media_path"], row["n_frames"]) == (
            None,
            None,
            None,
            None,
        )
        assert not bench.media(uid).exists()
    # naming subjects does not change it either, also when an incomplete one is named
    for row_uid in (two, three):
        with bench.catalog.write() as conn:
            conn.execute("DELETE FROM renders WHERE subject_uid = ?", (row_uid,))
    again = run_render(bench.ctx(), subjects=[two, one])
    assert (again.done, again.skipped) == (0, 1)
    assert failed(again) == failed(report) and bench.leased == [one]
    assert bench.rows()[two]["detail"] == "no file for lh_pial"
    assert bench.rows()[three]["reason"] == "missing surface"


def test_a_subject_that_gets_its_surface_is_rendered(new_bench: Callable[..., Bench]) -> None:
    bench = new_bench(["sub-01", "sub-02"], incomplete=["sub-02"])
    two = bench.uid("sub-02")
    assert failed(run_render(bench.ctx())) == {two: "missing surface"}
    shutil.copy(
        bench.subject_dir("sub-01") / "surf" / "lh.pial", bench.subject_dir("sub-02") / "surf"
    )
    bench.inventory()
    report = run_render(bench.ctx())
    assert (report.done, report.skipped, report.failed) == (1, 1, 0)
    assert bench.rows()[two]["status"] == "done" and bench.rows()[two]["reason"] is None


def test_the_failures_of_the_status_are_the_ones_it_counts(
    new_bench: Callable[..., Bench],
) -> None:
    bench = new_bench(["sub-01", "sub-02", "sub-03"], incomplete=["sub-02"])
    (bench.subject_dir("sub-03") / "mri" / "T1.mgz").write_bytes(b"")
    bench.inventory()
    unreadable = bench.store(
        1,
        SubjectRecord(
            bench.sources[0].dataset,
            "sub-04",
            "sub-04",
            {},
            f"unreadable: Permission denied: '{bench.subject_dir('sub-04')}'",
        ),
    )
    two, three = bench.uid("sub-02"), bench.uid("sub-03")
    ctx = bench.ctx()

    # before a render: the incomplete subjects, with the detail that their rows will have
    assert render_status(ctx) == (0, 2, 2)
    before = render_failures(ctx)
    assert before == {
        two: ("missing surface", "no file for lh_pial"),
        unreadable: (
            "missing surface",
            "no file for t1, lh_white, rh_white, lh_pial, rh_pial "
            "(unreadable: Permission denied: 'sub-04')",
        ),
    }
    assert bench.rows() == {} and bench.leased == []

    run_render(ctx)
    rows = bench.rows()
    assert {uid: rows[uid]["detail"] for uid in before} == {
        uid: detail for uid, (_, detail) in before.items()
    }
    assert render_status(ctx) == (1, 3, 0)
    assert render_failures(ctx) == {
        two: ("missing surface", None),
        three: ("render failure", None),
        unreadable: ("missing surface", None),
    }

    # under another configuration the failed render is pending, and the rows of the
    # incomplete subjects are still to be written
    other = bench.ctx(render=CONFIG.model_copy(update={"crosshair_alpha": 0.2}))
    assert render_status(other) == (0, 2, 2)
    assert render_failures(other) == before
    assert bench.rows() == rows


# ---------------------------------------------------------------- the skip rule


def spy_on_writes(ctx: RunContext, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The names of the writing calls on the context's catalog, as they happen."""
    calls: list[str] = []
    for name in ("record_renders", "set_file_digests"):
        original = getattr(ctx.public, name)

        def spy(*args: object, _name: str = name, _original: Callable = original) -> object:
            calls.append(_name)
            return _original(*args)

        monkeypatch.setattr(ctx.public, name, spy)
    return calls


def test_a_second_run_does_nothing(
    new_bench: Callable[..., Bench],
    rendered: list[tuple[str, bool, bool]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bench = new_bench(["sub-01", "sub-02", "sub-03"], incomplete=["sub-03"])
    first = run_render(bench.ctx())
    assert len(rendered) == 2
    media, rows, source = snapshot_tree(bench.run_dir / "media"), bench.rows(), tmp_path / "study"
    before = snapshot_tree(source)
    del rendered[:], bench.leased[:]

    ctx = bench.ctx()
    writes = spy_on_writes(ctx, monkeypatch)
    second = run_render(ctx)

    assert rendered == [] and bench.leased == [] and writes == []
    assert (second.done, second.skipped, second.failed) == (0, 2, 1)
    assert second.failures == first.failures
    assert snapshot_tree(bench.run_dir / "media") == media and bench.rows() == rows
    assert snapshot_tree(source) == before
    assert not (bench.run_dir / "logs").exists()


def test_a_missing_render_json_renders_that_subject_again(
    new_bench: Callable[..., Bench], rendered: list[tuple[str, bool, bool]]
) -> None:
    bench = new_bench(["sub-01", "sub-02"])
    run_render(bench.ctx())
    one = bench.uid("sub-01")
    (bench.media(one) / "render.json").unlink()
    del rendered[:]
    report = run_render(bench.ctx())
    assert uids_of(rendered) == [one] and (report.done, report.skipped) == (1, 1)
    assert (bench.media(one) / "render.json").is_file()


def test_a_row_without_which_the_files_are_there_is_recorded_again(
    new_bench: Callable[..., Bench], rendered: list[tuple[str, bool, bool]]
) -> None:
    """Files are written before their row. A run that died in between leaves a complete
    render without a row; the next run finds the files and records them."""
    bench = new_bench(["sub-01", "sub-02"])
    run_render(bench.ctx())
    one = bench.uid("sub-01")
    row, media = bench.rows()[one], snapshot_tree(bench.media(one))
    with bench.catalog.write() as conn:
        conn.execute("DELETE FROM renders WHERE subject_uid = ?", (one,))
    del rendered[:]
    report = run_render(bench.ctx())
    assert uids_of(rendered) == [one] and (report.done, report.skipped) == (1, 1)
    assert snapshot_tree(bench.media(one)) == media  # nothing was written a second time
    assert {k: v for k, v in bench.rows()[one].items() if k != "rendered_at"} == {
        k: v for k, v in row.items() if k != "rendered_at"
    }


def test_a_changed_config_renders_again(
    new_bench: Callable[..., Bench], rendered: list[tuple[str, bool, bool]]
) -> None:
    bench = new_bench(["sub-01", "sub-02"])
    run_render(bench.ctx())
    one = bench.uid("sub-01")
    before = bench.rows()[one]
    del rendered[:]

    # a field that changes the poster only: another config, the same tiles
    poster = CONFIG.model_copy(update={"crosshair_alpha": 0.2})
    report = run_render(bench.ctx(render=poster))
    assert len(rendered) == 2 and (report.done, report.skipped) == (2, 0)
    row = bench.rows()[one]
    assert row["config_hash"] == poster.config_hash() != before["config_hash"]
    assert (row["tile_hash"], row["render_key"]) == (before["tile_hash"], before["render_key"])
    assert RenderInfo.model_validate_json(
        (bench.media(one) / "render.json").read_bytes()
    ).config == (poster)

    # a field that changes the tiles: the render key says so
    lines = CONFIG.model_copy(update={"line_width_px": 2.0})
    run_render(bench.ctx(render=lines))
    row = bench.rows()[one]
    assert row["tile_hash"] == lines.tile_hash() != before["tile_hash"]
    assert row["render_key"] == bench.expected_key(one, lines) != before["render_key"]
    # the run as it was configured at first is no longer up to date
    assert run_render(bench.ctx()).done == 2


def test_force_renders_over_what_is_there(
    new_bench: Callable[..., Bench], rendered: list[tuple[str, bool, bool]]
) -> None:
    bench = new_bench(["sub-01", "sub-02"])
    run_render(bench.ctx())
    media = snapshot_tree(bench.run_dir / "media")
    content = tree_bytes(bench.run_dir / "media")
    del rendered[:]
    report = run_render(bench.ctx(), force=True)
    assert (report.done, report.skipped) == (2, 0)
    assert sorted(rendered) == sorted(
        (bench.uid(label), False, True) for label in ("sub-01", "sub-02")
    )
    after = snapshot_tree(bench.run_dir / "media")
    assert after.keys() == media.keys() and tree_bytes(bench.run_dir / "media") == content
    rewritten = [name for name in media if "." in name and after[name][1] != media[name][1]]
    assert len(rewritten) == sum("." in name for name in media)  # every file was written again


def test_a_touched_input_is_rendered_again_and_keeps_its_render_key(
    new_bench: Callable[..., Bench], rendered: list[tuple[str, bool, bool]]
) -> None:
    bench = new_bench(["sub-01", "sub-02"])
    run_render(bench.ctx())
    one, two = bench.uid("sub-01"), bench.uid("sub-02")
    before, media = bench.rows(), snapshot_tree(bench.run_dir / "media")
    touch(bench.subject_dir("sub-01") / "surf" / "rh.pial")
    bench.inventory()
    del rendered[:], bench.leased[:]

    report = run_render(bench.ctx())

    assert uids_of(rendered) == [one] == bench.leased
    assert (report.done, report.skipped, report.failed) == (1, 1, 0)
    rows = bench.rows()
    assert rows[one]["input_signature"] != before[one]["input_signature"]
    assert rows[one]["render_key"] == before[one]["render_key"]
    assert rows[two] == before[two]
    assert snapshot_tree(bench.run_dir / "media") == media  # the files were found complete


def test_changed_content_changes_the_render_key(
    new_bench: Callable[..., Bench], rendered: list[tuple[str, bool, bool]]
) -> None:
    bench = new_bench(["sub-01", "sub-02"])
    run_render(bench.ctx())
    one = bench.uid("sub-01")
    before = bench.rows()[one]
    alter_t1(bench.subject_dir("sub-01"))
    bench.inventory()
    del rendered[:]
    report = run_render(bench.ctx())
    assert uids_of(rendered) == [one] and (report.done, report.skipped) == (1, 1)
    row = bench.rows()[one]
    assert row["render_key"] == bench.expected_key(one) != before["render_key"]
    assert row["tile_hash"] == before["tile_hash"]


def test_a_pial_link_that_leads_to_another_variant_of_equal_size_and_time_is_rendered_again(
    new_bench: Callable[..., Bench], rendered: list[tuple[str, bool, bool]]
) -> None:
    # FreeSurfer 7: lh.pial -> lh.pial.T1, or -> lh.pial.T2 after a T2 refinement. The two
    # have one size, and here one modification time, so of the listing only the variant
    # differs; the tiles would otherwise stay those of the other surface
    bench = new_bench(["sub-01", "sub-02"])
    surf = bench.subject_dir("sub-01") / "surf"
    first, second, link = surf / "lh.pial.T1", surf / "lh.pial.T2", surf / "lh.pial"
    os.replace(link, first)
    link.symlink_to(first.name)
    bench.inventory()
    run_render(bench.ctx())
    one = bench.uid("sub-01")
    before = bench.rows()[one]
    listed = bench.catalog.subject_record(one).files[FileRole.LH_PIAL]
    assert (listed.location, listed.variant) == ("surf/lh.pial", "lh.pial.T1")

    coords, faces, info, stamp = nib.freesurfer.read_geometry(
        str(first), read_metadata=True, read_stamp=True
    )
    nib.freesurfer.write_geometry(
        str(second), coords * 0.9, faces, create_stamp=stamp, volume_info=info
    )
    os.utime(second, ns=(first.stat().st_atime_ns, first.stat().st_mtime_ns))
    link.unlink()
    link.symlink_to(second.name)
    bench.inventory()
    again = bench.catalog.subject_record(one).files[FileRole.LH_PIAL]
    assert again == dataclasses.replace(listed, variant="lh.pial.T2")
    del rendered[:]

    report = run_render(bench.ctx())

    assert uids_of(rendered) == [one] and (report.done, report.skipped) == (1, 1)
    row = bench.rows()[one]
    assert row["input_signature"] != before["input_signature"]
    assert row["render_key"] == bench.expected_key(one) != before["render_key"]
    render_json = bench.run_dir / "media" / one / "render.json"
    read = RenderInfo.model_validate_json(render_json.read_text()).inputs["lh_pial"]
    assert (read.name, read.sha256) == (
        "lh.pial.T2",
        hashlib.sha256(second.read_bytes()).hexdigest(),
    )


def test_a_file_that_no_render_reads_does_not_render_again(
    new_bench: Callable[..., Bench], rendered: list[tuple[str, bool, bool]]
) -> None:
    bench = new_bench(["sub-01"])
    run_render(bench.ctx())
    one = bench.uid("sub-01")
    scripts = bench.subject_dir("sub-01") / "scripts"
    scripts.mkdir()
    (scripts / "recon-all.log").write_text("a log\n")
    bench.inventory()
    assert FileRole.RECON_LOG in bench.catalog.subject_record(one).files
    del rendered[:]
    report = run_render(bench.ctx())
    assert rendered == [] and (report.done, report.skipped) == (0, 1)


# ---------------------------------------------------------------- which subjects, in which order


def test_without_a_selection_every_complete_present_subject_is_rendered(
    new_bench: Callable[..., Bench], tmp_path: Path
) -> None:
    """Three datasets: subjects go by pseudonym, then ``subject_uid``, and not in the order
    in which the datasets were found."""
    salt = privacy.new_local_salt()
    trees = [
        write_tree(tmp_path / name, "subjects_dir", ["sub-01", "sub-02", "sub-03"])
        for name in ("first", "second", "third")
    ]
    # found in the reverse order of their pseudonyms
    trees.sort(
        key=lambda tree: privacy.dataset_pseudonym(
            salt, "local", LocalSource(tree).dataset.identity
        ),
        reverse=True,
    )
    identity = IdentityConfig(
        kind="local-salt", local_salt=salt.secret.hex(), key_fingerprint=salt.fingerprint
    )
    bench = new_bench(trees=trees, identity=identity, cache=CacheConfig(batch_subjects=2))
    public = bench.catalog.public()
    discovery = [bench.pseudonym(number) for number in range(3)]
    assert discovery == sorted(discovery, reverse=True) and len(set(discovery)) == 3
    expected = [
        uid
        for number in (2, 1, 0)
        for uid in sorted(bench.uid(f"sub-0{n}", number) for n in (1, 2, 3))
    ]

    report = run_render(bench.ctx())
    assert bench.leased == expected and report.done == 9

    # a dataset by its pseudonym
    del bench.leased[:]
    again = run_render(bench.ctx(), datasets=[bench.pseudonym(1)], force=True)
    assert again.done == 3 and bench.leased == expected[3:6]
    assert [public.dataset_of(uid) for uid in bench.leased] == [bench.pseudonym(1)] * 3
    assert run_render(bench.ctx(), datasets=[], force=True).done == 0
    assert run_render(bench.ctx(), datasets=["D-00000000"], force=True).done == 0


def test_a_local_run_renders_its_selection_unless_all_are_asked_for(
    new_bench: Callable[..., Bench],
) -> None:
    bench = new_bench(["sub-01", "sub-02", "sub-03", "sub-04"], incomplete=["sub-04"])
    assert bench.config.select.render_scope == "all"  # which a local run does not look at
    one, two, three, four = (bench.uid(f"sub-0{n}") for n in (1, 2, 3, 4))
    select(bench, [two, four])
    select(bench, [three], reason="calibration")

    report = run_render(bench.ctx())
    assert sorted(bench.leased) == sorted([two, three]) and report.done == 2
    assert failed(report) == {four: "missing surface"}
    assert not bench.media(one).exists()

    del bench.leased[:]
    everything = run_render(bench.ctx(), all_subjects=True)
    assert bench.leased == [one] and (everything.done, everything.skipped) == (1, 2)


def test_a_census_run_renders_every_complete_subject_unless_its_scope_is_the_selection(
    new_bench: Callable[..., Bench],
) -> None:
    bench = new_bench(["sub-01", "sub-02", "sub-03"], census=True)
    one, two, three = (bench.uid(f"sub-0{n}") for n in (1, 2, 3))
    select(bench, [two], reason="prevalence")

    selected = bench.ctx(select=SelectConfig(render_scope="selected"))
    assert run_render(selected).done == 1 and bench.leased == [two]
    assert run_render(selected, all_subjects=True).done == 2
    del bench.leased[:]
    assert run_render(bench.ctx(), force=True).done == 3  # render_scope "all" is the default
    assert bench.leased == sorted([one, two, three])
    del bench.leased[:]
    assert run_render(bench.ctx(), all_subjects=False, force=True).done == 1
    assert bench.leased == [two]
    assert bench.rows()[one]["video"] == 1  # a census run has videos unless told otherwise


def test_named_subjects_are_rendered_in_the_order_given(new_bench: Callable[..., Bench]) -> None:
    bench = new_bench(["sub-01", "sub-02", "sub-03", "sub-04"], cache=CacheConfig(batch_subjects=1))
    uids = [bench.uid(f"sub-0{n}") for n in (1, 2, 3, 4)]
    order = [uids[2], uids[0], uids[3]]
    select(bench, [uids[1]])  # named subjects go before the selection
    report = run_render(bench.ctx(), subjects=[*order, uids[2]])
    assert bench.leased == order and report.done == 3 and report.notes == ()
    assert set(bench.rows()) == set(order)
    # one subject may be named by itself
    del bench.leased[:]
    assert run_render(bench.ctx(), subjects=uids[1]).done == 1 and bench.leased == [uids[1]]
    assert run_render(bench.ctx(), subjects=[]).done == 0 and bench.leased == [uids[1]]


def test_a_subject_the_run_does_not_have_is_refused(new_bench: Callable[..., Bench]) -> None:
    bench = new_bench(["sub-01"])
    with pytest.raises(ConfigError, match="has no subject 0123456789abcdef") as raised:
        run_render(bench.ctx(), subjects=[bench.uid("sub-01"), "0123456789abcdef"])
    assert raised.value.reason == "invalid"
    assert bench.leased == [] and bench.rows() == {}


def test_a_subject_that_left_the_source_keeps_its_render_and_is_not_rendered(
    new_bench: Callable[..., Bench], rendered: list[tuple[str, bool, bool]]
) -> None:
    bench = new_bench(["sub-01", "sub-02", "sub-03"], incomplete=["sub-03"])
    run_render(bench.ctx())
    one, two, three = (bench.uid(f"sub-0{n}") for n in (1, 2, 3))
    shutil.rmtree(bench.subject_dir("sub-02"))
    shutil.rmtree(bench.subject_dir("sub-03"))
    bench.inventory()
    assert bench.catalog.public().subject_uids() == [one]
    rows = bench.rows()
    del rendered[:], bench.leased[:]

    # another config: the subjects in the source are rendered again, the others stay as they are
    other = bench.ctx(render=CONFIG.model_copy(update={"crosshair_alpha": 0.2}))
    report = run_render(other)
    assert uids_of(rendered) == [one] and (report.done, report.failed) == (1, 0)
    assert bench.rows()[two] == rows[two] and bench.rows()[three] == rows[three]
    assert (bench.media(two) / "render.json").is_file()

    named = run_render(other, subjects=[two, one])
    assert (named.done, named.skipped, named.failed) == (0, 1, 0) and bench.leased == [one]
    assert len(named.notes) == 1 and named.notes[0].startswith("1 of the subjects named")
    assert bench.rows()[two] == rows[two]


def score_rows(entries: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    """Rows of ``scores.parquet``: ``entries`` over a value for every column that needs one."""
    defaults: dict[str, object] = {}
    for column in SCHEMAS["scores"]:
        if column.nullable:
            defaults[column.name] = None
        elif pa.types.is_string(column.type):
            defaults[column.name] = "none"
        else:
            defaults[column.name] = pa.types.is_boolean(column.type) or 0
    return pd.DataFrame([defaults | dict(entry) for entry in entries])


def write_scores(bench: Bench, ranks: Mapping[str, tuple[int, str]]) -> None:
    """``uid -> (rank, flags)`` as primary rows, and a second row per subject with the
    ranks reversed, which the order must not follow."""
    entries: list[dict[str, object]] = []
    for uid, (rank, flags) in ranks.items():
        entries.append({"subject_uid": uid, "rank": rank, "flags": flags})
        entries.append(
            {"subject_uid": uid, "reference_id": "other", "primary": False, "rank": 100 - rank}
        )
    store.write_table(
        store.table_path(bench.run_dir, "scores"),
        score_rows(entries),
        SCHEMAS["scores"],
        sort_by=KEYS["scores"],
    )


def test_selected_subjects_of_a_local_run_with_scores_go_in_queue_order(
    new_bench: Callable[..., Bench],
) -> None:
    labels = [f"sub-0{n}" for n in range(1, 7)]
    bench = new_bench(labels, cache=CacheConfig(batch_subjects=1))
    a, b, c, d, e, f = sorted(bench.uid(label) for label in labels)
    select(bench, [a, b, c, d, e])
    plain = run_render(bench.ctx())
    assert bench.leased == [a, b, c, d, e] and plain.done == 5  # no scores: by subject_uid

    # e and c could not be fully scored; d looks worst, then a; b has no score
    write_scores(
        bench,
        {a: (2, "no_annot"), c: (3, "no_stats,partial_score"), d: (1, ""), e: (4, "partial_score")},
    )
    del bench.leased[:]
    run_render(bench.ctx(), force=True)
    assert bench.leased == [c, e, d, a, b]

    for changes in ({"app": bench.config.app.model_copy(update={"blind": True})}, {}):
        del bench.leased[:]
        run_render(bench.ctx(**changes), force=True, all_subjects=not changes)
        expected = [a, b, c, d, e] if changes else [a, b, c, d, e, f]
        assert bench.leased == expected  # a blind run, and a run of all subjects: by subject_uid


def test_a_census_run_with_scores_goes_by_pseudonym_and_subject(
    new_bench: Callable[..., Bench],
) -> None:
    labels = ["sub-01", "sub-02", "sub-03"]
    bench = new_bench(labels, census=True, cache=CacheConfig(batch_subjects=1), render_video=False)
    a, b, c = sorted(bench.uid(label) for label in labels)
    select(bench, [a, b, c], reason="prevalence")
    write_scores(bench, {a: (3, ""), b: (2, ""), c: (1, "")})
    run_render(bench.ctx(select=SelectConfig(render_scope="selected")))
    assert bench.leased == [a, b, c]


# ---------------------------------------------------------------- the video


def test_the_video_follows_the_run_unless_it_is_asked_for(
    new_bench: Callable[..., Bench], rendered: list[tuple[str, bool, bool]]
) -> None:
    bench = new_bench(["sub-01"])
    one = bench.uid("sub-01")
    video = bench.media(one) / "anim.mp4"
    with_video = bench.ctx(render_video=True)

    assert run_render(with_video).done == 1
    assert video.is_file() and bench.rows()[one]["video"] == 1
    key = bench.rows()[one]["render_key"]
    assert run_render(with_video).skipped == 1 and len(rendered) == 1

    # asked for without: rendered again, and the video goes (decision D19)
    assert run_render(with_video, video=False).done == 1
    assert not video.exists() and bench.rows()[one]["video"] == 0
    assert bench.rows()[one]["render_key"] == key
    assert run_render(bench.ctx()).skipped == 1  # the run itself has no video
    assert run_render(bench.ctx(), video=True).done == 1 and video.is_file()
    assert uids_of(rendered) == [one] * 3 and [v for _, v, _ in rendered] == [True, False, True]


@pytest.mark.parametrize("problem", ["absent", "not executable"])
def test_without_an_encoder_the_phase_stops_and_records_nothing(
    new_bench: Callable[..., Bench],
    rendered: list[tuple[str, bool, bool]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    problem: str,
) -> None:
    """The encoder belongs to the environment. Its absence is no property of a subject, so
    no row is written, not even for the subject that lacks a surface."""
    bench = new_bench(["sub-01", "sub-02", "sub-03"], incomplete=["sub-03"])
    one = bench.uid("sub-01")
    exe = tmp_path / "ffmpeg-here"
    if problem == "not executable":
        exe.write_bytes(b"data")
    with monkeypatch.context() as broken:
        broken.setenv("IMAGEIO_FFMPEG_EXE", str(exe))
        with pytest.raises(ConfigError, match="the video cannot be encoded") as raised:
            run_render(bench.ctx(), video=True)
    assert raised.value.reason == "missing" and "ffmpeg" in raised.value.detail
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert rendered == [] and bench.leased == [] and bench.rows() == {}
    assert not (bench.run_dir / "media").exists()

    # what is up to date needs no encoder, and neither does a render without video
    assert run_render(bench.ctx(), subjects=[one], video=True).done == 1
    monkeypatch.setenv("IMAGEIO_FFMPEG_EXE", str(exe))
    done = run_render(bench.ctx(), subjects=[one], video=True)
    assert (done.done, done.skipped) == (0, 1)
    without = run_render(bench.ctx(), video=False)
    assert without.done == 2 and failed(without) == {bench.uid("sub-03"): "missing surface"}
    assert not (bench.media(one) / "anim.mp4").exists()


def test_a_video_of_an_odd_panel_size_is_refused_before_anything_is_recorded(
    new_bench: Callable[..., Bench],
) -> None:
    bench = new_bench(["sub-01", "sub-02"], incomplete=["sub-02"])
    odd = bench.ctx(render=RenderConfig(panel_px=65))
    with pytest.raises(ConfigError, match=r"even render\.panel_px, the run has 65") as raised:
        run_render(odd, video=True)
    assert raised.value.reason == "invalid"
    assert bench.leased == [] and bench.rows() == {} and not (bench.run_dir / "media").exists()
    assert run_render(odd).done == 1  # without a video the size is fine


# ---------------------------------------------------------------- failures


def test_failures_are_recorded_with_their_reason_and_the_batch_goes_on(
    new_bench: Callable[..., Bench],
    rendered: list[tuple[str, bool, bool]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    labels = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05"]
    bench = new_bench(labels)
    (bench.subject_dir("sub-02") / "mri" / "T1.mgz").write_bytes(b"")
    ras_t1(bench.subject_dir("sub-03"))
    bench.inventory()
    touch(bench.subject_dir("sub-04") / "surf" / "lh.white")  # after the inventory
    one, empty, turned, touched, last = (bench.uid(label) for label in labels)
    source = snapshot_tree(tmp_path / "study")

    report = run_render(bench.ctx())

    reasons = {empty: "render failure", turned: "wrong orientation", touched: "source changed"}
    assert failed(report) == reasons and (report.done, report.skipped) == (2, 0)
    assert sorted(bench.leased) == sorted([one, empty, turned, touched, last])
    assert sorted(uids_of(rendered)) == sorted([one, empty, turned, last])
    rows = bench.rows()
    assert rows[one]["status"] == rows[last]["status"] == "done"
    for uid, reason in reasons.items():
        row = rows[uid]
        assert (row["status"], row["reason"]) == ("failed", reason)
        assert row["config_hash"] == CONFIG.config_hash() and row["video"] == 0
        assert (row["tile_hash"], row["render_key"], row["media_path"]) == (None, None, None)
        assert not bench.media(uid).exists()
        assert str(tmp_path) not in row["detail"] and "/" not in row["detail"].replace("surf/", "")
    assert rows[empty]["detail"] == "unreadable input (T1.mgz: Empty file: 'T1.mgz')"
    assert "conformed orientation" in rows[turned]["detail"]
    assert rows[touched]["detail"] == (
        "surf/lh.white changed after it was listed: its modification time differs"
    )
    assert snapshot_tree(tmp_path / "study") == source

    # a second run tries nothing again and reports the same
    del rendered[:], bench.leased[:]
    ctx = bench.ctx()
    writes = spy_on_writes(ctx, monkeypatch)
    second = run_render(ctx)
    assert rendered == [] and bench.leased == [] and writes == []
    assert (second.done, second.skipped, second.failures) == (0, 2, report.failures)
    assert bench.rows() == rows

    # unless it is asked to
    third = run_render(bench.ctx(), retry_failed=True)
    assert sorted(bench.leased) == sorted([empty, turned, touched])
    assert sorted(uids_of(rendered)) == sorted([empty, turned])
    assert (third.done, third.skipped, third.failures) == (0, 2, report.failures)

    # a repaired subject is rendered without being asked for: its files are other files
    shutil.copy(bench.subject_dir("sub-01") / "mri" / "T1.mgz", bench.subject_dir("sub-02") / "mri")
    bench.inventory()  # which also lists the touched file as it now is
    del rendered[:]
    fourth = run_render(bench.ctx())
    assert sorted(uids_of(rendered)) == sorted([empty, touched])
    assert (fourth.done, fourth.skipped) == (2, 2) and failed(fourth) == {turned: reasons[turned]}
    assert bench.rows()[empty]["status"] == "done" and bench.rows()[empty]["detail"] is None


def test_a_failed_subject_is_tried_again_with_another_config(
    new_bench: Callable[..., Bench], rendered: list[tuple[str, bool, bool]]
) -> None:
    bench = new_bench(["sub-01"])
    ras_t1(bench.subject_dir("sub-01"))
    bench.inventory()
    one = bench.uid("sub-01")
    assert failed(run_render(bench.ctx())) == {one: "wrong orientation"}
    assert failed(run_render(bench.ctx())) == {one: "wrong orientation"} and len(rendered) == 1
    other = bench.ctx(render=CONFIG.model_copy(update={"crosshair_alpha": 0.2}))
    assert failed(run_render(other)) == {one: "wrong orientation"} and len(rendered) == 2
    assert bench.rows()[one]["config_hash"] == other.config.render.config_hash()
    assert failed(run_render(other, force=True)) == {one: "wrong orientation"}
    assert len(rendered) == 3


def test_any_other_exception_is_a_render_failure_with_its_type_and_a_scrubbed_text(
    new_bench: Callable[..., Bench], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bench = new_bench(["sub-01", "sub-02"])
    one, two = bench.uid("sub-01"), bench.uid("sub-02")
    original = api.render_subject

    def breaks(files: Mapping[FileRole, Path], out_dir: Path, *args: object, **kwargs: object):
        if Path(out_dir).name == two:
            raise RuntimeError(f"cannot  use\n'{files[FileRole.T1]}' of sub-02 in ds000102")
        return original(files, out_dir, *args, **kwargs)

    monkeypatch.setattr(api, "render_subject", breaks)
    report = run_render(bench.ctx())
    assert failed(report) == {two: "render failure"} and report.done == 1
    row = bench.rows()[two]
    # a local run keeps the labels of its subjects; a path keeps its last part only
    assert row["detail"] == "RuntimeError: cannot use 'T1.mgz' of sub-02 in D-?"
    assert bench.catalog.public().render(two).detail == row["detail"]
    assert bench.rows()[one]["status"] == "done"


@pytest.mark.parametrize("code", [errno.ENOSPC, errno.EROFS, errno.EDQUOT, errno.EACCES])
def test_media_that_cannot_be_written_stop_the_phase_and_are_not_charged_to_a_subject(
    new_bench: Callable[..., Bench], monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    """A full or read-only disk fails every subject alike. Recorded as failures, the
    subjects would stay failed after room was made, until ``retry_failed``."""
    labels = ["sub-01", "sub-02", "sub-03", "sub-04"]
    bench = new_bench(labels, incomplete=["sub-04"], cache=CacheConfig(batch_subjects=2))
    one, two, three = sorted(bench.uid(label) for label in labels[:3])  # as they are rendered
    four = bench.uid("sub-04")
    original = encode.write_rgb_webp

    def no_room(path: Path, *args: object, **kwargs: object) -> None:
        if Path(path).parent.name == three:  # the poster, after the tiles of the subject
            raise OSError(code, os.strerror(code), str(path))
        original(path, *args, **kwargs)

    with monkeypatch.context() as full:
        full.setattr(encode, "write_rgb_webp", no_room)
        with pytest.raises(ConfigError, match="the media cannot be written") as raised:
            run_render(bench.ctx())
    assert raised.value.reason == "invalid" and os.strerror(code) in raised.value.detail
    assert isinstance(raised.value.__cause__, OSError)
    # the batch before is recorded; of the batch that met the error, nothing
    rows = bench.rows()
    assert {uid: row["status"] for uid, row in rows.items()} == {
        one: "done",
        two: "done",
        four: "failed",
    }
    assert rows[four]["reason"] == "missing surface"

    # with room again, the same call does the rest
    report = run_render(bench.ctx())
    assert (report.done, report.skipped) == (1, 2) and failed(report) == {four: "missing surface"}
    assert bench.rows()[three]["status"] == "done"


def test_a_refusal_outside_the_media_is_charged_to_its_subject(
    new_bench: Callable[..., Bench], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bench = new_bench(["sub-01", "sub-02"])
    one, two = bench.uid("sub-01"), bench.uid("sub-02")
    original = api.render_subject
    elsewhere = tmp_path / "elsewhere" / "file"

    def breaks(files: Mapping[FileRole, Path], out_dir: Path, *args: object, **kwargs: object):
        if Path(out_dir).name == two:
            raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), str(elsewhere))
        return original(files, out_dir, *args, **kwargs)

    monkeypatch.setattr(api, "render_subject", breaks)
    report = run_render(bench.ctx())
    assert report.done == 1 and failed(report) == {two: "render failure"}
    assert bench.rows()[two]["detail"].startswith("PermissionError: ")
    assert bench.rows()[one]["status"] == "done"


def test_a_video_that_the_disk_has_no_room_for_stops_the_phase(
    new_bench: Callable[..., Bench], monkeypatch: pytest.MonkeyPatch
) -> None:
    """ffmpeg writes the video itself, so its failed write arrives in words. The text is
    what the ffmpeg of imageio-ffmpeg printed on a full tmpfs (Linux, seen once)."""
    bench = new_bench(["sub-01"])
    said = (
        "ffmpeg failed with status 228: [out#0/mp4 @ 0x55758b27a340] Error writing trailer: "
        "No space left on device"
    )

    def full(*args: object, **kwargs: object) -> int:
        raise RuntimeError(said)

    real = encode.write_mp4
    with monkeypatch.context() as patched:
        # the probe of the encoder is let through; the video of the subject is not
        patched.setattr(
            encode,
            "write_mp4",
            lambda path, *args: (real if Path(path).name == "probe.mp4" else full)(path, *args),
        )
        with pytest.raises(ConfigError, match="the media cannot be written") as raised:
            run_render(bench.ctx(), video=True)
    assert raised.value.reason == "invalid" and bench.rows() == {}
    assert run_render(bench.ctx(), video=True).done == 1


def test_an_encoder_that_cannot_encode_stops_the_phase_and_records_nothing(
    new_bench: Callable[..., Bench],
    rendered: list[tuple[str, bool, bool]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A binary that is there and fails (here: one that exits with 1, like a build without
    libx264) would otherwise fail every subject with the same ``render failure``."""
    bench = new_bench(["sub-01", "sub-02", "sub-03"], incomplete=["sub-03"])
    exe = tmp_path / "ffmpeg-broken"
    exe.write_text("#!/bin/sh\ncat > /dev/null\necho 'Unknown encoder libx264' >&2\nexit 1\n")
    exe.chmod(0o755)
    with monkeypatch.context() as broken:
        broken.setenv("IMAGEIO_FFMPEG_EXE", str(exe))
        with pytest.raises(ConfigError, match="the video cannot be encoded") as raised:
            run_render(bench.ctx(), video=True)
    assert raised.value.reason == "missing"
    assert "status 1" in raised.value.detail and "Unknown encoder libx264" in raised.value.detail
    assert rendered == [] and bench.leased == [] and bench.rows() == {}
    assert not (bench.run_dir / "media").exists()
    assert run_render(bench.ctx(), video=True).done == 2  # with the encoder back


def test_a_census_run_keeps_no_label_in_a_detail_and_shows_no_detail(
    new_bench: Callable[..., Bench], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bench = new_bench(["sub-01", "sub-02", "sub-03"], incomplete=["sub-03"], census=True)
    two, three = bench.uid("sub-02"), bench.uid("sub-03")

    def breaks(files: Mapping[FileRole, Path], out_dir: Path, *args: object, **kwargs: object):
        if Path(out_dir).name == two:
            raise OSError(f"cannot use {files[FileRole.T1]} of sub-02 in ds000102")
        raise api.RenderError("render failure", f"unreadable input ({files[FileRole.T1]})")

    monkeypatch.setattr(api, "render_subject", breaks)
    report = run_render(bench.ctx(), video=False)
    assert failed(report) == {
        bench.uid("sub-01"): "render failure",
        two: "render failure",
        three: "missing surface",
    }
    rows = bench.rows()
    assert rows[two]["detail"] == "OSError: cannot use T1.mgz of sub-? in D-?"
    assert rows[bench.uid("sub-01")]["detail"] == "unreadable input (T1.mgz)"
    assert rows[three]["detail"] == "no file for lh_pial"
    stored = " ".join(str(row["detail"]) for row in rows.values()) + repr(report)
    assert "sub-0" not in stored and str(tmp_path) not in stored and "study" not in stored
    public = bench.catalog.public()
    assert "detail" not in public.renders().columns
    assert public.render(two).detail is None and public.render(two).reason == "render failure"


def test_the_reason_and_detail_of_each_kind_of_exception() -> None:
    cases = [
        (api.RenderError("wrong orientation", "the axes"), ("wrong orientation", "the axes")),
        (
            SourceError("changed", "surf/lh.pial was removed"),
            ("source changed", "surf/lh.pial was removed"),
        ),
        (
            SourceError("unreadable", "mri/T1.mgz: Permission denied"),
            ("render failure", "SourceError: unreadable: mri/T1.mgz: Permission denied"),
        ),
        (WorkerError("Local", "it broke"), ("render failure", "Local: it broke")),
        (
            BrokenProcessPool("a process died"),
            ("render failure", "BrokenProcessPool: a process died"),
        ),
        (MemoryError(), ("render failure", "MemoryError: ")),
    ]
    for exception, expected in cases:
        assert batch._failure(exception) == expected


def test_a_run_directory_inside_a_subject_stops_the_phase(
    new_bench: Callable[..., Bench], tmp_path: Path
) -> None:
    """``render_subject`` writes nothing into a directory it reads from. That says something
    about where the run directory is and nothing about the subject."""
    tree = write_tree(tmp_path / "study", "subjects_dir", ["sub-01"])
    bench = new_bench(trees=[tree], run_dir=tree / "sub-01" / "qc")
    source = snapshot_tree(tree)
    with pytest.raises(ConfigError, match="run directory lies inside") as raised:
        run_render(bench.ctx())
    assert raised.value.reason == "invalid"
    assert isinstance(raised.value.__cause__, api.OutputInSourceError)
    assert bench.rows() == {} and snapshot_tree(tree) == source


# ---------------------------------------------------------------- interruptions


def test_an_interrupted_run_keeps_its_batches_and_a_rerun_does_the_rest(
    new_bench: Callable[..., Bench], monkeypatch: pytest.MonkeyPatch
) -> None:
    labels = [f"sub-0{n}" for n in range(1, 6)]
    bench = new_bench(labels, cache=CacheConfig(batch_subjects=2))
    order = bench.catalog.public().subject_uids()
    original = api.render_subject
    calls: list[str] = []

    def interrupted(files: Mapping[FileRole, Path], out_dir: Path, *args: object, **kwargs: object):
        calls.append(Path(out_dir).name)
        if len(calls) == 4:  # the second subject of the second batch
            raise KeyboardInterrupt
        return original(files, out_dir, *args, **kwargs)

    monkeypatch.setattr(api, "render_subject", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run_render(bench.ctx())
    assert calls == order[:4]
    rows = bench.rows()
    assert set(rows) == set(order[:2]) and all(row["status"] == "done" for row in rows.values())

    del calls[:]
    report = run_render(bench.ctx())
    assert calls == order[2:] and (report.done, report.skipped, report.failed) == (3, 2, 0)
    assert {uid: row["status"] for uid, row in bench.rows().items()} == dict.fromkeys(order, "done")
    assert {uid: bench.rows()[uid] for uid in order[:2]} == rows


def intensity_rows(uid: str, value: float) -> pd.DataFrame:
    schema = intensity_schema(2)
    rows = [
        {name: None for name in schema.names}
        | {"subject_uid": uid, "hemi": hemi, "region_kind": "annot", "gap_frac": value}
        for hemi in ("lh", "rh")
    ]
    return pd.DataFrame(rows)


def test_rows_computed_by_a_render_go_to_the_part_of_the_dataset(
    new_bench: Callable[..., Bench], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Milestone 5 computes the intensity metric inside the render. The table has the
    columns of the run's depths, here two, which is not the default schema."""
    trees = [
        write_tree(tmp_path / name, "subjects_dir", ["sub-01", "sub-02"]) for name in ("a", "b")
    ]
    bench = new_bench(trees=trees, intensity=IntensityConfig(depths=(0.5, 1.0)))
    ras_t1(tmp_path / "b" / "sub-02")
    bench.inventory()
    value = 0.25
    real = batch._render_one

    def with_rows(uid: str, files: Mapping[FileRole, Path], **kwargs: object) -> RenderResult:
        result = real(uid, files, **kwargs)
        return dataclasses.replace(result, extras={"intensity": intensity_rows(uid, value)})

    monkeypatch.setattr(batch, "_render_one", with_rows)
    schema = intensity_schema(2)

    def parts() -> dict[str, dict[str, float]]:
        found = {}
        for number in range(2):
            path = store.part_path(bench.run_dir, "intensity", bench.pseudonym(number))
            frame = store.read_table(path, schema)
            assert list(frame["subject_uid"]) == sorted(frame["subject_uid"])
            assert sorted(frame["hemi"].unique()) == ["lh", "rh"] and len(frame) % 2 == 0
            found[bench.pseudonym(number)] = dict(
                zip(frame["subject_uid"], frame["gap_frac"], strict=True)
            )
        return found

    assert run_render(bench.ctx()).done == 3
    expected = {
        bench.pseudonym(0): {bench.uid("sub-01"): 0.25, bench.uid("sub-02"): 0.25},
        bench.pseudonym(1): {bench.uid("sub-01", 1): 0.25},  # the other one failed
    }
    assert parts() == expected
    parts_dir = bench.run_dir / "tables" / "_parts" / "intensity"
    assert sorted(p.name for p in parts_dir.iterdir()) == sorted(
        f"{bench.pseudonym(number)}.parquet" for number in range(2)
    )

    # a subject rendered again replaces its rows and leaves the others
    value = 0.75
    run_render(bench.ctx(), subjects=[bench.uid("sub-02")], force=True)
    expected[bench.pseudonym(0)][bench.uid("sub-02")] = 0.75
    assert parts() == expected


def test_a_crash_before_the_rows_are_recorded_redoes_the_batch(
    new_bench: Callable[..., Bench], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Files and parts are written first and the rows last. A crash in between leaves no
    row, so the next run does the subjects again, and their rows are in the part once."""
    bench = new_bench(["sub-01", "sub-02"], intensity=IntensityConfig(depths=(0.5, 1.0)))
    uids = sorted(bench.uid(label) for label in ("sub-01", "sub-02"))
    real = batch._render_one
    calls: list[str] = []

    def with_rows(uid: str, files: Mapping[FileRole, Path], **kwargs: object) -> RenderResult:
        calls.append(uid)
        result = real(uid, files, **kwargs)
        return dataclasses.replace(result, extras={"intensity": intensity_rows(uid, 0.5)})

    monkeypatch.setattr(batch, "_render_one", with_rows)
    ctx = bench.ctx()

    def crash(rows: object) -> None:
        raise RuntimeError("the process died here")

    with monkeypatch.context() as patch:
        patch.setattr(ctx.public, "record_renders", crash)
        with pytest.raises(RuntimeError, match="died here"):
            run_render(ctx)
    part = store.part_path(bench.run_dir, "intensity", bench.pseudonym())
    assert part.is_file() and bench.rows() == {}
    assert all((bench.media(uid) / "render.json").is_file() for uid in uids)

    report = run_render(bench.ctx())
    assert calls == uids * 2 and (report.done, report.skipped) == (2, 0)
    assert set(bench.rows()) == set(uids)
    frame = store.read_table(part, intensity_schema(2))
    assert sorted(zip(frame["subject_uid"], frame["hemi"], strict=True)) == [
        (uid, hemi) for uid in uids for hemi in ("lh", "rh")
    ]


# ---------------------------------------------------------------- processes


def test_one_process_and_two_give_the_same_files_and_rows(
    new_bench: Callable[..., Bench], tmp_path: Path
) -> None:
    labels = [f"sub-0{n}" for n in range(1, 7)]
    bench = new_bench(labels, incomplete=["sub-06"], cache=CacheConfig(batch_subjects=3))
    alter_t1(bench.subject_dir("sub-02"))
    (bench.subject_dir("sub-03") / "mri" / "T1.mgz").write_bytes(b"")
    ras_t1(bench.subject_dir("sub-04"))
    bench.inventory()
    touch(bench.subject_dir("sub-05") / "surf" / "lh.white")
    source = snapshot_tree(tmp_path / "study")

    # the same run twice: a copy of the run directory has the same run secret
    shutil.copytree(bench.run_dir, tmp_path / "run2")
    twin = Bench(
        tmp_path / "run2",
        bench.config,
        bench.key,
        Catalog(tmp_path / "run2" / CATALOG_FILE),
        bench.sources,
    )
    try:
        alone = run_render(bench.ctx(), jobs=1, video=True)
        pair = run_render(twin.ctx(), jobs=2, video=True)
        rows, twin_rows = bench.rows(), twin.rows()
        digests = [
            b.catalog.query("SELECT subject_uid, role, sha256 FROM files ORDER BY 1, 2")
            for b in (bench, twin)
        ]
    finally:
        twin.catalog.close()

    assert dataclasses.replace(pair, seconds=0.0) == dataclasses.replace(alone, seconds=0.0)
    assert (alone.done, alone.skipped, alone.failed) == (2, 0, 4)
    assert sorted(set(failed(alone).values())) == [
        "missing surface",
        "render failure",
        "source changed",
        "wrong orientation",
    ]
    media = tree_bytes(bench.run_dir / "media")
    assert tree_bytes(twin.run_dir / "media") == media
    assert sum(name.endswith("anim.mp4") for name in media) == 2

    def comparable(table: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
        return {
            uid: {name: value for name, value in row.items() if name != "rendered_at"}
            for uid, row in table.items()
        }

    assert comparable(twin_rows) == comparable(rows) and len(rows) == 6
    # an error crosses the process boundary with its reason and its detail
    assert rows[bench.uid("sub-03")]["detail"] == "unreadable input (T1.mgz: Empty file: 'T1.mgz')"
    assert [tuple(r) for r in digests[1]] == [tuple(r) for r in digests[0]]
    assert sum(r["sha256"] is not None for r in digests[0]) == 2 * len(TIER_C)
    assert snapshot_tree(tmp_path / "study") == source


# ---------------------------------------------------------------- the sample for the check by eye


def test_the_check_sample_is_seeded_named_and_drawn_from_the_done_renders(
    new_bench: Callable[..., Bench],
) -> None:
    labels = [f"sub-0{n}" for n in range(1, 8)]
    bench = new_bench(labels, incomplete=["sub-06"])
    ras_t1(bench.subject_dir("sub-07"))
    bench.inventory()
    report = run_render(bench.ctx())
    assert (report.done, report.failed) == (5, 2)
    done = sorted(bench.uid(label) for label in labels[:5])
    ctx = bench.ctx()

    path = render_check_sample(ctx, fraction=0.3, seed=11)

    assert path == bench.run_dir / "logs" / "render-check-11.txt"
    lines = path.read_text().splitlines()
    assert lines[0] == "# render check: seed 11, 2 of 5 done renders (fraction 0.3)"
    assert math.ceil(0.3 * 5) == 2
    drawn = np.random.default_rng(11).choice(5, size=2, replace=False)
    expected = sorted(done[position] for position in drawn)
    assert lines[1:] == [f"{uid}\tmedia/{uid}/poster.webp" for uid in expected]
    for line in lines[1:]:
        assert (bench.run_dir / line.split("\t")[1]).is_file()

    page = path.with_suffix(".html")
    text = page.read_text()
    assert sorted(p.name for p in path.parent.iterdir()) == [page.name, path.name]
    assert "seed 11" in text and text.count("<img ") == 2
    for uid in expected:
        link = f"../media/{uid}/poster.webp"
        assert f'src="{link}"' in text and (page.parent / link).resolve().is_file()
    assert str(bench.run_dir) not in text + path.read_text()

    # the same seed gives the same files, another seed other subjects under another name
    before = (path.read_bytes(), page.read_bytes())
    assert render_check_sample(ctx, fraction=0.3, seed=11) == path
    assert (path.read_bytes(), page.read_bytes()) == before
    samples = set()
    for seed in range(8):
        other = render_check_sample(ctx, fraction=0.3, seed=seed)
        assert other.name == f"render-check-{seed}.txt"
        assert other.read_text().splitlines()[0].startswith(f"# render check: seed {seed}, 2 of 5")
        samples.add(tuple(other.read_text().splitlines()[1:]))
    assert len(samples) > 1
    assert path.read_bytes() == before[0]

    # the default fraction draws at least one render, a fraction of 1 all of them
    assert len(render_check_sample(ctx, seed=3).read_text().splitlines()) == 2
    everything = render_check_sample(ctx, fraction=1, seed=4).read_text().splitlines()
    assert [line.split("\t")[0] for line in everything[1:]] == done


def test_the_check_sample_of_a_run_without_renders_is_empty(
    new_bench: Callable[..., Bench],
) -> None:
    bench = new_bench(["sub-01", "sub-02"], incomplete=["sub-02"])
    ras_t1(bench.subject_dir("sub-01"))
    bench.inventory()
    assert run_render(bench.ctx()).failed == 2
    path = render_check_sample(bench.ctx(), seed=5)
    assert path.read_text() == "# render check: seed 5, 0 of 0 done renders (fraction 0.02)\n"
    assert "<img" not in path.with_suffix(".html").read_text()


@pytest.mark.parametrize(
    ("fraction", "seed", "problem"),
    [
        (0.0, 1, "fraction"),
        (-0.1, 1, "fraction"),
        (1.5, 1, "fraction"),
        (float("nan"), 1, "fraction"),
        (0.5, -1, "seed"),
    ],
)
def test_a_check_sample_needs_a_fraction_and_a_seed_that_can_be_used(
    new_bench: Callable[..., Bench], fraction: float, seed: int, problem: str
) -> None:
    bench = new_bench(["sub-01"])
    with pytest.raises(ValueError, match=problem):
        render_check_sample(bench.ctx(), fraction=fraction, seed=seed)
    assert not (bench.run_dir / "logs").exists()


# ---------------------------------------------------------------- the real subject


@pytest.mark.network
def test_the_real_subject_through_a_run_equals_its_direct_render(
    real_subject: Path, new_bench: Callable[..., Bench], tmp_path: Path
) -> None:
    """ds000102 sub-01 from the pytest cache, linked into a subjects directory."""
    source = snapshot_tree(real_subject)
    study = tmp_path / "study"
    study.mkdir()
    (study / "sub-01").symlink_to(real_subject, target_is_directory=True)
    bench = new_bench(trees=[study], render=RenderConfig())
    one = bench.uid("sub-01")

    report = run_render(bench.ctx())

    assert (report.done, report.failed) == (1, 0)
    info = api.three_plane(real_subject, tmp_path / "direct", video=False)
    assert tree_bytes(bench.media(one)) == tree_bytes(tmp_path / "direct")
    row = bench.rows()[one]
    assert row["render_key"] == bench.expected_key(one, RenderConfig())
    assert row["n_frames"] == len(info.modes["sync"].frames) == 120
    assert run_render(bench.ctx()).skipped == 1
    assert snapshot_tree(real_subject) == source
