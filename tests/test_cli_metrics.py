"""Tests of ``surfcensus metrics``: what it prints, its exit codes, the template options, and
what ``status`` and ``render`` say once a run has metrics."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pandas as pd
import pytest
from synth import snapshot_tree  # pyrefly: ignore[missing-import]
from synth_template import (  # pyrefly: ignore[missing-import]
    SyntheticTemplate,
    write_synthetic_template,
)
from synth_tierb import SyntheticCohort, synthetic_cohort  # pyrefly: ignore[missing-import]
from synth_tree import write_tier_a, write_tree  # pyrefly: ignore[missing-import]
from typer.testing import CliRunner, Result

from surfcensus.cli import app
from surfcensus.config import CONFIG_FILE, MetricsConfig, new_local_config
from surfcensus.render import RenderConfig
from surfcensus.run import Run
from surfcensus.sources.local import local

SECONDS = r"\d+\.\d s"
TOO_FEW = (
    "  {n} subjects with a map: too few for per-location scoring (30 needed). Ranked on raw "
    "ceiling fraction and occipital-to-control thickness."
)


def cli(*args: object) -> Result:
    return CliRunner().invoke(app, [str(arg) for arg in args])


def closing(result: Result, done: int, skipped: int, failed: int) -> bool:
    """Whether the first line of the output is the count line with these numbers."""
    line = rf"metrics: {done} done, {skipped} skipped, {failed} failed, {SECONDS}"
    return re.fullmatch(line, result.stdout.splitlines()[0]) is not None


def reference_id(run_dir: Path) -> str:
    (name,) = [path.name for path in (run_dir / "tables" / "reference").iterdir()]
    return name


def pinned(run_dir: Path) -> str | None:
    with Run(run_dir) as run:
        return run.public().meta("template_id")


@pytest.fixture
def three(tmp_path: Path, synth_template: SyntheticTemplate) -> Path:
    """A subjects directory of the test's own with three subjects and no template."""
    root = tmp_path / "subjects"
    synthetic_cohort(root, 3, synth_template, seed=7)
    return root


@pytest.fixture
def three_run(three: Path, tmp_path: Path) -> Path:
    """An inventoried local run on ``three`` that scores three subjects within the set."""
    run_dir = tmp_path / "run"
    config = new_local_config(
        three,
        render=RenderConfig(panel_px=32),
        metrics=MetricsConfig(min_reference_n=3, min_vertex_n=2),
    )
    with Run(run_dir, local(three), config=config) as run:
        run.inventory()
    return run_dir


def test_metrics_of_a_small_set_says_how_it_was_ranked(
    small_cohort: SyntheticCohort, tmp_path: Path
) -> None:
    before = snapshot_tree(small_cohort.root)
    run_dir = tmp_path / "run"
    assert cli("inventory", "--run-dir", run_dir, "--source", small_cohort.root).exit_code == 0
    stored = (run_dir / CONFIG_FILE).read_bytes()
    result = cli("metrics", "--run-dir", run_dir)
    assert result.exit_code == 0, result.output
    assert closing(result, 12, 0, 0)
    assert result.stdout.splitlines()[1:] == [
        TOO_FEW.format(n=5),
        "scores: 5 subjects; 5 ranked without a reference (decision D16)",
    ]
    assert result.stderr == ""

    again = cli("metrics", "--run-dir", run_dir, "--jobs", 2)
    assert again.exit_code == 0 and closing(again, 0, 12, 0)
    assert again.stdout.splitlines()[1:] == result.stdout.splitlines()[1:]
    assert (run_dir / CONFIG_FILE).read_bytes() == stored
    assert snapshot_tree(small_cohort.root) == before

    status = cli("status", "--run-dir", run_dir)
    assert status.stdout.splitlines()[2:] == [
        "metrics regional: 5 done, 0 failed",
        "metrics fsaverage: 5 done, 0 failed",
        "metrics reference: 1 done, 0 failed",
        "metrics score: 1 done, 0 failed",
        "render: 0 done, 0 failed, 5 pending",
    ]


def test_stages_and_a_dataset_filter(run_copy: Path) -> None:
    reference = reference_id(run_copy)
    scored = f"scores: 3 subjects; 3 against {reference}"
    result = cli("metrics", "--run-dir", run_copy)
    assert result.exit_code == 0 and closing(result, 0, 8, 0), result.output
    assert result.stdout.splitlines()[1:] == [scored]

    native = cli("metrics", "--run-dir", run_copy, "--stage", "native")
    assert native.exit_code == 0 and closing(native, 0, 1, 0), native.output
    forced = cli(
        "metrics", "--run-dir", run_copy, "--stage", "score", "--stage", "merge", "--force"
    )
    assert forced.exit_code == 0 and closing(forced, 3, 0, 0), forced.output
    assert reference_id(run_copy) == reference

    with Run(run_copy) as run:
        (dataset,) = run.public().datasets()["pseudonym"]
    filtered = cli("metrics", "--run-dir", run_copy, "--dataset", dataset, "--retry-failed")
    assert filtered.exit_code == 0 and closing(filtered, 0, 6, 0), filtered.output

    unknown = cli("metrics", "--run-dir", run_copy, "--stage", "clusters")
    assert unknown.exit_code == 2
    assert unknown.output.startswith("invalid: no metric stage clusters; the stages are regional,")
    assert len(unknown.output.splitlines()) == 1
    no_run = cli("metrics", "--run-dir", run_copy / "nowhere")
    assert no_run.exit_code == 2 and no_run.output.startswith("missing: ")


def test_a_failed_unit_is_exit_1_and_is_listed_by_status(tmp_path: Path) -> None:
    """Light subjects without a template; the stats file of ``sub-01`` is no parcel table."""
    study = write_tree(tmp_path / "study", "fmriprep", ("sub-01", "sub-02"))
    write_tier_a(study / "sourcedata" / "freesurfer" / "sub-01")
    run_dir = tmp_path / "run"
    assert cli("inventory", "--run-dir", run_dir, "--source", study).exit_code == 0
    result = cli("metrics", "--run-dir", run_dir)
    assert result.exit_code == 1, result.output
    assert closing(result, 1, 0, 1)
    out = result.stdout.splitlines()
    assert out[1] == "  regional: unreadable: 1 (sub-01)"
    assert out[2].startswith("  No fsaverage template: thickness is not compared per location.")
    assert out[3] == "scores: 2 subjects; 2 ranked without a reference (decision D16)"
    # where the template was looked for, once, on stderr
    assert result.stderr.startswith("WARNING: No fsaverage template found. Looked in:\n")
    assert result.stderr.count("No fsaverage template found") == 1
    assert "To enable: pass --fsaverage /path/to/fsaverage" in result.stderr

    again = cli("metrics", "--run-dir", run_dir)
    assert again.exit_code == 1 and closing(again, 0, 1, 1)
    status = cli("status", "--run-dir", run_dir, "--failures")
    assert status.exit_code == 0
    assert "metrics regional: 0 done, 1 failed" in status.stdout.splitlines()
    (listed,) = [line for line in status.stdout.splitlines() if line.startswith("regional ")]
    assert re.fullmatch(
        r"regional [0-9a-f]{16} \(sub-01\): unreadable: lh\.aparc\.stats: .+", listed
    )
    assert str(tmp_path) not in listed
    # a render says nothing about the template unless asked for more
    assert cli("render", "--run-dir", run_dir).stderr == ""


def test_fsaverage_gives_a_run_its_template_and_the_run_keeps_it(
    three: Path, three_run: Path, synth_template: SyntheticTemplate, tmp_path: Path
) -> None:
    stored = (three_run / CONFIG_FILE).read_bytes()
    empty = tmp_path / "empty"
    empty.mkdir()
    missing = cli("metrics", "--run-dir", three_run, "--fsaverage", empty)
    assert missing.exit_code == 2 and len(missing.output.splitlines()) == 1
    assert missing.output.startswith("not found: no usable fsaverage template; ")
    assert "incomplete: " in missing.output and pinned(three_run) is None

    result = cli("metrics", "--run-dir", three_run, "--fsaverage", synth_template.path)
    assert result.exit_code == 0 and closing(result, 8, 0, 0), result.output
    assert (
        result.stdout.splitlines()[1] == f"scores: 3 subjects; 3 against {reference_id(three_run)}"
    )
    template_id = pinned(three_run)
    assert template_id is not None and result.stderr == ""
    # the directory is not written down: the run holds the template itself
    assert (three_run / CONFIG_FILE).read_bytes() == stored
    assert (three_run / "templates" / template_id / "template.json").is_file()

    again = cli("metrics", "--run-dir", three_run)
    assert again.exit_code == 0 and closing(again, 0, 8, 0) and again.stderr == ""
    # a directory that cannot be used is passed over with a warning: the run has its template
    kept = cli("metrics", "--run-dir", three_run, "--fsaverage", empty)
    assert kept.exit_code == 0 and closing(kept, 0, 8, 0)
    assert kept.stderr.startswith("WARNING: The fsaverage directory that was given cannot be used")

    # the renders of this run carry the counterparts on that template
    rendered = cli("render", "--run-dir", three_run)
    assert rendered.exit_code == 0, rendered.output
    with Run(three_run) as run:
        assert set(run.public().renders()["template_id"]) == {template_id}
        assert not run.clusters()["native_vertex"].isna().all()


def test_another_template_stops_unless_retemplate(
    three: Path, three_run: Path, synth_template: SyntheticTemplate, tmp_path: Path
) -> None:
    assert cli("metrics", "--run-dir", three_run, "--fsaverage", synth_template.path).exit_code == 0
    first = pinned(three_run)
    scores = (three_run / "tables" / "scores.parquet").read_bytes()
    other = write_synthetic_template(tmp_path / "other", with_avg_area=False)

    refused = cli("metrics", "--run-dir", three_run, "--fsaverage", other.path)
    assert refused.exit_code == 2 and len(refused.output.splitlines()) == 1
    assert re.fullmatch(rf"changed: {first} -> [0-9a-f]{{12}}; pass --retemplate\n", refused.output)
    assert pinned(three_run) == first
    assert (three_run / "tables" / "scores.parquet").read_bytes() == scores

    taken = cli("metrics", "--run-dir", three_run, "--fsaverage", other.path, "--retemplate")
    assert taken.exit_code == 0, taken.output
    # the stats files are as they were; every map, the reference and the scores are redone
    assert closing(taken, 3 + 2, 3, 0)
    second = pinned(three_run)
    assert second not in (None, first)
    assert sorted(p.name for p in (three_run / "templates").iterdir()) == sorted([first, second])
    assert cli("metrics", "--run-dir", three_run).stdout.startswith("metrics: 0 done, 8 skipped")


def test_render_takes_the_template_option_as_well(
    three: Path, three_run: Path, synth_template: SyntheticTemplate
) -> None:
    bare = cli("render", "--run-dir", three_run)
    assert bare.exit_code == 0 and pinned(three_run) is None
    with_template = cli("render", "--run-dir", three_run, "--fsaverage", synth_template.path)
    assert with_template.exit_code == 0, with_template.output
    assert with_template.stdout.startswith("render: 3 done, 0 skipped, 0 failed")
    assert pinned(three_run) is not None
    sample = cli("render", "--run-dir", three_run, "--check-sample", "--fsaverage", "x")
    assert sample.exit_code == 2


def test_the_template_next_to_the_subjects_is_found_without_an_option(
    three: Path, three_run: Path, synth_template: SyntheticTemplate
) -> None:
    # the directory arrives after the inventory, and is found without a second one
    (three / "fsaverage").symlink_to(synth_template.path, target_is_directory=True)
    result = cli("metrics", "--run-dir", three_run)
    assert result.exit_code == 0 and closing(result, 8, 0, 0) and result.stderr == ""
    shutil.rmtree(three_run / "templates")
    assert closing(cli("metrics", "--run-dir", three_run), 0, 8, 0)


def test_a_damaged_map_of_the_run_is_one_line_and_exit_code_2(run_copy: Path) -> None:
    """It used to be a traceback with exit code 1, the code of failed units."""
    maps = sorted((run_copy / "tables" / "fsavg").iterdir())
    whole = maps[0].read_bytes()
    maps[0].write_bytes(whole[:300])
    stopped = cli("metrics", "--run-dir", run_copy, "--stage", "reference", "--force")
    assert stopped.exit_code == 2, stopped.output
    assert isinstance(stopped.exception, SystemExit)
    assert stopped.output == (
        "invalid: the fsaverage map of 1 subject(s) in tables/fsavg could not be read "
        f"({maps[0].stem}) and was removed; run the metrics again to make it anew\n"
    )
    # a fault of the run: no unit of any stage is in the failed state
    status = cli("status", "--run-dir", run_copy).stdout.splitlines()[1:]
    assert status and all(", 0 failed" in line for line in status), status
    # once the cause is gone, the plain command does the work
    again = cli("metrics", "--run-dir", run_copy)
    assert again.exit_code == 0 and closing(again, 1, 7, 0), again.output
    assert maps[0].read_bytes() == whole


def test_a_changed_byte_in_a_map_or_a_vertex_map_is_no_traceback(run_copy: Path) -> None:
    """One changed byte makes numpy or zipfile raise an error that no reader listed: both
    used to end the command in a traceback with exit code 1."""
    maps = sorted((run_copy / "tables" / "fsavg").iterdir())
    changed = bytearray(maps[0].read_bytes())
    changed[54] ^= 0x5A  # in the header of the first array
    maps[0].write_bytes(bytes(changed))
    stopped = cli("metrics", "--run-dir", run_copy, "--stage", "reference", "--force")
    assert stopped.exit_code == 2 and isinstance(stopped.exception, SystemExit), stopped.output
    assert stopped.output.startswith("invalid: the fsaverage map of 1 subject(s)")
    assert closing(cli("metrics", "--run-dir", run_copy), 1, 7, 0)

    vertex_map = sorted((run_copy / "media").glob("*/vertex_map.npz"))[0]
    whole = vertex_map.read_bytes()
    changed = bytearray(whole)
    changed[whole.index(b"PK\x01\x02") + 6] ^= 0x5A  # the zip version of the first entry
    vertex_map.write_bytes(bytes(changed))
    noted = cli("metrics", "--run-dir", run_copy, "--stage", "native", "--force")
    assert noted.exit_code == 0 and closing(noted, 1, 0, 0), noted.output
    assert f"cannot be read ({vertex_map.parent.name})" in noted.stdout


def test_a_damaged_reference_is_one_line_and_the_plain_command_builds_it_again(
    run_copy: Path,
) -> None:
    name = reference_id(run_copy)
    stratum = run_copy / "tables" / "reference" / name / "stratum_000.npz"
    whole = stratum.read_bytes()
    stratum.write_bytes(whole[:1000])
    stopped = cli("metrics", "--run-dir", run_copy, "--stage", "score", "--force")
    assert stopped.exit_code == 2 and isinstance(stopped.exception, SystemExit), stopped.output
    assert stopped.output == (
        f"invalid: the reference {name} in tables/reference cannot be read; "
        "run `surfcensus metrics`, which builds it again\n"
    )
    again = cli("metrics", "--run-dir", run_copy)
    assert again.exit_code == 0 and closing(again, 1, 7, 0), again.output
    assert stratum.read_bytes() == whole
    assert closing(cli("metrics", "--run-dir", run_copy), 0, 8, 0)


@pytest.mark.parametrize("damage", ["another schema", "no parquet"])
def test_a_cluster_table_that_is_not_one_is_one_line_and_exit_code_2(
    run_copy: Path, damage: str
) -> None:
    """A table of another schema under the name ``clusters.parquet`` (an export once wrote
    one there) ended ``render``, ``review`` and ``metrics`` in a traceback with exit code 1."""
    table = run_copy / "tables" / "clusters.parquet"
    whole = table.read_bytes()
    if damage == "another schema":
        shutil.copy(run_copy / "tables" / "scores.parquet", table)
    else:
        table.write_bytes(b"subject_uid,rank\n")
    for command in (("render",), ("metrics", "--stage", "native"), ("metrics",)):
        stopped = cli(command[0], "--run-dir", run_copy, *command[1:])
        assert stopped.exit_code == 2, (command, stopped.output)
        assert isinstance(stopped.exception, SystemExit)
        assert len(stopped.output.splitlines()) == 1, stopped.output
        assert stopped.output.startswith(
            "invalid: tables/clusters.parquet cannot be read as the cluster table of this run ("
        )
        assert stopped.output.endswith(
            "; run `surfcensus metrics --stage reference --force`, which writes it again\n"
        )
    # a fault of the run: no unit of any stage is in the failed state
    status = cli("status", "--run-dir", run_copy).stdout.splitlines()[1:]
    assert status and all(", 0 failed" in line for line in status), status
    # the command that the line names writes the table again, with the bytes it had
    again = cli("metrics", "--run-dir", run_copy, "--stage", "reference", "--force")
    assert again.exit_code == 0 and closing(again, 1, 0, 0), again.output
    assert table.read_bytes() == whole
    assert closing(cli("metrics", "--run-dir", run_copy), 0, 8, 0)
    assert cli("render", "--run-dir", run_copy).exit_code == 0


def test_a_damaged_vertex_map_is_a_note_and_no_failure(run_copy: Path) -> None:
    """It used to be a traceback at the end of ``render`` and of ``metrics --stage native``."""
    with Run(run_copy) as run:
        clusters = run.clusters()
    uid = str(clusters.loc[clusters["native_vertex"].notna(), "subject_uid"].iloc[0])
    vertex_map = run_copy / "media" / uid / "vertex_map.npz"
    vertex_map.write_bytes(vertex_map.read_bytes()[:500])
    note = (
        f"  media/<subject_uid>/vertex_map.npz of 1 rendered subject(s) cannot be read ({uid}): "
        "their cluster centroids have no native position. Remove the file and run "
        "`surfcensus render`, which makes such a render again."
    )
    native = cli("metrics", "--run-dir", run_copy, "--stage", "native", "--force")
    assert native.exit_code == 0 and closing(native, 1, 0, 0), native.output
    assert native.stdout.splitlines()[1] == note
    rendered = cli("render", "--run-dir", run_copy)
    assert rendered.exit_code == 0, rendered.output
    assert rendered.stdout.startswith("render: 0 done, 3 skipped, 0 failed")
    assert rendered.stdout.splitlines()[1:] == [note]
    with Run(run_copy) as run:
        assert uid not in set(run.clusters().dropna(subset=["native_vertex"])["subject_uid"])

    vertex_map.unlink()
    repaired = cli("render", "--run-dir", run_copy)
    assert repaired.exit_code == 0, repaired.output
    assert repaired.stdout.startswith("render: 1 done, 2 skipped, 0 failed")
    assert len(repaired.stdout.splitlines()) == 1
    with Run(run_copy) as run:
        pd.testing.assert_frame_equal(run.clusters(), clusters)
