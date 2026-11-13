"""Phase 4 through ``Run``: the metric stages, their tables and their skip tests, on synthetic
cohorts whose thickness is known in closed form.

``cohort_run`` is the 32-subject cohort scored within the set, read-only. Tests that change
a run work on a copy of its directory or on a small cohort of their own.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import nibabel.freesurfer as fs
import numpy as np
import pandas as pd
import pytest
from synth import snapshot_tree  # pyrefly: ignore[missing-import]
from synth_template import SyntheticTemplate  # pyrefly: ignore[missing-import]
from synth_tierb import (  # pyrefly: ignore[missing-import]
    CANARIES,
    SyntheticCohort,
    cohort_patches,
    in_patch,
    synthetic_cohort,
)

from surfcensus.config import ConfigError, MetricsConfig, RunConfig, new_local_config, with_changes
from surfcensus.metrics.fsaverage import MAD_TO_SD, global_median, read_maps
from surfcensus.metrics.pipeline import (
    map_path,
    merge_tables,
    reference_dir,
    run_metrics,
    update_native,
)
from surfcensus.metrics.regional import occ_ctrl_ratio
from surfcensus.metrics.scoring import quantile_scores
from surfcensus.records import ALL_METRIC_STAGES, METRIC_STAGES, PhaseReport
from surfcensus.reference import load
from surfcensus.reference.load import Reference, read_index, read_reference, write_reference
from surfcensus.render.model import RenderConfig
from surfcensus.render.vertexmap import read_vertex_map
from surfcensus.run import Run
from surfcensus.schemas import SCHEMAS
from surfcensus.sources.base import HEMIS
from surfcensus.sources.local import LocalSource, local
from surfcensus.store import CATALOG_FILE, Catalog, read_table, table_path
from surfcensus.templates import TemplateError

SET_OF_THREE = MetricsConfig(min_reference_n=3, min_vertex_n=2)
TABLES = ("regional", "vertex_summary", "clusters", "scores")


def labels(run: Run) -> dict[str, str]:
    """Label -> ``subject_uid`` of a local run."""
    subjects = run.public().subjects()
    return dict(zip(subjects["subject_label"], subjects["subject_uid"], strict=True))


def table(run_dir: Path, name: str) -> pd.DataFrame:
    """A table of the run, read with the schema it must have."""
    return read_table(table_path(run_dir, name), SCHEMAS[name])


def counts(report: PhaseReport) -> tuple[int, int, int]:
    return report.done, report.skipped, report.failed


def run_files(run_dir: Path) -> dict[str, bytes]:
    """Everything the metrics write, by path relative to the run directory."""
    return {
        path.relative_to(run_dir).as_posix(): path.read_bytes()
        for folder in ("tables", "templates")
        for path in sorted((run_dir / folder).rglob("*"))
        if path.is_file()
    }


def ledger(run: Run, stage: str) -> dict[str, str]:
    """Unit -> the time its row of the stage was written."""
    frame = run.public().work(stage)
    return dict(zip(frame["unit"], frame["finished_at"], strict=True))


def little_cohort(
    root: Path, n: int, template: SyntheticTemplate, *, with_template: bool = True
) -> SyntheticCohort:
    """``n`` subjects of the cohort kind in a subjects directory of the test's own,
    ``sub-001`` with the medial patch."""
    patch = cohort_patches()["medial"]
    subjects = synthetic_cohort(root, n, template, seed=11, bad={1: patch})
    if with_template:
        (root / "fsaverage").symlink_to(template.path, target_is_directory=True)
    return SyntheticCohort(
        root=root,
        subjects=tuple(subjects),
        template=template,
        bad={subjects[1].path.name: patch},
        no_sphere_reg=(),
        no_stats=(),
    )


@dataclass(frozen=True)
class CohortRun:
    run_dir: Path
    config: RunConfig
    cohort: SyntheticCohort
    report: PhaseReport
    source_unchanged: bool
    uids: dict[str, str]


@pytest.fixture(scope="module")
def cohort_run(
    tmp_path_factory: pytest.TempPathFactory, synth_cohort: SyntheticCohort
) -> CohortRun:
    """Inventory and metrics of the 32-subject cohort, with the default settings."""
    run_dir = tmp_path_factory.mktemp("cohort_run") / "run"
    before = snapshot_tree(synth_cohort.root)
    config = new_local_config(synth_cohort.root)
    with Run(run_dir, local(synth_cohort.root), config=config) as run:
        assert counts(run.inventory()) == (32, 0, 0)
        report = run.metrics()
        uids = labels(run)
    unchanged = snapshot_tree(synth_cohort.root) == before
    return CohortRun(run_dir, config, synth_cohort, report, unchanged, uids)


@pytest.fixture
def cohort_copy(cohort_run: CohortRun, tmp_path: Path) -> Path:
    return Path(shutil.copytree(cohort_run.run_dir, tmp_path / "run"))


# ---------------------------------------------------------------------------------------
# the cohort, scored within the set


def test_the_cohort_gets_every_table_with_its_schema(cohort_run: CohortRun) -> None:
    run_dir, cohort, uids = cohort_run.run_dir, cohort_run.cohort, cohort_run.uids
    # 31 subjects have stats files and all 32 a thickness file; then the two run-level units
    assert counts(cohort_run.report) == (31 + 32 + 2, 0, 0)
    assert cohort_run.report.notes == () and cohort_run.source_unchanged

    regional = table(run_dir, "regional")
    assert set(regional["subject_uid"]) == set(uids.values()) - {uids[cohort.no_stats[0]]}
    assert (regional.groupby(["subject_uid", "hemi"]).size() == 11).all()

    summary = table(run_dir, "vertex_summary")
    assert len(summary) == 64 and set(summary["hemi"]) == set(HEMIS)
    no_sphere = summary["subject_uid"] == uids[cohort.no_sphere_reg[0]]
    assert set(summary.loc[no_sphere, "flags"]) == {"no_sphere_reg"}
    assert summary.loc[no_sphere, "template_id"].isna().all()
    assert set(summary.loc[no_sphere, "region_kind"]) == {"annot"}
    mapped = summary[~no_sphere]
    assert set(mapped["flags"]) == {""} and set(mapped["region_kind"]) == {"template"}
    assert mapped["template_id"].nunique() == 1 and (mapped["n_fallback"] == 0).all()
    assert mapped["global_median_mm"].between(2.0, 3.0).all()  # subjects differ by an offset
    # the annotation-based count is there for every hemisphere, mapped or not
    assert summary["annot_search_n"].notna().all() and (summary["n_native"] == 642).all()
    assert sorted(p.stem for p in (run_dir / "tables" / "fsavg").iterdir()) == sorted(
        set(uids.values()) - {uids[cohort.no_sphere_reg[0]]}
    )

    clusters = table(run_dir, "clusters")
    assert not clusters.empty and clusters["reference_id"].nunique() == 1
    assert (clusters["rank"] <= 5).all() and (clusters["peak_z"] > 3).all()
    assert (clusters["excess_mass"] <= clusters["mass"]).all()
    assert not table_path(run_dir, "clusters_native").exists()  # nothing is rendered


def test_scores_have_one_primary_row_per_subject(cohort_run: CohortRun) -> None:
    cohort, uids = cohort_run.cohort, cohort_run.uids
    scores = table(cohort_run.run_dir, "scores")
    assert len(scores) == 32 and sorted(scores["subject_uid"]) == sorted(uids.values())
    assert scores["primary"].all()
    unmapped = scores["subject_uid"] == uids[cohort.no_sphere_reg[0]]
    (reference_id,) = set(scores.loc[~unmapped, "reference_id"])
    assert reference_id.startswith("set-")
    assert set(scores.loc[unmapped, "reference_id"]) == {"none"}
    assert set(scores["score_kind"]) == {"composite_z"}
    scored = scores[~unmapped]
    # one dataset with 31 mapped subjects: it is its own stratum
    (dataset,) = set(scores["dataset"])
    assert set(scored["stratum"]) == {f"dataset:{dataset}"}
    assert set(scored["n_reference"]) == {31} and scores.loc[unmapped, "n_reference"].isna().all()
    assert sorted(scored["rank"]) == list(range(1, 32))
    by_uid = scores.set_index("subject_uid")
    assert by_uid.loc[uids[cohort.no_sphere_reg[0]], "flags"] == "no_sphere_reg,partial_score"
    assert by_uid.loc[uids[cohort.no_stats[0]], "flags"] == "no_stats,partial_score"
    assert (scores["score_config_hash"].str.len() == 12).all()
    index = read_index(reference_dir(cohort_run.run_dir, reference_id))
    assert index.strata[f"dataset:{dataset}"].n == 31

    with Run(cohort_run.run_dir) as run:
        primary = run.scores()
        assert len(primary) == 32 and primary["reference_id"].iloc[-1] == "none"
        assert list(primary["rank"].iloc[:31]) == list(range(1, 32))
        joined = run.clusters()
        assert len(joined) == len(table(cohort_run.run_dir, "clusters"))
        assert joined["native_vertex"].isna().all()


def test_the_three_bad_subjects_have_the_largest_clusters(cohort_run: CohortRun) -> None:
    """Each of the three subjects with a planted 5 mm patch has a cluster in the search
    region that no other subject comes near, and ranks among the first of the run."""
    cohort, uids = cohort_run.cohort, cohort_run.uids
    bad = {uids[label] for label in cohort.bad}
    scores = table(cohort_run.run_dir, "scores")
    scored = scores[scores["reference_id"] != "none"]
    by_mass = scored.sort_values("mass_search", ascending=False)
    assert set(by_mass["subject_uid"].iloc[:3]) == bad
    assert by_mass["mass_search"].iloc[2] > 5 * by_mass["mass_search"].iloc[3]
    assert set(scored.loc[scored["rank"] <= 3, "subject_uid"]) == bad
    # the patch is at the ceiling on the native surface as well
    assert (scored.set_index("subject_uid").loc[sorted(bad), "ceiling_frac"] > 0.02).all()
    assert (scored.loc[~scored["subject_uid"].isin(bad), "ceiling_frac"] == 0).all()


def test_zmap_of_a_run_equals_a_direct_computation(cohort_run: CohortRun) -> None:
    cohort, run_dir = cohort_run.cohort, cohort_run.run_dir
    label, patch = next(iter(cohort.bad.items()))
    uid = cohort_run.uids[label]
    with Run(run_dir) as run:
        z = run.zmap(uid)
        template = run.template()
        assert template is not None
        (row,) = table(run_dir, "scores").query("subject_uid == @uid").to_dict("records")
        assert run.zmap(uid, row["reference_id"])["lh"].tobytes() == z["lh"].tobytes()
        with pytest.raises(KeyError, match="no z-map"):
            run.zmap(cohort_run.uids[cohort.no_sphere_reg[0]])
        with pytest.raises(KeyError, match="no z-map against set-000000000000"):
            run.zmap(uid, "set-000000000000")
    maps = read_maps(map_path(run_dir, uid))
    reference = read_reference(reference_dir(run_dir, row["reference_id"]))
    for hemi in HEMIS:
        half = template.hemis[hemi]
        ref = reference.strata[row["stratum"]][hemi]
        thickness = maps[hemi].astype(np.float64)
        centred = thickness - global_median(maps[hemi], half) - ref.median.astype(np.float64)
        direct = (centred / (MAD_TO_SD * ref.mad.astype(np.float64))).astype(np.float32)
        np.testing.assert_array_equal(z[hemi], direct)
        assert np.isnan(z[hemi][~half.cortex]).all()
        # above the threshold inside the planted patch, and almost nowhere else
        inside = in_patch(half.sphere, patch) & half.cortex
        assert inside.sum() > 100 and (z[hemi][inside] > 3).mean() > 0.9
        # a few degrees beyond the patch interpolation still carries the planted value
        wider = dataclasses.replace(patch, radius_deg=patch.radius_deg + 4.0)
        outside = half.cortex & ~in_patch(half.sphere, wider)
        assert (z[hemi][outside] > 3).mean() < 0.01


def test_nothing_private_and_no_canary_reaches_the_run_directory(cohort_run: CohortRun) -> None:
    """The stats files and logs carry an accession, a host and a user name in their headers
    and command lines. None of them is in any file of the run."""
    for path in sorted(cohort_run.run_dir.rglob("*")):
        if path.is_file():
            content = path.read_bytes()
            for canary in CANARIES:
                assert canary.encode() not in content, f"{canary} in {path.name}"
    root = str(cohort_run.cohort.root).encode()
    for name, content in run_files(cohort_run.run_dir).items():
        assert root not in content, name
        assert b"sub-0" not in content, name


def test_a_second_run_does_nothing(cohort_copy: Path) -> None:
    before = {p: p.stat().st_mtime_ns for p in sorted(cohort_copy.rglob("*")) if p.is_file()}
    with Run(cohort_copy) as run:
        rows = {stage: ledger(run, stage) for stage in METRIC_STAGES}
        assert counts(run.metrics()) == (0, 65, 0)
        assert rows == {stage: ledger(run, stage) for stage in METRIC_STAGES}
    after = {p: p.stat().st_mtime_ns for p in sorted(cohort_copy.rglob("*")) if p.is_file()}
    # the merged per-dataset tables are written again, with the same bytes; nothing else is
    rewritten = {p.name for p in before if after[p] != before[p]}
    assert rewritten <= {CATALOG_FILE, "regional.parquet", "vertex_summary.parquet"}
    assert set(after) == set(before)


def test_changing_the_cluster_threshold_redoes_the_run_level_stages_only(
    cohort_run: CohortRun, cohort_copy: Path
) -> None:
    lower = cohort_run.config.metrics.model_dump() | {"cluster_z": 2.5}
    config = with_changes(cohort_run.config, metrics=lower)
    was = run_files(cohort_copy)
    with Run(cohort_copy, config=config) as run:
        # the scores alone cannot be made: their clusters are those of the other threshold.
        # That is said and not recorded, so the full call that follows is not held up by it
        alone = run.metrics(stages=["score"])
        assert counts(alone) == (0, 0, 1)
        assert alone.failures == (("run", "score: reference not current"),)
        assert run_files(cohort_copy) == was
        assert counts(run.metrics()) == (2, 63, 0)
        assert counts(run.metrics()) == (0, 65, 0)
    now = run_files(cohort_copy)
    changed = {name for name in set(was) | set(now) if was.get(name) != now.get(name)}
    # a new reference under a new id, the clusters and the scores; no map and no part
    kinds = {name.split("/")[1] for name in changed}
    assert kinds == {"reference", "clusters.parquet", "scores.parquet"}
    (old_id,) = {name.split("/")[2] for name in was if name.startswith("tables/reference/")}
    (new_id,) = {name.split("/")[2] for name in now if name.startswith("tables/reference/")}
    assert old_id != new_id
    clusters = table(cohort_copy, "clusters")
    assert set(clusters["reference_id"]) == {new_id} and clusters["peak_z"].min() < 3
    assert len(clusters) > len(table(cohort_run.run_dir, "clusters"))


def test_a_dataset_filter_runs_the_per_subject_stages_only(cohort_copy: Path) -> None:
    with Run(cohort_copy) as run:
        (dataset,) = run.public().datasets()["pseudonym"]
        before = {stage: ledger(run, stage) for stage in METRIC_STAGES}
        report = run.metrics(datasets=[dataset], force=True)
        assert counts(report) == (63, 0, 0)
        after = {stage: ledger(run, stage) for stage in METRIC_STAGES}
        assert after["reference"] == before["reference"] and after["score"] == before["score"]
        assert all(
            after["fsaverage"][uid] >= before["fsaverage"][uid] for uid in after["fsaverage"]
        )
        # named stages run whatever the filter, and the run-level ones ignore it
        named = run.metrics(datasets=[dataset], stages=["score", "reference"])
        assert counts(named) == (0, 2, 0)
        with pytest.raises(ConfigError, match="D-00000000"):
            run.metrics(datasets=["D-00000000"])


def test_the_stage_functions_of_the_pipeline_work_on_a_context(cohort_copy: Path) -> None:
    with Run(cohort_copy) as run:
        template = run.template()
        ctx = run.context()
        assert ALL_METRIC_STAGES == (*METRIC_STAGES, "native", "merge")
        with pytest.raises(ConfigError, match="no metric stage clusters"):
            run_metrics(ctx, template=template, stages=["clusters"])
        was = run_files(cohort_copy)
        assert counts(merge_tables(ctx)) == (2, 0, 0)
        # nothing is rendered: the native stage writes an empty table and is then skipped
        assert counts(update_native(ctx, template=template)) == (1, 0, 0)
        assert counts(update_native(ctx, template=template)) == (0, 1, 0)
        assert table(cohort_copy, "clusters_native").empty
        now = run_files(cohort_copy)
        assert {name for name in now if now[name] != was.get(name)} == {
            "tables/clusters_native.parquet"
        }
        # a stage named alone, in any spelling of the list
        assert counts(run.metrics(stages=["merge"])) == (2, 0, 0)
        assert counts(run.metrics(stages=("score",))) == (0, 1, 0)


# ---------------------------------------------------------------------------------------
# small cohorts: too few subjects, no template, a template that has gone


def test_too_few_subjects_are_ranked_as_decision_d16_says(
    small_cohort: SyntheticCohort, tmp_path: Path
) -> None:
    with Run(tmp_path / "run", local(small_cohort.root)) as run:
        run.inventory()
        report = run.metrics()
        assert counts(report) == (5 + 5 + 2, 0, 0)
        assert report.notes == (
            "5 subjects with a map: too few for per-location scoring (30 needed). Ranked on "
            "raw ceiling fraction and occipital-to-control thickness.",
        )
        scores = run.scores()
        uids = labels(run)
    assert set(scores["score_kind"]) == {"fallback_rank"}
    assert set(scores["reference_id"]) == {"none"} and scores["primary"].all()
    assert list(scores["rank"]) == [1, 2, 3, 4, 5] and scores["composite"].notna().all()
    assert scores["stratum"].isna().all() and scores["n_reference"].isna().all()
    # the subject with the patch is the only one with vertices at the ceiling
    (bad,) = small_cohort.bad
    assert scores["subject_uid"].iloc[0] == uids[bad]
    assert scores["composite_component"].iloc[0] == "ceiling_frac"
    assert scores["mass_search"].isna().all() and scores["template_id"].notna().all()
    run_dir = tmp_path / "run"
    assert len(list((run_dir / "tables" / "fsavg").iterdir())) == 5  # the maps are made
    assert not table_path(run_dir, "clusters").exists()
    assert not (run_dir / "tables" / "reference").exists()


def test_without_a_template_the_message_comes_once_and_the_scores_fall_back(
    synth_template: SyntheticTemplate, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    cohort = little_cohort(tmp_path / "subjects", 3, synth_template, with_template=False)
    run_dir = tmp_path / "run"
    with Run(run_dir, local(cohort.root), config=new_local_config(cohort.root)) as run:
        run.inventory()
        with caplog.at_level(logging.INFO, logger="surfcensus"):
            report = run.metrics()
            assert counts(run.metrics()) == (0, 3 + 3 + 1, 0)
        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1
        lines = warnings[0].splitlines()
        assert lines[0] == "No fsaverage template found. Looked in:"
        assert lines[1].split() == ["$FREESURFER_HOME/subjects/fsaverage", "(not", "set)"]
        assert "To enable: pass --fsaverage /path/to/fsaverage" in lines[-1]
        assert run.template() is None and run.public().meta("template_id") is None
        scores = run.scores()
    # regional and fsaverage per subject, and the scores; the reference stage has no unit
    assert counts(report) == (3 + 3 + 1, 0, 0)
    assert len(report.notes) == 1 and report.notes[0].startswith("No fsaverage template:")
    assert set(scores["score_kind"]) == {"fallback_rank"} and scores["composite"].notna().all()
    assert scores["template_id"].isna().all() and set(scores["ceiling_region"]) == {"annot"}
    summary = table(run_dir, "vertex_summary")
    assert set(summary["flags"]) == {"no_template"} and summary["ceiling_frac"].notna().all()
    assert (summary["ceiling_frac"] == summary["ceiling_frac_annot"]).all()
    assert not table_path(run_dir, "clusters").exists()
    assert not (run_dir / "tables" / "fsavg").exists() and not (run_dir / "templates").exists()


def test_a_pinned_template_that_has_gone_raises_and_leaves_the_scores(
    synth_template: SyntheticTemplate, tmp_path: Path
) -> None:
    cohort = little_cohort(tmp_path / "subjects", 3, synth_template)
    run_dir = tmp_path / "run"
    config = new_local_config(cohort.root, metrics=SET_OF_THREE)
    with Run(run_dir, local(cohort.root), config=config) as run:
        run.inventory()
        assert counts(run.metrics()) == (3 + 3 + 2, 0, 0)
        template = run.template()
        assert template is not None and template.origin == "adjacent"
        assert run.public().meta("template_id") == template.template_id
    was = run_files(run_dir)
    assert f"templates/{template.template_id}/template.json" in was

    # the cache alone is enough: the fsaverage next to the subjects may go
    (cohort.root / "fsaverage").unlink()
    with Run(run_dir) as run:
        assert counts(run.metrics()) == (0, 8, 0)
    # without the cache the template is built again from where it was found
    shutil.rmtree(run_dir / "templates")
    (cohort.root / "fsaverage").symlink_to(synth_template.path, target_is_directory=True)
    with Run(run_dir) as run:
        assert counts(run.metrics()) == (0, 8, 0)
    assert run_files(run_dir) == was

    # with neither, a run that has scores does not fall back to the ranking of D16
    shutil.rmtree(run_dir / "templates")
    (cohort.root / "fsaverage").unlink()
    scores = table_path(run_dir, "scores").read_bytes()
    with Run(run_dir) as run:
        with pytest.raises(TemplateError, match="no longer available") as raised:
            run.metrics()
        assert raised.value.reason == "changed"
        with pytest.raises(TemplateError, match="no longer available"):
            run.render()
    assert table_path(run_dir, "scores").read_bytes() == scores


def test_adding_a_subject_redoes_the_set_level_stages_only(
    synth_template: SyntheticTemplate, tmp_path: Path
) -> None:
    cohort = little_cohort(tmp_path / "subjects", 4, synth_template)
    run_dir = tmp_path / "run"
    config = new_local_config(cohort.root, metrics=SET_OF_THREE)
    first = LocalSource(cohort.root, subjects=("sub-000", "sub-001", "sub-002"))
    with Run(run_dir, first, config=config) as run:
        run.inventory()
        assert counts(run.metrics()) == (3 + 3 + 2, 0, 0)
        (reference_of_three,) = set(run.scores()["reference_id"])
        per_subject = {stage: ledger(run, stage) for stage in ("regional", "fsaverage")}
    maps = {p.name: p.read_bytes() for p in (run_dir / "tables" / "fsavg").iterdir()}

    with Run(run_dir, local(cohort.root)) as run:
        assert counts(run.inventory()) == (1, 3, 0)
        assert counts(run.metrics()) == (1 + 1 + 2, 3 + 3, 0)
        for stage, rows in per_subject.items():
            assert {unit: at for unit, at in ledger(run, stage).items() if unit in rows} == rows
        scores = run.scores()
    assert len(scores) == 4 and set(scores["n_reference"]) == {4}
    (reference_of_four,) = set(scores["reference_id"])
    assert reference_of_four != reference_of_three
    # the earlier reference is replaced, and the earlier maps are as they were
    assert [p.name for p in (run_dir / "tables" / "reference").iterdir()] == [reference_of_four]
    now = {p.name: p.read_bytes() for p in (run_dir / "tables" / "fsavg").iterdir()}
    assert len(now) == 4 and {name: now[name] for name in maps} == maps


def test_two_worker_processes_write_the_bytes_of_one(
    synth_template: SyntheticTemplate, tmp_path: Path
) -> None:
    cohort = little_cohort(tmp_path / "subjects", 4, synth_template)
    config = new_local_config(cohort.root, metrics=SET_OF_THREE)  # one salt: the same uids
    written = []
    for jobs in (1, 2):
        run_dir = tmp_path / f"run{jobs}"
        with Run(run_dir, local(cohort.root), config=config) as run:
            run.inventory()
            assert counts(run.metrics(jobs=jobs)) == (4 + 4 + 2, 0, 0)
        written.append(run_files(run_dir))
    assert written[0].keys() == written[1].keys()
    assert [name for name in written[0] if written[0][name] != written[1][name]] == []
    assert {name.split("/")[1] for name in written[0]} >= {*(f"{t}.parquet" for t in TABLES)}


# ---------------------------------------------------------------------------------------
# failures


def catalog_failures(run_dir: Path) -> list[tuple[str, str, str, str]]:
    with Catalog(run_dir / CATALOG_FILE, readonly=True) as catalog:
        frame = catalog.failures()
    return list(frame.itertuples(index=False, name=None))


def test_an_unreadable_file_fails_its_subject_and_no_other(
    synth_template: SyntheticTemplate, tmp_path: Path
) -> None:
    cohort = little_cohort(tmp_path / "subjects", 3, synth_template)
    damaged = cohort.subjects[0].path / "surf" / "lh.thickness"
    whole = damaged.read_bytes()
    damaged.write_bytes(whole[: len(whole) // 2])
    stats = cohort.subjects[2].path / "stats" / "rh.aparc.stats"
    stats.write_text("not a stats file\n")
    run_dir = tmp_path / "run"
    config = new_local_config(cohort.root, metrics=SET_OF_THREE)
    with Run(run_dir, local(cohort.root), config=config) as run:
        run.inventory()
        uids = labels(run)
        report = run.metrics()
        assert counts(report) == (2 + 2 + 2, 0, 2)
        assert report.failures == (  # by stage, then unit
            (uids["sub-000"], "fsaverage: unreadable"),
            (uids["sub-002"], "regional: unreadable"),
        )
        failures = catalog_failures(run_dir)
        assert [(unit, phase, reason) for unit, phase, reason, _ in failures] == [
            (uids["sub-000"], "fsaverage", "unreadable"),
            (uids["sub-002"], "regional", "unreadable"),
        ]
        # the detail names the file and never its path
        assert failures[0][3].startswith("lh.thickness") and "/" not in failures[0][3]
        assert failures[1][3].startswith("rh.aparc.stats") and "/" not in failures[1][3]
        # the failed subject has no rows and no map; two subjects are too few for a reference
        assert uids["sub-000"] not in set(table(run_dir, "vertex_summary")["subject_uid"])
        assert not map_path(run_dir, uids["sub-000"]).exists()
        assert uids["sub-002"] not in set(table(run_dir, "regional")["subject_uid"])
        scores = run.scores()
        assert len(scores) == 3 and set(scores["score_kind"]) == {"fallback_rank"}
        flags = dict(zip(scores["subject_uid"], scores["flags"], strict=True))
        assert flags[uids["sub-000"]] == "no_thickness,partial_score"
        assert flags[uids["sub-002"]] == "no_stats,partial_score"

        # not tried again, and counted again
        again = run.metrics()
        assert counts(again) == (0, 2 + 2 + 2, 2) and again.failures == report.failures
        retried = run.metrics(retry_failed=True)
        assert counts(retried) == (0, 2 + 2 + 2, 2)

        # the files are repaired: the inventory sees other files, and the metrics follow
        damaged.write_bytes(whole)
        shutil.copyfile(cohort.subjects[1].path / "stats" / "rh.aparc.stats", stats)
        assert run.inventory().done == 2
        repaired = run.metrics()
        assert counts(repaired) == (1 + 1 + 2, 2 + 2, 0)
        assert catalog_failures(run_dir) == []
        assert set(run.scores()["score_kind"]) == {"composite_z"}


def test_a_file_that_changed_since_the_inventory_is_source_changed(
    synth_template: SyntheticTemplate, tmp_path: Path
) -> None:
    cohort = little_cohort(tmp_path / "subjects", 3, synth_template)
    run_dir = tmp_path / "run"
    with Run(run_dir, local(cohort.root), config=new_local_config(cohort.root)) as run:
        run.inventory()
        uids = labels(run)
        annot = cohort.subjects[1].path / "label" / "lh.aparc.annot"
        annot.write_bytes(annot.read_bytes() + b"\0")
        report = run.metrics()
        assert report.failures == ((uids["sub-001"], "fsaverage: source changed"),)
        assert counts(report) == (3 + 2 + 2, 0, 1)


def test_a_hemisphere_that_cannot_be_mapped_gets_a_flag_and_the_subject_goes_on(
    synth_template: SyntheticTemplate, tmp_path: Path
) -> None:
    """One hemisphere without ``sphere.reg``, one whose annotation is of another subject."""
    cohort = little_cohort(tmp_path / "subjects", 3, synth_template)
    (cohort.subjects[0].path / "surf" / "rh.sphere.reg").unlink()
    other = cohort.subjects[2].path / "label" / "lh.aparc.annot"
    annot_labels, ctab, names = fs.read_annot(str(other))
    fs.write_annot(str(other), np.asarray(annot_labels[:-7]), ctab, names, fill_ctab=True)
    run_dir = tmp_path / "run"
    config = new_local_config(cohort.root, metrics=SET_OF_THREE)
    with Run(run_dir, local(cohort.root), config=config) as run:
        run.inventory()
        uids = labels(run)
        assert counts(run.metrics()) == (3 + 3 + 2, 0, 0)
        scores = run.scores().set_index("subject_uid")
    summary = table(run_dir, "vertex_summary").set_index(["subject_uid", "hemi"])
    assert summary.loc[(uids["sub-000"], "rh"), "flags"] == "no_sphere_reg"
    assert summary.loc[(uids["sub-000"], "lh"), "flags"] == ""
    assert summary.loc[(uids["sub-002"], "lh"), "flags"] == "vertex_count_mismatch"
    assert summary.loc[(uids["sub-002"], "lh"), "region_kind"] == "none"
    assert pd.isna(summary.loc[(uids["sub-002"], "lh"), "template_id"])
    # each still has the map of its other hemisphere, all NaN on the one that failed
    maps = read_maps(map_path(run_dir, uids["sub-000"]))
    assert np.isnan(maps["rh"]).all() and np.isfinite(maps["lh"]).sum() > 30_000
    assert scores.loc[uids["sub-000"], "flags"] == "no_sphere_reg"
    assert scores.loc[uids["sub-002"], "flags"] == "vertex_count_mismatch"
    assert set(scores["reference_id"]) != {"none"} and scores["composite"].notna().all()


# ---------------------------------------------------------------------------------------
# what the run holds of a subject whose files went away, and outputs that were damaged


def test_a_subject_that_loses_its_stats_or_thickness_loses_its_rows_and_map(
    synth_template: SyntheticTemplate, tmp_path: Path
) -> None:
    """Without this the scores kept a ratio, or a ceiling fraction, of files that were no
    longer there."""
    cohort = little_cohort(tmp_path / "subjects", 4, synth_template)
    run_dir = tmp_path / "run"
    config = new_local_config(cohort.root, metrics=SET_OF_THREE)
    kept = tmp_path / "kept"
    kept.mkdir()
    stats = cohort.subjects[2].path / "stats"
    thickness = [cohort.subjects[3].path / "surf" / f"{hemi}.thickness" for hemi in HEMIS]
    with Run(run_dir, local(cohort.root), config=config) as run:
        run.inventory()
        assert counts(run.metrics()) == (4 + 4 + 2, 0, 0)
        uids = labels(run)
        lost_stats, lost_thickness = uids["sub-002"], uids["sub-003"]
        was = {name: table(run_dir, name) for name in ("regional", "vertex_summary", "scores")}
        map_was = map_path(run_dir, lost_thickness).read_bytes()

        # moved, not deleted: brought back below, the files are listed as they were
        stats.rename(kept / "stats")
        for path in thickness:
            path.rename(kept / path.name)
        assert run.inventory().done == 2
        # no subject is read: the two lost their rows, and the run-level units follow
        assert counts(run.metrics()) == (2, 3 + 3, 0)
        assert lost_stats not in set(table(run_dir, "regional")["subject_uid"])
        assert lost_thickness not in set(table(run_dir, "vertex_summary")["subject_uid"])
        assert not map_path(run_dir, lost_thickness).exists()
        assert lost_stats not in set(run.public().work("regional")["unit"])
        assert lost_thickness not in set(run.public().work("fsaverage")["unit"])
        scores = run.scores().set_index("subject_uid")
        assert scores.loc[lost_stats, "flags"] == "no_stats,partial_score"
        assert np.isnan(scores.loc[lost_stats, "occ_ctrl_ratio"])
        assert np.isnan(scores.loc[lost_stats, "occ_thick_mm"])
        assert scores.loc[lost_thickness, "flags"] == "no_thickness,partial_score"
        assert np.isnan(scores.loc[lost_thickness, "ceiling_frac"])
        # the other subject of each stage keeps what it had
        assert lost_thickness in set(table(run_dir, "regional")["subject_uid"])
        assert map_path(run_dir, lost_stats).is_file()
        assert counts(run.metrics()) == (0, 3 + 3 + 2, 0)

        # The files come back with the listing they had. A ledger row left from before
        # would say that the two are done, with their rows gone.
        (kept / "stats").rename(stats)
        for path in thickness:
            (kept / path.name).rename(path)
        assert run.inventory().done == 2
        assert counts(run.metrics()) == (1 + 1 + 2, 3 + 3, 0)
        for name, frame in was.items():
            pd.testing.assert_frame_equal(table(run_dir, name), frame)
        assert map_path(run_dir, lost_thickness).read_bytes() == map_was


def test_a_damaged_template_cache_is_written_again_and_fails_no_subject(
    synth_template: SyntheticTemplate, tmp_path: Path
) -> None:
    """Worker processes read the template from the run directory. A cache that does not
    read back used to fail every subject they worked on and to remove its earlier map."""
    cohort = little_cohort(tmp_path / "subjects", 4, synth_template)
    run_dir = tmp_path / "run"
    config = new_local_config(cohort.root, metrics=SET_OF_THREE)
    with Run(run_dir, local(cohort.root), config=config) as run:
        run.inventory()
        run.metrics()
        template_id = run.public().meta("template_id")
    assert template_id is not None
    files = run_files(run_dir)
    cached = run_dir / "templates" / template_id / "template.npz"
    cached.write_bytes(cached.read_bytes()[:1000])
    with Run(run_dir) as run:
        assert counts(run.metrics(force=True, jobs=2)) == (4 + 4 + 2, 0, 0)
    assert run_files(run_dir) == files


def test_a_damaged_map_stops_the_phase_and_is_made_again(
    synth_template: SyntheticTemplate, tmp_path: Path
) -> None:
    cohort = little_cohort(tmp_path / "subjects", 4, synth_template)
    run_dir = tmp_path / "run"
    config = new_local_config(cohort.root, metrics=SET_OF_THREE)
    with Run(run_dir, local(cohort.root), config=config) as run:
        run.inventory()
        run.metrics()
        uids = labels(run)
        files = run_files(run_dir)
        truncated, garbage = map_path(run_dir, uids["sub-001"]), map_path(run_dir, uids["sub-002"])
        truncated.write_bytes(truncated.read_bytes()[:300])
        garbage.write_bytes(b"not an archive")
        # nothing reads a map while nothing is pending
        assert counts(run.metrics()) == (0, 4 + 4 + 2, 0)
        with pytest.raises(ConfigError, match="could not be read") as raised:
            run.metrics(stages=["reference"], force=True)
        # a fault of the run: one error that names both subjects, and no failed unit
        assert raised.value.reason == "invalid"
        assert uids["sub-001"] in str(raised.value) and uids["sub-002"] in str(raised.value)
        assert "/" not in raised.value.detail.replace("tables/fsavg", "")
        assert not truncated.exists() and not garbage.exists()
        assert catalog_failures(run_dir) == []
        # the same command makes the two maps again, with the bytes they had
        assert counts(run.metrics()) == (2, 4 + 2 + 2, 0)
    assert run_files(run_dir) == files


# ---------------------------------------------------------------------------------------
# what a subject needs to be scored, and what its rows trace back to


def test_a_subject_with_one_stats_file_gets_its_rows_and_its_ratio(
    synth_template: SyntheticTemplate, tmp_path: Path
) -> None:
    """``tier_a`` asks for both stats files. The regional stage asks for one."""
    cohort = little_cohort(tmp_path / "subjects", 3, synth_template)
    (cohort.subjects[0].path / "stats" / "rh.aparc.stats").unlink()
    for name in ("lh.aparc.stats", "rh.aparc.stats"):
        (cohort.subjects[2].path / "stats" / name).unlink()
    run_dir = tmp_path / "run"
    config = new_local_config(cohort.root, metrics=SET_OF_THREE)
    with Run(run_dir, local(cohort.root), config=config) as run:
        run.inventory()
        uids = labels(run)
        subjects = run.public().subjects().set_index("subject_uid")
        assert not subjects.loc[uids["sub-000"], "tier_a"]
        # two subjects have a stats file, all three a thickness file
        assert counts(run.metrics()) == (2 + 3 + 2, 0, 0)
        assert counts(run.metrics()) == (0, 2 + 3 + 2, 0)
        scores = run.scores().set_index("subject_uid")
    regional = table(run_dir, "regional")
    assert set(regional["subject_uid"]) == {uids["sub-000"], uids["sub-001"]}
    one = regional[regional["subject_uid"] == uids["sub-000"]]
    assert set(one["hemi"]) == {"lh"} and len(one) == 11
    occipital, control, ratio = occ_ctrl_ratio(one, config.metrics.regions)
    row = scores.loc[uids["sub-000"]]
    assert row["occ_ctrl_ratio"] == pytest.approx(ratio, rel=1e-6)
    assert row["occ_thick_mm"] == pytest.approx(occipital, rel=1e-6)
    assert row["ctrl_thick_mm"] == pytest.approx(control, rel=1e-6)
    assert row["flags"] == ""
    # without any stats file there is nothing to score on, and the row says so
    assert np.isnan(scores.loc[uids["sub-002"], "occ_ctrl_ratio"])
    assert scores.loc[uids["sub-002"], "flags"] == "no_stats,partial_score"


METRIC_ROLES = tuple(
    f"{hemi}_{kind}" for hemi in HEMIS for kind in ("stats", "thickness", "annot", "sphere_reg")
)
ROLE_FILES = {
    "stats": "stats/{hemi}.aparc.stats",
    "thickness": "surf/{hemi}.thickness",
    "annot": "label/{hemi}.aparc.annot",
    "sphere_reg": "surf/{hemi}.sphere.reg",
}


def stored_digests(run_dir: Path, uid: str) -> dict[str, str | None]:
    """Role -> ``files.sha256`` of one subject."""
    with Catalog(run_dir / CATALOG_FILE, readonly=True) as catalog:
        rows = catalog.query("SELECT role, sha256 FROM files WHERE subject_uid = ?", (uid,))
    return {row["role"]: row["sha256"] for row in rows}


def digest_of(subject: Path, role: str) -> str:
    hemi, kind = role.split("_", 1)
    content = (subject / ROLE_FILES[kind].format(hemi=hemi)).read_bytes()
    return hashlib.sha256(content).hexdigest()


def test_the_files_that_the_metrics_read_get_their_digest_in_the_catalog(
    synth_template: SyntheticTemplate, tmp_path: Path
) -> None:
    cohort = little_cohort(tmp_path / "subjects", 3, synth_template)
    whole, partial = cohort.subjects[1].path, cohort.subjects[0].path
    # without its thickness the right hemisphere is not looked at: nothing of it is read
    (partial / "surf" / "rh.thickness").unlink()
    (partial / "stats" / "rh.aparc.stats").unlink()
    damaged = cohort.subjects[2].path / "surf" / "lh.sphere.reg"
    damaged.write_bytes(damaged.read_bytes()[:1000])
    run_dir = tmp_path / "run"
    config = new_local_config(cohort.root, metrics=SET_OF_THREE)
    with Run(run_dir, local(cohort.root), config=config) as run:
        run.inventory()
        uids = labels(run)
        # the inventory reads the stamp and the logs, and nothing that the metrics read
        assert all(stored_digests(run_dir, uids["sub-001"])[role] is None for role in METRIC_ROLES)
        report = run.metrics()
        assert report.failures == ((uids["sub-002"], "fsaverage: unreadable"),)

        stored = stored_digests(run_dir, uids["sub-001"])
        assert {role: stored[role] for role in METRIC_ROLES} == {
            role: digest_of(whole, role) for role in METRIC_ROLES
        }
        # the metrics read no other file of the subject
        read_by_inventory = {"build_stamp", "recon_log", "recon_log_lh", "recon_log_rh"}
        others = set(stored) - set(METRIC_ROLES) - read_by_inventory
        assert others and all(stored[role] is None for role in others)

        stored = stored_digests(run_dir, uids["sub-000"])
        left = [role for role in METRIC_ROLES if role.startswith("lh_")]
        assert {role: stored[role] for role in left} == {
            role: digest_of(partial, role) for role in left
        }
        assert stored["rh_annot"] is None and stored["rh_sphere_reg"] is None
        assert "rh_thickness" not in stored and "rh_stats" not in stored

        # a subject whose stage failed has no rows from it, and no digest either
        stored = stored_digests(run_dir, uids["sub-002"])
        assert stored["lh_stats"] == digest_of(cohort.subjects[2].path, "lh_stats")
        assert stored["lh_thickness"] is None and stored["rh_sphere_reg"] is None

        # an inventory that finds the same files keeps the digests, and nothing is redone
        assert counts(run.inventory()) == (0, 3, 0)
        assert counts(run.metrics()) == (0, 3 + 2 + 2, 1)
        assert stored_digests(run_dir, uids["sub-001"])["lh_sphere_reg"] == digest_of(
            whole, "lh_sphere_reg"
        )


def test_without_a_template_the_sphere_is_not_read_and_gets_no_digest(
    synth_template: SyntheticTemplate, tmp_path: Path
) -> None:
    cohort = little_cohort(tmp_path / "subjects", 3, synth_template, with_template=False)
    run_dir = tmp_path / "run"
    with Run(run_dir, local(cohort.root), config=new_local_config(cohort.root)) as run:
        run.inventory()
        uids = labels(run)
        assert counts(run.metrics(jobs=2)) == (3 + 3 + 1, 0, 0)
    subject = cohort.subjects[0].path
    stored = stored_digests(run_dir, uids["sub-000"])
    for role in METRIC_ROLES:
        expected = None if role.endswith("sphere_reg") else digest_of(subject, role)
        assert stored[role] == expected, role


# ---------------------------------------------------------------------------------------
# with renders: counterparts, native centroids


def test_the_rendered_run_has_counterparts_and_native_centroids(
    small_run: Path, small_run_cohort: SyntheticCohort
) -> None:
    with Run(small_run) as run:
        uids = labels(run)
        template = run.template()
        assert template is not None
        renders = run.public().renders()
        assert set(renders["status"]) == {"done"}
        assert set(renders["template_id"]) == {template.template_id}
        joined = run.clusters()
        scores = run.scores()
    assert set(scores["score_kind"]) == {"composite_z"} and len(scores) == 3
    for uid in uids.values():
        vmap = read_vertex_map(small_run / "media" / uid / "vertex_map.npz")
        for hemi in HEMIS:
            assert vmap[f"{hemi}_fsavg6"].min() >= 0
            assert vmap[f"{hemi}_fsavg6"].max() < 40_962

    native = table(small_run, "clusters_native")
    assert not native.empty and set(native["template_id"]) == {template.template_id}
    assert len(joined) == len(table(small_run, "clusters"))
    assert joined["native_vertex"].notna().sum() == len(native)
    # the subject with the patch: the centroid of its largest medial cluster lies on its own
    # pial surface, at a vertex of the vertex map, inside the patch
    (bad,) = small_run_cohort.bad
    rows = native[(native["subject_uid"] == uids[bad]) & (native["region"] == "medial")]
    assert not rows.empty
    row = rows.sort_values(["hemi", "rank"]).iloc[0]
    vmap = read_vertex_map(small_run / "media" / uids[bad] / "vertex_map.npz")
    position = np.flatnonzero(vmap[f"{row['hemi']}_index"] == row["native_vertex"])
    assert position.size == 1
    np.testing.assert_array_equal(
        vmap[f"{row['hemi']}_pial"][position[0]], row[["tkr_x", "tkr_y", "tkr_z"]].to_numpy("f4")
    )
    # a control cluster lies in frontal cortex, which no tile shows: it has no native row
    assert "control" not in set(native["region"])


def test_native_fills_its_table_again_and_leaves_the_clusters(run_copy: Path) -> None:
    clusters = table_path(run_copy, "clusters").read_bytes()
    native = table_path(run_copy, "clusters_native")
    was = native.read_bytes()
    native.unlink()
    with Run(run_copy) as run:
        assert counts(run.metrics(stages=["native"])) == (1, 0, 0)
        assert native.read_bytes() == was
        assert counts(run.metrics(stages=["native"])) == (0, 1, 0)
        # every render is recorded and whole: rendering again does nothing, native included
        assert counts(run.render()) == (0, 3, 0)
    assert native.read_bytes() == was and table_path(run_copy, "clusters").read_bytes() == clusters


def test_a_damaged_vertex_map_leaves_its_subject_without_native_rows(run_copy: Path) -> None:
    native = table(run_copy, "clusters_native")
    uid = str(native["subject_uid"].iloc[0])
    others = native[native["subject_uid"] != uid].reset_index(drop=True)
    assert len(others)
    vertex_map = run_copy / "media" / uid / "vertex_map.npz"
    whole = vertex_map.read_bytes()
    vertex_map.write_bytes(whole[:500])
    with Run(run_copy) as run:
        report = run.metrics(stages=["native"], force=True)
        # no unit fails; a note names the subject and says what to do
        assert counts(report) == (1, 0, 0)
        (note,) = report.notes
        assert uid in note and "surfcensus render" in note and str(run_copy) not in note
        pd.testing.assert_frame_equal(table(run_copy, "clusters_native"), others)
        # every later call reads the file again, a render with nothing to render included
        rendered = run.render()
        assert counts(rendered) == (0, 3, 0) and rendered.notes == report.notes
        # what the note says to do: without the file the render is not whole and is made again
        vertex_map.unlink()
        repaired = run.render()
        assert counts(repaired) == (1, 2, 0) and repaired.notes == ()
        assert vertex_map.read_bytes() == whole
        pd.testing.assert_frame_equal(table(run_copy, "clusters_native"), native)
        assert counts(run.metrics(stages=["native"])) == (0, 1, 0)


def test_a_template_that_arrives_later_updates_the_vertex_maps_only(
    synth_template: SyntheticTemplate, tmp_path: Path
) -> None:
    cohort = little_cohort(tmp_path / "subjects", 3, synth_template, with_template=False)
    run_dir = tmp_path / "run"
    config = new_local_config(cohort.root, render=RenderConfig(panel_px=64), metrics=SET_OF_THREE)
    with Run(run_dir, local(cohort.root), config=config) as run:
        run.inventory()
        assert counts(run.render()) == (3, 0, 0)
        keys = dict(
            zip(*(run.public().renders()[c] for c in ("subject_uid", "render_key")), strict=True)
        )
        assert run.public().renders()["template_id"].isna().all()
    stamps = {
        p: p.stat().st_mtime_ns for p in sorted((run_dir / "media").rglob("*")) if p.is_file()
    }
    one = next(iter(keys))
    assert read_vertex_map(run_dir / "media" / one / "vertex_map.npz")["lh_fsavg6"].max() == -1

    (cohort.root / "fsaverage").symlink_to(synth_template.path, target_is_directory=True)
    with Run(run_dir) as run:
        run.inventory()  # the listing now says that there is a template next to the subjects
        assert counts(run.metrics()) == (3 + 3 + 2, 0, 0)
        assert counts(run.render()) == (3, 0, 0)
        renders = run.public().renders()
        assert dict(zip(renders["subject_uid"], renders["render_key"], strict=True)) == keys
        assert renders["template_id"].notna().all()
        assert counts(run.render()) == (0, 3, 0)
    rewritten = {p.name for p, at in stamps.items() if p.stat().st_mtime_ns != at}
    assert rewritten == {"render.json", "vertex_map.npz"}
    assert read_vertex_map(run_dir / "media" / one / "vertex_map.npz")["lh_fsavg6"].min() >= 0
    # the renders came first here, and the call that rendered them placed the centroids
    assert not table(run_dir, "clusters_native").empty


def test_a_finished_run_renders_nothing_while_its_data_is_away(
    synth_template: SyntheticTemplate, tmp_path: Path
) -> None:
    """A render with nothing pending reads no source. It does read the run's template, which
    the run directory holds; without that too it stops, as the metrics do."""
    cohort = little_cohort(tmp_path / "subjects", 3, synth_template)
    run_dir = tmp_path / "run"
    config = new_local_config(cohort.root, render=RenderConfig(panel_px=64), metrics=SET_OF_THREE)
    with Run(run_dir, local(cohort.root), config=config) as run:
        run.inventory()
        run.metrics()
        assert counts(run.render()) == (3, 0, 0)
        positions = int(run.clusters()["native_vertex"].notna().sum())
        assert positions > 0
    cohort.root.rename(tmp_path / "unmounted")
    with Run(run_dir) as run:
        assert counts(run.render()) == (0, 3, 0)
        assert counts(run.metrics()) == (0, 3 + 3 + 2, 0)
        assert int(run.clusters()["native_vertex"].notna().sum()) == positions
    shutil.rmtree(run_dir / "templates")
    with Run(run_dir) as run:
        with pytest.raises(TemplateError, match="no longer available") as raised:
            run.render()
        assert raised.value.reason == "changed"
        assert set(run.public().renders()["status"]) == {"done"}


# ---------------------------------------------------------------------------------------
# a shipped reference


def as_shipped(reference: Reference) -> Reference:
    """The maps of a set reference as a shipped one would hold them: one stratum for all,
    no members, no template id."""
    (stratum,) = reference.strata
    return Reference(
        reference_id="shipped-0.test",
        kind="shipped",
        mesh=reference.mesh,
        template_id=None,
        sphere_sha256=reference.sphere_sha256,
        config_hash=reference.config_hash,
        strata={"all": reference.strata[stratum]},
        members={},
        metric_norms={},
        parcel_norms={},
    )


def test_a_shipped_reference_is_scored_beside_the_set_and_is_primary_in_a_local_run(
    run_copy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (set_id,) = [p.name for p in (run_copy / "tables" / "reference").iterdir()]
    built = read_reference(reference_dir(run_copy, set_id))
    write_reference(tmp_path / "shipped", as_shipped(built))
    monkeypatch.setattr(load, "SHIPPED_DIR", tmp_path / "shipped")
    was = table(run_copy, "scores")
    with Run(run_copy) as run:
        assert counts(run.metrics()) == (2, 6, 0)
        scores = table(run_copy, "scores")
        assert len(scores) == 6 and set(scores["reference_id"]) == {set_id, "shipped-0.test"}
        primary = scores[scores["primary"]]
        assert set(primary["reference_id"]) == {"shipped-0.test"} and len(primary) == 3
        assert set(primary["stratum"]) == {"all"} and set(primary["n_reference"]) == {3}
        clusters = table(run_copy, "clusters")
        assert set(clusters["reference_id"]) == {set_id, "shipped-0.test"}
        # the same maps give the same z: the two sets of rows agree
        columns = [c for c in was.columns if c.startswith(("mass_", "composite", "std_", "rank"))]
        for reference_id in (set_id, "shipped-0.test"):
            rows = scores[scores["reference_id"] == reference_id].reset_index(drop=True)
            pd.testing.assert_frame_equal(rows[columns], was[columns])
        uid = scores["subject_uid"].iloc[0]
        np.testing.assert_array_equal(run.zmap(uid)["lh"], run.zmap(uid, set_id)["lh"])

    # with percentiles of its own for every component, the shipped rows are placed in them
    grid = tuple(float(k) for k in range(101))
    components = ("mass_medial", "mass_inferior", "ceiling_frac", "occ_ctrl_ratio")
    normed = dataclasses.replace(
        as_shipped(built), metric_norms={"all": dict.fromkeys(components, grid)}
    )
    shutil.rmtree(tmp_path / "shipped")
    write_reference(tmp_path / "shipped", normed)
    with Run(run_copy) as run:
        # another content under the same id: the reference unit and the scores are redone
        assert counts(run.metrics()) == (2, 6, 0)
        scores = table(run_copy, "scores")
        placed = scores[scores["reference_id"] == "shipped-0.test"].reset_index(drop=True)
        expected = quantile_scores(placed["mass_medial"].astype("float64"), grid)
        np.testing.assert_allclose(placed["std_mass_medial"], expected, rtol=1e-6)
        assert placed["primary"].all() and sorted(placed["rank"]) == [1, 2, 3]
        # a mass of 0 lies at the bottom of that grid, one of 100 or more at its top
        assert placed["std_mass_medial"].min() == pytest.approx(-2.5758, abs=1e-3)
        within = scores[scores["reference_id"] == set_id].reset_index(drop=True)
        pd.testing.assert_frame_equal(within[columns], was[columns])

    # a reference made on another fsaverage does not apply
    other = as_shipped(built)
    other = Reference(**{**other.__dict__, "sphere_sha256": {"lh": "0" * 64, "rh": "1" * 64}})
    shutil.rmtree(tmp_path / "shipped")
    write_reference(tmp_path / "shipped", other)
    with Run(run_copy) as run:
        assert counts(run.metrics()) == (2, 6, 0)
        assert set(table(run_copy, "scores")["reference_id"]) == {set_id}
