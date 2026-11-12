"""Tests of ``surfcensus inventory``, ``render`` and ``status``: what they print, their exit
codes, what they store between calls, and that a source is only read."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pandas as pd
import pytest
from synth import snapshot_tree  # pyrefly: ignore[missing-import]
from synth_tree import write_tier_a, write_tree  # pyrefly: ignore[missing-import]
from typer.testing import CliRunner, Result

from surfcensus.cli import app
from surfcensus.config import CONFIG_FILE, load_config, new_local_config
from surfcensus.privacy import HashKey
from surfcensus.render import RenderConfig, api, encode
from surfcensus.run import Run
from surfcensus.sources.local import local

LABELS = ("sub-01", "sub-02", "sub-03")
FS_ROOT = "sourcedata/freesurfer"
DATASET_KEY = HashKey(bytes(range(1, 33)), "dataset-key")
SECONDS = r"\d+\.\d s"


def cli(*args: object) -> Result:
    return CliRunner().invoke(app, [str(arg) for arg in args])


def said(output: str) -> str:
    """The words of a usage error, whatever the terminal: without escape codes, without the
    frame Rich draws, with every run of white space as one space."""
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", output)
    return " ".join(re.sub(r"[─-╿]", " ", text).split())


def lines(result: Result) -> list[str]:
    return result.output.splitlines()


@pytest.fixture
def study(tmp_path: Path) -> Path:
    """An fMRIPrep output with three light subjects; ``sub-03`` lacks ``surf/lh.pial``, and
    ``sub-01`` has a build stamp, logs and stats files."""
    tree = write_tree(tmp_path / "study", "fmriprep", LABELS, incomplete=["sub-03"])
    write_tier_a(tree / FS_ROOT / "sub-01")
    return tree


@pytest.fixture
def run_dir(study: Path, tmp_path: Path) -> Path:
    """A local run on ``study`` with small tiles, not inventoried yet."""
    path = tmp_path / "run"
    Run(
        path, local(study), config=new_local_config(study, render=RenderConfig(panel_px=32))
    ).close()
    return path


def pseudonym(run_dir: Path) -> str:
    with Run(run_dir) as run:
        (name,) = run.public().datasets()["pseudonym"]
    return name


def uid(run_dir: Path, label: str) -> str:
    with Run(run_dir) as run:
        subjects = run.public().subjects()
    return dict(zip(subjects["subject_label"], subjects["subject_uid"], strict=True))[label]


# ---------------------------------------------------------------------------------------
# inventory


def test_inventory_creates_a_local_run_and_says_what_it_found(study: Path, tmp_path: Path) -> None:
    before = snapshot_tree(study)
    run_dir = tmp_path / "new" / "run"
    result = cli("inventory", "--run-dir", run_dir, "--source", study)
    assert result.exit_code == 0, result.output
    name = pseudonym(run_dir)
    roles_b = "lh_thickness, rh_thickness, lh_sphere_reg, rh_sphere_reg, lh_annot, rh_annot"
    assert lines(result)[:-1] == [
        f"{name}: layout fmriprep, pipeline fmriprep 23.2.0",
        "subjects: 3 found, 2 complete, 1 with Tier A, 0 with Tier B",
        "FreeSurfer: 6.0.1 (1), unknown (2)",
        f"  lacks lh_pial, {roles_b}, lh_stats, rh_stats: 1 (sub-03)",
        f"  lacks {roles_b}: 1 (sub-01)",
        f"  lacks {roles_b}, lh_stats, rh_stats: 1 (sub-02)",
    ]
    assert re.fullmatch(rf"inventory: 3 done, 0 skipped, 0 failed, {SECONDS}", lines(result)[-1])
    config = load_config(run_dir)
    assert config.mode == "local" and config.render_video is False
    assert config.sources[0].root == str(study.resolve())

    # again, without any option: nothing is read, nothing is stored anew
    stored = (run_dir / CONFIG_FILE).read_bytes()
    again = cli("inventory", "--run-dir", run_dir)
    assert again.exit_code == 0, again.output
    assert lines(again)[:-1] == lines(result)[:-1]
    assert re.fullmatch(rf"inventory: 0 done, 3 skipped, 0 failed, {SECONDS}", lines(again)[-1])
    assert (run_dir / CONFIG_FILE).read_bytes() == stored
    forced = cli("inventory", "--run-dir", run_dir, "--source", study, "--force")
    assert forced.exit_code == 0 and "inventory: 3 done, 0 skipped" in forced.output
    assert (run_dir / CONFIG_FILE).read_bytes() == stored
    assert snapshot_tree(study) == before


def test_inventory_options_are_stored_and_hold_until_they_are_given_again(
    study: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    first = cli(
        "inventory", "--run-dir", run_dir, "--source", study,
        "--subjects", "sub-01", "--subjects", "sub-03", "--field-strength", "7",
    )  # fmt: skip
    assert first.exit_code == 0, first.output
    assert "subjects: 2 found, 1 complete" in first.output
    source = load_config(run_dir).sources[0]
    assert (source.subjects, source.field_strength) == (("sub-01", "sub-03"), 7.0)

    stored = (run_dir / CONFIG_FILE).read_bytes()
    again = cli("inventory", "--run-dir", run_dir, "--source", study)
    assert again.exit_code == 0 and "subjects: 2 found" in again.output
    assert (run_dir / CONFIG_FILE).read_bytes() == stored

    wider = cli("inventory", "--run-dir", run_dir, "--subjects", "sub-0[12]")
    assert wider.exit_code == 0, wider.output
    assert "subjects: 2 found, 2 complete" in wider.output
    assert "1 subject(s) are no longer listed and were marked absent" in wider.output
    source = load_config(run_dir).sources[0]
    assert (source.subjects, source.field_strength) == (("sub-0[12]",), 7.0)
    with Run(run_dir) as run:
        subjects = run.public().subjects().set_index("subject_label")
    assert subjects["present"].to_dict() == {"sub-01": True, "sub-02": True, "sub-03": False}
    assert set(subjects.loc[["sub-01", "sub-02"], "field_strength"]) == {7.0}


def test_inventory_stops_with_exit_2_for_what_is_wrong_with_the_call(
    study: Path, run_dir: Path, tmp_path: Path
) -> None:
    def refused(*args: object) -> str:
        result = cli("inventory", *args)
        assert result.exit_code == 2, result.output
        return result.output

    nowhere = tmp_path / "nowhere"
    assert refused("--run-dir", nowhere) == (
        f"missing: {nowhere} holds no run; pass --source PATH to create one\n"
    )
    assert refused("--run-dir", nowhere, "--source", tmp_path / "absent").startswith("not found: ")
    empty = tmp_path / "empty"
    empty.mkdir()
    assert refused("--run-dir", nowhere, "--source", empty).startswith(
        "layout: no FreeSurfer subject found at "
    )
    out = refused("--run-dir", nowhere, "--source", study, "--subjects", "sub-9*")
    assert out.startswith("not found: no subject matches 'sub-9*'. Subjects in ")
    assert out.rstrip().endswith("sub-01, sub-02, sub-03")
    assert refused("--run-dir", study / FS_ROOT / "run", "--source", study).startswith(
        "invalid: the run directory "
    )
    assert refused("--run-dir", nowhere, "--source", study, "--relocate").startswith("missing: ")
    assert not nowhere.exists() and not (study / FS_ROOT / "run").exists()

    # on the run that exists
    stored = (run_dir / CONFIG_FILE).read_bytes()
    out = refused("--run-dir", run_dir, "--subjects", "sub-9*")
    assert out.startswith("not found: no subject matches 'sub-9*'")
    assert refused("--run-dir", run_dir, "--dataset", "ds000102") == (
        "invalid: no dataset named ds000102 in this run\n"
    )
    assert refused("--run-dir", run_dir, "--relocate").startswith("invalid: --relocate needs")
    assert refused("--run-dir", run_dir, "--field-strength", "-3").startswith("invalid: ")
    assert (run_dir / CONFIG_FILE).read_bytes() == stored
    usage = cli("inventory", "--run-dir", run_dir, "--source", study, "--relocate", "elsewhere")
    assert usage.exit_code == 2
    assert "takes a dataset pseudonym (D-xxxxxxxx) or nothing" in said(usage.output)
    assert cli("inventory").exit_code == 2  # --run-dir is required


def test_inventory_refuses_another_root_and_relocates_when_asked(
    study: Path, run_dir: Path, tmp_path: Path
) -> None:
    assert cli("inventory", "--run-dir", run_dir, "--subjects", "sub-0[13]").exit_code == 0
    assert cli("render", "--run-dir", run_dir).exit_code == 1
    name, one = pseudonym(run_dir), uid(run_dir, "sub-01")
    moved = tmp_path / "moved"
    shutil.move(study, moved)

    gone = cli("inventory", "--run-dir", run_dir)
    assert gone.exit_code == 2 and gone.output.startswith("not found: ")
    refused = cli("inventory", "--run-dir", run_dir, "--source", moved)
    assert refused.exit_code == 2
    assert refused.output == (
        f"other root: this run directory belongs to {study.resolve()}; pass --run-dir for a "
        "new run, or --relocate if the data moved\n"
    )

    relocated = cli("inventory", "--run-dir", run_dir, "--relocate", "--source", moved)
    assert relocated.exit_code == 0, relocated.output
    assert "inventory: 0 done, 2 skipped, 0 failed" in relocated.output
    source = load_config(run_dir).sources[0]
    assert (source.root, source.identity) == (str(moved.resolve()), str(study.resolve()))
    assert source.subjects == ("sub-0[13]",)  # the subject set moved along
    assert pseudonym(run_dir) == name and uid(run_dir, "sub-01") == one
    rendered = cli("render", "--run-dir", run_dir)
    assert "render: 0 done, 1 skipped, 1 failed" in rendered.output

    # and back, naming the dataset
    shutil.move(moved, study)
    back = cli("inventory", "--run-dir", run_dir, "--source", study, "--relocate", name)
    assert back.exit_code == 0 and "0 done, 2 skipped" in back.output
    other = cli("inventory", "--run-dir", run_dir, "--source", study, "--relocate", "D-00000000")
    assert other.exit_code == 0  # the root is the stored one: nothing to relocate


def test_an_unreadable_subject_is_a_failed_unit_with_exit_1(
    study: Path, run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refused = study.resolve() / FS_ROOT / "sub-02" / "mri"
    stat = Path.stat

    def no_entry(self: Path, *args: object, **kwargs: object) -> object:
        if self == refused:  # as in a directory without search permission
            raise PermissionError(13, "Permission denied", str(self))
        return stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", no_entry)
    for expected in ("3 done, 0 skipped, 1 failed", "0 done, 3 skipped, 1 failed"):
        result = cli("inventory", "--run-dir", run_dir)
        assert result.exit_code == 1, result.output
        assert "  unreadable: Permission denied: 1 (sub-02)" in lines(result)
        assert lines(result)[-2].startswith(f"inventory: {expected}")
        assert lines(result)[-1] == "  unreadable: 1 (sub-02)"
    status = cli("status", "--run-dir", run_dir, "--failures")
    assert status.exit_code == 0
    assert "inventory: 2 done, 1 failed" in lines(status)
    assert f"inventory {uid(run_dir, 'sub-02')} (sub-02): unreadable: Permission denied" in lines(
        status
    )


# ---------------------------------------------------------------------------------------
# render


def test_render_reports_counts_and_failures_and_exits_1_while_a_subject_is_failed(
    study: Path, run_dir: Path
) -> None:
    before = snapshot_tree(study)
    assert cli("inventory", "--run-dir", run_dir).exit_code == 0
    for counts in ("2 done, 0 skipped, 1 failed", "0 done, 2 skipped, 1 failed"):
        result = cli("render", "--run-dir", run_dir)
        assert result.exit_code == 1, result.output
        assert re.fullmatch(rf"render: {counts}, {SECONDS}", lines(result)[0])
        assert lines(result)[1:] == ["  missing surface: 1 (sub-03)"]
    assert cli("render", "--run-dir", run_dir, "--retry-failed", "--jobs", "1").exit_code == 1
    media = run_dir / "media"
    assert {p.name for p in media.iterdir()} == {uid(run_dir, "sub-01"), uid(run_dir, "sub-02")}
    assert not list(media.rglob("anim.mp4"))  # a local run renders no video unless asked

    # a run in which every subject renders ends with 0
    cli("inventory", "--run-dir", run_dir, "--subjects", "sub-0[12]")
    done = cli("render", "--run-dir", run_dir, "--dataset", pseudonym(run_dir))
    assert done.exit_code == 0, done.output
    assert re.fullmatch(rf"render: 0 done, 2 skipped, 0 failed, {SECONDS}\n", done.output)
    forced = cli("render", "--run-dir", run_dir, "--force", "--all")
    assert forced.exit_code == 0 and forced.output.startswith("render: 2 done, 0 skipped, 0 failed")
    assert snapshot_tree(study) == before


def test_a_subject_that_cannot_be_rendered_is_listed_with_a_scrubbed_detail(
    study: Path, run_dir: Path, tmp_path: Path
) -> None:
    (study / FS_ROOT / "sub-02" / "mri" / "T1.mgz").write_bytes(b"")
    assert cli("inventory", "--run-dir", run_dir).exit_code == 0
    result = cli("render", "--run-dir", run_dir)
    assert result.exit_code == 1
    assert lines(result)[1:] == [
        "  missing surface: 1 (sub-03)",
        "  render failure: 1 (sub-02)",
    ]
    status = cli("status", "--run-dir", run_dir, "--failures")
    assert status.exit_code == 0, status.output
    name = pseudonym(run_dir)
    assert lines(status)[:3] == [
        f"local run, datasets: {name}",
        "inventory: 3 done, 0 failed",
        "render: 1 done, 2 failed, 0 pending",
    ]
    failed = sorted(lines(status)[3:])
    two, three = uid(run_dir, "sub-02"), uid(run_dir, "sub-03")
    assert f"render {three} (sub-03): missing surface: no file for lh_pial" in failed
    (line,) = [text for text in failed if two in text]
    assert line.startswith(f"render {two} (sub-02): render failure: ") and "T1.mgz" in line
    assert str(tmp_path) not in status.output


def test_the_video_option_is_stored_and_a_missing_encoder_is_exit_2(
    study: Path, run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert cli("inventory", "--run-dir", run_dir, "--subjects", "sub-01").exit_code == 0
    assert cli("render", "--run-dir", run_dir).exit_code == 0
    one = run_dir / "media" / uid(run_dir, "sub-01")
    assert not (one / "anim.mp4").exists()

    with monkeypatch.context() as patched:
        patched.setattr(encode, "ffmpeg_exe", lambda: (_ for _ in ()).throw(RuntimeError("none")))
        missing = cli("render", "--run-dir", run_dir, "--video")
    assert missing.exit_code == 2
    assert missing.output == "missing: the video cannot be encoded: none\n"
    status = cli("status", "--run-dir", run_dir)
    assert "render: 0 done, 0 failed, 1 pending" in lines(status)  # the option was stored

    with_video = cli("render", "--run-dir", run_dir)  # not given again: the stored value holds
    assert with_video.exit_code == 0 and "render: 1 done" in with_video.output
    assert (one / "anim.mp4").is_file() and load_config(run_dir).render_video is True
    stored = (run_dir / CONFIG_FILE).read_bytes()
    assert "render: 0 done, 1 skipped" in cli("render", "--run-dir", run_dir, "--video").output
    assert (run_dir / CONFIG_FILE).read_bytes() == stored

    without = cli("render", "--run-dir", run_dir, "--no-video")
    assert without.exit_code == 0 and "render: 1 done" in without.output
    assert not (one / "anim.mp4").exists() and load_config(run_dir).render_video is False


def test_all_and_selected_choose_between_every_subject_and_the_selection(
    study: Path, run_dir: Path
) -> None:
    assert cli("inventory", "--run-dir", run_dir).exit_code == 0
    one = uid(run_dir, "sub-01")
    with Run(run_dir) as run:
        run.public().replace_selection(
            pd.DataFrame(
                {
                    "subject_uid": [one],
                    "reason": ["all"],
                    "stratum": ["none"],
                    "inclusion_prob": [1.0],
                    "seed": [0],
                }
            ),
            ["all"],
        )
    for flags in ((), ("--selected",)):
        result = cli("render", "--run-dir", run_dir, *flags)
        assert result.exit_code == 1 and "render: " in result.output
    assert {p.name for p in (run_dir / "media").iterdir()} == {one}
    # the status follows the selection too: sub-02 is not pending, sub-03 lacks a surface
    assert "render: 1 done, 1 failed, 0 pending" in lines(cli("status", "--run-dir", run_dir))
    everything = cli("render", "--run-dir", run_dir, "--all")
    assert "render: 1 done, 1 skipped, 1 failed" in everything.output
    assert len(list((run_dir / "media").iterdir())) == 2


def test_an_interrupted_render_keeps_what_was_recorded(
    study: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    config = new_local_config(study, render=RenderConfig(panel_px=32), cache={"batch_subjects": 1})
    Run(run_dir, local(study), config=config).close()
    assert cli("inventory", "--run-dir", run_dir).exit_code == 0
    original = api.render_subject
    calls = 0

    def interrupted(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return original(*args, **kwargs)

    with monkeypatch.context() as patched:
        patched.setattr(api, "render_subject", interrupted)
        stopped = cli("render", "--run-dir", run_dir)
    assert stopped.exit_code == 130 and stopped.output.startswith("interrupted; ")
    assert "render: 1 done, 1 failed, 1 pending" in lines(cli("status", "--run-dir", run_dir))
    rest = cli("render", "--run-dir", run_dir)
    assert rest.exit_code == 1 and "render: 1 done, 1 skipped, 1 failed" in rest.output


def test_the_check_sample_names_its_seed_and_is_the_same_for_the_same_seed(
    study: Path, run_dir: Path
) -> None:
    cli("inventory", "--run-dir", run_dir)
    cli("render", "--run-dir", run_dir)
    logs = run_dir / "logs"

    seeded = cli("render", "--run-dir", run_dir, "--check-sample", "--seed", "7", "--fraction", "1")
    assert seeded.exit_code == 0, seeded.output
    listing = logs / "render-check-7.txt"
    assert seeded.output == (
        f"render check: seed 7, 2 of 2 done renders (fraction 1): {listing} and "
        "render-check-7.html\n"
    )
    first = (listing.read_bytes(), listing.with_suffix(".html").read_bytes())
    assert listing.read_text().splitlines()[1:] == sorted(
        f"{uid(run_dir, label)}\tmedia/{uid(run_dir, label)}/poster.webp"
        for label in ("sub-01", "sub-02")
    )
    cli("render", "--run-dir", run_dir, "--check-sample", "--seed", "7", "--fraction", "1.0")
    assert (listing.read_bytes(), listing.with_suffix(".html").read_bytes()) == first

    drawn = cli("render", "--run-dir", run_dir, "--check-sample")
    assert drawn.exit_code == 0
    match = re.match(
        r"render check: seed (\d+), 1 of 2 done renders \(fraction 0\.02\): ", drawn.output
    )
    assert match is not None and (logs / f"render-check-{match.group(1)}.txt").is_file()


@pytest.mark.parametrize(
    ("flags", "words"),
    [
        (("--all", "--selected"), "--all contradicts --selected"),
        (("--check-sample", "--force"), "--check-sample renders nothing"),
        (("--check-sample", "--all"), "--check-sample renders nothing"),
        (("--seed", "3"), "--fraction and --seed belong to --check-sample"),
        (("--fraction", "0.5"), "--fraction and --seed belong to --check-sample"),
        (("--check-sample", "--fraction", "0"), "is above 0 and at most 1"),
        (("--check-sample", "--fraction", "1.5"), "is above 0 and at most 1"),
        (("--jobs", "0"), "--jobs"),
    ],
)
def test_render_refuses_options_that_do_not_go_together(
    run_dir: Path, flags: tuple[str, ...], words: str
) -> None:
    result = cli("render", "--run-dir", run_dir, *flags)
    assert result.exit_code == 2
    assert words in said(result.output)
    assert not (run_dir / "media").exists() and not (run_dir / "logs").exists()


# ---------------------------------------------------------------------------------------
# status, and runs that cannot be opened


def test_status_of_a_run_before_and_after_its_phases(study: Path, run_dir: Path) -> None:
    fresh = cli("status", "--run-dir", run_dir)
    assert fresh.exit_code == 0, fresh.output
    assert lines(fresh) == [
        "local run, datasets: none",
        "inventory: 0 done, 0 failed",
        "render: 0 done, 0 failed, 0 pending",
    ]
    cli("inventory", "--run-dir", run_dir)
    name = pseudonym(run_dir)
    listed = cli("status", "--run-dir", run_dir)
    assert lines(listed) == [
        f"local run, datasets: {name}",
        "inventory: 3 done, 0 failed",
        "render: 0 done, 1 failed, 2 pending",
    ]
    cli("render", "--run-dir", run_dir)
    shutil.rmtree(study / FS_ROOT / "sub-02")
    cli("inventory", "--run-dir", run_dir)
    after = cli("status", "--run-dir", run_dir)
    assert lines(after) == [
        f"local run, datasets: {name}",
        "inventory: 2 done, 0 failed (1 no longer in the source)",
        "render: 1 done, 1 failed, 0 pending",
    ]
    assert not list(run_dir.rglob("*.log"))  # status writes nothing


@pytest.mark.parametrize("command", ["inventory", "render", "status"])
def test_a_directory_without_a_run_is_exit_2(command: str, tmp_path: Path) -> None:
    result = cli(command, "--run-dir", tmp_path / "nothing")
    assert result.exit_code == 2
    assert result.output.startswith("missing: ") and result.output.count("\n") == 1
    assert not (tmp_path / "nothing").exists()


def test_a_configuration_with_an_unknown_key_is_exit_2(run_dir: Path) -> None:
    path = run_dir / CONFIG_FILE
    path.write_text(path.read_text() + "colour: blue\n")
    for command in ("inventory", "render", "status"):
        result = cli(command, "--run-dir", run_dir)
        assert result.exit_code == 2
        assert result.output == "invalid: config.yaml: colour: unknown key\n"


def test_a_census_run_shows_no_label_and_its_inventory_needs_the_key(
    study: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "census"
    with Run(run_dir, local(study), key=DATASET_KEY) as run:
        run.inventory()
        (name,) = run.public().datasets()["pseudonym"]
    result = cli("render", "--run-dir", run_dir)
    assert result.exit_code == 1, result.output
    status = cli("status", "--run-dir", run_dir, "--failures")
    assert lines(status)[:3] == [
        f"census run, datasets: {name}",
        "inventory: 3 done, 0 failed",
        "render: 2 done, 1 failed, 0 pending",
    ]
    for output in (result.output, status.output):
        assert "sub-0" not in output and str(tmp_path) not in output and "study" not in output
    assert re.fullmatch(r"  missing surface: 1 \([0-9a-f]{16}\)", lines(result)[1])

    keyless = cli("inventory", "--run-dir", run_dir)
    assert keyless.exit_code == 2
    assert keyless.output == "key: this command needs the dataset key of the census run\n"


def test_verbose_says_which_layout_was_found(study: Path, tmp_path: Path) -> None:
    quiet = cli("inventory", "--run-dir", tmp_path / "quiet", "--source", study)
    loud = cli("--verbose", "inventory", "--run-dir", tmp_path / "loud", "--source", study)
    assert quiet.exit_code == 0 and loud.exit_code == 0
    assert "INFO: " not in quiet.output
    assert (
        f"INFO: {study.resolve()} is an fMRIPrep output with subjects in sourcedata/freesurfer"
        in (loud.output)
    )
