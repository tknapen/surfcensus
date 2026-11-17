"""The input set and the pattern resolver (steps I0 and I1 of docs/dev/input_plan.md)."""

from __future__ import annotations

from pathlib import Path

import pytest

from surfcensus import inputs as inp
from surfcensus.inputs import InputError, Statements, compile_pattern, match_files, resolve


def touch(root: Path, *names: str) -> None:
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")


def runs_of(result: inp.InputSet) -> list[tuple[str, str | None, str, str | None, str | None]]:
    return [
        (s.label, se.label, r.name, r.raw, r.pre)
        for s in result.subjects
        for se in s.sessions
        for r in se.runs
    ]


# --- patterns ------------------------------------------------------------------------------


def test_a_pattern_must_be_absolute_and_name_known_fields(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="absolute"):
        compile_pattern("raw/*.nii.gz")
    with pytest.raises(InputError, match="unknown field"):
        compile_pattern(f"{tmp_path}/{{task}}.nii.gz")
    with pytest.raises(InputError, match="lone"):
        compile_pattern(f"{tmp_path}/{{sub.nii.gz")


def test_fields_match_within_one_component_and_repeat_with_one_value(tmp_path: Path) -> None:
    touch(tmp_path, "S01/S01_rest1.nii.gz", "S01/S02_rest1.nii.gz", "S02/x/S02_rest1.nii.gz")
    found = match_files(compile_pattern(f"{tmp_path}/{{sub}}/{{sub}}_{{run}}.nii.gz"))
    assert [(Path(p).name, f) for p, f in found] == [
        ("S01_rest1.nii.gz", {"sub": "S01", "run": "rest1"})
    ]


def test_literal_brackets_and_braces(tmp_path: Path) -> None:
    touch(tmp_path, "a[1]/{x}_r1.nii")
    found = match_files(compile_pattern(f"{tmp_path}/a[1]/{{{{x}}}}_{{run}}.nii"))
    assert [f for _, f in found] == [{"run": "r1"}]


def test_a_file_matched_in_two_ways_is_refused(tmp_path: Path) -> None:
    touch(tmp_path, "a_b_c.nii")
    with pytest.raises(InputError, match="two ways"):
        match_files(compile_pattern(f"{tmp_path}/{{sub}}_{{run}}.nii"))


def test_directories_and_files_are_told_apart(tmp_path: Path) -> None:
    touch(tmp_path, "fs/sub-01/mri/T1.mgz", "fs/sub-02")
    pattern = compile_pattern(f"{tmp_path}/fs/{{sub}}")
    assert [f["sub"] for _, f in match_files(pattern, directories=True)] == ["sub-01"]
    assert [f["sub"] for _, f in match_files(pattern)] == ["sub-02"]


# --- pairing by fields ----------------------------------------------------------------------


@pytest.fixture
def lab_tree(tmp_path: Path) -> Path:
    """Non-BIDS names, as a lab pipeline writes them: raw per subject and session, ANTs
    transforms per run, one fsnative transform per session, FreeSurfer subjects by label."""
    touch(
        tmp_path,
        "raw/S01/ses1/S01_ses1_bar1.nii.gz",
        "raw/S01/ses1/S01_ses1_bar2.nii.gz",
        "raw/S01/ses2/S01_ses2_bar1.nii.gz",
        "raw/S02/ses1/S02_ses1_bar1.nii.gz",
        "prep/S01/ses1/bar1_preproc.nii.gz",
        "prep/S01/ses1/bar2_preproc.nii.gz",
        "prep/S01/ses2/bar1_preproc.nii.gz",
        "prep/S02/ses1/bar1_preproc.nii.gz",
        "prep/S01/ses1/bar1_mask.nii.gz",
        "prep/S01/ses1/bar2_mask.nii.gz",
        "prep/S01/ses1/bar1_epi2anat0GenericAffine.txt",
        "prep/S01/ses1/bar2_epi2anat0GenericAffine.txt",
        "prep/S01/ses2/bar1_epi2anat0GenericAffine.txt",
        "prep/S01/ses1/anat2fs.txt",
        "fs/S01/mri/T1.mgz",
        "fs/S02/mri/T1.mgz",
    )
    return tmp_path


def lab_patterns(root: Path) -> dict[str, str]:
    return {
        "raw": f"{root}/raw/{{sub}}/{{ses}}/{{sub}}_{{ses}}_{{run}}.nii.gz",
        "pre": f"{root}/prep/{{sub}}/{{ses}}/{{run}}_preproc.nii.gz",
        "mask": f"{root}/prep/{{sub}}/{{ses}}/{{run}}_mask.nii.gz",
        "xfm": f"{root}/prep/{{sub}}/{{ses}}/{{run}}_epi2anat0GenericAffine.txt",
        "fsnative_xfm": f"{root}/prep/{{sub}}/{{ses}}/anat2fs.txt",
        "freesurfer": f"{root}/fs/{{sub}}",
    }


def test_fields_pair_runs_and_key_companions(lab_tree: Path) -> None:
    result = resolve(lab_patterns(lab_tree))
    assert result.pair == "fields"
    assert [(s, se, n) for s, se, n, _, _ in runs_of(result)] == [
        ("S01", "ses1", "bar1"),
        ("S01", "ses1", "bar2"),
        ("S01", "ses2", "bar1"),
        ("S02", "ses1", "bar1"),
    ]
    s01 = result.subjects[0]
    ses1, ses2 = s01.sessions
    assert ses1.freesurfer == f"{lab_tree}/fs/S01" == ses2.freesurfer
    assert ses1.fsnative_xfm == f"{lab_tree}/prep/S01/ses1/anat2fs.txt"
    assert ses2.fsnative_xfm is None
    bar1, bar2 = ses1.runs
    assert bar1.raw == f"{lab_tree}/raw/S01/ses1/S01_ses1_bar1.nii.gz"
    assert bar1.pre == f"{lab_tree}/prep/S01/ses1/bar1_preproc.nii.gz"
    assert bar1.mask == f"{lab_tree}/prep/S01/ses1/bar1_mask.nii.gz"
    assert bar2.xfm == f"{lab_tree}/prep/S01/ses1/bar2_epi2anat0GenericAffine.txt"
    assert ses2.runs[0].mask is None
    assert result.subjects[1].sessions[0].runs[0].xfm is None
    assert result.unused == ()


def test_a_session_label_with_its_prefix_is_read_without_it(tmp_path: Path) -> None:
    touch(tmp_path, "raw/sub-01/ses-2/a.nii", "pre/sub-01/ses-2/a.nii")
    result = resolve(
        {
            "raw": f"{tmp_path}/raw/{{sub}}/{{ses}}/{{run}}.nii",
            "pre": f"{tmp_path}/pre/{{sub}}/{{ses}}/{{run}}.nii",
        }
    )
    assert runs_of(result)[0][:3] == ("01", "2", "a")


def test_a_run_on_one_side_only_is_kept_with_the_other_side_none(lab_tree: Path) -> None:
    touch(lab_tree, "raw/S02/ses1/S02_ses1_bar2.nii.gz", "prep/S02/ses1/bar3_preproc.nii.gz")
    rows = runs_of(resolve(lab_patterns(lab_tree)))
    s02 = {n: (raw is not None, pre is not None) for s, _, n, raw, pre in rows if s == "S02"}
    assert s02 == {"bar1": (True, True), "bar2": (True, False), "bar3": (False, True)}


def test_a_companion_without_run_field_is_of_every_run_of_its_session(lab_tree: Path) -> None:
    touch(lab_tree, "xfm/S01/ses1/to_anat.txt")
    patterns = lab_patterns(lab_tree) | {"xfm": f"{lab_tree}/xfm/{{sub}}/{{ses}}/to_anat.txt"}
    ses1 = resolve(patterns).subjects[0].sessions[0]
    assert {r.xfm for r in ses1.runs} == {f"{lab_tree}/xfm/S01/ses1/to_anat.txt"}


def test_two_companions_for_one_run_stop_and_name_the_field(lab_tree: Path) -> None:
    patterns = lab_patterns(lab_tree) | {"mask": f"{lab_tree}/prep/{{sub}}/{{ses}}/*_mask.nii.gz"}
    with pytest.raises(InputError, match=r"two files for one run.*put \{run\}"):
        resolve(patterns)


def test_a_companion_that_no_run_takes_is_reported(lab_tree: Path) -> None:
    touch(lab_tree, "prep/S03/ses1/bar1_mask.nii.gz")
    result = resolve(lab_patterns(lab_tree))
    assert result.unused == (("mask", f"{lab_tree}/prep/S03/ses1/bar1_mask.nii.gz"),)
    assert any(line.startswith("Taken by no run (--mask-files)") for line in inp.table(result))


def test_run_field_on_one_side_only_stops(lab_tree: Path) -> None:
    patterns = lab_patterns(lab_tree) | {"pre": f"{lab_tree}/prep/{{sub}}/{{ses}}/*_preproc.nii.gz"}
    with pytest.raises(InputError, match=r"--pre-files has no \{run\}"):
        resolve(patterns)


def test_fields_that_disagree_with_bids_names_stop(tmp_path: Path) -> None:
    touch(
        tmp_path,
        "raw/a/sub-01_task-rest_run-1_bold.nii.gz",
        "pre/a/sub-01_task-rest_run-2_desc-preproc_bold.nii.gz",
    )
    with pytest.raises(InputError, match="different runs"):
        resolve(
            {
                "raw": f"{tmp_path}/raw/{{run}}/*_bold.nii.gz",
                "pre": f"{tmp_path}/pre/{{run}}/*_bold.nii.gz",
            }
        )


def test_no_subject_anywhere_needs_one_given(tmp_path: Path) -> None:
    touch(tmp_path, "raw/r1.nii", "pre/r1.nii")
    patterns = {"raw": f"{tmp_path}/raw/{{run}}.nii", "pre": f"{tmp_path}/pre/{{run}}.nii"}
    with pytest.raises(InputError, match="names no subject"):
        resolve(patterns)
    assert runs_of(resolve(patterns, subject="sub-P1"))[0][:3] == ("P1", None, "r1")
    with pytest.raises(InputError, match="ASCII"):
        resolve(patterns, subject="P 1")


# --- pairing by BIDS names and by order -----------------------------------------------------


def test_bids_names_pair_by_run_key_without_fields(tmp_path: Path) -> None:
    touch(
        tmp_path,
        "raw/sub-01/ses-1/func/sub-01_ses-1_task-rest_run-01_bold.nii.gz",
        "raw/sub-01/ses-1/func/sub-01_ses-1_task-rest_run-02_bold.nii.gz",
        "pre/sub-01_ses-1_task-rest_run-1_space-T1w_desc-preproc_bold.nii.gz",
        "pre/sub-01_ses-1_task-rest_run-2_space-T1w_desc-preproc_bold.nii.gz",
        "pre/sub-01_ses-1_task-rest_run-1_space-T1w_desc-brain_mask.nii.gz",
        "pre/sub-01_ses-1_from-T1w_to-fsnative_xfm.txt",
    )
    result = resolve(
        {
            "raw": f"{tmp_path}/raw/*/*/func/*_bold.nii.gz",
            "pre": f"{tmp_path}/pre/*_desc-preproc_bold.nii.gz",
            "mask": f"{tmp_path}/pre/*_desc-brain_mask.nii.gz",
            "fsnative_xfm": f"{tmp_path}/pre/*_to-fsnative_xfm.txt",
        }
    )
    assert result.pair == "bids"
    assert [n for _, _, n, _, _ in runs_of(result)] == [
        "ses-1_task-rest_run-01",
        "ses-1_task-rest_run-02",
    ]
    session = result.subjects[0].sessions[0]
    assert session.label == "1"
    assert session.runs[0].mask is not None and session.runs[1].mask is None
    assert session.fsnative_xfm == f"{tmp_path}/pre/sub-01_ses-1_from-T1w_to-fsnative_xfm.txt"


def test_names_that_are_not_bids_without_fields_stop_with_the_list(tmp_path: Path) -> None:
    touch(tmp_path, "raw/S01_bar1.nii", "pre/bar1.nii")
    with pytest.raises(InputError, match="cannot be paired") as info:
        resolve({"raw": f"{tmp_path}/raw/*.nii", "pre": f"{tmp_path}/pre/*.nii"})
    assert any("S01_bar1.nii" in line for line in info.value.lines)


def test_order_pairs_in_path_order_only_when_asked(tmp_path: Path) -> None:
    touch(tmp_path, "raw/S01_b.nii", "raw/S01_a.nii", "pre/2.nii", "pre/1.nii")
    patterns = {"raw": f"{tmp_path}/raw/*.nii", "pre": f"{tmp_path}/pre/*.nii"}
    result = resolve(patterns, pair="order", subject="S01")
    assert [(Path(r).name, Path(p).name) for *_, r, p in runs_of(result)] == [
        ("S01_a.nii", "1.nii"),
        ("S01_b.nii", "2.nii"),
    ]
    assert "path order" in inp.table(result)[0]
    touch(tmp_path, "pre/3.nii")
    with pytest.raises(InputError, match="2 raw and 3 preprocessed"):
        resolve(patterns, pair="order", subject="S01")


def test_two_raw_files_of_one_run_stop(tmp_path: Path) -> None:
    touch(tmp_path, "raw/a/S01_r1.nii", "raw/b/S01_r1.nii", "pre/S01_r1.nii")
    with pytest.raises(InputError, match="two files of one run"):
        resolve(
            {
                "raw": f"{tmp_path}/raw/*/{{sub}}_{{run}}.nii",
                "pre": f"{tmp_path}/pre/{{sub}}_{{run}}.nii",
            }
        )


def test_a_pattern_that_names_nothing_stops(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="names no file"):
        resolve({"raw": f"{tmp_path}/raw/*.nii", "pre": f"{tmp_path}/pre/*.nii"})
    with pytest.raises(InputError, match="--pre-files is needed"):
        resolve({"raw": f"{tmp_path}/raw/*.nii"})


# --- statements, the table and inputs.yaml --------------------------------------------------


def test_statements_that_are_missing_are_named(lab_tree: Path) -> None:
    result = resolve(lab_patterns(lab_tree))
    needs = inp.missing_statements(result)
    assert [n.split()[0] for n in needs] == ["--pre-space", "--xfm-fixed", "--fs-frame"]
    stated = resolve(
        lab_patterns(lab_tree),
        statements=Statements(pre_space="anat", xfm_fixed="anat", fs_frame="transform"),
    )
    assert inp.missing_statements(stated) == []
    assert not any(line.startswith("Not stated") for line in inp.table(stated))


def test_the_table_lists_each_run_and_what_it_lacks(lab_tree: Path) -> None:
    touch(lab_tree, "raw/S02/ses1/S02_ses1_bar2.nii.gz")
    lines = inp.table(resolve(lab_patterns(lab_tree)))
    assert lines[0].startswith("Paired by the {sub}, {ses} and {run} fields")
    assert "sub-S01 ses-ses1: 2 runs" in lines
    assert any(line.split() == ["pre", "none:", "no", "preprocessed", "series"] for line in lines)


def test_inputs_yaml_round_trips(lab_tree: Path, tmp_path: Path) -> None:
    result = resolve(lab_patterns(lab_tree), statements=Statements(xfm_fixed="epi"))
    run_dir = tmp_path / "run"
    assert inp.read_inputs(run_dir) is None
    inp.write_inputs(run_dir, result)
    assert inp.read_inputs(run_dir) == result
    (run_dir / inp.INPUTS_FILE).write_text("schema_version: 99\n")
    with pytest.raises(ValueError, match="schema"):
        inp.read_inputs(run_dir)


def test_resolving_opens_no_file(lab_tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    def refuse(*args: object, **kwargs: object) -> object:
        raise AssertionError(f"opened {args[0]!r}")

    monkeypatch.setattr(builtins, "open", refuse)
    monkeypatch.setattr(Path, "open", refuse)
    resolve(lab_patterns(lab_tree))


def test_every_pattern_role_has_an_option() -> None:
    assert set(inp.OPTIONS) == set(inp.PATTERN_ROLES)
    assert len(set(inp.OPTIONS.values())) == len(inp.OPTIONS)
