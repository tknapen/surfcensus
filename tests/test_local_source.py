"""sources/local: layout detection, the subject listing and file access of a local source."""

from __future__ import annotations

import ast
import builtins
import dataclasses
import io
import logging
import os
import shutil
from pathlib import Path

import pytest
from synth import file_modes_apply, snapshot_tree  # pyrefly: ignore[missing-import]
from synth_tree import (  # pyrefly: ignore[missing-import]
    FS_ROOT,
    write_light_subject,
    write_light_template,
    write_tree,
)

import surfcensus
import surfcensus.sources.local as local_module
from surfcensus.config import SourceConfig, new_local_config
from surfcensus.privacy import new_local_salt, subject_uid
from surfcensus.render import RenderConfig, render_subject
from surfcensus.run import from_config
from surfcensus.sources.base import (
    RENDER_ROLES,
    TIER_A,
    TIER_B,
    TIER_C,
    DatasetRecord,
    FileRole,
    Layout,
    Source,
    SourceError,
    SubjectRecord,
)
from surfcensus.sources.local import (
    LocalSource,
    detect_layout,
    find_adjacent_template,
    local,
    template_files_missing,
)
from surfcensus.sources.patterns import ROLE_CANDIDATES

LAYOUTS: tuple[Layout, ...] = ("subjects_dir", "fmriprep", "fmriprep_legacy", "bids_derivatives")
LABELS = ("sub-01", "sub-02", "sub-03")
PIPELINE = {
    "subjects_dir": ("freesurfer", None),
    "fmriprep": ("fmriprep", "23.2.0"),
    "fmriprep_legacy": ("fmriprep", "23.2.0"),
    "bids_derivatives": ("freesurfer", None),
}
REAL_SUBJECT_FILES = {  # ds000102 sub-01, bytes, as in tests/conftest.py (bucket listing)
    "mri/T1.mgz": 5_576_738,
    "mri/brainmask.mgz": 1_691_388,
    "surf/lh.white": 5_341_363,
    "surf/lh.pial": 5_341_363,
    "surf/rh.white": 5_291_503,
    "surf/rh.pial": 5_291_503,
    "stats/lh.aparc.stats": 6_717,
    "stats/rh.aparc.stats": 6_717,
    "stats/aseg.stats": 9_853,
    "scripts/build-stamp.txt": 58,
    "scripts/recon-all.log": 220_842,
    "scripts/recon-all-lh.log": 212_143,
    "scripts/recon-all-rh.log": 206_247,
    "surf/lh.thickness": 593_343,
    "surf/rh.thickness": 587_803,
    "surf/lh.sphere.reg": 5_343_555,
    "surf/rh.sphere.reg": 5_293_695,
    "label/lh.aparc.annot": 1_188_169,
    "label/rh.aparc.annot": 1_177_089,
}
TEXT_FILES = {
    FileRole.LH_STATS: "# Measure Cortex, NumVert, Number of Vertices, 148332, unitless\n",
    FileRole.RH_STATS: "# Measure Cortex, NumVert, Number of Vertices, 146931, unitless\n",
    FileRole.ASEG_STATS: "# Measure lhSurfaceHoles, lhSurfaceHoles, holes, 41, unitless\n",
    FileRole.BUILD_STAMP: "freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.1-f53a55a\n",
    FileRole.RECON_LOG: "".join(f"line {i:04d} of the log\n" for i in range(200)),
}


def _labels(source: LocalSource) -> list[str]:
    return [subject.subject_label for subject in source.subjects(source.dataset)]


def _subject(source: LocalSource, label: str) -> SubjectRecord:
    (record,) = [s for s in source.subjects(source.dataset) if s.subject_label == label]
    return record


def _add_text_files(subject_dir: Path) -> None:
    for role, text in TEXT_FILES.items():
        path = subject_dir / ROLE_CANDIDATES[role][0]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def _refuse_stat(monkeypatch: pytest.MonkeyPatch, *refused: Path) -> None:
    """``stat()`` of the given paths fails as it does in a directory without search
    permission. Replacing ``Path.stat`` lets the tests run as root, whom modes do not stop."""
    stat = Path.stat

    def refuse(self: Path, *args: object, **kwargs: object) -> object:
        if self in refused:
            raise PermissionError(13, "Permission denied", str(self))
        return stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", refuse)


# ---------------------------------------------------------------------------- layouts


@pytest.mark.parametrize("layout", LAYOUTS)
def test_each_layout_is_detected_with_its_root_subjects_directory_and_pipeline(
    tmp_path: Path, layout: Layout
) -> None:
    tree = write_tree(tmp_path / "study", layout, LABELS)
    source = LocalSource(tree)
    root = tree.resolve()
    pipeline, version = PIPELINE[layout]
    assert source.dataset == DatasetRecord(
        source="local",
        identity=str(root),
        accession=None,
        root=str(root),
        fs_root=FS_ROOT[layout],
        layout=layout,
        pipeline=pipeline,
        pipeline_version=version,
        template_location=None,
        derivative_ref=None,
        snapshot_tag=None,
    )
    assert source.kind == "local" and isinstance(source, Source)
    assert source.subjects_dir == root / FS_ROOT[layout]
    assert list(source.datasets()) == [source.dataset]
    assert list(source.datasets(only=["ds000102"])) == [source.dataset]
    assert detect_layout(tree) == (source.dataset, ())

    subjects = list(source.subjects(source.dataset))
    assert [s.subject_label for s in subjects] == list(LABELS)
    for subject in subjects:
        assert subject.dataset is source.dataset
        expected = subject.subject_label
        if FS_ROOT[layout] != ".":
            expected = f"{FS_ROOT[layout]}/{expected}"
        assert subject.fs_dir == expected
        assert (root / subject.fs_dir).is_dir()
        assert subject.problem is None
        assert set(subject.files) == set(TIER_C)


def test_the_root_is_resolved_and_a_string_path_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree = write_tree(tmp_path / "real" / "study", "fmriprep", LABELS)
    (tmp_path / "link").symlink_to(tmp_path / "real")
    expected = str(tree.resolve())
    assert LocalSource(tmp_path / "link" / "study").dataset.root == expected
    assert LocalSource(str(tree) + "/").dataset.root == expected
    assert LocalSource(tmp_path / "real" / "other" / ".." / "study").dataset.root == expected
    monkeypatch.chdir(tree)
    assert LocalSource(".").dataset.root == expected
    monkeypatch.chdir(tree / "sourcedata")
    assert LocalSource("..").dataset.root == expected
    assert LocalSource(Path("../sourcedata/..")).dataset.root == expected


def test_the_new_fmriprep_layout_wins_over_the_old_one_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    tree = write_tree(tmp_path / "study", "fmriprep", ["sub-01"])
    write_light_subject(tree / "freesurfer" / "sub-09")
    with caplog.at_level(logging.WARNING, logger="surfcensus"):
        source = LocalSource(tree)
    assert (source.dataset.layout, source.dataset.fs_root) == ("fmriprep", "sourcedata/freesurfer")
    assert _labels(source) == ["sub-01"]
    (warning,) = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert str(tree.resolve() / "freesurfer") in warning


def test_one_layout_gives_no_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    tree = write_tree(tmp_path / "study", "fmriprep", ["sub-01"])
    (tree / "freesurfer").mkdir()  # there, but without a subject: rule 3 does not match
    with caplog.at_level(logging.INFO, logger="surfcensus"):
        LocalSource(tree)
    assert [r.levelno for r in caplog.records] == [logging.INFO]
    assert "sourcedata/freesurfer" in caplog.records[0].getMessage()


def test_subjects_in_freesurfer_win_over_subjects_in_the_path_itself(tmp_path: Path) -> None:
    tree = write_tree(tmp_path / "study", "bids_derivatives", ["sub-01"])
    write_light_subject(tree / "sub-09")
    source = LocalSource(tree)
    assert (source.dataset.layout, source.dataset.fs_root) == ("bids_derivatives", "freesurfer")
    assert _labels(source) == ["sub-01"]

    tree = write_tree(tmp_path / "other", "fmriprep", ["sub-01"])
    write_light_subject(tree / "sub-09")
    source = LocalSource(tree)
    assert (source.dataset.layout, source.dataset.fs_root) == ("fmriprep", "sourcedata/freesurfer")
    assert _labels(source) == ["sub-01"]


def test_an_fmriprep_folder_next_to_freesurfer_makes_the_layout_legacy(tmp_path: Path) -> None:
    tree = write_tree(tmp_path / "study", "bids_derivatives", ["sub-01"])
    assert LocalSource(tree).dataset.layout == "bids_derivatives"
    (tree / "fmriprep").write_text("a file of that name is not the folder\n")
    assert LocalSource(tree).dataset.layout == "bids_derivatives"
    (tree / "fmriprep").unlink()
    (tree / "fmriprep").mkdir()
    record = LocalSource(tree).dataset
    assert (record.layout, record.pipeline, record.pipeline_version) == (
        "fmriprep_legacy",
        "fmriprep",
        None,
    )


def test_the_fmriprep_folder_of_the_old_layout_can_be_the_path(tmp_path: Path) -> None:
    tree = write_tree(tmp_path / "study", "fmriprep_legacy", LABELS)
    from_parent = LocalSource(tree).dataset
    source = LocalSource(tree / "fmriprep")
    assert source.dataset == from_parent
    assert source.dataset.root == str(tree.resolve())
    assert (source.dataset.layout, source.dataset.fs_root) == ("fmriprep_legacy", "freesurfer")
    assert source.dataset.pipeline_version == "23.2.0"
    assert _labels(source) == list(LABELS)

    # a link of another name to the folder, and a link of that name to a folder of another
    (tmp_path / "shortcut").symlink_to(tree / "fmriprep")
    assert LocalSource(tmp_path / "shortcut").dataset == from_parent
    renamed = write_tree(tmp_path / "renamed", "fmriprep_legacy", ["sub-07"])
    (renamed / "fmriprep").rename(tmp_path / "fp-v1")
    (renamed / "fmriprep").symlink_to(tmp_path / "fp-v1")
    record = LocalSource(renamed / "fmriprep").dataset
    assert (record.root, record.layout) == (str(renamed.resolve()), "fmriprep_legacy")
    assert record.pipeline_version == "23.2.0"


def test_a_folder_named_fmriprep_prefers_the_subjects_next_to_it(tmp_path: Path) -> None:
    tree = write_tree(tmp_path / "study", "fmriprep_legacy", ["sub-01"])
    write_light_subject(tree / "fmriprep" / "sub-09")  # rule 5 would find this one
    source = LocalSource(tree / "fmriprep")
    assert source.dataset.layout == "fmriprep_legacy"
    assert _labels(source) == ["sub-01"]


def test_a_folder_named_fmriprep_without_subjects_next_to_it_is_a_subjects_directory(
    tmp_path: Path,
) -> None:
    tree = write_tree(tmp_path / "fmriprep", "subjects_dir", ["sub-01"])
    (tmp_path / "freesurfer").mkdir()
    record = LocalSource(tree).dataset
    assert (record.layout, record.root, record.fs_root) == (
        "subjects_dir",
        str(tree.resolve()),
        ".",
    )


def test_no_layout_is_an_error_that_says_where_it_looked(tmp_path: Path) -> None:
    tree = tmp_path / "study"
    write_light_subject(tree / "fsaverage")  # a template is not a subject
    (tree / "sub-01" / "anat").mkdir(parents=True)  # fMRIPrep's own output is not one either
    (tree / "sub-02" / "mri").mkdir(parents=True)  # nor is a directory without surf/
    (tree / "freesurfer" / "fsaverage5" / "mri").mkdir(parents=True)
    (tree / "freesurfer" / "fsaverage5" / "surf").mkdir()
    with pytest.raises(SourceError) as caught:
        LocalSource(tree)
    assert caught.value.reason == "layout"
    detail = caught.value.detail
    assert str(tree.resolve()) in detail
    for place in ("the path itself", "sourcedata/freesurfer/", " freesurfer/", "../freesurfer/"):
        assert place in detail
    assert "own folders" in detail
    with pytest.raises(SourceError, match=r"^layout: "):
        detect_layout(tree)
    with pytest.raises(SourceError, match=r"^layout: "):
        local(tmp_path)


@pytest.mark.parametrize("kind", ["absent", "file", "dangling link"])
def test_a_path_that_is_not_a_directory_is_not_found(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "study"
    if kind == "file":
        path.write_text("x")
    elif kind == "dangling link":
        path.symlink_to(tmp_path / "gone")
    with pytest.raises(SourceError) as caught:
        LocalSource(path)
    assert caught.value.reason == "not found"
    assert str(path) in caught.value.detail


def _refuse_listdir(monkeypatch: pytest.MonkeyPatch, refused: Path) -> None:
    listdir = os.listdir

    def refuse(path: str | os.PathLike[str]) -> list[str]:
        if Path(path) == refused:
            raise PermissionError(13, "Permission denied", str(path))
        return listdir(path)

    monkeypatch.setattr(os, "listdir", refuse)


def test_a_place_that_cannot_be_listed_is_unreadable_not_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree = write_tree(tmp_path / "study", "fmriprep", ["sub-01"])
    source = LocalSource(tree)
    _refuse_listdir(monkeypatch, tree / "sourcedata" / "freesurfer")
    with pytest.raises(SourceError) as caught:
        LocalSource(tree)
    assert caught.value.reason == "unreadable"
    assert caught.value.detail == f"{tree / 'sourcedata' / 'freesurfer'}: Permission denied"
    with pytest.raises(SourceError) as caught:
        source.subjects(source.dataset)
    assert caught.value.reason == "unreadable"
    assert caught.value.detail == f"{tree / 'sourcedata' / 'freesurfer'}: Permission denied"


def test_a_folder_that_is_not_used_need_not_be_listable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree = write_tree(tmp_path / "study", "fmriprep", ["sub-01"])
    write_light_subject(tree / "freesurfer" / "sub-09")
    _refuse_listdir(monkeypatch, tree / "freesurfer")
    source = LocalSource(tree)
    assert (source.dataset.layout, source.dataset.fs_root) == ("fmriprep", "sourcedata/freesurfer")
    # when it is the only place with subjects, the refusal is reported
    shutil.rmtree(tree / "sourcedata")
    with pytest.raises(SourceError, match=r"^unreadable: .*freesurfer: Permission denied$"):
        LocalSource(tree)


def test_a_directory_whose_folders_cannot_be_examined_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree = write_tree(tmp_path / "study", "subjects_dir", ["sub-01"])
    _refuse_stat(monkeypatch, tree / "sub-01" / "mri")
    with pytest.raises(SourceError) as caught:
        LocalSource(tree)
    assert caught.value.reason == "layout"
    assert caught.value.detail.endswith("Could not be examined: sub-01")


def test_a_subjects_directory_without_search_permission(tmp_path: Path) -> None:
    if not file_modes_apply(tmp_path):
        pytest.skip("file modes do not restrict this user")
    tree = write_tree(tmp_path / "study", "subjects_dir", ["sub-01", "sub-02"])
    (tree / "sub-02").chmod(0)
    try:
        subjects = list(LocalSource(tree).subjects(LocalSource(tree).dataset))
        assert [(s.subject_label, s.problem) for s in subjects] == [
            ("sub-01", None),
            ("sub-02", "unreadable: Permission denied"),
        ]
        (tree / "sub-01").chmod(0)
        with pytest.raises(SourceError, match=r"^layout: .*Could not be examined: sub-01, sub-02$"):
            LocalSource(tree)
        tree.chmod(0)
        with pytest.raises(SourceError, match=r"^unreadable: "):
            LocalSource(tree)
    finally:
        tree.chmod(0o755)
        (tree / "sub-01").chmod(0o755)
        (tree / "sub-02").chmod(0o755)


# ---------------------------------------------------------------------------- one subject


def test_one_subject_directory_is_its_parent_restricted_to_that_name(tmp_path: Path) -> None:
    tree = write_tree(tmp_path / "study", "subjects_dir", LABELS, with_template=True)
    source = LocalSource(tree / "sub-02")
    record, implied = detect_layout(tree / "sub-02")
    assert source.dataset == record
    assert implied == ("sub-02",)
    assert (record.layout, record.root, record.fs_root) == (
        "subjects_dir",
        str(tree.resolve()),
        ".",
    )
    assert (record.pipeline, record.pipeline_version) == ("freesurfer", None)
    assert record.template_location == "fsaverage"
    assert _labels(source) == ["sub-02"]
    assert _subject(source, "sub-02").fs_dir == "sub-02"
    assert source.config().subjects == ("sub-02",)


def test_one_subject_inside_an_fmriprep_tree_wins_over_the_other_rules(tmp_path: Path) -> None:
    tree = write_tree(tmp_path / "study", "fmriprep", LABELS)
    subject_dir = tree / "sourcedata" / "freesurfer" / "sub-03"
    write_light_subject(subject_dir / "freesurfer" / "sub-09")  # rule 3 would match inside it
    source = LocalSource(subject_dir)
    assert source.dataset.root == str(subject_dir.parent.resolve())
    assert (source.dataset.layout, source.dataset.fs_root) == ("subjects_dir", ".")
    assert (source.dataset.pipeline, source.dataset.pipeline_version) == ("freesurfer", None)
    assert _labels(source) == ["sub-03"]


def test_the_implied_glob_is_escaped(tmp_path: Path) -> None:
    for name in ("sub-[1]", "sub-?", "sub-*", "sub-1", "sub-x"):
        write_light_subject(tmp_path / name)
    for name, glob in (("sub-[1]", "sub-[[]1]"), ("sub-?", "sub-[?]"), ("sub-*", "sub-[*]")):
        source = LocalSource(tmp_path / name)
        assert detect_layout(tmp_path / name)[1] == (glob,)
        assert _labels(source) == [name]


def test_one_subject_that_is_a_link_keeps_the_name_it_is_listed_under(tmp_path: Path) -> None:
    write_light_subject(tmp_path / "archive" / "a1b2c3")
    (tmp_path / "subjects").mkdir()
    (tmp_path / "subjects" / "sub-01").symlink_to(tmp_path / "archive" / "a1b2c3")
    source = LocalSource(tmp_path / "subjects" / "sub-01")
    assert source.dataset.root == str((tmp_path / "subjects").resolve())
    assert _labels(source) == ["sub-01"]
    assert LocalSource(tmp_path / "subjects").subjects_dir == source.subjects_dir
    assert _labels(LocalSource(tmp_path / "subjects")) == ["sub-01"]


def test_globs_given_with_one_subject_must_match_it(tmp_path: Path) -> None:
    tree = write_tree(tmp_path / "study", "subjects_dir", LABELS)
    source = LocalSource(tree / "sub-02", subjects=["sub-0[12]"])
    assert _labels(source) == ["sub-02"]
    assert source.config().subjects == ("sub-02",)
    with pytest.raises(SourceError) as caught:
        LocalSource(tree / "sub-02", subjects=["sub-01", "sub-03"])
    assert caught.value.reason == "not found"
    assert "sub-02" in caught.value.detail and "sub-01, sub-03" in caught.value.detail


def test_a_template_directory_as_path_is_not_a_subject(tmp_path: Path) -> None:
    tree = write_tree(tmp_path / "study", "subjects_dir", LABELS, with_template=True)
    with pytest.raises(SourceError, match=r"^layout: "):
        LocalSource(tree / "fsaverage")


# ---------------------------------------------------------------------------- listing


def test_templates_are_skipped_by_name_and_by_link_target(tmp_path: Path) -> None:
    tree = write_tree(tmp_path / "study", "subjects_dir", ["sub-01"], with_template=True)
    for name in ("cvs_avg35", "cvs_avg35_inMNI152", "V1_average", "lh.EC_average", "fsaverage_sym"):
        write_light_subject(tree / name)
    write_light_subject(tmp_path / "install" / "subjects" / "fsaverage6")
    (tree / "average").symlink_to(tmp_path / "install" / "subjects" / "fsaverage6")
    (tree / "my-template").symlink_to("fsaverage")
    write_light_subject(tree / "sub-01-moved")
    (tree / "sub-02").symlink_to(tmp_path / "elsewhere" / "sub-02")
    write_light_subject(tmp_path / "elsewhere" / "sub-02")
    # a link to a subject is a subject under the link's name; a link to a template is not
    assert _labels(LocalSource(tree)) == ["sub-01", "sub-01-moved", "sub-02"]


def test_only_directories_with_mri_and_surf_are_subjects(tmp_path: Path) -> None:
    tree = write_tree(tmp_path / "study", "fmriprep", ["sub-02", "sub-10", "sub-01"])
    subjects_dir = tree / "sourcedata" / "freesurfer"
    (subjects_dir / "logs").mkdir()
    (subjects_dir / ".datalad" / "mri").mkdir(parents=True)
    (subjects_dir / "sub-04.html").write_text("<html></html>\n")
    (subjects_dir / "sub-05" / "surf").mkdir(parents=True)
    (subjects_dir / "sub-06").symlink_to(tmp_path / "gone")
    assert _labels(LocalSource(tree)) == ["sub-01", "sub-02", "sub-10"]


def test_subjects_are_in_ascending_order_of_their_label(tmp_path: Path) -> None:
    names = ["sub-10", "sub-2", "Sub-03", "sub-01", "sub-01_ses-02", "sub-01.long.base", "a"]
    tree = write_tree(tmp_path / "study", "subjects_dir", names)
    labels = _labels(LocalSource(tree))
    assert labels == sorted(names)
    # longitudinal and session directories are ordinary subjects under those names
    assert {"sub-01.long.base", "sub-01_ses-02"} <= set(labels)


def test_incomplete_subjects_are_listed_with_the_files_they_have(tmp_path: Path) -> None:
    tree = write_tree(
        tmp_path / "study", "subjects_dir", ["sub-01", "sub-02", "sub-03"], incomplete=["sub-02"]
    )
    for name in ("mri", "surf"):  # a recon-all run that stopped before writing anything
        (tree / "sub-00" / name).mkdir(parents=True)
    (tree / "sub-03" / "mri" / "T1.mgz").unlink()
    subjects = {s.subject_label: s for s in LocalSource(tree).subjects(LocalSource(tree).dataset)}
    assert list(subjects) == ["sub-00", "sub-01", "sub-02", "sub-03"]
    assert all(s.problem is None for s in subjects.values())
    assert subjects["sub-00"].files == {}
    assert set(subjects["sub-01"].files) == set(TIER_C)
    assert set(subjects["sub-02"].files) == set(TIER_C) - {FileRole.LH_PIAL}
    assert set(subjects["sub-03"].files) == set(TIER_C) - {FileRole.T1}


@pytest.mark.parametrize("refused", ["mri", "surf/lh.white", "scripts/recon-all.log"])
def test_a_subject_that_cannot_be_examined_is_listed_with_its_problem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, refused: str
) -> None:
    tree = write_tree(tmp_path / "study", "subjects_dir", LABELS)
    source = LocalSource(tree)
    _refuse_stat(monkeypatch, tree / "sub-02" / refused)
    subjects = list(source.subjects(source.dataset))
    assert [s.subject_label for s in subjects] == list(LABELS)
    assert [s.problem for s in subjects] == [None, "unreadable: Permission denied", None]
    assert subjects[1].files == {}
    assert subjects[1].fs_dir == "sub-02"
    assert set(subjects[0].files) == set(TIER_C) == set(subjects[2].files)
    with source.lease(subjects[1], TIER_C) as paths:
        assert paths == {}


def test_two_names_for_one_directory_are_listed_once_under_the_first(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    tree = write_tree(tmp_path / "study", "subjects_dir", ["sub-01", "sub-02"])
    (tree / "sub-01b").symlink_to("sub-01")
    (tree / "alias").symlink_to(tree / "sub-02")
    with caplog.at_level(logging.WARNING, logger="surfcensus"):
        labels = _labels(LocalSource(tree))
    assert labels == ["alias", "sub-01"]
    warnings = sorted(r.getMessage() for r in caplog.records)
    assert len(warnings) == 2
    assert warnings[0].startswith("sub-01b is the same directory as sub-01")
    assert warnings[1].startswith("sub-02 is the same directory as alias")

    # a glob selects before duplicates are dropped, so a name can be asked for
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="surfcensus"):
        assert _labels(LocalSource(tree, subjects=["sub-0?"])) == ["sub-01", "sub-02"]
    assert caplog.records == []


def test_listing_is_repeatable_and_opens_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree = write_tree(tmp_path / "study", "fmriprep", LABELS, with_template=True)
    _add_text_files(tree / "sourcedata" / "freesurfer" / "sub-01")
    source = LocalSource(tree)
    first = list(source.subjects(source.dataset))

    def refuse(*args: object, **kwargs: object) -> object:
        raise AssertionError(f"opened {args[0]}")

    monkeypatch.setattr(builtins, "open", refuse)
    monkeypatch.setattr(io, "open", refuse)
    monkeypatch.setattr(os, "open", refuse)
    monkeypatch.setattr(Path, "open", refuse)
    assert list(source.subjects(source.dataset)) == first
    assert set(first[0].files) == {*TIER_C, *TEXT_FILES}


def test_the_dataset_given_to_subjects_is_the_one_on_the_records(tmp_path: Path) -> None:
    tree = write_tree(tmp_path / "study", "subjects_dir", LABELS)
    source = LocalSource(tree)
    stored = dataclasses.replace(source.dataset, identity="/where/the/data/was")
    subjects = list(source.subjects(stored))
    assert [s.subject_label for s in subjects] == list(LABELS)
    assert all(s.dataset is stored for s in subjects)


def test_a_subjects_directory_that_went_away_is_not_found(tmp_path: Path) -> None:
    tree = write_tree(tmp_path / "study", "fmriprep", LABELS)
    source = LocalSource(tree)
    shutil.rmtree(tree / "sourcedata")
    with pytest.raises(SourceError) as caught:
        source.subjects(source.dataset)
    assert caught.value.reason == "not found"


# ---------------------------------------------------------------------------- globs


def test_globs_are_a_union_on_the_directory_name(tmp_path: Path) -> None:
    names = ["sub-01", "sub-02", "sub-10", "sub-11", "pilot-01", "SUB-01"]
    tree = write_tree(tmp_path / "study", "subjects_dir", names)
    assert _labels(LocalSource(tree, subjects=["sub-0*"])) == ["sub-01", "sub-02"]
    assert _labels(LocalSource(tree, subjects=["sub-1?", "pilot-*"])) == [
        "pilot-01",
        "sub-10",
        "sub-11",
    ]
    assert _labels(LocalSource(tree, subjects=("sub-01", "sub-0[12]", "sub-01"))) == [
        "sub-01",
        "sub-02",
    ]
    assert _labels(LocalSource(tree, subjects=["*-01"])) == ["SUB-01", "pilot-01", "sub-01"]
    assert _labels(local(tree, subjects=["sub-0*"])) == ["sub-01", "sub-02"]
    # one glob given as a string is one glob, not its characters
    assert _labels(LocalSource(tree, subjects="sub-1*")) == ["sub-10", "sub-11"]
    assert _labels(LocalSource(tree)) == sorted(names)


def test_globs_do_not_select_templates_or_other_folders(tmp_path: Path) -> None:
    tree = write_tree(tmp_path / "study", "subjects_dir", ["sub-01"], with_template=True)
    (tree / "logs").mkdir()
    assert _labels(LocalSource(tree, subjects=["*"])) == ["sub-01"]
    with pytest.raises(SourceError, match=r"^not found: "):
        LocalSource(tree, subjects=["fsaverage"]).subjects(LocalSource(tree).dataset)


def test_a_glob_that_matches_nothing_is_an_error_naming_it_and_what_exists(tmp_path: Path) -> None:
    tree = write_tree(tmp_path / "study", "subjects_dir", LABELS)
    source = LocalSource(tree, subjects=["sub-0*", "Sub-01"])  # case matters
    with pytest.raises(SourceError) as caught:
        source.subjects(source.dataset)
    assert caught.value.reason == "not found"
    assert "'Sub-01'" in caught.value.detail
    assert "sub-01, sub-02, sub-03" in caught.value.detail
    assert "more" not in caught.value.detail


def test_a_glob_that_matches_nothing_names_at_most_ten_subjects(tmp_path: Path) -> None:
    names = [f"sub-{i:02d}" for i in range(1, 14)]
    for name in names:
        for folder in ("mri", "surf"):
            (tmp_path / name / folder).mkdir(parents=True)
    source = LocalSource(tmp_path, subjects=["ctl-*"])
    with pytest.raises(SourceError) as caught:
        source.subjects(source.dataset)
    detail = caught.value.detail
    assert "'ctl-*'" in detail
    assert ", ".join(names[:10]) + " and 3 more" in detail
    assert "sub-11" not in detail


# ---------------------------------------------------------------------------- file refs


def test_file_refs_say_where_and_what_without_opening(tmp_path: Path) -> None:
    tree = write_tree(tmp_path / "study", "fmriprep", ["sub-01"])
    subject_dir = tree / "sourcedata" / "freesurfer" / "sub-01"
    _add_text_files(subject_dir)
    subject = _subject(LocalSource(tree), "sub-01")
    assert list(subject.files) == [role for role in FileRole if role in subject.files]
    assert set(subject.files) == {*TIER_C, *TEXT_FILES}
    assert set(TIER_A) - set(subject.files) == {FileRole.RECON_LOG_LH, FileRole.RECON_LOG_RH}
    for role, ref in subject.files.items():
        path = subject_dir / ROLE_CANDIDATES[role][0]
        assert ref.role is role
        assert ref.location == ROLE_CANDIDATES[role][0]
        assert ref.size == path.stat().st_size
        assert ref.etag_or_mtime == str(path.stat().st_mtime_ns)
        assert ref.last_modified is None
    assert subject.files[FileRole.RECON_LOG].size == len(TEXT_FILES[FileRole.RECON_LOG])
    # a variant is recorded for the roles that have several candidate names
    variants = {role: ref.variant for role, ref in subject.files.items() if ref.variant}
    assert variants == {FileRole.LH_PIAL: "lh.pial", FileRole.RH_PIAL: "rh.pial"}


def test_the_pial_variant_of_freesurfer_7(tmp_path: Path) -> None:
    tree = write_tree(tmp_path / "study", "subjects_dir", ["sub-01", "sub-02", "sub-03"])
    # as recon-all 7 leaves it: lh.pial is a relative link to lh.pial.T1
    linked = tree / "sub-01" / "surf"
    (linked / "lh.pial").rename(linked / "lh.pial.T1")
    (linked / "lh.pial").symlink_to("lh.pial.T1")
    os.utime(linked / "lh.pial.T1", ns=(1_600_000_000_000_000_000, 1_600_000_000_123_456_789))
    # as the derivatives bucket has it: lh.pial.T1 and no lh.pial
    only = tree / "sub-02" / "surf"
    (only / "lh.pial").rename(only / "lh.pial.T1")
    (only / "rh.pial").rename(only / "rh.pial.T2")
    # a link into another directory reads under the candidate's own name
    foreign = tree / "sub-03" / "surf"
    (tmp_path / "edits").mkdir()
    (foreign / "lh.pial").rename(tmp_path / "edits" / "by-hand")
    (foreign / "lh.pial").symlink_to(tmp_path / "edits" / "by-hand")

    subjects = {s.subject_label: s for s in LocalSource(tree).subjects(LocalSource(tree).dataset)}
    ref = subjects["sub-01"].files[FileRole.LH_PIAL]
    assert (ref.location, ref.variant) == ("surf/lh.pial", "lh.pial.T1")
    assert ref.size == (linked / "lh.pial.T1").stat().st_size  # of the file, not of the link
    assert ref.size != (linked / "lh.pial").lstat().st_size
    assert ref.etag_or_mtime == "1600000000123456789"
    ref = subjects["sub-02"].files[FileRole.LH_PIAL]
    assert (ref.location, ref.variant) == ("surf/lh.pial.T1", "lh.pial.T1")
    ref = subjects["sub-02"].files[FileRole.RH_PIAL]
    assert (ref.location, ref.variant) == ("surf/rh.pial.T2", "rh.pial.T2")
    ref = subjects["sub-03"].files[FileRole.LH_PIAL]
    assert (ref.location, ref.variant) == ("surf/lh.pial", "lh.pial")
    assert all(set(s.files) == set(TIER_C) for s in subjects.values())

    # the name is the one render.json records for the same file
    source = LocalSource(tree)
    for label in ("sub-01", "sub-02", "sub-03"):
        with source.lease(subjects[label], TIER_C) as paths:
            info = render_subject(
                paths, tmp_path / "media" / label, RenderConfig(panel_px=64), video=False
            )
        for role in (FileRole.LH_PIAL, FileRole.RH_PIAL):
            assert info.inputs[role.value].name == subjects[label].files[role].variant
            assert info.inputs[role.value].size == subjects[label].files[role].size


# ---------------------------------------------------------------------------- reading


@pytest.fixture
def text_source(tmp_path: Path) -> tuple[LocalSource, SubjectRecord, Path]:
    tree = write_tree(tmp_path / "study", "fmriprep", ["sub-01", "sub-02"])
    subject_dir = tree / "sourcedata" / "freesurfer" / "sub-01"
    _add_text_files(subject_dir)
    source = LocalSource(tree)
    return source, _subject(source, "sub-01"), subject_dir


def test_read_bytes_returns_the_file_or_an_inclusive_range(
    text_source: tuple[LocalSource, SubjectRecord, Path],
) -> None:
    source, subject, subject_dir = text_source
    log = TEXT_FILES[FileRole.RECON_LOG].encode()
    for role, text in TEXT_FILES.items():
        assert source.read_bytes(subject, role) == text.encode()
    t1 = source.read_bytes(subject, FileRole.T1)
    assert t1 == (subject_dir / "mri" / "T1.mgz").read_bytes()
    read = source.read_bytes
    assert read(subject, FileRole.RECON_LOG, byte_range=(0, 0)) == log[:1]
    assert read(subject, FileRole.RECON_LOG, byte_range=(0, 99)) == log[:100]
    assert read(subject, FileRole.RECON_LOG, byte_range=(21, 62)) == log[21:63]
    assert read(subject, FileRole.RECON_LOG, byte_range=(len(log) - 1, len(log) - 1)) == log[-1:]
    # the head of a log that is shorter than the head asked for is the whole log
    assert read(subject, FileRole.RECON_LOG, byte_range=(0, 65535)) == log
    assert read(subject, FileRole.RECON_LOG, byte_range=(len(log) - 5, len(log) + 99)) == log[-5:]
    assert read(subject, FileRole.RECON_LOG, byte_range=(len(log), len(log) + 9)) == b""


@pytest.mark.parametrize("byte_range", [(-1, 4), (5, 4), (-3, -1)])
def test_read_bytes_refuses_a_range_that_is_not_one(
    text_source: tuple[LocalSource, SubjectRecord, Path], byte_range: tuple[int, int]
) -> None:
    source, subject, _ = text_source
    with pytest.raises(ValueError, match="byte_range"):
        source.read_bytes(subject, FileRole.RECON_LOG, byte_range=byte_range)


def test_read_bytes_of_a_role_the_subject_lacks_is_not_found(
    text_source: tuple[LocalSource, SubjectRecord, Path],
) -> None:
    source, subject, _ = text_source
    with pytest.raises(SourceError) as caught:
        source.read_bytes(subject, FileRole.RECON_LOG_LH)
    assert caught.value.reason == "not found"
    assert "recon_log_lh" in caught.value.detail


def test_lease_hands_out_the_files_in_place(
    text_source: tuple[LocalSource, SubjectRecord, Path],
) -> None:
    source, subject, subject_dir = text_source
    with source.lease(subject, TIER_C) as paths:
        assert list(paths) == [role for role in FileRole if role in TIER_C]
        for role, path in paths.items():
            assert path == subject_dir / ROLE_CANDIDATES[role][0]
            assert path.is_file()
    assert all(path.is_file() for path in paths.values())  # nothing is removed on exit

    # roles the subject lacks are absent, and asking for them is not an error
    with source.lease(subject, [FileRole.LH_THICKNESS, FileRole.RECON_LOG, FileRole.T1]) as paths:
        assert list(paths) == [FileRole.T1, FileRole.RECON_LOG]
    with source.lease(subject, ()) as paths:
        assert paths == {}
    with source.lease(subject, frozenset(FileRole)) as paths:
        assert set(paths) == set(subject.files)

    incomplete = dataclasses.replace(
        subject, files={r: f for r, f in subject.files.items() if r is not FileRole.LH_PIAL}
    )
    with source.lease(incomplete, RENDER_ROLES) as paths:
        assert set(paths) == set(RENDER_ROLES) - {FileRole.LH_PIAL}


def test_a_leased_subject_renders(tmp_path: Path) -> None:
    tree = write_tree(tmp_path / "study", "bids_derivatives", ["sub-01"])
    source = LocalSource(tree)
    subject = _subject(source, "sub-01")
    before = snapshot_tree(tree)
    with source.lease(subject, TIER_C) as paths:
        info = render_subject(paths, tmp_path / "media", RenderConfig(panel_px=64), video=False)
    assert snapshot_tree(tree) == before
    assert set(info.inputs) == {role.value for role in TIER_C}
    for role in TIER_C:
        assert info.inputs[role.value].size == subject.files[role].size


@pytest.mark.parametrize("change", ["mtime", "size", "removed", "replaced by a directory"])
def test_a_file_that_changed_after_the_listing_is_refused(
    text_source: tuple[LocalSource, SubjectRecord, Path], change: str
) -> None:
    source, subject, subject_dir = text_source
    path = subject_dir / "scripts" / "recon-all.log"
    stat = path.stat()
    if change == "mtime":
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        assert path.stat().st_size == stat.st_size
    elif change == "size":
        with path.open("a") as stream:
            stream.write("one more line\n")
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        assert path.stat().st_mtime_ns == stat.st_mtime_ns
    else:
        path.unlink()
        if change == "replaced by a directory":
            path.mkdir()

    with pytest.raises(SourceError) as caught:
        source.read_bytes(subject, FileRole.RECON_LOG)
    reason = "unreadable" if change == "replaced by a directory" else "changed"
    assert caught.value.reason == reason
    assert caught.value.detail.startswith("scripts/recon-all.log")
    assert str(subject_dir) not in caught.value.detail
    with pytest.raises(SourceError) as caught:
        source.read_bytes(subject, FileRole.RECON_LOG, byte_range=(0, 9))
    assert caught.value.reason == reason

    with pytest.raises(SourceError) as caught, source.lease(subject, TIER_A):
        raise AssertionError("the lease was entered")
    assert caught.value.reason == "changed"
    assert caught.value.detail.startswith("scripts/recon-all.log")

    # the other files, and a lease that does not ask for the changed one, are unaffected
    assert (
        source.read_bytes(subject, FileRole.BUILD_STAMP)
        == TEXT_FILES[FileRole.BUILD_STAMP].encode()
    )
    with source.lease(subject, TIER_C) as paths:
        assert set(paths) == set(TIER_C)
    # a new listing picks the change up
    fresh = _subject(source, "sub-01")
    if change in ("mtime", "size"):
        assert fresh.files[FileRole.RECON_LOG] != subject.files[FileRole.RECON_LOG]
        assert source.read_bytes(fresh, FileRole.RECON_LOG) == path.read_bytes()
    else:
        assert FileRole.RECON_LOG not in fresh.files


def test_a_touched_link_target_counts_as_changed(tmp_path: Path) -> None:
    tree = write_tree(tmp_path / "study", "subjects_dir", ["sub-01"])
    surf = tree / "sub-01" / "surf"
    (surf / "lh.pial").rename(surf / "lh.pial.T1")
    (surf / "lh.pial").symlink_to("lh.pial.T1")
    source = LocalSource(tree)
    subject = _subject(source, "sub-01")
    with source.lease(subject, [FileRole.LH_PIAL]) as paths:
        assert paths == {FileRole.LH_PIAL: surf / "lh.pial"}
    stat = (surf / "lh.pial.T1").stat()
    os.utime(surf / "lh.pial.T1", ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    with (
        pytest.raises(SourceError, match=r"^changed: surf/lh.pial "),
        source.lease(subject, TIER_C),
    ):
        pass


def test_a_file_that_cannot_be_opened_is_unreadable(
    text_source: tuple[LocalSource, SubjectRecord, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    source, subject, subject_dir = text_source
    target = subject_dir / "scripts" / "build-stamp.txt"
    opener = Path.open

    def refuse(self: Path, *args: object, **kwargs: object) -> object:
        if self == target:
            raise PermissionError(13, "Permission denied", str(self))
        return opener(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "open", refuse)
    with pytest.raises(SourceError) as caught:
        source.read_bytes(subject, FileRole.BUILD_STAMP)
    assert (caught.value.reason, caught.value.detail) == (
        "unreadable",
        "scripts/build-stamp.txt: Permission denied",
    )
    assert source.read_bytes(subject, FileRole.RECON_LOG).startswith(b"line 0000")


def test_files_are_read_from_where_the_source_is_now(tmp_path: Path) -> None:
    tree = write_tree(tmp_path / "study", "fmriprep", ["sub-01"])
    _add_text_files(tree / "sourcedata" / "freesurfer" / "sub-01")
    listed = _subject(LocalSource(tree), "sub-01")
    moved = tmp_path / "elsewhere" / "study"
    moved.parent.mkdir()
    shutil.copytree(tree, moved, symlinks=True)  # copytree keeps modification times
    shutil.rmtree(tree)
    source = LocalSource(moved, identity=listed.dataset.identity)
    # a record that still names the old root, as one rebuilt from a catalog does after a move
    assert listed.dataset.root != source.dataset.root
    assert (
        source.read_bytes(listed, FileRole.BUILD_STAMP) == TEXT_FILES[FileRole.BUILD_STAMP].encode()
    )
    with source.lease(listed, TIER_C) as paths:
        assert all(path.is_relative_to(moved) and path.is_file() for path in paths.values())


def test_the_source_never_changes_the_tree(tmp_path: Path) -> None:
    tree = write_tree(
        tmp_path / "study", "fmriprep", LABELS, with_template=True, incomplete=["sub-03"]
    )
    subjects_dir = tree / "sourcedata" / "freesurfer"
    _add_text_files(subjects_dir / "sub-01")
    (subjects_dir / "sub-01b").symlink_to("sub-01")
    before = snapshot_tree(tmp_path)

    source = LocalSource(tree)
    assert snapshot_tree(tmp_path) == before
    subjects = list(source.subjects(source.dataset))
    assert snapshot_tree(tmp_path) == before
    for subject in subjects:
        for role in subject.files:
            source.read_bytes(subject, role)
            source.read_bytes(subject, role, byte_range=(0, 15))
        with source.lease(subject, tuple(FileRole)) as paths:
            assert set(paths) == set(subject.files)
    assert snapshot_tree(tmp_path) == before
    assert source.template_dir(source.dataset) is not None
    source.config()
    LocalSource(subjects_dir / "sub-02").subjects(source.dataset)
    with pytest.raises(SourceError):
        LocalSource(tree, subjects=["nobody"]).subjects(source.dataset)
    with pytest.raises(SourceError):
        LocalSource(tree / "sub-01")  # fMRIPrep's own folder of that subject
    assert snapshot_tree(tmp_path) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["study"]


_WRITERS = {
    "write_text",
    "write_bytes",
    "mkdir",
    "makedirs",
    "touch",
    "unlink",
    "remove",
    "rmdir",
    "rmtree",
    "rename",
    "renames",
    "symlink_to",
    "hardlink_to",
    "link_to",
    "symlink",
    "link",
    "chmod",
    "chown",
    "utime",
    "truncate",
    "copy",
    "copy2",
    "copyfile",
    "copytree",
    "move",
    "atomic_path",
    "write_npz",
    "mkstemp",
    "mkdtemp",
    "NamedTemporaryFile",
    "TemporaryDirectory",
}


def test_the_module_has_no_call_that_writes_creates_or_removes() -> None:
    source = Path(str(local_module.__file__)).read_text()
    tree = ast.parse(source)
    opens = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", getattr(node.func, "id", None))
            assert name not in _WRITERS, f"line {node.lineno} calls {name}()"
            if name == "replace":  # dataclasses.replace and str.replace, never Path.replace
                owner = getattr(node.func, "value", None)
                assert isinstance(owner, ast.Name) and owner.id == "dataclasses"
            if name == "open":
                opens.append([ast.literal_eval(arg) for arg in node.args])
        elif isinstance(node, ast.Import | ast.ImportFrom):
            modules = [alias.name for alias in node.names]
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            assert not {"shutil", "tempfile", "surfcensus._io"} & set(modules)
    assert opens == [["rb"]]


# ---------------------------------------------------------------------------- pipeline version


def _description_dir(tree: Path, layout: Layout) -> Path:
    return tree if layout == "fmriprep" else tree / "fmriprep"


@pytest.mark.parametrize("layout", ["fmriprep", "fmriprep_legacy"])
def test_the_pipeline_version_is_read_from_the_dataset_description(
    tmp_path: Path, layout: Layout
) -> None:
    # write_tree gives the new layout the GeneratedBy form and the old one PipelineDescription
    tree = write_tree(tmp_path / "a", layout, ["sub-01"], fmriprep_version="21.0.2")
    assert LocalSource(tree).dataset.pipeline_version == "21.0.2"

    path = _description_dir(tree, layout) / "dataset_description.json"
    path.write_text(
        '{"GeneratedBy": [{"Name": "other", "Version": "1.0"}, '
        '{"Name": "fmriprep", "Version": "20.2.7"}]}'
    )
    assert LocalSource(tree).dataset.pipeline_version == "20.2.7"
    path.write_text('{"PipelineDescription": {"Name": "fMRIPrep", "Version": "1.5.8"}}')
    assert LocalSource(tree).dataset.pipeline_version == "1.5.8"


@pytest.mark.parametrize("layout", ["fmriprep", "fmriprep_legacy"])
@pytest.mark.parametrize(
    "content",
    [None, b"", b"{ not json", b'["a list"]', b'{"Name": "x"}', b"\xff\xfe\x00{", "directory"],
)
def test_a_missing_or_broken_dataset_description_gives_no_version(
    tmp_path: Path, layout: Layout, content: bytes | str | None
) -> None:
    tree = write_tree(tmp_path / "study", layout, ["sub-01"], fmriprep_version=None)
    path = _description_dir(tree, layout) / "dataset_description.json"
    assert not path.exists()
    if content == "directory":
        path.mkdir()
    elif isinstance(content, bytes):
        path.write_bytes(content)
    record = LocalSource(tree).dataset
    assert (record.layout, record.pipeline, record.pipeline_version) == (layout, "fmriprep", None)
    assert _labels(LocalSource(tree)) == ["sub-01"]


def test_the_description_is_read_only_where_the_layout_has_it(tmp_path: Path) -> None:
    text = '{"GeneratedBy": [{"Name": "fMRIPrep", "Version": "23.2.0"}]}'
    # the old layout reads fmriprep/dataset_description.json, not the one beside it
    tree = write_tree(tmp_path / "legacy", "fmriprep_legacy", ["sub-01"], fmriprep_version=None)
    (tree / "dataset_description.json").write_text(text)
    assert LocalSource(tree).dataset.pipeline_version is None
    # the other layouts are not fMRIPrep's and read none
    for layout in ("bids_derivatives", "subjects_dir"):
        tree = write_tree(tmp_path / layout, layout, ["sub-01"])
        (tree / "dataset_description.json").write_text(text)
        if layout == "subjects_dir":  # a folder of that name does not make it fMRIPrep's
            (tree / "fmriprep").mkdir()
            (tree / "fmriprep" / "dataset_description.json").write_text(text)
        record = LocalSource(tree).dataset
        assert (record.layout, record.pipeline, record.pipeline_version) == (
            layout,
            "freesurfer",
            None,
        )


# ---------------------------------------------------------------------------- template


@pytest.mark.parametrize("layout", LAYOUTS)
def test_the_template_next_to_the_subjects_is_found(tmp_path: Path, layout: Layout) -> None:
    tree = write_tree(tmp_path / "study", layout, ["sub-01"], with_template=True)
    source = LocalSource(tree)
    expected = tree.resolve() / FS_ROOT[layout] / "fsaverage"
    assert source.template_dir(source.dataset) == expected
    assert find_adjacent_template(source.subjects_dir) == expected
    assert find_adjacent_template(str(source.subjects_dir)) == expected
    assert Path(source.dataset.root, source.dataset.template_location or "x") == expected
    assert source.dataset.template_location == (
        "fsaverage" if layout == "subjects_dir" else f"{FS_ROOT[layout]}/fsaverage"
    )
    assert template_files_missing(expected) == ()
    assert _labels(source) == ["sub-01"]  # fsaverage and fsaverage5 are not subjects


def test_no_template_next_to_the_subjects(tmp_path: Path) -> None:
    tree = write_tree(tmp_path / "study", "fmriprep", ["sub-01"])
    write_light_template(tree / "fsaverage")  # next to fMRIPrep's output, not to the subjects
    write_light_template(tree / "sourcedata" / "freesurfer" / "fsaverage6")  # another name
    source = LocalSource(tree)
    assert source.dataset.template_location is None
    assert source.template_dir(source.dataset) is None
    assert find_adjacent_template(source.subjects_dir) is None
    assert template_files_missing(source.subjects_dir / "fsaverage") is None
    (source.subjects_dir / "fsaverage").write_text("a file is not a template\n")
    assert LocalSource(tree).dataset.template_location is None
    assert find_adjacent_template(source.subjects_dir) is None


def test_a_dangling_template_link_is_no_template(tmp_path: Path) -> None:
    tree = write_tree(tmp_path / "study", "subjects_dir", ["sub-01"], dangling_template=True)
    assert (tree / "fsaverage").is_symlink()
    source = LocalSource(tree)
    assert source.dataset.template_location is None
    assert source.template_dir(source.dataset) is None
    assert find_adjacent_template(tree) is None
    assert template_files_missing(tree / "fsaverage") is None
    assert _labels(source) == ["sub-01"]


def test_a_template_link_to_a_freesurfer_installation_is_found(tmp_path: Path) -> None:
    tree = write_tree(tmp_path / "study", "subjects_dir", ["sub-01"])
    write_light_template(tmp_path / "freesurfer" / "subjects" / "fsaverage")
    (tree / "fsaverage").symlink_to(tmp_path / "freesurfer" / "subjects" / "fsaverage")
    source = LocalSource(tree)
    assert source.dataset.template_location == "fsaverage"
    assert source.template_dir(source.dataset) == tree.resolve() / "fsaverage"
    assert _labels(source) == ["sub-01"]


def test_an_incomplete_template_is_not_used_and_its_missing_files_are_named(
    tmp_path: Path,
) -> None:
    tree = write_tree(tmp_path / "study", "subjects_dir", ["sub-01"])
    write_light_template(tree / "fsaverage", complete=False)
    source = LocalSource(tree)
    assert source.template_dir(source.dataset) is None
    assert find_adjacent_template(tree) is None
    assert template_files_missing(tree / "fsaverage") == (
        "label/lh.aparc.annot",
        "label/rh.aparc.annot",
    )
    # the directory is recorded all the same, so the template lookup can say what it lacks
    assert source.dataset.template_location == "fsaverage"

    (tree / "fsaverage" / "surf" / "rh.sphere.reg").unlink()
    assert template_files_missing(tree / "fsaverage") == (
        "surf/rh.sphere.reg",
        "label/lh.aparc.annot",
        "label/rh.aparc.annot",
    )
    # one hemisphere is not enough, and the template is looked up on each call
    shutil.rmtree(tree / "fsaverage")
    write_light_template(tree / "fsaverage")
    assert source.template_dir(source.dataset) == tree.resolve() / "fsaverage"
    (tree / "fsaverage" / "label" / "rh.cortex.label").unlink()
    assert source.template_dir(source.dataset) is None
    assert template_files_missing(tree / "fsaverage") == ("label/rh.cortex.label",)


def test_a_template_that_cannot_be_examined_is_not_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree = write_tree(tmp_path / "study", "subjects_dir", ["sub-01"], with_template=True)
    source = LocalSource(tree)
    assert source.template_dir(source.dataset) is not None
    _refuse_stat(monkeypatch, tree / "fsaverage" / "surf" / "lh.white")
    assert source.template_dir(source.dataset) is None
    with pytest.raises(PermissionError):
        template_files_missing(tree / "fsaverage")


# ---------------------------------------------------------------------------- config, identity


@pytest.mark.parametrize("layout", LAYOUTS)
def test_the_config_rebuilds_an_equal_source(tmp_path: Path, layout: Layout) -> None:
    tree = write_tree(tmp_path / "study", layout, LABELS, with_template=True)
    source = LocalSource(tree, subjects=["sub-0[13]", "sub-03"], field_strength=7.0)
    config = source.config()
    assert config == SourceConfig(
        kind="local",
        root=str(tree.resolve()),
        identity=str(tree.resolve()),
        subjects=("sub-0[13]", "sub-03"),
        field_strength=7.0,
    )
    run_config = new_local_config(tree, sources=(config,))
    assert run_config.sources == (config,)
    rebuilt = from_config(config, run_config, tmp_path / "run")
    assert isinstance(rebuilt, LocalSource)
    assert rebuilt.dataset == source.dataset
    assert rebuilt.config() == config
    assert list(rebuilt.subjects(rebuilt.dataset)) == list(source.subjects(source.dataset))
    assert _labels(rebuilt) == ["sub-01", "sub-03"]


def test_the_config_of_a_source_without_options(tmp_path: Path) -> None:
    tree = write_tree(tmp_path / "study", "subjects_dir", LABELS)
    config = local(tree).config()
    assert (config.subjects, config.field_strength) == ((), None)
    assert (config.dataset, config.bucket, config.prefix) == (None, None, None)
    assert local(tree, field_strength=3).config().field_strength == 3.0


def test_the_config_of_one_subject_rebuilds_an_equal_source(tmp_path: Path) -> None:
    tree = write_tree(tmp_path / "study", "subjects_dir", ["sub-01", "sub-[2]", "sub-2"])
    source = LocalSource(tree / "sub-[2]", field_strength=1.5)
    config = source.config()
    assert (config.root, config.subjects) == (str(tree.resolve()), ("sub-[[]2]",))
    rebuilt = from_config(config, new_local_config(tree, sources=(config,)), tmp_path / "run")
    assert isinstance(rebuilt, LocalSource)
    assert rebuilt.dataset == source.dataset
    assert rebuilt.config() == config
    assert _labels(rebuilt) == ["sub-[2]"] == _labels(source)


@pytest.mark.parametrize("value", [0, -1.5, float("nan")])
def test_the_field_strength_is_positive(tmp_path: Path, value: float) -> None:
    tree = write_tree(tmp_path / "study", "subjects_dir", ["sub-01"])
    with pytest.raises(ValueError, match="field_strength"):
        LocalSource(tree, field_strength=value)


def test_a_given_identity_survives_a_new_root(tmp_path: Path) -> None:
    tree = write_tree(tmp_path / "old" / "study", "fmriprep", LABELS)
    source = LocalSource(tree, subjects=["sub-0[12]"])
    stored = source.config()
    assert source.dataset.identity == str(tree.resolve())

    moved = tmp_path / "new" / "place"
    moved.parent.mkdir()
    tree.rename(moved)
    relocated = stored.model_copy(update={"root": str(moved)})
    rebuilt = from_config(
        relocated, new_local_config(moved, sources=(relocated,)), tmp_path / "run"
    )
    assert isinstance(rebuilt, LocalSource)
    assert rebuilt.dataset.root == str(moved.resolve())
    assert rebuilt.dataset.identity == source.dataset.identity
    assert rebuilt.dataset == dataclasses.replace(source.dataset, root=str(moved.resolve()))
    assert rebuilt.config() == relocated
    assert _labels(rebuilt) == ["sub-01", "sub-02"]

    key = new_local_salt()

    def uids(src: LocalSource) -> list[str]:
        return [
            subject_uid(
                key, s.dataset.source, s.dataset.identity, s.dataset.pipeline, s.subject_label
            )
            for s in src.subjects(src.dataset)
        ]

    fresh = LocalSource(moved, subjects=["sub-0[12]"])
    assert fresh.dataset.identity == str(moved.resolve())
    kept = uids(rebuilt)
    assert len(set(kept)) == 2 and kept != uids(fresh)
    assert LocalSource(moved, identity=stored.identity).dataset.identity == stored.identity
    # the uid is a function of the identity, not of where the files are
    moved.rename(tree)
    assert uids(LocalSource(tree, subjects=["sub-0[12]"])) == kept


def test_sources_local_is_the_function_of_the_public_api(tmp_path: Path) -> None:
    tree = write_tree(tmp_path / "study", "subjects_dir", LABELS)
    source = surfcensus.sources.local(tree, subjects=["sub-02"], field_strength=3.0)
    assert isinstance(source, surfcensus.sources.LocalSource)
    assert source.config() == LocalSource(tree, subjects=["sub-02"], field_strength=3.0).config()
    assert _labels(source) == ["sub-02"]


# ---------------------------------------------------------------------------- the real subject


def _check_real_subject(root: Path) -> None:
    """ds000102 sub-01 (FreeSurfer 6.0.1): the sizes of the bucket listing, plain pial files."""
    before = snapshot_tree(root)
    source = LocalSource(root)
    assert (source.dataset.layout, source.dataset.root) == (
        "subjects_dir",
        str(root.parent.resolve()),
    )
    (subject,) = source.subjects(source.dataset)
    assert subject.subject_label == root.name and subject.problem is None
    assert {ref.location: ref.size for ref in subject.files.values()} == REAL_SUBJECT_FILES
    assert set(subject.files) == {*TIER_A, *TIER_B, *TIER_C}
    assert subject.files[FileRole.LH_PIAL].variant == "lh.pial"
    assert subject.files[FileRole.RH_PIAL].variant == "rh.pial"
    with source.lease(subject, TIER_C) as paths:
        assert {role: path.stat().st_size for role, path in paths.items()} == {
            role: ref.size for role, ref in subject.files.items() if role in TIER_C
        }
    # an MGH file is gzip, a surface starts with the triangle magic number ff ff fe
    assert source.read_bytes(subject, FileRole.T1, byte_range=(0, 1)) == b"\x1f\x8b"
    assert source.read_bytes(subject, FileRole.LH_PIAL, byte_range=(0, 2)) == b"\xff\xff\xfe"
    assert snapshot_tree(root) == before


def test_the_cached_real_subject_is_listed_as_the_bucket_lists_it(
    request: pytest.FixtureRequest,
) -> None:
    root = request.config.rootpath / ".pytest_cache" / "d" / "ds000102-sub-01"
    present = all(
        (root / name).is_file() and (root / name).stat().st_size == size
        for name, size in REAL_SUBJECT_FILES.items()
    )
    if not present:
        pytest.skip("ds000102 sub-01 is not in the pytest cache (the network tests fetch it)")
    _check_real_subject(root)


@pytest.mark.network
def test_the_real_subject_is_listed_as_the_bucket_lists_it(real_subject: Path) -> None:
    _check_real_subject(real_subject)
