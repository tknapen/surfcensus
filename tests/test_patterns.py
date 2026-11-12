"""sources/patterns and sources/base: file roles inside a FreeSurfer subject directory, what
a source lists as a subject, and the records and tiers every source shares."""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Collection, Iterator, Mapping
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path, PurePosixPath

import numpy as np
import pytest
from synth import (  # pyrefly: ignore[missing-import]
    SyntheticSubject,
    file_modes_apply,
    snapshot_tree,
    write_synthetic_subject,
)

from surfcensus.config import SourceConfig
from surfcensus.fsio.surfaces import read_surface
from surfcensus.sources.base import (
    HEMIS,
    RENDER_ROLES,
    TIER_A,
    TIER_A_REQUIRED,
    TIER_B,
    TIER_B_REQUIRED,
    TIER_C,
    DatasetRecord,
    FileRef,
    FileRole,
    Source,
    SourceError,
    SubjectRecord,
    input_signature,
)
from surfcensus.sources.patterns import (
    ROLE_CANDIDATES,
    TEMPLATE_FILES,
    TEMPLATE_NAME_PATTERNS,
    TEMPLATE_OPTIONAL_FILES,
    is_file,
    is_freesurfer_subject_dir,
    is_subject_candidate,
    is_template_name,
    missing_roles,
    resolve_subject_files,
    variant_name,
)

PIAL_SUFFIXES = ("", ".T2", ".FLAIR", ".T1")  # order of preference


def _fake_subject(root: Path, name: str, files: tuple[str, ...]) -> Path:
    """A subject directory of empty files: only names matter to the patterns."""
    subject = root / name
    for relative in files:
        (subject / relative).parent.mkdir(parents=True, exist_ok=True)
        (subject / relative).touch()
    return subject


def _small_subject(root: Path, **kwargs: str) -> SyntheticSubject:
    return write_synthetic_subject(root, shape=(8, 8, 8), subdivisions=1, **kwargs)


def test_candidates_cover_every_role_with_relative_paths() -> None:
    assert list(ROLE_CANDIDATES) == list(FileRole)
    for candidates in ROLE_CANDIDATES.values():
        assert candidates and len(set(candidates)) == len(candidates)
        for candidate in candidates:
            path = PurePosixPath(candidate)
            assert not path.is_absolute() and ".." not in path.parts and "\\" not in candidate
    everything = [c for candidates in ROLE_CANDIDATES.values() for c in candidates]
    assert len(set(everything)) == len(everything)


def test_candidates_follow_the_data_sources_table() -> None:
    # docs/data_sources.md, "Files required per subject"
    assert {role.value: candidates[0] for role, candidates in ROLE_CANDIDATES.items()} == {
        "t1": "mri/T1.mgz",
        "brainmask": "mri/brainmask.mgz",
        "lh_white": "surf/lh.white",
        "rh_white": "surf/rh.white",
        "lh_pial": "surf/lh.pial",
        "rh_pial": "surf/rh.pial",
        "lh_thickness": "surf/lh.thickness",
        "rh_thickness": "surf/rh.thickness",
        "lh_sphere_reg": "surf/lh.sphere.reg",
        "rh_sphere_reg": "surf/rh.sphere.reg",
        "lh_annot": "label/lh.aparc.annot",
        "rh_annot": "label/rh.aparc.annot",
        "lh_stats": "stats/lh.aparc.stats",
        "rh_stats": "stats/rh.aparc.stats",
        "aseg_stats": "stats/aseg.stats",
        "talairach": "mri/transforms/talairach.xfm",
        "recon_log": "scripts/recon-all.log",
        "build_stamp": "scripts/build-stamp.txt",
        "recon_log_lh": "scripts/recon-all-lh.log",
        "recon_log_rh": "scripts/recon-all-rh.log",
    }
    for hemi, role in (("lh", FileRole.LH_PIAL), ("rh", FileRole.RH_PIAL)):
        assert ROLE_CANDIDATES[role] == tuple(f"surf/{hemi}.pial{s}" for s in PIAL_SUFFIXES)
    multiple = {role for role, candidates in ROLE_CANDIDATES.items() if len(candidates) > 1}
    assert multiple == {FileRole.LH_PIAL, FileRole.RH_PIAL}


@pytest.mark.parametrize(
    ("fixture", "pial"),
    [
        ("synth_subject", "pial"),
        ("synth_subject_hires", "pial"),
        ("synth_subject_pial_t1", "pial.T1"),
    ],
)
def test_resolve_on_the_synthetic_subjects(
    fixture: str, pial: str, request: pytest.FixtureRequest
) -> None:
    s: SyntheticSubject = request.getfixturevalue(fixture)
    files = resolve_subject_files(s.path)
    assert files == {
        FileRole.T1: s.path / "mri" / "T1.mgz",
        FileRole.BRAINMASK: s.path / "mri" / "brainmask.mgz",
        FileRole.LH_WHITE: s.path / "surf" / "lh.white",
        FileRole.RH_WHITE: s.path / "surf" / "rh.white",
        FileRole.LH_PIAL: s.path / "surf" / f"lh.{pial}",
        FileRole.RH_PIAL: s.path / "surf" / f"rh.{pial}",
    }
    assert list(files) == [r for r in FileRole if r in files]  # in role order
    assert all(path.is_file() for path in files.values())
    assert resolve_subject_files(str(s.path)) == files
    assert missing_roles(files) == []
    assert missing_roles(files, required=tuple(FileRole)) == [r for r in FileRole if r not in files]
    assert is_freesurfer_subject_dir(s.path) and is_freesurfer_subject_dir(str(s.path))


def test_pial_t1_variant_has_no_plain_pial(synth_subject_pial_t1: SyntheticSubject) -> None:
    surf = sorted(p.name for p in (synth_subject_pial_t1.path / "surf").iterdir())
    assert surf == ["lh.pial.T1", "lh.white", "rh.pial.T1", "rh.white"]


@pytest.mark.parametrize(
    ("present", "chosen"),
    [
        (("", ".T2", ".FLAIR", ".T1"), ""),
        ((".T1", ".FLAIR", ".T2"), ".T2"),
        ((".T1", ".FLAIR"), ".FLAIR"),
        ((".T1",), ".T1"),
        (("", ".T1"), ""),
    ],
)
def test_pial_fallback_order(tmp_path: Path, present: tuple[str, ...], chosen: str) -> None:
    names = tuple(f"surf/{hemi}.pial{suffix}" for hemi in ("lh", "rh") for suffix in present)
    subject = _fake_subject(tmp_path, "sub-01", ("mri/T1.mgz", *names))
    files = resolve_subject_files(subject)
    assert files[FileRole.LH_PIAL] == subject / "surf" / f"lh.pial{chosen}"
    assert files[FileRole.RH_PIAL] == subject / "surf" / f"rh.pial{chosen}"


def test_hemispheres_resolve_independently(tmp_path: Path) -> None:
    subject = _fake_subject(
        tmp_path, "sub-01", ("mri/T1.mgz", "surf/lh.pial", "surf/rh.pial.T2", "surf/rh.pial.T1")
    )
    files = resolve_subject_files(subject)
    assert (
        files[FileRole.LH_PIAL].name == "lh.pial" and files[FileRole.RH_PIAL].name == "rh.pial.T2"
    )


def test_resolve_finds_every_role_and_reports_missing_ones(tmp_path: Path) -> None:
    full = _fake_subject(tmp_path, "full", tuple(c[0] for c in ROLE_CANDIDATES.values()))
    files = resolve_subject_files(full)
    assert list(files) == list(FileRole)
    assert missing_roles(files, required=tuple(FileRole)) == []

    partial = _fake_subject(tmp_path, "partial", ("mri/T1.mgz", "surf/rh.white", "surf/lh.pial"))
    files = resolve_subject_files(partial)
    assert set(files) == {FileRole.T1, FileRole.RH_WHITE, FileRole.LH_PIAL}
    assert missing_roles(files) == [FileRole.LH_WHITE, FileRole.RH_PIAL]  # order of RENDER_ROLES
    assert missing_roles({}) == list(RENDER_ROLES)
    assert missing_roles(files, required=[FileRole.BRAINMASK, FileRole.T1]) == [FileRole.BRAINMASK]

    assert resolve_subject_files(tmp_path / "does-not-exist") == {}


def test_brainmask_is_optional(tmp_path: Path) -> None:
    s = write_synthetic_subject(tmp_path, shape=(8, 8, 8), subdivisions=1, with_brainmask=False)
    files = resolve_subject_files(s.path)
    assert FileRole.BRAINMASK not in files and missing_roles(files) == []


def test_a_directory_is_not_a_file(tmp_path: Path) -> None:
    subject = _fake_subject(tmp_path, "sub-01", ("surf/lh.pial.T1", "surf/rh.pial"))
    (subject / "mri" / "T1.mgz").mkdir(parents=True)
    (subject / "surf" / "lh.pial").mkdir()
    files = resolve_subject_files(subject)
    assert FileRole.T1 not in files
    assert files[FileRole.LH_PIAL].name == "lh.pial.T1"
    assert not is_freesurfer_subject_dir(subject)


def test_dangling_symlink_does_not_count(tmp_path: Path) -> None:
    s = _small_subject(tmp_path, pial_variant="pial.T1")
    surf = s.path / "surf"
    (surf / "lh.pial").symlink_to("lh.pial.T2")  # FreeSurfer 7 link whose target was not copied
    (surf / "rh.pial").symlink_to("rh.pial.T2")
    files = resolve_subject_files(s.path)
    assert files[FileRole.LH_PIAL] == surf / "lh.pial.T1"  # falls through to the next candidate
    assert is_freesurfer_subject_dir(s.path)

    (surf / "lh.pial.T1").unlink()
    files = resolve_subject_files(s.path)
    assert FileRole.LH_PIAL not in files and FileRole.RH_PIAL in files
    assert missing_roles(files) == [FileRole.LH_PIAL]
    assert not is_freesurfer_subject_dir(s.path)
    assert (surf / "lh.pial").is_symlink()  # the link itself was left alone


def test_freesurfer7_symlinked_pial(tmp_path: Path) -> None:
    s = _small_subject(tmp_path, pial_variant="pial.T1")
    surf = s.path / "surf"
    for hemi in ("lh", "rh"):
        (surf / f"{hemi}.pial").symlink_to(f"{hemi}.pial.T1")  # relative, as recon-all makes it
    before = snapshot_tree(s.path)
    files = resolve_subject_files(s.path)
    assert files[FileRole.LH_PIAL] == surf / "lh.pial"  # the preferred name, through the link
    assert files[FileRole.LH_PIAL].resolve().name == "lh.pial.T1"  # the variant it names
    assert files[FileRole.RH_PIAL].resolve().name == "rh.pial.T1"
    assert missing_roles(files) == [] and is_freesurfer_subject_dir(s.path)
    linked, target = read_surface(files[FileRole.LH_PIAL]), read_surface(surf / "lh.pial.T1")
    assert np.array_equal(linked.vertices, target.vertices)
    assert np.array_equal(linked.faces, target.faces)
    assert snapshot_tree(s.path) == before


def test_is_freesurfer_subject_dir_rejects_templates_and_incomplete_dirs(tmp_path: Path) -> None:
    complete = ("mri/T1.mgz", "surf/lh.pial", "surf/rh.pial", "surf/lh.white", "surf/rh.white")
    assert is_freesurfer_subject_dir(_fake_subject(tmp_path, "sub-01", complete))
    # a left pial variant is enough to recognize a subject
    assert is_freesurfer_subject_dir(
        _fake_subject(tmp_path, "sub-02", ("mri/T1.mgz", "surf/lh.pial.T1"))
    )
    assert is_freesurfer_subject_dir(_fake_subject(tmp_path, "my-fsaverage-copy", complete))

    for template in ("fsaverage", "fsaverage5", "fsaverage6", "fsaverage_sym"):
        assert not is_freesurfer_subject_dir(_fake_subject(tmp_path, template, complete))
    assert not is_freesurfer_subject_dir(_fake_subject(tmp_path, "no-t1", ("surf/lh.pial",)))
    assert not is_freesurfer_subject_dir(_fake_subject(tmp_path, "no-pial", ("mri/T1.mgz",)))
    assert not is_freesurfer_subject_dir(
        _fake_subject(tmp_path, "rh-only", ("mri/T1.mgz", "surf/rh.pial", "surf/lh.white"))
    )
    assert not is_freesurfer_subject_dir(
        _fake_subject(tmp_path, "mgh", ("mri/T1.mgh", "surf/lh.pial"))
    )
    assert not is_freesurfer_subject_dir(tmp_path / "does-not-exist")
    assert not is_freesurfer_subject_dir(tmp_path / "sub-01" / "mri" / "T1.mgz")  # a file
    assert not is_freesurfer_subject_dir(tmp_path)  # the SUBJECTS_DIR itself


def test_is_freesurfer_subject_dir_through_directory_symlinks(tmp_path: Path) -> None:
    complete = ("mri/T1.mgz", "surf/lh.pial")
    home = tmp_path / "freesurfer_home"
    subjects = tmp_path / "subjects"
    subjects.mkdir()
    # SUBJECTS_DIR/fsaverage is usually a link into the FreeSurfer installation
    (subjects / "fsaverage").symlink_to(_fake_subject(home, "fsaverage", complete))
    assert not is_freesurfer_subject_dir(subjects / "fsaverage")
    # renaming the link does not turn the template into a subject
    (subjects / "template").symlink_to(home / "fsaverage")
    assert not is_freesurfer_subject_dir(subjects / "template")
    # a linked subject is a subject
    (subjects / "sub-01").symlink_to(_fake_subject(tmp_path / "elsewhere", "sub-01", complete))
    assert is_freesurfer_subject_dir(subjects / "sub-01")
    (subjects / "sub-gone").symlink_to(tmp_path / "nowhere")
    assert not is_freesurfer_subject_dir(subjects / "sub-gone")


def test_is_file_tells_absent_from_present(tmp_path: Path) -> None:
    (tmp_path / "file").write_text("x")
    (tmp_path / "link").symlink_to("file")
    (tmp_path / "dangling").symlink_to("nothing")
    (tmp_path / "loop").symlink_to("loop")
    (tmp_path / "folder").mkdir()
    assert is_file(tmp_path / "file") and is_file(str(tmp_path / "link"))
    for absent in ("nothing", "dangling", "loop", "folder", "file/below", "a" * 300, "nul\0"):
        assert not is_file(tmp_path / absent), absent


def test_names_too_long_to_exist_are_absent(tmp_path: Path) -> None:
    """Such a name raises OSError (ENAMETOOLONG) from ``Path.is_dir()`` up to Python 3.13."""
    subject = _fake_subject(tmp_path, "sub-01", ("mri/T1.mgz", "surf/lh.pial"))
    assert not is_freesurfer_subject_dir(tmp_path / ("a" * 300))
    assert not is_freesurfer_subject_dir(subject / ("a" * 300) / "sub-02")
    assert resolve_subject_files(tmp_path / ("a" * 300)) == {}


@pytest.mark.parametrize("unreadable", ["mri/T1.mgz", "surf/lh.pial"])
def test_a_path_that_cannot_be_examined_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unreadable: str
) -> None:
    """Whether the file exists is unknown, so the answer is neither "absent" nor "not a
    subject". ``Path.is_file()`` cannot be relied on for this: it raises up to Python 3.13 and
    answers False from 3.14. The error is that of ``stat()`` in a directory without search
    permission, raised here by a replaced ``Path.stat`` so that the test also runs as root."""
    subject = _fake_subject(
        tmp_path, "sub-01", ("mri/T1.mgz", "surf/lh.pial", "surf/lh.pial.T1", "surf/rh.pial")
    )
    stat = Path.stat

    def refuse(self: Path, *args: object, **kwargs: object) -> object:
        if self == subject / unreadable:
            raise PermissionError(13, "Permission denied", str(self))
        return stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", refuse)
    with pytest.raises(PermissionError):
        is_file(subject / unreadable)
    with pytest.raises(PermissionError):
        is_freesurfer_subject_dir(subject)
    with pytest.raises(PermissionError):
        resolve_subject_files(subject)
    assert is_file(subject / "surf" / "rh.pial")


@pytest.mark.parametrize(("target", "mode"), [("surf", 0), ("mri", 0o444), (".", 0)])
def test_a_directory_without_permission_raises(tmp_path: Path, target: str, mode: int) -> None:
    if not file_modes_apply(tmp_path):
        pytest.skip("file modes do not restrict this user")
    subject = _fake_subject(tmp_path, "sub-01", ("mri/T1.mgz", "surf/lh.pial", "surf/rh.pial"))
    (subject / target).chmod(mode)
    try:
        with pytest.raises(PermissionError):
            is_freesurfer_subject_dir(subject)
        with pytest.raises(PermissionError):
            resolve_subject_files(subject)
    finally:
        (subject / target).chmod(0o755)


def test_patterns_leave_the_tree_untouched(
    synth_subject: SyntheticSubject, synth_subject_pial_t1: SyntheticSubject
) -> None:
    for s in (synth_subject, synth_subject_pial_t1):
        before = snapshot_tree(s.path)
        listing = sorted(p.name for p in s.path.parent.iterdir())
        files = resolve_subject_files(s.path)
        missing_roles(files, required=tuple(FileRole))
        is_freesurfer_subject_dir(s.path)
        assert snapshot_tree(s.path) == before
        assert sorted(p.name for p in s.path.parent.iterdir()) == listing


# --- roles, tiers and records of sources/base.py ---------------------------------------------


def test_the_first_eighteen_roles_keep_their_order_and_the_two_logs_follow() -> None:
    # role order is the order of every files mapping and of the input signature
    assert [role.value for role in FileRole] == [
        "t1",
        "brainmask",
        "lh_white",
        "rh_white",
        "lh_pial",
        "rh_pial",
        "lh_thickness",
        "rh_thickness",
        "lh_sphere_reg",
        "rh_sphere_reg",
        "lh_annot",
        "rh_annot",
        "lh_stats",
        "rh_stats",
        "aseg_stats",
        "talairach",
        "recon_log",
        "build_stamp",
        "recon_log_lh",
        "recon_log_rh",
    ]
    assert HEMIS == ("lh", "rh")


def test_the_tiers_are_those_of_the_data_sources_document() -> None:
    assert [role.value for role in TIER_A] == [
        "lh_stats",
        "rh_stats",
        "aseg_stats",
        "build_stamp",
        "recon_log",
        "recon_log_lh",
        "recon_log_rh",
    ]
    assert [role.value for role in TIER_B] == [
        "lh_thickness",
        "rh_thickness",
        "lh_sphere_reg",
        "rh_sphere_reg",
        "lh_annot",
        "rh_annot",
    ]
    assert TIER_C == (*RENDER_ROLES, FileRole.BRAINMASK)
    assert TIER_A_REQUIRED == (FileRole.LH_STATS, FileRole.RH_STATS)
    assert [role.value for role in TIER_B_REQUIRED] == [
        "lh_thickness",
        "rh_thickness",
        "lh_sphere_reg",
        "rh_sphere_reg",
    ]
    assert set(TIER_A_REQUIRED) < set(TIER_A) and set(TIER_B_REQUIRED) < set(TIER_B)
    tiers = [*TIER_A, *TIER_B, *TIER_C]
    assert len(set(tiers)) == len(tiers), "no role is in two tiers"
    assert set(FileRole) - set(tiers) == {FileRole.TALAIRACH}


def test_the_text_files_of_a_subject_resolve_to_their_roles(tmp_path: Path) -> None:
    names = [ROLE_CANDIDATES[role][0] for role in (*TIER_A, *TIER_B)]
    subject = _fake_subject(tmp_path, "sub-01", tuple(names))
    files = resolve_subject_files(subject)
    assert set(files) == {*TIER_A, *TIER_B}
    assert files[FileRole.RECON_LOG_LH] == subject / "scripts" / "recon-all-lh.log"
    assert files[FileRole.RECON_LOG_RH] == subject / "scripts" / "recon-all-rh.log"
    assert missing_roles(files, TIER_A_REQUIRED) == []
    assert missing_roles(files, TIER_B_REQUIRED) == []
    assert missing_roles(files) == list(RENDER_ROLES)


def _records() -> tuple[DatasetRecord, SubjectRecord]:
    dataset = DatasetRecord(
        source="local",
        identity="/data/study",
        accession=None,
        root="/data/study",
        fs_root="sourcedata/freesurfer",
        layout="fmriprep",
        pipeline="fmriprep",
        pipeline_version="23.2.0",
    )
    files = {
        FileRole.T1: FileRef(FileRole.T1, "mri/T1.mgz", 5_576_738, "1731400000000000000"),
        FileRole.LH_PIAL: FileRef(
            FileRole.LH_PIAL, "surf/lh.pial.T1", 4_731_523, "1731400000000000001", "lh.pial.T1"
        ),
        FileRole.BUILD_STAMP: FileRef(FileRole.BUILD_STAMP, "scripts/build-stamp.txt", 58, "17"),
    }
    subject = SubjectRecord(dataset, "sub-01", "sourcedata/freesurfer/sub-01", files)
    return dataset, subject


def test_records_are_frozen_and_compare_by_value() -> None:
    dataset, subject = _records()
    again_dataset, again = _records()
    assert dataset == again_dataset and subject == again
    assert dataset.template_location is None and dataset.derivative_ref is None
    assert subject.problem is None and subject.files[FileRole.LH_PIAL].variant == "lh.pial.T1"
    assert subject.files[FileRole.T1].last_modified is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        dataset.root = "/elsewhere"  # pyrefly: ignore[read-only]
    with pytest.raises(dataclasses.FrozenInstanceError):
        subject.files = {}  # pyrefly: ignore[read-only]
    moved = dataclasses.replace(dataset, root="/elsewhere")
    assert moved != dataset and moved.identity == dataset.identity
    unreadable = SubjectRecord(dataset, "sub-02", "x/sub-02", {}, "unreadable: Permission denied")
    assert unreadable.files == {} and unreadable.problem == "unreadable: Permission denied"


def test_source_error_has_a_reason_and_a_detail() -> None:
    error = SourceError("changed", "mri/T1.mgz is 12 bytes, the inventory saw 5576738")
    assert error.reason == "changed"
    assert error.detail == "mri/T1.mgz is 12 bytes, the inventory saw 5576738"
    assert str(error) == "changed: mri/T1.mgz is 12 bytes, the inventory saw 5576738"
    assert isinstance(error, Exception)


def test_the_input_signature_is_the_documented_hash() -> None:
    _, subject = _records()
    lines = [
        "t1\tmri/T1.mgz\t5576738\t1731400000000000000",
        # a role with several candidate names also says which one was read
        "lh_pial\tsurf/lh.pial.T1\t4731523\t1731400000000000001\tlh.pial.T1",
        "build_stamp\tscripts/build-stamp.txt\t58\t17",
    ]
    expected = hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16]
    assert input_signature(subject.files) == expected
    # in role order, whatever the order of the mapping or of the request
    assert input_signature(dict(reversed(list(subject.files.items())))) == expected
    assert input_signature(subject.files, list(reversed(FileRole))) == expected
    # the requested roles that the subject has; absent ones add nothing
    two = hashlib.sha256("\n".join(lines[:2]).encode()).hexdigest()[:16]
    assert input_signature(subject.files, TIER_C) == two
    assert input_signature(subject.files, (FileRole.T1, FileRole.LH_PIAL)) == two
    # then one line per extra string
    with_extra = hashlib.sha256("\n".join([*lines[:2], "abc", "def"]).encode()).hexdigest()[:16]
    assert input_signature(subject.files, TIER_C, ("abc", "def")) == with_extra
    assert input_signature(subject.files, TIER_C, ("def", "abc")) != with_extra
    assert input_signature({}) == hashlib.sha256(b"").hexdigest()[:16]


@pytest.mark.parametrize(
    "change",
    [
        {"size": 5_576_739},
        {"etag_or_mtime": "1731400000000000002"},
        {"location": "mri/T1.mgz.gz"},
    ],
)
def test_the_input_signature_changes_with_size_mtime_and_location(change: dict) -> None:
    _, subject = _records()
    files = dict(subject.files)
    files[FileRole.T1] = dataclasses.replace(files[FileRole.T1], **change)
    assert input_signature(files) != input_signature(subject.files)
    assert input_signature(files, TIER_C) != input_signature(subject.files, TIER_C)
    # a stage that does not read the file is not redone
    assert input_signature(files, TIER_A) == input_signature(subject.files, TIER_A)


def test_the_input_signature_ignores_what_is_not_in_its_lines() -> None:
    _, subject = _records()
    files = dict(subject.files)
    files[FileRole.LH_PIAL] = dataclasses.replace(
        files[FileRole.LH_PIAL], last_modified="2026-11-12T00:00:00Z"
    )
    assert input_signature(files) == input_signature(subject.files)


@pytest.mark.parametrize("variant", ["lh.pial.T2", None])
def test_the_input_signature_changes_with_the_file_a_link_leads_to(variant: str | None) -> None:
    # lh.pial -> lh.pial.T1 and lh.pial -> lh.pial.T2 are one location, and the two files
    # have one size: with the same modification time only the variant tells them apart
    _, subject = _records()
    link = dataclasses.replace(subject.files[FileRole.LH_PIAL], location="surf/lh.pial")
    files = dict(subject.files) | {FileRole.LH_PIAL: link}
    other = files | {FileRole.LH_PIAL: dataclasses.replace(link, variant=variant)}
    assert input_signature(other) != input_signature(files)
    assert input_signature(other, TIER_C) != input_signature(files, TIER_C)
    assert input_signature(other, TIER_A) == input_signature(files, TIER_A)
    # a role without a variant has the four fields and no trailing tab
    assert (
        input_signature({FileRole.T1: files[FileRole.T1]})
        == (hashlib.sha256(b"t1\tmri/T1.mgz\t5576738\t1731400000000000000").hexdigest()[:16])
    )


class _MinimalSource:
    """The least a source has to be: what the protocol names, and nothing else."""

    kind = "local"

    def datasets(self, only: Collection[str] | None = None) -> Iterator[DatasetRecord]:
        yield _records()[0]

    def subjects(self, dataset: DatasetRecord) -> Iterator[SubjectRecord]:
        yield _records()[1]

    def read_bytes(
        self,
        subject: SubjectRecord,
        role: FileRole,
        *,
        byte_range: tuple[int, int] | None = None,
    ) -> bytes:
        return b""

    def lease(
        self, subject: SubjectRecord, roles: Collection[FileRole]
    ) -> AbstractContextManager[Mapping[FileRole, Path]]:
        return nullcontext({})

    def template_dir(self, dataset: DatasetRecord) -> Path | None:
        return None

    def config(self) -> SourceConfig:
        return SourceConfig(kind="local", root="/data/study")


def test_source_is_a_protocol_that_can_be_checked_at_run_time() -> None:
    source = _MinimalSource()
    assert isinstance(source, Source)
    assert not isinstance(object(), Source)

    class WithoutLease:
        kind = "local"
        datasets = subjects = read_bytes = template_dir = config = None

    assert not isinstance(WithoutLease(), Source)
    members = {name for name in dir(Source) if not name.startswith("_")}
    assert members | set(Source.__annotations__) == {
        "kind",
        "datasets",
        "subjects",
        "read_bytes",
        "lease",
        "template_dir",
        "config",
    }


# --- templates and subject candidates ---------------------------------------------------------


def test_template_files_are_per_hemisphere_paths_inside_an_fsaverage_directory() -> None:
    assert TEMPLATE_FILES == (
        "surf/{h}.sphere.reg",
        "surf/{h}.white",
        "label/{h}.cortex.label",
        "label/{h}.aparc.annot",
    )
    assert TEMPLATE_OPTIONAL_FILES == ("surf/{h}.white.avg.area.mgh",)
    names = [name.format(h=hemi) for name in TEMPLATE_FILES for hemi in HEMIS]
    assert len(set(names)) == 8
    for name in (*TEMPLATE_FILES, *TEMPLATE_OPTIONAL_FILES):
        assert name.count("{h}") == 1 and not PurePosixPath(name).is_absolute()


@pytest.mark.parametrize(
    "name",
    [
        "fsaverage",
        "fsaverage5",
        "fsaverage6",
        "fsaverage_sym",
        "cvs_avg35",
        "cvs_avg35_inMNI152",
        "V1_average",
        "lh.EC_average",
        "rh.EC_average",
    ],
)
def test_template_names(name: str) -> None:
    assert is_template_name(name)


@pytest.mark.parametrize(
    "name",
    [
        "sub-01",
        "sub-fsaverage",
        "Fsaverage",
        "FSAVERAGE",
        "V1_average2",
        "v1_average",
        "xh.EC_averag",
        "lh.EC_average.old",
        "bert",
        "",
    ],
)
def test_names_that_are_not_templates(name: str) -> None:
    assert not is_template_name(name)


def test_template_name_patterns_are_matched_with_case() -> None:
    assert TEMPLATE_NAME_PATTERNS == ("fsaverage*", "cvs_avg35*", "V1_average", "?h.EC_average")


def test_a_subject_candidate_has_mri_and_surf_and_is_not_a_template(tmp_path: Path) -> None:
    def directories(name: str, *children: str) -> Path:
        for child in children:
            (tmp_path / name / child).mkdir(parents=True)
        (tmp_path / name).mkdir(exist_ok=True)
        return tmp_path / name

    crashed = directories("sub-01", "mri", "surf", "scripts")
    assert is_subject_candidate(crashed) and is_subject_candidate(str(crashed))
    # a candidate is not yet a subject that can be rendered
    assert not is_freesurfer_subject_dir(crashed)
    assert is_subject_candidate(directories("sub-01.long.base", "mri", "surf"))
    assert not is_subject_candidate(directories("sub-02", "mri"))
    assert not is_subject_candidate(directories("sub-03", "surf", "label"))
    assert not is_subject_candidate(directories("logs"))
    assert not is_subject_candidate(tmp_path / "absent")
    for template in (
        "fsaverage",
        "fsaverage6",
        "cvs_avg35_inMNI152",
        "V1_average",
        "lh.EC_average",
    ):
        assert not is_subject_candidate(directories(template, "mri", "surf"))
    # mri and surf must be directories
    files = directories("sub-04", "mri")
    (files / "surf").touch()
    assert not is_subject_candidate(files)
    (tmp_path / "sub-05").touch()
    assert not is_subject_candidate(tmp_path / "sub-05")


def test_subject_candidates_through_links(tmp_path: Path) -> None:
    for name in ("sub-01", "fsaverage"):
        for child in ("mri", "surf"):
            (tmp_path / "store" / name / child).mkdir(parents=True)
    listing = tmp_path / "subjects"
    listing.mkdir()
    (listing / "sub-01").symlink_to(tmp_path / "store" / "sub-01")
    (listing / "alias").symlink_to(tmp_path / "store" / "sub-01")
    (listing / "average").symlink_to(tmp_path / "store" / "fsaverage")
    (listing / "fsaverage").symlink_to(tmp_path / "store" / "sub-01")
    (listing / "gone").symlink_to(tmp_path / "store" / "missing")
    assert is_subject_candidate(listing / "sub-01")
    assert is_subject_candidate(listing / "alias")
    # a template by the name of what the link points to, and by the name of the link
    assert not is_subject_candidate(listing / "average")
    assert not is_subject_candidate(listing / "fsaverage")
    assert not is_subject_candidate(listing / "gone")


def test_a_candidate_that_cannot_be_examined_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    subject = tmp_path / "sub-01"
    (subject / "mri").mkdir(parents=True)
    (subject / "surf").mkdir()
    stat = Path.stat

    def refuse(self: Path, *args: object, **kwargs: object) -> object:
        if self == subject / "surf":
            raise PermissionError(13, "Permission denied", str(self))
        return stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", refuse)
    with pytest.raises(PermissionError):
        is_subject_candidate(subject)


def test_a_candidate_directory_without_search_permission_raises(tmp_path: Path) -> None:
    if not file_modes_apply(tmp_path):
        pytest.skip("file modes do not restrict this user")
    subject = tmp_path / "sub-01"
    (subject / "mri").mkdir(parents=True)
    (subject / "surf").mkdir()
    subject.chmod(0)
    try:
        with pytest.raises(PermissionError):
            is_subject_candidate(subject)
    finally:
        subject.chmod(0o755)


def test_the_candidate_test_leaves_the_tree_untouched(synth_subject: SyntheticSubject) -> None:
    before = snapshot_tree(synth_subject.path)
    assert is_subject_candidate(synth_subject.path)
    assert snapshot_tree(synth_subject.path) == before


# --- variant_name -----------------------------------------------------------------------------


def test_the_variant_of_a_file_is_the_name_freesurfer_gives_it(tmp_path: Path) -> None:
    subject = _fake_subject(
        tmp_path, "sub-01", ("mri/T1.mgz", "surf/lh.pial.T1", "surf/rh.pial", "surf/lh.white")
    )
    surf = subject / "surf"
    assert variant_name(FileRole.T1, subject / "mri" / "T1.mgz") == "T1.mgz"
    assert variant_name(FileRole.RH_PIAL, surf / "rh.pial") == "rh.pial"
    assert variant_name(FileRole.LH_PIAL, str(surf / "lh.pial.T1")) == "lh.pial.T1"
    # FreeSurfer 7: lh.pial is a relative link to lh.pial.T1
    (surf / "lh.pial").symlink_to("lh.pial.T1")
    assert variant_name(FileRole.LH_PIAL, surf / "lh.pial") == "lh.pial.T1"
    # a link to a name that is no candidate of the role reads under its own name
    (surf / "rh.white").symlink_to("lh.white")
    assert variant_name(FileRole.RH_WHITE, surf / "rh.white") == "rh.white"
    # a link into another directory: no foreign name
    store = tmp_path / "store" / "ds000102_sub-01_lh.pial.T2"
    store.parent.mkdir()
    store.touch()
    (surf / "rh.pial.T2").symlink_to(store)
    assert variant_name(FileRole.RH_PIAL, surf / "rh.pial.T2") == "rh.pial.T2"
    # a file under a name of someone's choosing reads as the first candidate
    assert variant_name(FileRole.LH_WHITE, store) == "lh.white"


def test_a_link_into_another_directory_reads_under_its_own_name(tmp_path: Path) -> None:
    subject = _fake_subject(tmp_path, "sub-01", ("mri/T1.mgz", "surf/rh.pial"))
    surf = subject / "surf"
    for name in ("lh.pial.T1", "lh.pial.T2"):  # candidate names, in another directory
        (tmp_path / "elsewhere" / name).parent.mkdir(exist_ok=True)
        (tmp_path / "elsewhere" / name).touch()
    (surf / "lh.pial").symlink_to(tmp_path / "elsewhere" / "lh.pial.T1")
    assert variant_name(FileRole.LH_PIAL, surf / "lh.pial") == "lh.pial"
    (surf / "lh.pial.FLAIR").symlink_to(Path("..", "..", "..", "elsewhere", "lh.pial.T2"))
    assert variant_name(FileRole.LH_PIAL, surf / "lh.pial.FLAIR") == "lh.pial.FLAIR"
    # the directory itself may be reached through a link: the target is still inside it
    (surf / "rh.pial.T1").touch()
    (surf / "rh.pial").unlink()
    (surf / "rh.pial").symlink_to("rh.pial.T1")
    (tmp_path / "linked").symlink_to(subject, target_is_directory=True)
    assert variant_name(FileRole.RH_PIAL, tmp_path / "linked" / "surf" / "rh.pial") == "rh.pial.T1"


def test_the_variant_is_the_name_that_render_json_records(tmp_path: Path) -> None:
    from surfcensus.render import api  # imports matplotlib

    plain = _small_subject(tmp_path / "plain")
    linked = _small_subject(tmp_path / "linked", pial_variant="pial.T1")
    for hemi in ("lh", "rh"):
        (linked.path / "surf" / f"{hemi}.pial").symlink_to(f"{hemi}.pial.T1")
    moved = tmp_path / "elsewhere" / "some-name"
    moved.parent.mkdir()
    moved.write_bytes((plain.path / "surf" / "lh.pial").read_bytes())
    (plain.path / "surf" / "lh.pial.T2").symlink_to(moved)
    named = tmp_path / "elsewhere" / "lh.pial.T1"  # a candidate name, in another directory
    named.write_bytes(moved.read_bytes())
    (plain.path / "surf" / "lh.pial.FLAIR").symlink_to(named)
    cases = [
        *resolve_subject_files(plain.path).items(),
        *resolve_subject_files(linked.path).items(),
        (FileRole.LH_PIAL, plain.path / "surf" / "lh.pial.T2"),
        (FileRole.LH_PIAL, plain.path / "surf" / "lh.pial.FLAIR"),
        (FileRole.LH_PIAL, moved),
    ]
    assert (FileRole.LH_PIAL, linked.path / "surf" / "lh.pial") in cases
    for role, path in cases:
        assert variant_name(role, path) == api._input_file(role, path).name, (role, path)
    assert variant_name(FileRole.LH_PIAL, linked.path / "surf" / "lh.pial") == "lh.pial.T1"
    foreign = api._input_file(FileRole.LH_PIAL, plain.path / "surf" / "lh.pial.FLAIR")
    assert (foreign.name, foreign.size) == ("lh.pial.FLAIR", named.stat().st_size)
