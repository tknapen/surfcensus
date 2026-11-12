"""sources/patterns: file roles inside a FreeSurfer subject directory."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import numpy as np
import pytest
from synth import (  # pyrefly: ignore[missing-import]
    SyntheticSubject,
    snapshot_tree,
    write_synthetic_subject,
)

from surfcensus.fsio.surfaces import read_surface
from surfcensus.sources.base import RENDER_ROLES, FileRole
from surfcensus.sources.patterns import (
    ROLE_CANDIDATES,
    is_freesurfer_subject_dir,
    missing_roles,
    resolve_subject_files,
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
