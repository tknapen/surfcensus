"""The trees of tests/synth_tree.py are what they say: layouts, light subjects, templates."""

from __future__ import annotations

import json
import os
from pathlib import Path

import nibabel.freesurfer as fs
import numpy as np
import pytest
from synth import snapshot_tree  # pyrefly: ignore[missing-import]
from synth_tree import (  # pyrefly: ignore[missing-import]
    DANGLING_TARGET,
    FS_ROOT,
    write_light_subject,
    write_light_template,
    write_tree,
)

from surfcensus.inventory.provenance import parse_dataset_description
from surfcensus.render import RenderConfig, render_subject
from surfcensus.sources.base import HEMIS, RENDER_ROLES, FileRole
from surfcensus.sources.patterns import (
    TEMPLATE_FILES,
    is_freesurfer_subject_dir,
    is_subject_candidate,
    missing_roles,
    resolve_subject_files,
)

LAYOUTS = ("subjects_dir", "fmriprep", "fmriprep_legacy", "bids_derivatives")
LABELS = ("sub-01", "sub-02", "sub-03")


def test_a_light_subject_is_small_complete_and_renders(tmp_path: Path) -> None:
    subject = tmp_path / "tree" / "sub-01"
    write_light_subject(subject)
    assert is_freesurfer_subject_dir(subject) and is_subject_candidate(subject)
    files = resolve_subject_files(subject)
    assert set(files) == {*RENDER_ROLES, FileRole.BRAINMASK}
    assert sum(p.stat().st_size for p in subject.rglob("*") if p.is_file()) < 20_000
    vertices, faces = fs.read_geometry(str(subject / "surf" / "lh.pial"))
    assert vertices.shape == (42, 3) and faces.shape == (80, 3)

    before = snapshot_tree(subject)
    info = render_subject(files, tmp_path / "media", RenderConfig(panel_px=64), video=False)
    assert snapshot_tree(subject) == before
    assert all(len(panel.slices) >= 5 for panel in info.panels.values())
    assert info.vertex_map is not None and info.vertex_map.n_lh > 0 and info.vertex_map.n_rh > 0
    assert (tmp_path / "media" / "render.json").is_file()


def test_an_incomplete_light_subject_is_a_candidate_that_cannot_be_rendered(
    tmp_path: Path,
) -> None:
    subject = tmp_path / "sub-01"
    write_light_subject(subject, complete=False)
    assert not (subject / "surf" / "lh.pial").exists()
    assert is_subject_candidate(subject)
    assert not is_freesurfer_subject_dir(subject)
    assert missing_roles(resolve_subject_files(subject)) == [FileRole.LH_PIAL]


@pytest.mark.parametrize("layout", LAYOUTS)
def test_each_layout_puts_the_subjects_where_it_says(tmp_path: Path, layout: str) -> None:
    path = write_tree(tmp_path / "study", layout, LABELS, incomplete=["sub-02"])
    assert path == tmp_path / "study"
    subjects_dir = path / FS_ROOT[layout]
    assert sorted(p.name for p in subjects_dir.iterdir() if is_subject_candidate(p)) == [*LABELS]
    assert [is_freesurfer_subject_dir(subjects_dir / label) for label in LABELS] == [
        True,
        False,
        True,
    ]
    assert not (subjects_dir / "fsaverage").exists()
    # where the subjects are is the only place with subject candidates
    candidates = {p.parent for p in path.rglob("*") if p.is_dir() and is_subject_candidate(p)}
    assert candidates == {subjects_dir}


def test_the_layouts_differ_in_what_lies_beside_the_subjects(tmp_path: Path) -> None:
    plain = write_tree(tmp_path / "plain", "subjects_dir", LABELS)
    assert sorted(p.name for p in plain.iterdir()) == [*LABELS]

    new = write_tree(tmp_path / "new", "fmriprep", LABELS)
    assert sorted(p.name for p in new.iterdir()) == sorted(
        ["dataset_description.json", "sourcedata", *LABELS, *(f"{x}.html" for x in LABELS)]
    )
    description = json.loads((new / "dataset_description.json").read_text())
    assert description["GeneratedBy"] == [{"Name": "fMRIPrep", "Version": "23.2.0"}]
    assert "PipelineDescription" not in description

    legacy = write_tree(tmp_path / "legacy", "fmriprep_legacy", LABELS, fmriprep_version="1.5.0")
    assert sorted(p.name for p in legacy.iterdir()) == ["fmriprep", "freesurfer"]
    description = json.loads((legacy / "fmriprep" / "dataset_description.json").read_text())
    assert description["PipelineDescription"] == {"Name": "fMRIPrep", "Version": "1.5.0"}
    assert "GeneratedBy" not in description
    assert (legacy / "fmriprep" / "sub-01" / "anat").is_dir()

    bids = write_tree(tmp_path / "bids", "bids_derivatives", LABELS)
    assert sorted(p.name for p in bids.iterdir()) == ["freesurfer"]

    # both forms give the version back
    assert parse_dataset_description((new / "dataset_description.json").read_text()) == "23.2.0"
    assert (
        parse_dataset_description((legacy / "fmriprep" / "dataset_description.json").read_text())
        == "1.5.0"
    )

    for layout in ("fmriprep", "fmriprep_legacy"):
        bare = write_tree(tmp_path / f"bare-{layout}", layout, LABELS, fmriprep_version=None)
        assert not list(bare.rglob("dataset_description.json"))


def test_the_template_has_every_template_file_and_would_pass_for_a_subject_but_for_its_name(
    tmp_path: Path,
) -> None:
    path = write_tree(tmp_path, "fmriprep", LABELS, with_template=True)
    subjects_dir = path / FS_ROOT["fmriprep"]
    template = subjects_dir / "fsaverage"
    for name in TEMPLATE_FILES:
        for hemi in HEMIS:
            assert (template / name.format(h=hemi)).is_file()
    assert sorted(p.name for p in subjects_dir.iterdir()) == ["fsaverage", "fsaverage5", *LABELS]
    for name in ("fsaverage", "fsaverage5"):
        assert not is_subject_candidate(subjects_dir / name)
        assert not is_freesurfer_subject_dir(subjects_dir / name)
        # under another name the same directory is a subject
        os.rename(subjects_dir / name, subjects_dir / f"x{name}")
        assert is_subject_candidate(subjects_dir / f"x{name}")
        assert is_freesurfer_subject_dir(subjects_dir / f"x{name}")
    # fsaverage5 is a template by name only: it lacks the label files
    assert not (subjects_dir / "xfsaverage5" / "label").exists()


def test_the_template_files_are_read_by_nibabel(tmp_path: Path) -> None:
    template = tmp_path / "fsaverage"
    write_light_template(template)
    for hemi in HEMIS:
        sphere, faces = fs.read_geometry(str(template / "surf" / f"{hemi}.sphere.reg"))
        white, white_faces = fs.read_geometry(str(template / "surf" / f"{hemi}.white"))
        assert sphere.shape == white.shape == (42, 3)
        np.testing.assert_array_equal(faces, white_faces)
        np.testing.assert_allclose(np.linalg.norm(sphere, axis=1), 100.0, atol=1e-4)
        cortex = fs.read_label(str(template / "label" / f"{hemi}.cortex.label"))
        assert cortex.dtype.kind == "i" and len(cortex) == 40 and len(set(cortex.tolist())) == 40
        labels, _, names = fs.read_annot(str(template / "label" / f"{hemi}.aparc.annot"))
        assert labels.shape == (42,)
        assert [name.decode() for name in names] == [
            "unknown",
            "cuneus",
            "lateraloccipital",
            "superiorfrontal",
        ]
        # the two vertices outside the cortex label carry no parcel, as on fsaverage
        assert sorted(np.flatnonzero(labels == -1).tolist()) == sorted(
            set(range(42)) - set(cortex.tolist())
        )


def test_an_incomplete_template_lacks_the_annotations(tmp_path: Path) -> None:
    write_light_template(tmp_path / "fsaverage", complete=False)
    present = {p.relative_to(tmp_path / "fsaverage").as_posix() for p in tmp_path.rglob("*")}
    assert "label/lh.cortex.label" in present and "surf/rh.sphere.reg" in present
    assert "label/lh.aparc.annot" not in present and "label/rh.aparc.annot" not in present


def test_a_dangling_template_is_a_link_to_nothing(tmp_path: Path) -> None:
    path = write_tree(tmp_path, "subjects_dir", LABELS, dangling_template=True)
    link = path / "fsaverage"
    assert link.is_symlink() and not link.exists()
    assert os.readlink(link) == DANGLING_TARGET
    assert not is_subject_candidate(link)
    assert not (path / "fsaverage5").exists()


def test_write_tree_refuses_contradictions(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not both"):
        write_tree(tmp_path, "subjects_dir", LABELS, with_template=True, dangling_template=True)
    with pytest.raises(ValueError, match="not in labels"):
        write_tree(tmp_path, "subjects_dir", LABELS, incomplete=["sub-09"])


@pytest.mark.parametrize(
    ("description", "version"),
    [
        # the entry of fmriprep/ds000102-fmriprep/dataset_description.json
        (
            {
                "Name": "ds000102-fmriprep",
                "BIDSVersion": "1.4.0",
                "DatasetType": "derivative",
                "GeneratedBy": [
                    {
                        "Name": "fMRIPrep",
                        "Version": "21.0.2",
                        "CodeURL": "https://github.com/nipreps/fmriprep/archive/21.0.2.tar.gz",
                    }
                ],
            },
            "21.0.2",
        ),
        ({"GeneratedBy": [{"Name": "FMRIPREP", "Version": " 20.2.7 "}]}, "20.2.7"),
        (
            {
                "GeneratedBy": [
                    {"Name": "dcm2niix", "Version": "1.0"},
                    {"Name": "fmriprep", "Version": "22.1.1"},
                ]
            },
            "22.1.1",
        ),
        ({"PipelineDescription": {"Name": "fMRIPrep", "Version": "1.5.0"}}, "1.5.0"),
        (
            {
                "GeneratedBy": [{"Name": "fMRIPrep"}],
                "PipelineDescription": {"Version": "1.4.1"},
            },
            "1.4.1",
        ),
        ({"GeneratedBy": [{"Name": "FreeSurfer", "Version": "7.3.2"}]}, None),
        ({"GeneratedBy": [{"Name": "fMRIPrep", "Version": 21}]}, None),
        ({"GeneratedBy": "fMRIPrep 21.0.2", "PipelineDescription": "1.5.0"}, None),
        ({"GeneratedBy": [None, "x", 3]}, None),
        ({"Name": "a raw dataset"}, None),
        (["a", "list"], None),
        ("a string", None),
        (None, None),
    ],
)
def test_the_pipeline_version_of_a_dataset_description(
    description: object, version: str | None
) -> None:
    assert parse_dataset_description(json.dumps(description)) == version


@pytest.mark.parametrize("text", ["", "{", "not json", "\x00", "[" * 100_000, '{"GeneratedBy": '])
def test_a_broken_dataset_description_gives_no_version_and_no_error(text: str) -> None:
    assert parse_dataset_description(text) is None
