"""Synthetic directory trees in the layouts a local source detects, written with nibabel.

A tree holds light subjects (16^3 voxels of 10 mm, 42 vertices per surface, about 8 kB)
that pass ``is_freesurfer_subject_dir`` and render in a fraction of a second, so tests of
sources, inventory and runs can afford several of them. Nothing here is a fixture: a test
writes its own tree under ``tmp_path``.

The template directories have the file names of a FreeSurfer ``fsaverage`` and, like the
real ones (``fsaverage`` and ``fsaverage5`` next to the subjects of ds000102-fmriprep hold
``mri/`` and ``surf/``, the latter with ``mri/T1.mgz``), would pass for subjects but for
their name. Their content is a 42-vertex mesh: valid files, not a usable template. The
template of the metrics tests is ``synth_template.write_synthetic_template`` (milestone 3).
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Collection, Sequence
from pathlib import Path

import nibabel.freesurfer as fs
import numpy as np
from synth import icosphere, write_synthetic_subject  # pyrefly: ignore[missing-import]

from surfcensus.sources.base import HEMIS, Layout
from surfcensus.sources.patterns import TEMPLATE_FILES

FS_ROOT: dict[Layout, str] = {
    "subjects_dir": ".",
    "fmriprep": "sourcedata/freesurfer",
    "fmriprep_legacy": "freesurfer",
    "bids_derivatives": "freesurfer",
}
"""Where each layout keeps its subjects, relative to the path that ``write_tree`` returns."""

DANGLING_TARGET = "../missing-freesurfer/subjects/fsaverage"
RECON_LOGS = Path(__file__).parent / "data" / "recon_logs"
BUILD_STAMPS = {  # the build-stamp.txt of the two subjects the log excerpts are cut from
    "fs6": "freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.1-f53a55a",
    "fs7": "freesurfer-linux-ubuntu22_x86_64-7.3.2-20220804-6354275",
}
_PARCELS = ("unknown", "cuneus", "lateraloccipital", "superiorfrontal")


def write_light_subject(path: Path, *, complete: bool = True) -> None:
    """The smallest directory that passes ``is_freesurfer_subject_dir`` and renders.

    ``mri/T1.mgz`` and ``mri/brainmask.mgz`` of 16^3 voxels (10 mm, so the anatomy of
    ``tests/synth.py`` fits), and 42-vertex white and pial surfaces of both hemispheres.
    ``complete=False`` leaves out ``surf/lh.pial``: a subject candidate that cannot be
    rendered.
    """
    path = Path(path)
    write_synthetic_subject(
        path.parent, name=path.name, voxel_size=10.0, shape=(16, 16, 16), subdivisions=1
    )
    if not complete:
        (path / "surf" / "lh.pial").unlink()


def write_tier_a(subject_dir: Path, version: str = "fs6") -> None:
    """Add what the inventory reads to a subject: the build stamp and the three recon-all
    logs, which are the excerpts of a real subject under ``tests/data/recon_logs``
    (``fs6``: ds000102 sub-01, FreeSurfer 6.0.1; ``fs7``: ds003020 sub-UTS01, 7.3.2), and
    stats files of one line, which the inventory lists and does not read."""
    scripts = Path(subject_dir) / "scripts"
    scripts.mkdir(exist_ok=True)
    (scripts / "build-stamp.txt").write_text(BUILD_STAMPS[version] + "\n")
    for name in ("recon-all.log", "recon-all-lh.log", "recon-all-rh.log"):
        shutil.copyfile(RECON_LOGS / f"{version}_{name}", scripts / name)
    stats = Path(subject_dir) / "stats"
    stats.mkdir(exist_ok=True)
    for name in ("lh.aparc.stats", "rh.aparc.stats", "aseg.stats"):
        (stats / name).write_text("# Measure Cortex, NumVert, Number of Vertices, 42, unitless\n")


def write_label(path: Path, vertices: np.ndarray, coords: np.ndarray) -> None:
    """A FreeSurfer ``.label`` file: plain text, which nibabel reads and does not write."""
    lines = ["#!ascii label  , from subject fsaverage vox2ras=TkReg", str(len(vertices))]
    lines += [
        f"{vertex} {x:.3f} {y:.3f} {z:.3f} 0.0000000000"
        for vertex, (x, y, z) in zip(vertices, coords, strict=True)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_light_template(path: Path, *, complete: bool = True) -> None:
    """A directory with every file of ``patterns.TEMPLATE_FILES`` for both hemispheres, on
    the 42-vertex mesh of a light subject. ``complete=False`` leaves out the annotations."""
    path = Path(path)
    write_light_subject(path)
    unit, faces = icosphere(1)
    cortex = np.flatnonzero(np.abs(unit[:, 0]) < 0.9)  # two vertices stand for the medial wall
    parcel = np.where(unit[:, 1] < 0, 1 + (unit[:, 2] < 0), 3)
    parcel[np.abs(unit[:, 0]) >= 0.9] = -1
    colours = np.array([[25, 5, 25, 0], [220, 20, 100, 0], [20, 30, 140, 0], [20, 220, 160, 0]])
    for hemi in HEMIS:
        fs.write_geometry(str(path / "surf" / f"{hemi}.sphere.reg"), unit * 100.0, faces)
        white = fs.read_geometry(str(path / "surf" / f"{hemi}.white"))[0]
        write_label(path / "label" / f"{hemi}.cortex.label", cortex, white[cortex])
        if complete:
            fs.write_annot(
                str(path / "label" / f"{hemi}.aparc.annot"),
                parcel.astype(np.int32),
                colours,
                list(_PARCELS),
                fill_ctab=True,
            )
    expected = {name.format(h=hemi) for name in TEMPLATE_FILES for hemi in HEMIS}
    missing = {name for name in expected if not (path / name).is_file()}
    assert complete == (not missing), f"template files of this fixture are out of date: {missing}"


def _description(version: str, *, legacy: bool) -> dict[str, object]:
    base: dict[str, object] = {"Name": "fMRIPrep - fMRI PREProcessing workflow"}
    if legacy:  # the form before BIDS derivatives named GeneratedBy; from memory, not from a file
        return base | {
            "BIDSVersion": "1.1.1",
            "PipelineDescription": {"Name": "fMRIPrep", "Version": version},
        }
    # the keys of fmriprep/ds000102-fmriprep/dataset_description.json (fMRIPrep 21.0.2)
    return base | {
        "BIDSVersion": "1.4.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "fMRIPrep", "Version": version}],
    }


def write_tree(
    root: Path,
    layout: Layout,
    labels: Sequence[str],
    *,
    with_template: bool = False,
    dangling_template: bool = False,
    incomplete: Collection[str] = (),
    fmriprep_version: str | None = "23.2.0",
) -> Path:
    """Write a tree of light subjects in ``layout`` under ``root``; returns the path a user
    would pass, which is ``root`` in every layout.

    * ``subjects_dir``: ``root/<label>``.
    * ``fmriprep``: ``root/sourcedata/freesurfer/<label>``, ``root/dataset_description.json``
      (``GeneratedBy``), and fMRIPrep's own ``root/<label>/anat`` and ``root/<label>.html``.
    * ``fmriprep_legacy``: ``root/freesurfer/<label>`` beside ``root/fmriprep/`` with a
      ``dataset_description.json`` in the old form (``PipelineDescription``) and the
      per-subject outputs. A user may also pass ``root / "fmriprep"``.
    * ``bids_derivatives``: ``root/freesurfer/<label>`` and no ``root/fmriprep``.

    ``incomplete`` labels lack ``surf/lh.pial``. ``with_template`` adds ``fsaverage`` (every
    template file) and ``fsaverage5`` next to the subjects; ``dangling_template`` adds
    ``fsaverage`` as a link to nothing instead. ``fmriprep_version=None`` writes no
    ``dataset_description.json``.
    """
    if with_template and dangling_template:
        raise ValueError("a tree has a template or a dangling link to one, not both")
    unknown = set(incomplete) - set(labels)
    if unknown:
        raise ValueError(f"incomplete names subjects that are not in labels: {sorted(unknown)}")
    root = Path(root)
    subjects_dir = root / FS_ROOT[layout]
    subjects_dir.mkdir(parents=True, exist_ok=True)
    for label in labels:
        write_light_subject(subjects_dir / label, complete=label not in incomplete)
    if with_template:
        write_light_template(subjects_dir / "fsaverage")
        write_light_subject(subjects_dir / "fsaverage5")
    if dangling_template:
        (subjects_dir / "fsaverage").symlink_to(DANGLING_TARGET)

    outputs = {"fmriprep": root, "fmriprep_legacy": root / "fmriprep"}.get(layout)
    if outputs is not None:
        for label in labels:
            (outputs / label / "anat").mkdir(parents=True, exist_ok=True)
            (outputs / f"{label}.html").write_text("<html></html>\n")
        if fmriprep_version is not None:
            description = _description(fmriprep_version, legacy=layout == "fmriprep_legacy")
            (outputs / "dataset_description.json").write_text(
                json.dumps(description, indent=2) + "\n"
            )
    return root


def write_bids_dataset(
    root: Path, labels: Sequence[str] = ("sub-01", "sub-02"), *, description: bool = True
) -> Path:
    """The raw part of a BIDS dataset as OpenNeuro hands one out, under ``root``:
    ``dataset_description.json`` (unless ``description`` is false), ``participants.tsv``,
    ``README`` and ``<label>/anat/<label>_T1w.nii.gz`` (a file of a few bytes: nothing reads
    it). No ``derivatives/`` folder: a test adds what it needs with :func:`write_tree`, for
    example ``write_tree(root / "derivatives" / "fmriprep", "fmriprep", labels)`` for
    ``derivatives/fmriprep/sourcedata/freesurfer/<label>`` or
    ``write_tree(root / "derivatives", "bids_derivatives", labels)`` for
    ``derivatives/freesurfer/<label>``. Returns ``root``, the path a user would pass.

    The file names are those of a BIDS dataset from memory of the specification; no real
    OpenNeuro dataset root was listed for this.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if description:
        raw = {"Name": "A synthetic dataset", "BIDSVersion": "1.8.0", "DatasetType": "raw"}
        (root / "dataset_description.json").write_text(json.dumps(raw, indent=2) + "\n")
    (root / "README").write_text("Synthetic.\n")
    (root / "participants.tsv").write_text(
        "participant_id\n" + "".join(f"{label}\n" for label in labels)
    )
    for label in labels:
        (root / label / "anat").mkdir(parents=True, exist_ok=True)
        (root / label / "anat" / f"{label}_T1w.nii.gz").write_bytes(b"not an image\n")
    return root


def write_linked_tree(
    root: Path, store: Path, labels: Sequence[str], *, with_template: bool = False
) -> Path:
    """A subjects directory ``root`` whose subjects are links to light subjects in ``store``,
    as a study that takes its subjects from a shared place; with ``with_template`` also
    ``root/fsaverage`` as a link to ``store/fsaverage``. Returns ``root``."""
    root, store = Path(root), Path(store)
    root.mkdir(parents=True, exist_ok=True)
    for label in labels:
        write_light_subject(store / label)
        (root / label).symlink_to(store / label, target_is_directory=True)
    if with_template:
        write_light_template(store / "fsaverage")
        (root / "fsaverage").symlink_to(store / "fsaverage", target_is_directory=True)
    return root
