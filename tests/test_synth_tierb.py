"""The Tier A and Tier B fixtures and the cohorts are what ``tests/synth_tierb.py`` says.

Checked against nibabel, numpy and the parts of the package that milestone 2 built (the
local source, the provenance rules, the inventory, the renderer), not against the metrics
code, so these tests hold while that code is being written and say what it can rely on.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import nibabel.freesurfer as fs
import numpy as np
import pytest
from scipy.spatial import cKDTree
from synth import (  # pyrefly: ignore[missing-import]
    PIAL_SEMI_AXES,
    SyntheticSubject,
    snapshot_tree,
    write_synthetic_subject,
)
from synth_template import (  # pyrefly: ignore[missing-import]
    DESIKAN_NAMES,
    MEDIAL_WALL_M,
    SyntheticTemplate,
    desikan_colours,
    direction,
    expected_regions,
    medial_coordinate,
    parcel_of,
    template_mesh,
    write_synthetic_template,
)
from synth_tierb import (  # pyrefly: ignore[missing-import]
    CANARIES,
    NOISE_SD_MM,
    WARP_AXIS,
    SyntheticCohort,
    ThickPatch,
    add_tier_a,
    add_tier_b,
    cohort_patches,
    expected_thickness,
    in_patch,
    subject_directions,
    synthetic_cohort,
    unwarp_directions,
    warp_directions,
)

from surfcensus.inventory.provenance import decode_text, subject_provenance
from surfcensus.records import Provenance
from surfcensus.render import RenderConfig, render_subject
from surfcensus.run import Run
from surfcensus.sources.base import HEMIS, RENDER_ROLES, TIER_A, TIER_B, FileRole
from surfcensus.sources.local import LocalSource, find_adjacent_template, local
from surfcensus.sources.patterns import resolve_subject_files

OCCIPITAL = ("cuneus", "lateraloccipital", "lingual", "pericalcarine")
CONTROL = ("caudalmiddlefrontal", "rostralmiddlefrontal", "superiorfrontal")
LOG_ROLES = (FileRole.BUILD_STAMP, FileRole.RECON_LOG, FileRole.RECON_LOG_LH, FileRole.RECON_LOG_RH)


def light_subject(root: Path, name: str = "sub-x", subdivisions: int = 3) -> SyntheticSubject:
    return write_synthetic_subject(
        root, name=name, voxel_size=10.0, shape=(16, 16, 16), subdivisions=subdivisions
    )


def tree_digest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def provenance(subjects_dir: Path, label: str) -> Provenance:
    """What the inventory of milestone 2 would learn about one subject of a directory."""
    source = LocalSource(subjects_dir)
    dataset = next(iter(source.datasets()))
    record = next(r for r in source.subjects(dataset) if r.subject_label == label)
    texts = {
        role: decode_text(source.read_bytes(record, role))
        for role in LOG_ROLES
        if role in record.files
    }
    return subject_provenance(texts, record.files, label)


def stats_rows(path: Path) -> dict[str, tuple[int, int, int, float, float]]:
    """Parcel -> (NumVert, SurfArea, GrayVol, ThickAvg, ThickStd) of an aparc.stats file."""
    rows = {}
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        fields = line.split()
        assert len(fields) == 10, line
        rows[fields[0]] = (
            int(fields[1]),
            int(fields[2]),
            int(fields[3]),
            float(fields[4]),
            float(fields[5]),
        )
    return rows


def occipital_control_ratio(subject_dir: Path) -> float:
    """Area-weighted occipital over control thickness from the two aparc.stats files."""

    def mean(parcels: tuple[str, ...]) -> float:
        area = thick = 0.0
        for hemi in HEMIS:
            rows = stats_rows(subject_dir / "stats" / f"{hemi}.aparc.stats")
            for parcel in parcels:
                area += rows[parcel][1]
                thick += rows[parcel][1] * rows[parcel][3]
        return thick / area

    return mean(OCCIPITAL) / mean(CONTROL)


@pytest.fixture(scope="module")
def template(tmp_path_factory: pytest.TempPathFactory) -> SyntheticTemplate:
    return write_synthetic_template(tmp_path_factory.mktemp("template4"), order=4)


# ---------------------------------------------------------------------------------------
# the warp


def test_the_warp_is_a_twist_about_its_axis_with_a_closed_form_inverse() -> None:
    directions, _ = template_mesh(5)
    assert np.linalg.norm(WARP_AXIS) == pytest.approx(1.0)
    assert np.array_equal(warp_directions(directions, 0.0), directions)
    for warp in (0.15, 0.3, -0.2):
        warped = warp_directions(directions, warp)
        np.testing.assert_allclose(np.linalg.norm(warped, axis=1), 1.0, atol=1e-12)
        # every latitude circle about the axis turns rigidly
        np.testing.assert_allclose(warped @ WARP_AXIS, directions @ WARP_AXIS, atol=1e-12)
        np.testing.assert_allclose(unwarp_directions(warped, warp), directions, atol=1e-12)
        np.testing.assert_allclose(
            warp_directions(unwarp_directions(directions, warp), warp), directions, atol=1e-12
        )
        shift = np.arccos(np.clip(np.einsum("ij,ij->i", warped, directions), -1, 1))
        assert 0.8 * abs(warp) < shift.max() < abs(warp)
    # the points on the axis, and those on its equator, stay where they are
    np.testing.assert_allclose(warp_directions(WARP_AXIS, 0.3), WARP_AXIS, atol=1e-12)
    on_equator = np.cross(WARP_AXIS, [1.0, 0.0, 0.0])
    on_equator /= np.linalg.norm(on_equator)
    np.testing.assert_allclose(warp_directions(on_equator, 0.3), on_equator, atol=1e-12)


@pytest.mark.parametrize("subdivisions", [3, 5])
@pytest.mark.parametrize("warp", [0.0, 0.15, 0.3])
def test_the_registered_sphere(
    tmp_path: Path, template: SyntheticTemplate, subdivisions: int, warp: float
) -> None:
    subject = light_subject(tmp_path, subdivisions=subdivisions)
    add_tier_b(subject, template, warp=warp)
    directions, faces = subject_directions(subject)
    assert len(directions) == subject.n_vertices
    for hemi in HEMIS:
        sphere, sphere_faces = fs.read_geometry(str(subject.path / "surf" / f"{hemi}.sphere.reg"))
        white_faces = fs.read_geometry(str(subject.path / "surf" / f"{hemi}.white"))[1]
        assert np.array_equal(sphere_faces, faces) and np.array_equal(white_faces, faces)
        np.testing.assert_allclose(np.linalg.norm(sphere, axis=1), 100.0, atol=1e-4)
        np.testing.assert_allclose(sphere, warp_directions(directions, warp) * 100.0, atol=1e-5)
        # no face is folded or wound inward: det = a . (b x c) > 0 for every face
        a, b, c = (sphere[sphere_faces[:, k]] for k in range(3))
        assert (np.einsum("ij,ij->i", a, np.cross(b, c)) > 0).all()
    if warp == 0.0:
        # the subject's vertices are the first vertices of the template mesh
        sphere = fs.read_geometry(str(subject.path / "surf" / "lh.sphere.reg"))[0]
        coarse = min(subdivisions, template.order)
        count = 10 * 4**coarse + 2
        np.testing.assert_allclose(sphere[:count], template_mesh(coarse)[0] * 100.0, atol=1e-5)


def test_flip_winding_reverses_every_face(tmp_path: Path, template: SyntheticTemplate) -> None:
    subject = light_subject(tmp_path)
    add_tier_b(subject, template, flip_winding=True, hemis=("lh",))
    sphere, faces = fs.read_geometry(str(subject.path / "surf" / "lh.sphere.reg"))
    a, b, c = (sphere[faces[:, k]] for k in range(3))
    assert (np.einsum("ij,ij->i", a, np.cross(b, c)) < 0).all()
    assert np.array_equal(faces[:, ::-1], subject_directions(subject)[1])
    np.testing.assert_allclose(
        sphere, warp_directions(subject_directions(subject)[0], 0.15) * 100, atol=1e-5
    )
    # hemis=("lh",) wrote the left hemisphere only
    assert not (subject.path / "surf" / "rh.sphere.reg").exists()
    assert not (subject.path / "surf" / "rh.thickness").exists()
    assert not (subject.path / "label" / "rh.aparc.annot").exists()


def test_the_template_pole_has_a_closed_form_native_position(
    tmp_path: Path, template: SyntheticTemplate
) -> None:
    # decision D12: the template's pole vertex, taken to the native surface through the
    # registered sphere, lies where the inverse warp puts it. What the nearest native
    # vertex adds is at most the spacing of the subject's mesh.
    subject = light_subject(tmp_path, subdivisions=5)
    warp = 0.25
    add_tier_b(subject, template, warp=warp)
    directions, _ = subject_directions(subject)
    pole = template_mesh(template.order)[0][template.pole_vertex]
    sphere = fs.read_geometry(str(subject.path / "surf" / "lh.sphere.reg"))[0]
    native = int(np.argmin(np.linalg.norm(sphere / 100.0 - pole, axis=1)))
    expected = unwarp_directions(pole, warp)
    spacing = math.sqrt(4 * math.pi / subject.n_vertices)
    assert np.linalg.norm(directions[native] - expected) < spacing
    assert np.linalg.norm(expected - pole) > 5 * spacing  # the warp moved it by much more
    pial = fs.read_geometry(str(subject.path / "surf" / "lh.pial"))[0]
    centre, semi = np.array(subject.lh_centre), np.array(PIAL_SEMI_AXES)
    assert np.linalg.norm(pial[native] - (centre + expected * semi)) < spacing * max(semi)
    geometric = pial[int(np.argmin(pial[:, 1]))]
    np.testing.assert_allclose(geometric, centre + pole * semi, atol=1e-4)
    constructed = float(np.linalg.norm((expected - pole) * semi))
    assert np.linalg.norm(pial[native] - geometric) == pytest.approx(
        constructed, abs=spacing * max(semi)
    )
    assert constructed > 5.0  # mm


# ---------------------------------------------------------------------------------------
# thickness and annotation


def test_expected_thickness_is_a_smooth_field_with_an_optional_patch() -> None:
    directions, _ = template_mesh(5)
    plain = expected_thickness(directions)
    np.testing.assert_allclose(
        plain, 2.5 + 0.35 * directions[:, 2] + 0.25 * directions[:, 0] * directions[:, 1]
    )
    assert 2.0 < plain.min() < 2.2 and 2.8 < plain.max() < 3.0
    patch = ThickPatch(centre=(0.0, -2.0, 0.0), radius_deg=12.0, value_mm=4.2)
    inside = in_patch(directions, patch)
    assert np.array_equal(inside, directions[:, 1] <= -math.cos(math.radians(12.0)))
    patched = expected_thickness(directions, patch)
    assert (patched[inside] == 4.2).all() and np.array_equal(patched[~inside], plain[~inside])
    clamped = expected_thickness(directions, ThickPatch((0.0, -1.0, 0.0), 12.0, 7.5))
    assert (clamped[inside] == 5.0).all()
    assert expected_thickness(np.array([0.0, -1.0, 0.0])) == pytest.approx(2.5)


@pytest.mark.parametrize("hemi", HEMIS)
def test_thickness_and_annotation_follow_the_registered_direction(
    tmp_path: Path, template: SyntheticTemplate, hemi: str
) -> None:
    subject = light_subject(tmp_path, subdivisions=5)
    patch = cohort_patches(value_mm=6.0)["inferior"]
    add_tier_b(subject, template, warp=0.2, patch=patch, offset_mm=0.3)
    directions, _ = subject_directions(subject)
    registered = warp_directions(directions, 0.2)
    medial_strip = medial_coordinate(registered, hemi) > MEDIAL_WALL_M
    patched = in_patch(registered, patch)
    assert 100 < patched.sum() < 400 and not (patched & medial_strip).any()

    thickness = fs.read_morph_data(str(subject.path / "surf" / f"{hemi}.thickness"))
    assert thickness.dtype == np.dtype(">f4") and thickness.shape == (subject.n_vertices,)
    assert (thickness[medial_strip] == 0).all() and (thickness[~medial_strip] > 0).all()
    assert (thickness[patched] == 5.0).all()  # the value, clamped; no offset inside the patch
    plain = ~medial_strip & ~patched
    np.testing.assert_allclose(
        thickness[plain], expected_thickness(registered)[plain] + 0.3, atol=1e-6
    )
    assert thickness.max() == 5.0

    labels, ctab, names = fs.read_annot(str(subject.path / "label" / f"{hemi}.aparc.annot"))
    assert tuple(name.decode() for name in names) == DESIKAN_NAMES
    assert np.array_equal(ctab[:, :4], desikan_colours())
    assert np.array_equal(labels, parcel_of(registered, hemi))
    assert np.array_equal(labels == -1, medial_strip)
    # the parcel of a vertex is the parcel of the template point it is registered to, not
    # of its own direction: the warp moves the borders
    assert (labels != parcel_of(directions, hemi)).sum() > 200


def test_the_noise_is_seeded_and_the_default_has_none(
    tmp_path: Path, template: SyntheticTemplate
) -> None:
    def thickness(name: str, **options: object) -> dict[str, np.ndarray]:
        subject = light_subject(tmp_path, name=name, subdivisions=5)
        add_tier_b(subject, template, **options)  # pyrefly: ignore[bad-argument-type]
        return {
            hemi: fs.read_morph_data(str(subject.path / "surf" / f"{hemi}.thickness"))
            for hemi in HEMIS
        }

    plain, seven, again, eight = (
        thickness("a"),
        thickness("b", noise_seed=7),
        thickness("c", noise_seed=7),
        thickness("d", noise_seed=8),
    )
    for hemi in HEMIS:
        assert np.array_equal(seven[hemi], again[hemi])
        assert not np.array_equal(seven[hemi], eight[hemi])
        cortex = plain[hemi] > 0
        assert np.array_equal(seven[hemi] > 0, cortex)
        noise = (seven[hemi] - plain[hemi])[cortex]
        assert noise.std() == pytest.approx(NOISE_SD_MM, rel=0.05)
        assert abs(noise.mean()) < 0.01
    # the two hemispheres do not share their noise
    both = (plain["lh"] > 0) & (plain["rh"] > 0)
    left, right = (seven[hemi] - plain[hemi] for hemi in HEMIS)
    assert abs(np.corrcoef(left[both], right[both])[0, 1]) < 0.05


def test_the_map_on_the_template_is_the_expected_thickness_of_its_directions(
    tmp_path: Path, template: SyntheticTemplate
) -> None:
    # The claim the resampling tests build on, checked with the crudest mapping there is:
    # each template vertex takes the value of the native vertex nearest to it on the
    # registered sphere. Barycentric interpolation has to do better than this.
    subject = light_subject(tmp_path, subdivisions=5)
    patch = cohort_patches()["medial"]
    add_tier_b(subject, template, warp=0.3, patch=patch)
    targets, _ = template_mesh(template.order)
    for hemi in HEMIS:
        sphere = fs.read_geometry(str(subject.path / "surf" / f"{hemi}.sphere.reg"))[0]
        thickness = fs.read_morph_data(str(subject.path / "surf" / f"{hemi}.thickness"))
        nearest = cKDTree(sphere / 100.0).query(targets)[1]
        resampled = thickness[nearest]
        expected = expected_thickness(targets, patch)
        # away from the two edges where the field jumps: the patch rim and the medial strip
        wider = ThickPatch(patch.centre, patch.radius_deg + 4.0, patch.value_mm)
        narrower = ThickPatch(patch.centre, patch.radius_deg - 4.0, patch.value_mm)
        smooth = ~in_patch(targets, wider) & (
            medial_coordinate(targets, hemi) < MEDIAL_WALL_M - 0.1
        )
        assert smooth.mean() > 0.8
        assert np.abs(resampled - expected)[smooth].max() < 0.02
        inside = in_patch(targets, narrower)
        assert inside.sum() > 20 and (resampled[inside] == 5.0).all()
        strip = medial_coordinate(targets, hemi) > MEDIAL_WALL_M + 0.05
        assert strip.any() and (resampled[strip] == 0).all()
        assert np.array_equal(
            template.masks[hemi]["medial_wall"], medial_coordinate(targets, hemi) > MEDIAL_WALL_M
        )


def test_tier_b_bytes_depend_on_the_arguments_only(
    tmp_path: Path, template: SyntheticTemplate
) -> None:
    digests = []
    for folder in ("one", "two"):
        subject = light_subject(tmp_path / folder)
        add_tier_b(subject, template, noise_seed=3, patch=cohort_patches()["superior"])
        add_tier_a(subject)
        digests.append(tree_digest(subject.path))
    assert digests[0] == digests[1]
    assert set(digests[0]) >= {
        "surf/lh.sphere.reg",
        "surf/rh.thickness",
        "label/lh.aparc.annot",
        "stats/aseg.stats",
        "scripts/recon-all-rh.log",
    }


def test_add_tier_b_refuses_another_template_and_another_mesh(
    tmp_path: Path, template: SyntheticTemplate
) -> None:
    subject = light_subject(tmp_path)
    with pytest.raises(TypeError, match="write_synthetic_template"):
        add_tier_b(subject, template.path)  # pyrefly: ignore[bad-argument-type]
    odd = SyntheticSubject(**{**subject.__dict__, "n_vertices": 643})
    with pytest.raises(ValueError, match="not an icosphere"):
        subject_directions(odd)


# ---------------------------------------------------------------------------------------
# Tier A


def test_fs6_fmriprep_logs_and_stamp_parse_as_the_real_ones_do(
    tmp_path: Path, template: SyntheticTemplate
) -> None:
    subject = light_subject(tmp_path, name="sub-07")
    add_tier_b(subject, template)
    add_tier_a(subject)
    files = resolve_subject_files(subject.path)
    assert set(TIER_A) | set(TIER_B) | set(RENDER_ROLES) <= set(files)
    found = provenance(tmp_path, "sub-07")
    assert found.build_stamp == "freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.1-f53a55a"
    assert (found.fs_version, found.fs_major) == ("6.0.1", 6)
    assert found.n_invocations == 8  # 4 + 2 + 2, as for ds000102 sub-01
    assert found.finished is True
    assert found.holes == (3, 5)
    assert found.truncations == {"lh": (12, 140), "rh": (30, 201)}
    assert (found.t2pial, found.flairpial, found.hires, found.expert, found.edited) == (False,) * 5
    assert "-autorecon-hemi" in found.flags and "-autorecon1" in found.flags
    assert not any("/" in flag or "ds999999" in flag for flag in found.flags)
    assert found.pial_variant == "pial" and found.participant == "sub-07"


def test_fs7_fmriprep_logs(tmp_path: Path) -> None:
    subject = light_subject(tmp_path, name="sub-08")
    add_tier_a(
        subject,
        log_style="fs7-fmriprep",
        fs_version="7.3.2",
        holes=(4, 15),
        truncations={"lh": (22, 230)},
    )
    found = provenance(tmp_path, "sub-08")
    assert found.build_stamp == "freesurfer-linux-ubuntu22_x86_64-7.3.2-20220804-6354275"
    assert (found.fs_version, found.fs_major) == ("7.3.2", 7)
    assert found.n_invocations == 9  # 5 + 2 + 2, as for ds003020 sub-UTS01
    assert found.finished is True and found.holes == (4, 15)
    assert found.truncations == {"lh": (22, 230), "rh": None}  # left out: no line
    assert "-noT2pial" in found.flags and "-noFLAIRpial" in found.flags
    assert (found.t2pial, found.flairpial) == (False, False)
    log = (subject.path / "scripts" / "recon-all.log").read_text()
    assert log.count("#New# invocation of recon-all") == 4
    assert "/opt/freesurfer/bin/recon-all -autorecon1 " in log  # arguments after the script


def test_plain_log_and_flags(tmp_path: Path) -> None:
    subject = light_subject(tmp_path, name="sub-09")
    add_tier_a(subject, log_style="fs7-fmriprep", fs_version="7.3.2")
    add_tier_a(subject, log_style="plain", flags=("-T2pial", "-hires"), holes=(21, 12))
    scripts = subject.path / "scripts"
    # the per-hemisphere logs of the earlier style are gone
    assert sorted(path.name for path in scripts.iterdir()) == ["build-stamp.txt", "recon-all.log"]
    found = provenance(tmp_path, "sub-09")
    assert found.n_invocations == 1 and found.finished is True
    assert (found.fs_version, found.holes) == ("6.0.1", (21, 12))
    assert found.truncations == {"lh": (12, 140), "rh": (30, 201)}  # both in recon-all.log
    assert (found.t2pial, found.hires, found.flairpial) == (True, True, False)
    assert {"-T2pial", "-hires", "-all"} <= set(found.flags)


def test_flags_are_added_to_the_first_invocation_of_the_fmriprep_styles(tmp_path: Path) -> None:
    subject = light_subject(tmp_path, name="sub-10")
    add_tier_a(
        subject, log_style="fs7-fmriprep", fs_version="7.3.2", flags=("-FLAIRpial", "-expert")
    )
    found = provenance(tmp_path, "sub-10")
    # -FLAIRpial after -noFLAIRpial on one command line: the last one decides
    assert (found.t2pial, found.flairpial, found.expert) == (False, True, True)


@pytest.mark.parametrize(("fs_version", "has_volumes"), [("6.0.1", True), ("7.3.2", False)])
def test_stats_files_have_the_real_layout_and_the_canaries(
    tmp_path: Path, fs_version: str, has_volumes: bool
) -> None:
    subject = light_subject(tmp_path, name="sub-11")
    add_tier_a(subject, fs_version=fs_version, holes=(41, 26))
    for hemi in HEMIS:
        text = (subject.path / "stats" / f"{hemi}.aparc.stats").read_text()
        assert "# SUBJECTS_DIR /data/ds999999-fmriprep/sourcedata/freesurfer\n" in text
        assert "# hostname canary-host\n" in text and "# user     canary-user\n" in text
        assert f"# hemi {hemi}\n" in text and "# subjectname sub-11\n" in text
        # FreeSurfer 6 writes one measure line without the comma after its name
        malformed = (
            "# Measure Cortex, CortexVol Total cortical gray matter volume, 484608.441133, mm^3"
        )
        assert (malformed in text) == has_volumes
        assert ("-noglobal" in text) == (not has_volumes)
        assert (f"# cvs_version {fs_version}\n" in text) == (not has_volumes)
        header = (
            "# ColHeaders StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv "
            "GausCurv FoldInd CurvInd"
        )
        assert header in text and "# NTableCols 10\n" in text
        # without Tier B: every Desikan parcel but unknown and corpuscallosum, by rule
        rows = stats_rows(subject.path / "stats" / f"{hemi}.aparc.stats")
        assert list(rows) == [n for n in DESIKAN_NAMES if n not in ("unknown", "corpuscallosum")]
        assert len(rows) == 34 and all(row[3] == 2.5 for row in rows.values())
        assert occipital_control_ratio(subject.path) == pytest.approx(1.0)
    aseg = (subject.path / "stats" / "aseg.stats").read_text()
    assert ", 41, unitless" in aseg and ", 26, unitless" in aseg and ", 67, unitless" in aseg
    for measure in ("lhSurfaceHoles", "rhSurfaceHoles", "SurfaceHoles", "eTIV"):
        assert f", {measure}, " in aseg
    for canary in CANARIES:
        assert canary in aseg
    logs = "".join(path.read_text() for path in (subject.path / "scripts").glob("*.log"))
    assert all(canary in logs for canary in CANARIES[:2])


def test_stats_follow_the_thickness_when_tier_b_is_there(
    tmp_path: Path, template: SyntheticTemplate
) -> None:
    subject = light_subject(tmp_path, name="sub-12", subdivisions=4)
    patch = cohort_patches()["medial"]
    add_tier_b(subject, template, patch=patch, noise_seed=1)
    add_tier_a(subject)
    for hemi in HEMIS:
        thickness = fs.read_morph_data(str(subject.path / "surf" / f"{hemi}.thickness"))
        labels, _, _ = fs.read_annot(str(subject.path / "label" / f"{hemi}.aparc.annot"))
        rows = stats_rows(subject.path / "stats" / f"{hemi}.aparc.stats")
        present = [DESIKAN_NAMES[index] for index in np.unique(labels[labels >= 0])]
        assert list(rows) == present and len(rows) == 11
        for name, (num_vert, surf_area, gray_vol, thick_avg, thick_std) in rows.items():
            members = labels == DESIKAN_NAMES.index(name)
            assert num_vert == members.sum() and surf_area > 0
            assert thick_avg == pytest.approx(thickness[members].mean(), abs=5.1e-4)
            assert thick_std == pytest.approx(thickness[members].std(), abs=5.1e-4)
            assert gray_vol == round(surf_area * thick_avg)
        text = (subject.path / "stats" / f"{hemi}.aparc.stats").read_text()
        assert (
            f"# Measure Cortex, NumVert, Number of Vertices, {(labels >= 0).sum()}, unitless"
            in text
        )
    # the patch thickens occipital cortex, and the stats show it
    assert occipital_control_ratio(subject.path) > 1.1
    before = snapshot_tree(subject.path / "surf")
    add_tier_a(subject)
    assert snapshot_tree(subject.path / "surf") == before  # Tier A reads Tier B and leaves it


# ---------------------------------------------------------------------------------------
# cohorts


def test_synthetic_cohort_is_reproducible_and_varies_between_subjects(
    tmp_path: Path, template: SyntheticTemplate
) -> None:
    patch = cohort_patches()["medial"]
    first = synthetic_cohort(tmp_path / "a", 4, template, seed=11, bad={2: patch})
    synthetic_cohort(tmp_path / "b", 4, template, seed=11, bad={2: patch})
    synthetic_cohort(tmp_path / "c", 4, template, seed=12, bad={2: patch})
    assert [subject.path.name for subject in first] == ["sub-000", "sub-001", "sub-002", "sub-003"]
    assert all(subject.n_vertices == 642 for subject in first)
    assert tree_digest(tmp_path / "a") == tree_digest(tmp_path / "b")
    assert tree_digest(tmp_path / "a") != tree_digest(tmp_path / "c")
    spheres = [(s.path / "surf" / "lh.sphere.reg").read_bytes() for s in first]
    thickness = [(s.path / "surf" / "lh.thickness").read_bytes() for s in first]
    assert len(set(spheres)) == 4 and len(set(thickness)) == 4
    maxima = [fs.read_morph_data(str(s.path / "surf" / "lh.thickness")).max() for s in first]
    assert [value == 5.0 for value in maxima] == [False, False, True, False]
    assert max(value for value in maxima if value != 5.0) < 4.0


def test_cohort_patches_lie_in_the_search_region_of_the_left_hemisphere() -> None:
    patches = cohort_patches()
    directions, _ = template_mesh(5)
    regions = expected_regions(directions, "lh")
    parcel = parcel_of(directions, "lh")
    expected_parcel = {"medial": "pericalcarine", "inferior": "lingual", "superior": "cuneus"}
    expected_part = {"medial": "medial", "inferior": "inferior", "superior": "medial"}
    for name, patch in patches.items():
        assert patch.value_mm == 5.0 and patch.radius_deg == 16.0
        centre = np.array(patch.centre)
        assert DESIKAN_NAMES[int(parcel_of(centre, "lh"))] == expected_parcel[name]
        inside = in_patch(directions, patch)
        assert inside.any() and not (inside & ~regions["search"]).any()
        assert (inside & regions[expected_part[name]]).mean() > 0.5 * inside.mean()
        assert np.unique(parcel[inside]).size >= 1
    assert patches["medial"].centre == pytest.approx(direction("lh", from_pole_deg=20.0))
    assert cohort_patches(value_mm=4.0, radius_deg=10.0)["medial"].value_mm == 4.0


def test_the_cohort_of_the_session(synth_cohort: SyntheticCohort) -> None:
    names = [subject.path.name for subject in synth_cohort.subjects]
    assert names == [f"sub-{position:03d}" for position in range(32)]
    assert sorted(synth_cohort.bad) == ["sub-004", "sub-013", "sub-022"]
    assert synth_cohort.no_sphere_reg == ("sub-009",) and synth_cohort.no_stats == ("sub-027",)
    assert synth_cohort.template.order == 7
    assert find_adjacent_template(synth_cohort.root) == synth_cohort.root / "fsaverage"
    assert (synth_cohort.root / "fsaverage").resolve() == synth_cohort.template.path.resolve()
    for subject in synth_cohort.subjects:
        files = resolve_subject_files(subject.path)
        name = subject.path.name
        assert set(RENDER_ROLES) <= set(files)
        assert (FileRole.LH_SPHERE_REG in files) == (name not in synth_cohort.no_sphere_reg)
        assert (FileRole.RH_SPHERE_REG in files) == (name not in synth_cohort.no_sphere_reg)
        assert (FileRole.LH_STATS in files) == (name not in synth_cohort.no_stats)
        assert (FileRole.ASEG_STATS in files) == (name not in synth_cohort.no_stats)
        assert {FileRole.LH_THICKNESS, FileRole.RH_ANNOT, FileRole.RECON_LOG_LH} <= set(files)


def test_the_bad_subjects_of_the_cohort_stand_out_on_the_raw_numbers(
    synth_cohort: SyntheticCohort,
) -> None:
    # what the composite score has to find: only the three bad subjects have vertices at
    # the ceiling, and their occipital-to-control ratio is the highest of the cohort
    ceiling, ratios = {}, {}
    for subject in synth_cohort.subjects:
        name = subject.path.name
        at_ceiling = 0
        for hemi in HEMIS:
            thickness = fs.read_morph_data(str(subject.path / "surf" / f"{hemi}.thickness"))
            assert thickness.max() <= 5.0
            at_ceiling += int((thickness >= 4.9).sum())
            if name not in synth_cohort.bad:
                assert thickness.max() < 4.5
        ceiling[name] = at_ceiling
        if name not in synth_cohort.no_stats:
            ratios[name] = occipital_control_ratio(subject.path)
    assert {name for name, count in ceiling.items() if count > 0} == set(synth_cohort.bad)
    assert all(ceiling[name] >= 10 for name in synth_cohort.bad)
    ranked = sorted(ratios, key=lambda name: ratios[name], reverse=True)
    assert set(ranked[:3]) == set(synth_cohort.bad)
    assert ratios[ranked[2]] - ratios[ranked[3]] > 0.05
    good = [ratio for name, ratio in ratios.items() if name not in synth_cohort.bad]
    assert len(good) == 28 and max(good) - min(good) < 0.05


def test_the_cohort_inventories_and_renders_with_the_code_of_milestone_2(
    synth_cohort: SyntheticCohort, tmp_path: Path
) -> None:
    before = snapshot_tree(synth_cohort.root)
    run = Run(tmp_path / "run", source=local(synth_cohort.root))
    report = run.inventory()
    assert (report.done, report.failed) == (32, 0)
    subjects = run.public().subjects()
    assert len(subjects) == 32 and "fsaverage" not in set(subjects["subject_label"])
    assert int(subjects["complete"].sum()) == 32
    by_label = subjects.set_index("subject_label")
    assert sorted(by_label.index[~by_label["tier_a"].astype(bool)]) == ["sub-027"]
    assert sorted(by_label.index[~by_label["tier_b"].astype(bool)]) == ["sub-009"]
    assert set(subjects["fs_version"]) == {"6.0.1"}
    assert set(subjects["n_invocations"]) == {8}
    assert subjects["holes_lh"].nunique() > 5 and subjects["trunc_rh_2"].nunique() > 5

    bad = synth_cohort.subjects[4]
    info = render_subject(
        resolve_subject_files(bad.path), tmp_path / "media", RenderConfig(panel_px=64), video=False
    )
    assert info.vertex_map is not None and info.vertex_map.n_lh > 0
    assert snapshot_tree(synth_cohort.root) == before


def test_the_small_cohort(small_cohort: SyntheticCohort, synth_cohort: SyntheticCohort) -> None:
    assert [subject.path.name for subject in small_cohort.subjects] == [
        f"sub-{position:03d}" for position in range(5)
    ]
    assert sorted(small_cohort.bad) == ["sub-002"]
    assert small_cohort.no_sphere_reg == () and small_cohort.no_stats == ()
    assert small_cohort.root != synth_cohort.root
    assert find_adjacent_template(small_cohort.root) is not None
    assert occipital_control_ratio(small_cohort.subjects[2].path) > 1.1
    # another seed: not the first five subjects of the large cohort
    assert (small_cohort.subjects[0].path / "surf" / "lh.thickness").read_bytes() != (
        synth_cohort.subjects[0].path / "surf" / "lh.thickness"
    ).read_bytes()
