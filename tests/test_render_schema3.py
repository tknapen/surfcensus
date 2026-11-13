"""Render schema 3: the template in ``render.json``, fsaverage counterparts in the vertex map,
the template's occipital pole on the native surface (decision D12), and the update of a
render that writes two files (docs/rendering.md, docs/dev/contract.md section 6.11).

The synthetic subject is registered to the synthetic template by a warp that is known in
closed form (``tests/synth_tierb.py``), so every counterpart and the pole can be worked out
here without the package's sphere mapping. The tests at the end read the real subject and
the real ``fsaverage`` where the network tests have put them, and skip otherwise.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import nibabel.freesurfer as fs
import numpy as np
import pytest
from pydantic import ValidationError
from scipy.spatial import cKDTree
from synth import (  # pyrefly: ignore[missing-import]
    PIAL_SEMI_AXES,
    SyntheticSubject,
    icosphere,
    snapshot_tree,
    write_synthetic_subject,
)
from synth_template import (  # pyrefly: ignore[missing-import]
    SyntheticTemplate,
    template_mesh,
    write_synthetic_template,
)
from synth_tierb import (  # pyrefly: ignore[missing-import]
    add_tier_b,
    subject_directions,
    unwarp_directions,
    warp_directions,
)

from surfcensus import privacy
from surfcensus.render import OutputInSourceError, api, render_subject, three_plane
from surfcensus.render.model import (
    RENDER_SCHEMA_VERSION,
    FsaveragePole,
    RenderConfig,
    RenderInfo,
    TemplateInfo,
)
from surfcensus.render.vertexmap import KEYS, fill_fsaverage, read_vertex_map
from surfcensus.sources.base import HEMIS, TIER_C, FileRole
from surfcensus.sources.patterns import resolve_subject_files
from surfcensus.templates import Template, TemplateError, TemplateNotFound, build_template

CONFIG = RenderConfig(panel_px=64)  # the default box, which shows both hemispheres
WARP = 0.15
OLD_NS = 1_000_000_000_000_000_000  # an mtime in 2001
TWO_FILES = {"render.json", "vertex_map.npz"}
NOT_FOUND = "no fsaverage template found: click marks will not map to fsaverage"


def tree(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def age(root: Path) -> None:
    for p in root.rglob("*"):
        os.utime(p, ns=(OLD_NS, OLD_NS))


def rewritten(root: Path) -> set[str]:
    """The files under ``root`` that were written since :func:`age`."""
    return {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.stat().st_mtime_ns != OLD_NS
    }


def nearest_rows(points: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """For each point the nearest target, the lowest index among equally near ones. Every
    distance is computed; nothing here is shared with the package's sphere mapping."""
    out = np.empty(len(points), dtype=np.int64)
    for start in range(0, len(points), 128):
        chunk = points[start : start + 128]
        d2 = ((chunk[:, None, :] - targets[None, :, :]) ** 2).sum(axis=2)
        out[start : start + 128] = d2.argmin(axis=1)
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_key(info: RenderInfo) -> str:
    """The key a run stores with a render (``render/batch.py``): the tile settings and the
    digests of the files the tiles are made from."""
    digests = {r.value: info.inputs[r.value].sha256 for r in TIER_C if r.value in info.inputs}
    return privacy.render_key(b"k" * 32, info.config.tile_hash(), digests)


def sphere_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and r.getMessage().startswith("no fsaverage counterparts")
    ]


@pytest.fixture(autouse=True)
def _no_freesurfer_install(monkeypatch: pytest.MonkeyPatch) -> None:
    """``three_plane`` looks for a template in ``$FREESURFER_HOME`` first."""
    monkeypatch.delenv("FREESURFER_HOME", raising=False)


@dataclass(frozen=True)
class World:
    """One synthetic subject of 2,562 vertices per surface with Tier B files, two templates
    built from the session's ``fsaverage`` directory, and two renders of the subject. All
    of it is read-only; a test that writes copies what it needs."""

    subject: SyntheticSubject
    templates: dict[str, Template]
    plain_dir: Path
    plain: RenderInfo
    """Rendered without a template."""
    mapped_dir: Path
    mapped: RenderInfo
    """Rendered with the template on fsaverage4."""

    @property
    def files(self) -> dict[FileRole, Path]:
        return resolve_subject_files(self.subject.path)

    @property
    def template(self) -> Template:
        return self.templates["fsaverage4"]


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory, synth_template: SyntheticTemplate) -> World:
    root = tmp_path_factory.mktemp("schema3")
    subject = write_synthetic_subject(root / "subjects", subdivisions=4)
    add_tier_b(subject, synth_template, warp=WARP)
    templates = {
        mesh: build_template(synth_template.path, "explicit", mesh=mesh)
        for mesh in ("fsaverage4", "fsaverage6")
    }
    files = resolve_subject_files(subject.path)
    plain = render_subject(files, root / "plain", CONFIG, video=False)
    mapped = render_subject(
        files, root / "mapped", CONFIG, template=templates["fsaverage4"], video=False
    )
    return World(subject, templates, root / "plain", plain, root / "mapped", mapped)


@pytest.fixture
def plain_copy(world: World, tmp_path: Path) -> Path:
    return Path(shutil.copytree(world.plain_dir, tmp_path / "media"))


@pytest.fixture
def mapped_copy(world: World, tmp_path: Path) -> Path:
    return Path(shutil.copytree(world.mapped_dir, tmp_path / "media"))


@pytest.fixture
def subject_copy(world: World, tmp_path: Path) -> Path:
    """The subject in a directory of its own, with no ``fsaverage`` next to it."""
    return Path(shutil.copytree(world.subject.path, tmp_path / "study" / "sub-copy"))


# ---------------------------------------------------------------- the model


def test_schema_3_changes_the_config_hash_and_not_the_tile_hash() -> None:
    """The schema number is hashed into ``config_hash``, so a render of schema 2 is out of
    date. A rating follows ``tile_hash``, which was ``5a10b062eba2`` for the default config
    under schema 2 as well (computed with the model of the commit before this one)."""
    assert RENDER_SCHEMA_VERSION == 3
    default = RenderConfig()
    payload = json.dumps(
        {"schema": 3, "config": default.model_dump(mode="json")},
        sort_keys=True,
        separators=(",", ":"),
    )
    assert default.config_hash() == hashlib.sha256(payload.encode()).hexdigest()[:12]
    assert default.config_hash() == "61154ffcd5ab" != "17f83a0c9b6c"  # schema 2 gave the latter
    assert default.tile_hash() == "5a10b062eba2"


def test_template_and_pole_are_frozen_and_closed() -> None:
    template = TemplateInfo(
        mesh="fsaverage6", template_id="0123456789ab", sphere_sha256={"lh": "a", "rh": "b"}
    )
    pole = FsaveragePole(template_vertex=21, native_vertex=3, tkr=(1.0, 2.0, 3.0), distance_mm=0.5)
    for model, field in ((template, "mesh"), (pole, "distance_mm")):
        with pytest.raises(ValidationError):
            setattr(model, field, getattr(model, field))
        with pytest.raises(ValidationError):
            type(model).model_validate({**model.model_dump(), "path": "/data/fsaverage"})
    with pytest.raises(ValidationError):
        TemplateInfo.model_validate({**template.model_dump(), "sphere_sha256": {"mid": "a"}})
    assert set(TemplateInfo.model_fields) == {"mesh", "template_id", "sphere_sha256"}
    assert set(FsaveragePole.model_fields) == {
        "template_vertex",
        "native_vertex",
        "tkr",
        "distance_mm",
    }


def test_render_info_round_trips_with_and_without_a_template(world: World) -> None:
    for info in (world.plain, world.mapped):
        text = info.to_json()
        back = RenderInfo.model_validate_json(text)
        assert back == info and back.to_json() == text
    doc = json.loads(world.plain.to_json())
    assert doc["schema_version"] == 3
    assert doc["template"] is None and doc["fsaverage_poles"] is None
    doc = json.loads(world.mapped.to_json())
    assert set(doc["template"]) == {"mesh", "template_id", "sphere_sha256"}
    assert set(doc["fsaverage_poles"]) == {"lh", "rh"}


@pytest.mark.parametrize(
    "change",
    [
        "poles without template",
        "mesh without template",
        "template without poles",
        "template with empty poles",
        "template without mesh",
        "another mesh",
    ],
)
def test_render_info_says_one_thing_about_the_template(world: World, change: str) -> None:
    """``template``, ``fsaverage_poles`` and ``vertex_map.fsaverage`` are three places that
    say whether the vertex map has counterparts; a ``render.json`` in which they disagree
    is not one."""
    doc = world.mapped.model_dump()
    RenderInfo.model_validate(doc)
    if change == "poles without template":
        doc["template"] = None
        doc["vertex_map"]["fsaverage"] = None
    elif change == "mesh without template":
        doc["template"] = None
        doc["fsaverage_poles"] = None
    elif change == "template without poles":
        doc["fsaverage_poles"] = None
    elif change == "template with empty poles":
        doc["fsaverage_poles"] = {}
    elif change == "template without mesh":
        doc["vertex_map"]["fsaverage"] = None
    else:
        doc["vertex_map"]["fsaverage"] = "fsaverage6"
    with pytest.raises(ValidationError):
        RenderInfo.model_validate(doc)


# ---------------------------------------------------------------- fill_fsaverage


def toy_map() -> dict[str, np.ndarray]:
    vmap: dict[str, np.ndarray] = {}
    for hemi, index in (("lh", [0, 2, 5]), ("rh", [1, 4])):
        k = len(index)
        vmap[f"{hemi}_index"] = np.array(index, dtype="<i4")
        vmap[f"{hemi}_pial"] = np.arange(3 * k, dtype="<f4").reshape(k, 3)
        vmap[f"{hemi}_white"] = np.arange(3 * k, dtype="<f4").reshape(k, 3) + 0.5
        vmap[f"{hemi}_fsavg6"] = np.full(k, -1, dtype="<i4")
        vmap[f"{hemi}_visible"] = np.ones(k, dtype=bool)
    return vmap


def test_fill_fsaverage_looks_each_vertex_up_by_its_index() -> None:
    vmap = toy_map()
    nearest = {
        "lh": np.array([10, 11, 12, 13, 14, 15], dtype=np.int32),
        "rh": np.array([20, 21, 22, 23, 24], dtype=np.int64),
    }
    filled = fill_fsaverage(vmap, nearest)
    assert set(filled) == set(KEYS)
    assert filled["lh_fsavg6"].tolist() == [10, 12, 15]
    assert filled["rh_fsavg6"].tolist() == [21, 24]
    for hemi in HEMIS:
        assert filled[f"{hemi}_fsavg6"].dtype == np.dtype("<i4")
        for field in ("index", "pial", "white", "visible"):
            assert filled[f"{hemi}_{field}"] is vmap[f"{hemi}_{field}"]
        # the map that was passed is as it was
        assert vmap[f"{hemi}_fsavg6"].tolist() == [-1] * len(vmap[f"{hemi}_index"])


def test_fill_fsaverage_resets_a_hemisphere_without_a_mapping() -> None:
    """The result depends on the indices and the mappings alone, not on what was filled
    before: a map filled for one template and then for none is the unfilled map."""
    nearest = {"lh": np.arange(6) + 100, "rh": np.arange(5) + 200}
    once = fill_fsaverage(toy_map(), nearest)
    left = fill_fsaverage(once, {"lh": nearest["lh"]})
    assert left["lh_fsavg6"].tolist() == [100, 102, 105]
    assert left["rh_fsavg6"].tolist() == [-1, -1] and left["rh_fsavg6"].dtype == np.dtype("<i4")
    none = fill_fsaverage(once, {})
    for key in KEYS:
        np.testing.assert_array_equal(none[key], toy_map()[key])


def test_fill_fsaverage_refuses_a_mapping_that_does_not_reach_the_vertices() -> None:
    vmap = toy_map()
    with pytest.raises(ValueError, match=r"lh mapping covers 5 vertices.*names vertices 0 to 5"):
        fill_fsaverage(vmap, {"lh": np.arange(5)})
    vmap["rh_index"] = np.array([-1, 4], dtype="<i4")
    with pytest.raises(ValueError, match=r"rh mapping covers 5 vertices.*vertices -1 to 4"):
        fill_fsaverage(vmap, {"rh": np.arange(5)})
    empty = {key: value[:0] for key, value in toy_map().items()}
    filled = fill_fsaverage(empty, {"lh": np.arange(0), "rh": np.arange(3)})
    assert filled["lh_fsavg6"].shape == (0,) and filled["rh_fsavg6"].shape == (0,)


# ---------------------------------------------------------------- a render with a template


@pytest.mark.parametrize("mesh", ["fsaverage4", "fsaverage6"])
def test_counterparts_are_the_nearest_template_vertices_of_the_warped_directions(
    world: World, tmp_path: Path, mesh: str
) -> None:
    """A native vertex with direction ``d`` is registered to ``warp(d)``, so its counterpart
    is the template vertex nearest that direction."""
    if mesh == "fsaverage4":
        out = world.mapped_dir
    else:
        out = tmp_path / "media"
        render_subject(world.files, out, CONFIG, template=world.templates[mesh], video=False)
    info = RenderInfo.model_validate_json((out / "render.json").read_bytes())
    assert info.vertex_map is not None and info.vertex_map.fsaverage == mesh
    order = {"fsaverage4": 4, "fsaverage6": 6}[mesh]
    targets = template_mesh(order)[0]
    directions = subject_directions(world.subject)[0]
    registered = warp_directions(directions, WARP)
    vmap = read_vertex_map(out / "vertex_map.npz")
    # every vertex against the 2,562 targets of fsaverage4, every fifth against the 40,962
    step = 1 if mesh == "fsaverage4" else 5
    for hemi in HEMIS:
        index, found = vmap[f"{hemi}_index"], vmap[f"{hemi}_fsavg6"]
        assert len(index) > 900 and found.dtype == np.int32
        assert found.min() >= 0 and found.max() < len(targets)
        expected = nearest_rows(registered[index[::step]], targets)
        np.testing.assert_array_equal(found[::step], expected)
        # the warp decides: the nearest vertices of the directions as they are differ
        assert (nearest_rows(directions[index[::step]], targets) != expected).mean() > 0.5
    # the two hemispheres of the fixture share mesh and warp, and so their counterparts
    lh = dict(zip(vmap["lh_index"].tolist(), vmap["lh_fsavg6"].tolist(), strict=True))
    rh = dict(zip(vmap["rh_index"].tolist(), vmap["rh_fsavg6"].tolist(), strict=True))
    shared = set(lh) & set(rh)
    assert len(shared) > 500 and all(lh[v] == rh[v] for v in shared)


def test_the_fsaverage_pole_lies_where_the_warp_puts_it(
    world: World, synth_template: SyntheticTemplate
) -> None:
    """The template's pole is the vertex at ``(0, -1, 0)``. On the native surface it is the
    vertex whose registered direction is nearest that, which the inverse of the warp puts
    at ``unwarp((0, -1, 0))``, about 4.4 mm from the geometric pole."""
    info = world.mapped
    assert info.fsaverage_poles is not None and set(info.fsaverage_poles) == {"lh", "rh"}
    directions = subject_directions(world.subject)[0]
    registered = warp_directions(directions, WARP)
    u_pole = template_mesh(4)[0][synth_template.pole_vertex]
    np.testing.assert_array_equal(u_pole, [0.0, -1.0, 0.0])
    native = int(nearest_rows(u_pole[None, :], registered)[0])
    preimage = unwarp_directions(u_pole, WARP)
    for hemi, centre in (("lh", world.subject.lh_centre), ("rh", world.subject.rh_centre)):
        pole = info.fsaverage_poles[hemi]
        assert pole.template_vertex == synth_template.pole_vertex == 21
        assert pole.template_vertex == world.template.hemis[hemi].pole_vertex
        assert pole.native_vertex == native != info.poles[hemi].vertex
        pial = fs.read_geometry(str(world.subject.path / "surf" / f"{hemi}.pial"))[0]
        assert pole.tkr == tuple(float(x) for x in pial[native])
        geometric = pial[info.poles[hemi].vertex]
        assert info.poles[hemi].tkr == tuple(float(x) for x in geometric)
        distance = float(np.sqrt(((pial[native].astype(float) - geometric) ** 2).sum()))
        assert pole.distance_mm == pytest.approx(distance, abs=1e-12)
        # in closed form, up to the spacing of the 2,562-vertex mesh (edges of 2.2 to 4.7 mm)
        there = np.array(centre) + preimage * np.array(PIAL_SEMI_AXES)
        assert np.linalg.norm(pial[native] - there) < 2.4
        closed = float(np.linalg.norm((preimage - u_pole) * np.array(PIAL_SEMI_AXES)))
        assert 4.0 < closed < 5.0 and abs(pole.distance_mm - closed) < 2.4
    assert info.fsaverage_poles["lh"].distance_mm == pytest.approx(4.3549, abs=1e-4)


def test_without_a_warp_the_two_poles_are_one_vertex(
    synth_template: SyntheticTemplate, world: World, tmp_path: Path
) -> None:
    """The subject's mesh is the fsaverage4 mesh. Registered without a warp, every vertex
    is its own counterpart and the template's pole is the geometric pole."""
    subject = write_synthetic_subject(tmp_path / "subjects", subdivisions=4)
    add_tier_b(subject, synth_template, warp=0.0)
    info = render_subject(
        resolve_subject_files(subject.path),
        tmp_path / "media",
        CONFIG,
        template=world.template,
        video=False,
    )
    vmap = read_vertex_map(tmp_path / "media" / "vertex_map.npz")
    assert info.fsaverage_poles is not None
    for hemi in HEMIS:
        np.testing.assert_array_equal(vmap[f"{hemi}_fsavg6"], vmap[f"{hemi}_index"])
        pole = info.fsaverage_poles[hemi]
        assert pole.native_vertex == info.poles[hemi].vertex == 21
        assert pole.tkr == info.poles[hemi].tkr and pole.distance_mm == 0.0


def test_the_template_and_the_sphere_files_are_recorded(
    world: World, synth_template: SyntheticTemplate
) -> None:
    info, subject = world.mapped, world.subject
    assert (world.mapped_dir / "render.json").read_text() == info.to_json()
    assert info.schema_version == 3 and info.config_hash == CONFIG.config_hash()
    assert set(info.inputs) == set(world.plain.inputs) | {"lh_sphere_reg", "rh_sphere_reg"}
    for hemi in HEMIS:
        path = subject.path / "surf" / f"{hemi}.sphere.reg"
        used = info.inputs[f"{hemi}_sphere_reg"]
        assert (used.name, used.size, used.sha256) == (
            f"{hemi}.sphere.reg",
            path.stat().st_size,
            sha256(path),
        )
    for role, entry in world.plain.inputs.items():  # the inputs of the tiles are the same
        assert info.inputs[role] == entry
    assert info.template == TemplateInfo(
        mesh="fsaverage4",
        template_id=world.template.template_id,
        sphere_sha256={
            hemi: sha256(synth_template.path / "surf" / f"{hemi}.sphere.reg") for hemi in HEMIS
        },
    )
    assert info.vertex_map is not None and info.vertex_map.fsaverage == "fsaverage4"
    text = info.to_json()
    assert str(subject.path.parent) not in text and str(synth_template.path) not in text
    assert "/" not in "".join(entry.name for entry in info.inputs.values())


def test_without_a_template_nothing_is_mapped_and_the_tiles_are_the_same(world: World) -> None:
    """The subject has both ``sphere.reg`` files. Without a template they are not read, and
    with one the render differs in ``render.json`` and in the vertex map only."""
    plain = world.plain
    assert plain.template is None and plain.fsaverage_poles is None
    assert plain.vertex_map is not None and plain.vertex_map.fsaverage is None
    assert not any("sphere" in role for role in plain.inputs)
    without, with_template = tree(world.plain_dir), tree(world.mapped_dir)
    assert set(without) == set(with_template) and len(without) == 2 * 176 + 3
    assert {name for name in without if without[name] != with_template[name]} == TWO_FILES
    a, b = (
        read_vertex_map(world.plain_dir / "vertex_map.npz"),
        read_vertex_map(world.mapped_dir / "vertex_map.npz"),
    )
    for key in KEYS:
        if key.endswith("fsavg6"):
            assert set(a[key].tolist()) == {-1} and (b[key] >= 0).all()
        else:
            np.testing.assert_array_equal(a[key], b[key])
    one, other = plain.model_dump(), world.mapped.model_dump()
    assert {key for key in one if one[key] != other[key]} == {
        "inputs",
        "template",
        "fsaverage_poles",
        "vertex_map",
    }
    assert render_key(plain) == render_key(world.mapped)


def test_render_on_disk_does_not_ask_about_the_template(world: World) -> None:
    """A run decides from its own row whether the template of a render is current
    (``renders.input_signature``); the test on disk is about the files."""
    assert api.render_on_disk(world.plain_dir, CONFIG, video=False)
    assert api.render_on_disk(world.mapped_dir, CONFIG, video=False)
    assert not api.render_on_disk(world.mapped_dir, CONFIG, video=True)


# ---------------------------------------------------------------- the update of two files


def test_a_template_added_later_rewrites_two_files(
    world: World, plain_copy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_tiles(*args: object, **kwargs: object) -> None:
        raise AssertionError("a tile was drawn")

    monkeypatch.setattr(api.frames, "gray_tile", no_tiles)
    files = world.files
    before = render_key(world.plain)
    age(plain_copy)
    info = render_subject(files, plain_copy, CONFIG, template=world.template, video=False)
    assert info == world.mapped
    assert rewritten(plain_copy) == TWO_FILES
    assert tree(plain_copy) == tree(world.mapped_dir)
    assert render_key(info) == before

    # asked again, nothing is written
    age(plain_copy)
    again = render_subject(files, plain_copy, CONFIG, template=world.template, video=False)
    assert again == info and rewritten(plain_copy) == set()

    # another template: the same two files
    other = world.templates["fsaverage6"]
    info6 = render_subject(files, plain_copy, CONFIG, template=other, video=False)
    assert rewritten(plain_copy) == TWO_FILES
    assert info6.template is not None and info6.template.template_id == other.template_id
    assert info6.vertex_map is not None and info6.vertex_map.fsaverage == "fsaverage6"
    assert info6.template.template_id != world.template.template_id
    assert read_vertex_map(plain_copy / "vertex_map.npz")["lh_fsavg6"].max() > 2562
    assert render_key(info6) == before

    # and none: the render that was there at the start, byte for byte
    age(plain_copy)
    back = render_subject(files, plain_copy, CONFIG, video=False)
    assert back == world.plain and rewritten(plain_copy) == TWO_FILES
    assert tree(plain_copy) == tree(world.plain_dir)


def test_the_update_keeps_the_video_and_what_else_is_there(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "media"
    first = render_subject(world.files, out, CONFIG)
    assert first.video is not None and (out / "anim.mp4").is_file()
    (out / "notes.txt").write_text("kept")
    (out / ".vertex_map.npz.0123456789abcdef.tmp").write_text("of a writer that was killed")
    before = tree(out)
    age(out)
    monkeypatch.setattr(api.encode, "write_mp4", lambda *a, **k: pytest.fail("video encoded"))
    info = render_subject(world.files, out, CONFIG, template=world.template)
    assert rewritten(out) == TWO_FILES
    after = tree(out)
    assert {name for name in before if before[name] != after[name]} == TWO_FILES
    assert set(after) == set(before)
    assert info.video == first.video and info.template is not None
    assert info.model_dump(exclude={"inputs", "template", "fsaverage_poles", "vertex_map"}) == (
        first.model_dump(exclude={"inputs", "template", "fsaverage_poles", "vertex_map"})
    )


def test_force_renders_everything_again(world: World, mapped_copy: Path) -> None:
    before = tree(mapped_copy)
    age(mapped_copy)
    info = render_subject(
        world.files, mapped_copy, CONFIG, template=world.template, video=False, force=True
    )
    assert info == world.mapped
    assert rewritten(mapped_copy) == set(before) and tree(mapped_copy) == before


@pytest.mark.parametrize("changed", ["tile setting", "pial surface", "video"])
def test_a_change_of_the_tiles_is_a_full_render_also_with_a_template(
    world: World, mapped_copy: Path, tmp_path: Path, changed: str
) -> None:
    """Only the template and the ``sphere.reg`` files are left to the update. Anything the
    tiles are made from renders in full."""
    files, config, video = world.files, CONFIG, False
    if changed == "tile setting":
        config = CONFIG.model_copy(update={"pial_colour": "#00FF00"})
    elif changed == "video":
        video = True
    else:
        subject = Path(shutil.copytree(world.subject.path, tmp_path / "study" / "sub-copy"))
        vertices, faces = fs.read_geometry(str(subject / "surf" / "lh.pial"))
        fs.write_geometry(
            str(subject / "surf" / "lh.pial"), vertices + np.array([0.0, 0.0, 0.5]), faces
        )
        files = resolve_subject_files(subject)
    age(mapped_copy)
    info = render_subject(files, mapped_copy, config, template=world.template, video=video)
    assert info.template == world.mapped.template
    assert rewritten(mapped_copy) == set(tree(mapped_copy))
    assert (info.video is not None) == video


@pytest.mark.parametrize("old", ["as schema 2 wrote it", "version alone", "hash alone"])
@pytest.mark.parametrize("with_template", [False, True])
def test_a_render_of_schema_2_is_not_accepted_as_complete(
    world: World, plain_copy: Path, old: str, with_template: bool
) -> None:
    """A ``render.json`` of schema 2 has neither of the two new entries and loads under the
    present model all the same, since both have a default. The version number and the config
    hash of schema 2 are each a reason of its own to render again, in full: the update of
    two files is for a render of this schema."""
    doc = json.loads((plain_copy / "render.json").read_text())
    if old != "hash alone":
        doc["schema_version"] = 2
    if old != "version alone":  # the hash that schema 2 gave the very same config
        payload = json.dumps(
            {"schema": 2, "config": doc["config"]}, sort_keys=True, separators=(",", ":")
        )
        doc["config_hash"] = hashlib.sha256(payload.encode()).hexdigest()[:12]
        assert doc["config_hash"] != CONFIG.config_hash()
    if old == "as schema 2 wrote it":
        del doc["template"], doc["fsaverage_poles"]
    text = json.dumps(doc, indent=1, sort_keys=True) + "\n"
    (plain_copy / "render.json").write_text(text)
    assert RenderInfo.model_validate_json(text).inputs == world.plain.inputs
    assert not api.render_on_disk(plain_copy, CONFIG, video=False)
    age(plain_copy)
    template = world.template if with_template else None
    info = render_subject(world.files, plain_copy, CONFIG, template=template, video=False)
    assert info == (world.mapped if with_template else world.plain)
    assert rewritten(plain_copy) == set(tree(plain_copy))
    assert tree(plain_copy) == tree(world.mapped_dir if with_template else world.plain_dir)


@pytest.mark.parametrize(
    "damage",
    [
        "garbage",
        "of another render",
        "an array missing",
        "one vertex more",
        "other visible flags",
        "absent",
    ],
)
def test_a_vertex_map_that_is_not_the_renders_is_no_base_for_an_update(
    world: World, plain_copy: Path, tmp_path: Path, damage: str
) -> None:
    """The update fills the vertex map that lies in the directory. When that file is not
    the one ``render.json`` describes, the directory is rendered in full."""
    target = plain_copy / "vertex_map.npz"
    if damage == "garbage":
        target.write_bytes(b"not a zip archive")
    elif damage == "of another render":  # a whole vertex map, of a narrower box
        narrow = CONFIG.model_copy(update={"box_x_mm": (-30.0, 0.0)})
        render_subject(world.files, tmp_path / "narrow", narrow, video=False)
        shutil.copy(tmp_path / "narrow" / "vertex_map.npz", target)
    elif damage == "an array missing":
        vmap = read_vertex_map(target)
        with target.open("wb") as stream:
            np.savez(stream, **{key: a for key, a in vmap.items() if key != "rh_visible"})
    elif damage == "one vertex more":  # in the indices alone; the flags are those of the render
        vmap = read_vertex_map(target)
        vmap["rh_index"] = np.append(vmap["rh_index"], 2561).astype("<i4")
        with target.open("wb") as stream:
            np.savez(stream, **vmap)
    elif damage == "other visible flags":  # every array of the length that render.json names
        vmap = read_vertex_map(target)
        vmap["lh_visible"] = ~vmap["lh_visible"]
        assert vmap["lh_visible"].sum() != world.plain.vertex_map.n_lh_visible
        with target.open("wb") as stream:
            np.savez(stream, **vmap)
    else:
        target.unlink()
    age(plain_copy)
    info = render_subject(world.files, plain_copy, CONFIG, template=world.template, video=False)
    assert info == world.mapped
    assert rewritten(plain_copy) == set(tree(plain_copy))
    assert tree(plain_copy) == tree(world.mapped_dir)


def test_an_interrupted_update_leaves_an_incomplete_render(
    world: World, plain_copy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``render.json`` goes first and comes last, as in a full render: an update that dies
    in between leaves a directory that the next call renders, not a ``render.json`` that
    describes another vertex map than the one beside it."""

    def fail(*args: object) -> None:
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(api.vertexmap, "write_vertex_map", fail)
        with pytest.raises(OSError, match="disk full"):
            render_subject(world.files, plain_copy, CONFIG, template=world.template, video=False)
    assert not (plain_copy / "render.json").exists()
    assert set(tree(plain_copy)) == set(tree(world.plain_dir)) - {"render.json"}
    info = render_subject(world.files, plain_copy, CONFIG, template=world.template, video=False)
    assert info == world.mapped and tree(plain_copy) == tree(world.mapped_dir)


def test_the_update_is_written_while_the_directory_is_held(
    world: World, plain_copy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two renders into one directory run one after the other, and the update is one."""
    state = {"held": False, "written": 0}
    exclusive, write = api._exclusive, api.vertexmap.write_vertex_map

    class Held:
        def __init__(self, out_dir: Path) -> None:
            self.inner = exclusive(out_dir)

        def __enter__(self) -> bool:
            locked = self.inner.__enter__()
            state["held"] = True
            return locked

        def __exit__(self, *exc: object) -> None:
            state["held"] = False
            self.inner.__exit__(None, None, None)

    def held_write(path: Path, vmap: dict[str, np.ndarray]) -> None:
        assert state["held"], "the vertex map was written without the lock"
        assert not (plain_copy / "render.json").exists()
        state["written"] += 1
        write(path, vmap)

    monkeypatch.setattr(api, "_exclusive", Held)
    monkeypatch.setattr(api.vertexmap, "write_vertex_map", held_write)
    render_subject(world.files, plain_copy, CONFIG, template=world.template, video=False)
    assert state == {"held": False, "written": 1}


def test_a_render_that_another_one_finished_meanwhile_is_not_written_twice(
    world: World, plain_copy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """While this call waits for the directory, another one brings it up to date. Under the
    lock the directory is looked at again, and nothing is written."""
    exclusive = api._exclusive

    class Waited:
        def __init__(self, out_dir: Path) -> None:
            self.inner = exclusive(out_dir)

        def __enter__(self) -> bool:
            # what the render that held the directory left behind
            shutil.copy(world.mapped_dir / "vertex_map.npz", plain_copy / "vertex_map.npz")
            shutil.copy(world.mapped_dir / "render.json", plain_copy / "render.json")
            age(plain_copy)
            return self.inner.__enter__()

        def __exit__(self, *exc: object) -> None:
            self.inner.__exit__(None, None, None)

    monkeypatch.setattr(api, "_exclusive", Waited)
    info = render_subject(world.files, plain_copy, CONFIG, template=world.template, video=False)
    assert info == world.mapped and rewritten(plain_copy) == set()


# ---------------------------------------------------------------- sphere.reg problems


def break_sphere(subject: Path, hemi: str, problem: str) -> None:
    path = subject / "surf" / f"{hemi}.sphere.reg"
    if problem == "another vertex count":
        vertices, faces = icosphere(3)
        fs.write_geometry(str(path), vertices * 100.0, faces)
    elif problem == "no sphere":  # an ellipsoid: radii from 60 to 140
        vertices, faces = fs.read_geometry(str(path))
        fs.write_geometry(str(path), vertices * [0.6, 1.0, 1.4], faces)
    elif problem == "at the origin":
        vertices, faces = fs.read_geometry(str(path))
        fs.write_geometry(str(path), vertices * 0.0, faces)
    elif problem == "faces out of range":
        vertices, faces = fs.read_geometry(str(path))
        fs.write_geometry(str(path), vertices, faces + 1)
    elif problem == "garbage":
        path.write_bytes(b"this is no surface file\n" * 40)
    elif problem == "empty":
        path.write_bytes(b"")
    elif problem == "absent":
        path.unlink()
    elif problem == "dangling link":
        path.unlink()
        path.symlink_to(subject / "surf" / "nowhere.sphere.reg")
    else:
        raise AssertionError(problem)


SPHERE_PROBLEMS = {
    "another vertex count": "rh.sphere.reg has 642 vertices and the pial surface 2562",
    "no sphere": "rh.sphere.reg: degenerate sphere",
    "at the origin": "rh.sphere.reg: degenerate sphere",
    "faces out of range": "rh.sphere.reg: degenerate sphere",
    "garbage": "unreadable input (rh.sphere.reg: ",
    "empty": "unreadable input (rh.sphere.reg: ",
    "absent": "no rh.sphere.reg",
    "dangling link": "no rh.sphere.reg",
}


@pytest.mark.parametrize("problem", list(SPHERE_PROBLEMS))
def test_a_sphere_that_cannot_be_used_leaves_its_hemisphere_unmapped(
    world: World,
    subject_copy: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    problem: str,
) -> None:
    """Not a failure of the render: the tiles are the same, the other hemisphere is mapped,
    and a warning says why this one is not."""
    break_sphere(subject_copy, "rh", problem)
    files = resolve_subject_files(subject_copy)
    out = tmp_path / "media"
    with caplog.at_level(logging.WARNING, logger="surfcensus"):
        info = render_subject(files, out, CONFIG, template=world.template, video=False)
    messages = sphere_warnings(caplog)
    assert len(messages) == 1, messages
    assert messages[0].startswith("no fsaverage counterparts for rh: " + SPHERE_PROBLEMS[problem])
    assert str(tmp_path) not in messages[0]

    assert info.fsaverage_poles == {"lh": world.mapped.fsaverage_poles["lh"]}
    assert info.template == world.mapped.template
    assert info.vertex_map is not None and info.vertex_map.fsaverage == "fsaverage4"
    vmap, full = (
        read_vertex_map(out / "vertex_map.npz"),
        read_vertex_map(world.mapped_dir / "vertex_map.npz"),
    )
    np.testing.assert_array_equal(vmap["lh_fsavg6"], full["lh_fsavg6"])
    assert set(vmap["rh_fsavg6"].tolist()) == {-1} and len(vmap["rh_fsavg6"]) > 900
    # a file that was read is an input, whether or not it could be used
    read = problem not in ("absent", "dangling link")
    assert ("rh_sphere_reg" in info.inputs) == read and "lh_sphere_reg" in info.inputs
    if read:
        assert info.inputs["rh_sphere_reg"].sha256 == sha256(subject_copy / "surf/rh.sphere.reg")
    written, reference = tree(out), tree(world.mapped_dir)
    assert {name for name in written if written[name] != reference[name]} == TWO_FILES

    # asked again, with this template or with another, the render is complete as it is
    age(out)
    caplog.clear()
    assert render_subject(files, out, CONFIG, template=world.template, video=False) == info
    assert rewritten(out) == set() and sphere_warnings(caplog) == []


@pytest.mark.parametrize("problem", ["another vertex count", "garbage", "absent"])
def test_with_no_usable_sphere_the_render_names_no_template(
    world: World,
    subject_copy: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    problem: str,
) -> None:
    for hemi in HEMIS:
        break_sphere(subject_copy, hemi, problem)
    files = resolve_subject_files(subject_copy)
    out = tmp_path / "media"
    with caplog.at_level(logging.WARNING, logger="surfcensus"):
        info = render_subject(files, out, CONFIG, template=world.template, video=False)
    assert [m.split(":")[0] for m in sphere_warnings(caplog)] == [
        "no fsaverage counterparts for lh",
        "no fsaverage counterparts for rh",
    ]
    assert info.template is None and info.fsaverage_poles is None
    assert info.vertex_map is not None and info.vertex_map.fsaverage is None
    vmap = read_vertex_map(out / "vertex_map.npz")
    assert set(vmap["lh_fsavg6"].tolist()) == set(vmap["rh_fsavg6"].tolist()) == {-1}
    spheres = {role for role in info.inputs if "sphere" in role}
    assert spheres == (set() if problem == "absent" else {"lh_sphere_reg", "rh_sphere_reg"})
    written, plain = tree(out), tree(world.plain_dir)
    differing = {name for name in written if written[name] != plain[name]}
    assert differing == (set() if problem == "absent" else {"render.json"})

    # that no hemisphere can be mapped does not depend on the template: with another one
    # the render is complete as it is, and nothing is read again
    age(out)
    caplog.clear()
    other = world.templates["fsaverage6"]
    assert render_subject(files, out, CONFIG, template=other, video=False) == info
    assert rewritten(out) == set() and sphere_warnings(caplog) == []

    # the file comes back: the two files are written
    shutil.copy(world.subject.path / "surf" / "lh.sphere.reg", subject_copy / "surf")
    files = resolve_subject_files(subject_copy)
    fixed = render_subject(files, out, CONFIG, template=world.template, video=False)
    assert rewritten(out) == TWO_FILES
    assert fixed.fsaverage_poles == {"lh": world.mapped.fsaverage_poles["lh"]}
    assert fixed.template == world.mapped.template


def test_a_sphere_that_cannot_be_opened_is_left_out_with_its_reason(
    world: World,
    subject_copy: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A file that may not be read fails when its digest is taken. For a file the tiles are
    made from that is a failure of the render; for a ``sphere.reg`` it is one hemisphere
    without counterparts."""
    digest = api.sha256_file

    def denied(path: Path) -> str:
        if Path(path).name == "lh.sphere.reg":
            raise PermissionError(13, "Permission denied", str(path))
        return digest(path)

    monkeypatch.setattr(api, "sha256_file", denied)
    files = resolve_subject_files(subject_copy)
    with caplog.at_level(logging.WARNING, logger="surfcensus"):
        info = render_subject(
            files, tmp_path / "media", CONFIG, template=world.template, video=False
        )
    assert sphere_warnings(caplog) == [
        "no fsaverage counterparts for lh: unreadable input (lh.sphere.reg: Permission denied)"
    ]
    assert info.fsaverage_poles is not None and set(info.fsaverage_poles) == {"rh"}
    assert "lh_sphere_reg" not in info.inputs and "rh_sphere_reg" in info.inputs
    # without a template the file is not looked at
    caplog.clear()
    render_subject(files, tmp_path / "plain", CONFIG, video=False)
    assert sphere_warnings(caplog) == []


def test_sphere_files_are_sources_like_the_others(world: World, tmp_path: Path) -> None:
    """A ``sphere.reg`` handed over from a directory of its own: nothing is rendered into
    that directory when the file is read, and it is no source when it is not."""
    elsewhere = tmp_path / "cache"
    elsewhere.mkdir()
    files = dict(world.files)
    files[FileRole.LH_SPHERE_REG] = Path(
        shutil.copy(files[FileRole.LH_SPHERE_REG], elsewhere / "0123abcd")
    )
    before = snapshot_tree(elsewhere)
    with pytest.raises(OutputInSourceError):
        render_subject(files, elsewhere / "media", CONFIG, template=world.template, video=False)
    assert snapshot_tree(elsewhere) == before
    info = render_subject(files, tmp_path / "out", CONFIG, template=world.template, video=False)
    # recorded under the name FreeSurfer gives the file, not the one it was stored under
    assert info.inputs["lh_sphere_reg"].name == "lh.sphere.reg"
    assert info.fsaverage_poles == world.mapped.fsaverage_poles
    render_subject(files, elsewhere / "media", CONFIG, video=False)
    assert (elsewhere / "media" / "render.json").is_file()


# ---------------------------------------------------------------- three_plane


@pytest.fixture
def study(world: World, synth_template: SyntheticTemplate, tmp_path: Path) -> Path:
    """A subjects directory with the subject and the template next to it, as fMRIPrep
    leaves them. The template is the session's, through a link."""
    root = tmp_path / "study"
    shutil.copytree(world.subject.path, root / "sub-01")
    (root / "fsaverage").symlink_to(synth_template.path, target_is_directory=True)
    return root


def not_found_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and "template" in r.getMessage()
    ]


def test_three_plane_finds_the_template_next_to_the_subject(
    world: World, study: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    default = world.templates["fsaverage6"]
    with caplog.at_level(logging.WARNING, logger="surfcensus"):
        info = three_plane(study / "sub-01", tmp_path / "media", CONFIG, video=False)
    assert caplog.records == []
    assert info.template is not None
    assert (info.template.mesh, info.template.template_id) == ("fsaverage6", default.template_id)
    assert info.fsaverage_poles is not None and set(info.fsaverage_poles) == {"lh", "rh"}
    vmap = read_vertex_map(tmp_path / "media" / "vertex_map.npz")
    assert (vmap["lh_fsavg6"] >= 0).all() and (vmap["rh_fsavg6"] >= 0).all()
    # the same render as with the template handed over
    same = render_subject(
        resolve_subject_files(study / "sub-01"),
        tmp_path / "direct",
        CONFIG,
        template=default,
        video=False,
    )
    assert same == info and tree(tmp_path / "direct") == tree(tmp_path / "media")


def test_three_plane_finds_the_template_from_inside_the_subject_directory(
    world: World, study: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(study / "sub-01")
    info = three_plane(Path("."), tmp_path / "media", CONFIG, video=False)
    assert info.template is not None
    assert info.template.template_id == world.templates["fsaverage6"].template_id
    monkeypatch.chdir(study)
    again = three_plane("sub-01", tmp_path / "media", CONFIG, video=False)
    assert again == info


def test_three_plane_prefers_a_freesurfer_install(
    world: World, study: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Decision D14: ``$FREESURFER_HOME/subjects/fsaverage`` comes before the directory
    next to the subjects. The installed one here has no average-area files, which makes it
    another template."""
    installed = write_synthetic_template(
        tmp_path / "freesurfer" / "subjects", order=6, with_avg_area=False
    )
    monkeypatch.setenv("FREESURFER_HOME", str(tmp_path / "freesurfer"))
    info = three_plane(study / "sub-01", tmp_path / "media", CONFIG, video=False)
    expected = build_template(installed.path, "freesurfer_home")
    assert info.template is not None and info.template.template_id == expected.template_id
    assert info.template.template_id != world.templates["fsaverage6"].template_id
    # and with no directory next to the subject it is the only one
    (study / "fsaverage").unlink()
    assert three_plane(study / "sub-01", tmp_path / "media", CONFIG, video=False) == info


def test_three_plane_without_any_template_renders_and_says_so(
    world: World, subject_copy: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="surfcensus"):
        info = three_plane(subject_copy, tmp_path / "media", CONFIG, video=False)
    (message,) = not_found_warnings(caplog)
    assert message.startswith(NOT_FOUND + ". Looked in ")
    assert str(subject_copy.parent / "fsaverage") + ": absent" in message
    assert "$FREESURFER_HOME/subjects/fsaverage: not set" in message
    # and how to get one
    assert "--fsaverage DIR" in message and "set FREESURFER_HOME" in message
    assert "next to the subject directory" in message and "\n" not in message
    assert info.template is None and info.fsaverage_poles is None
    assert tree(tmp_path / "media") == tree(world.plain_dir)

    # said again for a render that is found complete: it is as true of that one
    caplog.clear()
    age(tmp_path / "media")
    with caplog.at_level(logging.WARNING, logger="surfcensus"):
        assert three_plane(subject_copy, tmp_path / "media", CONFIG, video=False) == info
    assert not_found_warnings(caplog) == [message] and rewritten(tmp_path / "media") == set()

    # a video has no marks to map
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="surfcensus"):
        three_plane(subject_copy, tmp_path / "sub.mp4", CONFIG)
    assert caplog.records == []


@pytest.mark.parametrize("failure", ["missing surface", "no encoder", "output in source"])
def test_a_render_that_fails_says_nothing_about_the_template(
    subject_copy: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    """One message for one problem: that marks will not map is of no interest for a subject
    that is not rendered."""
    out, error = tmp_path / "media", api.RenderError
    if failure == "missing surface":
        (subject_copy / "surf" / "rh.white").unlink()
    elif failure == "no encoder":
        monkeypatch.setenv("IMAGEIO_FFMPEG_EXE", str(tmp_path / "no-ffmpeg"))
    else:
        out, error = subject_copy / "media", OutputInSourceError
    with caplog.at_level(logging.WARNING, logger="surfcensus"), pytest.raises(error):
        three_plane(subject_copy, out, CONFIG)
    assert caplog.records == [] and not out.exists()


@pytest.mark.parametrize("unusable", ["incomplete", "of a lower order", "damaged"])
def test_three_plane_next_to_an_unusable_fsaverage_renders_without_counterparts(
    world: World,
    synth_template: SyntheticTemplate,
    subject_copy: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    unusable: str,
) -> None:
    """With ``"auto"`` a template is used when there is one. A directory that cannot serve
    as one is no reason to refuse the render; the warning says what is wrong with it."""
    adjacent = subject_copy.parent / "fsaverage"
    if unusable == "incomplete":
        (adjacent / "surf").mkdir(parents=True)
        shutil.copy(synth_template.path / "surf" / "lh.sphere.reg", adjacent / "surf")
    elif unusable == "of a lower order":  # a whole fsaverage5: no fsaverage6 can be cut from it
        write_synthetic_template(subject_copy.parent, order=5)
    else:
        write_synthetic_template(subject_copy.parent, order=6)
        (adjacent / "label" / "lh.aparc.annot").write_bytes(b"no annotation")
    before = snapshot_tree(adjacent)
    with caplog.at_level(logging.WARNING, logger="surfcensus"):
        info = three_plane(subject_copy, tmp_path / "media", CONFIG, video=False)
    (message,) = not_found_warnings(caplog)
    if unusable == "incomplete":
        assert message.startswith(NOT_FOUND) and f"{adjacent}: incomplete" in message
    else:
        reason = "not icosahedral" if unusable == "of a lower order" else "unreadable"
        assert message.startswith(
            f"the fsaverage template that was found cannot be used ({reason}: "
        )
        assert message.endswith("click marks will not map to fsaverage")
    assert info.template is None and info.fsaverage_poles is None
    assert tree(tmp_path / "media") == tree(world.plain_dir)
    assert snapshot_tree(adjacent) == before


def test_three_plane_with_an_fsaverage_directory(
    world: World, synth_template: SyntheticTemplate, subject_copy: Path, tmp_path: Path
) -> None:
    """The explicit forms, which ``render-subject`` offers as ``--fsaverage DIR`` and
    ``--no-template``."""
    default = world.templates["fsaverage6"]
    out = tmp_path / "media"
    info = three_plane(subject_copy, out, CONFIG, template=synth_template.path, video=False)
    assert info.template is not None
    assert (info.template.mesh, info.template.template_id) == ("fsaverage6", default.template_id)

    # a Template is used as it is, whatever its mesh
    age(out)
    on4 = three_plane(subject_copy, out, CONFIG, template=world.template, video=False)
    assert on4 == world.mapped.model_copy(update={"inputs": on4.inputs})
    assert rewritten(out) == TWO_FILES

    # None: no counterparts, also next to a template, and no warning about it
    (subject_copy.parent / "fsaverage").symlink_to(synth_template.path, target_is_directory=True)
    plain = three_plane(subject_copy, out, CONFIG, template=None, video=False)
    assert plain.template is None and rewritten(out) == TWO_FILES
    assert tree(out) == tree(world.plain_dir)


@pytest.mark.parametrize("unusable", ["absent", "incomplete", "of a lower order"])
def test_three_plane_refuses_an_fsaverage_directory_that_cannot_be_used(
    synth_template: SyntheticTemplate, subject_copy: Path, tmp_path: Path, unusable: str
) -> None:
    """A template that was asked for is not dropped silently, and nothing is written."""
    given = tmp_path / "given" / "fsaverage"
    if unusable == "incomplete":
        (given / "surf").mkdir(parents=True)
        shutil.copy(synth_template.path / "surf" / "lh.sphere.reg", given / "surf")
    elif unusable == "of a lower order":
        write_synthetic_template(given.parent, order=5)
    error = TemplateError if unusable == "of a lower order" else TemplateNotFound
    for out in (tmp_path / "media", tmp_path / "videos" / "sub.mp4"):
        with pytest.raises(error) as caught:
            three_plane(subject_copy, out, CONFIG, template=given)
        assert not out.exists() and not out.parent.joinpath("media").exists()
        if unusable == "of a lower order":
            assert caught.value.reason == "not icosahedral"
        else:
            ((origin, where, state),) = caught.value.looked_in
            assert (origin, where) == ("explicit", str(given))
            assert state == ("absent" if unusable == "absent" else state)
            assert state.startswith("absent" if unusable == "absent" else "incomplete: ")
    assert not (tmp_path / "videos").exists()


def test_three_plane_to_a_video_reports_the_template(
    world: World, study: Path, tmp_path: Path
) -> None:
    target = tmp_path / "videos" / "sub-01.mp4"
    info = three_plane(study / "sub-01", target, CONFIG)
    assert sorted(p.name for p in target.parent.iterdir()) == ["sub-01.mp4"]
    assert info.video is not None and info.template is not None
    assert info.template.template_id == world.templates["fsaverage6"].template_id
    assert info.fsaverage_poles is not None and set(info.fsaverage_poles) == {"lh", "rh"}
    assert {"lh_sphere_reg", "rh_sphere_reg"} <= set(info.inputs)
    bare = three_plane(study / "sub-01", tmp_path / "videos" / "bare.mp4", CONFIG, template=None)
    assert bare.template is None and not any("sphere" in role for role in bare.inputs)
    assert target.read_bytes() == (tmp_path / "videos" / "bare.mp4").read_bytes()


@pytest.mark.parametrize("how", ["given", "found next to the subject"])
@pytest.mark.parametrize("target", ["media", "media/sub.mp4", "surf"])
def test_nothing_is_rendered_into_the_template_directory(
    world: World, subject_copy: Path, tmp_path: Path, how: str, target: str
) -> None:
    """The fsaverage directory is read like the subject directory, and written to as little."""
    if how == "given":
        fsaverage = write_synthetic_template(tmp_path / "templates", order=6).path
        template: Path | Literal["auto"] = fsaverage
    else:
        fsaverage = write_synthetic_template(subject_copy.parent, order=6).path
        template = "auto"
    before = snapshot_tree(fsaverage)
    with pytest.raises(OutputInSourceError, match="only read"):
        three_plane(subject_copy, fsaverage / target, CONFIG, template=template)
    assert snapshot_tree(fsaverage) == before
    # through a link to the directory as well
    link = tmp_path / "link-to-template"
    link.symlink_to(fsaverage, target_is_directory=True)
    with pytest.raises(OutputInSourceError):
        three_plane(subject_copy, link / target, CONFIG, template=template)
    assert snapshot_tree(fsaverage) == before
    # without the template the directory is no source of this render
    if how == "given" and target == "media":
        three_plane(subject_copy, tmp_path / "elsewhere", CONFIG, template=template, video=False)
        assert (tmp_path / "elsewhere" / "render.json").is_file()


# ---------------------------------------------------------------- the real subject

REAL_SUBJECT = "ds000102-sub-01"
REAL_FSAVERAGE = "ds000102-fsaverage"
FSAVERAGE_URL = (
    "https://openneuro-derivatives.s3.amazonaws.com/fmriprep/ds000102-fmriprep/"
    "sourcedata/freesurfer/fsaverage/"
)
REAL_FSAVERAGE_FILES = {  # bytes, from the bucket listing of 2026-11-13
    "surf/lh.sphere.reg": 5_898_546,
    "surf/rh.sphere.reg": 5_898_546,
    "surf/lh.white": 5_898_808,
    "surf/rh.white": 5_898_808,
    "label/lh.cortex.label": 6_117_569,
    "label/rh.cortex.label": 5_964_904,
    "label/lh.aparc.annot": 1_312_250,
    "label/rh.aparc.annot": 1_312_250,
    "surf/lh.white.avg.area.mgh": 655_672,
    "surf/rh.white.avg.area.mgh": 655_672,
    "scripts/build-stamp.txt": 49,
}
REAL_SUBJECT_FILES = (
    "mri/T1.mgz",
    "mri/brainmask.mgz",
    *(f"surf/{h}.{name}" for h in HEMIS for name in ("white", "pial", "sphere.reg")),
)
# measured 2026-11-13 on ds000102 sub-01 (FreeSurfer 6.0.1) with the fsaverage next to it:
# template pole vertex, the native vertex it maps to, and its distance from the geometric pole
REAL_POLES = {"lh": (12_950, 1, 1.1632), "rh": (5_558, 206, 6.2925)}
REAL_VERTICES = {"lh": 37_356, "rh": 37_298}


def check_real_render(subject: Path, fsaverage: Path, scratch: Path) -> None:
    """The real subject with the real template: a plain render, the update of two files,
    and counterparts that an independent nearest-vertex search confirms."""
    config = RenderConfig(panel_px=64)
    files = resolve_subject_files(subject)
    source_before = snapshot_tree(subject)
    out = scratch / "media"
    plain = render_subject(files, out, config, video=False)
    assert plain.schema_version == 3 and plain.template is None
    tiles = tree(out)
    age(out)
    template = build_template(fsaverage, "adjacent")
    info = render_subject(files, out, config, template=template, video=False)
    assert rewritten(out) == TWO_FILES
    after = tree(out)
    assert {name for name in tiles if tiles[name] != after[name]} == TWO_FILES
    assert render_key(info) == render_key(plain)

    assert info.template == TemplateInfo(
        mesh="fsaverage6",
        template_id=template.template_id,
        sphere_sha256={h: sha256(fsaverage / "surf" / f"{h}.sphere.reg") for h in HEMIS},
    )
    assert info.vertex_map is not None and info.vertex_map.fsaverage == "fsaverage6"
    assert (info.vertex_map.n_lh, info.vertex_map.n_rh) == (
        REAL_VERTICES["lh"],
        REAL_VERTICES["rh"],
    )
    assert info.fsaverage_poles is not None
    vmap = read_vertex_map(out / "vertex_map.npz")
    for hemi in HEMIS:
        used = info.inputs[f"{hemi}_sphere_reg"]
        path = subject / "surf" / f"{hemi}.sphere.reg"
        assert (used.name, used.sha256) == (f"{hemi}.sphere.reg", sha256(path))

        # the counterparts, from the two files alone: fsaverage6 is the first 40,962
        # vertices of the template's sphere
        registered = fs.read_geometry(str(path))[0].astype(float)
        target = fs.read_geometry(str(fsaverage / "surf" / f"{hemi}.sphere.reg"))[0][:40_962]
        target = target / np.linalg.norm(target, axis=1, keepdims=True)
        registered /= np.linalg.norm(registered, axis=1, keepdims=True)
        index, found = vmap[f"{hemi}_index"], vmap[f"{hemi}_fsavg6"]
        assert len(index) == REAL_VERTICES[hemi]
        assert found.min() >= 0 and found.max() < 40_962
        np.testing.assert_array_equal(found, cKDTree(target).query(registered[index])[1])
        assert len(np.unique(found)) > 5_000

        pole = info.fsaverage_poles[hemi]
        pial = fs.read_geometry(str(subject / "surf" / f"{hemi}.pial"))[0]
        template_vertex, native_vertex, distance = REAL_POLES[hemi]
        assert (pole.template_vertex, pole.native_vertex) == (template_vertex, native_vertex)
        assert native_vertex == int(cKDTree(registered).query(target[template_vertex])[1])
        assert pole.tkr == tuple(float(x) for x in pial[native_vertex])
        geometric = pial[int(np.argmin(pial[:, 1]))].astype(float)
        assert pole.distance_mm == pytest.approx(
            float(np.linalg.norm(pial[native_vertex] - geometric)), abs=1e-9
        )
        assert pole.distance_mm == pytest.approx(distance, abs=1e-3)
        print(f"{hemi}: fsaverage pole {pole.distance_mm:.2f} mm from the geometric pole")
    assert info.fsaverage_poles["lh"].distance_mm < 3.0
    assert snapshot_tree(subject) == source_before


def real_files_present(subject: Path, fsaverage: Path) -> bool:
    return all((subject / name).is_file() for name in REAL_SUBJECT_FILES) and all(
        (fsaverage / name).is_file() and (fsaverage / name).stat().st_size == size
        for name, size in REAL_FSAVERAGE_FILES.items()
    )


def test_cached_real_subject_with_the_real_template(cached_subjects: Path, tmp_path: Path) -> None:
    subject, fsaverage = cached_subjects / REAL_SUBJECT, cached_subjects / REAL_FSAVERAGE
    if not real_files_present(subject, fsaverage):
        pytest.skip("the real subject and the real fsaverage are not in the pytest cache")
    check_real_render(subject, fsaverage, tmp_path)


@pytest.fixture(scope="module")
def real_fsaverage(request: pytest.FixtureRequest) -> Iterator[Path]:
    """The template files of the ``fsaverage`` next to ds000102 (40 MB) in the pytest cache.
    Anonymous, read-only GETs; a file of the expected size is not fetched again."""
    import httpx

    root = request.config.cache.mkdir(REAL_FSAVERAGE)
    with httpx.Client(timeout=120.0) as client:
        for relative, size in REAL_FSAVERAGE_FILES.items():
            path = root / relative
            if path.is_file() and path.stat().st_size == size:
                continue
            content = client.get(FSAVERAGE_URL + relative).raise_for_status().content
            assert len(content) == size, f"{relative}: {len(content)} bytes, expected {size}"
            path.parent.mkdir(parents=True, exist_ok=True)
            part = path.with_name(path.name + ".part")
            part.write_bytes(content)
            part.replace(path)
    yield root


@pytest.mark.network
def test_real_subject_with_the_real_template(
    real_subject: Path, real_fsaverage: Path, tmp_path: Path
) -> None:
    check_real_render(real_subject, real_fsaverage, tmp_path)
