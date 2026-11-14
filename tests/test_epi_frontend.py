"""The modules of the EPI page under Node (work package E1.D): the timeline of the subject's
one stream, the slices of another grid at the same place in the anatomy, the common intensity
scale, the chunks the page holds, and the slices of a volume, each against a computation in
Python from the shapes and the packs. Skipped where there is no ``node``.

The packs are those of ``tests/epi_packs.py``; the shape is the one the server answers with
(``app.epi.Packs.shape``)."""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from epi_packs import PackedStream, make_stream, stream_order  # pyrefly: ignore[missing-import]

import surfcensus.app
from surfcensus.app.epi import Packs
from surfcensus.func import PLANES
from surfcensus.func.pack import slice_planes
from surfcensus.func.packread import common_scale, decode_volume, read_pack
from surfcensus.records import Grid, to_matrix4

# the files of the package that is imported (a copy on PYTHONPATH as well)
APP_DIR = Path(surfcensus.app.__file__).resolve().parent
EPI_JS = APP_DIR / "static" / "js" / "epi"
JS_FILES = ("epi.js", "logic.js", "store.js", "traces.js", "views.js", "worker.js")

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")


@pytest.fixture(scope="module")
def packed(tmp_path_factory: pytest.TempPathFactory) -> PackedStream:
    return make_stream(tmp_path_factory.mktemp("epi"), n_volumes=14)


@pytest.fixture(scope="module")
def shape(packed: PackedStream) -> dict[str, Any]:
    return Packs(packed.out).shape().model_dump(mode="json")


@pytest.fixture(scope="module")
def js_copy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("epijs")
    shutil.copytree(EPI_JS, root / "js")
    (root / "package.json").write_text('{"type": "module"}\n', encoding="utf-8")
    return root


def run_js(js_copy: Path, script: str, data: Any = None) -> Any:
    assert NODE is not None
    (js_copy / "input.json").write_text(json.dumps(data), encoding="utf-8")
    main = (
        'import { readFileSync } from "node:fs";\n'
        'import * as L from "./js/logic.js";\n'
        'const text = readFileSync(new URL("./input.json", import.meta.url), "utf8");\n'
        "const input = JSON.parse(text);\n"
        "const output = (value) => console.log(JSON.stringify(value));\n" + script
    )
    (js_copy / "main.mjs").write_text(main, encoding="utf-8")
    done = subprocess.run(
        [NODE, "main.mjs"], cwd=js_copy, capture_output=True, text=True, timeout=120, check=False
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


# --- the files ------------------------------------------------------------------------------


def test_the_files_of_the_page() -> None:
    assert {p.name for p in EPI_JS.iterdir()} == set(JS_FILES)
    assert (APP_DIR / "templates" / "epi.html").is_file()


@needs_node
@pytest.mark.parametrize("name", JS_FILES)
def test_every_module_parses(js_copy: Path, name: str) -> None:
    assert NODE is not None
    done = subprocess.run(
        [NODE, "--check", f"js/{name}"], cwd=js_copy, capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr


def test_modules_import_each_other_by_relative_path_and_load_nothing_from_elsewhere() -> None:
    url = re.compile(r"""(?:https?:)?//[\w.-]+\.[a-z]{2,}""", flags=re.I)
    for name in JS_FILES:
        source = (EPI_JS / name).read_text(encoding="utf-8")
        for target in re.findall(r"""^\s*import\s[^;]*?from\s+["']([^"']+)["']""", source, re.M):
            assert target.startswith("./") and (EPI_JS / target[2:]).is_file(), (name, target)
        assert "import(" not in source and "eval(" not in source and "new Function" not in source
        assert url.findall(source) == [], name
        # the page writes nothing: no storage, no download, no request but a GET
        for word in ("localStorage", "sessionStorage", "indexedDB", "sendBeacon", "method:"):
            assert word not in source, (name, word)
    template = (APP_DIR / "templates" / "epi.html").read_text(encoding="utf-8")
    assert url.findall(template) == []


def test_every_element_the_page_looks_up_is_in_the_template() -> None:
    source = (EPI_JS / "epi.js").read_text(encoding="utf-8")
    wanted = set(re.findall(r"\$\('([\w-]+)'\)", source))
    template = (APP_DIR / "templates" / "epi.html").read_text(encoding="utf-8")
    present = set(re.findall(r'\bid="([\w-]+)"', template))
    assert wanted and wanted <= present, wanted - present


def test_every_field_the_page_reads_of_a_run_is_in_the_shapes() -> None:
    from surfcensus.app import epi_schemas as shapes

    fields: set[str] = set()
    for model in (
        shapes.EpiStreamOut,
        shapes.EpiSessionOut,
        shapes.EpiRunOut,
        shapes.EpiPackOut,
        shapes.GridOut,
        shapes.StreamOut,
        shapes.ChunkOut,
        shapes.ContourOut,
        shapes.TracesOut,
        shapes.RegistrationOut,
    ):
        fields |= set(model.model_fields)
    from surfcensus.func.model import CorrectionsInfo, FrameInfo, StatementInfo

    for model in (CorrectionsInfo, FrameInfo, StatementInfo):
        fields |= set(model.model_fields)
    from surfcensus.func.model import RULES

    fields |= set(RULES)  # stream.rules.<name>
    source = "".join((EPI_JS / name).read_text(encoding="utf-8") for name in JS_FILES)
    # a field of the answer is read as .snake_case; every one must be a field of a shape
    read = set(re.findall(r"\.([a-z]+_[a-z_]+)\b", source))
    assert read and read <= fields, read - fields


# --- the timeline ---------------------------------------------------------------------------


@needs_node
def test_the_timeline_is_every_packed_run_of_every_session_in_order(
    js_copy: Path, packed: PackedStream, shape: dict[str, Any]
) -> None:
    out = run_js(
        js_copy,
        "const t = L.buildTimeline(input.sessions);\n"
        "const located = [];\n"
        "for (let g = 0; g < t.total; g++) located.push(L.locate(t, g));\n"
        "const chunks = L.streamChunks(t);\n"
        "const chunkOf = [];\n"
        "for (let g = 0; g < t.total; g++) chunkOf.push(L.chunkIndexOf(chunks, g));\n"
        "output({ total: t.total, segments: t.segments, located, chunks, chunkOf,"
        " boundaries: L.boundaries(t) });\n",
        shape,
    )
    order = stream_order(packed)
    assert out["total"] == sum(n for *_, n in order) == 8 * 14
    runs = [
        (
            shape["sessions"][s["session"]]["session_uid"],
            shape["sessions"][s["session"]]["runs"][s["run"]]["run_uid"],
            s["first"],
            s["n"],
        )
        for s in out["segments"]
    ]
    assert runs == order
    # every volume of the stream lies in the segment whose stretch holds it
    for g, k in enumerate(out["located"]):
        seg = out["segments"][k]
        assert seg["first"] <= g < seg["first"] + seg["n"]
    # the chunks cover the stream once, in order, and each volume is in its chunk
    at = 0
    for c in out["chunks"]:
        assert c["first"] == at
        at += c["count"]
    assert at == out["total"]
    for g, c in enumerate(out["chunkOf"]):
        chunk = out["chunks"][c]
        assert chunk["first"] <= g < chunk["first"] + chunk["count"]
    # boundaries: every run start but the first; a session boundary where the session changes
    assert [b["at"] for b in out["boundaries"]] == [first for _, _, first, _ in order[1:]]
    sessions = [s for s, *_ in order]
    assert [b["session"] for b in out["boundaries"]] == [
        sessions[k] != sessions[k - 1] for k in range(1, len(sessions))
    ]
    assert sum(b["session"] for b in out["boundaries"]) == 1


@needs_node
def test_the_traces_of_the_stream_are_those_of_its_runs(
    js_copy: Path, packed: PackedStream, shape: dict[str, Any]
) -> None:
    out = run_js(
        js_copy,
        "const t = L.buildTimeline(input.sessions);\noutput(L.streamTraces(t, input.sessions));\n",
        shape,
    )
    fd: list[float | None] = []
    ns: list[int] = []
    for session in shape["sessions"]:
        for run in session["runs"]:
            if run["pack"] is None:
                continue
            ns += [len(fd) + t for t in run["pack"]["traces"]["non_steady"]]
            fd += run["pack"]["traces"]["fd"]
    assert out["fd"] == fd and out["nonSteady"] == ns


# --- the same place across grids ------------------------------------------------------------


def nearest_slices(grid: dict[str, Any], x: np.ndarray) -> list[int]:
    out = []
    for plane in PLANES:
        target = float(np.dot(grid["slice_normal"][plane], x))
        mm = np.asarray(grid["slice_mm"][plane])
        out.append(int(np.argmin(np.abs(mm - target))))  # argmin: the lower of two equals
    return out


def meet(grid: dict[str, Any], slices: list[int]) -> np.ndarray:
    normals = np.asarray([grid["slice_normal"][p] for p in PLANES])
    distances = np.asarray([grid["slice_mm"][p][k] for p, k in zip(PLANES, slices, strict=True)])
    return np.linalg.solve(normals, distances)


def rotated_grid(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    angles = rng.uniform(-0.4, 0.4, 3)
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    affine = np.eye(4)
    affine[:3, :3] = rz @ ry @ rx @ np.diag(rng.uniform(2.0, 4.0, 3))
    affine[:3, 3] = rng.uniform(-90, -40, 3)
    shape = tuple(int(n) for n in rng.integers(30, 60, 3))
    normals, distances = slice_planes(Grid(shape, to_matrix4(affine)))  # type: ignore[arg-type]
    return {"shape": list(shape), "slice_normal": normals, "slice_mm": distances}


@needs_node
def test_the_slices_of_another_grid_are_those_nearest_to_the_place_in_mm(
    js_copy: Path, shape: dict[str, Any]
) -> None:
    grids = [g for s in shape["sessions"] for g in s["grids"]]
    grids += [rotated_grid(seed) for seed in range(4)]
    rng = np.random.default_rng(7)
    cases = []
    for _ in range(300):
        a, b = rng.choice(len(grids), 2)
        slices = [int(rng.integers(0, n)) for n in grids[a]["shape"]]
        cases.append({"from": int(a), "to": int(b), "slices": slices})
    out = run_js(
        js_copy,
        "output(input.cases.map((c) => {\n"
        "  const x = L.pointOf(input.grids[c.from], c.slices);\n"
        "  const to = L.slicesAt(input.grids[c.to], x);\n"
        "  return { x, to, back: L.slicesAt(input.grids[c.from], x) };\n"
        "}));\n",
        {"grids": grids, "cases": cases},
    )
    for case, got in zip(cases, out, strict=True):
        x = meet(grids[case["from"]], case["slices"])
        assert np.allclose(got["x"], x, atol=1e-6)
        assert got["back"] == case["slices"]
        want = nearest_slices(grids[case["to"]], np.asarray(got["x"]))
        assert got["to"] == want


@needs_node
def test_a_place_half_way_between_two_slices_goes_to_the_lower_index(js_copy: Path) -> None:
    grid = {
        "shape": [4, 4, 4],
        "slice_normal": {p: [float(p == q) for q in PLANES] for p in PLANES},
        "slice_mm": {p: [0.0, 2.0, 4.0, 6.0] for p in PLANES},
    }
    out = run_js(js_copy, "output(L.slicesAt(input, [1.0, 3.0, 5.0]));\n", grid)
    assert out == [0, 1, 2]


# --- intensity ------------------------------------------------------------------------------


@needs_node
def test_the_common_scale_is_that_of_packread_for_every_byte(
    js_copy: Path, shape: dict[str, Any]
) -> None:
    windows = [r["window"] for s in shape["sessions"] for r in s["runs"] if r["pack"]]
    rng = np.random.default_rng(3)
    pairs = [(w[1], max(x[1] for x in windows)) for w in windows]
    for _ in range(200):
        hi = float(rng.uniform(1, 5000))
        pairs.append((hi, hi * float(rng.uniform(1, 3))))
    pairs += [(1.0, 1.0), (255.0, 510.0), (1.0, 3.0), (1.0, 2.0), (0.1, 0.30000000000000004)]
    out = run_js(
        js_copy, "output(input.map(([h, t]) => Array.from(L.commonScale(h, t))));\n", pairs
    )
    for (hi, top), table in zip(pairs, out, strict=True):
        assert table == common_scale((0.0, hi), top).tolist(), (hi, top)


@needs_node
def test_normalization_on_is_the_stored_values_and_off_one_window_for_the_stream(
    js_copy: Path, shape: dict[str, Any]
) -> None:
    runs = [r for s in shape["sessions"] for r in s["runs"] if r["pack"]]
    data = {"value": [r["window"] for r in runs], "sd": [r["sd_window"] for r in runs]}
    out = run_js(
        js_copy,
        "const pick = (t) => t.map((x) => ({ value: Array.from(x.value),"
        " sd: Array.from(x.sd), top: x.top, topSd: x.topSd }));\n"
        "output({ on: pick(L.intensityTables(input.value, input.sd, true)),"
        " off: pick(L.intensityTables(input.value, input.sd, false)) });\n",
        data,
    )
    top = max(w[1] for w in data["value"])
    top_sd = max(w[1] for w in data["sd"])
    for k, run in enumerate(runs):
        assert out["on"][k]["value"] == list(range(256)) == out["on"][k]["sd"]
        assert out["on"][k]["top"] == run["window"][1]
        assert out["off"][k]["value"] == common_scale(tuple(run["window"]), top).tolist()
        assert out["off"][k]["sd"] == common_scale(tuple(run["sd_window"]), top_sd).tolist()
        assert out["off"][k]["top"] == top
    # the brighter run sets the common window and keeps its values; the others come out darker
    brightest = max(range(len(runs)), key=lambda k: runs[k]["window"][1])
    assert out["off"][brightest]["value"] == list(range(256))
    assert out["off"][0]["value"][255] < 255


@needs_node
def test_a_slice_through_a_table_is_the_decoded_volume_through_it(
    js_copy: Path, packed: PackedStream, shape: dict[str, Any]
) -> None:
    run = packed.runs[4]  # the run on grid C
    session_dir = packed.out / run.session
    info = read_pack(session_dir, run.run)
    assert info is not None
    volume = decode_volume(session_dir, info, "raw", 3)
    table = list(range(255, -1, -1))
    flat = volume.ravel(order="F").tolist()
    out = run_js(
        js_copy,
        "const r = [];\n"
        "for (let p = 0; p < 3; p++) {\n"
        "  const [w, h] = L.planeSize(input.shape, p);\n"
        "  const ax = L.SPEC[p].axis;\n"
        "  const k = Math.floor(input.shape[ax] / 3);\n"
        "  const out = new Uint8Array(w * h);\n"
        "  const v = Uint8Array.from(input.v);\n"
        "  L.extractSlice(v, input.shape, p, k, out, Uint8Array.from(input.t));\n"
        "  r.push({ w, h, k, a: Array.from(out) });\n"
        "}\noutput(r);\n",
        {"shape": list(volume.shape), "v": flat, "t": table},
    )
    lut = np.asarray(table, np.uint8)
    for plane, got in enumerate(out):
        sl = np.take(volume, got["k"], axis=plane)  # the other two axes in order: (h, v)
        rows = lut[sl.T[::-1]]  # vertical runs against the voxel axis, top row first
        if plane > 0:
            rows = rows[:, ::-1]  # coronal and axial: horizontal against the voxel axis
        assert got["a"] == rows.ravel().tolist()


# --- chunks ---------------------------------------------------------------------------------


@needs_node
def test_the_chunks_after_a_boundary_are_wanted_before_it(
    js_copy: Path, shape: dict[str, Any]
) -> None:
    out = run_js(
        js_copy,
        "const t = L.buildTimeline(input.sessions);\n"
        "const chunks = L.streamChunks(t);\n"
        "const sizes = chunks.map((c) => c.count * 1000);\n"
        "const r = [];\n"
        "for (let cursor = 0; cursor < chunks.length; cursor++) {\n"
        "  for (const limit of [1, 5000, 1e9]) {\n"
        "    const d = L.desiredChunks({ sizes, cursor, streams: ['raw'], limitBytes: limit });\n"
        "    r.push({ cursor, limit, d });\n"
        "  }\n"
        "}\n"
        "output({ n: chunks.length, segs: chunks.map((c) => c.seg), sizes, r });\n",
        shape,
    )
    n = out["n"]
    for case in out["r"]:
        keys = case["d"]
        cursor = case["cursor"]
        nxt = (cursor + 1) % n
        # the chunk on screen and the next one, across a run or session boundary and round the
        # end of the stream, whatever the limit
        assert keys[:2] == [f"{cursor}|raw", f"{nxt}|raw"]
        assert len(set(keys)) == len(keys)
        total = sum(out["sizes"][int(k.split("|")[0])] for k in keys)
        assert total <= max(case["limit"], out["sizes"][cursor] + out["sizes"][nxt])
        if case["limit"] == 1e9:
            assert len(keys) == 2 * n  # everything, the stream off screen last
            assert all(k.endswith("|raw") for k in keys[:n])
    # somewhere the next chunk is in another segment: the prefetch crosses boundaries
    assert any(out["segs"][c] != out["segs"][(c + 1) % n] for c in range(n))


@needs_node
def test_the_playback_clock_paints_every_volume(js_copy: Path) -> None:
    out = run_js(
        js_copy,
        "let due = 0; let shown = 0; const at = [];\n"
        "for (let now = 0; now < 2000; now += 16.7) {\n"
        "  const s = L.playbackStep(due, now, 60);\n"
        "  if (s.advance) { shown += 1; due = s.nextDue; at.push(now); }\n"
        "}\noutput({ shown });\n",
    )
    assert math.isclose(out["shown"], 120, abs_tol=2)
