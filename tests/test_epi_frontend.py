"""The modules of the EPI page under Node (work package E1.D): the timeline of the subject's
one stream, the slices of another grid at the same place in the anatomy, the common intensity
scale, the chunks the page holds, and the slices of a volume, each against a computation in
Python from the shapes and the packs. Skipped where there is no ``node``.

The packs are those of ``tests/epi_packs.py``; the shape is the one the server answers with
(``app.epi.Packs.shape``)."""

from __future__ import annotations

import itertools
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
JS_FILES = ("epi.js", "index.js", "logic.js", "store.js", "traces.js", "views.js", "worker.js")
"""The modules of ``js/epi/``: the page of a stream and its parts, and ``index.js``, the
index of the participants of a run directory (``templates/epi_index.html``)."""

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
    assert (APP_DIR / "templates" / "epi_index.html").is_file()


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
        shapes.EpiIndexOut,
        shapes.EpiParticipantOut,
        shapes.EpiIndexSessionOut,
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
def test_the_segment_of_a_run_is_where_the_page_starts_when_its_address_names_it(
    js_copy: Path, packed: PackedStream, shape: dict[str, Any]
) -> None:
    """``#run=<run_uid>`` (the links of the subject pages, step 3): the segment of that run on
    the timeline, and -1 for a run that is not on it, which the page starts at 0 for."""
    absent = shape["sessions"][0]["runs"][0]["run_uid"][::-1]
    wanted = [r["run_uid"] for s in shape["sessions"] for r in s["runs"]] + [absent, "", None]
    out = run_js(
        js_copy,
        "const t = L.buildTimeline(input.shape.sessions);\n"
        "output(input.wanted.map((uid) => L.segmentOfRun(t, input.shape.sessions, uid)));\n",
        {"shape": shape, "wanted": wanted},
    )
    order = [run for _, run, _, _ in stream_order(packed)]
    for uid, k in zip(wanted, out, strict=True):
        if uid in order:
            assert k == order.index(uid), uid
        else:
            assert k == -1, uid
    assert sorted(k for k in out if k >= 0) == list(range(len(order)))


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


@needs_node
def test_a_step_in_one_plane_moves_the_place_along_its_normal_only(
    js_copy: Path, shape: dict[str, Any]
) -> None:
    """Finding 12 of the review of the page: a slice step on a coarse grid moved the place to
    that grid's slice centres, so on coming back to a fine grid the other two planes changed.
    With ``movePlace`` only the stepped plane's distance changes."""
    grids = {g["grid_key"]: g for s in shape["sessions"] for g in s["grids"]}
    fine, coarse = sorted(grids.values(), key=lambda g: g["shape"][0], reverse=True)[:2]
    rng = np.random.default_rng(12)
    cases = [
        {"slices": [int(rng.integers(0, n)) for n in fine["shape"]], "plane": int(p), "dir": d}
        for p in range(3)
        for d in (-1, 1)
        for _ in range(15)
    ]
    out = run_js(
        js_copy,
        "output(input.cases.map((c) => {\n"
        "  const x = L.pointOf(input.fine, c.slices);\n"
        "  const on = L.slicesAt(input.coarse, x);\n"
        "  const n = input.coarse.shape[c.plane];\n"
        "  const to = Math.min(n - 1, Math.max(0, on[c.plane] + c.dir));\n"
        "  const stepped = on.slice(); stepped[c.plane] = to;\n"
        "  const moved = L.movePlace(input.coarse, x, c.plane, to);\n"
        "  const reset = L.pointOf(input.coarse, stepped);\n"
        "  return { back: L.slicesAt(input.fine, moved), old: L.slicesAt(input.fine, reset),"
        " d: L.slicesAt(input.coarse, moved), to };\n"
        "}));\n",
        {"fine": fine, "coarse": coarse, "cases": cases},
    )
    changed_by_old_rule = 0
    for case, got in zip(cases, out, strict=True):
        p = case["plane"]
        others = [q for q in range(3) if q != p]
        # the other two planes come back where they were; the stepped plane is on its slice
        assert [got["back"][q] for q in others] == [case["slices"][q] for q in others], case
        assert got["d"][p] == got["to"]  # on the coarse grid, the slice that was stepped to
        changed_by_old_rule += [got["old"][q] for q in others] != [
            case["slices"][q] for q in others
        ]
    # the cases do reach what the earlier rule (the place from all three slices) got wrong
    assert changed_by_old_rule > 0


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
    # exact halves, where floor(b * (hi / top) + 0.5) and the rounding of b * hi / top differ
    # (review of the page: windows (0, 7) and (0, 10), stored value 45: 31, not 32)
    pairs += [(7.0, 10.0), (3.0, 10.0), (7.0, 20.0)]
    out = run_js(
        js_copy, "output(input.map(([h, t]) => Array.from(L.commonScale(h, t))));\n", pairs
    )
    for (hi, top), table in zip(pairs, out, strict=True):
        assert table == common_scale((0.0, hi), top).tolist(), (hi, top)
    halves = dict(zip(pairs, out, strict=True))
    assert halves[(7.0, 10.0)][45] == 31 == int(common_scale((0.0, 7.0), 10.0)[45])
    assert round(45 * 7.0 / 10.0 + 1e-9) == 32  # the other rounding, which the test must catch


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


# --- the playback rate (D43) ---------------------------------------------------------------------


@needs_node
def test_the_rates_and_the_keys_that_step_through_them(js_copy: Path) -> None:
    out = run_js(
        js_copy,
        "const up = [1]; while (up.length < 20) up.push(L.stepRate(up[up.length - 1], 1));\n"
        "const down = [300];\n"
        "while (down.length < 20) down.push(L.stepRate(down[down.length - 1], -1));\n"
        "output({ rates: L.RATES, max: L.PAINT_EVERY_MAX, up, down, odd: L.stepRate(100, 1) });\n",
    )
    assert out["rates"] == [1, 2, 5, 12, 24, 30, 60, 120, 200, 300]
    assert out["max"] == 60
    assert out["up"][:10] == out["rates"] and set(out["up"][10:]) == {300}
    assert out["down"][:10] == out["rates"][::-1] and set(out["down"][10:]) == {1}
    assert out["odd"] == 200  # a rate not in the list steps from the next one above it


@needs_node
def test_volumes_are_left_out_only_above_60_and_above_the_frame_rate(js_copy: Path) -> None:
    out = run_js(
        js_copy,
        "const r = {};\n"
        "for (const rate of L.RATES) for (const fps of [null, 30, 60, 120, 144, 300])\n"
        "  r[`${rate}|${fps}`] = L.maySkip(rate, fps);\n"
        "output(r);\n",
    )
    for key, skip in out.items():
        rate, fps = key.split("|")
        rate = int(rate)
        if rate <= 60 or fps == "null":
            assert skip is False, key  # at 60 and below every volume is painted, on any machine
        else:
            assert skip is (rate > 1.1 * float(fps)), key


def advance_js(rate: int, frame_ms: float, skip: bool, seconds: float = 2.0) -> str:
    return (
        f"let due = 0; const steps = [];\n"
        f"for (let now = 0; now < {seconds * 1000}; now += {frame_ms}) {{\n"
        f"  const s = L.playbackAdvance(due, now, {rate}, {str(skip).lower()});\n"
        "  if (s.n > 0) { steps.push(s.n); due = s.nextDue; }\n"
        "}\noutput({ steps });\n"
    )


@needs_node
@pytest.mark.parametrize("rate", [1, 12, 30, 60])
def test_without_skipping_the_clock_paints_every_volume_one_at_a_time(
    js_copy: Path, rate: int
) -> None:
    for frame in (16.7, 40.0):  # a page at 60 frames per second, and one that is slower
        steps = run_js(js_copy, advance_js(rate, frame, False))["steps"]
        assert set(steps) == {1}
        if 1000 / frame >= rate:
            assert math.isclose(len(steps), 2 * rate, abs_tol=2)


@needs_node
@pytest.mark.parametrize("rate", [120, 200, 300])
def test_with_skipping_the_stream_advances_at_the_rate_asked(js_copy: Path, rate: int) -> None:
    for frame in (16.7, 8.3, 25.0):
        steps = run_js(js_copy, advance_js(rate, frame, True))["steps"]
        # stream time: the volumes due by each frame, so the stream moves on at the rate
        assert math.isclose(sum(steps), 2 * rate, abs_tol=rate * frame / 1000 + 2), (
            frame,
            sum(steps),
        )
        assert all(n >= 1 for n in steps)
        assert len(steps) <= 2000 / frame + 1  # at most one paint per frame
        expected = rate * frame / 1000
        assert max(steps) <= math.ceil(expected) + 1


@needs_node
def test_a_clock_far_behind_starts_again_and_does_not_jump(js_copy: Path) -> None:
    out = run_js(
        js_copy,
        "output([L.playbackAdvance(0, 2000, 300, true), L.playbackAdvance(0, 2000, 30, false),\n"
        "        L.playbackAdvance(100, 50, 300, true), L.playbackAdvance(0, 16, 300, true)]);\n",
    )
    assert out[0] == {"n": 1, "nextDue": pytest.approx(2000 + 1000 / 300), "resynced": True}
    assert out[1]["n"] == 1 and out[1]["resynced"] is True
    assert out[2]["n"] == 0  # not due yet
    assert out[3]["n"] == 1 + math.floor(20 / (1000 / 300))


@needs_node
def test_the_frame_rate_is_the_median_interval_of_the_last_second(js_copy: Path) -> None:
    out = run_js(
        js_copy,
        "const t = []; for (let k = 0; k < 120; k++) t.push(k * 1000 / 60);\n"
        "t.push(t[t.length - 1] + 200);\n"  # one long frame does not move the median
        "output([L.frameRate(t, t[t.length - 1]), L.frameRate(t.slice(0, 5), 70),\n"
        "        L.frameRate([], 0)]);\n",
    )
    assert out[0] == pytest.approx(60, rel=1e-6)
    assert out[1] is None and out[2] is None


@needs_node
def test_what_is_said_beside_the_rate(js_copy: Path) -> None:
    ratios = [1.0, 1.03, 1.2, 1.4, 1.6, 1.9, 2.0, 2.4, 3.0, 3.4, 5.0, 11.0, 21.0, 22.0, 23.0]
    out = run_js(
        js_copy,
        "output({ fresh: input.map((r) => L.skipNote(r)),\n"
        "  ord: [1, 2, 3, 4, 11, 12, 13, 21, 22, 23, 101, 111].map(L.ordinal) });\n",
        ratios,
    )
    texts = [n["text"] for n in out["fresh"]]
    assert texts == [
        "",
        "",
        "80% of volumes shown",
        "70% of volumes shown",
        "60% of volumes shown",
        "every 2nd volume shown",
        "every 2nd volume shown",
        "every 2nd volume shown",
        "every 3rd volume shown",
        "every 3rd volume shown",
        "every 5th volume shown",
        "every 11th volume shown",
        "every 21st volume shown",
        "every 22nd volume shown",
        "every 23rd volume shown",
    ]
    assert out["ord"] == [
        "1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "23rd", "101st", "111th"
    ]  # fmt: skip
    assert max(len(t) for t in texts) <= 24  # the box of the note is 24 characters wide


@needs_node
def test_a_ratio_near_a_rounding_boundary_keeps_its_text(js_copy: Path) -> None:
    out = run_js(
        js_copy,
        "const run = (ratios) => {\n"
        "  let prev = null;\n"
        "  return ratios.map((r) => (prev = L.skipNote(r, prev)).text);\n"
        "};\n"
        "output([run([4.4, 4.6, 4.45, 4.55, 4.6]), run([4.6, 4.4, 4.6, 5.2, 5.4]),\n"
        "        run([1.04, 1.03, 1.06, 1.03]), run([1.06, 1.03, 1.0])]);\n",
    )
    assert out[0] == ["every 4th volume shown"] * 5
    assert out[1] == ["every 5th volume shown"] * 3 + ["every 5th volume shown"] * 2
    assert out[2] == ["", "", "90% of volumes shown", "90% of volumes shown"]
    assert out[3] == ["90% of volumes shown", "90% of volumes shown", ""]


# --- the random order of play (D48) ---------------------------------------------------------


def toy_timeline(lengths: list[int]) -> dict[str, Any]:
    segments, at = [], 0
    for n in lengths:
        segments.append({"first": at, "n": n, "chunks": [{"first": 0, "count": n}]})
        at += n
    return {"total": at, "segments": segments}


@needs_node
def test_the_generator_and_the_shuffle_follow_the_seed_alone(js_copy: Path) -> None:
    out = run_js(
        js_copy,
        "const draw = (seed, n) => {\n"
        "  const r = L.mulberry32(seed); return Array.from({ length: n }, r);\n"
        "};\n"
        "const t = input;\n"
        "output({ a: draw(1, 8), b: draw(1, 8), c: draw(2, 8),\n"
        "  o1: Array.from(L.playOrder(t, 'stream', 1)),\n"
        "  o1b: Array.from(L.playOrder(t, 'stream', 1)),\n"
        "  o2: Array.from(L.playOrder(t, 'stream', 2)), big: draw(4294967295, 3) });\n",
        toy_timeline([30, 50, 20]),
    )
    assert out["a"] == out["b"] and out["a"] != out["c"]
    assert all(0 <= x < 1 for x in out["a"] + out["c"] + out["big"])
    assert out["o1"] == out["o1b"] and out["o1"] != out["o2"]
    # pinned: the first numbers of seed 1, so that a change of the generator is seen (every
    # browser and Node compute these with the same integer arithmetic)
    assert out["a"][:3] == pytest.approx(
        [0.6270739405881613, 0.002735721180215478, 0.5274470399599522]
    )


@needs_node
@pytest.mark.parametrize("lengths", [[30, 50, 20], [1], [7], [1, 1, 1], [24] * 8])
def test_each_scope_is_a_permutation_and_run_by_run_keeps_the_runs_in_order(
    js_copy: Path, lengths: list[int]
) -> None:
    timeline = toy_timeline(lengths)
    out = run_js(
        js_copy,
        "const r = {};\n"
        "for (const scope of L.ORDER_SCOPES) for (const seed of [1, 2, 3]) {\n"
        "  const o = L.playOrder(input, scope, seed);\n"
        "  r[`${scope}|${seed}`] = { o: Array.from(o), pos: Array.from(L.positions(o)) };\n"
        "}\noutput(r);\n",
        timeline,
    )
    total = sum(lengths)
    for key, got in out.items():
        order, pos = got["order"] if "order" in got else got["o"], got["pos"]
        assert sorted(order) == list(range(total)), key
        assert all(order[pos[g]] == g for g in range(total)) and all(
            pos[order[p]] == p for p in range(total)
        )
        if key.startswith("run|"):
            for seg in timeline["segments"]:
                block = order[seg["first"] : seg["first"] + seg["n"]]
                assert sorted(block) == list(range(seg["first"], seg["first"] + seg["n"]))
    if total >= 20:
        # the scope is not ignored: the whole-stream order mixes the runs, run by run does not
        stream = out["stream|1"]["o"]
        firsts = [seg["first"] for seg in timeline["segments"]]
        run_of = [max(k for k, f in enumerate(firsts) if f <= g) for g in stream[: lengths[0]]]
        assert len(set(run_of)) > 1
        assert out["stream|1"]["o"] != list(range(total))
        assert out["run|1"]["o"] != list(range(total))


@needs_node
def test_an_empty_stream_has_an_empty_order_and_no_step(js_copy: Path) -> None:
    out = run_js(
        js_copy,
        "const t = { total: 0, segments: [] };\n"
        "const bad = () => {\n"
        "  try { L.playOrder(t, 'x', 1); return 'no'; } catch { return 'raised'; }\n"
        "};\n"
        "output({ s: Array.from(L.playOrder(t, 'stream', 1)),\n"
        "  r: Array.from(L.playOrder(t, 'run', 1)),\n"
        "  step: L.advanceInOrder(0, 1, 0, () => true), bad: bad() });\n",
    )
    assert out == {"s": [], "r": [], "step": None, "bad": "raised"}


def advance_order_js(scope: str, rate: int, frame_ms: float, skip: bool, loops: float) -> str:
    """Play an order with the clock of the page (playbackAdvance, then advanceInOrder) on a
    stream where every volume is loaded, from the last place, and return the volumes painted."""
    return (
        f"const order = L.playOrder(input, '{scope}', 5);\n"
        "const pos = L.positions(order);\n"
        f"let g = order[order.length - 1]; let due = 0; const painted = [];\n"
        f"const want = {loops} * order.length;\n"
        f"for (let now = 0; painted.length < want && now < 60000; now += {frame_ms}) {{\n"
        f"  const s = L.playbackAdvance(due, now, {rate}, {str(skip).lower()});\n"
        "  if (s.n === 0) continue;\n"
        "  const a = L.advanceInOrder(pos[g], s.n, order.length, () => true);\n"
        "  g = order[a.pos]; due = s.nextDue; painted.push(g);\n"
        "}\noutput({ order: Array.from(order), painted });\n"
    )


@needs_node
@pytest.mark.parametrize("scope", ["stream", "run"])
def test_the_clock_paints_the_order_in_force_every_volume_once_per_loop(
    js_copy: Path, scope: str
) -> None:
    timeline = toy_timeline([30, 50, 20])
    for rate, frame in ((12, 16.7), (60, 16.7), (60, 40.0)):
        out = run_js(js_copy, advance_order_js(scope, rate, frame, False, 2), timeline)
        order = out["order"]
        assert out["painted"] == (order + order)[: len(out["painted"])], (rate, frame)
        assert len(out["painted"]) == 2 * len(order)


@needs_node
@pytest.mark.parametrize("scope", ["stream", "run"])
def test_with_skipping_the_painted_volumes_are_a_subsequence_of_the_order(
    js_copy: Path, scope: str
) -> None:
    timeline = toy_timeline([30, 50, 20])
    out = run_js(js_copy, advance_order_js(scope, 300, 16.7, True, 3), timeline)
    order = out["order"]
    place = {g: p for p, g in enumerate(order)}
    places = [place[g] for g in out["painted"]]
    loops = 0
    for a, b in itertools.pairwise(places):
        if b <= a:
            assert b == 0, (a, b)  # a loop starts again at the first place, none twice in a loop
            loops += 1
    assert loops >= 2
    assert len(set(out["painted"])) < len(order)  # volumes were left out


@needs_node
def test_a_place_that_is_not_loaded_holds_or_waits(js_copy: Path) -> None:
    out = run_js(
        js_copy,
        "const ready = new Set([3, 4, 5, 6]);\n"
        "const r = (q) => ready.has(q);\n"
        "output([L.advanceInOrder(3, 3, 10, r), L.advanceInOrder(3, 1, 10, r),\n"
        "  L.advanceInOrder(6, 1, 10, r), L.advanceInOrder(4, 5, 10, r),\n"
        "  L.advanceInOrder(8, 5, 10, () => true), L.advanceInOrder(0, 3, 1, () => true)]);\n",
    )
    assert out[0] == {"ahead": 3, "pos": 6, "held": False}
    assert out[1] == {"ahead": 1, "pos": 4, "held": False}
    assert out[2] is None  # the next place is not loaded: the clock waits
    assert out[3] == {
        "ahead": 2,
        "pos": 6,
        "held": True,
    }  # the latest loaded place before the due one
    assert out[4] == {
        "ahead": 2,
        "pos": 0,
        "held": False,
    }  # past the end: place 0, not carried over
    assert out[5] == {"ahead": 0, "pos": 0, "held": False}  # one volume


@needs_node
def test_the_chunks_are_wanted_in_the_order_the_coming_volumes_need_them(
    js_copy: Path, shape: dict[str, Any]
) -> None:
    out = run_js(
        js_copy,
        "const t = L.buildTimeline(input.sessions);\n"
        "const chunks = L.streamChunks(t);\n"
        "const chunkOf = [];\n"
        "for (let g = 0; g < t.total; g++) chunkOf.push(L.chunkIndexOf(chunks, g));\n"
        "const order = L.playOrder(t, 'stream', 3);\n"
        "const r = [];\n"
        "for (const p of [0, 7, t.total - 1]) {\n"
        "  const c = L.chunksInOrder(order, chunkOf, chunks.length, p);\n"
        "  const sizes = chunks.map((x) => x.count * 1000);\n"
        "  const d = L.desiredChunks({ sizes, cursor: c[0], streams: ['raw'],\n"
        "    limitBytes: 3 * 14000, order: c });\n"
        "  r.push({ p, c, d });\n"
        "}\n"
        "output({ order: Array.from(order), chunkOf, n: chunks.length, r });\n",
        shape,
    )
    order, chunk_of, n = out["order"], out["chunkOf"], out["n"]
    for case in out["r"]:
        p = case["p"]
        want: list[int] = []
        for k in range(len(order)):
            c = chunk_of[order[(p + k) % len(order)]]
            if c not in want:
                want.append(c)
        assert case["c"] == want and sorted(want) == list(range(n))
        # the chunk on screen first, then that of the next volume of the order, within the limit
        assert case["d"][0] == f"{chunk_of[order[p]]}|raw"
        assert (
            case["d"][1] == f"{chunk_of[order[(p + 1) % len(order)]]}|raw"
            or chunk_of[order[(p + 1) % len(order)]] == chunk_of[order[p]]
        )
        assert [k.split("|")[0] for k in case["d"]] == [str(c) for c in want[: len(case["d"])]]


def test_the_runs_on_the_timeline_are_labelled_by_what_tells_them_apart(js_copy: Path) -> None:
    """The labels of the segments leave out the entities that every run shares, so that the
    ellipsis of a narrow segment does not cut off what differs (design review, 2026-11-24)."""
    cases = [
        ["task-flanker_run-1", "task-flanker_run-2"],
        [
            "ses-retest_task-covertverbgeneration",
            "ses-retest_task-linebisection",
            "ses-test_task-covertverbgeneration",
        ],
        ["ses-1_task-rest_dir-AP_run-1", "ses-1_task-rest_dir-PA_run-1"],
        ["task-rest"],
        ["task-rest", "task-rest_run-2"],
    ]
    out = run_js(js_copy, "output(input.map((names) => L.runLabels(names)));\n", cases)
    assert out == [
        ["run 1", "run 2"],
        ["retest · covertverbgeneration", "retest · linebisection", "test · covertverbgeneration"],
        ["dir AP", "dir PA"],
        ["task-rest"],
        ["task-rest", "run 2"],
    ]


def test_the_track_and_the_traces_share_the_plot_area() -> None:
    """The track of the stream is lined up with the plot area of the traces under it: the two
    values of traces.js are those app.css gives the track."""
    traces = (EPI_JS / "traces.js").read_text(encoding="utf-8")
    css = (EPI_JS.parent.parent / "app.css").read_text(encoding="utf-8")
    for js_name, css_name in (("LEFT", "--trace-left"), ("RIGHT", "--trace-right")):
        js_value = re.search(rf"^const {js_name} = (\d+);$", traces, re.M)
        css_value = re.search(rf"{css_name}: (\d+)px;", css)
        assert js_value and css_value and js_value.group(1) == css_value.group(1), js_name
