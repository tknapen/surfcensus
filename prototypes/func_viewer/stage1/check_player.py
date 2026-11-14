"""Drive the player page in a browser on one pack and measure what it does.

usage: check_player.py PACK MAP DATA OUT [--only PART[,PART...]] [--seconds 10] [--volumes 80]
                       [--mem MB] [--seed 1]

PACK  a pack of pack_run.py           MAP   the table that pack_run.py wrote with --map
DATA  the local files the pack was made from (deriv/, raw/), for the expected values
OUT   a directory for check_player.json and shots/*.png; not in the repository

What "exact" checks, and what it does not. The expected values are computed here from the
NIfTI files, but with the transform reader and the resampler of pipeline/geom.py, which are the
ones the pack was made with (session.py uses the same expression), and with the same rule for
the raw header's affine. So it checks that the pack and the page show the values that the
prototype's own resampling gives: encoding, decoding, windows, layout and display. It cannot
catch an error in reading the transform, in its direction or in placing the raw series, since
such an error would be in both. On the first try of one selected session, whose raw stream lay
about 22 mm from the anatomy, it passed every check, and it passes on a copy of the worked
example packed with a wrong raw affine on purpose (README, stage 1). check_geometry.py is the
independent check of where the raw stream lies.

The parts, all of them unless --only names some:
  node     the logic of the page that needs no page, under Node (player/logic_test.mjs)
  load     three times in a new browser context: ms to the first frame and to everything
           loaded, requests and bytes, decoded memory, resident memory of the renderer
  exact    for --volumes random volumes, three random combinations each of plane, slice and
           stream (raw, preprocessed, difference; a third of them in the side-by-side view): the
           pixels of the canvas against the 8-bit values computed here from the NIfTI files
           with numpy, scipy and pipeline/geom.py (the pack's own resampling, see above); then
           twelve of them once more as the pixels of a screenshot
  stills   the same for the mean and SD stills of every run and stream (reads every volume)
  fps      volumes painted per second at 12, 30 and 60 asked, --seconds each, single stream
           and side by side (and the difference at 60), from the page's log of what it drew
  scrub    jumps of the time slider, keys, slice sliders and the wheel: the volume and slices
           shown are those asked for; ms from the jump to the frame
  markers  played at 60 per second: every marker of a volume above the threshold stays at
           least the stated time, seen by an observer of the page's elements
  shots    screenshots of every view at windows 1400 and 400 px wide

--mem MB is the memory limit the page is opened with (its own default without it).
Exit status 1 when a check fails, 0 when none does; measurements never fail by their size.

The browser is one that is installed: the executable named by SURFCENSUS_BROWSER, else Chrome
(Playwright's channel 'chrome'). It needs the Python package playwright, which is not a core
dependency of surfcensus (the repository has it in the dependency group 'browser').
"""

import io
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent / "pipeline"))  # also under -I
from geom import read_itk_ras, resample

HERE = Path(__file__).resolve().parent
PARTS = ("node", "load", "exact", "stills", "fps", "scrub", "markers", "shots")
DIFF_SCALES = (8, 16, 32, 64, 128)
# the stops of the difference colours, as player/logic.js documents them
ZERO, POS, NEG = (24, 24, 24), ((214, 48, 39), (255, 224, 144)), ((33, 102, 172), (166, 226, 255))


# ---- the values, computed here ----------------------------------------------------------------


def raw_affine(img, reference):
    """The affine of a raw series in the frame of its stored transform, decided by the run's
    BOLD-space reference image (deriv/<stem>_boldref.nii.gz) when DATA has it: the raw header's
    sform-first affine when the reference agrees with it, else its qform when that agrees, else
    the reference's own. Written here apart from session.raw_affine, which follows the same rule,
    so that the check reads the files itself; on the worked example both give img.affine."""
    if not reference.is_file():
        return img.affine
    ref = nib.load(reference)
    if tuple(ref.shape[:3]) != tuple(img.shape[:3]):
        return img.affine
    for candidate in (img.affine, img.header.get_qform()):
        if np.allclose(ref.affine, candidate, atol=1e-4):
            return candidate
    return ref.affine


def q8(v, hi):
    """value = clip(floor(v * 255 / hi + 0.5), 0, 255) in float32 (the manifest's rule)."""
    x = v.astype(np.float32) * np.float32(255.0 / hi) + np.float32(0.5)
    return np.clip(np.floor(x), 0, 255).astype(np.uint8)


def diff_rgb(d, scale):
    """The colours of a signed difference, by the rule of player/logic.js, for an int array."""
    d = np.asarray(d, np.int64)
    a = np.minimum(np.abs(d), scale)
    first = 2 * a <= scale
    num = np.where(first, 2 * a, 2 * a - scale)
    out = np.empty((*d.shape, 3), np.int64)
    for k in range(3):
        mid = np.where(d > 0, POS[0][k], NEG[0][k])
        end = np.where(d > 0, POS[1][k], NEG[1][k])
        c0 = np.where(first, ZERO[k], mid)
        c1 = np.where(first, mid, end)
        colour = c0 + ((c1 - c0) * num * 2 + scale) // (2 * scale)
        out[..., k] = np.where(d == 0, ZERO[k], colour)
    return out.astype(np.uint8)


def panel(vol, plane, index):
    """A slice as the page lays it out (rows from the top): sagittal with posterior at the
    left, coronal and axial with the subject's right at the left; superior or anterior up."""
    if plane == 0:
        return vol[index].T[::-1, :]
    if plane == 1:
        return vol[:, index, :].T[::-1, ::-1]
    return vol[:, :, index].T[::-1, ::-1]


class Source:
    """The NIfTI files of the runs of a pack, and their 8-bit volumes by the pack's rules."""

    def __init__(self, data, map_path, manifest):
        nib.arrayproxy.KEEP_FILE_OPEN_DEFAULT = True
        data = Path(data)
        rows = [line.split("\t") for line in Path(map_path).read_text().splitlines()[1:]]
        rows = {r[1]: r for r in rows if r[1].startswith("R")}
        self.runs = []
        anchor = next(r for r in rows.values() if r[2] == "anchor")
        self.anchor_xfm = read_itk_ras(
            data / "deriv" / f"{anchor[3]}_from-scanner_to-T1w_mode-image_xfm.txt"
        )
        for entry in manifest["runs"]:
            stem, raw = rows[entry["id"]][3], rows[entry["id"]][4]
            pre = nib.load(data / "deriv" / f"{stem}_space-T1w_desc-preproc_bold.nii.gz")
            raw_img = nib.load(data / "raw" / raw)
            mask = nib.load(data / "deriv" / f"{stem}_space-T1w_desc-brain_mask.nii.gz")
            self.runs.append(
                {
                    "pre": pre,
                    "raw": raw_img,
                    "raw_affine": raw_affine(raw_img, data / "deriv" / f"{stem}_boldref.nii.gz"),
                    "mask": np.asarray(mask.dataobj) > 0,
                    "hi": None,
                    "n": pre.shape[3],
                }
            )
        self.shape, self.affine = self.runs[0]["pre"].shape[:3], self.runs[0]["pre"].affine

    @staticmethod
    def _volume(img, t):
        """Volume t as pipeline/make_videos.py scales it (stored values as float32, times the
        slope, plus the intercept). Only that volume is read: get_unscaled() would read the whole
        series at every call, which for a long run is gigabytes."""
        proxy = img.dataobj
        stored = proxy._get_unscaled((slice(None), slice(None), slice(None), int(t)))
        return np.asarray(stored).astype(np.float32) * float(proxy.slope) + float(proxy.inter)

    def window(self, run):
        """0 to the 99.5th percentile of the preprocessed series in the mask, volumes 0, 10, ..."""
        r = self.runs[run]
        if r["hi"] is None:
            sub = [self._volume(r["pre"], t)[r["mask"]] for t in range(0, r["n"], 10)]
            r["hi"] = float(np.percentile(np.stack(sub), 99.5))
        return r["hi"]

    def floats(self, run, stream, t):
        r = self.runs[run]
        if stream == "pre":
            return self._volume(r["pre"], t)
        vol = self._volume(r["raw"], t)
        return resample(vol, r["raw_affine"], self.shape, self.affine, self.anchor_xfm, order=3)

    def volume(self, run, stream, t):
        return q8(self.floats(run, stream, t), self.window(run))

    def stills(self, run, non_steady):
        """8-bit mean and SD of both streams of a run, and the SD window, from every volume."""
        r = self.runs[run]
        keep = [t for t in range(r["n"]) if t not in set(non_steady)]
        out = {}
        for stream in ("raw", "pre"):
            total = np.zeros(self.shape, np.float64)
            squares = np.zeros(self.shape, np.float64)
            for t in keep:
                v = self.floats(run, stream, t).astype(np.float64)
                total += v
                squares += v * v
            mean = total / len(keep)
            var = np.maximum(squares - total**2 / len(keep), 0.0) / (len(keep) - 1)
            out[stream] = (mean, np.sqrt(var))
        sd_hi = float(np.percentile(out["pre"][1][r["mask"]], 99.5))
        return {
            s: {"mean": q8(m, self.window(run)), "sd": q8(sd, sd_hi)} for s, (m, sd) in out.items()
        }, sd_hi


# ---- the browser ------------------------------------------------------------------------------


def launch(playwright):
    path = os.environ.get("SURFCENSUS_BROWSER")
    if path and not Path(path).is_file():
        sys.exit(f"SURFCENSUS_BROWSER names {path}, which is not a file")
    options = {"executable_path": path} if path else {"channel": "chrome"}
    return playwright.chromium.launch(**options), (
        "SURFCENSUS_BROWSER" if path else "channel chrome"
    )


def wait_for(page, expression, timeout=120.0):
    """Poll an expression of the page until it is true (the page's policy forbids eval, which
    Playwright's own wait uses)."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if page.evaluate(expression):
            return time.perf_counter() - t0
        time.sleep(0.02)
    raise TimeoutError(f"the page did not reach: {expression}")


def renderer_rss(browser):
    """Resident memory (bytes) of the largest renderer process of this browser, from /proc;
    None where the process list or /proc cannot be read (not Linux)."""
    try:
        session = browser.new_browser_cdp_session()
        processes = session.send("SystemInfo.getProcessInfo")["processInfo"]
        session.detach()
    except Exception:  # the browser does not answer this over its debugging protocol
        return None
    best = None
    for process in processes:
        if process.get("type") != "renderer":
            continue
        try:
            text = Path(f"/proc/{process['id']}/status").read_text()
        except OSError:
            continue
        for line in text.splitlines():
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) * 1024
                best = rss if best is None else max(best, rss)
    return best


def canvas_pixels(page, group, plane):
    data = page.evaluate(
        """([group, plane]) => {
            const where = `.group[data-group="${group}"] .panel[data-plane="${plane}"]`;
            const c = document.querySelector(`${where} canvas.gray`);
            const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
            return [c.width, c.height, Array.from(d)];
        }""",
        [group, plane],
    )
    w, h, values = data
    return np.array(values, np.uint8).reshape(h, w, 4)


def expected_rgba(source, run, stream, t, plane, index, scale, cache):
    for s in ("raw", "pre"):
        if (run, s, t) not in cache and stream in (s, "diff"):
            cache[run, s, t] = source.volume(run, s, t)
    if stream == "diff":
        a = panel(cache[run, "raw", t], plane, index).astype(np.int64)
        b = panel(cache[run, "pre", t], plane, index).astype(np.int64)
        rgb = diff_rgb(a - b, scale)
    else:
        rgb = np.repeat(panel(cache[run, stream, t], plane, index)[..., None], 3, axis=2)
    return np.concatenate([rgb, np.full((*rgb.shape[:2], 1), 255, np.uint8)], axis=2)


class Check:
    def __init__(self):
        self.failed = []
        self.results = {}

    def expect(self, ok, text):
        print(("ok      " if ok else "FAILED  ") + text, flush=True)
        if not ok:
            self.failed.append(text)

    def note(self, text):
        print("        " + text, flush=True)


# ---- the parts --------------------------------------------------------------------------------


def part_node(check):
    node = shutil.which("node")
    if not node:
        check.note("node is not installed: the logic checks under Node were not run")
        check.results["node"] = None
        return
    run = subprocess.run(
        [node, str(HERE / "player" / "logic_test.mjs")], capture_output=True, text=True, check=False
    )
    lines = run.stdout.strip().splitlines()
    n_ok = sum(line.startswith("ok") for line in lines)
    check.expect(run.returncode == 0, f"node: {n_ok} of {len(lines) - 1} logic checks hold")
    for line in lines:
        if line.startswith("FAILED"):
            check.note(line)
    check.results["node"] = {"ok": n_ok, "checks": len(lines) - 1, "status": run.returncode}


def open_page(browser, url, width=1400, height=900, scale=1):
    context = browser.new_context(
        viewport={"width": width, "height": height}, device_scale_factor=scale
    )
    page = context.new_page()
    seen = {"problems": [], "finished": []}
    page.on("pageerror", lambda e: seen["problems"].append(f"page error: {e}"))
    page.on(
        "console", lambda m: m.type == "error" and seen["problems"].append(f"console: {m.text}")
    )
    page.on("requestfailed", lambda r: seen["problems"].append(f"request failed: {r.url}"))
    page.on("requestfinished", lambda r: seen["finished"].append(r))
    page.goto(url)
    wait_for(page, "!!(window.__player && window.__player.shown)")
    return context, page, seen


def pack_sizes(manifest):
    """Bytes of the pack by part: per stream the chunks and the stills, the contours, in all."""
    out = {"raw": 0, "pre": 0, "stills": 0, "contours": 0}
    for run in manifest["runs"]:
        for stream in ("raw", "pre"):
            entry = run["streams"][stream]
            out[stream] += sum(c["bytes"] for c in entry["chunks"])
            out["stills"] += sum(entry[k]["bytes"] for k in ("ref", "mean", "sd"))
    if manifest["contours"]:
        out["contours"] = sum(p["bytes"] for p in manifest["contours"]["planes"].values())
    shape = manifest["grid"]["shape"]
    volumes = sum(run["n_volumes"] for run in manifest["runs"])
    out["volumes"] = volumes
    out["decoded_series"] = 2 * volumes * shape[0] * shape[1] * shape[2]
    out["all"] = out["raw"] + out["pre"] + out["stills"] + out["contours"]
    return out


def part_load(check, browser, url, manifest):
    sizes = pack_sizes(manifest)
    check.note(
        f"pack: {sizes['volumes']} volumes per stream; raw {sizes['raw'] / 1e6:.2f} MB,"
        f" preprocessed {sizes['pre'] / 1e6:.2f} MB, stills and reference volumes"
        f" {sizes['stills'] / 1e6:.2f} MB,"
        f" contours {sizes['contours'] / 1e6:.2f} MB, {sizes['all'] / 1e6:.2f} MB in all (the"
        f" manifest's sizes); both series decoded: {sizes['decoded_series'] / 1e6:.0f} MB"
    )
    rows = []
    for _ in range(3):
        t0 = time.perf_counter()
        context, page, seen = open_page(browser, url)
        wall_first = time.perf_counter() - t0
        wait_for(page, "window.__player.allLoaded()", 600)
        wall_all = time.perf_counter() - t0
        timing = page.evaluate("window.__player.page.timing")
        stats = page.evaluate("window.__player.stats()")
        sizes = [r.sizes() for r in seen["finished"]]
        rows.append(
            {
                "first_frame_ms": timing["firstFrame"],
                "contours_ms": timing["contours"],
                "all_loaded_ms": timing["allLoaded"],
                "wall_first_s": wall_first,
                "wall_all_s": wall_all,
                "requests": len(sizes),
                "body_bytes": sum(s["responseBodySize"] for s in sizes),
                "header_bytes": sum(s["responseHeadersSize"] for s in sizes),
                "page_counted_requests": stats["requests"] + stats["pageRequests"],
                "page_counted_bytes": stats["fetchedBytes"] + stats["pageFetchedBytes"],
                "decoded_chunk_bytes": stats["decodedChunkBytes"],
                "still_bytes": stats["stillBytes"],
                "chunks_held": stats["chunksHeld"],
                "limit_bytes": stats["limitBytes"],
                "workers": stats["workers"],
                "renderer_rss": renderer_rss(browser),
                "problems": list(seen["problems"]),
            }
        )
        context.close()

    def spread(key):
        values = [r[key] for r in rows if r[key] is not None]
        return f"{statistics.median(values):.0f} ({min(values):.0f} to {max(values):.0f})"

    r = rows[-1]
    check.expect(
        not any(row["problems"] for row in rows),
        "load: no page error, console error or failed request",
    )
    check.expect(
        all(row["decoded_chunk_bytes"] <= row["limit_bytes"] for row in rows),
        f"load: decoded chunks {r['decoded_chunk_bytes'] / 1e6:.1f} MB within the limit of"
        f" {r['limit_bytes'] / 1e6:.0f} MB; stills {r['still_bytes'] / 1e6:.1f} MB",
    )
    check.note(
        f"ms from navigation, median (range) of 3: first frame {spread('first_frame_ms')},"
        f" contours {spread('contours_ms')}, everything wanted loaded {spread('all_loaded_ms')}"
    )
    check.note(
        f"{r['requests']} requests, {r['body_bytes'] / 1e6:.2f} MB of bodies and"
        f" {r['header_bytes'] / 1e3:.1f} kB of headers (the page counted"
        f" {r['page_counted_requests']} fetches and {r['page_counted_bytes'] / 1e6:.2f} MB);"
        f" {r['chunks_held']} chunks held by {r['workers']} workers"
    )
    if r["renderer_rss"]:
        rss = [row["renderer_rss"] / 1e6 for row in rows if row["renderer_rss"]]
        check.note(
            f"resident memory of the renderer process: {statistics.median(rss):.0f} MB"
            f" ({min(rss):.0f} to {max(rss):.0f})"
        )
    check.results["load"] = {"pack": sizes, "opens": rows}


def part_exact(check, page, source, manifest, rng, n_volumes):
    hi = [source.window(k) for k in range(len(source.runs))]
    same = all(
        abs(h - run["window"][1]) <= 1e-9 * h for h, run in zip(hi, manifest["runs"], strict=True)
    )
    check.expect(same, "exact: the window of every run, computed here, is the manifest's")
    picks = []
    for _ in range(n_volumes):
        run = rng.randrange(len(source.runs))
        t = rng.randrange(source.runs[run]["n"])
        for _ in range(3):
            plane = rng.randrange(3)
            stream = rng.choice(("raw", "pre", "diff"))
            view = "side" if stream != "diff" and rng.random() < 1 / 3 else stream
            picks.append(
                (
                    run,
                    t,
                    plane,
                    rng.randrange(source.shape[plane]),
                    stream,
                    view,
                    rng.choice(DIFF_SCALES),
                )
            )
    picks.sort(key=lambda p: (p[0], p[1]))  # volumes in order: the series are read front to back
    wrong, cache, count = [], {}, {"raw": 0, "pre": 0, "diff": 0}
    for run, t, plane, index, stream, view, scale in picks:
        cache = {k: v for k, v in cache.items() if k[0] == run and k[2] == t}
        slices = [rng.randrange(n) for n in source.shape]
        slices[plane] = index
        page.evaluate(
            "(c) => window.__player.set(c)",
            {
                "run": run,
                "t": t,
                "view": view,
                "kind": "series",
                "slices": slices,
                "diffScale": scale,
            },
        )
        got = canvas_pixels(page, stream, plane)
        want = expected_rgba(source, run, stream, t, plane, index, scale, cache)
        count[stream] += 1
        if got.shape != want.shape or not np.array_equal(got, want):
            wrong.append((run, t, plane, index, stream, view))
    check.expect(
        not wrong,
        f"exact: {len(picks)} combinations of volume, plane, slice and stream ({count['raw']} raw,"
        f" {count['pre']} preprocessed, {count['diff']} difference): every canvas pixel is the"
        f" value computed from the NIfTI files; {len(wrong)} differ",
    )
    # the same on screen: a screenshot of the canvas element, one sample at the centre of each voxel
    hide = "document.querySelectorAll('.letter').forEach((e) => { e.style.visibility = '%s'; })"
    off, samples = 0, 0
    for run, t, plane, index, stream, _view, scale in picks[:: max(1, len(picks) // 12)][:12]:
        slices = [rng.randrange(n) for n in source.shape]
        slices[plane] = index
        change = {"run": run, "t": t, "view": stream, "slices": slices, "diffScale": scale}
        page.evaluate(
            "(c) => window.__player.set({ ...c, contours: false, crosshair: false })", change
        )
        page.evaluate(hide % "hidden")  # the edge letters lie over the voxels
        box = page.locator(
            f'.group[data-group="{stream}"] .panel[data-plane="{plane}"] canvas.gray'
        )
        rect = box.bounding_box()
        shot = np.array(Image.open(io.BytesIO(box.screenshot())).convert("RGB"))
        want = expected_rgba(source, run, stream, t, plane, index, scale, {})[..., :3]
        # the screenshot starts at the whole pixel in which the canvas starts
        y0, x0 = rect["y"] - np.floor(rect["y"]), rect["x"] - np.floor(rect["x"])
        rows = (y0 + (np.arange(want.shape[0]) + 0.5) * rect["height"] / want.shape[0]).astype(int)
        cols = (x0 + (np.arange(want.shape[1]) + 0.5) * rect["width"] / want.shape[1]).astype(int)
        seen = shot[np.ix_(rows, cols)]
        off += int((seen != want).any(axis=2).sum())
        samples += seen.shape[0] * seen.shape[1]
    page.evaluate(hide % "")
    page.evaluate("window.__player.set({ contours: true, crosshair: true })")
    check.expect(
        off == 0,
        f"exact: on screenshots of 12 of them (contours and crosshair off), the pixel at the"
        f" centre of every voxel is that value: {off} of {samples} differ",
    )
    check.results["exact"] = {
        "combinations": len(picks),
        "wrong": wrong,
        "screen_samples": samples,
        "screen_off": off,
    }


def part_stills(check, page, source, manifest, rng):
    wrong, n = [], 0
    for run, entry in enumerate(manifest["runs"]):
        stills, sd_hi = source.stills(run, entry["non_steady"])
        if abs(sd_hi - entry["sd_window"][1]) > 1e-6 * sd_hi:
            wrong.append((run, "sd window"))
        for kind in ("mean", "sd"):
            for stream in ("raw", "pre", "diff"):
                for plane in range(3):
                    index = rng.randrange(source.shape[plane])
                    slices = [rng.randrange(k) for k in source.shape]
                    slices[plane] = index
                    change = {
                        "run": run,
                        "view": stream,
                        "kind": kind,
                        "slices": slices,
                        "diffScale": 32,
                    }
                    page.evaluate("(c) => window.__player.set(c)", change)
                    got = canvas_pixels(page, stream, plane)
                    if stream == "diff":
                        a = panel(stills["raw"][kind], plane, index).astype(np.int64)
                        b = panel(stills["pre"][kind], plane, index).astype(np.int64)
                        rgb = diff_rgb(a - b, 32)
                    else:
                        rgb = np.repeat(
                            panel(stills[stream][kind], plane, index)[..., None], 3, axis=2
                        )
                    n += 1
                    if not np.array_equal(got[..., :3], rgb) or not (got[..., 3] == 255).all():
                        wrong.append((run, kind, stream, plane, index))
    page.evaluate("window.__player.set({ kind: 'series', view: 'raw' })")
    check.expect(
        not wrong,
        f"stills: {n} slices of the mean and SD stills and their differences, every run: the"
        f" canvas is the still computed here from every volume; {len(wrong)} differ",
    )
    check.results["stills"] = {"slices": n, "wrong": wrong}


def play(page, change, seconds):
    """Play for some seconds; returns the page's log of painted volumes and counters."""
    page.evaluate("(c) => window.__player.set(c)", {**change, "playing": False})
    wait_for(page, "window.__player.idle()", 600)
    page.evaluate(
        "() => { const p = window.__player.page; p.log.length = 0; p.stalls = 0; p.resyncs = 0; }"
    )
    page.evaluate("window.__player.set({ playing: true })")
    peak = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds + 0.3:
        time.sleep(0.5)
        peak = max(peak, page.evaluate("window.__player.stats().decodedChunkBytes"))
    out = page.evaluate(
        """() => {
            const p = window.__player.page;
            const log = p.log.filter((e) => e.reason === 'play').map((e) => [e.ms, e.t]);
            return { log, stalls: p.stalls, resyncs: p.resyncs };
        }"""
    )
    page.evaluate("window.__player.set({ playing: false })")
    stats = page.evaluate("window.__player.stats()")
    out["peak_decoded"] = max(peak, stats["peakDecoded"])
    out["limit"] = stats["limitBytes"]
    out["evictions"] = stats["evictions"]
    return out


def part_fps(check, page, manifest, seconds):
    rows = []
    n = manifest["runs"][0]["n_volumes"]
    cases = [(view, rate) for view in ("raw", "side") for rate in (12, 30, 60)] + [("diff", 60)]
    for view, rate in cases:
        got = play(page, {"run": 0, "t": 0, "view": view, "rate": rate, "kind": "series"}, seconds)
        log = [e for e in got["log"] if e[0] - got["log"][0][0] <= seconds * 1000]
        ms = np.array([e[0] for e in log])
        vols = np.array([e[1] for e in log])
        steps = (np.diff(vols) + n) % n
        gaps = np.diff(ms)
        row = {
            "view": view,
            "asked": rate,
            "painted": len(log),
            "seconds": float(ms[-1] - ms[0]) / 1000,
            "per_second": float((len(log) - 1) * 1000 / (ms[-1] - ms[0])),
            "repeated": int((steps == 0).sum()),
            "dropped": int(np.clip(steps - 1, 0, None).sum()),
            "gap_median_ms": float(np.median(gaps)),
            "gap_p95_ms": float(np.percentile(gaps, 95)),
            "gap_max_ms": float(gaps.max()),
            "stalls": got["stalls"],
            "resyncs": got["resyncs"],
            "peak_decoded_bytes": got["peak_decoded"],
            "limit_bytes": got["limit"],
            "evictions": got["evictions"],
        }
        rows.append(row)
        check.note(
            f"fps {view:4s} asked {rate:2d}: painted {row['per_second']:.2f} per s over"
            f" {row['seconds']:.1f} s ({row['painted']} volumes); {row['repeated']} repeated,"
            f" {row['dropped']} dropped; gap median {row['gap_median_ms']:.1f}, p95"
            f" {row['gap_p95_ms']:.1f}, max {row['gap_max_ms']:.1f} ms; {row['stalls']} stalls,"
            f" {row['resyncs']} resyncs; peak {row['peak_decoded_bytes'] / 1e6:.0f} MB decoded"
        )
    check.expect(
        all(r["repeated"] == 0 and r["dropped"] == 0 for r in rows),
        "fps: no volume was painted twice in a row and none was skipped, at any rate and view",
    )
    limit = rows[-1]["limit_bytes"]
    check.expect(
        all(r["peak_decoded_bytes"] <= limit for r in rows),
        f"fps: the decoded chunks never exceeded the limit of {limit / 1e6:.0f} MB (largest"
        f" {max(r['peak_decoded_bytes'] for r in rows) / 1e6:.0f} MB, as the page counts it"
        f" whenever a chunk is kept; {rows[-1]['evictions']} chunks dropped by then)",
    )
    check.results["fps"] = rows


def part_scrub(check, page, manifest, rng):
    n = manifest["runs"][0]["n_volumes"]
    shape = manifest["grid"]["shape"]
    page.evaluate("window.__player.set({ run: 0, t: 0, view: 'side', kind: 'series' })")
    wait_for(page, "window.__player.idle()", 600)
    jump = """(t) => {
        const input = document.getElementById('time');
        const had = window.__player.has(0, 'side', t);
        input.value = String(t);
        input.dispatchEvent(new Event('input', { bubbles: true }));
        return window.__player.ready().then((shown) => {
            const ms = window.__player.page.lastSeek.ms;
            return { shown: shown.t, had, ms, slider: Number(input.value) };
        });
    }"""
    rows = []
    for _ in range(40):
        t = rng.randrange(n)
        got = page.evaluate(jump, t)
        rows.append({"asked": t, **got})
    bad = [r for r in rows if r["shown"] != r["asked"]]
    check.expect(
        not bad,
        f"scrub: after 40 jumps of the time slider the volume shown is the one asked for;"
        f" {len(bad)} are not",
    )
    for label, had in (("decoded in memory", True), ("not in memory", False)):
        ms = [r["ms"] for r in rows if r["had"] is had]
        if ms:
            check.note(
                f"jump to a volume {label} ({len(ms)}): median {statistics.median(ms):.1f} ms,"
                f" largest {max(ms):.1f} ms from the slider's event to the painted frame"
            )
    # a real click on the slider, then keys
    box = page.locator("#time").bounding_box()
    clicks = []
    for _ in range(10):
        f = rng.random()
        page.mouse.click(box["x"] + f * box["width"], box["y"] + box["height"] / 2)
        page.evaluate("window.__player.ready()")
        got = page.evaluate(
            "[Number(document.getElementById('time').value), window.__player.shown.t]"
        )
        clicks.append((f, *got))
    ok = all(v == s and abs(v - f * (n - 1)) <= max(2.0, 0.02 * n) for f, v, s in clicks)
    check.expect(
        ok,
        "scrub: 10 clicks on the time slider show the volume the slider then holds, at the place"
        " clicked",
    )
    page.evaluate("window.__player.set({ t: 5 })")
    page.keyboard.press("ArrowRight")
    page.keyboard.press("ArrowRight")
    page.keyboard.press("Shift+ArrowRight")
    page.keyboard.press("ArrowLeft")
    page.evaluate("window.__player.ready()")
    t_keys = page.evaluate("window.__player.shown.t")
    page.keyboard.press("End")
    page.evaluate("window.__player.ready()")
    t_end = page.evaluate("window.__player.shown.t")
    check.expect(
        t_keys == 5 + 1 + 1 + 10 - 1 and t_end == n - 1,
        f"scrub: arrow keys step 1 and 10 volumes, End goes to the last (volume {t_keys}, then"
        f" {t_end})",
    )
    # slices: the sliders, the wheel over a panel, a click in a panel
    asked = [rng.randrange(k) for k in shape]
    page.evaluate(
        """(asked) => {
            document.querySelectorAll('#slices input').forEach((input, k) => {
                input.value = String(asked[k]);
                input.dispatchEvent(new Event('input', { bubbles: true }));
            });
            return window.__player.ready();
        }""",
        asked,
    )
    got = page.evaluate("window.__player.shown.slices")
    check.expect(
        got == asked, f"scrub: the three slice sliders show the slices asked for ({asked})"
    )
    panel_box = page.locator('.group[data-group="raw"] .panel[data-plane="2"]').bounding_box()
    page.mouse.move(
        panel_box["x"] + panel_box["width"] / 2, panel_box["y"] + panel_box["height"] / 2
    )
    before = page.evaluate("window.__player.shown.slices[2]")
    page.mouse.wheel(0, -100)
    page.mouse.wheel(0, -100)
    time.sleep(0.2)
    page.evaluate("window.__player.ready()")
    after = page.evaluate("window.__player.shown.slices[2]")
    want = min(before + 2, shape[2] - 1)
    check.expect(
        after == want,
        f"scrub: two wheel steps over the axial panel move two axial slices ({before} to {after})",
    )
    # a click at a quarter from the left and top of the axial panel: R at the left, A at the top
    page.mouse.click(
        panel_box["x"] + panel_box["width"] * 0.25, panel_box["y"] + panel_box["height"] * 0.25
    )
    page.evaluate("window.__player.ready()")
    got = page.evaluate("window.__player.shown.slices")
    want_i = shape[0] - 1 - int(0.25 * shape[0])
    want_j = shape[1] - 1 - int(0.25 * shape[1])
    check.expect(
        abs(got[0] - want_i) <= 1 and abs(got[1] - want_j) <= 1,
        f"scrub: a click in the axial panel sends the sagittal and coronal planes there"
        f" (i {got[0]}, j {got[1]})",
    )
    check.results["scrub"] = {"jumps": rows, "clicks": clicks}


def part_markers(check, page, manifest):
    hold = float(page.evaluate("document.getElementById('help-hold').textContent"))
    run = max(
        range(len(manifest["runs"])),
        key=lambda k: sum(
            1 for v in manifest["runs"][k]["fd"] if v and v > manifest["fd_threshold_mm"]
        ),
    )
    entry = manifest["runs"][run]
    above = [
        t for t, v in enumerate(entry["fd"]) if v is not None and v > manifest["fd_threshold_mm"]
    ]
    if not above:
        check.note("markers: no volume of this pack is above the threshold; nothing to measure")
        check.results["markers"] = None
        return
    # paused on a marked volume: every group carries the marker, also after the view changes,
    # and it leaves after the stated time once another volume is on screen
    carried = (
        "Array.from(document.querySelectorAll('.group'), (g) => g.classList.contains('marker-fd'))"
    )
    change = {"run": run, "t": above[0], "view": "raw", "kind": "series"}
    page.evaluate("(c) => window.__player.set(c)", change)
    one = page.evaluate(carried)
    page.evaluate("window.__player.set({ view: 'side' })")
    two = page.evaluate(carried)
    marked = {*above, *entry["non_steady"]}
    quiet = next(t for t in range(entry["n_volumes"]) if t not in marked)
    page.evaluate("(t) => window.__player.set({ t })", quiet)
    time.sleep(hold / 1000 + 0.25)
    gone = page.evaluate(carried)
    check.expect(
        one == [True] and two == [True, True] and gone == [False, False],
        "markers: a marked volume carries the marker in every group, also after a change of"
        " view, and the marker leaves once another volume is on screen",
    )
    page.evaluate(
        "(c) => window.__player.set(c)",
        {"run": run, "t": 0, "view": "raw", "kind": "series", "rate": 60},
    )
    wait_for(page, "window.__player.idle()", 600)
    page.evaluate(
        """() => {
            window.__seen = [];
            const group = document.querySelector('.group');
            let on = group.classList.contains('marker-fd');
            new MutationObserver(() => {
                const now = group.classList.contains('marker-fd');
                if (now !== on) { on = now; window.__seen.push([performance.now(), now]); }
            }).observe(group, { attributes: true, attributeFilter: ['class'] });
        }"""
    )
    seconds = min(30.0, max(8.0, 2.2 * entry["n_volumes"] / 60))
    page.evaluate("window.__player.set({ playing: true })")
    time.sleep(seconds)
    page.evaluate("window.__player.set({ playing: false, t: 0 })")
    time.sleep(hold / 1000 + 0.3)
    seen = page.evaluate("window.__seen")
    spans = [b[0] - a[0] for a, b in zip(seen[::2], seen[1::2], strict=False) if a[1] and not b[1]]
    check.expect(
        bool(spans) and min(spans) >= hold - 1,
        f"markers: played at 60 per s for {seconds:.0f} s, the frame carried the marker"
        f" {len(spans)} times, for {min(spans, default=0):.0f} to {max(spans, default=0):.0f} ms"
        f" each (stated minimum {hold:.0f} ms; {len(above)} volumes of {entry['n_volumes']} are"
        " above the threshold)",
    )
    check.results["markers"] = {
        "spans_ms": spans,
        "hold_ms": hold,
        "above": above,
        "seconds": seconds,
    }


def part_shots(check, browser, url, manifest, out):
    shots = Path(out) / "shots"
    shots.mkdir(parents=True, exist_ok=True)
    last = len(manifest["runs"]) - 1
    marked = next(
        (
            t
            for t, v in enumerate(manifest["runs"][0]["fd"])
            if v is not None and v > manifest["fd_threshold_mm"]
        ),
        0,
    )
    views = [
        ("raw", {"run": 0, "t": marked, "view": "raw", "kind": "series"}),
        ("pre", {"run": 0, "t": marked, "view": "pre", "kind": "series"}),
        ("side", {"run": 0, "t": marked, "view": "side", "kind": "series"}),
        ("diff", {"run": last, "t": marked, "view": "diff", "kind": "series"}),
        ("mean_side", {"run": last, "view": "side", "kind": "mean"}),
        ("mean_diff", {"run": last, "view": "diff", "kind": "mean"}),
        ("sd_side", {"run": last, "view": "side", "kind": "sd"}),
        ("sd_raw", {"run": 0, "view": "raw", "kind": "sd"}),
    ]
    made = []
    for width, height in ((1400, 900), (400, 800)):
        context, page, seen = open_page(browser, url, width, height)
        wait_for(page, "window.__player.allLoaded()", 600)
        for name, change in views:
            page.evaluate("(c) => window.__player.set(c)", change)
            time.sleep(0.15)
            path = shots / f"{name}_{width}.png"
            page.screenshot(path=str(path), full_page=True)
            made.append(path.name)
        page.evaluate("window.__player.set({ kind: 'series', view: 'raw' })")
        page.keyboard.press("s")
        page.keyboard.press("c")
        time.sleep(0.15)
        page.screenshot(path=str(shots / f"raw_bare_{width}.png"), full_page=True)
        page.keyboard.press("?")
        time.sleep(0.15)
        page.screenshot(path=str(shots / f"help_{width}.png"), full_page=False)
        made += [f"raw_bare_{width}.png", f"help_{width}.png"]
        overflow = page.evaluate(
            "document.documentElement.scrollWidth - document.documentElement.clientWidth"
        )
        check.expect(
            overflow <= 0,
            f"shots: at {width} px wide the page is not wider than the window ({overflow} px over)",
        )
        check.expect(not seen["problems"], f"shots: no page error at {width} px wide")
        context.close()
    check.note(f"{len(made)} screenshots in {shots}")
    check.results["shots"] = made


def main():
    args = sys.argv[1:]
    options = {
        "--only": ",".join(PARTS),
        "--seconds": 10.0,
        "--volumes": 80,
        "--mem": 0,
        "--seed": 1,
    }
    positional = []
    while args:
        arg = args.pop(0)
        if arg in options and args:
            options[arg] = type(options[arg])(args.pop(0))
        elif arg.startswith("--"):
            positional = []
            break
        else:
            positional.append(arg)
    parts = [p for p in options["--only"].split(",") if p]
    if len(positional) != 4 or any(p not in PARTS for p in parts):
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(2)
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.exit("check_player.py needs the Python package playwright (dependency group 'browser')")
    pack, map_path, data, out = positional
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(pack) / "manifest.json").read_text())
    rng = random.Random(options["--seed"])
    check = Check()
    if "node" in parts:
        part_node(check)
    server = subprocess.Popen(
        [sys.executable, "-I", str(HERE / "serve.py"), str(pack), "--port", "0"],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        url = server.stdout.readline().strip()
        if options["--mem"]:
            url += f"?mem={options['--mem']}"
        with sync_playwright() as playwright:
            browser, how = launch(playwright)
            check.results["browser"] = {"version": browser.version, "found_by": how}
            check.note(f"Chromium {browser.version} ({how}), Python Playwright; page at {url}")
            if "load" in parts:
                part_load(check, browser, url, manifest)
            if {"exact", "stills", "fps", "scrub", "markers"} & set(parts):
                context, page, seen = open_page(browser, url)
                source = (
                    Source(data, map_path, manifest) if {"exact", "stills"} & set(parts) else None
                )
                if "exact" in parts:
                    part_exact(check, page, source, manifest, rng, options["--volumes"])
                if "stills" in parts:
                    part_stills(check, page, source, manifest, rng)
                if "fps" in parts:
                    part_fps(check, page, manifest, options["--seconds"])
                if "scrub" in parts:
                    part_scrub(check, page, manifest, rng)
                if "markers" in parts:
                    part_markers(check, page, manifest)
                stats = page.evaluate("window.__player.stats()")
                rss = renderer_rss(browser)
                check.results["end"] = {
                    "stats": stats,
                    "renderer_rss": rss,
                    "problems": seen["problems"],
                }
                check.expect(
                    not seen["problems"] and not stats["errors"],
                    "the page reported no error during the checks",
                )
                per_load = stats["chunkMs"] / max(1, stats["chunkLoads"])
                check.note(
                    f"at the end: {stats['decodedChunkBytes'] / 1e6:.0f} MB of chunks decoded"
                    f" (limit {stats['limitBytes'] / 1e6:.0f} MB), {stats['evictions']} chunks"
                    f" dropped, {stats['chunkLoads']} chunk loads of {per_load:.0f} ms on average"
                    " in a worker" + (f"; renderer {rss / 1e6:.0f} MB resident" if rss else "")
                )
                context.close()
            if "shots" in parts:
                part_shots(check, browser, url, manifest, out)
            browser.close()
    finally:
        server.terminate()
    check.results["failed"] = check.failed
    (out / "check_player.json").write_text(json.dumps(check.results, indent=1, default=str) + "\n")
    print(f"{len(check.failed)} checks failed" if check.failed else "all checks hold")
    sys.exit(1 if check.failed else 0)


if __name__ == "__main__":
    main()
