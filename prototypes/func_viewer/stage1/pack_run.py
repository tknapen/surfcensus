"""Pack the runs of one session for the player page: both streams of every run as 8-bit volumes
that a browser reslices, the stills, the traces, the contours and the frame check.

usage: pack_run.py DATA FS_SUBJECT_DIR OUT --id ID [--session LABEL] [--map FILE]
                   [--chunk N] [--portable] [--no-contours]

DATA             local files of one subject: deriv/ and raw/ (pack_inputs.py says which files)
FS_SUBJECT_DIR   the FreeSurfer subject directory (mri/T1.mgz, surf/?h.white, surf/?h.pial);
                 not read with --no-contours, but an argument must be given
OUT              the pack directory; it must not exist or be empty. Nothing else is written.
--id ID          the neutral id that the pack and the page carry (H01, Q07, ...)
--session LABEL  the session (without ses-) when DATA holds more than one
--map FILE       also write the table that maps the run ids to file names. It names a dataset,
                 so it may not lie inside OUT or inside the repository.
--chunk N        volumes per chunk file (default: as many as fit 4 MiB decoded, at most 32)
--portable       the voxel map of the raw stream without numpy's linear algebra (session.py)
--no-contours    pack no surface contours

Takes local files only and sends no request. A last line of JSON on standard output has the
sizes and the times; the pack itself holds no time, no path and no file name of the input, so
two packs of the same input are the same bytes (pack_digest.py compares two).

THE STREAMS. The display grid is the grid of the preprocessed series (fMRIPrep's T1w space).
'pre' is space-T1w_desc-preproc_bold as stored. 'raw' is the raw BIDS series, every volume
resampled once onto that grid through ONE rigid transform for the whole session: the stored
from-scanner_to-T1w transform of the session's anchor run (session.py: pick_anchor,
session_stream; decision of Tomas Knapen of 2026-11-13 as read there). No motion correction, no
slice-timing correction, no distortion correction. Every run's own stored transform is compared
with the anchor's (session.check_frame) and the result is in the manifest.

THE PACK (format 'func-viewer-pack' version 1). Every .bin file is a zlib stream.

  manifest.json                    everything the page needs but the voxels; the rules as text
  contours/<plane>.bin             pack_contours.py
  runs/<run>/<stream>/ref.bin      the reference volume of the stream (below)
  runs/<run>/<stream>/mean.bin     temporal mean, 8 bit, the window of the run
  runs/<run>/<stream>/sd.bin       temporal SD, 8 bit, the SD window of the run
  runs/<run>/<stream>/cNNNN.bin    chunk N: volumes N * chunk_volumes and following

A volume is nx * ny * nz bytes, the voxel (i, j, k) at i + nx * (j + ny * k). A chunk holds its
volumes one after the other as RESIDUALS: (value - ref) mod 256 per voxel, where ref is the
rounded mean of the 8-bit volumes 0, 10, 20, ... of that run and stream. The page adds ref back
(mod 256), which gives the stored 8-bit values exactly. Residuals compress better than values:
on the worked example 56 kB against 87 kB per volume with zlib, and 74 kB for a lossless WebP
mosaic of the values (measured 2026-11-13).

8 BIT. One window per run for both streams: 0 to the 99.5th percentile of the preprocessed
series inside the run's brain mask, taken over the volumes 0, 10, 20, ... (the rule of the
feasibility study). value = clip(floor((v - lo) * 255 / (hi - lo) + 0.5), 0, 255) in float32.
The SD stills have a window of their own: 0 to the 99.5th percentile of the SD of the
preprocessed series inside the mask. Mean and SD are taken over the volumes that the confounds
table does not mark as non-steady-state, in float64, before the 8 bits.
"""

import hashlib
import json
import sys
import time
import zlib
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))  # the sibling modules, also under -I
sys.path.append(str(Path(__file__).resolve().parent.parent / "pipeline"))
from geom import surfaces_in_t1w
from pack_contours import PLANES, SURFACES, UNIT, session_contours
from pack_inputs import InputError, find_session, read_confounds, repetition_time
from session import FRAME_LIMIT_MM, check_frame, read_itk_ras, session_stream

HERE = Path(__file__).resolve().parent
FORMAT, VERSION = "func-viewer-pack", 1
ZLIB_LEVEL = 6
CHUNK_BYTES = 4 * 1024 * 1024  # decoded size that a chunk aims at
MAX_CHUNK = 32
WINDOW_STEP = 10  # the window and the reference volume use the volumes 0, 10, 20, ...
PERCENTILE = 99.5
FD_THRESHOLD_MM = 0.5
STREAMS = ("raw", "pre")

READING = (
    "Decision of Tomas Knapen, 2026-11-13: 'Raw bids means single per session registration to"
    " the anatomy to ensure comparability across sessions, processing stages, and subjects.'"
    " As read by the session that built this (the reading is not his words): the raw stream is"
    " the raw series resampled once into fMRIPrep's T1w space through one rigid transform per"
    " session, the stored scanner-to-T1w transform of the session's anchor run, the same for"
    " every run and volume. Nothing is motion-corrected, slice-time-corrected or unwarped."
)
RULES = {
    "order": "a volume is nx*ny*nz bytes; voxel (i, j, k) is at i + nx*(j + ny*k)",
    "files": "every .bin file is a zlib stream",
    "chunks": (
        "a chunk holds chunk_volumes volumes (the last one fewer) as residuals:"
        " (value - ref) mod 256, ref being ref.bin of the run and stream"
    ),
    "ref": f"rounded mean of the 8-bit volumes 0, {WINDOW_STEP}, {2 * WINDOW_STEP}, ...",
    "window": (
        f"per run, both streams: 0 to the {PERCENTILE}th percentile of the preprocessed series"
        f" inside the run's brain mask, over the volumes 0, {WINDOW_STEP}, {2 * WINDOW_STEP}, ..."
    ),
    "quantise": "value = clip(floor((v - lo) * 255 / (hi - lo) + 0.5), 0, 255), float32",
    "sd_window": (
        f"per run, both streams: 0 to the {PERCENTILE}th percentile of the temporal SD of the"
        " preprocessed series inside the run's brain mask"
    ),
    "stills": (
        "temporal mean and SD (n - 1) over the volumes that the confounds table does not mark"
        " as non-steady-state, in float64, then 8 bit"
    ),
    "difference": (
        "every difference view is raw minus pre of the two stored 8-bit values, in gray levels"
    ),
    "frame": (
        "per run, the displacement between its own stored transform and the anchor's over its"
        " brain mask (mm); flagged when the maximum exceeds limit_mm"
    ),
}


def quantise(vol, lo, hi):
    """The 8 bits of a volume under the window lo..hi (RULES['quantise'])."""
    k = np.float32(255.0 / (hi - lo))
    v = np.nan_to_num(np.asarray(vol, np.float32), nan=0.0)
    return np.clip((v - np.float32(lo)) * k + np.float32(0.5), 0, 255).astype(np.uint8)


def volume_bytes(vol):
    """A 3D uint8 volume in the byte order of the pack (RULES['order'])."""
    return np.ascontiguousarray(vol, np.uint8).tobytes(order="F")


def stored(path):
    """(array proxy of the stored values, slope, intercept, image) of a series."""
    img = nib.load(path)
    return img.dataobj.get_unscaled(), float(img.dataobj.slope), float(img.dataobj.inter), img


def scaled(proxy, slope, inter, t):
    return proxy[..., t].astype(np.float32) * slope + inter


class Writer:
    """Writes the files of a pack and keeps their sizes and SHA-256."""

    def __init__(self, out):
        self.out, self.total = Path(out), 0

    def put(self, name, data, compress=True):
        blob = zlib.compress(data, ZLIB_LEVEL) if compress else data
        path = self.out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
        self.total += len(blob)
        return {"file": name, "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}


def chunk_volumes(n_voxels, asked=None):
    if asked:
        return max(1, int(asked))
    return max(1, min(MAX_CHUNK, CHUNK_BYTES // n_voxels))


def pack_one_run(run, grid, anchor_xfm, k_chunk, writer, portable):
    """Pack both streams of one run; returns its manifest entry (without id, anchor, frame)."""
    shape, affine = grid
    proxy, slope, inter, img = stored(run.preproc)
    n = proxy.shape[3] if len(proxy.shape) > 3 else 1
    raw_n = nib.load(run.raw).shape[3:] or (1,)
    if raw_n[0] != n:
        raise InputError(f"raw series has {raw_n[0]} volumes, the preprocessed series {n}")
    mask = np.asarray(nib.load(run.mask).dataobj) > 0
    if mask.shape != tuple(shape) or not mask.any():
        raise InputError("the brain mask is empty or not on the grid of the series")
    fd, dvars, non_steady = read_confounds(run.confounds)
    if len(fd) != n:
        raise InputError(f"the confounds table has {len(fd)} rows, the series {n} volumes")
    steady = np.ones(n, bool)
    steady[[t for t in non_steady if t < n]] = False
    if steady.sum() < 2:
        steady[:] = True

    # pass A, the volumes 0, 10, 20, ...: the window of the run and the reference volumes
    sub = range(0, n, WINDOW_STEP)
    pre_sub = [scaled(proxy, slope, inter, t) for t in sub]
    hi = float(np.percentile(np.stack([v[mask] for v in pre_sub]), PERCENTILE))
    if not hi > 0:
        raise InputError("the preprocessed series has no signal inside the brain mask")
    window = (0.0, hi)
    ref = {"pre": np.zeros(shape, np.float64), "raw": np.zeros(shape, np.float64)}
    for v in pre_sub:
        ref["pre"] += quantise(v, *window)
    del pre_sub
    for _, v in session_stream(run.raw, shape, affine, anchor_xfm, sub, portable):
        ref["raw"] += quantise(v, *window)
    ref = {s: np.floor(ref[s] / len(sub) + 0.5).astype(np.uint8) for s in STREAMS}

    # pass B, every volume: the chunks, and the sums for the stills
    sums = {s: np.zeros(shape, np.float64) for s in STREAMS}
    squares = {s: np.zeros(shape, np.float64) for s in STREAMS}
    chunks = {s: [] for s in STREAMS}
    pending = {s: [] for s in STREAMS}
    seconds = {"raw": 0.0, "pre": 0.0}

    def flush(stream, first):
        entry = writer.put(
            f"runs/{run.id}/{stream}/c{len(chunks[stream]):04d}.bin", b"".join(pending[stream])
        )
        chunks[stream].append({**entry, "first": first, "count": len(pending[stream])})
        pending[stream].clear()

    raw_stream = session_stream(run.raw, shape, affine, anchor_xfm, None, portable)
    for t in range(n):
        for stream in STREAMS:
            t0 = time.perf_counter()
            if stream == "raw":
                _, vol = next(raw_stream)
            else:
                vol = scaled(proxy, slope, inter, t)
            if steady[t]:
                v64 = vol.astype(np.float64)
                sums[stream] += v64
                squares[stream] += v64 * v64
            q = quantise(vol, *window)
            pending[stream].append(volume_bytes(q - ref[stream]))  # uint8: mod 256
            if len(pending[stream]) == k_chunk or t == n - 1:
                flush(stream, t + 1 - len(pending[stream]))
            seconds[stream] += time.perf_counter() - t0

    # the stills
    n_steady = int(steady.sum())
    mean = {s: sums[s] / n_steady for s in STREAMS}
    sd = {
        s: np.sqrt(np.maximum(squares[s] - sums[s] ** 2 / n_steady, 0.0) / (n_steady - 1))
        for s in STREAMS
    }
    sd_hi = float(np.percentile(sd["pre"][mask], PERCENTILE))
    sd_window = (0.0, sd_hi if sd_hi > 0 else 1.0)
    streams = {}
    for s in STREAMS:
        base = f"runs/{run.id}/{s}/"
        streams[s] = {
            "ref": writer.put(base + "ref.bin", volume_bytes(ref[s])),
            "mean": writer.put(base + "mean.bin", volume_bytes(quantise(mean[s], *window))),
            "sd": writer.put(base + "sd.bin", volume_bytes(quantise(sd[s], *sd_window))),
            "chunks": chunks[s],
        }
    entry = {
        "n_volumes": n,
        "tr_s": round(repetition_time(run, img.header.get_zooms()), 6),
        "window": list(window),
        "sd_window": list(sd_window),
        "fd": fd,
        "dvars": dvars,
        "non_steady": [t for t in non_steady if t < n],
        "stills_volumes": n_steady,
        "streams": streams,
    }
    return entry, seconds


def grid_of(runs):
    """(shape, affine) of the display grid; every run of the session must lie on it."""
    first = nib.load(runs[0].preproc)
    shape, affine = tuple(int(x) for x in first.shape[:3]), np.asarray(first.affine, float)
    for run in runs[1:]:
        img = nib.load(run.preproc)
        if tuple(img.shape[:3]) != shape or not np.allclose(img.affine, affine, atol=1e-4):
            raise InputError("the runs of the session lie on different T1w-space grids")
    if nib.aff2axcodes(affine) != ("R", "A", "S"):
        raise InputError("the display grid is not in RAS order; the page cannot show it")
    return shape, affine


def grid_entry(shape, affine):
    """What the page needs of the grid. The affine itself stays out of the pack: its numbers
    are those of a public file. The contours come in voxel coordinates."""
    size = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    cosines = np.abs(np.diag(affine[:3, :3]) / size)
    return {
        "shape": list(shape),
        "voxel_mm": [round(float(x), 6) for x in size],
        "oblique_deg": round(float(np.degrees(np.arccos(np.clip(cosines.min(), -1, 1)))), 4),
    }


def frame_entry(d):
    keys = ("mean_mm", "max_mm", "rotation_deg", "translation_mm")
    return {
        **{k: round(float(d[k]), 4) for k in keys},
        "translation_ras": [round(float(x), 4) for x in d["translation_ras"]],
        "flagged": bool(d["flagged"]),
    }


def pack(data, fs_dir, out, pack_id, session=None, chunk=None, portable=False, contours=True):
    """Write the pack; returns (manifest, timings, the Session of pack_inputs)."""
    nib.arrayproxy.KEEP_FILE_OPEN_DEFAULT = True  # a series is read front to back, not re-opened
    t_start = time.perf_counter()
    ses = find_session(data, session)
    if contours and ses.fsnative is None:
        raise InputError("deriv/ has no from-T1w_to-fsnative transform (or use --no-contours)")
    shape, affine = grid = grid_of(ses.runs)
    anchor = ses.runs[0]
    transforms = {run.id: read_itk_ras(run.xfm) for run in ses.runs}
    masks = {}
    for run in ses.runs:
        img = nib.load(run.mask)
        masks[run.id] = (np.asarray(img.dataobj) > 0, img.affine)
    frame = check_frame(transforms, anchor.id, masks, FRAME_LIMIT_MM)
    writer = Writer(out)
    timings = {"runs": {}}
    manifest = {
        "format": FORMAT,
        "version": VERSION,
        "id": pack_id,
        "grid": grid_entry(shape, affine),
        "chunk_volumes": chunk_volumes(int(np.prod(shape)), chunk),
        "fd_threshold_mm": FD_THRESHOLD_MM,
        "rules": RULES,
        "registration": {
            "reading": READING,
            "anchor": anchor.id,
            "rule": ses.rule,
            "limit_mm": frame["limit_mm"],
            "shared_frame_plausible": frame["shared_frame_plausible"],
            "anchor_is_outlier": frame["anchor_is_outlier"],
            "portable_voxel_map": bool(portable),
        },
        "runs_left_out": len(ses.left_out),
        "contours": None,
        "runs": [],
    }
    if contours:
        t0 = time.perf_counter()
        surfaces, _ = surfaces_in_t1w(fs_dir, ses.fsnative)
        files, counts = session_contours(surfaces, shape, affine)
        manifest["contours"] = {
            "unit": UNIT,
            "surfaces": list(SURFACES),
            "planes": {
                p: {**writer.put(f"contours/{p}.bin", files[p]), **counts[p]} for p in PLANES
            },
        }
        timings["contours_s"] = round(time.perf_counter() - t0, 2)
    for run in ses.runs:
        t0 = time.perf_counter()
        before = writer.total
        entry, seconds = pack_one_run(
            run, grid, transforms[anchor.id], manifest["chunk_volumes"], writer, portable
        )
        manifest["runs"].append(
            {
                "id": run.id,
                "anchor": run.anchor,
                "frame": frame_entry(frame["runs"][run.id]),
                **entry,
            }
        )
        timings["runs"][run.id] = {
            "seconds": round(time.perf_counter() - t0, 2),
            "raw_s": round(seconds["raw"], 2),
            "pre_s": round(seconds["pre"], 2),
            "bytes": writer.total - before,
            "bytes_raw": sum(c["bytes"] for c in entry["streams"]["raw"]["chunks"]),
            "bytes_pre": sum(c["bytes"] for c in entry["streams"]["pre"]["chunks"]),
            "volumes": entry["n_volumes"],
        }
    text = json.dumps(manifest, indent=1, sort_keys=True, allow_nan=False) + "\n"
    writer.put("manifest.json", text.encode(), compress=False)
    timings["bytes"] = writer.total
    timings["seconds"] = round(time.perf_counter() - t_start, 2)
    return manifest, timings, ses


def write_map(path, pack_id, ses):
    lines = ["pack\trun\tanchor\tstem\traw\tsession\trule\twhy"]
    for run in ses.runs:
        cells = (pack_id, run.id, "anchor" if run.anchor else "", run.stem, run.raw.name)
        lines.append("\t".join((*cells, ses.label, ses.rule, ses.why)))
    for stem, reason in ses.left_out:
        lines.append("\t".join((pack_id, "left out", "", stem, "", ses.label, "", reason)))
    Path(path).write_text("\n".join(lines) + "\n")


def _inside(path, folder):
    path, folder = Path(path).resolve(), Path(folder).resolve()
    return path == folder or folder in path.parents


def main():
    args = sys.argv[1:]
    options = {"--id": None, "--session": None, "--map": None, "--chunk": None}
    flags = {"--portable": False, "--no-contours": False}
    positional = []
    while args:
        arg = args.pop(0)
        if arg in options and args:
            options[arg] = args.pop(0)
        elif arg in flags:
            flags[arg] = True
        elif arg.startswith("--"):
            positional = []  # an option that is unknown or has no value
            break
        else:
            positional.append(arg)
    if len(positional) != 3 or not options["--id"]:
        print(__doc__.strip().split("\n\nTHE STREAMS")[0], file=sys.stderr)
        sys.exit(2)
    data, fs_dir, out = positional
    out = Path(out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        sys.exit(f"{out} exists and is not an empty directory; a pack is written into a new one")
    if options["--map"] and (
        _inside(options["--map"], out) or _inside(options["--map"], HERE.parents[2])
    ):
        sys.exit("--map names datasets: it may lie neither in the pack nor in the repository")
    out.mkdir(parents=True, exist_ok=True)
    try:
        manifest, timings, ses = pack(
            data,
            fs_dir,
            out,
            options["--id"],
            session=options["--session"],
            chunk=options["--chunk"],
            portable=flags["--portable"],
            contours=not flags["--no-contours"],
        )
    except InputError as err:
        sys.exit(f"cannot pack: {err}")
    if options["--map"]:
        write_map(options["--map"], options["--id"], ses)
    reg = manifest["registration"]
    print(f"pack {manifest['id']}: {len(manifest['runs'])} runs, grid {manifest['grid']['shape']}")
    print(
        f"anchor {reg['anchor']} by rule {reg['rule']}; {manifest['runs_left_out']} runs left out"
    )
    for run in manifest["runs"]:
        f, t = run["frame"], timings["runs"][run["id"]]
        flag = "; FLAGGED" if f["flagged"] else ""
        print(
            f"{run['id']}: {run['n_volumes']} volumes, window 0 to {run['window'][1]:.1f};"
            f" from the anchor {f['mean_mm']:.2f} mm mean, {f['max_mm']:.2f} mm max{flag};"
            f" raw {t['bytes_raw'] / 1e6:.2f} MB, pre {t['bytes_pre'] / 1e6:.2f} MB,"
            f" {t['seconds']:.1f} s"
        )
    if not reg["shared_frame_plausible"]:
        print("a run is flagged: the page shows it with a warning, not as sharing the frame")
    print(json.dumps(timings, sort_keys=True))


if __name__ == "__main__":
    main()
