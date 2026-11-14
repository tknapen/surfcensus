"""An independent check of where a pack puts the raw stream: recomputed from the source files
with code that shares nothing with pipeline/geom.py, session.py or pack_run.py.

usage: check_geometry.py PACK MAP DATA [--volumes N] [--json FILE]

PACK   a pack of pack_run.py
MAP    the table that pack_run.py wrote with --map (run ids to file names; it names a dataset)
DATA   the local files the pack was made from (deriv/, raw/), as fetch_run.py or
       pipeline/fetch.py writes them
--volumes N   raw volumes per run that are recomputed and compared (default 3: the first, the
              middle and the last)
--json FILE   also write the results as JSON (outside the repository: it holds no name, but
              a run's correlations are as public as its files)

What check_player.py checks is that the pack and the page show the values that the prototype's
own resampling gives: it computes them with pipeline/geom.py, the reader and resampler that the
pack was made with, so an error in reading the transform or in placing the raw series would be
in both and pass. This script reads every file itself, with nibabel, numpy and scipy only:

- The ITK text transform of the anchor run is read here (Parameters, FixedParameters: y = M (x
  - c) + t + c in LPS) and turned into RAS. It maps a point of fMRIPrep's T1w space to a point of
  the scanner space of the run's BOLD-space reference image (the convention of nitransforms,
  fMRIPrep's transform library, as the review of stage 1 read it in its source; not checked
  against the ITK documentation, which this session could not reach).
- The raw volume is placed by the affine of fMRIPrep's BOLD-space reference image of the run
  (deriv/<stem>_boldref.nii.gz), which is the scanner frame that the stored transform refers
  to, and not by a choice between the raw header's qform and sform. Without the reference image
  the raw header's sform-first affine is used, and the check says that it could not check the
  header.
- Each display voxel (i, j, k) of the preprocessed grid goes to T1w space by the grid's affine,
  to scanner space by the transform, to a raw voxel by the inverse of that affine, and is
  sampled with scipy.ndimage.map_coordinates: cubic B-spline with the edge repeated, 0 where the
  point lies outside the outermost raw voxel centres. (This is the rule that the pack follows,
  so that the values can be compared one by one; it is written here from that statement, not
  taken from the pack's code.)
- The window is recomputed (0 to the 99.5th percentile of the preprocessed series in the brain
  mask over the volumes 0, 10, 20, ...) and the volume quantised to 8 bits, then compared voxel
  by voxel with the volume decoded from the pack. The preprocessed stream of the same volumes is
  compared with the stored series in the same way.

Then the placement itself, which no comparison with the pack can show: the temporal mean of the
raw series (in its own voxels, over every volume) is resampled with linear interpolation through
the run's OWN stored transform (where fMRIPrep's registration puts the run) and correlated, inside
the run's brain mask, with the temporal mean of the preprocessed series. (The pack draws every
run through the anchor's transform by design, so a run that is not the anchor is off by the
head's movement between runs; that correlation is printed as well and is not tested.) The
same is done with the raw header's other affine (the qform where the reference agrees with the
sform-first affine, and the reverse), and with the raw stream shifted in T1w space by up to 3 mm
along each axis (1 mm steps, then 0.25 mm around the best). A pack whose raw stream lay well
off the anatomy would show a much higher correlation for the other affine or for a shift.
Last, the pack's own stills are decoded (the 8-bit temporal means of both streams) and
correlated in the same mask: that is the placement the pack shows, compared with what this
script computes for the same placement through the anchor's transform.

A run FAILS when:
- more than 0.1% of its brain-mask voxels in the compared volumes differ from the pack by more
  than 1 gray level (raw or preprocessed stream);
- the raw header's other affine gives a correlation higher than this script's placement by
  more than 0.05;
- the correlation of the pack's two mean stills is lower than this script's correlation through
  the anchor's transform by more than 0.05.
These limits are choices of this script; the README (stage 1) gives what it found on the packs
it was run on and on a pack made with a wrong raw affine on purpose.

The best shift is reported and is not a failure. In the first version a gain of more than 0.03
from a shift failed a run. It failed a pack whose raw stream the comparison of values and the
other checks place correctly (a quiet run with fieldmap-less distortion correction: 1.9 mm, r
0.892 to 0.943): distortion as acquired displaces the raw image along the phase-encoding axis,
and a rigid shift then lines up part of it. That is what the raw stream is meant to show
(decision 7 of the README), so a shift can say "look here", not "misplaced". A misplacement of
the kind the check exists for (a wrong affine of the raw header, about 22 mm) lies far outside
the search anyway, and the other tests catch it.

A correlation below 1 is not a failure by itself either: the raw stream is not motion-corrected
and, where fMRIPrep applied distortion correction, not unwarped (the manifest says per run). Exit
status 0 when no run fails, 1 when one does, 2 for usage. Prints one line per check, by the
pack's run ids; names no file.
"""

import json
import sys
import zlib
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

LPS = np.diag([-1.0, -1.0, 1.0, 1.0])
VALUE_LIMIT = 1  # gray levels
VALUE_SHARE = 1e-3  # of the brain-mask voxels
OTHER_AFFINE_GAIN = 0.05
STILLS_LOSS = 0.05
SHIFT_MM = 3.0
PERCENTILE = 99.5
WINDOW_STEP = 10


# ---- reading ---------------------------------------------------------------------------------


def itk_to_ras(path):
    """The 4 x 4 RAS matrix of a single affine ITK text transform: y = M (x - c) + t + c in LPS.
    Refuses a file with more or fewer than one transform."""
    params = fixed = None
    transforms = 0
    for line in Path(path).read_text().splitlines():
        if line.startswith("Transform:"):
            transforms += 1
        elif line.startswith("Parameters:"):
            params = [float(x) for x in line.split()[1:]]
        elif line.startswith("FixedParameters:"):
            fixed = [float(x) for x in line.split()[1:]]
    if transforms != 1 or params is None or len(params) != 12 or fixed is None:
        raise ValueError("not a single 3D affine ITK transform")
    m = np.array(params[:9]).reshape(3, 3)
    t = np.array(params[9:12])
    c = np.array(fixed[:3])
    lps = np.eye(4)
    lps[:3, :3] = m
    lps[:3, 3] = t + c - m @ c
    return LPS @ lps @ LPS


def decode(pack, entry, shape, t=None):
    """A volume of the pack: ref.bin when t is None, else volume t of the chunks."""
    n = int(np.prod(shape))
    ref = np.frombuffer(zlib.decompress((pack / entry["ref"]["file"]).read_bytes()), np.uint8)
    if t is None:
        return ref.reshape(shape, order="F")
    for chunk in entry["chunks"]:
        if chunk["first"] <= t < chunk["first"] + chunk["count"]:
            data = zlib.decompress((pack / chunk["file"]).read_bytes())
            res = np.frombuffer(data, np.uint8)[
                (t - chunk["first"]) * n : (t - chunk["first"] + 1) * n
            ]
            return (res + ref).astype(np.uint8).reshape(shape, order="F")  # uint8 adds mod 256
    raise KeyError(t)


def decode_still(pack, entry, shape, what="mean"):
    data = zlib.decompress((pack / entry[what]["file"]).read_bytes())
    return np.frombuffer(data, np.uint8).reshape(shape, order="F")


class Series:
    """A 3D or 4D image read once: volume t in float32, stored value * slope + intercept."""

    def __init__(self, path):
        self.img = nib.load(path)
        self.data = np.asanyarray(self.img.dataobj.get_unscaled())
        self.slope = np.float32(self.img.dataobj.slope)
        self.inter = np.float32(self.img.dataobj.inter)
        self.n = self.data.shape[3] if self.data.ndim > 3 else 1

    def volume(self, t):
        vol = self.data[..., t] if self.data.ndim > 3 else self.data
        return vol.astype(np.float32) * self.slope + self.inter


# ---- placing ---------------------------------------------------------------------------------


def raw_coordinates(shape, grid_affine, t1w_to_scanner, raw_affine, shift=(0.0, 0.0, 0.0)):
    """Raw voxel coordinates (3, n) of every display voxel; shift (mm, RAS) moves the T1w point."""
    ijk = np.indices(shape, dtype=np.float64).reshape(3, -1)
    xyz = grid_affine[:3, :3] @ ijk + grid_affine[:3, 3:4] + np.asarray(shift, float)[:, None]
    scanner = t1w_to_scanner[:3, :3] @ xyz + t1w_to_scanner[:3, 3:4]
    inverse = np.linalg.inv(raw_affine)
    return inverse[:3, :3] @ scanner + inverse[:3, 3:4]


def sample(vol, coords, shape, order):
    out = ndimage.map_coordinates(vol, coords, order=order, mode="nearest", output=np.float32)
    top = np.array(vol.shape, float)[:, None] - 1
    inside = np.all((coords >= 0) & (coords <= top), axis=0)
    out[~inside] = 0
    return out.reshape(shape)


def quantise(v, hi):
    x = np.nan_to_num(v.astype(np.float32)) * np.float32(255.0 / hi) + np.float32(0.5)
    return np.clip(x, 0, 255).astype(np.uint8)


def correlation(a, b, mask):
    x, y = a[mask].astype(np.float64), b[mask].astype(np.float64)
    x, y = x - x.mean(), y - y.mean()
    den = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / den) if den > 0 else float("nan")


def header_affines(img, reference):
    """(used, other, how): the affine that places the raw volume (the reference image's, or the
    sform-first img.affine without one), the other affine of the header, and how it was chosen."""
    qform, first = img.header.get_qform(), img.affine
    if reference is None:
        other = qform if not np.allclose(qform, first, atol=1e-4) else None
        return first, other, "no reference image: the raw header's img.affine, unchecked"
    used = reference.affine
    if np.allclose(used, first, atol=1e-4):
        other = qform if not np.allclose(qform, first, atol=1e-4) else None
        how = "reference image (equals the header's sform-first affine)"
    elif np.allclose(used, qform, atol=1e-4):
        other, how = first, "reference image (equals the header's qform, not its sform)"
    else:
        other, how = first, "reference image (equals neither affine of the header)"
    return used, other, how


# ---- one run ---------------------------------------------------------------------------------


def check_run(pack, entry, files, anchor_xfm, own_xfm, n_check):
    pre, raw = Series(files["pre"]), Series(files["raw"])
    pre_img, raw_img, mask_img = pre.img, raw.img, nib.load(files["mask"])
    reference = nib.load(files["boldref"]) if files["boldref"].is_file() else None
    if reference is not None and tuple(reference.shape[:3]) != tuple(raw_img.shape[:3]):
        reference = None
    shape = tuple(int(x) for x in pre_img.shape[:3])
    grid = np.asarray(pre_img.affine, float)
    mask = np.asarray(mask_img.dataobj) > 0
    used, other, how = header_affines(raw_img, reference)
    n = pre.n
    if raw.n != n:
        raise ValueError(f"{entry['id']}: raw and preprocessed series differ in length")
    out = {"run": entry["id"], "placed_by": how, "fails": []}

    # the window and the value comparison
    sub = list(range(0, n, WINDOW_STEP))
    hi = float(np.percentile(np.stack([pre.volume(t)[mask] for t in sub]), PERCENTILE))
    out["window_hi"] = round(hi, 4)
    out["window_hi_manifest"] = round(float(entry["window"][1]), 4)
    if abs(hi - entry["window"][1]) > 1e-3 * max(abs(hi), 1.0):
        out["fails"].append("window differs from the manifest")
    picks = sorted({int(x) for x in np.linspace(0, n - 1, max(n_check, 1)).round()})
    coords = raw_coordinates(shape, grid, anchor_xfm, used)
    compared = {"raw": [0, 0, 0], "pre": [0, 0, 0]}  # differing, beyond the limit, in mask
    for t in picks:
        mine = {
            "raw": quantise(sample(raw.volume(t), coords, shape, 3), hi),
            "pre": quantise(pre.volume(t), hi),
        }
        for stream in ("raw", "pre"):
            theirs = decode(pack, entry["streams"][stream], shape, t)
            d = np.abs(mine[stream].astype(np.int16) - theirs.astype(np.int16))[mask]
            compared[stream][0] += int((d > 0).sum())
            compared[stream][1] += int((d > VALUE_LIMIT).sum())
            compared[stream][2] += int(d.size)
    out["volumes_compared"] = [int(t) for t in picks]
    for stream, (differ, beyond, total) in compared.items():
        out[f"{stream}_voxels_differ"] = differ
        out[f"{stream}_voxels_beyond_limit"] = beyond
        out[f"{stream}_voxels_in_mask"] = total
        if beyond > VALUE_SHARE * total:
            out["fails"].append(
                f"{stream} stream: {beyond} of {total} voxels differ by more than {VALUE_LIMIT}"
            )

    # the placement: temporal means
    raw_sum = np.zeros(raw_img.shape[:3], np.float64)
    pre_sum = np.zeros(shape, np.float64)
    for t in range(n):
        raw_sum += raw.volume(t)
        pre_sum += pre.volume(t)
    raw_mean = (raw_sum / n).astype(np.float32)
    pre_mean = pre_sum / n

    def r_of(affine, shift=(0.0, 0.0, 0.0), xfm=own_xfm):
        c = raw_coordinates(shape, grid, xfm, affine, shift)
        return correlation(sample(raw_mean, c, shape, 1), pre_mean, mask)

    r0 = r_of(used)
    out["r_mean"] = round(r0, 4)
    r_anchor = r_of(used, xfm=anchor_xfm)
    out["r_mean_anchor_transform"] = round(r_anchor, 4)
    stills = [decode_still(pack, entry["streams"][k], shape) for k in ("raw", "pre")]
    r_stills = correlation(stills[0], stills[1], mask)
    out["r_pack_stills"] = round(r_stills, 4)
    if r_anchor - r_stills > STILLS_LOSS:
        out["fails"].append(
            f"the pack's mean stills correlate at {r_stills:.3f}, this placement at {r_anchor:.3f}"
        )
    if other is not None:
        r_other = r_of(other)
        out["r_other_affine"] = round(r_other, 4)
        out["other_affine_apart_mm"] = round(float(np.abs(other[:3, 3] - used[:3, 3]).max()), 2)
        if r_other - r0 > OTHER_AFFINE_GAIN:
            out["fails"].append(f"the header's other affine gives r {r_other:.3f} against {r0:.3f}")
    else:
        out["r_other_affine"] = None
    best, best_r = (0.0, 0.0, 0.0), r0
    steps = np.arange(-SHIFT_MM, SHIFT_MM + 1e-9, 1.0)
    for dx in steps:
        for dy in steps:
            for dz in steps:
                if dx == dy == dz == 0:
                    continue
                r = r_of(used, (dx, dy, dz))
                if r > best_r:
                    best, best_r = (float(dx), float(dy), float(dz)), r
    fine = np.arange(-0.75, 0.76, 0.25)
    centre = best
    for dx in fine:
        for dy in fine:
            for dz in fine:
                s = (centre[0] + dx, centre[1] + dy, centre[2] + dz)
                if max(abs(x) for x in s) > SHIFT_MM + 0.75 or s == centre:
                    continue
                r = r_of(used, s)
                if r > best_r:
                    best, best_r = tuple(float(x) for x in s), r
    out["best_shift_mm"] = [round(x, 2) for x in best]
    out["r_best_shift"] = round(best_r, 4)
    out["shift_gain"] = round(best_r - r0, 4)  # reported, not a failure (see the docstring)
    return out


def main():
    args = sys.argv[1:]
    options = {"--volumes": "3", "--json": None}
    positional = []
    while args:
        arg = args.pop(0)
        if arg in options and args:
            options[arg] = args.pop(0)
        elif arg.startswith("--"):
            positional = []
            break
        else:
            positional.append(arg)
    if len(positional) != 3:
        print(__doc__.strip().split("\n\nWhat check_player")[0], file=sys.stderr)
        sys.exit(2)
    pack, map_path, data = (Path(p) for p in positional)
    manifest = json.loads((pack / "manifest.json").read_text())
    rows = [line.split("\t") for line in map_path.read_text().splitlines()[1:]]
    rows = {r[1]: r for r in rows if r[1].startswith("R")}
    anchor = next(r for r in rows.values() if r[2] == "anchor")
    anchor_xfm = itk_to_ras(data / "deriv" / f"{anchor[3]}_from-scanner_to-T1w_mode-image_xfm.txt")
    results, failed = [], False
    for entry in manifest["runs"]:
        stem, raw = rows[entry["id"]][3], rows[entry["id"]][4]
        files = {
            "pre": data / "deriv" / f"{stem}_space-T1w_desc-preproc_bold.nii.gz",
            "mask": data / "deriv" / f"{stem}_space-T1w_desc-brain_mask.nii.gz",
            "boldref": data / "deriv" / f"{stem}_boldref.nii.gz",
            "raw": data / "raw" / raw,
        }
        own_xfm = itk_to_ras(data / "deriv" / f"{stem}_from-scanner_to-T1w_mode-image_xfm.txt")
        res = check_run(pack, entry, files, anchor_xfm, own_xfm, int(options["--volumes"]))
        results.append(res)
        failed = failed or bool(res["fails"])
        other = "none" if res["r_other_affine"] is None else f"{res['r_other_affine']:.4f}"
        print(f"{manifest['id']} {res['run']}: placed by {res['placed_by']}")
        print(
            f"{manifest['id']} {res['run']}: volumes {res['volumes_compared']}: raw"
            f" {res['raw_voxels_differ']} of {res['raw_voxels_in_mask']} mask voxels differ"
            f" ({res['raw_voxels_beyond_limit']} by more than {VALUE_LIMIT}), preprocessed"
            f" {res['pre_voxels_differ']} ({res['pre_voxels_beyond_limit']})"
        )
        print(
            f"{manifest['id']} {res['run']}: r of the temporal means {res['r_mean']:.4f}"
            f" (through the anchor's transform, as packed: {res['r_mean_anchor_transform']:.4f});"
            f" pack's mean stills {res['r_pack_stills']:.4f}; other"
            f" header affine {other}; best shift {res['best_shift_mm']} mm gives"
            f" {res['r_best_shift']:.4f}"
        )
        for text in res["fails"]:
            print(f"FAIL {manifest['id']} {res['run']}: {text}")
    if options["--json"]:
        Path(options["--json"]).write_text(
            json.dumps({"id": manifest["id"], "runs": results}, indent=1)
        )
    print("check_geometry: " + ("FAILED" if failed else "ok"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
