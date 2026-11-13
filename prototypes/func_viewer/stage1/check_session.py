"""Self-check of the session logic on the worked example (ds000102 sub-01, two runs).

usage: check_session.py DATA      (DATA as pipeline/fetch.py wrote it: deriv/ and raw/)

Exit status 1 when one of these does not hold, 0 when all do:

1. The anchor of the one session is run 1, by name order (the raw runs have no JSON sidecar).
2. The raw stream of the anchor run through the session logic is identical, bit for bit and in
   every volume, to the per-run stream of the feasibility study: the volumes that
   pipeline/make_videos.py makes with geom.resample, and the three-plane strips that
   pipeline/epiview.py cuts from them. With the portable voxel map (session.voxel_map) the
   stream may differ from that in the last bits: the maps must agree to 1e-12 voxel and the
   volumes to 0.001, and how many volumes differ is printed.
3. The raw stream of run 2 through the anchor's transform has the same SHA-256 in two calls in
   this process and in a call in a new process, and differs from run 2 through its own
   transform (so the anchor's transform is the one in use).
4. The displacement between the stored transform of run 2 and the anchor's, over the brain
   mask of run 2, is what the README reports (EXPECTED below), within TOLERANCE.

Reads the two raw series and the small files of the derivative; writes nothing. About a minute
on one core.
"""

import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))  # the sibling module, also under -I
sys.path.append(str(Path(__file__).resolve().parent.parent / "pipeline"))  # the study's modules
from epiview import Panels
from geom import resample
from session import (
    FRAME_LIMIT_MM,
    RULE_NAME,
    check_frame,
    pick_anchor,
    raw_volumes,
    read_itk_ras,
    resample_voxels,
    session_stream,
    stream_digest,
    voxel_map,
)

HERE = Path(__file__).resolve().parent
# Measured 2026-11-13 on ds000102 sub-01: run 2 against the anchor (run 1), over the 39,801
# voxels of the T1w-space brain mask of run 2.
EXPECTED = {"mean_mm": 0.829, "max_mm": 1.087, "rotation_deg": 0.205, "translation_mm": 0.821}
TOLERANCE = 0.002
RUNS = ("sub-01_task-flanker_run-1_bold", "sub-01_task-flanker_run-2_bold")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__.strip(), file=sys.stderr)  # too few arguments: say how to run it
        sys.exit(2)
    data = Path(sys.argv[1])
    failed = []

    def report(ok, text):
        print(("ok    " if ok else "FAILED ") + text)
        if not ok:
            failed.append(text)

    def xfm(run):
        return data / f"deriv/run-{run}_from-scanner_to-T1w_mode-image_xfm.txt"

    def raw(run):
        return data / f"raw/run-{run}_bold.nii.gz"

    # 1. the anchor
    anchor = pick_anchor([(name, None) for name in reversed(RUNS)])
    report(
        anchor.name == RUNS[0] and anchor.rule == RULE_NAME,
        f"anchor: {anchor.name.rsplit('_', 2)[1]} by {anchor.rule} ({anchor.why})",
    )
    grid = nib.load(data / "deriv/run-1_space-T1w_boldref.nii.gz")
    shape, affine = grid.shape, grid.affine
    transforms = {run: read_itk_ras(xfm(run)) for run in (1, 2)}
    masks = {}
    for run in (1, 2):
        img = nib.load(data / f"deriv/run-{run}_space-T1w_desc-brain_mask.nii.gz")
        masks[run] = (np.asarray(img.dataobj) > 0, img.affine)

    # 2. the anchor run: session stream against the per-run stream of the feasibility study
    stored, run_affine, slope, inter = raw_volumes(raw(1))
    union = masks[1][0] | masks[2][0]
    slices = np.round(np.array(np.nonzero(union)).mean(axis=1) + np.array([4, 0, 1])).astype(int)
    panels = Panels(shape, affine, 2.0, slices)
    portable_map = voxel_map(run_affine, transforms[1], affine, portable=True)
    n_vol = n_diff = n_strip = n_portable = 0
    worst = worst_portable = 0.0
    for t, vol in session_stream(raw(1), shape, affine, transforms[1]):
        source = stored[..., t].astype(np.float32) * slope + inter
        per_run = resample(source, run_affine, shape, affine, transforms[1], order=3)
        if vol.tobytes() != per_run.tobytes():
            n_diff += 1
            worst = max(worst, float(np.abs(vol - per_run).max()))
        hi = float(per_run.max()) or 1.0
        n_strip += panels.strip(vol, 0.0, hi).tobytes() != panels.strip(per_run, 0.0, hi).tobytes()
        portable = resample_voxels(source, portable_map, shape)
        if portable.tobytes() != per_run.tobytes():
            n_portable += 1
            worst_portable = max(worst_portable, float(np.abs(portable - per_run).max()))
        n_vol += 1
    report(
        n_diff == 0 and n_strip == 0 and n_vol == stored.shape[3],
        f"anchor run: {n_vol} volumes, {n_diff} differ from the per-run stream"
        f" (largest difference {worst:g}), {n_strip} three-plane strips differ",
    )
    maps = np.abs(portable_map - voxel_map(run_affine, transforms[1], affine)).max()
    report(
        maps < 1e-12 and worst_portable < 1e-3,
        f"anchor run with the portable voxel map: the maps differ by at most {maps:.1e} voxel,"
        f" {n_portable} of {n_vol} volumes differ from the per-run stream, by at most"
        f" {worst_portable:g}",
    )

    # 3. run 2 through the anchor's transform: the same bytes in every call
    first = stream_digest(raw(2), shape, affine, transforms[1])
    second = stream_digest(raw(2), shape, affine, transforms[1])
    command = [sys.executable, "-I", str(HERE / "session.py"), "digest"]
    command += [str(raw(2)), str(data / "deriv/run-1_space-T1w_boldref.nii.gz"), str(xfm(1))]
    child = subprocess.run(command, capture_output=True, text=True, check=False)
    third = child.stdout.strip()
    own = stream_digest(raw(2), shape, affine, transforms[2])
    report(
        first == second == third,
        f"run 2 through the anchor's transform: SHA-256 {first[:16]}... in two calls and in a"
        f" new process ({'same' if first == second == third else 'DIFFERENT'})",
    )
    report(first != own, "run 2 through its own transform gives other bytes")

    # 4. the displacement between the two stored transforms
    frame = check_frame(transforms, 1, masks)
    d = frame["runs"][2]
    off = {k: d[k] for k in EXPECTED if abs(d[k] - EXPECTED[k]) > TOLERANCE}
    report(
        not off and not d["flagged"] and frame["shared_frame_plausible"],
        f"run 2 against the anchor: mean {d['mean_mm']:.3f} mm, max {d['max_mm']:.3f} mm,"
        f" rotation {d['rotation_deg']:.3f} deg, translation {d['translation_mm']:.3f} mm over"
        f" {d['n_voxels']} voxels; expected {EXPECTED} within {TOLERANCE}; not flagged at"
        f" {FRAME_LIMIT_MM:g} mm",
    )
    if failed:
        print(f"{len(failed)} of the checks failed", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
