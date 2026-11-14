"""What one registration per session does to the runs of a subject, in numbers.

usage: session_measure.py DATA OUT

DATA holds deriv/ and raw/ as pipeline/fetch.py writes them for the worked example, or deriv/,
raw/ and means/ as session_fetch.py writes them for another subject. Per session it picks the
anchor (session.pick_anchor) and reports for every run:

* the displacement between the run's own stored transform and the anchor's over the run's brain
  mask (session.check_frame);
* two pictures of the run in T1w space, one through its own transform (the per-run stream of the
  feasibility study) and one through the anchor's (one registration per session), each
  correlated inside the brain mask with what fMRIPrep made of the run: the reference image of
  the run's own space against space-T1w_boldref, and, where the series or their means are there,
  the temporal mean of the raw series against the temporal mean of the T1w-space preprocessed
  series; and how the two pictures differ from each other (correlation, and mean absolute
  difference in percent of the median in the mask).

For a subject with several sessions it also reports the anchor of every later session against
the anchor of the first, which is what would be shown if the sessions were taken for one.

The lines it prints call sessions and runs S1, S2 and R1, R2 (runs in name order within the
session) and name nothing. OUT/session_measure.json has every number with the file names, and
OUT/session_ids.tsv maps the ids to the names; both stay out of the repository.
"""

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))  # the sibling module, also under -I
from session import (
    check_frame,
    frame_displacement,
    name_order,
    parse_name,
    pick_anchor,
    read_itk_ras,
    run_key,
    session_of,
    session_volume,
    sidecar_time,
)

XFM = "_from-scanner_to-T1w_mode-image_xfm.txt"


def load_runs(data):
    """The runs under DATA, by the stem of their transform file, in name order."""
    data = Path(data)
    runs = []
    for xfm in sorted((data / "deriv").glob("*" + XFM)):
        stem = xfm.name[: -len(XFM)]
        deriv, raw, means = data / "deriv" / stem, data / "raw" / stem, data / "means" / stem
        files = {
            "mask": deriv.with_name(stem + "_space-T1w_desc-brain_mask.nii.gz"),
            "t1w_boldref": deriv.with_name(stem + "_space-T1w_boldref.nii.gz"),
            "boldref": deriv.with_name(stem + "_boldref.nii.gz"),
            "preproc": deriv.with_name(stem + "_space-T1w_desc-preproc_bold.nii.gz"),
            "raw": raw.with_name(stem + "_bold.nii.gz"),
            "raw_mean": means.with_name(stem + "_raw_mean.nii.gz"),
            "preproc_mean": means.with_name(stem + "_preproc_mean.nii.gz"),
        }
        if not files["raw"].exists():  # run-01 in the raw name where the derivative has run-1
            key = (session_of(parse_name(stem)), run_key(parse_name(stem)))
            for candidate in sorted((data / "raw").glob("*_bold.nii.gz")):
                ent = parse_name(candidate.name[: -len("_bold.nii.gz")])
                if "echo" not in ent and (session_of(ent), run_key(ent)) == key:
                    files["raw"] = candidate
                    break
        runs.append(
            {
                "stem": stem,
                "session": session_of(parse_name(stem)),
                "xfm": xfm,
                "time": sidecar_time(Path(str(files["raw"])[: -len(".nii.gz")] + ".json")),
                **{k: (p if p.exists() else None) for k, p in files.items()},
            }
        )
    return sorted(runs, key=lambda r: (r["session"], name_order(parse_name(r["stem"])), r["stem"]))


def corr(a, b, mask):
    return float(np.corrcoef(a[mask].astype(np.float64), b[mask].astype(np.float64))[0, 1])


def volume(path):
    img = nib.load(path)
    return np.asarray(img.dataobj, dtype=np.float32), img.affine


def temporal_mean(series, mean):
    """The temporal mean of a run: a stored mean image, else computed from the series."""
    if mean is not None:
        return volume(mean)
    if series is None:
        return None
    data, affine = volume(series)
    return data.mean(axis=3), affine


def two_pictures(image, affine, grid, own, anchor, target, mask):
    """An image of the run's own space through its own and through the anchor's transform."""
    a = session_volume(image, affine, grid[0], grid[1], own)
    b = session_volume(image, affine, grid[0], grid[1], anchor)
    median = float(np.median(a[mask]))
    return {
        "r_own": corr(a, target, mask),
        "r_anchor": corr(b, target, mask),
        "r_between": corr(a, b, mask),
        "mean_abs_diff_percent": float(100 * np.abs(a - b)[mask].mean() / median),
    }


def measure_run(run, own, anchor):
    """The two pictures of one run, from its reference image and from its temporal means."""
    mask_img = nib.load(run["mask"])
    mask = np.asarray(mask_img.dataobj) > 0
    grid = (mask_img.shape, mask_img.affine)
    out = {}
    if run["boldref"] and run["t1w_boldref"]:
        ref, ref_affine = volume(run["boldref"])
        target, _ = volume(run["t1w_boldref"])
        out["reference"] = two_pictures(ref, ref_affine, grid, own, anchor, target, mask)
    raw = temporal_mean(run["raw"], run["raw_mean"])
    pre = temporal_mean(run["preproc"], run["preproc_mean"])
    if raw is not None and pre is not None:
        out["series_mean"] = two_pictures(raw[0], raw[1], grid, own, anchor, pre[0], mask)
    return out


def line(tag, d, pictures):
    text = (
        f"  {tag}: mean {d['mean_mm']:.2f} mm, max {d['max_mm']:.2f} mm; rotation"
        f" {d['rotation_deg']:.2f} deg, translation {d['translation_mm']:.2f} mm"
        f" (R {d['translation_ras'][0]:+.2f}, A {d['translation_ras'][1]:+.2f},"
        f" S {d['translation_ras'][2]:+.2f})"
    )
    if d.get("flagged"):
        text += "; FLAGGED"
    for key, label in (("reference", "reference image"), ("series_mean", "temporal mean")):
        if key in pictures:
            p = pictures[key]
            text += (
                f"\n      {label}: r {p['r_own']:.4f} through its own transform,"
                f" {p['r_anchor']:.4f} through the anchor's; the two pictures: r"
                f" {p['r_between']:.4f}, mean abs difference {p['mean_abs_diff_percent']:.1f}%"
            )
    return text


def main():
    if len(sys.argv) < 3 or sys.argv[1] in ("-h", "--help"):
        print(__doc__.strip(), file=sys.stderr)  # too few arguments: say how to run it
        sys.exit(2)
    out_dir = Path(sys.argv[2])
    runs = load_runs(sys.argv[1])
    if not runs:
        sys.exit("no from-scanner_to-T1w transform under DATA/deriv")
    by_session = {}
    for run in runs:
        by_session.setdefault(run["session"], []).append(run)
    results, ids, anchors = {}, [], {}
    for s, (ses, group) in enumerate(sorted(by_session.items()), start=1):
        anchor = pick_anchor([(r["stem"], r["time"]) for r in group])
        transforms = {r["stem"]: read_itk_ras(r["xfm"]) for r in group}
        masks = {}
        for r in group:
            img = nib.load(r["mask"])
            masks[r["stem"]] = (np.asarray(img.dataobj) > 0, img.affine)
        frame = check_frame(transforms, anchor.name, masks)
        anchors[ses] = next(r for r in group if r["stem"] == anchor.name)
        print(
            f"S{s}: {len(group)} runs; the anchor is"
            f" R{[r['stem'] for r in group].index(anchor.name) + 1} by {anchor.rule}"
            f" ({anchor.why}); a shared frame is"
            f" {'plausible' if frame['shared_frame_plausible'] else 'NOT plausible'}"
        )
        results[ses] = {"anchor": anchor.name, "rule": anchor.rule, "why": anchor.why, "runs": {}}
        for i, r in enumerate(group, start=1):
            ids.append(f"S{s}\tR{i}\t{r['stem']}")
            d = frame["runs"][r["stem"]]
            pictures = measure_run(r, transforms[r["stem"]], transforms[anchor.name])
            results[ses]["runs"][r["stem"]] = {"id": f"S{s} R{i}", "frame": d, **pictures}
            print(line(f"R{i}{' (anchor)' if r['stem'] == anchor.name else ''}", d, pictures))
    labels = sorted(anchors)
    first = anchors[labels[0]]
    between = {}
    for s, ses in enumerate(labels[1:], start=2):
        other = anchors[ses]
        own, ref = read_itk_ras(other["xfm"]), read_itk_ras(first["xfm"])
        img = nib.load(other["mask"])
        d = frame_displacement(own, ref, np.asarray(img.dataobj) > 0, img.affine)
        pictures = measure_run(other, own, ref)
        between[ses] = {"frame": d, **pictures}
        print(f"between sessions: the anchor of S{s} through the transform of the anchor of S1")
        print(line(f"S{s} anchor", d, pictures))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "session_measure.json").write_text(
        json.dumps({"sessions": results, "between_sessions": between}, indent=1)
    )
    (out_dir / "session_ids.tsv").write_text("session\trun\tname\n" + "\n".join(ids) + "\n")


if __name__ == "__main__":
    main()
