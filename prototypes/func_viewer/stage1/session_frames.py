"""Do the runs of a session share a scanner frame? The stored transforms of many sessions.

usage: session_frames.py OUT    (after session_find.py SURVEY_DIR OUT)

For the first subject of every older derivative that session_find.py listed, this fetches the
from-scanner_to-T1w transform (text, about 230 bytes) and the T1w-space brain mask (a few kB) of
every run, lists the subject in the raw bucket and fetches the JSON sidecars of its runs where
there are any. Anonymous, read-only, at most 4 requests at a time; a file that is already under
OUT/frames is not fetched again. No BOLD series is read.

Per session it picks the anchor (session.pick_anchor) and measures, for every other run, the
displacement between the run's own transform and the anchor's over the run's brain mask
(session.frame_displacement). For a subject with several sessions it also measures the anchor of
every later session against the anchor of the first: two runs that are known not to share a
scanner frame. Writes OUT/session_frames.json, which names datasets and stays out of the
repository, and prints the two distributions without a name.
"""

import io
import json
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))  # the sibling module, also under -I
sys.path.append(str(Path(__file__).resolve().parent.parent / "survey"))  # the listing functions
from session import (
    FRAME_LIMIT_MM,
    RULE_TIME,
    check_frame,
    frame_displacement,
    name_order,
    parse_name,
    pick_anchor,
    read_itk_ras,
    run_key,
    session_of,
    sidecar_time,
)
from survey import DERIV, MAX_REQUESTS, RAW, get, list_prefix

XFM = "_from-scanner_to-T1w_mode-image_xfm.txt"
MASK = "_space-T1w_desc-brain_mask.nii.gz"
QUANTILES = (0, 25, 50, 75, 90, 95, 100)


def fetch(base, key, dst):
    """One object into dst unless it is there; True when dst then exists."""
    if dst.exists():
        return True
    r = get(f"{base}/{key}")
    if r.status_code != 200:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    tmp.write_bytes(r.content)
    tmp.rename(dst)
    return True


def raw_grid(key, folder, name):
    """Shape and affine of a raw series from the first 4 kB of the file; None when unreadable."""
    dst = folder / "raw" / (name + "_bold_header.json")
    if dst.exists():
        return json.loads(dst.read_text())
    if key is None:
        return None
    r = get(f"{RAW}/{key}", headers={"Range": "bytes=0-4095"})
    if r.status_code not in (200, 206):
        return None
    try:
        head = zlib.decompressobj(31).decompress(r.content)
        hdr = nib.Nifti1Header.from_fileobj(io.BytesIO(head[:348]), check=False)
        grid = {
            "shape": [int(n) for n in hdr.get_data_shape()],
            "affine": [[float(x) for x in row] for row in hdr.get_best_affine()],
        }
    except (zlib.error, ValueError, OSError):
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(grid))
    return grid


def subject_runs(item, out_dir):
    """The runs of one listed first subject, with their small files fetched."""
    prefix, subject = item["prefix"], item["subject"]
    accession = prefix.split("/")[1].rsplit("-", 1)[0]
    label = subject.rstrip("/").rsplit("/", 1)[-1]
    folder = out_dir / "frames" / accession
    keys = {k for k, _ in item["keys"]}
    stems = sorted(k[: -len(XFM)] for k in keys if k.endswith(XFM) and "/func/" in k)
    raw_keys, _ = list_prefix(RAW, f"{accession}/{label}/")
    sidecars, series = {}, {}
    for key, _, _, _ in raw_keys:
        ent = parse_name(key)
        if "/func/" in key and key.endswith("_bold.json"):
            sidecars[session_of(ent), run_key(ent)] = key
        elif "/func/" in key and key.endswith("_bold.nii.gz") and "echo" not in ent:
            series[session_of(ent), run_key(ent)] = key
    runs = []
    for stem in stems:
        name = stem.rsplit("/", 1)[-1]
        ent = parse_name(name)
        run = {"name": name, "session": session_of(ent), "xfm": None, "mask": None, "time": None}
        run["grid"] = raw_grid(series.get((session_of(ent), run_key(ent))), folder, name)
        if fetch(DERIV, stem + XFM, folder / (name + XFM)):
            run["xfm"] = str(folder / (name + XFM))
        if stem + MASK in keys and fetch(DERIV, stem + MASK, folder / (name + MASK)):
            run["mask"] = str(folder / (name + MASK))
        side = sidecars.get((session_of(ent), run_key(ent)))
        if side and fetch(RAW, side, folder / "raw" / (name + "_bold.json")):
            run["time"] = sidecar_time(folder / "raw" / (name + "_bold.json"))
        runs.append(run)
    return {
        "prefix": prefix,
        "subject": label,
        "runs": runs,
        "raw_subject_listed": bool(raw_keys),
        "n_raw_sidecars": len(sidecars),
    }


def displacement(run, anchor):
    """frame_displacement of a run against an anchor over the run's own mask; None if unusable."""
    if not (run["xfm"] and anchor["xfm"] and run["mask"]):
        return None
    try:
        own, ref = read_itk_ras(run["xfm"]), read_itk_ras(anchor["xfm"])
    except ValueError:
        return None
    img = nib.load(run["mask"])
    return frame_displacement(own, ref, np.asarray(img.dataobj) > 0, img.affine)


def anchor_is_outlier(runs, anchor):
    """session.check_frame's verdict on the anchor; False when a run lacks a file."""
    try:
        transforms = {r["name"]: read_itk_ras(r["xfm"]) for r in runs if r["xfm"] and r["mask"]}
    except ValueError:
        return False
    if len(transforms) < len(runs):
        return False
    masks = {}
    for r in runs:
        img = nib.load(r["mask"])
        masks[r["name"]] = (np.asarray(img.dataobj) > 0, img.affine)
    return check_frame(transforms, anchor, masks)["anchor_is_outlier"]


def measure(subject):
    """Within-session and between-session displacements of one subject."""
    by_session = {}
    for run in subject["runs"]:
        by_session.setdefault(run["session"], []).append(run)
    anchors, within, between = {}, [], []
    for ses, runs in sorted(by_session.items()):
        anchor = pick_anchor([(r["name"], r["time"]) for r in runs])
        by_name = pick_anchor([(r["name"], None) for r in runs])
        anchors[ses] = {
            "name": anchor.name,
            "rule": anchor.rule,
            "why": anchor.why,
            "n_runs": len(runs),
            "name_order_agrees": by_name.name == anchor.name,
            "anchor_is_outlier": anchor_is_outlier(runs, anchor.name),
        }
        ref = next(r for r in runs if r["name"] == anchor.name)
        for run in sorted(runs, key=lambda r: name_order(parse_name(r["name"]))):
            d = None if run is ref else displacement(run, ref)
            if d is not None:
                same = None
                if run["grid"] and ref["grid"]:
                    same = bool(
                        np.allclose(run["grid"]["affine"], ref["grid"]["affine"], atol=1e-3)
                    )
                within.append({"session": ses, "name": run["name"], "same_grid": same, **d})
    labels = sorted(anchors)
    first = next(r for r in by_session[labels[0]] if r["name"] == anchors[labels[0]]["name"])
    for ses in labels[1:]:
        other = next(r for r in by_session[ses] if r["name"] == anchors[ses]["name"])
        d = displacement(other, first)
        if d is not None:
            between.append({"session": ses, "name": other["name"], **d})
    return {**subject, "anchors": anchors, "within": within, "between": between}


def quantiles(values):
    q = np.percentile(values, QUANTILES)
    return ", ".join(f"p{p} {v:.2f}" for p, v in zip(QUANTILES, q, strict=True))


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__.strip(), file=sys.stderr)  # too few arguments: say how to run it
        sys.exit(2)
    out_dir = Path(sys.argv[1])
    listed = json.loads((out_dir / "session_candidates.json").read_text())
    listed = [item for item in listed if item["subject"]]
    with ThreadPoolExecutor(max_workers=MAX_REQUESTS) as pool:
        subjects = list(pool.map(lambda item: subject_runs(item, out_dir), listed))
    results = [measure(s) for s in subjects if s["runs"]]
    (out_dir / "session_frames.json").write_text(json.dumps(results, indent=1))

    n_runs = sum(len(s["runs"]) for s in results)
    n_sessions = sum(len(s["anchors"]) for s in results)
    by_time = sum(a["rule"] == RULE_TIME for s in results for a in s["anchors"].values())
    with_sidecars = sum(s["n_raw_sidecars"] > 0 for s in results)
    not_in_raw = sum(not s["raw_subject_listed"] for s in results)
    print(
        f"{len(results)} first subjects with a stored transform, {n_sessions} sessions,"
        f" {n_runs} runs; anchor by AcquisitionTime in {by_time} sessions, by name order in"
        f" {n_sessions - by_time}; {with_sidecars} subjects have a JSON sidecar beside a raw"
        f" run; {not_in_raw} subjects are absent from the raw bucket"
    )
    within = [d for s in results for d in s["within"]]
    between = [d for s in results for d in s["between"]]
    sessions_with_pairs = {(s["prefix"], d["session"]) for s in results for d in s["within"]}
    worst = {}
    for s in results:
        for d in s["within"]:
            key = (s["prefix"], d["session"])
            worst[key] = max(worst.get(key, 0.0), d["max_mm"])
    for label, rows in (("within a session, run against anchor", within), ("between", between)):
        if not rows:
            continue
        print(f"{label}: {len(rows)} pairs")
        for field in ("max_mm", "mean_mm", "rotation_deg", "translation_mm"):
            print(f"  {field:15s} {quantiles([d[field] for d in rows])}")
    print(
        f"sessions with at least two runs: {len(sessions_with_pairs)}; largest max_mm per"
        f" session: {quantiles(list(worst.values()))}"
    )
    over = sorted(v for v in worst.values() if v > FRAME_LIMIT_MM)
    print(
        f"sessions with a run beyond the limit of {FRAME_LIMIT_MM} mm: {len(over)}"
        f" ({', '.join(f'{v:.1f}' for v in over)})"
    )
    under = sorted(d["max_mm"] for d in between if d["max_mm"] <= FRAME_LIMIT_MM)
    print(
        f"between-session pairs within the limit: {len(under)} of {len(between)}"
        f" ({', '.join(f'{v:.1f}' for v in under)})"
    )
    multi = [a for s in results for a in s["anchors"].values() if a["n_runs"] > 1]
    timed = [a for a in multi if a["rule"] == RULE_TIME]
    print(
        f"sessions with at least two runs and the anchor by AcquisitionTime: {len(timed)};"
        f" name order would pick another run in {sum(not a['name_order_agrees'] for a in timed)}"
    )
    print(
        "sessions in which every other run is beyond the limit and they agree with each other"
        f" (the anchor is the odd one): {sum(a['anchor_is_outlier'] for a in multi)}"
    )
    for label, same in (("the same", True), ("another", False)):
        rows = [d["max_mm"] for d in within if d["same_grid"] is same]
        if rows:
            print(
                f"pairs whose raw header has {label} affine as the anchor's: {len(rows)},"
                f" {sum(v > FRAME_LIMIT_MM for v in rows)} beyond the limit; max_mm"
                f" {quantiles(rows)}"
            )
    shear = max((d["shear_scale"] for d in within + between), default=0.0)
    print(f"largest deviation of a pair from a rigid motion (singular value - 1): {shear:.2e}")


if __name__ == "__main__":
    main()
