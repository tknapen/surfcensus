"""One line of measurements for a packed session, by neutral ids only.

usage: run_line.py SELECTION WORK_ID PACK

SELECTION  the directory of select_runs.py (selection.parquet)
WORK_ID    the work folder of one id that run_stage1.sh made: fetch.json (the last line of
           fetch_run.py), pack.json (the last line of pack_run.py), map.tsv (pack_run.py --map)
           and measure/session_measure.json (session_measure.py)
PACK       the pack directory (manifest.json)

Prints one JSON line: the id; per run of the pack its id in the pack (R1, ...), its id in the
session (<ID>-NN), whether it is the anchor, volumes, the frame check (mean and maximum
displacement from the anchor in mm, flagged or not) and the correlation inside the brain mask of
the temporal mean of the raw stream with the temporal mean of the preprocessed series, through
the run's own transform and through the anchor's; the grid; the pack size; bytes, requests and
seconds of the fetch and seconds of the pack. It names no dataset, subject, task or file.
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))  # the sibling module, also under -I
from select_runs import read_selection

PREPROC = "_space-T1w_desc-preproc_bold.nii.gz"


def main():
    if len(sys.argv) != 4:
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(2)
    sel_dir, work, pack = (Path(p) for p in sys.argv[1:])
    manifest = json.loads((pack / "manifest.json").read_text())
    fetch = json.loads((work / "fetch.json").read_text())
    packed = json.loads((work / "pack.json").read_text())
    measure = json.loads((work / "measure" / "session_measure.json").read_text())
    sel = read_selection(sel_dir / "selection.parquet")
    rows = sel[sel["id"] == manifest["id"]]
    run_id = {}
    for key, rid in zip(rows.preproc_key, rows.run_id, strict=True):
        run_id[key.rsplit("/", 1)[1][: -len(PREPROC)]] = rid
    stems = {}
    for line in (work / "map.tsv").read_text().splitlines()[1:]:
        cells = line.split("\t")
        if cells[1] != "left out":
            stems[cells[1]] = cells[3]
    series = {}
    for ses in measure["sessions"].values():
        for stem, entry in ses["runs"].items():
            series[stem] = entry.get("series_mean", {})
    runs = []
    for run in manifest["runs"]:
        stem = stems[run["id"]]
        s = series.get(stem, {})
        f = run["frame"]
        runs.append(
            {
                "pack_run": run["id"],
                "session_run": run_id.get(stem),
                "anchor": run["anchor"],
                "volumes": run["n_volumes"],
                "tr_s": run["tr_s"],
                "frame_mean_mm": round(f["mean_mm"], 3),
                "frame_max_mm": round(f["max_mm"], 3),
                "flagged": f["flagged"],
                "r_mean_own": round(s["r_own"], 4) if "r_own" in s else None,
                "r_mean_anchor": round(s["r_anchor"], 4) if "r_anchor" in s else None,
                "pack_bytes": packed["runs"][run["id"]]["bytes"],
                "pack_s": packed["runs"][run["id"]]["seconds"],
            }
        )
    line = {
        "id": manifest["id"],
        "grid": manifest["grid"]["shape"],
        "voxel_mm": manifest["grid"]["voxel_mm"],
        "session_runs": len(rows),
        "runs_packed": len(runs),
        "shared_frame_plausible": manifest["registration"]["shared_frame_plausible"],
        "anchor_is_outlier": manifest["registration"]["anchor_is_outlier"],
        "anchor_rule": manifest["registration"]["rule"],
        "pack_bytes": packed["bytes"],
        "pack_s": packed["seconds"],
        "fetch_bytes": fetch["bytes_transferred"],
        "fetch_requests": fetch["requests"],
        "fetch_s": fetch["seconds"],
        "runs": runs,
    }
    print(json.dumps(line, sort_keys=True))


if __name__ == "__main__":
    main()
