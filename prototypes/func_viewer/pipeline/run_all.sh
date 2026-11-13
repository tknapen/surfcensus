#!/bin/bash
# Everything of the feasibility study after the download, in order.
#
# usage: run_all.sh DATA FS_SUBJECT_DIR OUT WORK
#
#   DATA            the directory that fetch.py filled
#   FS_SUBJECT_DIR  FreeSurfer subject directory of ds000102 sub-01 (mri/T1.mgz, surf/?h.white,
#                   surf/?h.pial), for example what fetch_freesurfer.py wrote
#   OUT             the seven example videos, the frames picked from them, OUT/checks
#   WORK            text and JSON results, and the videos made only to be measured
#
# OUT and WORK should be new or empty: a video left there from an earlier run is decoded and
# reported with the others. The interpreter is $PYTHON, else `python`; from the repository root:
#   uv run bash prototypes/func_viewer/pipeline/run_all.sh DATA FS_SUBJECT_DIR OUT WORK
# -I keeps the interpreter from importing anything that lies next to the data.
# Every call pins itself to the first CPU that the script may use (--one-core). Two of these
# scripts started at the same time therefore share one CPU and take about twice as long; the
# results are the same. To use another CPU, start the script under `taskset -c N`.
set -Eeuo pipefail
if [ "$#" -ne 4 ]; then
    sed -n '2,18p' "$0"
    exit 2
fi
DATA=$1; FS=$2; OUT=$3; WORK=$4
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=${PYTHON:-python}

# Inputs are checked before anything is made, so a wrong path is said on the terminal.
missing=0
for f in "$DATA"/raw/run-{1,2}_bold.nii.gz \
    "$DATA"/deriv/run-{1,2}_space-T1w_desc-preproc_bold.nii.gz \
    "$DATA"/deriv/run-{1,2}_from-scanner_to-T1w_mode-image_xfm.txt \
    "$DATA"/deriv/run-{1,2}_desc-confounds_timeseries.tsv \
    "$DATA"/deriv/anat_desc-preproc_T1w.nii.gz \
    "$DATA"/deriv/anat_from-T1w_to-fsnative_mode-image_xfm.txt \
    "$FS"/mri/T1.mgz "$FS"/surf/{lh,rh}.{white,pial}; do
    if [ ! -f "$f" ]; then
        echo "run_all.sh: missing input $f" >&2
        missing=1
    fi
done
if [ "$missing" -ne 0 ]; then
    echo "run_all.sh: DATA is what fetch.py fills, FS_SUBJECT_DIR for example what fetch_freesurfer.py wrote" >&2
    exit 2
fi

# One line per step goes to the terminal. A step that fails is named there, with the last lines
# of the text file that received its output, where the step has one.
STEP=""; LOG=""
step() {
    STEP=$1; LOG=${2:-}
    echo "run_all.sh: $STEP" >&2
}
failed() {
    echo "run_all.sh: failed at: $STEP" >&2
    if [ -n "$LOG" ] && [ -s "$LOG" ]; then
        echo "run_all.sh: last lines of $LOG:" >&2
        tail -n 5 "$LOG" >&2
    fi
}
trap failed ERR

mkdir -p "$OUT/checks" "$WORK"
rm -f "$OUT/timings.jsonl" "$WORK/timings.jsonl"

step "verify_geometry.py" "$WORK/geometry.txt"
"$PY" -I "$S/verify_geometry.py" "$DATA" "$FS" "$WORK/geometry.json" 2>&1 \
    | grep -v -e Warning -e "c /=" > "$WORK/geometry.txt"
step "overlays.py"
"$PY" -I "$S/overlays.py" "$DATA" "$FS" "$OUT/checks" > "$WORK/overlays.txt"

videos() {  # the arguments common to every call of make_videos.py
    step "make_videos.py ${*:2}"
    "$PY" -I "$S/make_videos.py" "$DATA" "$FS" "$@" --one-core
}
for v in A B C; do
    videos "$OUT" --video $v --fps 12 --png 0,59,60,95,145,146,200
done
for v in A B C; do
    n=$(echo $v | sed 's/A/example_raw_rigid/;s/B/example_preproc/;s/C/example_side_by_side/')
    videos "$OUT" --video $v --fps 24 --name "${n}_24fps"
done
videos "$OUT" --video C --fps 12 --no-contours --name example_side_by_side_no_contours --png 59
videos "$WORK" --video C --fps 12 --scale 1.5 --name c_scale15
videos "$WORK" --video A --fps 12 --scale 3 --name a_scale3
for crf in 12 18 28; do
    videos "$WORK" --video C --fps 12 --no-contours --crf $crf --name c_nocont_crf$crf
done
videos "$WORK" --video B --fps 12 --no-video --no-contours --name b_novideo_nocontours
cp "$OUT/example_side_by_side_no_contours.mp4" "$WORK/c_nocont_crf23.mp4"

step "check_videos.py"
"$PY" -I "$S/check_videos.py" "$OUT" > "$WORK/check_videos.txt"
step "analyze.py" "$WORK/analysis.txt"
"$PY" -I "$S/analyze.py" "$DATA" "$OUT/checks" 2>&1 | grep -v Warn > "$WORK/analysis.txt"
mv "$OUT/checks/analysis.json" "$WORK/"
step "fidelity.py"
"$PY" -I "$S/fidelity.py" "$DATA" "$WORK" > "$WORK/fidelity.txt"
step "tile_sizes.py"
"$PY" -I "$S/tile_sizes.py" "$DATA" > "$WORK/tile_sizes.txt"
echo "run_all.sh: done; videos and frames in $OUT, text results in $WORK" >&2
