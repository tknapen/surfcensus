#!/bin/bash
# The measurements for fast playback with scrubbing (README, "Fast playback and scrubbing"):
# the side-by-side video at 60 volumes per second, the same with every frame a key frame, and
# one image per volume and panel. One core. Not part of the feasibility study; added with the
# hand-over on 2026-11-13.
#
# usage: run_playback.sh DATA FS_SUBJECT_DIR OUT
#
# OUT should be new or empty. It receives the videos, timings.jsonl (one line per video, from
# make_videos.py) and keyint_check.txt, check_videos.txt, fidelity.txt and tile_sizes.txt.
# From the repository root:
#   uv run bash prototypes/func_viewer/pipeline/run_playback.sh DATA FS_SUBJECT_DIR OUT
# Every call pins itself to the first CPU that the script may use (--one-core). Two of these
# scripts started at the same time therefore share one CPU and take about twice as long; the
# results are the same. To use another CPU, start the script under `taskset -c N`.
set -Eeuo pipefail
if [ "$#" -ne 3 ]; then
    sed -n '2,15p' "$0"
    exit 2
fi
DATA=$1; FS=$2; OUT=$3
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
        echo "run_playback.sh: missing input $f" >&2
        missing=1
    fi
done
if [ "$missing" -ne 0 ]; then
    echo "run_playback.sh: DATA is what fetch.py fills, FS_SUBJECT_DIR for example what fetch_freesurfer.py wrote" >&2
    exit 2
fi

# One line per step goes to the terminal. A step that fails is named there, with the last lines
# of the text file that received its output, where the step has one.
STEP=""; LOG=""
step() {
    STEP=$1; LOG=${2:-}
    echo "run_playback.sh: $STEP" >&2
}
failed() {
    echo "run_playback.sh: failed at: $STEP" >&2
    if [ -n "$LOG" ] && [ -s "$LOG" ]; then
        echo "run_playback.sh: last lines of $LOG:" >&2
        tail -n 5 "$LOG" >&2
    fi
}
trap failed ERR

mkdir -p "$OUT"
rm -f "$OUT/timings.jsonl"

videos() {
    step "make_videos.py --video C $*"
    "$PY" -I "$S/make_videos.py" "$DATA" "$FS" "$OUT" --video C --one-core "$@"
}
# as the feasibility study made it, at 60 volumes per second: the key frames x264 chooses
videos --fps 60 --name example_side_by_side_60fps
# every frame a key frame
videos --fps 60 --keyint 1 --name example_side_by_side_60fps_intra
# the prototype's own encoder must give the package's file when asked for x264's default
videos --fps 60 --keyint 250 --name check_keyint250
if cmp -s "$OUT/example_side_by_side_60fps.mp4" "$OUT/check_keyint250.mp4"; then
    echo "gop_encode.py with --keyint 250 writes the same bytes as write_mp4" > "$OUT/keyint_check.txt"
    rm "$OUT/check_keyint250.mp4"
else
    echo "gop_encode.py with --keyint 250 DIFFERS from write_mp4" | tee "$OUT/keyint_check.txt"
    exit 1
fi
# without contours, for the comparison of decoded frames with their source (fidelity.py)
for crf in 23 12; do
    videos --fps 60 --no-contours --crf $crf --name c_nocont_60fps_crf$crf
    videos --fps 60 --no-contours --crf $crf --keyint 1 --name c_nocont_60fps_intra_crf$crf
done

step "check_videos.py"
"$PY" -I "$S/check_videos.py" "$OUT" > "$OUT/check_videos.txt"
step "fidelity.py"
"$PY" -I "$S/fidelity.py" "$DATA" "$OUT" c_nocont_60fps_crf23 c_nocont_60fps_intra_crf23 \
    c_nocont_60fps_crf12 c_nocont_60fps_intra_crf12 > "$OUT/fidelity.txt"
step "tile_sizes.py --panels"
"$PY" -I "$S/tile_sizes.py" "$DATA" --panels --fs-dir "$FS" --one-core > "$OUT/tile_sizes.txt"
echo "run_playback.sh: done; videos and text results in $OUT" >&2
