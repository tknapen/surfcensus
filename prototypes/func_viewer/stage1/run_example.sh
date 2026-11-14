#!/usr/bin/env bash
# Stage 1 on the worked example (ds000102 sub-01): every self-check of stage 1 that needs no
# listing, then a pack that the page shows. Needs no listing and sends no request.
#
# usage: run_example.sh DATA FS_SUBJECT WORK [--no-browser]
#
#   DATA        the worked example as pipeline/fetch.py writes it (deriv/, raw/)
#   FS_SUBJECT  its FreeSurfer subject as pipeline/fetch_freesurfer.py writes it (.../sub-01)
#   WORK        a new directory outside the repository for the pack and the check results
#   --no-browser  leave out check_player.py, which needs Playwright and an installed Chrome
#
# Steps: check_session.py (the anchor run is the per-run stream bit for bit; the session stream
# is deterministic; the between-run displacement), select_runs.py --self-check (the selection
# rule on made-up tables), pack_run.py twice and pack_digest.py (two packs are the same bytes),
# check_geometry.py (the raw stream recomputed with code that shares nothing with the pack's,
# and where it sits on the anatomy), check_player.py (the page shows the pack's values; frame
# rate, scrubbing, markers, screenshots). The pack carries the neutral id W01. At the end it
# prints the command that serves the pack.
set -euo pipefail

usage() {
    sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//' >&2
    exit 2
}
[ $# -ge 3 ] || usage
DATA=$1 FS=$2 WORK=$3
BROWSER=1
[ "${4:-}" = "--no-browser" ] && BROWSER=0
[ $# -le 3 ] || [ "$4" = "--no-browser" ] || usage
S=$(cd "$(dirname "$0")" && pwd)
for f in "$DATA/deriv/run-1_space-T1w_desc-preproc_bold.nii.gz" "$DATA/raw/run-2_bold.nii.gz" \
    "$FS/mri/T1.mgz" "$FS/surf/lh.pial"; do
    [ -f "$f" ] || { echo "missing input: $f" >&2; exit 2; }
done
if [ -e "$WORK" ] && [ -n "$(ls -A "$WORK")" ]; then
    echo "$WORK exists and is not empty; give a new directory" >&2
    exit 2
fi
mkdir -p "$WORK"
PY=(uv run --quiet python -I)

step() {  # step NAME LOG COMMAND...
    local name=$1 log=$2 t0
    shift 2
    t0=$(date +%s)
    if ! "$@" >"$log" 2>&1; then
        echo "step failed: $name (see $log)" >&2
        tail -5 "$log" >&2
        exit 1
    fi
    echo "ok: $name ($(($(date +%s) - t0)) s; $log)"
}

step "check_session" "$WORK/check_session.txt" "${PY[@]}" "$S/check_session.py" "$DATA"
step "select_runs --self-check" "$WORK/select_self_check.txt" "${PY[@]}" "$S/select_runs.py" --self-check
step "pack" "$WORK/pack.txt" "${PY[@]}" "$S/pack_run.py" "$DATA" "$FS" "$WORK/pack" --id W01 --map "$WORK/map.tsv"
step "pack again" "$WORK/pack2.txt" "${PY[@]}" "$S/pack_run.py" "$DATA" "$FS" "$WORK/pack2" --id W01
step "pack_digest: the two packs are the same bytes" "$WORK/digest.txt" "${PY[@]}" "$S/pack_digest.py" "$WORK/pack" "$WORK/pack2"
rm -rf "$WORK/pack2"
step "check_geometry: independent placement of the raw stream" "$WORK/check_geometry.txt" "${PY[@]}" "$S/check_geometry.py" "$WORK/pack" "$WORK/map.tsv" "$DATA" --json "$WORK/check_geometry.json"
if [ $BROWSER = 1 ]; then
    step "check_player" "$WORK/check_player.txt" "${PY[@]}" "$S/check_player.py" "$WORK/pack" "$WORK/map.tsv" "$DATA" "$WORK/check"
fi
echo "the pack: $WORK/pack; to look at it:"
echo "  uv run python -I $S/serve.py $WORK/pack     # then open the address it prints"
