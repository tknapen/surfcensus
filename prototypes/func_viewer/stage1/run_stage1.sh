#!/usr/bin/env bash
# Stage 1 on the selection: for each selected id, fetch its session, measure it, pack it under
# its neutral id, delete the downloaded series, keep the pack and one line of measurements.
#
# usage: run_stage1.sh LISTING SELECTION WORK PACKS [--runs ID=N,N]... [--check ID]... [ID ...]
#
#   LISTING    the directory of list_runs.py          SELECTION  the directory of select_runs.py
#   WORK       a directory for the downloads and the logs of each id (names datasets)
#   PACKS      a directory for the packs, one folder per id (names nothing)
#   --runs ID=N,N   the runs of that session to fetch and pack, by number in the session
#                   (default: fetch_run.py's rule, at most 4 runs: anchor, selected, name order)
#   --check ID      run check_player.py on that pack before its series are deleted
#   ID ...          the ids to do, in this order (default: H01 to H06, Q01, Q02)
#
# All four directories lie outside the repository. Run from the repository root. An id whose
# pack is complete (PACKS/ID/manifest.json) and whose line is in WORK/measurements.jsonl is
# skipped, so the script can be started again after an interruption; a pack without a manifest
# is removed and made again. Each step prints one line and the script names a step that fails.
set -euo pipefail

usage() {
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//' >&2
    exit 2
}
[ $# -ge 4 ] || usage
LISTING=$1 SELECTION=$2 WORK=$3 PACKS=$4
shift 4
S=$(cd "$(dirname "$0")" && pwd)
declare -A RUNS=() CHECK=()
IDS=()
while [ $# -gt 0 ]; do
    case $1 in
    --runs) [ $# -ge 2 ] || usage; RUNS[${2%%=*}]=${2#*=}; shift 2 ;;
    --check) [ $# -ge 2 ] || usage; CHECK[$2]=1; shift 2 ;;
    --*) usage ;;
    *) IDS+=("$1"); shift ;;
    esac
done
[ ${#IDS[@]} -gt 0 ] || IDS=(H01 H02 H03 H04 H05 H06 Q01 Q02)
[ -f "$LISTING/runs.parquet" ] || { echo "no runs.parquet in $LISTING" >&2; exit 2; }
[ -f "$SELECTION/selection.parquet" ] || { echo "no selection.parquet in $SELECTION" >&2; exit 2; }
mkdir -p "$WORK" "$PACKS"
PY=(uv run --quiet python -I)

step() {  # step NAME LOG COMMAND...: run, keep the output in LOG, say what failed
    local name=$1 log=$2
    shift 2
    echo "  $name"
    if ! "$@" >"$log" 2>&1; then
        echo "step failed: $name (see $log)" >&2
        tail -5 "$log" >&2
        exit 1
    fi
}

for ID in "${IDS[@]}"; do
    W=$WORK/$ID
    if [ -f "$PACKS/$ID/manifest.json" ] && grep -q "\"id\": \"$ID\"" "$WORK/measurements.jsonl" 2>/dev/null; then
        echo "$ID: packed already"
        continue
    fi
    echo "$ID: $(date -u +%H:%M:%S) UTC"
    mkdir -p "$W"
    extra=()
    [ -n "${RUNS[$ID]:-}" ] && extra=(--runs "${RUNS[$ID]}")
    step "fetch" "$W/fetch.out" "${PY[@]}" "$S/fetch_run.py" "$LISTING" "$SELECTION" "$ID" "$W/data" "${extra[@]}"
    head -1 "$W/fetch.out" | sed 's/^/    /'
    tail -1 "$W/fetch.out" >"$W/fetch.json"
    step "measure" "$W/measure.out" "${PY[@]}" "$S/session_measure.py" "$W/data" "$W/measure"
    [ -f "$PACKS/$ID/manifest.json" ] || rm -rf "${PACKS:?}/$ID"
    step "pack" "$W/pack.out" "${PY[@]}" "$S/pack_run.py" "$W/data" "$W/data/fs" "$PACKS/$ID" --id "$ID" --map "$W/map.tsv"
    tail -1 "$W/pack.out" >"$W/pack.json"
    if [ -n "${CHECK[$ID]:-}" ]; then
        step "check_player" "$W/check.out" "${PY[@]}" "$S/check_player.py" "$PACKS/$ID" "$W/map.tsv" "$W/data" "$W/check"
    fi
    find "$W/data" \( -name '*_bold.nii.gz' -o -name '*_bold.nii' -o -name '*.part' \) -delete
    echo "  series deleted; $(du -sh "$PACKS/$ID" | cut -f1) packed"
    "${PY[@]}" "$S/run_line.py" "$SELECTION" "$W" "$PACKS/$ID" >>"$WORK/measurements.jsonl"
done
echo "done: one line per id in $WORK/measurements.jsonl"
