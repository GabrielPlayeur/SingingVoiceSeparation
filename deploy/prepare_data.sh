#!/usr/bin/env bash
# (server) Prepares EVERYTHING that comes before the first training, in a resumable way:
#   1. checks the mp4 files (manifest)  2. mp4 -> wav (src/data/preprocessing.py, skips what exists)
#   3. split.json (identical to the local split)  4. oracle cache (avoids recomputing it in every evaluation, and the
#   concurrent accesses between parallel runs).
# Best launched in no-GPU mode (CPU only). Usage: deploy/prepare_data.sh [--workers N]
# shellcheck source=deploy/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
WORKERS="${CPU_CORES:-$(nproc)}"
if [ "${1:-}" = "--workers" ]; then WORKERS="${2:?--workers needs a number}"; fi
cd "$REPO_ROOT" || die "repository not found: $REPO_ROOT"

"$PYTHON_BIN" "$DEPLOY_DIR/make_manifest.py" "$MUSDB_RAW" --check "$DEPLOY_DIR/manifest_musdb18.json"
# space: the wavs weigh ~35 GB; we require what is left to write (+ 4 GB for runs and the oracle cache)
used_gb=$(du -s --block-size=1G "$MUSDB_PROCESSED" 2>/dev/null | cut -f1 || true)
avail_gb=$(df --output=avail --block-size=1G "$MUSDB_RAW" | tail -1 | tr -d ' G')
need_gb=$(( 39 - ${used_gb:-0} ))
if [ "$avail_gb" -lt "$need_gb" ]; then
    die "not enough space on the data disk: ${avail_gb} GB free, ~${need_gb} GB needed for the wavs (df -h; delete the zip in $DOWNLOAD_DIR or rent more disk)"
fi
log "data disk: ${avail_gb} GB free, ~${need_gb} GB to write"
log "preprocessing mp4 -> wav with $WORKERS processes (ffmpeg decoding)"
"$PYTHON_BIN" src/data/preprocessing.py --input "$MUSDB_RAW" --output "$MUSDB_PROCESSED" --workers "$WORKERS"
"$PYTHON_BIN" "$DEPLOY_DIR/prepare_data.py"
log "preparation finished. Next step: deploy/check_env.py then deploy/benchmark_parallel.sh"
