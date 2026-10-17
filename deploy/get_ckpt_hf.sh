#!/usr/bin/env bash
# (local) Fetches the slimmed checkpoints uploaded by push_ckpt_hf.sh from the private Hugging Face model repository, into
# outputs/server/<tag>/<run>/ (same place as fetch_results.sh --ckpt), then copies each one where the analysis scripts read them:
# outputs/VoiceSeparation/<W&B id>/checkpoints/best_slim.ckpt (the name is historical: it is the checkpoint of the queue entry, last or best).
# Prerequisites: READ token in HF_TOKEN (read -rs HF_TOKEN; export HF_TOKEN), HF_REPO or HF_CKPT_REPO (env.local.sh).
# Usage: deploy/get_ckpt_hf.sh TAG RUN [RUN...]
# shellcheck source=deploy/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
TAG="${1:?usage: get_ckpt_hf.sh TAG RUN [RUN...]}"; shift
[ $# -gt 0 ] || die "give at least one run name"
[ -n "${HF_TOKEN:-}" ] || die "HF_TOKEN missing: read -rs HF_TOKEN; export HF_TOKEN (read token)"
REPO="${HF_CKPT_REPO:-}"
if [ -z "$REPO" ]; then require_var HF_REPO; REPO="${HF_REPO%%/*}/svs-checkpoints"; fi
require_cmd uv "https://github.com/astral-sh/uv"
patterns=()
for run in "$@"; do patterns+=("$TAG/$run/*"); done
log "downloading ${patterns[*]} from hf.co/$REPO -> $LOCAL_RESULTS_DIR"
uv run --no-project --with huggingface_hub python - "$REPO" "$LOCAL_RESULTS_DIR" "${patterns[@]}" <<'PY' || die "download failed: rerun the same command"
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], repo_type="model", local_dir=sys.argv[2], allow_patterns=sys.argv[3:])
PY
for run in "$@"; do
    d="$LOCAL_RESULTS_DIR/$TAG/$run"
    [ -f "$d/checkpoint_slim.ckpt" ] || die "$d/checkpoint_slim.ckpt missing after download (pushed with push_ckpt_hf.sh?)"
    id="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["wandb_run_id"])' "$d/metadata.json")"
    mkdir -p "$REPO_ROOT/outputs/VoiceSeparation/$id/checkpoints"
    cp "$d/checkpoint_slim.ckpt" "$REPO_ROOT/outputs/VoiceSeparation/$id/checkpoints/best_slim.ckpt"
    log "$run -> outputs/VoiceSeparation/$id/checkpoints/best_slim.ckpt ($(du -h "$d/checkpoint_slim.ckpt" | cut -f1))"
done
