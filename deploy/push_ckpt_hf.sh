#!/usr/bin/env bash
# (server) Checkpoint relay through Hugging Face: slims the checkpoint of each run (weights + hyper-parameters, the one chosen by
# its queue entry: best or last, deploy/slim_ckpt.py) and uploads it with the run's metadata.json to a PRIVATE model repository,
# path <tag>/<run>/. The PC fetches them with get_ckpt_hf.sh. Why: rsync of a 39-157 MB file from AutoDL to the PC takes hours,
# while the AutoDL network accelerator covers Hugging Face (measured with the MUSDB18 relay, get_data_hf.sh).
# Works from the repository or from the bootstrap copy of deploy/ (push_bootstrap.sh): it uses $SERVER_REPO/.venv explicitly.
# Prerequisites: WRITE token in HF_TOKEN (read -rs HF_TOKEN; export HF_TOKEN), HF_REPO or HF_CKPT_REPO (env.local.sh).
# Usage: deploy/push_ckpt_hf.sh TAG RUN [RUN...]     e.g. deploy/push_ckpt_hf.sh final-v2 f_legacy_s42 f_power_loss_s42
# shellcheck source=deploy/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
TAG="${1:?usage: push_ckpt_hf.sh TAG RUN [RUN...]}"; shift
[ $# -gt 0 ] || die "give at least one run name (ls $RUNS_DIR/$TAG)"
[ -n "${HF_TOKEN:-}" ] || die "HF_TOKEN missing: read -rs HF_TOKEN; export HF_TOKEN (token with write access, never in a file)"
REPO="${HF_CKPT_REPO:-}"
if [ -z "$REPO" ]; then require_var HF_REPO; REPO="${HF_REPO%%/*}/svs-checkpoints"; fi
PY="$SERVER_REPO/.venv/bin/python"
[ -x "$PY" ] || die "$PY not found (setup_server.sh done?)"
SLIM="$DEPLOY_DIR/slim_ckpt.py"
"$PY" -c "import huggingface_hub" 2>/dev/null || { log "installing huggingface_hub (the image's pip mirror)"; uv pip install -q --python "$PY" huggingface_hub --index-url "${PYPI_MIRROR:-https://pypi.org/simple/}"; }

files=()
for run in "$@"; do
    d="$RUNS_DIR/$TAG/$run"
    [ -f "$d/metadata.json" ] || die "$d/metadata.json not found (exact run name? ls $RUNS_DIR/$TAG)"
    if [ ! -f "$d/checkpoint_slim.ckpt" ]; then
        log "slimming the checkpoint of $run"
        "$PY" "$SLIM" "$d" "$d/checkpoint_slim.ckpt" || die "cannot slim the checkpoint of $run"
    fi
    files+=("$d/checkpoint_slim.ckpt" "$d/metadata.json")
done

# Xet (default of recent huggingface_hub) goes through a CDN that the AutoDL proxy does not cover (measured, get_data_hf.sh).
export HF_HUB_DISABLE_XET=1
enable_net_turbo
log "uploading ${#files[@]} file(s) to hf.co/$REPO (private model repository), path $TAG/<run>/"
"$PY" - "$REPO" "$RUNS_DIR" "${files[@]}" <<'PY' || die "upload failed (token? accelerator?): rerun the same command"
import os, sys
from huggingface_hub import HfApi
repo, runs_dir, files = sys.argv[1], sys.argv[2], sys.argv[3:]
api = HfApi()
api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
for f in files:
    dest = os.path.relpath(f, runs_dir)                         # <tag>/<run>/<file>
    api.upload_file(path_or_fileobj=f, path_in_repo=dest, repo_id=repo, repo_type="model", commit_message=f"add {dest}")
    print(f"  {dest} ({os.path.getsize(f) / 1e6:.1f} MB)", flush=True)
PY
log "OK. On the PC: deploy/get_ckpt_hf.sh $TAG $*"
