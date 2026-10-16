#!/usr/bin/env bash
# (server) Fetches the .stem.mp4 files from the private Hugging Face repository of hf_upload.sh, through the AutoDL network accelerator,
# then checks the manifest. Resumable. Prerequisites: HF_TOKEN (read) in the environment, HF_REPO (env.local.sh).
# Usage: deploy/get_data_hf.sh
# shellcheck source=deploy/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
require_var HF_REPO
[ -n "${HF_TOKEN:-}" ] || die "HF_TOKEN missing: read -rs HF_TOKEN; export HF_TOKEN (read token)"
# huggingface_hub is installed BEFORE the accelerator (the proxy slows pip down)
"$PYTHON_BIN" -c "import huggingface_hub" 2>/dev/null || { log "installing huggingface_hub (the image's pip mirror)"; "$PYTHON_BIN" -m pip install -q huggingface_hub 2>/dev/null || uv pip install -q --python "$PYTHON_BIN" huggingface_hub --index-url "${PYPI_MIRROR:-https://pypi.org/simple/}"; }
# Xet (default of recent huggingface_hub versions) goes through a CDN that the AutoDL proxy does not cover: "error sending request" (measured).
export HF_HUB_DISABLE_XET=1
enable_net_turbo
mkdir -p "$MUSDB_RAW"
log "downloading hf.co/datasets/$HF_REPO to $MUSDB_RAW (rerun to resume)"
"$PYTHON_BIN" - "$HF_REPO" "$MUSDB_RAW" <<'PY' || die "download failed (token? network accelerator? rerun the script)"
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], repo_type="dataset", local_dir=sys.argv[2], allow_patterns=["train/*.stem.mp4", "test/*.stem.mp4"], max_workers=8)
PY
"$PYTHON_BIN" "$DEPLOY_DIR/make_manifest.py" "$MUSDB_RAW" --check "$DEPLOY_DIR/manifest_musdb18.json"
log "data OK. Next step: deploy/prepare_data.sh"
