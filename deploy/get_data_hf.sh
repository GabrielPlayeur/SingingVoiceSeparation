#!/usr/bin/env bash
# (serveur) Récupère les .stem.mp4 depuis le dépôt Hugging Face privé de hf_upload.sh, via l'accélérateur réseau AutoDL, puis vérifie
# le manifeste. Reprenable. Prérequis : HF_TOKEN (lecture) dans l'environnement, HF_REPO (env.local.sh).
# Usage : deploy/get_data_hf.sh
# shellcheck source=deploy/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
require_var HF_REPO
[ -n "${HF_TOKEN:-}" ] || die "HF_TOKEN absent : read -rs HF_TOKEN; export HF_TOKEN (jeton en lecture)"
# huggingface_hub s'installe AVANT l'accélérateur (le proxy ralentit pip)
"$PYTHON_BIN" -c "import huggingface_hub" 2>/dev/null || { log "installation de huggingface_hub (miroir pip de l'image)"; "$PYTHON_BIN" -m pip install -q huggingface_hub 2>/dev/null || uv pip install -q --python "$PYTHON_BIN" huggingface_hub --index-url "${PYPI_MIRROR:-https://pypi.org/simple/}"; }
enable_net_turbo
mkdir -p "$MUSDB_RAW"
log "téléchargement de hf.co/datasets/$HF_REPO vers $MUSDB_RAW (relancer pour reprendre)"
"$PYTHON_BIN" - "$HF_REPO" "$MUSDB_RAW" <<'PY' || die "téléchargement échoué (jeton ? accélérateur réseau ? relancer le script)"
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], repo_type="dataset", local_dir=sys.argv[2], allow_patterns=["train/*.stem.mp4", "test/*.stem.mp4"], max_workers=8)
PY
"$PYTHON_BIN" "$DEPLOY_DIR/make_manifest.py" "$MUSDB_RAW" --check "$DEPLOY_DIR/manifest_musdb18.json"
log "données OK. Étape suivante : deploy/prepare_data.sh"
