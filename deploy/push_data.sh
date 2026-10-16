#!/usr/bin/env bash
# (local) Sends the .stem.mp4 files of MUSDB18 (~5 GB) to the server with rsync, resumable. Does NOT send processed/ (35 GB).
# Usage: deploy/push_data.sh
# shellcheck source=deploy/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
ssh_setup
require_cmd rsync "apt install rsync"
[ -d "$LOCAL_MUSDB/train" ] || die "local dataset not found: $LOCAL_MUSDB/train (variable LOCAL_MUSDB)"
remote "mkdir -p '$MUSDB_RAW'"
log "rsync $LOCAL_MUSDB -> $SSH_TARGET:$MUSDB_RAW (rerun the same command after a disconnection)"
rsync -avP --prune-empty-dirs -e "$RSYNC_SSH" \
    --include='train/' --include='test/' --include='*.stem.mp4' --exclude='*' \
    "$LOCAL_MUSDB/" "$SSH_TARGET:$MUSDB_RAW/"
log "checking the manifest on the server"
remote "cd '$SERVER_REPO' && python3 deploy/make_manifest.py '$MUSDB_RAW' --check deploy/manifest_musdb18.json" \
    || die "the server does not have the same files as the manifest: rerun this script"
log "data OK. Next step, on the server: deploy/prepare_data.sh"
