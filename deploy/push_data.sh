#!/usr/bin/env bash
# (local) Envoie les .stem.mp4 de MUSDB18 (~5 Go) vers le serveur par rsync, reprenable. N'envoie PAS processed/ (35 Go).
# Usage : deploy/push_data.sh
# shellcheck source=deploy/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
ssh_setup
require_cmd rsync "apt install rsync"
[ -d "$LOCAL_MUSDB/train" ] || die "dataset local introuvable : $LOCAL_MUSDB/train (variable LOCAL_MUSDB)"
remote "mkdir -p '$MUSDB_RAW'"
log "rsync $LOCAL_MUSDB -> $SSH_TARGET:$MUSDB_RAW (relancer la même commande après une coupure)"
rsync -avP --prune-empty-dirs -e "$RSYNC_SSH" \
    --include='train/' --include='test/' --include='*.stem.mp4' --exclude='*' \
    "$LOCAL_MUSDB/" "$SSH_TARGET:$MUSDB_RAW/"
log "vérification du manifeste sur le serveur"
remote "cd '$SERVER_REPO' && python3 deploy/make_manifest.py '$MUSDB_RAW' --check deploy/manifest_musdb18.json" \
    || die "le serveur n'a pas les mêmes fichiers que le manifeste : relancer ce script"
log "données OK. Étape suivante, sur le serveur : deploy/prepare_data.sh"
