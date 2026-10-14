#!/usr/bin/env bash
# (local) Copie deploy/ sur le serveur AVANT le premier clone : setup_server.sh doit exister là-bas pour cloner le dépôt.
# Usage : deploy/push_bootstrap.sh
# shellcheck source=deploy/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
ssh_setup
require_cmd rsync "apt install rsync"
log "copie de deploy/ vers $SSH_TARGET:$BOOTSTRAP_DIR/deploy/"
remote "mkdir -p '$BOOTSTRAP_DIR/deploy'"
rsync -az -e "$RSYNC_SSH" --exclude='__pycache__' "$DEPLOY_DIR/" "$SSH_TARGET:$BOOTSTRAP_DIR/deploy/"
log "OK. Étape suivante, sur le serveur : bash $BOOTSTRAP_DIR/deploy/setup_server.sh $GIT_TAG"
