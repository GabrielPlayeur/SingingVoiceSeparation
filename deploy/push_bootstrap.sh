#!/usr/bin/env bash
# (local) Copies deploy/ to the server BEFORE the first clone: setup_server.sh must exist there to clone the repository.
# Usage: deploy/push_bootstrap.sh
# shellcheck source=deploy/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
ssh_setup
require_cmd rsync "apt install rsync"
log "copying deploy/ to $SSH_TARGET:$BOOTSTRAP_DIR/deploy/"
remote "mkdir -p '$BOOTSTRAP_DIR/deploy'"
rsync -az -e "$RSYNC_SSH" --exclude='__pycache__' "$DEPLOY_DIR/" "$SSH_TARGET:$BOOTSTRAP_DIR/deploy/"
log "OK. Next step, on the server: bash $BOOTSTRAP_DIR/deploy/setup_server.sh $GIT_TAG"
