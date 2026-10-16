#!/usr/bin/env bash
# Functions shared by the deploy/*.sh scripts. To be sourced (not executed). Contains NO AutoDL-specific value: see env.sh.
# shellcheck disable=SC2034  # PYTHON_BIN, RSYNC_SSH... are used by the scripts that source this file
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$DEPLOY_DIR")"
# shellcheck source=deploy/env.sh
source "$DEPLOY_DIR/env.sh"

LOCAL_MUSDB="${LOCAL_MUSDB:-$REPO_ROOT/musdb18}"
LOCAL_RESULTS_DIR="${LOCAL_RESULTS_DIR:-$REPO_ROOT/outputs/server}"
if [ -x "$REPO_ROOT/.venv/bin/python" ]; then PYTHON_BIN="$REPO_ROOT/.venv/bin/python"; else PYTHON_BIN="python3"; fi

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
require_cmd() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1 ($2)"; }
require_var() { [ -n "${!1:-}" ] || die "$1 is empty: set it in deploy/env.local.sh"; }

enable_net_turbo() {
    if [ -n "${NET_TURBO_FILE:-}" ] && [ -f "$NET_TURBO_FILE" ]; then
        set +u
        # shellcheck source=/dev/null
        source "$NET_TURBO_FILE"
        set -u
        log "network accelerator enabled ($NET_TURBO_FILE)"
    fi
}

# --- SSH (local side) ---
ssh_setup() {
    require_var SSH_HOST
    require_cmd ssh "package openssh-client"
    SSH_ARGS=(-p "$SSH_PORT" -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
    if [ -n "$SSH_KEY" ]; then SSH_ARGS+=(-i "$SSH_KEY"); fi
    SSH_TARGET="$SSH_USER@$SSH_HOST"
    RSYNC_SSH="ssh ${SSH_ARGS[*]}"
}
# shellcheck disable=SC2029  # client-side expansion is intended
remote() { ssh "${SSH_ARGS[@]}" "$SSH_TARGET" "$@"; }
