#!/usr/bin/env bash
# Fonctions communes des scripts deploy/*.sh. À sourcer (pas à exécuter). Ne contient AUCUNE valeur propre à AutoDL : voir env.sh.
# shellcheck disable=SC2034  # PYTHON_BIN, RSYNC_SSH... sont utilisés par les scripts qui sourcent ce fichier
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$DEPLOY_DIR")"
# shellcheck source=deploy/env.sh
source "$DEPLOY_DIR/env.sh"

LOCAL_MUSDB="${LOCAL_MUSDB:-$REPO_ROOT/musdb18}"
LOCAL_RESULTS_DIR="${LOCAL_RESULTS_DIR:-$REPO_ROOT/outputs/server}"
if [ -x "$REPO_ROOT/.venv/bin/python" ]; then PYTHON_BIN="$REPO_ROOT/.venv/bin/python"; else PYTHON_BIN="python3"; fi

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { printf 'ERREUR: %s\n' "$*" >&2; exit 1; }
require_cmd() { command -v "$1" >/dev/null 2>&1 || die "commande absente : $1 ($2)"; }
require_var() { [ -n "${!1:-}" ] || die "$1 est vide : renseignez-le dans deploy/env.local.sh"; }

enable_net_turbo() {
    if [ -n "${NET_TURBO_FILE:-}" ] && [ -f "$NET_TURBO_FILE" ]; then
        set +u
        # shellcheck source=/dev/null
        source "$NET_TURBO_FILE"
        set -u
        log "accélérateur réseau activé ($NET_TURBO_FILE)"
    fi
}

# --- SSH (côté local) ---
ssh_setup() {
    require_var SSH_HOST
    require_cmd ssh "paquet openssh-client"
    SSH_ARGS=(-p "$SSH_PORT" -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
    if [ -n "$SSH_KEY" ]; then SSH_ARGS+=(-i "$SSH_KEY"); fi
    SSH_TARGET="$SSH_USER@$SSH_HOST"
    RSYNC_SSH="ssh ${SSH_ARGS[*]}"
}
# shellcheck disable=SC2029  # l’expansion côté client est voulue
remote() { ssh "${SSH_ARGS[@]}" "$SSH_TARGET" "$@"; }
