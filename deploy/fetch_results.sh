#!/usr/bin/env bash
# (local) Rapatrie les résultats d'un tag depuis le serveur : JSON, logs, metadata, config résolue, dossier W&B hors ligne.
# SANS checkpoint par défaut (118 Mo chacun, très lent depuis la Chine) : `--ckpt RUN` ajoute le checkpoint ALLÉGÉ (poids seuls,
# ~39 Mo) d'un run, par exemple `--ckpt lr3e4_cosine_s42`. Vérifie que rien ne manque, puis pose RUNS_DIR/<tag>/FETCHED sur le
# serveur : `run_queue.py --shutdown` attend ce fichier. ATTENTION : les checkpoints non rapatriés restent sur le disque du serveur ;
# ils ne survivent à l'extinction que si les données de l'instance sont conservées (à vérifier dans la doc AutoDL).
# Usage : deploy/fetch_results.sh TAG [--wait] [--ckpt RUN]...
# shellcheck source=deploy/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
TAG="${1:?usage : fetch_results.sh TAG [--wait] [--ckpt RUN]...}"; shift
WAIT=0; CKPT_RUNS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --wait) WAIT=1 ;;
        --ckpt) shift; [ $# -gt 0 ] || die "--ckpt demande un nom de run (ex. lr3e4_cosine_s42)"; CKPT_RUNS+=("$1") ;;
        *) die "option inconnue : $1" ;;
    esac
    shift
done
ssh_setup
require_cmd rsync "apt install rsync"
SRC="$RUNS_DIR/$TAG"
DEST="$LOCAL_RESULTS_DIR/$TAG"

if [ "$WAIT" = "1" ]; then
    log "attente de $SRC/DONE (Ctrl-C pour arrêter ; relancer ensuite sans --wait)"
    until remote "test -f '$SRC/DONE'"; do sleep 60; done
fi
if [ "${FORCE:-0}" != "1" ]; then
    remote "test -f '$SRC/DONE'" || die "$SRC/DONE absent : la file n'est pas terminée (utiliser --wait, ou copier quand même avec FORCE=1)"
fi
mkdir -p "$DEST"

FILTERS=(--exclude='_code/')
for run in "${CKPT_RUNS[@]}"; do
    log "création du checkpoint allégé de $run sur le serveur (si absent)"
    remote "d='$SRC/$run'; slim='$SERVER_REPO/deploy/slim_ckpt.py'; [ -f \"\$slim\" ] || slim='$BOOTSTRAP_DIR/deploy/slim_ckpt.py'; \
            [ -f \"\$d/checkpoint_slim.ckpt\" ] || { ck=\$(ls -t \"\$d\"/outputs/VoiceSeparation/*/checkpoints/*.ckpt | head -1); \
            '$SERVER_REPO/.venv/bin/python' \"\$slim\" \"\$ck\" \"\$d/checkpoint_slim.ckpt\"; }" \
        || die "impossible de créer le checkpoint allégé de $run (nom de run exact ? voir ls $SRC)"
    FILTERS+=(--include="$run/checkpoint_slim.ckpt")
done
FILTERS+=(--include='*/' --include='*.json' --include='*.log' --include='*.md' --include='*.yaml' --include='*.txt'
          --include='*.wandb' --include='DONE' --exclude='*')
log "rapatriement $SSH_TARGET:$SRC -> $DEST"
rsync -avP --prune-empty-dirs --no-links -e "$RSYNC_SSH" "${FILTERS[@]}" "$SSH_TARGET:$SRC/" "$DEST/"

log "vérification : une seconde passe ne doit plus rien trouver à copier"
left="$(rsync -a --dry-run --itemize-changes --prune-empty-dirs --no-links -e "$RSYNC_SSH" "${FILTERS[@]}" "$SSH_TARGET:$SRC/" "$DEST/" | grep -c '^>f' || true)"
[ "$left" = "0" ] || die "$left fichier(s) pas (ou mal) copiés : relancer fetch_results.sh"

# Les scripts d'analyse lisent outputs/VoiceSeparation/<id>/checkpoints/*.ckpt : on y place les checkpoints allégés reçus.
for ck in "$DEST"/*/checkpoint_slim.ckpt; do
    [ -f "$ck" ] || continue
    id="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["wandb_run_id"])' "$(dirname "$ck")/metadata.json")"
    mkdir -p "$REPO_ROOT/outputs/VoiceSeparation/$id/checkpoints"
    cp "$ck" "$REPO_ROOT/outputs/VoiceSeparation/$id/checkpoints/best_slim.ckpt"
    log "checkpoint allégé de $(basename "$(dirname "$ck")") -> outputs/VoiceSeparation/$id/checkpoints/"
done

n_ckpt="$(find "$DEST" -name '*.ckpt' | wc -l)"; n_runs="$(find "$DEST" -maxdepth 2 -name metadata.json | wc -l)"
log "reçu : $n_runs run(s), $n_ckpt checkpoint(s). Résumé :"
cat "$DEST/SUMMARY.md" 2>/dev/null || log "(pas de SUMMARY.md)"
remote "touch '$SRC/FETCHED'" && log "FETCHED posé sur le serveur : l'extinction automatique (--shutdown) peut avoir lieu."
log "N'oubliez pas : deploy/sync_wandb.sh $TAG, puis vérifier dans le panneau AutoDL que l'instance est ARRÊTÉE (facturation)."
