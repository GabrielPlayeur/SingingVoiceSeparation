#!/usr/bin/env bash
# (local) Rapatrie les résultats d'un tag depuis le serveur : JSON, figures, logs, metadata, config résolue, dossier W&B offline et
# UN checkpoint par run (save_top_k=1). Vérifie que RIEN ne manque, puis pose RUNS_DIR/<tag>/FETCHED sur le serveur : c'est ce
# fichier que `run_queue.py --shutdown` attend avant d'éteindre.
# Usage : deploy/fetch_results.sh TAG [--wait]        (--wait : attend le fichier DONE de la file avant de copier)
# shellcheck source=deploy/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
TAG="${1:?usage : fetch_results.sh TAG [--wait]}"; WAIT="${2:-}"
ssh_setup
require_cmd rsync "apt install rsync"
SRC="$RUNS_DIR/$TAG"
DEST="$LOCAL_RESULTS_DIR/$TAG"

if [ "$WAIT" = "--wait" ]; then
    log "attente de $SRC/DONE (Ctrl-C pour arrêter ; relancer ensuite sans --wait)"
    until remote "test -f '$SRC/DONE'"; do sleep 60; done
fi
if [ "${FORCE:-0}" != "1" ]; then
    remote "test -f '$SRC/DONE'" || die "$SRC/DONE absent : la file n'est pas terminée (utiliser --wait, ou copier quand même avec FORCE=1)"
fi
mkdir -p "$DEST"

FILTERS=(--exclude='_code/' --include='*/' --include='*.json' --include='*.log' --include='*.md' --include='*.yaml' --include='*.txt'
         --include='*.ckpt' --include='*.wandb' --include='DONE' --exclude='*')
log "rapatriement $SSH_TARGET:$SRC -> $DEST"
rsync -avP --prune-empty-dirs --no-links -e "$RSYNC_SSH" "${FILTERS[@]}" "$SSH_TARGET:$SRC/" "$DEST/"

log "vérification : une seconde passe ne doit plus rien trouver à copier"
left="$(rsync -a --dry-run --itemize-changes --prune-empty-dirs --no-links -e "$RSYNC_SSH" "${FILTERS[@]}" "$SSH_TARGET:$SRC/" "$DEST/" | grep -c '^>f' || true)"
[ "$left" = "0" ] || die "$left fichier(s) pas (ou mal) copiés : relancer fetch_results.sh"

# Les scripts d'analyse lisent outputs/VoiceSeparation/<id>/checkpoints/ : on y recopie les dossiers de chaque run.
mkdir -p "$REPO_ROOT/outputs/VoiceSeparation"
for d in "$DEST"/*/outputs/VoiceSeparation; do
    [ -d "$d" ] && rsync -a --no-links "$d/" "$REPO_ROOT/outputs/VoiceSeparation/"
done

n_ckpt="$(find "$DEST" -name '*.ckpt' | wc -l)"; n_runs="$(find "$DEST" -maxdepth 2 -name metadata.json | wc -l)"
log "reçu : $n_runs run(s), $n_ckpt checkpoint(s). Résumé :"
cat "$DEST/SUMMARY.md" 2>/dev/null || log "(pas de SUMMARY.md)"
remote "touch '$SRC/FETCHED'" && log "FETCHED posé sur le serveur : l'extinction automatique (--shutdown) peut avoir lieu."
log "N'oubliez pas : deploy/sync_wandb.sh $TAG, puis vérifier dans le panneau AutoDL que l'instance est ARRÊTÉE (facturation)."
