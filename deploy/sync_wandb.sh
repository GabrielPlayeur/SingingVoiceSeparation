#!/usr/bin/env bash
# (local) Envoie à W&B les runs enregistrés hors ligne (WANDB_MODE=offline) rapatriés par fetch_results.sh.
# Prérequis : `wandb login` (ou WANDB_API_KEY dans l'environnement). Usage : deploy/sync_wandb.sh TAG
# shellcheck source=deploy/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
TAG="${1:?usage : sync_wandb.sh TAG}"
DEST="$LOCAL_RESULTS_DIR/$TAG"
[ -d "$DEST" ] || die "$DEST introuvable : lancer fetch_results.sh $TAG d'abord"
mapfile -t runs < <(find "$DEST" -type d -name 'offline-run-*' | sort)
[ "${#runs[@]}" -gt 0 ] || die "aucun dossier offline-run-* sous $DEST (runs lancés avec WANDB_MODE=online ?)"
for r in "${runs[@]}"; do
    log "wandb sync $r"
    "$PYTHON_BIN" -m wandb sync "$r" || log "ECHEC pour $r (réseau / clé ? relancer le script : les runs déjà envoyés sont ignorés)"
done
