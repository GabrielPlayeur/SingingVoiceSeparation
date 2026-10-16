#!/usr/bin/env bash
# (local) Sends to W&B the runs recorded offline (WANDB_MODE=offline) and fetched by fetch_results.sh.
# Prerequisites: `wandb login` (or WANDB_API_KEY in the environment). Usage: deploy/sync_wandb.sh TAG
# shellcheck source=deploy/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
TAG="${1:?usage: sync_wandb.sh TAG}"
DEST="$LOCAL_RESULTS_DIR/$TAG"
[ -d "$DEST" ] || die "$DEST not found: run fetch_results.sh $TAG first"
mapfile -t runs < <(find "$DEST" -type d -name 'offline-run-*' | sort)
[ "${#runs[@]}" -gt 0 ] || die "no offline-run-* folder under $DEST (runs launched with WANDB_MODE=online?)"
for r in "${runs[@]}"; do
    log "wandb sync $r"
    "$PYTHON_BIN" -m wandb sync "$r" || log "FAILED for $r (network / key? rerun the script: runs already sent are ignored)"
done
