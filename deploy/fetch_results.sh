#!/usr/bin/env bash
# (local) Fetches the results of a tag from the server: JSON, logs, metadata, resolved config, offline W&B folder.
# WITHOUT checkpoints by default (118 MB each, very slow from China): `--ckpt RUN` adds the SLIMMED checkpoint (weights only,
# ~39 MB) of a run, e.g. `--ckpt lr3e4_cosine_s42`. Checks that nothing is missing, then sets RUNS_DIR/<tag>/FETCHED on the
# server: `run_queue.py --shutdown` waits for this file. WARNING: the checkpoints that are not fetched stay on the server's disk;
# they only survive the shutdown if the instance's data is kept (to check in the AutoDL docs).
# Usage: deploy/fetch_results.sh TAG [--wait] [--ckpt RUN]...
# shellcheck source=deploy/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
TAG="${1:?usage: fetch_results.sh TAG [--wait] [--ckpt RUN]...}"; shift
WAIT=0; CKPT_RUNS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --wait) WAIT=1 ;;
        --ckpt) shift; [ $# -gt 0 ] || die "--ckpt needs a run name (e.g. lr3e4_cosine_s42)"; CKPT_RUNS+=("$1") ;;
        *) die "unknown option: $1" ;;
    esac
    shift
done
ssh_setup
require_cmd rsync "apt install rsync"
SRC="$RUNS_DIR/$TAG"
DEST="$LOCAL_RESULTS_DIR/$TAG"

if [ "$WAIT" = "1" ]; then
    log "waiting for $SRC/DONE (Ctrl-C to stop; then rerun without --wait)"
    until remote "test -f '$SRC/DONE'"; do sleep 60; done
fi
if [ "${FORCE:-0}" != "1" ]; then
    remote "test -f '$SRC/DONE'" || die "$SRC/DONE missing: the queue is not finished (use --wait, or copy anyway with FORCE=1)"
fi
mkdir -p "$DEST"

FILTERS=(--exclude='_code/')
for run in "${CKPT_RUNS[@]}"; do
    log "creating the slimmed checkpoint of $run on the server (if absent)"
    remote "d='$SRC/$run'; slim='$SERVER_REPO/deploy/slim_ckpt.py'; [ -f \"\$slim\" ] || slim='$BOOTSTRAP_DIR/deploy/slim_ckpt.py'; \
            [ -f \"\$d/checkpoint_slim.ckpt\" ] || '$SERVER_REPO/.venv/bin/python' \"\$slim\" \"\$d\" \"\$d/checkpoint_slim.ckpt\"" \
        || die "cannot create the slimmed checkpoint of $run (exact run name? see ls $SRC)"
    FILTERS+=(--include="$run/checkpoint_slim.ckpt")
done
FILTERS+=(--include='*/' --include='*.json' --include='*.log' --include='*.md' --include='*.yaml' --include='*.txt'
          --include='*.wandb' --include='DONE' --exclude='*')
log "fetching $SSH_TARGET:$SRC -> $DEST"
rsync -avP --prune-empty-dirs --no-links -e "$RSYNC_SSH" "${FILTERS[@]}" "$SSH_TARGET:$SRC/" "$DEST/"

log "check: a second pass must find nothing left to copy"
left="$(rsync -a --dry-run --itemize-changes --prune-empty-dirs --no-links -e "$RSYNC_SSH" "${FILTERS[@]}" "$SSH_TARGET:$SRC/" "$DEST/" | grep -c '^>f' || true)"
[ "$left" = "0" ] || die "$left file(s) not (or badly) copied: rerun fetch_results.sh"

# The analysis scripts read outputs/VoiceSeparation/<id>/checkpoints/*.ckpt: the slimmed checkpoints received are placed there.
for ck in "$DEST"/*/checkpoint_slim.ckpt; do
    [ -f "$ck" ] || continue
    id="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["wandb_run_id"])' "$(dirname "$ck")/metadata.json")"
    mkdir -p "$REPO_ROOT/outputs/VoiceSeparation/$id/checkpoints"
    cp "$ck" "$REPO_ROOT/outputs/VoiceSeparation/$id/checkpoints/best_slim.ckpt"
    log "slimmed checkpoint of $(basename "$(dirname "$ck")") -> outputs/VoiceSeparation/$id/checkpoints/"
done

n_ckpt="$(find "$DEST" -name '*.ckpt' | wc -l)"; n_runs="$(find "$DEST" -maxdepth 2 -name metadata.json | wc -l)"
log "received: $n_runs run(s), $n_ckpt checkpoint(s). Summary:"
cat "$DEST/SUMMARY.md" 2>/dev/null || log "(no SUMMARY.md)"
remote "touch '$SRC/FETCHED'" && log "FETCHED set on the server: the automatic shutdown (--shutdown) may now happen."
log "Do not forget: deploy/sync_wandb.sh $TAG, then check in the AutoDL panel that the instance is STOPPED (billing)."
