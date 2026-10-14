#!/usr/bin/env bash
# (serveur) Mesure le débit total de la baseline avec 1, puis 2, puis 3 entraînements simultanés, pour choisir --parallel.
# Chaque entraînement fait EPOCHS epochs (epoch_size de la config = 63 pas), sans évaluation. Mesure aussi le pic de VRAM et la
# RAM minimale disponible pendant les runs. Le temps de démarrage (lecture de la validation) est inclus : il pèse pareil sur chaque cas.
# Usage : deploy/benchmark_parallel.sh [EPOCHS=6] [MAX_PARALLEL=3]
# shellcheck source=deploy/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
EPOCHS="${1:-6}"; MAX_PARALLEL="${2:-3}"
cd "$REPO_ROOT" || die "dépôt introuvable : $REPO_ROOT"
[ -d "$MUSDB_PROCESSED/train" ] || die "données prétraitées absentes ($MUSDB_PROCESSED) : lancer deploy/prepare_data.sh"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/svs_bench.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
cores="${CPU_CORES:-$(nproc)}"

run_case() {
    local n="$1" i
    : > "$WORK/mem.txt"; : > "$WORK/vram.txt"
    ( while :; do
          awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo >> "$WORK/mem.txt"
          if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 >> "$WORK/vram.txt"; fi
          sleep 3
      done ) &
    local sampler=$!
    local start end pids=()
    start=$(date +%s.%N)
    for i in $(seq 1 "$n"); do
        WANDB_MODE=offline "$PYTHON_BIN" -m src.train experiment=stereo seed=$((41 + i)) \
            training.max_epochs="$EPOCHS" training.evaluate_at_end=false \
            data.num_workers=$(( cores / n > 4 ? 4 : (cores / n < 1 ? 1 : cores / n) )) \
            data.dataset_path="$MUSDB_PROCESSED" data.split_path="$SPLIT_PATH" \
            output_dir="$WORK/out_${n}_$i" hydra.run.dir="$WORK/hydra_${n}_$i" > "$WORK/log_${n}_$i.txt" 2>&1 &
        pids+=($!)
    done
    local failed=0
    for p in "${pids[@]}"; do wait "$p" || failed=1; done
    end=$(date +%s.%N)
    kill "$sampler" 2>/dev/null || true; wait "$sampler" 2>/dev/null || true
    if [ "$failed" -ne 0 ]; then
        tail -n 15 "$WORK"/log_"${n}"_*.txt >&2
        die "au moins un entraînement a échoué avec $n processus (log ci-dessus ; OOM ? voir free -g et nvidia-smi)"
    fi
    local secs minmem maxvram
    secs=$(awk -v s="$start" -v e="$end" 'BEGIN {printf "%.0f", e - s}')
    minmem=$(sort -n "$WORK/mem.txt" | head -1)
    maxvram=$(sort -n "$WORK/vram.txt" 2>/dev/null | tail -1)
    printf '%s\t%s\t%s\t%s\n' "$n" "$secs" "${minmem:-?}" "${maxvram:--}" >> "$WORK/results.tsv"
}

: > "$WORK/results.tsv"
for n in $(seq 1 "$MAX_PARALLEL"); do
    log "cas $n processus simultané(s), $EPOCHS epochs chacun ..."
    run_case "$n"
done

echo
printf '%-9s %-10s %-22s %-18s %-14s\n' "parallèle" "durée (s)" "débit total (epoch/min)" "RAM dispo min (Mo)" "VRAM max (Mo)"
while IFS=$'\t' read -r n secs mem vram; do
    printf '%-9s %-10s %-22s %-18s %-14s\n' "$n" "$secs" "$(awk -v n="$n" -v e="$EPOCHS" -v s="$secs" 'BEGIN {printf "%.1f", n * e * 60 / s}')" "$mem" "$vram"
done < "$WORK/results.tsv"
echo
echo "Choisir le plus grand parallèle dont le débit total progresse encore nettement ET dont la RAM dispo min reste > ~3000 Mo"
echo "(l'évaluation, en plus, consomme ~2,5 Go par processus de métriques). Puis : run_queue.py --parallel N"
