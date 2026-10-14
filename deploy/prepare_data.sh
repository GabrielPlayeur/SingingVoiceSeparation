#!/usr/bin/env bash
# (serveur) Prépare TOUT ce qui précède le premier entraînement, de façon reprenable :
#   1. vérifie les mp4 (manifeste)  2. mp4 -> wav (src/data/preprocessing.py, saute ce qui existe)
#   3. split.json (identique au split local)  4. cache de l'oracle (évite de le recalculer dans chaque évaluation, et les
#   accès concurrents entre runs parallèles).
# À lancer de préférence en mode sans GPU (CPU seul). Usage : deploy/prepare_data.sh [--workers N]
# shellcheck source=deploy/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
WORKERS="${CPU_CORES:-$(nproc)}"
if [ "${1:-}" = "--workers" ]; then WORKERS="${2:?--workers demande un nombre}"; fi
cd "$REPO_ROOT" || die "dépôt introuvable : $REPO_ROOT"

"$PYTHON_BIN" "$DEPLOY_DIR/make_manifest.py" "$MUSDB_RAW" --check "$DEPLOY_DIR/manifest_musdb18.json"
# espace : les wav pèsent ~35 Go ; on exige ce qu'il reste à écrire (+ 4 Go pour runs et cache de l'oracle)
used_gb=$(du -s --block-size=1G "$MUSDB_PROCESSED" 2>/dev/null | cut -f1 || true)
avail_gb=$(df --output=avail --block-size=1G "$MUSDB_RAW" | tail -1 | tr -d ' G')
need_gb=$(( 39 - ${used_gb:-0} ))
if [ "$avail_gb" -lt "$need_gb" ]; then
    die "espace insuffisant sur le disque de données : ${avail_gb} Go libres, ~${need_gb} Go nécessaires pour les wav (df -h ; supprimer le zip de $DOWNLOAD_DIR ou louer plus de disque)"
fi
log "disque de données : ${avail_gb} Go libres, ~${need_gb} Go à écrire"
log "prétraitement mp4 -> wav avec $WORKERS processus (décodage ffmpeg)"
"$PYTHON_BIN" src/data/preprocessing.py --input "$MUSDB_RAW" --output "$MUSDB_PROCESSED" --workers "$WORKERS"
"$PYTHON_BIN" "$DEPLOY_DIR/prepare_data.py"
log "préparation terminée. Étape suivante : deploy/check_env.py puis deploy/benchmark_parallel.sh"
