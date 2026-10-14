#!/usr/bin/env bash
# (serveur) Télécharge MUSDB18 compressé (zip), l'extrait dans MUSDB_RAW et vérifie le manifeste. Reprenable (wget -c).
# Usage : deploy/get_data.sh [URL]            (URL par défaut : DATASET_URL de env.sh)
# shellcheck source=deploy/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
URL="${1:-$DATASET_URL}"
MANIFEST="$DEPLOY_DIR/manifest_musdb18.json"
require_cmd wget "apt install wget"; require_cmd unzip "apt install unzip"
enable_net_turbo
mkdir -p "$DOWNLOAD_DIR" "$MUSDB_RAW"
zip="$DOWNLOAD_DIR/musdb18.zip"
log "téléchargement de $URL (reprise automatique si interrompu : relancer la commande)"
wget -c --tries=20 --retry-connrefused -O "$zip" "$URL" || die "téléchargement échoué ; alternative : deploy/push_data.sh depuis le PC"
log "extraction dans $MUSDB_RAW"
unzip -q -n "$zip" -d "$MUSDB_RAW" || die "unzip a échoué (archive incomplète ? relancer ce script : wget -c la complète)"
"$PYTHON_BIN" "$DEPLOY_DIR/make_manifest.py" "$MUSDB_RAW" --check "$MANIFEST"
if [ "${KEEP_ZIP:-0}" != "1" ]; then rm -f "$zip"; log "zip supprimé (KEEP_ZIP=1 pour le garder) : place libérée"; fi
log "données OK. Étape suivante : deploy/prepare_data.sh"
