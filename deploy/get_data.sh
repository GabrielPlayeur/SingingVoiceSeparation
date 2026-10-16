#!/usr/bin/env bash
# (server) Downloads the compressed MUSDB18 (zip), extracts it into MUSDB_RAW and checks the manifest. Resumable (wget -c).
# Usage: deploy/get_data.sh [URL]            (default URL: DATASET_URL of env.sh)
# shellcheck source=deploy/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
URL="${1:-$DATASET_URL}"
MANIFEST="$DEPLOY_DIR/manifest_musdb18.json"
require_cmd wget "apt install wget"; require_cmd unzip "apt install unzip"
# the network accelerator (proxy) is reserved for GitHub/Hugging Face: on Zenodo it slows things down (measured: 23 KB/s). USE_NET_TURBO=1 to enable it anyway.
if [ "${USE_NET_TURBO:-0}" = "1" ]; then enable_net_turbo; fi
mkdir -p "$DOWNLOAD_DIR" "$MUSDB_RAW"
zip="$DOWNLOAD_DIR/musdb18.zip"
log "downloading $URL (automatic resume if interrupted: rerun the command)"
wget -c --tries=20 --retry-connrefused -O "$zip" "$URL" || die "download failed; alternative: deploy/push_data.sh from the PC"
log "extracting into $MUSDB_RAW"
unzip -q -n "$zip" -d "$MUSDB_RAW" || die "unzip failed (incomplete archive? rerun this script: wget -c completes it)"
"$PYTHON_BIN" "$DEPLOY_DIR/make_manifest.py" "$MUSDB_RAW" --check "$MANIFEST"
if [ "${KEEP_ZIP:-0}" != "1" ]; then rm -f "$zip"; log "zip deleted (KEEP_ZIP=1 to keep it): space freed"; fi
log "data OK. Next step: deploy/prepare_data.sh"
