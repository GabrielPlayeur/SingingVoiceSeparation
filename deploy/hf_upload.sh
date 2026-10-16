#!/usr/bin/env bash
# (local) Hugging Face relay: uploads the .stem.mp4 files of MUSDB18 (5.7 GB) to a PRIVATE dataset repository, from which the server fetches them
# with get_data_hf.sh (the AutoDL network accelerator covers Hugging Face; Zenodo is unreachable from there, and rsync from the PC is too slow).
# MUSDB18 is restricted to academic use: keep the repository private. Resumable (`hf upload` resumes an interrupted folder when the same command is rerun).
# Prerequisites: Hugging Face account, WRITE token in HF_TOKEN (read -rs HF_TOKEN; export HF_TOKEN), HF_REPO=user/name (env.local.sh).
# Usage: deploy/hf_upload.sh
# shellcheck source=deploy/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
require_var HF_REPO
[ -n "${HF_TOKEN:-}" ] || die "HF_TOKEN missing: read -rs HF_TOKEN; export HF_TOKEN (token with write access, never in a file)"
[ -d "$LOCAL_MUSDB/train" ] || die "local dataset not found: $LOCAL_MUSDB/train"
HF=(uvx --from huggingface_hub hf)
"${HF[@]}" repos create "$HF_REPO" --type dataset --private --exist-ok >/dev/null || die "cannot create the repository $HF_REPO (token? write access?)"
log "uploading $LOCAL_MUSDB to hf.co/datasets/$HF_REPO (private)"
"${HF[@]}" upload "$HF_REPO" "$LOCAL_MUSDB" --type dataset \
    --include 'train/*.stem.mp4' --include 'test/*.stem.mp4' || die "upload interrupted: rerun the same command (resume)"
log "OK. On the server: deploy/get_data_hf.sh"
