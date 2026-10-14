#!/usr/bin/env bash
# (local) Relais Hugging Face : envoie les .stem.mp4 de MUSDB18 (5,7 Go) vers un dépôt dataset PRIVÉ, d'où le serveur les récupère
# avec get_data_hf.sh (l'accélérateur réseau AutoDL couvre Hugging Face ; Zenodo y est inaccessible, et rsync depuis le PC est trop lent).
# MUSDB18 est réservé à un usage académique : garder le dépôt privé. Reprenable (`hf upload` reprend un dossier interrompu quand on relance la même commande).
# Prérequis : compte Hugging Face, jeton ÉCRITURE dans HF_TOKEN (read -rs HF_TOKEN; export HF_TOKEN), HF_REPO=utilisateur/nom (env.local.sh).
# Usage : deploy/hf_upload.sh
# shellcheck source=deploy/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
require_var HF_REPO
[ -n "${HF_TOKEN:-}" ] || die "HF_TOKEN absent : read -rs HF_TOKEN; export HF_TOKEN (jeton avec droit d'écriture, jamais dans un fichier)"
[ -d "$LOCAL_MUSDB/train" ] || die "dataset local introuvable : $LOCAL_MUSDB/train"
HF=(uvx --from huggingface_hub hf)
"${HF[@]}" repos create "$HF_REPO" --type dataset --private --exist-ok >/dev/null || die "création du dépôt $HF_REPO impossible (jeton ? droits d'écriture ?)"
log "envoi de $LOCAL_MUSDB vers hf.co/datasets/$HF_REPO (privé)"
"${HF[@]}" upload "$HF_REPO" "$LOCAL_MUSDB" --type dataset \
    --include 'train/*.stem.mp4' --include 'test/*.stem.mp4' || die "envoi interrompu : relancer la même commande (reprise)"
log "OK. Sur le serveur : deploy/get_data_hf.sh"
