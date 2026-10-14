#!/usr/bin/env bash
# Configuration UNIQUE du déploiement. Sourcé par tous les deploy/*.sh, lu aussi par les scripts Python (deploy/_env.py).
#
# - Toute valeur qui dépend d'AutoDL est marquée « TODO: vérifier dans la doc AutoDL » : elle n'est écrite qu'ici.
# - Les valeurs propres à CETTE location (SSH_HOST, SSH_PORT...) vont dans deploy/env.local.sh (ignoré par git, pour que
#   l'arbre reste propre : la file d'expériences refuse de démarrer sinon).
# - Les secrets (GITHUB_TOKEN, WANDB_API_KEY) ne vont JAMAIS dans un fichier : variables d'environnement uniquement.
# - Toute variable peut être surchargée par l'environnement (ex. DATA_ROOT=/tmp/x deploy/...).

_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
if [ -f "$_ENV_DIR/env.local.sh" ]; then source "$_ENV_DIR/env.local.sh"; fi

# ---- Connexion SSH (côté local) : à remplir dans env.local.sh avec les valeurs du panneau AutoDL ----------------------
: "${SSH_HOST:=}"                 # ex. connect.xxx.seetacloud.com
: "${SSH_PORT:=22}"
: "${SSH_USER:=root}"
: "${SSH_KEY:=}"                  # chemin de la clé privée ; vide = clé par défaut / mot de passe

# ---- Dépôt ------------------------------------------------------------------------------------------------------------
: "${REPO_URL:=https://github.com/GabrielPlayeur/SingingVoiceSeparation.git}"   # ou un fichier .bundle (git bundle) en secours
: "${GIT_TAG:=lr-v1}"             # tag à déployer par défaut (setup_server.sh et run_queue.py --tag)

# ---- Disques et dossiers (côté serveur) --------------------------------------------------------------------------------
: "${DATA_ROOT:=/root/autodl-tmp}"   # TODO: vérifier dans la doc AutoDL (disque de données, et sa taille : il faut >= 60 Go)
: "${SERVER_REPO:=$HOME/SingingVoiceSeparation}"           # clone + .venv (~7 Go) : disque SYSTÈME (30 Go), pour garder les 50 Go de données pour MUSDB18 (~41 Go)
: "${BOOTSTRAP_DIR:=/root/svs_bootstrap}"                  # où push_bootstrap.sh copie deploy/ avant le premier clone
: "${MUSDB_RAW:=$DATA_ROOT/musdb18}"                       # {train,test}/*.stem.mp4
: "${MUSDB_PROCESSED:=$MUSDB_RAW/processed}"               # wav générés par src/data/preprocessing.py (~35 Go)
: "${SPLIT_PATH:=$MUSDB_RAW/split.json}"
: "${ORACLE_CACHE_DIR:=$DATA_ROOT/oracle_cache}"
: "${RUNS_DIR:=$DATA_ROOT/runs}"                           # runs/<tag>/<nom>_s<seed>/
: "${DOWNLOAD_DIR:=$DATA_ROOT/downloads}"

# ---- Données (côté local) ----------------------------------------------------------------------------------------------
: "${LOCAL_MUSDB:=}"              # vide = <dépôt>/musdb18
# KEEP_ZIP=1 pour garder le zip (4,7 Go) après extraction ; supprimé par défaut (disque de données de 50 Go)
: "${DATASET_URL:=https://zenodo.org/records/1117372/files/musdb18.zip}"   # TODO: vérifier l'URL (MUSDB18 compressé, ~4,7 Go)
: "${HF_REPO:=}"                    # relais de données : utilisateur/nom d'un dépôt Hugging Face dataset PRIVÉ (hf_upload.sh / get_data_hf.sh)
: "${LOCAL_RESULTS_DIR:=}"        # vide = <dépôt>/outputs/server

# ---- Spécifique AutoDL : à vérifier dans la doc AutoDL avant d'acheter ---------------------------------------------------
: "${NET_TURBO_FILE:=/etc/network_turbo}"   # TODO: vérifier dans la doc AutoDL (accélérateur réseau GitHub : fichier à « source »). Vide = aucun
: "${SHUTDOWN_CMD:=shutdown}"               # TODO: vérifier dans la doc AutoDL (extinction depuis le terminal) ; vérifier que la facturation s'arrête
: "${CPU_CORES:=}"                          # TODO: vérifier dans la doc AutoDL (cœurs réellement alloués ; os.cpu_count() peut voir ceux de l'hôte). Vide = nproc
# TODO: vérifier dans la doc AutoDL : sources lentes depuis la Chine ? Si oui, décommenter/adapter (variables lues par uv) :
# export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"      # miroir PyPI (pyproject.toml pointe aussi vers download.pytorch.org)
# export UV_PYTHON_INSTALL_MIRROR="..."                                   # miroir de Python (sinon uv télécharge depuis GitHub)
# export UV_PYTHON="/usr/bin/python3.12"                                  # ou utiliser le Python 3.12 de l'image
# Installation via miroir (Chine) : si PYPI_MIRROR est non vide, setup_server.sh n'utilise pas `uv sync --frozen` (qui télécharge les
# URL figées dans uv.lock, depuis pythonhosted.org / pytorch.org : ~0,7 à 1,3 Mo/s mesurés depuis AutoDL) mais `uv export` +
# `uv pip install` : MÊMES versions exactes que le lock, depuis le miroir (~7,6 Mo/s mesurés). Valeurs mesurées/vérifiées le 2026-10-14.
# Hors de Chine : PYPI_MIRROR= (vide) pour revenir à `uv sync --frozen`.
: "${PYPI_MIRROR=https://mirrors.aliyun.com/pypi/simple/}"                 # TODO: vérifier dans la doc AutoDL (c'est aussi le miroir du pip.conf de l'image)
: "${TORCH_WHEELS_URL=https://mirrors.aliyun.com/pytorch-wheels/cu128/}"   # dossier des roues torch/torchvision +cu128 (fichiers joignables, pas leur listing) ; vide ou en échec : repli sur TORCH_INDEX
: "${TORCH_INDEX:=https://download.pytorch.org/whl/cu128}"
: "${UV_INSTALL_CMD:=python3 -m pip install uv}"   # TODO: vérifier dans la doc AutoDL (pip ou accès à astral.sh ?)

# ---- Exécution ---------------------------------------------------------------------------------------------------------
: "${DEFAULT_PARALLEL:=3}"        # nombre de runs (entraînement + évaluation) simultanés ; à fixer avec benchmark_parallel.sh
: "${WANDB_MODE:=offline}"        # offline | online (online demande WANDB_API_KEY dans l'environnement)
: "${SHUTDOWN_WAIT_MIN:=180}"     # run_queue.py --shutdown : attente max du fichier FETCHED avant d'abandonner l'extinction

export HF_REPO SSH_HOST SSH_PORT SSH_USER SSH_KEY REPO_URL GIT_TAG DATA_ROOT SERVER_REPO BOOTSTRAP_DIR MUSDB_RAW MUSDB_PROCESSED SPLIT_PATH \
       ORACLE_CACHE_DIR RUNS_DIR DOWNLOAD_DIR LOCAL_MUSDB DATASET_URL LOCAL_RESULTS_DIR NET_TURBO_FILE SHUTDOWN_CMD CPU_CORES \
       UV_INSTALL_CMD PYPI_MIRROR TORCH_INDEX TORCH_WHEELS_URL DEFAULT_PARALLEL WANDB_MODE SHUTDOWN_WAIT_MIN
