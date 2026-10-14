#!/usr/bin/env bash
# (serveur) Installe tout ce qu'il faut. Idempotent : peut être relancé (aussi pour passer à un nouveau tag).
# Usage : deploy/setup_server.sh [TAG]        (TAG par défaut : GIT_TAG de env.sh)
# ALLOW_NO_GPU=1 : accepte un serveur démarré sans GPU (préparation à bas coût).
# Secret : si le dépôt est privé, exporter GITHUB_TOKEN dans la session (read -rs GITHUB_TOKEN; export GITHUB_TOKEN). Il n'est
# jamais écrit sur disque (ni dans .git/config) ni affiché.
# shellcheck source=deploy/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
TAG="${1:-$GIT_TAG}"
enable_net_turbo

# 1. paquets système (apt seulement si quelque chose manque : apt update peut être lent)
missing=()
for c in ffmpeg tmux rsync git unzip wget; do command -v "$c" >/dev/null 2>&1 || missing+=("$c"); done
if [ "${#missing[@]}" -gt 0 ]; then
    log "installation des paquets manquants : ${missing[*]}"
    apt-get update -y >/dev/null || die "apt-get update a échoué (réseau ? miroir apt) ; installer à la main : ${missing[*]}"
    apt-get install -y "${missing[@]}" || die "apt-get install a échoué pour : ${missing[*]}"
fi

# 2. uv
if ! command -v uv >/dev/null 2>&1; then
    log "installation de uv : $UV_INSTALL_CMD"
    bash -c "$UV_INSTALL_CMD" || die "installation de uv échouée (voir UV_INSTALL_CMD dans env.sh)"
    export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || die "uv installé mais introuvable dans le PATH (essayer : export PATH=\$HOME/.local/bin:\$PATH)"

# 3. dépôt : clone puis checkout du tag. Le jeton passe par GIT_ASKPASS (script sans secret, qui lit l'environnement).
askpass="$(mktemp)"; trap 'rm -f "$askpass"' EXIT
cat > "$askpass" <<'ASK'
#!/usr/bin/env bash
case "$1" in Username*) echo "x-access-token" ;; *) echo "${GITHUB_TOKEN:-}" ;; esac
ASK
chmod 700 "$askpass"
git_net() { GIT_ASKPASS="$askpass" GIT_TERMINAL_PROMPT=0 git -c credential.helper= "$@"; }
if [ ! -d "$SERVER_REPO/.git" ]; then
    log "clone de $REPO_URL dans $SERVER_REPO"
    mkdir -p "$(dirname "$SERVER_REPO")"
    git_net clone "$REPO_URL" "$SERVER_REPO" || die "clone impossible (dépôt privé sans GITHUB_TOKEN ? GitHub lent ? voir « git bundle » dans docs/DEPLOY_AUTODL.md)"
else
    log "dépôt déjà cloné, récupération des tags"
    git_net -C "$SERVER_REPO" fetch --tags --force origin || die "git fetch échoué (réseau / jeton)"
fi
git -C "$SERVER_REPO" rev-parse --verify --quiet "refs/tags/$TAG" >/dev/null || die "le tag '$TAG' n'existe pas dans le dépôt (l'avez-vous poussé : git push origin $TAG ?)"
git -C "$SERVER_REPO" checkout --quiet "$TAG"
log "dépôt sur le tag $TAG ($(git -C "$SERVER_REPO" rev-parse --short HEAD))"

# 4. environnement Python figé par uv.lock
if [ -z "$PYPI_MIRROR" ]; then
    log "uv sync --frozen (téléchargement de torch + CUDA : ~4 Go)"
    (cd "$SERVER_REPO" && uv sync --frozen) || die "uv sync a échoué (réseau ? index PyTorch ? espace disque : df -h /)"
else
    # mêmes versions que uv.lock (uv export les fige), téléchargées depuis PYPI_MIRROR ; torch+cu128 depuis TORCH_FIND_LINKS ou TORCH_INDEX
    log "installation via miroir $PYPI_MIRROR (versions de uv.lock)"
    req="$(mktemp)"
    (
        cd "$SERVER_REPO"
        uv export --frozen --no-hashes --no-emit-project -o "$req" >/dev/null
        uv venv --allow-existing .venv
        pip_install() { uv pip install --python .venv/bin/python -r "$req" --index-url "$PYPI_MIRROR" "$@" --index-strategy unsafe-best-match; }
        if [ -z "$TORCH_FIND_LINKS" ] || ! pip_install --find-links "$TORCH_FIND_LINKS"; then
            log "miroir torch indisponible ou non utilisé : repli sur $TORCH_INDEX (plus lent)"
            pip_install --extra-index-url "$TORCH_INDEX"
        fi
    ) || { rm -f "$req"; die "installation via miroir échouée (miroir injoignable ? espace disque : df -h /) ; relancer : les roues déjà reçues sont en cache"; }
    rm -f "$req"
fi

# 5. dossiers
mkdir -p "$MUSDB_RAW" "$ORACLE_CACHE_DIR" "$RUNS_DIR" "$DOWNLOAD_DIR"

# 6. vérifications
log "vérification de l'environnement"
ALLOW_NO_GPU="${ALLOW_NO_GPU:-0}" "$SERVER_REPO/.venv/bin/python" - <<'PY' || die "vérification échouée : voir le message ci-dessus"
import os, shutil, sys, torch
print(f"torch {torch.__version__} | CUDA (build) {torch.version.cuda}")
if not torch.cuda.is_available() and os.environ["ALLOW_NO_GPU"] == "1":
    print("AVERTISSEMENT : pas de GPU (ALLOW_NO_GPU=1, mode préparation sans GPU) ; refaire check_env.py avec le GPU")
elif not torch.cuda.is_available():
    sys.exit("ERREUR: torch.cuda.is_available() est False : pilote NVIDIA trop ancien pour cette version de CUDA, "
             "ou instance démarrée sans GPU. Vérifier nvidia-smi (version du pilote) ; torch du projet = cu128.")
if torch.cuda.is_available():
    print("GPU :", torch.cuda.get_device_name(0), f"| {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} Gio")
if not shutil.which("ffmpeg"):
    sys.exit("ERREUR: ffmpeg absent du PATH")
PY
log "setup terminé. Étape suivante : deploy/get_data.sh (ou push_data.sh depuis le PC), puis deploy/prepare_data.sh"
