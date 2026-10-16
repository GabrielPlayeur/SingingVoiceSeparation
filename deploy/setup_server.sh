#!/usr/bin/env bash
# (server) Installs everything needed. Idempotent: can be rerun (also to switch to a new tag).
# Usage: deploy/setup_server.sh [TAG]        (default TAG: GIT_TAG of env.sh)
# ALLOW_NO_GPU=1: accepts a server started without a GPU (low-cost preparation).
# Secret: if the repository is private, export GITHUB_TOKEN in the session (read -rs GITHUB_TOKEN; export GITHUB_TOKEN). It is
# never written to disk (nor to .git/config) nor displayed.
# shellcheck source=deploy/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
TAG="${1:-$GIT_TAG}"
# (the network accelerator, a proxy, is only for GitHub: it is enabled in git_net only; it breaks the PyPI/torch mirrors)

# 1. system packages (apt only if something is missing: apt update can be slow)
missing=()
for c in ffmpeg tmux rsync git unzip wget; do command -v "$c" >/dev/null 2>&1 || missing+=("$c"); done
if [ "${#missing[@]}" -gt 0 ]; then
    log "installing the missing packages: ${missing[*]}"
    apt-get update -y >/dev/null || die "apt-get update failed (network? apt mirror); install by hand: ${missing[*]}"
    apt-get install -y "${missing[@]}" || die "apt-get install failed for: ${missing[*]}"
fi

# 2. uv
if ! command -v uv >/dev/null 2>&1; then
    log "installing uv: $UV_INSTALL_CMD"
    bash -c "$UV_INSTALL_CMD" || die "uv installation failed (see UV_INSTALL_CMD in env.sh)"
    export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || die "uv installed but not found in the PATH (try: export PATH=\$HOME/.local/bin:\$PATH)"

# 3. repository: clone then checkout of the tag. The token goes through GIT_ASKPASS (a script without secret, which reads the environment).
askpass="$(mktemp)"; trap 'rm -f "$askpass"' EXIT
cat > "$askpass" <<'ASK'
#!/usr/bin/env bash
case "$1" in Username*) echo "x-access-token" ;; *) echo "${GITHUB_TOKEN:-}" ;; esac
ASK
chmod 700 "$askpass"
git_net() { ( enable_net_turbo; GIT_ASKPASS="$askpass" GIT_TERMINAL_PROMPT=0 git -c credential.helper= "$@" ); }
if [ ! -d "$SERVER_REPO/.git" ]; then
    log "cloning $REPO_URL into $SERVER_REPO"
    mkdir -p "$(dirname "$SERVER_REPO")"
    git_net clone "$REPO_URL" "$SERVER_REPO" || die "cannot clone (private repository without GITHUB_TOKEN? slow GitHub? see \"git bundle\" in docs/DEPLOY_AUTODL.md)"
else
    log "repository already cloned, fetching the tags"
    git_net -C "$SERVER_REPO" fetch --tags --force origin || die "git fetch failed (network / token)"
fi
git -C "$SERVER_REPO" rev-parse --verify --quiet "refs/tags/$TAG" >/dev/null || die "tag '$TAG' does not exist in the repository (did you push it: git push origin $TAG ?)"
git -C "$SERVER_REPO" checkout --quiet "$TAG"
log "repository on tag $TAG ($(git -C "$SERVER_REPO" rev-parse --short HEAD))"

# 4. Python environment pinned by uv.lock
if [ -z "$PYPI_MIRROR" ]; then
    log "uv sync --frozen (downloading torch + CUDA: ~4 GB)"
    (cd "$SERVER_REPO" && uv sync --frozen) || die "uv sync failed (network? PyTorch index? disk space: df -h /)"
else
    # same versions as uv.lock (uv export pins them), downloaded from PYPI_MIRROR; torch+cu128 from TORCH_WHEELS_URL (direct URLs) or TORCH_INDEX
    log "installing through mirror $PYPI_MIRROR (uv.lock versions)"
    # a proxy possibly inherited from the session (network accelerator) breaks the mirrors
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
    req="$(mktemp)"
    (
        cd "$SERVER_REPO"
        uv export --frozen --no-hashes --no-emit-project -o "$req" >/dev/null
        uv venv --allow-existing .venv
        tv_pins=()   # torch and torchvision (+cu128 versions): the only ones to come from the PyTorch index; their dependencies come from the PyPI mirror
        for pkg in torch torchvision; do
            pin="$(grep -E "^${pkg}==" "$req" | head -1 | sed -E "s/^${pkg}==//; s/[ ;].*//")"
            [ -n "$pin" ] || { echo "ERROR: $pkg missing from uv.lock" >&2; exit 1; }
            tv_pins+=("$pkg==$pin")
        done
        grep -vE '^(torch|torchvision)==' "$req" > "$req.rest"
        tag="$(.venv/bin/python -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
        urls=()
        for p in "${tv_pins[@]}"; do urls+=("${TORCH_WHEELS_URL%/}/${p%%==*}-${p##*==}-$tag-$tag-manylinux_2_28_x86_64.whl"); done
        urls=("${urls[@]//+/%2B}")
        if [ -n "$TORCH_WHEELS_URL" ] && uv pip install --python .venv/bin/python --no-deps "${urls[@]}"; then :; else
            log "torch wheels from the mirror unavailable: falling back to $TORCH_INDEX (slower)"
            uv pip install --python .venv/bin/python --no-deps --index-url "$TORCH_INDEX" "${tv_pins[@]}"
        fi
        uv pip install --python .venv/bin/python -r "$req.rest" --index-url "$PYPI_MIRROR"
        rm -f "$req.rest"
    ) || { rm -f "$req"; die "installation through the mirror failed (mirror unreachable? disk space: df -h /); rerun: the wheels already received are cached"; }
    rm -f "$req"
fi

# 5. folders
mkdir -p "$MUSDB_RAW" "$ORACLE_CACHE_DIR" "$RUNS_DIR" "$DOWNLOAD_DIR"

# 6. checks
log "checking the environment"
ALLOW_NO_GPU="${ALLOW_NO_GPU:-0}" "$SERVER_REPO/.venv/bin/python" - <<'PY' || die "check failed: see the message above"
import os, shutil, sys, torch
print(f"torch {torch.__version__} | CUDA (build) {torch.version.cuda}")
if not torch.cuda.is_available() and os.environ["ALLOW_NO_GPU"] == "1":
    print("WARNING: no GPU (ALLOW_NO_GPU=1, GPU-less preparation mode); rerun check_env.py with the GPU")
elif not torch.cuda.is_available():
    sys.exit("ERROR: torch.cuda.is_available() is False: NVIDIA driver too old for this CUDA version, "
             "or instance started without GPU. Check nvidia-smi (driver version); the project's torch = cu128.")
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), f"| {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
if not shutil.which("ffmpeg"):
    sys.exit("ERROR: ffmpeg not in PATH")
PY
log "setup finished. Next step: deploy/get_data.sh (or push_data.sh from the PC), then deploy/prepare_data.sh"
