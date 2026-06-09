#!/usr/bin/env bash
#
# Strawberry setup script.
#
# Idempotent — safe to re-run. Each step skips work that's already done, and
# large downloads resume rather than restart. It will:
#   1. Ensure `uv` is installed
#   2. Install Python dependencies (uv sync)
#   3. Download the smart-turn VAD model
#   4. Download the Gemma LLM weights (~3.7 GB)
#   5. Scaffold config.local.json (pointed at the downloaded weights)
#   6. Report on external requirements (VLC, OrbStack)
#
# Usage: ./setup.sh

set -euo pipefail

cd "$(dirname "$0")"

# --- pretty output helpers ---------------------------------------------------
bold() { printf '\033[1m%s\033[0m\n' "$1"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$1"; }
info() { printf '  \033[36m→\033[0m %s\n' "$1"; }

VAD_MODEL="models/smart-turn-v3.2-cpu.onnx"
VAD_URL="https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx"

GEMMA_MODEL="models/gemma-4-E4B-it.litertlm"
GEMMA_URL="https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/resolve/main/gemma-4-E4B-it.litertlm"

# --- download helper ---------------------------------------------------------
# Idempotent, resumable download.
#   - Skips if the destination already exists and matches the remote size.
#   - Resumes a partial download into <dest>.part, then atomically moves it
#     into place only once the transfer completes successfully.
download() {
    local url="$1" dest="$2"
    local remote_size local_size

    # HuggingFace serves a redirect to a CDN; x-linked-size carries the true
    # size of the underlying (xet-deduplicated) file. Fall back to 0 if absent.
    remote_size=$(curl -sIL "$url" 2>/dev/null \
        | tr -d '\r' \
        | awk -F': ' 'tolower($1)=="x-linked-size" || tolower($1)=="content-length" {v=$2} END{print v+0}')

    if [[ -f "$dest" ]]; then
        local_size=$(wc -c < "$dest" | tr -d ' ')
        if [[ "$remote_size" -gt 0 && "$local_size" == "$remote_size" ]]; then
            ok "$(basename "$dest") already present ($local_size bytes)"
            return 0
        elif [[ "$remote_size" -eq 0 ]]; then
            # Couldn't determine remote size; trust the existing file.
            ok "$(basename "$dest") already present (size unverified)"
            return 0
        else
            warn "$(basename "$dest") size mismatch ($local_size vs $remote_size) — re-downloading"
            rm -f "$dest"
        fi
    fi

    info "Downloading $(basename "$dest")…"
    # -C - resumes from <dest>.part if a previous run was interrupted.
    curl -L --fail --retry 3 -C - -o "$dest.part" "$url"
    mv "$dest.part" "$dest"
    ok "Downloaded $dest"
}

# --- 1. uv -------------------------------------------------------------------
bold "1/6  Checking for uv"
if command -v uv >/dev/null 2>&1; then
    ok "uv already installed ($(uv --version))"
else
    info "Installing uv…"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Make uv available in this shell for the rest of the script.
    export PATH="$HOME/.local/bin:$PATH"
    ok "uv installed ($(uv --version))"
fi

# --- 2. Python dependencies --------------------------------------------------
bold "2/6  Installing Python dependencies"
# `uv sync` reads pyproject.toml + uv.lock, provisions the right Python
# (3.12, per .python-version) and creates .venv with locked versions.
# It's a no-op when everything is already in sync.
uv sync
ok "Dependencies installed into .venv"

# --- 3. smart-turn VAD model -------------------------------------------------
bold "3/6  Downloading smart-turn VAD model"
mkdir -p models
download "$VAD_URL" "$VAD_MODEL"

# --- 4. Gemma LLM weights ----------------------------------------------------
bold "4/6  Downloading Gemma LLM weights (~3.7 GB, this can take a while)"
download "$GEMMA_URL" "$GEMMA_MODEL"

# --- 5. config.local.json ----------------------------------------------------
bold "5/6  Scaffolding config.local.json"
GEMMA_ABS="$(pwd)/$GEMMA_MODEL"
if [[ -f config.local.json ]]; then
    # Self-heal a leftover placeholder MODEL_PATH (e.g. "/path/to/..."), but
    # never clobber a real path the user has set.
    if grep -q '"MODEL_PATH"[[:space:]]*:[[:space:]]*"/path/to' config.local.json; then
        # Rewrite just the MODEL_PATH value, in place.
        tmp=$(mktemp)
        sed "s|\"MODEL_PATH\"[[:space:]]*:[[:space:]]*\"/path/to[^\"]*\"|\"MODEL_PATH\": \"$GEMMA_ABS\"|" \
            config.local.json > "$tmp" && mv "$tmp" config.local.json
        ok "Repaired placeholder MODEL_PATH in config.local.json → downloaded weights"
    else
        ok "config.local.json already exists (left untouched)"
    fi
else
    cat > config.local.json <<JSON
{
  "MODEL_PATH": "$GEMMA_ABS"
}
JSON
    ok "Created config.local.json (MODEL_PATH set to downloaded weights)"
    info "Add an optional \"MUSIC_DIR\" pointing at your music folder to enable playback."
fi

# --- 6. External requirements (can't be auto-installed reliably) -------------
bold "6/6  Checking external requirements"

# VLC — required for music playback.
if command -v vlc >/dev/null 2>&1 || [[ -e "/Applications/VLC.app" ]]; then
    ok "VLC found"
else
    warn "VLC not found — needed for music playback. Install from https://www.videolan.org/vlc/"
fi

# OrbStack / Docker — required for the code execution sandbox.
if command -v docker >/dev/null 2>&1 || [[ -e "/Applications/OrbStack.app" ]]; then
    ok "Docker/OrbStack found"
else
    warn "OrbStack not found — needed for the code execution sandbox. Install from https://orbstack.dev/"
fi

echo
bold "Setup complete."
echo "  Run Strawberry with:  uv run main.py"
echo "  Then click 'Connect Microphone' in the browser tab that opens, and say 'hey strawberry'."
