#!/usr/bin/env bash
#
# llm-router one-line installer.
#
#   curl -fsSL https://raw.githubusercontent.com/Ninzalo/llm-router/main/install.sh | bash
#
# What it does (idempotent — safe to re-run):
#   1. clones (or updates) the repo at ~/llm-router          [override: LLM_ROUTER_DIR]
#   2. creates / reuses a Python virtualenv at ~/llm-router/.venv
#   3. installs the `llm-router` command (editable) into that venv
#   4. symlinks ~/llm-router/.venv/bin/llm-router -> ~/.local/bin/llm-router
#   5. drops a starter config.yaml (from config.example.yaml) if none exists
#
# Requirements: bash, git, and Python 3.10+. If `uv` is on your PATH it is used
# (faster), otherwise the standard `python -m venv` is used.
set -euo pipefail

REPO_SSH_URL="git@github.com:Ninzalo/llm-router.git"
REPO_HTTPS_URL="https://github.com/Ninzalo/llm-router.git"
INSTALL_DIR="${LLM_ROUTER_DIR:-$HOME/llm-router}"
BIN_DIR="${LLM_ROUTER_BIN_DIR:-$HOME/.local/bin}"

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# --- 0. preflight ------------------------------------------------------------
command -v git  >/dev/null 2>&1 || die "git is required (install it, then re-run)."
command -v curl >/dev/null 2>&1 || die "curl is required."

# find a Python 3.10+ interpreter
PY=""
for cand in python3.13 python3.12 python3.11 python3.10 python3; do
  command -v "$cand" >/dev/null 2>&1 || continue
  ver="$("$cand" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo 0.0)"
  major="${ver%%.*}"; minor="${ver#*.}"
  if { [ "$major" -gt 3 ] || { [ "$major" -eq 3 ] && [ "$minor" -ge 10 ]; }; }; then
    PY="$(command -v "$cand")"; break
  fi
done
[ -n "$PY" ] || die "Python 3.10+ not found (looked for python3.13..python3.10, python3)."
log "Using Python: $PY ($("$PY" --version 2>&1))"

# --- 1. clone / update the repo ----------------------------------------------
if [ -d "$INSTALL_DIR/.git" ]; then
  log "Updating existing checkout at $INSTALL_DIR ..."
  git -C "$INSTALL_DIR" fetch --quiet origin \
    && git -C "$INSTALL_DIR" checkout --quiet main \
    && git -C "$INSTALL_DIR" pull --ff-only --quiet origin main \
    || warn "git update failed; continuing with the current checkout."
elif [ -e "$INSTALL_DIR" ]; then
  die "$INSTALL_DIR exists but is not a git checkout. Move it aside, or set LLM_ROUTER_DIR."
else
  log "Cloning $REPO_SSH_URL -> $INSTALL_DIR ..."
  git clone --quiet "$REPO_SSH_URL" "$INSTALL_DIR" \
    || git clone --quiet "$REPO_HTTPS_URL" "$INSTALL_DIR" \
    || die "could not clone the repo (try manually: git clone $REPO_SSH_URL $INSTALL_DIR)."
fi

# --- 2. virtualenv ------------------------------------------------------------
VENV="$INSTALL_DIR/.venv"
if [ -x "$VENV/bin/python" ]; then
  log "Reusing existing venv at $VENV"
else
  log "Creating venv at $VENV ..."
  if command -v uv >/dev/null 2>&1; then
    uv venv "$VENV" --python "$PY"
  else
    "$PY" -m venv "$VENV"
  fi
fi
VPY="$VENV/bin/python"

# --- 3. install the package (editable) ---------------------------------------
log "Installing llm-router into the venv (pulls fastapi/uvicorn/httpx/PyYAML) ..."
if command -v uv >/dev/null 2>&1; then
  ( cd "$INSTALL_DIR" && uv pip install -e . --python "$VPY" )
else
  ( cd "$INSTALL_DIR" && "$VPY" -m pip install --upgrade pip --quiet )
  ( cd "$INSTALL_DIR" && "$VPY" -m pip install -e . --quiet )
fi

# --- 4. expose the `llm-router` command --------------------------------------
mkdir -p "$BIN_DIR"
ln -sf "$VENV/bin/llm-router" "$BIN_DIR/llm-router"
log "Linked $BIN_DIR/llm-router -> $VENV/bin/llm-router"

# --- 5. starter config --------------------------------------------------------
CFG="$INSTALL_DIR/config.yaml"
if [ ! -f "$CFG" ]; then
  if [ -f "$INSTALL_DIR/config.example.yaml" ]; then
    cp "$INSTALL_DIR/config.example.yaml" "$CFG"
    log "Created a starter config at $CFG — edit it for your models."
  else
    warn "no config.example.yaml found; create $CFG yourself (see the README)."
  fi
else
  log "Config already present at $CFG (left untouched)."
fi

# --- 6. PATH check ------------------------------------------------------------
case ":$PATH:" in
  *":$BIN_DIR:"*) : ;;
  *)
    warn "$BIN_DIR is not on your PATH. Add it to your shell, e.g.:"
    printf '      echo "export PATH=%s:$PATH" >> ~/.zshrc && source ~/.zshrc\n' "$BIN_DIR"
    ;;
esac

# --- 7. verify ----------------------------------------------------------------
if "$BIN_DIR/llm-router" --version >/dev/null 2>&1; then
  log "Success! $( "$BIN_DIR/llm-router" --version 2>&1 )"
else
  die "llm-router --version failed; check the steps above."
fi

cat <<'EOF'

Next steps:
  1. Add your model(s) to your config:
        llm-router config          # opens config.yaml in your $EDITOR
  2. Validate it (nothing starts):
        llm-router check
  3. Start the router (background daemon; the default model loads in the background):
        llm-router run
  4. Point your client at it (see README -> "Wiring it up (the custom OpenAI endpoint)"):
        base_url: http://127.0.0.1:8787/v1
  5. Open the web dashboard:
        llm-router dashboard
EOF
