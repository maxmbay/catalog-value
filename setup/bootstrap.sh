#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

HOOK_MARKER="# catalog-value: direnv hook"

echo "Bootstrapping development environment..."

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed."
    echo "Install it from: https://docs.astral.sh/uv/"
    exit 1
fi

uv sync

if [[ ! -f .env ]]; then
    cp .env.example .env
    echo "Created .env from .env.example (fill in TMDB_API_KEY)."
else
    echo ".env already exists; leaving it unchanged."
fi

install_direnv() {
    if command -v direnv >/dev/null 2>&1; then
        echo "direnv already installed: $(command -v direnv)"
        return
    fi
    if command -v brew >/dev/null 2>&1; then
        echo "Installing direnv with Homebrew..."
        HOMEBREW_NO_AUTO_UPDATE=1 brew install direnv
        return
    fi
    echo "direnv is not installed and Homebrew was not found."
    echo "Install direnv from: https://direnv.net/docs/installation.html"
    exit 1
}

ensure_hook() {
    local rcfile="$1"
    local shell_name="$2"
    local hook_line="eval \"\$(direnv hook ${shell_name})\""

    mkdir -p "$(dirname "$rcfile")"
    if [[ ! -f "$rcfile" ]]; then
        touch "$rcfile"
    fi
    if grep -Eq 'direnv hook '"${shell_name}" "$rcfile"; then
        echo "direnv ${shell_name} hook already present in ${rcfile}"
        return
    fi
    {
        echo ""
        echo "${HOOK_MARKER}"
        echo "${hook_line}"
    } >>"$rcfile"
    echo "Added direnv ${shell_name} hook to ${rcfile}"
}

ensure_bash_login_sources_bashrc() {
    local profile="$HOME/.bash_profile"
    if [[ ! -f "$profile" ]]; then
        cat >"$profile" <<'EOF'
# Load bashrc for login shells (macOS Terminal, SSH).
if [ -f "$HOME/.bashrc" ]; then
    . "$HOME/.bashrc"
fi
EOF
        echo "Created ${profile} to source ~/.bashrc"
        return
    fi
    if grep -Eq '\.bashrc|source bashrc' "$profile"; then
        echo "${profile} already sources bashrc"
        return
    fi
    {
        echo ""
        echo "${HOOK_MARKER}: source bashrc for login bash"
        echo 'if [ -f "$HOME/.bashrc" ]; then'
        echo '    . "$HOME/.bashrc"'
        echo "fi"
    } >>"$profile"
    echo "Appended bashrc sourcing to ${profile}"
}

install_direnv
ensure_hook "$HOME/.zshrc" zsh
ensure_hook "$HOME/.bashrc" bash
ensure_bash_login_sources_bashrc

# Allow this directory even if the current shell is not hooked yet.
direnv allow "$ROOT"

echo "Done."
echo
echo "direnv will load .env automatically in new bash and zsh sessions."
echo "This shell:  source ~/.zshrc    # or: source ~/.bashrc"
echo "Then:        cd ${ROOT}"
echo
echo "Run commands with:"
echo "  uv run python ..."
echo "  uv run pytest"
