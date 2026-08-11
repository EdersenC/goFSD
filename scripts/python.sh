#!/usr/bin/env bash
set -euo pipefail

stop_sign_project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -x "${stop_sign_project_root}/.venv/bin/python" ]]; then
    exec "${stop_sign_project_root}/.venv/bin/python" "$@"
fi

if [[ -f "${stop_sign_project_root}/.venv/Scripts/python.exe" ]]; then
    exec "${stop_sign_project_root}/.venv/Scripts/python.exe" "$@"
fi

if command -v python3 >/dev/null 2>&1; then
    exec python3 "$@"
fi

echo "No Python runtime found. Create .venv or install python3." >&2
exit 1
