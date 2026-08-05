#!/usr/bin/env bash
set -euo pipefail

parking_project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -x "${parking_project_root}/.venv/bin/python" ]]; then
    exec "${parking_project_root}/.venv/bin/python" "$@"
fi

if [[ -f "${parking_project_root}/.venv/Scripts/python.exe" ]]; then
    exec "${parking_project_root}/.venv/Scripts/python.exe" "$@"
fi

if command -v python3 >/dev/null 2>&1; then
    exec python3 "$@"
fi

echo "No Python runtime found. Create .venv or install python3." >&2
exit 1
