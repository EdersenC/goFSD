#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" == "Linux" ]] \
    && uname -r | grep -qi microsoft \
    && command -v go.exe >/dev/null 2>&1; then
    exec go.exe "$@"
fi

if command -v go >/dev/null 2>&1; then
    exec go "$@"
fi

echo "No Go runtime found. Install Windows Go for WSL Stop Sign Lab work, or Go for this host." >&2
exit 1
