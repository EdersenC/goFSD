#!/usr/bin/env bash
set -euo pipefail

parking_project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$(uname -s)" == "Linux" ]] \
    && uname -r | grep -qi microsoft \
    && command -v powershell.exe >/dev/null 2>&1 \
    && command -v wslpath >/dev/null 2>&1; then
    parking_windows_script="$(wslpath -w "${parking_project_root}/scripts/dev-backend.ps1")"
    parking_powershell_args=(
        -NoLogo
        -NoProfile
        -NonInteractive
        -ExecutionPolicy Bypass
        -File "${parking_windows_script}"
    )
    if [[ -n "${FSD_DATA_ROOT:-}" ]]; then
        parking_data_root="${FSD_DATA_ROOT}"
        if [[ "${parking_data_root}" == /* ]]; then
            parking_data_root="$(wslpath -w "${parking_data_root}")"
        fi
        parking_powershell_args+=(-DataRoot "${parking_data_root}")
    fi
    exec powershell.exe "${parking_powershell_args[@]}"
fi

cd "${parking_project_root}/backend"
exec go run ./cmd
