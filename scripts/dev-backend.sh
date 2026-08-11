#!/usr/bin/env bash
set -euo pipefail

stop_sign_project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
stop_sign_backend_mode="${1:-serve}"

if [[ "${stop_sign_backend_mode}" == "serve" ]]; then
    npm --prefix "${stop_sign_project_root}/backend/cmd/web" run build
fi

if [[ "$(uname -s)" == "Linux" ]] \
    && uname -r | grep -qi microsoft \
    && command -v powershell.exe >/dev/null 2>&1 \
    && command -v wslpath >/dev/null 2>&1; then
    stop_sign_windows_script="$(wslpath -w "${stop_sign_project_root}/scripts/dev-backend.ps1")"
    stop_sign_powershell_args=(
        -NoLogo
        -NoProfile
        -NonInteractive
        -ExecutionPolicy Bypass
        -File "${stop_sign_windows_script}"
    )
    if [[ -n "${FSD_DATA_ROOT:-}" ]]; then
        stop_sign_data_root="${FSD_DATA_ROOT}"
        if [[ "${stop_sign_data_root}" == /* ]]; then
            stop_sign_data_root="$(wslpath -w "${stop_sign_data_root}")"
        fi
        stop_sign_powershell_args+=(-DataRoot "${stop_sign_data_root}")
    fi
    # The WSL launcher already built the bundle with the working Linux Node runtime.
    stop_sign_powershell_args+=(-SkipWebBuild)
    stop_sign_powershell_args+=("$@")
    exec powershell.exe "${stop_sign_powershell_args[@]}"
fi

cd "${stop_sign_project_root}/backend"
exec go run ./cmd "$@"
