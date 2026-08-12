#!/usr/bin/env bash
set -euo pipefail

stop_sign_project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$(uname -s)" == "Linux" ]] \
    && uname -r | grep -qi microsoft \
    && [[ -f "${stop_sign_project_root}/.venv/Scripts/python.exe" ]] \
    && command -v wslpath >/dev/null 2>&1; then
    stop_sign_windows_server="$(wslpath -w "${stop_sign_project_root}/fsd_trainer/src/gta_fsd/server.py")"
    stop_sign_windows_config="$(wslpath -w "${stop_sign_project_root}/fsd_trainer/train_config.toml")"
    if [[ -n "${FSD_DATA_ROOT:-}" ]]; then
        stop_sign_data_root="${FSD_DATA_ROOT}"
        if [[ "${stop_sign_data_root}" == /* ]]; then
            stop_sign_data_root="$(wslpath -w "${stop_sign_data_root}")"
        fi
        IFS=: read -r -a stop_sign_shared_entries <<< "${WSLENV:-}"
        stop_sign_shared_environment=""
        for stop_sign_shared_entry in "${stop_sign_shared_entries[@]}"; do
            if [[ -z "${stop_sign_shared_entry}" \
                || "${stop_sign_shared_entry}" == "FSD_DATA_ROOT" \
                || "${stop_sign_shared_entry}" == FSD_DATA_ROOT/* ]]; then
                continue
            fi
            stop_sign_shared_environment="${stop_sign_shared_environment:+${stop_sign_shared_environment}:}${stop_sign_shared_entry}"
        done
        stop_sign_shared_environment="${stop_sign_shared_environment:+${stop_sign_shared_environment}:}FSD_DATA_ROOT"
        exec env WSLENV="${stop_sign_shared_environment}" FSD_DATA_ROOT="${stop_sign_data_root}" \
            "${stop_sign_project_root}/.venv/Scripts/python.exe" "${stop_sign_windows_server}" \
            --config "${stop_sign_windows_config}" "$@"
    fi
    exec "${stop_sign_project_root}/.venv/Scripts/python.exe" "${stop_sign_windows_server}" \
        --config "${stop_sign_windows_config}" "$@"
fi

cd "${stop_sign_project_root}/fsd_trainer/src/gta_fsd"
exec "${stop_sign_project_root}/scripts/python.sh" server.py \
    --config "${stop_sign_project_root}/fsd_trainer/train_config.toml" "$@"
