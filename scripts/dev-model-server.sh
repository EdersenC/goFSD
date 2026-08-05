#!/usr/bin/env bash
set -euo pipefail

parking_project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$(uname -s)" == "Linux" ]] \
    && uname -r | grep -qi microsoft \
    && [[ -f "${parking_project_root}/.venv/Scripts/python.exe" ]] \
    && command -v wslpath >/dev/null 2>&1; then
    parking_windows_server="$(wslpath -w "${parking_project_root}/fsd_trainer/src/gta_fsd/server.py")"
    parking_windows_config="$(wslpath -w "${parking_project_root}/fsd_trainer/train_config.toml")"
    if [[ -n "${FSD_DATA_ROOT:-}" ]]; then
        parking_data_root="${FSD_DATA_ROOT}"
        if [[ "${parking_data_root}" == /* ]]; then
            parking_data_root="$(wslpath -w "${parking_data_root}")"
        fi
        IFS=: read -r -a parking_shared_entries <<< "${WSLENV:-}"
        parking_shared_environment=""
        for parking_shared_entry in "${parking_shared_entries[@]}"; do
            if [[ -z "${parking_shared_entry}" \
                || "${parking_shared_entry}" == "FSD_DATA_ROOT" \
                || "${parking_shared_entry}" == FSD_DATA_ROOT/* ]]; then
                continue
            fi
            parking_shared_environment="${parking_shared_environment:+${parking_shared_environment}:}${parking_shared_entry}"
        done
        parking_shared_environment="${parking_shared_environment:+${parking_shared_environment}:}FSD_DATA_ROOT"
        exec env WSLENV="${parking_shared_environment}" FSD_DATA_ROOT="${parking_data_root}" \
            "${parking_project_root}/.venv/Scripts/python.exe" "${parking_windows_server}" \
            --config "${parking_windows_config}" "$@"
    fi
    exec "${parking_project_root}/.venv/Scripts/python.exe" "${parking_windows_server}" \
        --config "${parking_windows_config}" "$@"
fi

cd "${parking_project_root}/fsd_trainer/src/gta_fsd"
exec "${parking_project_root}/scripts/python.sh" server.py \
    --config "${parking_project_root}/fsd_trainer/train_config.toml" "$@"
