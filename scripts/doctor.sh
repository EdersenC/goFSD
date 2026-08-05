#!/usr/bin/env bash
set -uo pipefail

parking_project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
parking_failures=0

pass() {
    printf 'ok    %s\n' "$1"
}

warn() {
    printf 'warn  %s\n' "$1"
}

fail() {
    printf 'fail  %s\n' "$1"
    parking_failures=$((parking_failures + 1))
}

check_command() {
    local command_name="$1"
    if command -v "${command_name}" >/dev/null 2>&1; then
        pass "${command_name}: $(command -v "${command_name}")"
    else
        fail "${command_name} is not on PATH"
    fi
}

echo "Parking Lab doctor"
check_command node
check_command npm

parking_is_wsl=false
if [[ "$(uname -s)" == "Linux" ]] && uname -r | grep -qi microsoft; then
    parking_is_wsl=true
fi

if [[ "${parking_is_wsl}" == true ]]; then
    if command -v powershell.exe >/dev/null 2>&1; then
        pass "WSL-to-Windows PowerShell bridge is available"
        if command -v go.exe >/dev/null 2>&1; then
            pass "Windows Go runtime: $(command -v go.exe)"
        else
            fail "Windows Go is unavailable; install it for the capture backend"
        fi
        if command -v ffmpeg.exe >/dev/null 2>&1; then
            pass "Windows ffmpeg: $(command -v ffmpeg.exe)"
        else
            fail "Windows ffmpeg is unavailable; install it for parking video capture"
        fi
    else
        fail "powershell.exe is unavailable; live capture and virtual-controller work must run on Windows"
    fi
else
    check_command go
    check_command ffmpeg
fi

if "${parking_project_root}/scripts/python.sh" -B -c \
    'import PIL, torch, torchvision; print(f"ok    python stack: torch={torch.__version__} torchvision={torchvision.__version__} pillow={PIL.__version__}")'; then
    :
else
    warn "Python ML dependencies are unavailable; collection works, but training/inference needs fsd_trainer/requirements.txt"
fi

if [[ -d "${parking_project_root}/fivem/node_modules" ]]; then
    pass "FiveM dependencies are installed"
else
    fail "FiveM dependencies are missing; run npm --prefix fivem install"
fi

if [[ -n "${FSD_DATA_ROOT:-}" ]]; then
    pass "FSD_DATA_ROOT=${FSD_DATA_ROOT}"
elif [[ -d /mnt/s/fsd_fivem_data ]]; then
    pass "data root: /mnt/s/fsd_fivem_data (Windows S:\\fsd_fivem_data)"
else
    warn "FSD_DATA_ROOT is unset; Windows runtime will default to S:\\fsd_fivem_data"
fi

if ((parking_failures > 0)); then
    printf '\nDoctor found %d blocking issue(s).\n' "${parking_failures}"
    exit 1
fi

printf '\nReady for Parking Lab development.\n'
