#!/usr/bin/env bash
set -euo pipefail

parking_project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
parking_backend_url="${PARKING_BACKEND_URL:-http://127.0.0.1:8080}"
parking_model_url="${PARKING_MODEL_URL:-http://127.0.0.1:8090}"
parking_data_root="${FSD_DATA_ROOT:-}"
parking_no_model=false
parking_no_browser=false
parking_dry_run=false
parking_wait_seconds=90

usage() {
    cat <<'EOF'
Usage: npm start -- [options]

Options:
  --collect-only       Start the backend without the model server.
  --no-browser         Do not open the Stop Sign Lab workspace.
  --data-root PATH     Use the same capture/model data root for both services.
  --wait-seconds N     Startup timeout for each service (default: 90).
  --dry-run            Print the launch plan without starting anything.
  -h, --help           Show this help.
EOF
}

service_identity_matches() {
    local service_url="$1"
    local expected_service="$2"
    local response=""

    response="$(curl --fail --silent --connect-timeout 1 --max-time 2 \
        "${service_url%/}/healthz" 2>/dev/null)" || return 1
    printf '%s' "${response}" | node -e '
let raw = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => { raw += chunk; });
process.stdin.on("end", () => {
    try {
        const value = JSON.parse(raw);
        if (value?.status === "ok" && value?.service === process.argv[1]) {
            process.exit(0);
        }
    } catch {}
    process.exit(1);
});
' "${expected_service}" >/dev/null 2>&1
}

while (($# > 0)); do
    case "$1" in
        --collect-only|--no-model)
            parking_no_model=true
            ;;
        --no-browser)
            parking_no_browser=true
            ;;
        --dry-run)
            parking_dry_run=true
            ;;
        --data-root)
            if (($# < 2)) || [[ -z "$2" ]]; then
                printf 'error --data-root requires a path\n' >&2
                exit 2
            fi
            parking_data_root="$2"
            shift
            ;;
        --data-root=*)
            parking_data_root="${1#*=}"
            if [[ -z "${parking_data_root}" ]]; then
                printf 'error --data-root requires a path\n' >&2
                exit 2
            fi
            ;;
        --wait-seconds)
            if (($# < 2)) || [[ ! "$2" =~ ^[0-9]+$ ]] || ((10#$2 < 1 || 10#$2 > 300)); then
                printf 'error --wait-seconds must be an integer from 1 to 300\n' >&2
                exit 2
            fi
            parking_wait_seconds="$2"
            shift
            ;;
        --wait-seconds=*)
            parking_wait_seconds="${1#*=}"
            if [[ ! "${parking_wait_seconds}" =~ ^[0-9]+$ ]] \
                || ((10#${parking_wait_seconds} < 1 || 10#${parking_wait_seconds} > 300)); then
                printf 'error --wait-seconds must be an integer from 1 to 300\n' >&2
                exit 2
            fi
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'error unknown option: %s\n\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

if [[ "$(uname -s)" != "Linux" ]] \
    || ! uname -r | grep -qi microsoft \
    || ! command -v powershell.exe >/dev/null 2>&1 \
    || ! command -v wslpath >/dev/null 2>&1; then
    cat >&2 <<'EOF'
error The one-command launcher must run from WSL because live capture and controller access run on Windows.
From Windows PowerShell, use: .\scripts\start-parking-lab.ps1
EOF
    exit 1
fi

parking_windows_data_root="${parking_data_root}"
if [[ -n "${parking_windows_data_root}" && "${parking_windows_data_root}" == /* ]]; then
    parking_windows_data_root="$(wslpath -w "${parking_windows_data_root}")"
fi

parking_windows_script="$(wslpath -w "${parking_project_root}/scripts/start-parking-lab.ps1")"
parking_powershell_args=(
    -NoLogo
    -NoProfile
    -NonInteractive
    -ExecutionPolicy Bypass
    -File "${parking_windows_script}"
    -BackendUrl "${parking_backend_url}"
    -ModelUrl "${parking_model_url}"
    -WaitSeconds "${parking_wait_seconds}"
)
if [[ -n "${parking_windows_data_root}" ]]; then
    parking_powershell_args+=(-DataRoot "${parking_windows_data_root}")
fi
if [[ "${parking_no_model}" == true ]]; then
    parking_powershell_args+=(-NoModel)
fi
if [[ "${parking_no_browser}" == true ]]; then
    parking_powershell_args+=(-NoBrowser)
fi
if [[ "${parking_dry_run}" == true ]]; then
    parking_powershell_args+=(-DryRun)
else
    if ! service_identity_matches "${parking_backend_url}" "stop-sign-lab-backend"; then
		printf 'build Stop Sign Lab web workspace\n'
        npm --prefix "${parking_project_root}/backend/cmd/web" run build
    fi
fi

# WSL uses its working Node installation, then the Windows backend consumes the embedded bundle.
parking_powershell_args+=(-SkipWebBuild)
exec powershell.exe "${parking_powershell_args[@]}"
