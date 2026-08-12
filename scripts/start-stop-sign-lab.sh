#!/usr/bin/env bash
set -euo pipefail

stop_sign_project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
stop_sign_backend_url="${STOP_SIGN_BACKEND_URL:-http://127.0.0.1:8080}"
stop_sign_model_url="${STOP_SIGN_MODEL_URL:-http://127.0.0.1:8090}"
stop_sign_data_root="${FSD_DATA_ROOT:-}"
stop_sign_no_model=false
stop_sign_no_browser=false
stop_sign_dry_run=false
stop_sign_wait_seconds=90

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
            stop_sign_no_model=true
            ;;
        --no-browser)
            stop_sign_no_browser=true
            ;;
        --dry-run)
            stop_sign_dry_run=true
            ;;
        --data-root)
            if (($# < 2)) || [[ -z "$2" ]]; then
                printf 'error --data-root requires a path\n' >&2
                exit 2
            fi
            stop_sign_data_root="$2"
            shift
            ;;
        --data-root=*)
            stop_sign_data_root="${1#*=}"
            if [[ -z "${stop_sign_data_root}" ]]; then
                printf 'error --data-root requires a path\n' >&2
                exit 2
            fi
            ;;
        --wait-seconds)
            if (($# < 2)) || [[ ! "$2" =~ ^[0-9]+$ ]] || ((10#$2 < 1 || 10#$2 > 300)); then
                printf 'error --wait-seconds must be an integer from 1 to 300\n' >&2
                exit 2
            fi
            stop_sign_wait_seconds="$2"
            shift
            ;;
        --wait-seconds=*)
            stop_sign_wait_seconds="${1#*=}"
            if [[ ! "${stop_sign_wait_seconds}" =~ ^[0-9]+$ ]] \
                || ((10#${stop_sign_wait_seconds} < 1 || 10#${stop_sign_wait_seconds} > 300)); then
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
From Windows PowerShell, use: .\scripts\start-stop-sign-lab.ps1
EOF
    exit 1
fi

stop_sign_windows_data_root="${stop_sign_data_root}"
if [[ -n "${stop_sign_windows_data_root}" && "${stop_sign_windows_data_root}" == /* ]]; then
    stop_sign_windows_data_root="$(wslpath -w "${stop_sign_windows_data_root}")"
fi

stop_sign_windows_script="$(wslpath -w "${stop_sign_project_root}/scripts/start-stop-sign-lab.ps1")"
stop_sign_powershell_args=(
    -NoLogo
    -NoProfile
    -NonInteractive
    -ExecutionPolicy Bypass
    -File "${stop_sign_windows_script}"
    -BackendUrl "${stop_sign_backend_url}"
    -ModelUrl "${stop_sign_model_url}"
    -WaitSeconds "${stop_sign_wait_seconds}"
)
if [[ -n "${stop_sign_windows_data_root}" ]]; then
    stop_sign_powershell_args+=(-DataRoot "${stop_sign_windows_data_root}")
fi
if [[ "${stop_sign_no_model}" == true ]]; then
    stop_sign_powershell_args+=(-NoModel)
fi
if [[ "${stop_sign_no_browser}" == true ]]; then
    stop_sign_powershell_args+=(-NoBrowser)
fi
if [[ "${stop_sign_dry_run}" == true ]]; then
    stop_sign_powershell_args+=(-DryRun)
else
    if ! service_identity_matches "${stop_sign_backend_url}" "stop-sign-lab-backend"; then
		printf 'build Stop Sign Lab web workspace\n'
        npm --prefix "${stop_sign_project_root}/backend/cmd/web" run build
    fi
fi

# WSL uses its working Node installation, then the Windows backend consumes the embedded bundle.
stop_sign_powershell_args+=(-SkipWebBuild)
exec powershell.exe "${stop_sign_powershell_args[@]}"
