#!/usr/bin/env bash
set -euo pipefail

stop_sign_project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
stop_sign_launcher="${stop_sign_project_root}/scripts/start-stop-sign-lab.sh"
stop_sign_setup="${stop_sign_project_root}/scripts/setup.mjs"

fail() {
    printf 'launcher test failed: %s\n' "$1" >&2
    exit 1
}

bash -n "${stop_sign_launcher}"

stop_sign_setup_output="$(node "${stop_sign_setup}" --dry-run)"
grep -Fq 'npm --prefix fivem ci --no-audit --no-fund' <<< "${stop_sign_setup_output}" \
    || fail "FiveM locked install is missing"
grep -Fq 'npm --prefix backend/cmd/web ci --no-audit --no-fund' <<< "${stop_sign_setup_output}" \
    || fail "web locked install is missing"
if node "${stop_sign_setup}" --not-a-real-option >/dev/null 2>&1; then
    fail "unknown setup options must fail"
fi

if [[ "$(uname -s)" != "Linux" ]] \
    || ! uname -r | grep -qi microsoft \
    || ! command -v powershell.exe >/dev/null 2>&1; then
    printf 'skip  launcher dry-run requires WSL with Windows PowerShell\n'
    exit 0
fi

stop_sign_output="$("${stop_sign_launcher}" --dry-run --no-browser --data-root /mnt/s/fsd_fivem_data)"
grep -Fq 'dry   backend:' <<< "${stop_sign_output}" || fail "backend plan is missing"
grep -Fq 'dry   model server:' <<< "${stop_sign_output}" || fail "model plan is missing"
grep -Fq 'skip  browser (--no-browser)' <<< "${stop_sign_output}" || fail "browser opt-out is missing"
grep -Fq 'S:\fsd_fivem_data' <<< "${stop_sign_output}" || fail "WSL data root was not converted"

stop_sign_collect_output="$("${stop_sign_launcher}" --dry-run --collect-only)"
grep -Fq 'skip  model server (--no-model)' <<< "${stop_sign_collect_output}" || fail "collection-only mode is missing"
grep -Fq 'dry   browser: http://127.0.0.1:8080' <<< "${stop_sign_collect_output}" || fail "browser plan is missing"

if "${stop_sign_launcher}" --dry-run --not-a-real-option >/dev/null 2>&1; then
    fail "unknown options must fail"
fi
if STOP_SIGN_BACKEND_URL=http://192.0.2.1:8080 "${stop_sign_launcher}" --dry-run >/dev/null 2>&1; then
    fail "non-loopback control URLs must fail"
fi

stop_sign_test_port=$((31000 + ($$ % 1000)))
node -e '
const http = require("node:http");
const port = Number(process.argv[1]);
const identities = new Map([
    ["/backend/healthz", "stop-sign-lab-backend"],
    ["/model/healthz", "stop-sign-lab-model"],
    ["/wrong/healthz", "not-stop-sign-lab"],
]);
const server = http.createServer((request, response) => {
    const service = identities.get(request.url);
    response.writeHead(200, { "Content-Type": "application/json" });
    response.end(JSON.stringify({ status: "ok", service }));
});
server.listen(port, "0.0.0.0");
process.on("SIGTERM", () => server.close());
' "${stop_sign_test_port}" >/dev/null 2>&1 &
stop_sign_test_server_pid=$!
cleanup_identity_server() {
    kill "${stop_sign_test_server_pid}" >/dev/null 2>&1 || true
    wait "${stop_sign_test_server_pid}" 2>/dev/null || true
}
trap cleanup_identity_server EXIT

stop_sign_test_server_ready=false
for _ in {1..30}; do
    if curl --fail --silent --max-time 1 \
        "http://127.0.0.1:${stop_sign_test_port}/backend/healthz" >/dev/null 2>&1; then
        stop_sign_test_server_ready=true
        break
    fi
    sleep 0.1
done
[[ "${stop_sign_test_server_ready}" == true ]] || fail "identity test server did not start"

stop_sign_identity_output="$(STOP_SIGN_BACKEND_URL="http://127.0.0.1:${stop_sign_test_port}/backend" \
    "${stop_sign_launcher}" --collect-only --no-browser --wait-seconds 2)"
grep -Fq 'ok    backend already running:' <<< "${stop_sign_identity_output}" \
    || fail "matching backend identity was not reused"

stop_sign_windows_launcher="$(wslpath -w "${stop_sign_project_root}/scripts/start-stop-sign-lab.ps1")"
if powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass \
    -File "${stop_sign_windows_launcher}" \
    -BackendUrl "http://127.0.0.1:${stop_sign_test_port}/wrong" \
    -ModelUrl "http://127.0.0.1:${stop_sign_test_port}/model" \
    -NoModel -NoBrowser -HealthCheckOnly >/dev/null 2>&1; then
    fail "wrong backend identity must not be reused"
fi
if powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass \
    -File "${stop_sign_windows_launcher}" \
    -BackendUrl "http://127.0.0.1:${stop_sign_test_port}/backend" \
    -ModelUrl "http://127.0.0.1:${stop_sign_test_port}/wrong" \
    -NoBrowser -HealthCheckOnly >/dev/null 2>&1; then
    fail "wrong model identity must not be reused"
fi
if ! powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass \
    -File "${stop_sign_windows_launcher}" \
    -BackendUrl "http://127.0.0.1:${stop_sign_test_port}/backend" \
    -ModelUrl "http://127.0.0.1:${stop_sign_test_port}/model" \
    -NoBrowser -HealthCheckOnly >/dev/null 2>&1; then
    fail "matching backend and model identities were not accepted"
fi

printf 'ok    launcher dry-run, argument guards, and service identities\n'
