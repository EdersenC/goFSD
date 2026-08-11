#!/usr/bin/env bash
set -euo pipefail

parking_project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
parking_launcher="${parking_project_root}/scripts/start-parking-lab.sh"
parking_setup="${parking_project_root}/scripts/setup.mjs"

fail() {
    printf 'launcher test failed: %s\n' "$1" >&2
    exit 1
}

bash -n "${parking_launcher}"

parking_setup_output="$(node "${parking_setup}" --dry-run)"
grep -Fq 'npm --prefix fivem ci --no-audit --no-fund' <<< "${parking_setup_output}" \
    || fail "FiveM locked install is missing"
grep -Fq 'npm --prefix backend/cmd/web ci --no-audit --no-fund' <<< "${parking_setup_output}" \
    || fail "web locked install is missing"
if node "${parking_setup}" --not-a-real-option >/dev/null 2>&1; then
    fail "unknown setup options must fail"
fi

if [[ "$(uname -s)" != "Linux" ]] \
    || ! uname -r | grep -qi microsoft \
    || ! command -v powershell.exe >/dev/null 2>&1; then
    printf 'skip  launcher dry-run requires WSL with Windows PowerShell\n'
    exit 0
fi

parking_output="$("${parking_launcher}" --dry-run --no-browser --data-root /mnt/s/fsd_fivem_data)"
grep -Fq 'dry   backend:' <<< "${parking_output}" || fail "backend plan is missing"
grep -Fq 'dry   model server:' <<< "${parking_output}" || fail "model plan is missing"
grep -Fq 'skip  browser (--no-browser)' <<< "${parking_output}" || fail "browser opt-out is missing"
grep -Fq 'S:\fsd_fivem_data' <<< "${parking_output}" || fail "WSL data root was not converted"

parking_collect_output="$("${parking_launcher}" --dry-run --collect-only)"
grep -Fq 'skip  model server (--no-model)' <<< "${parking_collect_output}" || fail "collection-only mode is missing"
grep -Fq 'dry   browser: http://127.0.0.1:8080' <<< "${parking_collect_output}" || fail "browser plan is missing"

if "${parking_launcher}" --dry-run --not-a-real-option >/dev/null 2>&1; then
    fail "unknown options must fail"
fi
if PARKING_BACKEND_URL=http://192.0.2.1:8080 "${parking_launcher}" --dry-run >/dev/null 2>&1; then
    fail "non-loopback control URLs must fail"
fi

parking_test_port=$((31000 + ($$ % 1000)))
node -e '
const http = require("node:http");
const port = Number(process.argv[1]);
const identities = new Map([
    ["/backend/healthz", "stop-sign-lab-backend"],
    ["/model/healthz", "stop-sign-lab-model"],
    ["/wrong/healthz", "not-parking-lab"],
]);
const server = http.createServer((request, response) => {
    const service = identities.get(request.url);
    response.writeHead(200, { "Content-Type": "application/json" });
    response.end(JSON.stringify({ status: "ok", service }));
});
server.listen(port, "0.0.0.0");
process.on("SIGTERM", () => server.close());
' "${parking_test_port}" >/dev/null 2>&1 &
parking_test_server_pid=$!
cleanup_identity_server() {
    kill "${parking_test_server_pid}" >/dev/null 2>&1 || true
    wait "${parking_test_server_pid}" 2>/dev/null || true
}
trap cleanup_identity_server EXIT

parking_test_server_ready=false
for _ in {1..30}; do
    if curl --fail --silent --max-time 1 \
        "http://127.0.0.1:${parking_test_port}/backend/healthz" >/dev/null 2>&1; then
        parking_test_server_ready=true
        break
    fi
    sleep 0.1
done
[[ "${parking_test_server_ready}" == true ]] || fail "identity test server did not start"

parking_identity_output="$(PARKING_BACKEND_URL="http://127.0.0.1:${parking_test_port}/backend" \
    "${parking_launcher}" --collect-only --no-browser --wait-seconds 2)"
grep -Fq 'ok    backend already running:' <<< "${parking_identity_output}" \
    || fail "matching backend identity was not reused"

parking_windows_launcher="$(wslpath -w "${parking_project_root}/scripts/start-parking-lab.ps1")"
if powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass \
    -File "${parking_windows_launcher}" \
    -BackendUrl "http://127.0.0.1:${parking_test_port}/wrong" \
    -ModelUrl "http://127.0.0.1:${parking_test_port}/model" \
    -NoModel -NoBrowser -HealthCheckOnly >/dev/null 2>&1; then
    fail "wrong backend identity must not be reused"
fi
if powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass \
    -File "${parking_windows_launcher}" \
    -BackendUrl "http://127.0.0.1:${parking_test_port}/backend" \
    -ModelUrl "http://127.0.0.1:${parking_test_port}/wrong" \
    -NoBrowser -HealthCheckOnly >/dev/null 2>&1; then
    fail "wrong model identity must not be reused"
fi
if ! powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass \
    -File "${parking_windows_launcher}" \
    -BackendUrl "http://127.0.0.1:${parking_test_port}/backend" \
    -ModelUrl "http://127.0.0.1:${parking_test_port}/model" \
    -NoBrowser -HealthCheckOnly >/dev/null 2>&1; then
    fail "matching backend and model identities were not accepted"
fi

printf 'ok    launcher dry-run, argument guards, and service identities\n'
