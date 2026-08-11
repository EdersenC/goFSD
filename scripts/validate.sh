#!/usr/bin/env bash
set -euo pipefail

stop_sign_project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"${stop_sign_project_root}/scripts/test-start-stop-sign-lab.sh"
node --check "${stop_sign_project_root}/scripts/audit-stop-sign-data.mjs"
node --check "${stop_sign_project_root}/scripts/queue-stop-sign-training.mjs"
node --check "${stop_sign_project_root}/scripts/lib/stop-sign-data-audit.mjs"

npm --prefix "${stop_sign_project_root}/backend/cmd/web" run typecheck
npm --prefix "${stop_sign_project_root}/backend/cmd/web" test
npm --prefix "${stop_sign_project_root}/backend/cmd/web" run build

npm --prefix "${stop_sign_project_root}/fivem" run typecheck
npm --prefix "${stop_sign_project_root}/fivem" test
npm --prefix "${stop_sign_project_root}/fivem" run build

(
    cd "${stop_sign_project_root}/backend"
    "${stop_sign_project_root}/scripts/go.sh" test ./...
)

(
    cd "${stop_sign_project_root}/fsd_trainer/src/gta_fsd"
    "${stop_sign_project_root}/scripts/python.sh" -B -m unittest discover -p 'test_*.py'
)
