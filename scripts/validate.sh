#!/usr/bin/env bash
set -euo pipefail

parking_project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"${parking_project_root}/scripts/test-start-parking-lab.sh"

npm --prefix "${parking_project_root}/backend/cmd/web" run typecheck
npm --prefix "${parking_project_root}/backend/cmd/web" test
npm --prefix "${parking_project_root}/backend/cmd/web" run build

npm --prefix "${parking_project_root}/fivem" run typecheck
npm --prefix "${parking_project_root}/fivem" test
npm --prefix "${parking_project_root}/fivem" run build

(
    cd "${parking_project_root}/backend"
    "${parking_project_root}/scripts/go.sh" test ./...
)

(
    cd "${parking_project_root}/fsd_trainer/src/gta_fsd"
    "${parking_project_root}/scripts/python.sh" -B -m unittest discover -p 'test_*.py'
)
