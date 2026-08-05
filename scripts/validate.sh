#!/usr/bin/env bash
set -euo pipefail

parking_project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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
