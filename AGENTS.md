# Repository Guidelines

## Project Structure & Module Organization
The repo has three active modules:
- `fivem/`: TypeScript FiveM resource code. Main files are in `fivem/src/` (for example `client.ts`, `server.ts`, `egoService.ts`); the forward-bay parking workflow is isolated under `fivem/src/parking/`, and legacy driving datasets remain under `fivem/src/datasets/`. Build output goes to `fivem/dist/`.
- `backend/`: Go capture and control API. HTTP entrypoint is `backend/cmd/main.go`; embedded control-page assets are in `backend/cmd/web/`; capture logic lives in `backend/internal/capture/`; post-processing and dataset generation live in `backend/internal/dataset/`; command queue/state logic lives in `backend/internal/control/`. Runtime artifacts default to `FSD_DATA_ROOT` when set, otherwise `S:\\fsd_fivem_data` on Windows.
- `fsd_trainer/`: Python temporal-planner dataset, training, inference, and model-server code. Shared parking-first runtime settings live in `fsd_trainer/train_config.toml`.
- Run outputs land under `FSD_DATA_ROOT\\runs\\<run-id>\\<scene-id>_<variant>\\` when `FSD_DATA_ROOT` is set, otherwise under `S:\\fsd_fivem_data\\runs\\...`: the run-level manifest is `run.jsonl` inside the scene folder, and each trip gets its own folder such as `trip-000/` with `video.mkv`, `video.log`, `metadata.json`, `processing.json`, `dataset.jsonl`, and extracted `frames/`.

## Build, Test, and Development Commands
- `npm --prefix fivem install`: install FiveM TypeScript dependencies.
- `npm --prefix fivem run build`: one-time bundle of client/server scripts into `fivem/dist/`.
- `npm --prefix fivem run watch`: rebuild on changes for local iteration.
- `npm --prefix fivem test`: run focused TypeScript unit tests.
- `cd backend && ../scripts/go.sh test ./...`: run Go unit tests with native Go, or Windows Go when working from WSL.
- `cd backend && go run ./cmd`: start the capture API on `127.0.0.1:8080` (`HOST` / `PORT` override).
- `cd backend && FSD_DATA_ROOT=S:\\fsd_fivem_data go run ./cmd process-runs --workers 4`: backfill trip frame extraction and dataset manifests in parallel from `FSD_DATA_ROOT\\runs`.
- `cd backend && go run ./cmd process-runs --workers 4 --force`: rerun processing and overwrite existing `frames/` and `dataset.jsonl`.
- `cd fsd_trainer/src/gta_fsd && ../../../scripts/python.sh -B -m unittest discover -p 'test_*.py'`: run Python unit tests with the project interpreter without generating bytecode caches.
- `bash scripts/validate.sh`: run the repository-wide TypeScript, Go, Python, and build checks.

Quick API smoke test:
```bash
curl -s http://localhost:8080/capture/sources
curl -s -X POST http://localhost:8080/capture/start -H "Content-Type: application/json" -d '{}'
curl -s -X POST http://localhost:8080/capture/stop
curl -s http://localhost:8080/control/state
curl -s -X POST http://localhost:8080/control/command -H "Content-Type: application/json" -d '{"type":"setParkingTarget"}'
curl -s -X POST http://localhost:8080/control/command -H "Content-Type: application/json" -d '{"type":"startParkingRun","attemptCount":5,"seed":"smoke-1"}'
curl -s -X POST http://localhost:8080/control/command -H "Content-Type: application/json" -d '{"type":"prepareParkingEvaluation","seed":"smoke-1"}'
```

## Coding Style & Naming Conventions
- TypeScript: `strict` mode is enabled (`fivem/tsconfig.json`), so avoid `any` unless justified. Use `camelCase` for variables/functions, `PascalCase` for types/classes, and kebab-case filenames for datasets (for example `inner-city-driving.ts`).
- Go: use standard library HTTP (`net/http`) for backend routes. Run `gofmt` before committing. Use `PascalCase` for exported names and `camelCase` for internals.

## Testing Guidelines
- Go tests live in `backend/internal/capture/service_test.go` and `backend/internal/control/store_test.go`.
- Minimum PR validation:
  - `cd backend && ../scripts/go.sh test ./...`
  - `npm --prefix fivem test`
  - `npm --prefix fivem run build`
  - `cd fsd_trainer/src/gta_fsd && ../../../scripts/python.sh -B -m unittest discover -p 'test_*.py'`
- Add new Go tests as `*_test.go` adjacent to the package under test.
- For control/capture changes, include manual endpoint verification (`/control/state`, `/control/command`, `/capture/start`, `/capture/stop`) in the PR notes.
- If trip capture behavior changes, verify `run.jsonl` plus the per-trip `video.mkv`, `video.log`, `metadata.json`, `processing.json`, `dataset.jsonl`, and `frames/` outputs under `FSD_DATA_ROOT\\runs\\<run-id>\\<scene-id>_<variant>\\` or the fallback root.
- If parking behavior changes, verify one successful and one failed attempt. Confirm the calibrated goal/start offset and `parkingOutcome` are stored, and that failed attempts remain inspectable but are excluded from expert training by default.
- For dataset backfills, note whether the run used normal skip behavior or `--force` overwrite mode.

## Commit & Pull Request Guidelines
- Existing commits are short, imperative summaries, often with a prefix like `refactor:` or `add`.
- Follow `type: concise description` (example: `feat: auto-select capture source on start`).
- PRs should include:
  - What changed and why.
  - Local verification commands/results.
  - Linked issue/task (if applicable).
  - Notes on runtime output impacts (`FSD_DATA_ROOT` / storage root, API response shape, ffmpeg args, browser control flow).
