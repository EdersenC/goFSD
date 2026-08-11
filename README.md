# GTA Stop Sign Lab

This repository is a temporal-driving lab for one focused V0 behavior: approach a stop sign, brake to an exact stop, then continue through the intersection. Each scene is collected as three short, independent temporal clips instead of one long recording.

The three clip stages are:

1. **Approach** — launch and establish the approach before braking begins.
2. **Brake + Stop** — slow down and end immediately after a brief zero-speed confirmation.
3. **Release** — begin at the captured Stop pose and drive to the captured End pose.

Every stage has its own synchronization flash and capture lifecycle. Temporal windows and future labels stay inside that stage, so training never crosses an Approach → Brake or Brake → Release boundary. There is no long dwell recording or dwell control in the collection UI.

## Start the lab

1. Open [`stop-sign-lab.code-workspace`](stop-sign-lab.code-workspace).
2. Run the **Stop Sign Lab: Start** task, or run:

   ```bash
   npm start
   ```

3. Keep the backend and optional model-server terminals open. The launcher opens the workbench at [http://127.0.0.1:8080/](http://127.0.0.1:8080/) after the backend is ready.

Use `npm run start:collect` when you only need collection and data preparation. The model server is not required to collect expert demonstrations. The architecture view is available at [http://127.0.0.1:8080/architecture](http://127.0.0.1:8080/architecture).

### First-machine setup and FiveM deployment

Install the locked dependencies once:

```bash
npm run setup
```

Build and deploy the FiveM resource whenever `fivem/` changes:

```bash
export FIVEM_RESOURCE_DIR=/absolute/path/to/cfx-server-data/resources/FSD
npm --prefix fivem run build:deploy
```

The deploy command validates the existing resource directory and copies only `dist/client.js`, `dist/server.js`, and `fxmanifest.lua`. It does not restart FiveM. In the FiveM server console, run:

```text
restart FSD
```

Then join the session and confirm the workbench status strip reports the API online, FiveM linked, and Control synchronized. Do not run the deploy watcher with `sudo`; point `FIVEM_RESOURCE_DIR` at the resource owned by the normal FiveM user.

## One operator workbench

The Stop Sign Lab uses one persistent **Collect → Data → Train → Evaluate** workbench instead of multiple workflow shells:

| Area | Purpose |
|---|---|
| **Collect** | Search all 463 verified main-map signs, teleport to one, capture Start → Stop → End, and replay the scene through deterministic auto-variants. |
| **Data** | Process completed trips, inspect readiness and phase coverage, and keep failed attempts diagnosable but out of expert training by default. |
| **Train** | Train from independent stop-sign locations, with phase-balanced sampling and a fixed temporal output contract. |
| **Evaluate** | Load a compatible checkpoint, pass the guarded safety preflight, and compare predicted speed/stop intent with the closed-loop result. |

The red **Hold** control stays available at the bottom-right. `Alt+Shift+H` is the keyboard shortcut. Hold stops collection, vehicle motion, and inference, invalidates stale motion-start epochs, and waits for FiveM to confirm a fresh idle safety state.

## First stop-sign collection

1. Search the **Stop-sign catalog** and press **Teleport** on any of the 463 signs. Teleport creates or reopens that sign's scene and places the managed setup car on a nearby drivable lane.
2. Move the car to the exact beginning of the demonstration and press **Capture Start**.
3. Move it to the exact vehicle-center stopping point and press **Capture Stop**.
4. Move it beyond the sign to the desired continuation point and press **Capture End**.
5. Keep the default **50 variants** and **20% motion variance**, or adjust them. Motion variance is capped at `25%`; weather, clock time, and vehicle color vary broadly.
6. Press **Collect this scene**. Every variant produces separate `approach`, `brake_stop`, and `release` clips.
7. Choose the next catalog sign and repeat. Saved scenes remain in the browser so the workflow is Teleport → Start → Stop → End → Collect.
8. Use **End collection** for an orderly stop, or **Hold** when motion must stop immediately.

The seed makes expansion reproducible: the same scene, seed, variant count, and variance bound produce the same jobs. Variant `auto-001` preserves the captured scene exactly; later variants perturb route distance and speed within the selected bound, apply only centimeter-scale Stop jitter, and sample broader visual conditions. See [`docs/stop-sign-collection.md`](docs/stop-sign-collection.md) for the plan contract and collection checklist.

## Scene geometry contract

The operator directly captures the three poses that define behavior:

```text
Start  ── approach ──> braking boundary ── brake_stop ──> Stop
                                                               └── release ──> End
```

- `Start` is the exact reset pose for the approach clip.
- `Stop` is the exact target for the vehicle center at zero speed.
- `End` is the exact destination for the release clip.
- The selected catalog prop supplies stable physical-sign identity and navigation coordinates, not a model input.

The UI and backend require Start to be before Stop, End to be beyond Stop, and all three poses to remain compatible with one approach lane. FiveM receives explicit validated poses for every generated variant.

## Data and model contract

Set `FSD_DATA_ROOT` to choose the shared capture/training root. Without an override, the Windows runtime defaults to `S:\fsd_fivem_data`; use the same root for the backend and model server.

Fresh runs use scene `stop-sign:temporal-v1` and this layout:

```text
<data-root>/runs/<run-id>/stop-sign_temporal-v1/
├── run.jsonl
└── trip-000/
    ├── video.mkv
    ├── video.log
    ├── metadata.json
    ├── processing.json
    ├── dataset.jsonl
    └── frames/
```

`metadata.json` and `run.jsonl` retain `stopSignGoal`, `stopSignOutcome`, `clipStage`, catalog/location identity, resolved variant, seed, attempt index, and synchronized telemetry. One variant produces three trip folders—one each for `approach`, `brake_stop`, and `release`. `dataset.jsonl` contains causal RGB windows, telemetry history/future, fine phase, future targets, eligibility, and a location-level split group.

The current model consumes five causal RGB frames plus current-speed history. Stop-sign pose, stop-line distance, ego error, and other oracle geometry are labels/scoring metadata only; they are not perception inputs. The primary outputs are:

- `future_speed_mps`
- `stop_intent`

They are predicted at short fixed horizons of `100, 250, 500, 1000 ms`. Samples are stage-locked: image history and future targets must be available inside the same clip, and incomplete edge windows are excluded. A deterministic controller converts the nearest motion target into slew-limited, mutually exclusive throttle or brake, then the virtual controller applies it to GTA. Recorded expert throttle, expert brake, and physical brake pressure are auxiliary diagnostics—not the model-to-game interface.

See [`docs/stop-sign-model-control.md`](docs/stop-sign-model-control.md) for the complete data, training, controller, and safety boundary.

## Training from a clean slate

The checked-in [`fsd_trainer/train_config.toml`](fsd_trainer/train_config.toml) intentionally contains no run IDs or checkpoint. A fresh workspace showing zero runs and zero checkpoints is expected.

Training admits successful, complete stage clips by default, balances the observed `(clip stage, fine phase)` groups, and prevents a physical stop-sign location from appearing in both training and validation. Failed clips remain inspectable but are excluded from expert training by default. A few clips prove the pipeline, not model readiness; collect many variants across many independent physical signs before training a useful checkpoint.

Before training, run the contract audit. It verifies every referenced RGB image, dense 50 ms telemetry ordering, frame spacing, phase/variant/location labels, horizon target alignment, and stable RGB-to-game-time offset:

```bash
npm run data:audit
```

With the full lab running, one command reconciles pending processing, chooses a location-disjoint train/validation split, and queues the job:

```bash
npm run train:stop-sign -- --epochs 150
```

Use `npm run train:stop-sign -- --dry-run` to print the exact split without starting training.

Do not process or rebuild paths while a training job is actively reading them. The normal workflow is to finish collection, use **Data** to make processing current, then queue training from **Train**.

## Components

| Area | Responsibility |
|---|---|
| `fivem/` | Stop-sign scene replay, three-stage capture, deterministic expert, world/vehicle execution, telemetry, and trip metadata. |
| `backend/` | Capture and processing, guarded command queue, Stop Sign Lab web UI, training bridge, and deterministic controller safety. |
| `fsd_trainer/` | Stop-sign dataset loading, phase balancing, temporal planner training, checkpoint contract, and inference serving. |
| `scripts/` | Setup, diagnostics, launchers, deployment support, and repository validation. |

## Validation

Run the repository validation from the root:

```bash
bash scripts/validate.sh
```

Or run the layers independently:

```bash
npm --prefix backend/cmd/web test
npm --prefix backend/cmd/web run build
npm --prefix fivem run typecheck
npm --prefix fivem test
npm --prefix fivem run build
(cd backend && ../scripts/go.sh test ./...)
(cd fsd_trainer/src/gta_fsd && ../../../scripts/python.sh -B -m unittest discover -p 'test_*.py')
```

Automated validation does not prove live GTA behavior, capture-source selection, controller calibration, or a checkpoint's closed-loop performance. Complete [`docs/stop-sign-live-acceptance.md`](docs/stop-sign-live-acceptance.md) before calling the V0 loop accepted.
