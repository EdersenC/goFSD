# GTA Stop Sign Lab

This repository is a temporal-driving lab for one focused V0 behavior: launch from rest, approach a stop sign, brake to the correct stopping point, hold for about five seconds, and leave under a scripted release policy.

The four operator milestones are:

1. **Launch** — apply throttle and establish forward motion.
2. **Approach + Brake** — cruise toward the sign, then decelerate without crossing the stop line.
3. **Stop + Dwell** — stop at the derived ego pose and remain stopped for the configured dwell, normally `5000 ms`.
4. **Go** — V0 releases with `scripted_dwell_release_v0`. The model does not decide when the dwell is finished yet.

Parking and general-driving modules may remain in the tree for reference, but they are legacy paths. They are not the active product, dataset contract, or operator workflow.

## Start the lab

1. Open [`parking-lab.code-workspace`](parking-lab.code-workspace). The filename is retained for compatibility; its tasks and folder name are Stop Sign Lab.
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

Then join the session and confirm the workbench status strip reports both the API and FiveM online. Do not run the deploy watcher with `sudo`; point `FIVEM_RESOURCE_DIR` at the resource owned by the normal FiveM user.

## One operator workbench

The Stop Sign Lab uses one persistent **Collect → Data → Train → Evaluate** workbench instead of multiple workflow shells:

| Area | Purpose |
|---|---|
| **Collect** | Register stop-sign poses, set base conditions and variants, queue ordered attempts, and watch the live phase/telemetry. |
| **Data** | Process completed trips, inspect readiness and phase coverage, and keep failed attempts diagnosable but out of expert training by default. |
| **Train** | Train from independent stop-sign locations, with phase-balanced sampling and a fixed temporal output contract. |
| **Evaluate** | Load a compatible checkpoint, pass the guarded safety preflight, and compare predicted speed/stop intent with the closed-loop result. |

The red **Hold** control stays available at the bottom-right. `Alt+Shift+H` is the keyboard shortcut. Hold stops collection, vehicle motion, and inference, invalidates stale motion-start epochs, and waits for FiveM to confirm a fresh idle safety state.

## First stop-sign collection

1. Select **Start setup car** and enter the driver seat.
2. Place the setup car at the stop-sign reference point with its heading aligned to the vehicle's intended travel direction. The heading describes travel through the intersection, not the physical sign prop's facing direction.
3. Select **Calibrate current sign**, then copy the accepted live pose into the desired plan entry with **Use live pose**.
4. Set the stop-line distance, ego-center offset, approach distance, target speed, dwell, attempts, and base environment.
5. Add variations for weather, time, vehicle model/color, speed, distances, dwell, or attempt count. Blank variation fields inherit the entry's base values.
6. Add more physical stop-sign entries as needed, review the total signs/variations/attempts, then select **Queue collection**.
7. Watch the fine phase labels: `accelerate`, `cruise_approach`, `decelerate`, `stop_hold`, and `release`.
8. Use **End collection** for an orderly stop, or **Hold** when motion must stop immediately.

The browser keeps the draft plan locally. The backend expands it deterministically in entry order and then variation order. See [`docs/stop-sign-collection.md`](docs/stop-sign-collection.md) for the plan JSON, limits, geometry, and collection checklist.

## Geometry contract

Every attempt has exactly four distinct poses:

```text
signPose
   └─ back by stopDistanceM ──────────> stopLinePose
          └─ back by egoCenterOffsetM ─> egoStopPose
                 └─ back by startDistanceM ─> startPose
```

- `signPose` is the registered world reference and travel heading.
- `stopLinePose` is where the vehicle's front must not cross before completing the stop.
- `egoStopPose` is the target for the vehicle center, accounting for front overhang and clearance.
- `startPose` is the deterministic reset pose for the approach.

Derived poses are validated again at the FiveM boundary; clients cannot send contradictory geometry. The defaults are a `3.0 m` sign-to-line offset, `2.5 m` line-to-ego-center offset, `40.0 m` approach, `8.0 m/s` target speed, and `5000 ms` dwell.

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

`metadata.json` and `run.jsonl` retain `stopSignGoal`, `stopSignOutcome`, sign/location identity, resolved variant, seed, attempt index, and synchronized telemetry. `dataset.jsonl` contains causal RGB windows, telemetry history/future, the fine phase, future targets, eligibility, and a location-level split group.

The current model consumes five causal RGB frames plus current-speed history. Stop-sign pose, stop-line distance, ego error, and other oracle geometry are labels/scoring metadata only; they are not perception inputs. The primary outputs are:

- `future_speed_mps`
- `stop_intent`

They are predicted at fixed horizons of `250, 500, 1000, 2000, 3000, 5000 ms`. A deterministic controller converts the nearest motion target into slew-limited, mutually exclusive throttle or brake, then the virtual controller applies it to GTA. Recorded expert throttle, expert brake, and physical brake pressure are auxiliary diagnostics—not the model-to-game interface.

See [`docs/stop-sign-model-control.md`](docs/stop-sign-model-control.md) for the complete data, training, controller, and safety boundary.

## Training from a clean slate

The checked-in [`fsd_trainer/train_config.toml`](fsd_trainer/train_config.toml) intentionally contains no run IDs or checkpoint. A fresh workspace showing zero runs and zero checkpoints is expected.

Training admits successful, complete `stop-sign-goal.v1` attempts by default, balances the five behavior phases, and prevents a physical stop-sign location from appearing in both training and validation. Collect at least two independent runs across distinct locations before expecting the workbench to report training ready.

Do not process or rebuild paths while a training job is actively reading them. The normal workflow is to finish collection, use **Data** to make processing current, then queue training from **Train**.

## Components

| Area | Responsibility |
|---|---|
| `fivem/` | Stop-sign calibration, derived geometry, deterministic expert, world/vehicle execution, telemetry, and trip metadata. |
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
