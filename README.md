# GTA Stop Sign Lab

This repository is a temporal-driving lab for one focused V0 behavior: launch from rest, approach a stop sign, brake to the correct stopping point, hold for about five seconds, and leave under a scripted release policy.

The four operator milestones are:

1. **Launch** — apply throttle and establish forward motion.
2. **Approach + Brake** — cruise toward the sign, then decelerate without crossing the stop line.
3. **Stop + Dwell** — stop at the derived ego pose and remain stopped for the configured dwell, normally `5000 ms`.
4. **Go** — V0 releases with `scripted_dwell_release_v0`. The model does not decide when the dwell is finished yet.

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
| **Collect** | Search the 463-prop GTA catalog, navigate to signs, register calibrated lane poses, queue ordered attempts, and watch live telemetry. |
| **Data** | Process completed trips, inspect readiness and phase coverage, and keep failed attempts diagnosable but out of expert training by default. |
| **Train** | Train from independent stop-sign locations, with phase-balanced sampling and a fixed temporal output contract. |
| **Evaluate** | Load a compatible checkpoint, pass the guarded safety preflight, and compare predicted speed/stop intent with the closed-loop result. |

The red **Hold** control stays available at the bottom-right. `Alt+Shift+H` is the keyboard shortcut. Hold stops collection, vehicle motion, and inference, invalidates stale motion-start epochs, and waits for FiveM to confirm a fresh idle safety state.

## First stop-sign collection

1. Search the **Stop-sign catalog**, select a physical prop, choose **Stage in plan**, then **Set GTA waypoint**. Run `/tpwaypoint` in FiveM to travel there.
2. Select **Start setup car** and remain in its driver seat.
3. Place the setup car at the lane reference point with its heading aligned to the vehicle's intended travel direction. The catalog prop position and quaternion are navigation references, not a training pose.
4. Select **Calibrate current sign**, then copy the accepted live pose into the desired plan entry with **Use live pose**.
5. Set the stop-line distance, ego-center offset, approach distance, exit distance, target speed, dwell, attempts, and base environment.
6. Select **Load proof set** for the seeded clear baseline, short/slow, long/brisk, and rainy daytime variants, or add your own. Blank variation fields inherit the entry's base values.
7. Add more physical stop-sign entries as needed, review the total signs/variations/attempts, then select **Queue collection**.
8. Watch the fine phase labels: `accelerate`, `cruise_approach`, `decelerate`, `stop_hold`, and `release`.
9. Use **End collection** for an orderly stop, or **Hold** when motion must stop immediately.

The browser keeps the draft plan locally. The backend expands it deterministically in entry order and then variation order. See [`docs/stop-sign-collection.md`](docs/stop-sign-collection.md) for the plan JSON, limits, geometry, and collection checklist.

Catalog staging copies only the stable sign ID. Roadside prop coordinates and quaternions are never copied into the lane target, and an origin-placeholder pose is rejected by the UI, backend, and FiveM until a live pose is applied.

## Geometry contract

Every attempt has exactly five distinct poses:

```text
signPose
   ├─ forward by exitDistanceM ──────> exitPose
   └─ back by stopDistanceM ──────────> stopLinePose
          └─ back by egoCenterOffsetM ─> egoStopPose
                 └─ back by startDistanceM ─> startPose
```

- `signPose` is the registered world reference and travel heading.
- `stopLinePose` is where the vehicle's front must not cross before completing the stop.
- `egoStopPose` is the target for the vehicle center, accounting for front overhang and clearance.
- `startPose` is the deterministic reset pose for the approach.
- `exitPose` is the exact scripted-release waypoint beyond the sign.

Derived poses are validated again at the FiveM boundary; clients cannot send contradictory geometry. The workbench defaults are the proven `4.0 m` sign-to-line offset, `2.5 m` line-to-ego-center offset, `35.0 m` approach, `8.0 m` sign-to-exit distance, `5.0 m/s` target speed, and `5000 ms` dwell.

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

Training admits successful, complete `stop-sign-goal.v1` attempts by default, balances the observed anchor phases, and prevents a physical stop-sign location from appearing in both training and validation. The scripted release remains present in future phase/speed labels even when the five-second future horizon leaves no release anchor rows. Collect successful clips at two distinct physical signs before expecting the workbench to report training ready.

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
