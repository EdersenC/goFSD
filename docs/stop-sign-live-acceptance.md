# Stop Sign Lab live acceptance

Automated tests validate contracts and deterministic logic. They do not prove the Windows capture stack, the selected display, FiveM deployment, a vehicle-specific controller calibration, or closed-loop GTA behavior.

## Start and deploy

From the repository root:

```bash
npm run setup
npm start
```

In another terminal, deploy the resource:

```bash
export FIVEM_RESOURCE_DIR=/absolute/path/to/cfx-server-data/resources/FSD
npm --prefix fivem run deploy:check
npm --prefix fivem run build:deploy
```

In the FiveM server console:

```text
restart FSD
```

Join the session, open [http://127.0.0.1:8080/](http://127.0.0.1:8080/), and keep a manual brake/stop path available throughout the test.

## Runtime acceptance

- [ ] API, FiveM, capture source, and model status match the processes actually running.
- [ ] **Hold** or `Alt+Shift+H` stops motion and settles to a fresh idle safety epoch.
- [ ] A stale pre-Hold start is rejected rather than executed later.
- [ ] The setup command puts the player in the driver seat and preserves the controlled car.
- [ ] The catalog loads all 463 verified main-map stop signs and search filters the complete list.
- [ ] Pressing **Teleport** moves the setup car but does not add a scene; **Use this stop sign** is the only catalog action that adds or opens one.
- [ ] **Capture Start**, **Capture Stop**, and **Capture End** store the setup car's exact world pose and heading.
- [ ] An incomplete scene cannot be queued; invalid Start → Stop → End ordering is rejected.
- [ ] A queued variant resets to its resolved Start pose once before the continuous attempt.
- [ ] The selected monitor contains GTA, and the captured RGB recording shows the complete motion through End.
- [ ] Nearby traffic/NPC isolation behaves as intended without deleting the controlled vehicle.

## Behavior acceptance

Collect at least one clean continuous variant and deliberate failures that exercise the safety labels.

- [ ] One flash begins the attempt at Start; there are no capture restarts or teleports before End.
- [ ] `approach`: the car launches cleanly and establishes a stable approach speed.
- [ ] `brake_stop`: the car transitions smoothly into braking, reaches Stop, and briefly confirms zero speed without a long dwell.
- [ ] `release`: the same vehicle and recording continue smoothly from Stop to End.
- [ ] The one video spans both stage boundaries without a visual or controller discontinuity.
- [ ] The default scene expands to 50 deterministic variants, repeating the same seed reproduces them, and changing the seed changes them.
- [ ] Route distance and speed stay within the selected bound (default `20%`, maximum `25%`), while weather/time/color cover broader conditions.
- [ ] Crossing without stopping, stopping too early, collision, timeout, and operator stop produce inspectable failed outcomes.

## Data acceptance

For a completed attempt, verify:

```text
<FSD_DATA_ROOT>/runs/<run-id>/stop-sign_continuous-v2/run.jsonl
<FSD_DATA_ROOT>/runs/<run-id>/stop-sign_continuous-v2/trip-000/video.mkv
<FSD_DATA_ROOT>/runs/<run-id>/stop-sign_continuous-v2/trip-000/video.log
<FSD_DATA_ROOT>/runs/<run-id>/stop-sign_continuous-v2/trip-000/metadata.json
<FSD_DATA_ROOT>/runs/<run-id>/stop-sign_continuous-v2/trip-000/processing.json
<FSD_DATA_ROOT>/runs/<run-id>/stop-sign_continuous-v2/trip-000/dataset.jsonl
<FSD_DATA_ROOT>/runs/<run-id>/stop-sign_continuous-v2/trip-000/frames/
```

- [ ] RGB and 50 ms telemetry align at the capture flash.
- [ ] `stopSignGoal.captureMode` is `continuous`, the three logical stages are declared, and `stageTransitions` contains ordered game-time boundaries.
- [ ] Each dataset row's `clip_stage` matches its anchor phase; temporal history and future labels can cross a boundary.
- [ ] Current speed, expert desired speed, expert throttle/brake, physical brake pressure, and outcome are finite and plausible.
- [ ] `stopSignGoal` contains the captured/resolved Start, Stop, and End geometry, catalog ID, continuous capture contract, variant ID, and seed.
- [ ] A successful trip is training eligible; a failed trip is retained but excluded by default.
- [ ] Trips from one physical sign share a location split group; distinct signs have distinct groups.
- [ ] The model input excludes oracle sign/line/error geometry.

## Training and closed-loop acceptance

- [ ] Training and validation use distinct physical stop-sign locations.
- [ ] Every logical stage has usable samples and `(logical stage, fine phase)` balancing is active.
- [ ] Boundary samples preserve real pre-transition history and post-transition targets from the same recording.
- [ ] The checkpoint declares `temporal_stop_sign_v1`, `stop_sign_motion_plan_v1`, and horizons `100, 250, 500, 1000 ms`.
- [ ] Offline inference returns finite `future_speed_mps` and `stop_intent` sequences.
- [ ] The live controller calibration matches the tested vehicle, GTA build, and adapter.
- [ ] The deterministic controller never commands throttle and brake together.
- [ ] Stale telemetry, stale plans, incompatible checkpoints, and calibration mismatch fail closed.
- [ ] A guarded live evaluation completes Approach, Brake + Stop, and Release without crossing the line early.

Until every applicable live item is checked, report the work as automated-contract complete with live acceptance pending—not as proven in GTA.
