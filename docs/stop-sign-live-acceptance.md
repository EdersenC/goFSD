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
- [ ] Live sign calibration returns the expected world pose and travel heading.
- [ ] The derived stop line, ego stop pose, and start pose are on the correct side of the sign and aligned with the lane.
- [ ] A queued attempt resets to the exact derived start pose before capture.
- [ ] The selected monitor contains GTA, and the captured RGB clip shows the intended approach.
- [ ] Nearby traffic/NPC isolation behaves as intended without deleting the controlled vehicle.

## Behavior acceptance

Collect at least one clean success and deliberate failures that exercise the safety labels.

- [ ] `accelerate`: the car launches cleanly from rest.
- [ ] `cruise_approach`: it establishes a stable approach speed.
- [ ] `decelerate`: it brakes smoothly without throttle/brake overlap or visible oscillation.
- [ ] `stop_hold`: the front remains behind the stop line, the vehicle center reaches the ego stop tolerance, and speed remains near zero for the full dwell.
- [ ] `release`: the car leaves only after the scripted V0 dwell completes.
- [ ] Crossing without stopping, stopping too early, collision, timeout, and operator stop produce inspectable failed outcomes.

## Data acceptance

For a completed attempt, verify:

```text
<FSD_DATA_ROOT>/runs/<run-id>/stop-sign_temporal-v1/run.jsonl
<FSD_DATA_ROOT>/runs/<run-id>/stop-sign_temporal-v1/trip-000/video.mkv
<FSD_DATA_ROOT>/runs/<run-id>/stop-sign_temporal-v1/trip-000/video.log
<FSD_DATA_ROOT>/runs/<run-id>/stop-sign_temporal-v1/trip-000/metadata.json
<FSD_DATA_ROOT>/runs/<run-id>/stop-sign_temporal-v1/trip-000/processing.json
<FSD_DATA_ROOT>/runs/<run-id>/stop-sign_temporal-v1/trip-000/dataset.jsonl
<FSD_DATA_ROOT>/runs/<run-id>/stop-sign_temporal-v1/trip-000/frames/
```

- [ ] RGB and 50 ms telemetry align at the capture flash.
- [ ] The five fine phases appear in the expected temporal order.
- [ ] Current speed, expert desired speed, expert throttle/brake, physical brake pressure, and outcome are finite and plausible.
- [ ] `stopSignGoal` contains all four poses, the resolved variation, dwell, seed, and `scripted_dwell_release_v0`.
- [ ] A successful trip is training eligible; a failed trip is retained but excluded by default.
- [ ] Trips from one physical sign share a location split group; distinct signs have distinct groups.
- [ ] The model input excludes oracle sign/line/error geometry.

## Training and closed-loop acceptance

- [ ] Training and validation use distinct physical stop-sign locations.
- [ ] Every fine phase has usable samples; phase-balanced sampling is active.
- [ ] The checkpoint declares `temporal_stop_sign_v1`, `stop_sign_motion_plan_v1`, the six horizon timings, and `scripted_dwell_release_v0`.
- [ ] Offline inference returns finite `future_speed_mps` and `stop_intent` sequences.
- [ ] The live controller calibration matches the tested vehicle, GTA build, and adapter.
- [ ] The deterministic controller never commands throttle and brake together.
- [ ] Stale telemetry, stale plans, incompatible checkpoints, and calibration mismatch fail closed.
- [ ] A guarded live evaluation completes the four user milestones without crossing the line early or releasing before dwell.

Until every applicable live item is checked, report the work as automated-contract complete with live acceptance pending—not as proven in GTA.
