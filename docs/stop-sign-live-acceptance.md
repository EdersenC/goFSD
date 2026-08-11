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
- [ ] Pressing **Teleport** on a catalog row moves the managed setup car near that physical sign and creates or reopens its saved scene.
- [ ] **Capture Start**, **Capture Stop**, and **Capture End** store the setup car's exact world pose and heading.
- [ ] An incomplete scene cannot be queued; invalid Start → Stop → End ordering is rejected.
- [ ] A queued variant resets to its resolved Start pose before the Approach clip.
- [ ] The selected monitor contains GTA, and the captured RGB clip shows the intended approach.
- [ ] Nearby traffic/NPC isolation behaves as intended without deleting the controlled vehicle.

## Behavior acceptance

Collect at least one clean three-clip variant and deliberate failures that exercise the safety labels.

- [ ] `approach`: a flash begins the clip, the car launches cleanly, establishes a stable approach speed, and the clip ends at the braking boundary.
- [ ] `brake_stop`: a new flash begins the clip, the car brakes smoothly without throttle/brake overlap, reaches Stop, briefly confirms zero speed, and capture ends without a long dwell.
- [ ] `release`: a new flash begins the clip from Stop and the car reaches End.
- [ ] The three videos are independent; no video spans two stage boundaries.
- [ ] The default scene expands to 50 deterministic variants, repeating the same seed reproduces them, and changing the seed changes them.
- [ ] Route distance and speed stay within the selected bound (default `20%`, maximum `25%`), while weather/time/color cover broader conditions.
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
- [ ] `stopSignGoal.clipStage` is exactly `approach`, `brake_stop`, or `release`, and three consecutive trip folders represent one complete variant.
- [ ] Fine phases are valid for the declared clip stage and never leak across clips.
- [ ] Current speed, expert desired speed, expert throttle/brake, physical brake pressure, and outcome are finite and plausible.
- [ ] `stopSignGoal` contains the captured/resolved Start, Stop, and End geometry, catalog ID, clip stage, variant ID, and seed.
- [ ] A successful trip is training eligible; a failed trip is retained but excluded by default.
- [ ] Trips from one physical sign share a location split group; distinct signs have distinct groups.
- [ ] The model input excludes oracle sign/line/error geometry.

## Training and closed-loop acceptance

- [ ] Training and validation use distinct physical stop-sign locations.
- [ ] Every clip stage has usable samples and `(clip stage, fine phase)` balancing is active.
- [ ] No temporal history or future target crosses a clip boundary.
- [ ] The checkpoint declares `temporal_stop_sign_v1`, `stop_sign_motion_plan_v1`, and horizons `100, 250, 500, 1000 ms`.
- [ ] Offline inference returns finite `future_speed_mps` and `stop_intent` sequences.
- [ ] The live controller calibration matches the tested vehicle, GTA build, and adapter.
- [ ] The deterministic controller never commands throttle and brake together.
- [ ] Stale telemetry, stale plans, incompatible checkpoints, and calibration mismatch fail closed.
- [ ] A guarded live evaluation completes Approach, Brake + Stop, and Release without crossing the line early.

Until every applicable live item is checked, report the work as automated-contract complete with live acceptance pending—not as proven in GTA.
