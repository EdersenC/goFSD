# Virtual controller actuator

The Go backend owns a persistent virtual Xbox 360 controller for model-driven inference. Expert parking collection does **not** use this path: FiveM's native parking task drives those demonstrations directly.

## Runtime boundary

- Live actuation is Windows-only; non-Windows builds expose an unsupported actuator state.
- Install ViGEmBus and run the backend with enough permission to create a virtual controller.
- Start the backend with `scripts/dev-backend.ps1` so the data root and working directory are consistent.
- Start `scripts/dev-model-server.ps1`, load a checkpoint from the Models/Advanced UI, and only then start inference.

The request flow is:

```text
FiveM telemetry + captured frames
             ↓
Go inference bridge → Python planner
             ↓
Go translation and actuator safety
             ↓
virtual Xbox controller → GTA/FiveM
```

FiveM publishes vehicle state, but it does not apply predicted steer/throttle/brake values itself.

## Parking safety boundary

The checked-in parking profile limits inferred motion to `8 km/h`. The actuator also:

- clamps and smooths steering, throttle, and brake;
- makes brake win if throttle and brake conflict;
- applies overspeed braking;
- releases or decays stale enabled drive commands;
- preserves reverse lockout by translating a real brake request below the lockout speed into a handbrake hold, then releasing that automatic hold on forward throttle; and
- exposes the received, resolved, and applied values at `GET /actuator/state`.

Inference starts only when FiveM reports a fresh, valid ego vehicle in the `ready` evaluation phase, a calibrated parking target, and a pose inside the forward curriculum start envelope: `-17.0..-9.5 m` longitudinal, `+/-2.75 m` lateral, and `+/-17 deg` heading error relative to the bay. The loaded checkpoint must enable the target-present and signed longitudinal/lateral/heading parking inputs and expose steering, acceleration, and brake heads. The session binds both the calibrated target pose and checkpoint identity so neither can be replaced while the car is moving.

Any prediction/actuator error, target loss, collision, reverse motion, off-ground state, tilt beyond `5 deg`, frame/telemetry skew beyond the configured limit, unexpected capture stream/process exit, or failure to settle within `45 s` latches the session. Every arm and terminal hold carries a monotonic command ID; the backend waits until that exact neutral or disabled-handbrake command is reported as physically applied. An apply fault or confirmation timeout is terminal and can never be reported as a parking success. Once confirmed, the safety handbrake remains held beyond the normal stale-command timeout and the inference capture process is canceled. `GET /inference/status` reports `active: true` until process cleanup finishes and then preserves the terminal `succeeded` or `error` result with `active: false`. A new Start is the only operation that rechecks every parking/model/actuator precondition and confirms neutral actuation before motion can resume; legacy cruising fallback decay is never applied in this parking-first path.

Safety holds and neutral Start arms are acknowledged only after the actuator reports that the exact submitted command ID reached the controller with the expected applied state. A queued command, an older matching state, or a controller apply error does not satisfy this gate; parking success becomes visible only after the hold is confirmed.

The parking start gate compares the checkpoint's exact ordered image, telemetry, future-offset, telemetry-feature, control-head, auxiliary-head, image-size, frame-window, and enabled state-input contracts. Temporal-horizon actuation currently fails closed at Start because its control timing is not yet derived from the checkpoint's future offsets; keep `temporal_horizon_actuator_enabled = false` for parking.

The reverse lockout is why the current milestone is forward-bay parking only. Adding reverse or parallel parking requires an explicit direction/gear output and an independently tested interlock; negative acceleration is not treated as reverse.

## Manual smoke test

Before collecting fresh demonstrations, perform a steering-sign acceptance check at walking speed: send `+0.20` steering, verify GTA turns right, and verify normalized `steering_actual` is positive. If either sign is wrong, stop and disable actuation; invert the sign exactly once in the FiveM telemetry normalization boundary, then repeat this check before collecting data.

With the Windows backend running, send a short normalized command:

```powershell
.\.venv\Scripts\python.exe fsd_trainer\src\gta_fsd\send_control.py `
  --config fsd_trainer\train_config.toml --steer 0.20 --throttle 0.20
```

Release every control explicitly:

```powershell
.\.venv\Scripts\python.exe fsd_trainer\src\gta_fsd\send_control.py `
  --config fsd_trainer\train_config.toml --disabled
```

Inspect state:

```powershell
Invoke-RestMethod http://127.0.0.1:8080/actuator/state
```

## Configuration

`fsd_trainer/train_config.toml` is the shared runtime configuration. The primary actuator fields are:

- `tick_hz`, `request_timeout`, and optional `stale_timeout`;
- `steering_gain`, `throttle_gain`, and `throttle_floor`;
- `speed_limit_kph`, overspeed margin/brake, and `model_brake_threshold`;
- `reverse_lockout_speed_kph`; and
- the optional temporal-horizon safety settings.

The Advanced UI can tune supported values live. Save deliberately: the checked-in TOML remains the reproducible baseline for the next run.
