# Virtual controller and parking setpoint runtime

The Go backend owns a persistent virtual Xbox 360 controller for model evaluation. Expert demonstration collection does not use it: FiveM's native parking task produces the successful forward-bay examples.

## Runtime boundary

- Live actuation is Windows-only and requires ViGEmBus.
- The backend and model server must use the same `fsd_trainer/train_config.toml` and data root.
- Load a `temporal_telemetry_gru_v2` / `parking_setpoint_v1` checkpoint before starting inference.
- Keep the backend bound to loopback; its control endpoints are intentionally local and unauthenticated.

```text
FiveM wheel/speed telemetry + captured frames
                    ↓
          Python parking planner
                    ↓
      versioned physical setpoint plan
                    ↓
 calibrated Go feedback controller + safety
                    ↓
       virtual Xbox controller → GTA
```

The Python model never outputs trigger values. It outputs desired normalized physical wheel steer, desired speed in meters per second, and stop probability at the exact future times declared by the checkpoint.

## Arming and ownership

Inference starts only when all of these agree:

- FiveM ego telemetry is fresh and valid;
- a forward-bay goal is calibrated and the car is inside the curriculum start envelope;
- the loaded checkpoint exposes the exact parking contract, timing, tensors, and enabled state inputs;
- the virtual controller reports ready with no apply fault;
- the parking controller has a verified calibration profile; and
- the current `vehicleModelHash` equals the calibrated hash.

The session first claims parking ownership. Before the first plan, the actuator holds the car with service brake or low-speed handbrake and waits for that exact ownership command ID to be applied. Once armed, parking inference may submit only `parking_setpoint_v1` plans or request the actuator-owned safety stop. Other enabled commands are rejected; an explicit disabled manual safety command can preempt the session.

Success, manual stop, target loss, collision, reverse motion, invalid pose, stale telemetry/plan, model error, controller error, or capture-process failure requests a persistent terminal stop from the actuator. The actuator uses service brake while moving and handbrake only after fresh telemetry confirms low speed. The backend waits for the exact stop command ID and a safe applied brake/hold state before reporting confirmation.

## Feedback and fail-safe behavior

At the actuator tick, the backend samples the plan using its observation age plus configured actuation latency. Steering uses a calibrated monotonic feed-forward map plus bounded PI correction from measured wheel steer. Speed uses bounded PI control and one signed longitudinal effort, so throttle and brake are mutually exclusive.

Stop probability has hysteresis. While moving, a stop request uses service brake. Handbrake is used only when fresh telemetry confirms speed is below the hold threshold. If speed is unknown, the fail-safe chooses service braking rather than assuming the vehicle is stopped.

Every plan and telemetry sample has a freshness limit. Frame PTS, source telemetry time, backend receipt time, and source/receipt skew are checked independently. Wrong contract, planner version, direction, horizon timing, output bounds, timestamp echo, vehicle identity, or controller `dt` fails closed. `GET /actuator/state` shows the plan receipt, selected setpoint, measured state, P/I terms, requested effort, final applied controls, and fault.

## Calibration profile

The checked-in config is deliberately unverified. Obtain the current hash from `GET /control/state`, measure the selected vehicle, then fill `[backend.parking_controller]`:

```toml
[backend.parking_controller]
calibration_verified = true
calibration_profile_id = "vehicle-name-gamebuild-adapter-v1"
vehicle_model_hash = 123456789
game_build = "replace-with-tested-build"
adapter_version = "vgamepad-go-v1"
steering_convention = "positive_wheel_is_positive_xinput"
steering_profile = [
  { wheel_steer = -1.0, command = -1.0 },
  { wheel_steer =  0.0, command =  0.0 },
  { wheel_steer =  1.0, command =  1.0 },
]
```

Replace the identity points with measured monotonic points when the game response is nonlinear. The vehicle hash is checked live. `game_build` and `adapter_version` are required test provenance but are not currently sensed by FiveM telemetry, so the operator must verify them. Restart the backend after editing and confirm `parkingController.ready: true` at `GET /actuator/state`.

Before model evaluation, use the manual controller only for calibration:

```powershell
.\.venv\Scripts\python.exe fsd_trainer\src\gta_fsd\send_control.py `
  --config fsd_trainer\train_config.toml --steer 0.20 --throttle 0.12

.\.venv\Scripts\python.exe fsd_trainer\src\gta_fsd\send_control.py `
  --config fsd_trainer\train_config.toml --disabled
```

Verify small positive/negative steering, settled wheel response, low trigger response, braking, latency, and release. Keep the car at walking speed with room around it. If steering signs disagree, stop and correct the convention boundary once before collecting or evaluating data.

## Shared configuration

`fsd_trainer/train_config.toml` separates manual/device safety from the model controller:

- `[backend.actuator]`: controller tick, request timeout, manual calibration gains, and final speed/overspeed safety.
- `[backend.parking_controller]`: calibration identity, steering map, PI gains, effort/slew bounds, stop hysteresis, freshness, latency, and expected horizon timing.
- `[backend.inference]`: exact planner/control contract, capture cadence, model server, and frame/telemetry alignment limits.
- `[dataset]`: the 50 ms sample interval and future offsets from which `50..300 ms` control timing is derived.

Manual actuator tuning does not alter model setpoints or the parking PI controller. It remains available only for deliberate calibration and direct manual commands.

## Live acceptance

Unit and replay checks are not a live acceptance test. On the actual Windows/FiveM stack, verify:

1. steering sign, monotonicity, center, and saturation;
2. service-brake versus low-speed handbrake transition;
3. no simultaneous applied throttle and brake;
4. stale plan, stale telemetry, wrong vehicle, and controller apply failures;
5. one successful and one deliberately failed evaluation with trace receipts; and
6. comparable setpoint tracking at the supported inference cadence.

The current contract is forward-only. Reverse or parallel parking requires explicit direction/gear supervision and a separate safety interlock.
