# Virtual Controller Actuator

This project now actuates GTA/FiveM through a Go-owned virtual Xbox 360 controller.

## Requirements

- Windows only for live actuation. The actuator endpoint reports `501 Not Implemented` on non-Windows hosts.
- Install the ViGEmBus driver before running the backend on Windows.
- Run the Go backend as Administrator so `vgamepad-go` can create and drive the virtual controller.

## Data Flow

1. FiveM still runs the existing `startEgoControl` setup path and prepares the ego vehicle/session.
2. The Go backend captures frames and requests predictions from the Python model server.
3. The Python model server posts raw model outputs to `POST /actuator/command`, and the Go actuator normalizes them before applying controller-specific gains and limits.
4. The Go actuator stores the latest command and applies it to a persistent virtual Xbox 360 controller at the configured `backend.actuator.tick_hz` cadence. The checked-in training config uses `15Hz`.
5. GTA/FiveM sees normal controller input from the virtual gamepad.

FiveM no longer applies predicted steer/throttle/brake itself.

## Start

1. Start the Python model server:

```bash
cd fsd_trainer/src/gta_fsd
python server.py --config ../../train_config.toml
```

2. Start the Go backend on Windows as Administrator:

```bash
cd backend
go run ./cmd
```

3. Start FiveM as usual and use the existing ego-control flow:

- Control page button: `Start Ego Control`
- Or FiveM command: `startEgo`

That still spawns/prepares the ego vehicle and capture state, but it does not apply controls anymore.

## Test Fixed Commands

Send a manual command from Python:

```bash
cd fsd_trainer/src/gta_fsd
python send_control.py --config ../../train_config.toml --steer -0.20 --throttle 0.35
```

`send_control.py` explicitly marks those inputs as `normalized`; the actuator API otherwise assumes raw model-scale inputs by default.
For manual testing, `send_control.py` repeats non-disabled commands for about one second at `15Hz` by default so the command stays visible across the actuator stale timeout.

Release controls explicitly:

```bash
cd fsd_trainer/src/gta_fsd
python send_control.py --config ../../train_config.toml --disabled
```

You can also inspect backend state:

```bash
curl -s http://127.0.0.1:8080/actuator/state
```

## Safety Behavior

- Steering is clamped, deadzoned, scaled, and rate-limited before it is applied.
- Raw model steer/acceleration values are normalized first using `model_steer_scale` and `model_accel_scale`.
- Throttle and brake are ramp-limited.
- If both throttle and brake are present, brake wins and throttle is forced to zero.
- If a fresh command is not received within `backend.actuator.stale_timeout`, Go recenters steering and releases throttle/brake.
- `/actuator/state` reports the last received request, the resolved controller target, and the last controller state that was successfully applied.

## Config

`fsd_trainer/train_config.toml` now contains a `[backend.actuator]` section used by both Python and Go:

- `url`
- `request_timeout`
- `tick_hz`
- `stale_timeout`
- `steer_deadzone`
- `max_steer_scale`
- `steer_rate_per_second`
- `throttle_rate_per_second`
- `brake_rate_per_second`
- `model_steer_scale`
- `model_accel_scale`
