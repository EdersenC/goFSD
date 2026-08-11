# Stop-sign temporal collection

The Collect area at `http://127.0.0.1:8080/` is the primary data-collection surface. It stores the draft in this browser, accepts a live `signPose` from FiveM, and queues one deterministic batch through `POST /control/stop-sign-batches`.

## Operator checklist

1. Start the backend, deploy the current FiveM resource, run `restart FSD` in the server console, and join the session.
2. Confirm **API** is online, **FiveM** is linked, and **Control** is synchronized. Use **Hold** once and verify it settles before starting a batch.
3. Search the 463-location **Stop-sign catalog**, select a physical prop, and choose **Set GTA waypoint**. Run `/tpwaypoint` in FiveM to travel there.
4. Select **Start setup car** and remain in its driver seat.
5. Move the car to the lane reference point and align its heading to the direction the ego vehicle will travel through the intersection.
6. Select **Calibrate current sign**, then select **Use live pose** on the intended entry.
7. Review the five-pose geometry and base run settings. The setup car does not need to be moved to a start, stop, or exit pose; all are derived from the accepted sign pose.
8. Add condition variations and additional physical signs. Keep every entry and variation ID unique.
9. Review total signs, expanded variations, and attempts before queueing.
10. Select **Queue collection**. FiveM runs one job and one attempt at a time.
11. Watch the behavior phase and telemetry. Use **End collection** for orderly cancellation, or **Hold** for immediate safety intervention.
12. In **Data**, process completed trips and inspect the RGB clip, phase sequence, speed, expert throttle/brake, physical brake pressure, and outcome.

The draft survives a browser refresh. The saved plan contains no `safetyEpoch`; the UI adds the current epoch immediately before queueing. A missing epoch returns `428`. A Hold or FiveM reconnect invalidates an old epoch, and the backend returns `409` without starting motion.

## Geometry

`signPose.heading` is the GTA travel heading, not the physical sign prop's facing direction. With GTA's forward vector

The bundled CSV registry stores roadside prop positions and prop quaternions for navigation. Those values are never copied into `signPose` or model inputs. The operator establishes the lane-center reference and travel heading with the setup car before queueing.

```text
forward(heading) = (-sin(heading), cos(heading))
```

the backend derives:

```text
stopLinePose = signPose    - forward * stopDistanceM
egoStopPose  = stopLinePose - forward * egoCenterOffsetM
startPose    = egoStopPose  - forward * startDistanceM
exitPose     = signPose     + forward * exitDistanceM
```

This separation is deliberate:

| Pose | Meaning |
|---|---|
| `signPose` | Stable map/location identity and travel heading. |
| `stopLinePose` | Crossing boundary for the vehicle's front bumper. |
| `egoStopPose` | Target vehicle-center pose before the line. |
| `startPose` | Exact reset pose for the temporal approach. |
| `exitPose` | Exact waypoint for the scripted release beyond the sign. |

The expanded command contains all five poses plus their distances. FiveM recomputes the chain and rejects any contradiction, preventing UI/backend/controller geometry drift.

## Plan contract

```json
{
  "id": "city-stops-v0",
  "seed": "fresh-stop-data-1",
  "entries": [
    {
      "id": "mission-row-01",
      "signPose": {"x": 120.0, "y": -45.0, "z": 8.0, "heading": 90.0},
      "stopDistanceM": 3.0,
      "egoCenterOffsetM": 2.5,
      "startDistanceM": 40.0,
      "exitDistanceM": 8.0,
      "targetSpeedMps": 8.0,
      "dwellMs": 5000,
      "attemptCount": 5,
      "weather": "EXTRASUNNY",
      "time": {"hour": 12, "minute": 0},
      "vehicle": {"model": "blista", "color": {"r": 230, "g": 60, "b": 45}},
      "variations": [
        {"id": "base"},
        {
          "id": "rain-blue-slower",
          "weather": "RAIN",
          "targetSpeedMps": 6.0,
          "vehicle": {"color": {"r": 40, "g": 80, "b": 220}}
        }
      ]
    }
  ]
}
```

Top-level `id` and `seed` are required. A plan allows 1–100 entries, at most 100 expanded jobs, and 1–50 attempts per job. Empty `variations` expands to one implicit `base` job. Expansion is stable: entry order first, then variation order. Each job seed is `<plan-seed>:<entry-id>:<variation-id>`.

Base defaults are:

| Setting | Default | Accepted range |
|---|---:|---:|
| `stopDistanceM` | `3.0` | `0.5–15.0 m` |
| `egoCenterOffsetM` | `2.5` | `0.5–8.0 m` |
| `startDistanceM` | `40.0` | `5.0–250.0 m` |
| `exitDistanceM` | `8.0` | `2.0–50.0 m` |
| `targetSpeedMps` | `8.0` | `0.5–8.0 m/s` |
| `dwellMs` | `5000` | `500–30000 ms` |
| `attemptCount` | `1` | `1–50` |
| `weather` | `EXTRASUNNY` | Supported GTA weather name |
| `time` | `12:00` | `00:00–23:59` |

Variations may override stop distance, ego offset, start distance, exit distance, target speed, dwell, attempts, weather, time, vehicle model, and vehicle RGB color. Omitted fields inherit from the entry. Geometry remains sign-relative; variations do not provide arbitrary derived poses.

## Temporal guarantees

- Jobs and attempts never overlap.
- Every attempt resets to the resolved `startPose` before capture.
- RGB video and 50 ms telemetry synchronize around the capture flash.
- The behavior sequence uses `accelerate`, `cruise_approach`, `decelerate`, `stop_hold`, and `release`.
- The normal dwell is five seconds. V0 release after the dwell is scripted and labeled honestly.
- Collision, timeout, early stop, stop-line crossing, invalid vehicle, and operator stop remain inspectable failures.
- Failed attempts are excluded from expert training by default.
- A location-level split key derived from `signPose` prevents train/validation leakage across frames or attempts from the same physical sign.

## Output layout

```text
<FSD_DATA_ROOT>/runs/<run-id>/stop-sign_temporal-v1/
├── run.jsonl
├── trip-000/
│   ├── video.mkv
│   ├── video.log
│   ├── metadata.json
│   ├── processing.json
│   ├── dataset.jsonl
│   └── frames/
└── trip-001/
```

Capture writes the raw video, log, trip metadata, and scene `run.jsonl`. Processing extracts RGB frames and publishes `dataset.jsonl` plus `processing.json`. A temporary processing lock/workspace may appear while processing is active; do not edit or rebuild that trip concurrently.
