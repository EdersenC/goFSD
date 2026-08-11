# Stop-sign scene collection

The Collect area at `http://127.0.0.1:8080/` is the primary data-collection surface. It turns each physical stop sign into a reusable scene defined by three operator-captured poses: **Start**, **Stop**, and **End**. The browser saves the scene library locally and queues deterministic batches through `POST /control/stop-sign-batches`.

## Operator checklist

1. Start the backend, deploy the current FiveM resource, run `restart FSD` in the server console, and join the session.
2. Confirm **API** is online, **FiveM** is linked, and **Control** is synchronized. Use **Hold** once and verify it settles before starting a batch.
3. Search the complete 463-sign catalog and press **Teleport**. The managed setup car is moved to a nearby drivable lane and the sign is added to the scene library.
4. Drive to the exact beginning of the desired demonstration and press **Capture Start**.
5. Drive to the exact vehicle-center stopping point and press **Capture Stop**.
6. Drive beyond the sign to the desired continuation point and press **Capture End**.
7. Review the automatic variant count and motion bound. Defaults are `50` variants and `20%`; the motion bound cannot exceed `25%`.
8. Select **Collect this scene**. FiveM runs one generated variant at a time and records three separate stage clips for it.
9. Choose the next catalog sign and repeat the Teleport → Start → Stop → End workflow.
10. Watch the clip stage and telemetry. Use **End collection** for orderly cancellation, or **Hold** for immediate safety intervention.
11. In **Data**, process completed trips and inspect each stage clip, speed, expert throttle/brake, physical brake pressure, and outcome.

The draft survives a browser refresh. The saved plan contains no `safetyEpoch`; the UI adds the current epoch immediately before queueing. A missing epoch returns `428`. A Hold or FiveM reconnect invalidates an old epoch, and the backend returns `409` without starting motion.

## Scene geometry

The catalog stores roadside prop positions for identification and one-click navigation. These positions are never model inputs and do not replace the operator's demonstrated route.

The operator captures:

| Pose | Meaning |
|---|---|
| `startPose` | Exact reset pose and heading for the Approach clip. |
| `egoStopPose` | Exact vehicle-center pose and heading for the end of Brake + Stop. |
| `exitPose` | Exact destination pose and heading for the Release clip. |

Start must be at least `5 m` before Stop, End must be at least `2 m` beyond Stop, and both must remain within the accepted approach-lane corridor. Invalid or incomplete scenes cannot be queued.

## Plan contract

```json
{
  "id": "city-stops-v0",
  "seed": "fresh-stop-data-1",
  "entries": [
    {
      "id": "mission-row-01",
      "catalogId": "gta-v-sign-0023",
      "catalogPosition": {"x": -1830.77, "y": 3206.39, "z": 31.85},
      "startPose": {"x": -1810.0, "y": 3180.0, "z": 32.0, "heading": 320.0},
      "egoStopPose": {"x": -1827.0, "y": 3201.0, "z": 32.0, "heading": 320.0},
      "exitPose": {"x": -1836.0, "y": 3212.0, "z": 32.0, "heading": 320.0},
      "autoVariations": {"count": 50, "motionVariancePct": 20},
      "targetSpeedMps": 5.0
    }
  ]
}
```

Top-level `id` and `seed` are required. A scene library allows 1–100 signs and at most 5,000 expanded jobs. Each captured scene requires catalog identity, catalog navigation position, Start, Stop, End, and an automatic-variation specification.

Expansion is deterministic. Each generated job uses seed `<plan-seed>:<entry-id>:auto-NNN`. `auto-001` is the exact captured baseline. Remaining jobs vary:

- Start-to-Stop distance, Stop-to-End distance, and target speed within `motionVariancePct`.
- Start/End lateral position and heading within small bounded tolerances.
- Stop by only a small centimeter-scale longitudinal/lateral tolerance and a small heading tolerance.
- Weather, time of day, and vehicle color broadly across their supported pools.

The default is `50` variants with `20%` motion variance. The accepted motion range is `0–25%`. Changing the seed intentionally creates a different deterministic set; restoring the old seed reproduces the old set. Manual variation entry is not part of the normal operator workflow.

## Temporal guarantees

- Jobs, attempts, and stage captures never overlap.
- Every variant produces exactly three clips in order: `approach`, `brake_stop`, and `release`.
- Each clip starts its own RGB capture, emits its own synchronization flash, records its own telemetry, and finalizes before the next clip begins.
- Approach ends at the braking-stage boundary. Brake + Stop ends immediately after the car reaches the Stop pose and completes a brief zero-speed confirmation. Release begins from Stop and ends at End.
- Long stationary dwell footage is not collected.
- Temporal histories and future targets are stage-locked; samples that cannot satisfy the complete window inside one clip are excluded rather than borrowing frames or labels from another stage.
- Model target horizons are `100`, `250`, `500`, and `1000 ms`.
- Fine behavior phases remain `accelerate`, `cruise_approach`, `decelerate`, `stop_hold`, and `release`, while `clipStage` identifies the independent clip boundary.
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

Each generated variant writes three consecutive trip folders, one per clip stage. `stopSignGoal.clipStage` identifies `approach`, `brake_stop`, or `release` in metadata and processed samples. Capture writes the raw video, log, trip metadata, and scene `run.jsonl`; processing extracts RGB frames and publishes `dataset.jsonl` plus `processing.json`. A temporary processing lock/workspace may appear while processing is active; do not edit or rebuild that trip concurrently.
