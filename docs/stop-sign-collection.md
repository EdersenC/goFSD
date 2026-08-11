# Stop-sign scene collection

The Collect area at `http://127.0.0.1:8080/` is the primary data-collection surface. It turns each physical stop sign into a reusable scene defined by three operator-captured poses: **Start**, **Stop**, and **End**. The browser saves the scene library locally and queues deterministic batches through `POST /control/stop-sign-batches`.

## Operator checklist

1. Start the backend, deploy the current FiveM resource, run `restart FSD` in the server console, and join the session.
2. Confirm **API** is online, **FiveM** is linked, and **Control** is synchronized. Use **Hold** once and verify it settles before starting a batch.
3. Search the complete 463-sign catalog and press **Teleport**. This only previews the location; browsing never changes the scene library.
4. Inspect the location and press **Use this stop sign** when it is suitable.
5. Drive to the exact beginning of the desired demonstration and press **Capture Start**.
6. Drive to the exact vehicle-center stopping point and press **Capture Stop**.
7. Drive beyond the sign to the desired continuation point and press **Capture End**.
8. Review the automatic variant count and motion bound. Defaults are `50` variants and `20%`; the motion bound cannot exceed `25%`.
9. Select **Collect this scene**. FiveM records one uninterrupted attempt per generated variant.
10. Choose the next catalog sign and repeat the Preview → Use → Start → Stop → End workflow.
11. Watch the phase and telemetry. Use **End collection** for orderly cancellation, or **Hold** for immediate safety intervention.
12. In **Data**, process completed trips and inspect logical stage windows, speed, expert throttle/brake, physical brake pressure, and outcome.

The draft survives a browser refresh. The saved plan contains no `safetyEpoch`; the UI adds the current epoch immediately before queueing. A missing epoch returns `428`. A Hold or FiveM reconnect invalidates an old epoch, and the backend returns `409` without starting motion.

## Scene geometry

The catalog stores roadside prop positions for identification and one-click navigation. These positions are never model inputs and do not replace the operator's demonstrated route.

The operator captures:

| Pose | Meaning |
|---|---|
| `startPose` | Exact reset pose and heading for the continuous attempt. |
| `egoStopPose` | Exact vehicle-center pose and heading for the end of Brake + Stop. |
| `exitPose` | Exact destination pose and heading where the attempt ends. |

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

- Jobs and attempts never overlap.
- Every variant produces one physical recording with three logical stages in order: `approach`, `brake_stop`, and `release`.
- Capture and synchronization happen once at Start. FiveM emits exact transition timestamps and a phase on every 50 ms telemetry row, then capture finalizes once at End.
- The vehicle, camera, controller history, and recording stay continuous through braking, the brief zero-speed confirmation, and release.
- Long stationary dwell footage is not collected.
- Each sample's `clip_stage` comes from its anchor phase. Causal histories and future targets intentionally overlap stage boundaries so transition motion is preserved.
- Model target horizons are `100`, `250`, `500`, and `1000 ms`.
- Fine behavior phases remain `accelerate`, `cruise_approach`, `decelerate`, `stop_hold`, and `release`; processed `clip_stage` is a logical balancing label, not a physical recording boundary.
- Collision, timeout, early stop, stop-line crossing, invalid vehicle, and operator stop remain inspectable failures.
- Failed attempts are excluded from expert training by default.
- A location-level split key derived from `signPose` prevents train/validation leakage across frames or attempts from the same physical sign.

## Output layout

```text
<FSD_DATA_ROOT>/runs/<run-id>/stop-sign_continuous-v2/
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

Each generated variant writes one trip folder. `stopSignGoal` declares continuous capture and all logical stages; `stopSignOutcome.stageTransitions` records their synchronized game-time boundaries. Processing extracts RGB frames and publishes anchor-frame `clip_stage` labels in `dataset.jsonl`. A temporary processing lock/workspace may appear while processing is active; do not edit or rebuild that trip concurrently.
