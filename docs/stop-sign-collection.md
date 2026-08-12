# Stop-sign scene collection

The Collect area at `http://127.0.0.1:8080/` is the primary data-collection surface. It turns each physical stop sign into a reusable scene defined by three operator-captured poses: **Start**, **Stop**, and **End**. The browser saves the scene library locally and queues deterministic batches through `POST /control/stop-sign-batches`.

## Operator checklist

1. Start the backend, deploy the current FiveM resource, run `restart FSD` in the server console, and join the session.
2. Confirm **API** is online, **FiveM** is linked, and **Control** is synchronized. Use **Hold** once and verify it settles before starting a batch.
3. Search the complete 463-sign catalog and press **Teleport**. This only previews the location; browsing never changes the scene library.
4. Inspect the location and press **Use this stop sign** when it is suitable.
5. Drive at least `16 m` before Stop, aligned with the approach lane, and press **Capture Start**. This is a direction/cruise-buffer anchor; generated jobs derive their exact reset pose from speed.
6. Drive to the exact vehicle-center stopping point and press **Capture Stop**.
7. Drive beyond the sign to the desired continuation point and press **Capture End**.
8. Review the automatic variant count and motion bound. Defaults are `50` variants and `20%`; the motion bound cannot exceed `50%`.
9. Select **Collect this scene**. FiveM records one uninterrupted attempt per generated variant.
10. Choose the next catalog sign and repeat the Preview → Use → Start → Stop → End workflow. Completed calibrations appear under **Saved signs**; open one to restore all three positions by catalog ID.
11. Watch the phase and telemetry. Use **End collection** for orderly cancellation, or **Hold** for immediate safety intervention.
12. In **Data**, process completed trips and inspect logical stage windows, speed, expert throttle/brake, physical brake pressure, and outcome.

Drafts and completed calibrations survive a browser refresh. Each completed scene is keyed by its stable `catalogId`, so selecting the same sign reopens and updates its existing Start, Stop, and End instead of creating a duplicate. Deleting a scene from the Scene builder intentionally removes that saved calibration. The saved plan contains no `safetyEpoch`; the UI adds the current epoch immediately before queueing. A missing epoch returns `428`. A Hold or FiveM reconnect invalidates an old epoch, and the backend returns `409` without starting motion.

## Scene geometry

The catalog stores roadside prop positions for identification and one-click navigation. These positions are never model inputs and do not replace the operator's demonstrated route.

The operator captures:

| Pose | Meaning |
|---|---|
| `startPose` | Captured approach direction and optional extra cruise buffer; expansion derives each job's exact reset pose. |
| `egoStopPose` | Exact vehicle-center pose and heading for the end of Brake + Stop. |
| `exitPose` | Exact destination pose and heading where the attempt ends. |

Captured Start must be at least `16 m` before Stop and stay in the approach-lane corridor. Expansion derives each exact Start from deterministic acceleration distance, `1.25 s` of stable cruise, braking distance, and any extra buffer represented by the captured anchor. Automatic target speed uses seeded-random `22.4–33.6 mph` buckets. End must be at least `8 m` beyond Stop. Invalid or incomplete scenes cannot be queued.

At replay launch, the expert follows the saved Start heading first and blends toward the Stop heading over the opening approach. Steering is distance-bounded and rate-limited, so a curved Start-to-Stop capture cannot request full wheel lock on the first frame. The stop target and phase labels remain unchanged.

The saved-scene collection queue can contain every ready sign or any ordered subset. The backend expands the queue in sign order and then seeded-variant order, while FiveM teleports to each saved Start automatically. The queue persists in the browser so an unattended overnight collection can be prepared once and started as one safety-fenced batch.

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
      "targetSpeedMps": 10.0
    }
  ]
}
```

Top-level `id` and `seed` are required. A scene library allows 1–100 signs and at most 5,000 expanded jobs. Each captured scene requires catalog identity, catalog navigation position, Start, Stop, End, and an automatic-variation specification.

Expansion is deterministic. Each generated job uses seed `<plan-seed>:<entry-id>:auto-NNN`. `auto-001` preserves baseline conditions and uses the `10 m/s` endpoint; its Start is still speed-coupled. Remaining jobs vary:

- Target speed selected from `22.4`, `24.4`, `26.4`, `28.4`, `30.4`, `32.4`, or `33.6 mph`, plus Stop-to-End distance within `motionVariancePct`. The captured Start supplies lane direction and any extra cruise buffer. Target speed and the bounded braking profile are generated first; Start is then moved using that run's exact acceleration/cruise/braking profile, even when the captured anchor itself is closer than the generated distance.
- Small behavior changes that scale with `motionVariancePct`: braking can become only slightly harder (`3.80–4.05 m/s²` at the 50% maximum), while release acceleration can become moderately faster (`3.50–4.20 m/s²`). The generator never creates softer braking or a slower-than-standard release, limiting contradictory imitation targets. The changed desired-speed trajectory drives the physical replay; expert throttle/brake remain derived diagnostics.
- Start/End lateral position and heading within small bounded tolerances.
- Stop by only a small centimeter-scale longitudinal/lateral tolerance and a small heading tolerance.
- Weather, time of day, and vehicle color broadly across their supported pools.
- A visual route waypoint `120–300 m` beyond End with a seeded `-75–75 m` lateral offset. This marker does not move End or change the controller's completion target; its resolved pose and offsets are stored in `stopSignGoal`.

The default is `50` variants with `20%` motion variance. The accepted range is `0–50%`. Speed is independent of that percentage: the first run uses the `10 m/s` (`22.4 mph` displayed) baseline and every later run independently chooses a deterministic random speed bucket. Start is coupled to the generated speed and braking deceleration, so a target is never silently reduced to fit an independently generated Start. Each run combines its independently seeded speed/Start, End distance, lane offsets, headings, Stop jitter, weather, time, vehicle color, braking deceleration, and release acceleration; it is not assigned only one changed axis. Changing the seed intentionally creates a different deterministic set; restoring the old seed reproduces the old set. Manual variation entry is not part of the normal operator workflow.

`auto-001` is the exact scene baseline. Every resolved job carries `variationProfile` contract `stop-sign-variation-profile.v2`, containing exact physical/categorical/behavior deltas from that baseline, a canonical `changedDimensions` list, `changeCount`, and `combinationMagnitudePct`. The percentage is the RMS magnitude across 15 fixed-range normalized dimensions: target speed, braking deceleration, release acceleration, Start distance, End distance, Stop position/heading, Start lane/heading, End lane/heading, weather, time, vehicle model, and vehicle color. It is coverage/audit metadata only and must never be passed to the RGB policy. Because normalization uses the system's fixed maximum ranges rather than the selected slider bound, increasing Motion variance increases the actual deltas and their contribution to the score.

## Temporal guarantees

- Jobs and attempts never overlap.
- Every variant produces one physical recording with three logical stages in order: `approach`, `brake_stop`, and `release`.
- The raw recording begins at launch, but the training index begins only after one full second at stable cruise (within 2% of the peak approach command). Launch-heavy frames remain available for inspection and are not selected for training.
- Capture and synchronization happen once at Start. FiveM emits exact transition timestamps and a phase on every 50 ms telemetry row, then capture finalizes once at End.
- The vehicle, camera, controller history, and recording stay continuous through braking, the brief zero-speed confirmation, and release.
- Long stationary dwell footage is not collected.
- Each sample's `clip_stage` comes from the latest telemetry phase at or before its RGB anchor. Causal histories never borrow a future phase; future targets intentionally cross stage boundaries so transition motion is preserved.
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
