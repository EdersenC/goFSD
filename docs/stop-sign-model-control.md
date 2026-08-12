# Stop-sign model and control contract

The active contract is `temporal_stop_sign_v1` with model output contract `stop_sign_motion_plan_v1`. It deliberately separates visual understanding, motion planning, deterministic actuation, and GTA execution.

```text
causal RGB + speed history
          │
          ▼
 temporal planner
          │  future_speed_mps + stop_intent
          ▼
 deterministic longitudinal controller
          │  mutually exclusive, slew-limited throttle or brake
          ▼
 virtual controller adapter
          │
          ▼
         GTA
```

## Model input boundary

Each sample uses five causal RGB frames and synchronized current-speed telemetry at offsets `[-20, -15, -10, -5, 0]` on the 50 ms telemetry timeline. Offset zero uses the latest telemetry row at or before the RGB anchor; neither the current label nor the image history borrows from the future. Future training targets remain future-facing. Training anchors begin only after one stable cruise second, so the earliest selected window contains cruise rather than launch acceleration. Every anchor is labeled `approach`, `brake_stop`, or `release`, while its temporal context remains continuous across boundaries.

The perception boundary is RGB-only with respect to the stop sign and stopping geometry. These values may be recorded for expert generation, labels, scoring, debugging, and safety, but are not model perception inputs:

- `signPose`, `stopLinePose`, `egoStopPose`, `startPose`, and `exitPose`
- distance to the sign or stop line
- longitudinal, lateral, or heading error to the oracle pose
- stop confirmation completion or ground-truth phase as an input feature

Current speed is allowed proprioception, not a map oracle. This lets the network learn what the stop sign looks like and when the visual approach requires slowing while still knowing the vehicle's own motion state.

## Dataset supervision

Raw 50 ms telemetry retains enough evidence to reproduce and diagnose the expert:

- current speed and acceleration
- normalized physical steering and wheel-angle diagnostics
- physical brake pressure
- expert desired speed and steering
- expert throttle and brake
- expert stop/go probabilities

Collection jobs may apply a small seeded behavior profile: braking deceleration only increases from `3.80` up to `4.05 m/s²`, and release acceleration only increases from `3.50` up to `4.20 m/s²` at maximum Motion variance. These values alter the expert's desired-speed trajectory and are retained in `stopSignGoal.variationProfile` for auditing, but the profile itself is not a policy input. The range is intentionally narrow so the temporal policy sees bounded robustness variation instead of several contradictory driving styles for the same visual state.
- fine phase
- oracle-relative geometry and final outcome

Processed samples retain a causal RGB/history window and a future label sequence. The primary targets are:

| Target | Horizons |
|---|---|
| `future_speed_mps` | `100, 250, 500, 1000 ms` |
| `stop_intent` | the same four horizons |

`expert_throttle`, `expert_brake`, and `actual_brake_pressure` are low-weight auxiliary diagnostics. They help expose poor demonstrations or controller mismatch, but they are not the deployed controller interface.

Every recorded telemetry row carries one fine phase; rolling-capture training normally starts at `cruise_approach`, while `accelerate` remains in the raw audit timeline and closed phase contract:

1. `accelerate`
2. `cruise_approach`
3. `decelerate`
4. `stop_hold`
5. `release`

Training balances observed `(logical stage, fine phase)` groups so long cruise spans do not drown out braking, stopping, or release. Launch-heavy anchors are excluded before balancing; raw launch telemetry and video remain inspectable. History and future labels intentionally cross a stage transition when the anchor is near it. Frames are not independently randomized across train and validation. Split isolation uses the physical stop-sign location so all attempts and generated variants from one sign remain on one side of the split.

Only successful attempts with a complete `stopSignGoal` and `stopSignOutcome` are admitted by default. Failed attempts stay available for inspection and later hard-negative work.

## Continuous release policy

Collection preserves one real motion sequence:

1. `approach` records launch and cruise; processing indexes only the stable-cruise tail.
2. `brake_stop` begins when the planned speed decreases and includes the brief zero-speed confirmation.
3. `release` begins immediately after confirmation without resetting the vehicle, camera, or temporal history.

The controller owns the 250 ms confirmation and release trigger in V0, while release motion is still collected as supervision. Stage flags are labels only and never model inputs. This avoids redundant stationary frames without fabricating or stitching transition motion.

## Controller translation

The model never emits trigger percentages or game key presses. It emits a physical motion plan. The deterministic controller then:

1. validates checkpoint contract, horizon timing, freshness, and finite bounded values;
2. selects the nearest future speed target;
3. compares it with measured vehicle speed;
4. applies a bounded feedback law;
5. rate-limits longitudinal effort;
6. emits throttle **or** brake, never both;
7. latches stop intent with separate enter/release thresholds;
8. fails closed on stale telemetry, stale plans, invalid calibration, or controller errors.

This is the important translation boundary:

```text
model: what motion should happen
controller: what actuator effort should produce it
adapter: how that effort is encoded for the virtual gamepad
game: the observed vehicle response
```

Changing a vehicle, game build, or controller adapter can change the actuator response without changing the model's semantic output. The controller calibration profile therefore records the vehicle/build/adapter identity and remains fail-closed until verified on the live setup.

## Training and evaluation

The checked-in config starts with empty `train_run_ids`, `val_run_ids`, and checkpoint fields. The normal loop is:

1. Capture Start → Stop → End scenes at multiple physical signs and generate many seeded variants from each scene.
2. Process all selected trips with the current frame-window fingerprint.
3. Inspect logical-stage, phase, location, and variant coverage and remove unusable demonstrations.
4. Reserve entire stop-sign locations for validation.
5. Train and compare future-speed error, stop-intent quality, phase coverage, and validation loss.
6. Evaluate offline first, then run a guarded live test with a calibrated controller.

Closed-loop sequence results matter more than a low per-frame loss. A useful evaluation records whether the car launched, approached without oscillation, stopped before the line at the intended pose, avoided collision, and smoothly released to End.

## Safety boundary

Inference must refuse motion when the virtual controller is unavailable, the controller calibration profile is unverified or mismatched, telemetry is stale, the plan is stale, the checkpoint contract is incompatible, or another runtime owns motion. **Hold** supersedes collection and inference starts and requires a fresh FiveM safety acknowledgment before new motion can begin.

## Design references

The separation between privileged expert/scoring signals and deployed visual inputs follows the teacher/student boundary demonstrated by [Learning by Cheating](https://vladlen.info/papers/learning-by-cheating.pdf). Predicting a physical motion plan instead of raw game controls follows the planning boundary used by [TransFuser](https://openaccess.thecvf.com/content/CVPR2021/papers/Prakash_Multi-Modal_Fusion_Transformer_for_End-to-End_Autonomous_Driving_CVPR_2021_paper.pdf).

The geometry and scenario-variation vocabulary also aligns with [CARLA map landmarks](https://carla.readthedocs.io/en/0.9.12/core_map/) and [ASAM OpenSCENARIO parameter distributions](https://publications.pages.asam.net/standards/ASAM_OpenSCENARIO/ASAM_OpenSCENARIO_XML/v1.3.0/09_reuse_mechanisms/09_03_parameter_distribution.html). Closed-loop sequence acceptance is treated as primary for the same reason planning benchmarks such as [nuPlan](https://arxiv.org/abs/2403.04133) evaluate complete rollout behavior rather than isolated action error.
