# Stop-sign temporal model

The active planner learns one causal RGB/telemetry contract from three independent stage clips: `approach`, `brake_stop`, and `release`. Every usable sample also carries the fine phase `accelerate`, `cruise_approach`, `decelerate`, `stop_hold`, or `release`. A processed trip is admitted for expert training only when its metadata contains a successful `stop-sign-goal.v2` / `stopSignOutcome` pair for `stop-sign:temporal-v1` and a valid clip stage.

## Inputs and outputs

- Inputs are RGB frames at `[-20, -15, -10, -5, 0]` and current-speed telemetry at the same offsets. All offsets are causal; zero is the current sample.
- Primary outputs are `future_speed_mps` and `stop_intent` at `[100, 250, 500, 1000]` ms.
- `expert_throttle`, `expert_brake`, and physical `actual_brake_pressure` remain low-weight auxiliary diagnostics. They are not the controller interface.
- Oracle sign/stop-line distances are supervision and scoring metadata only. They are intentionally absent from the checked-in model inputs.

Training balances the joint `(clip_stage, phase)` groups so long approach clips cannot dominate short brake or release clips. Windows are constructed inside one trip, so temporal history and targets cannot cross stage boundaries. Train/validation isolation is enforced by catalog stop-sign identity (with quantized pose only as a compatibility fallback); frames or attempts from the same physical stop cannot appear in both splits.

## Controller handoff

The model server returns `motion_plan.future_speed_mps`, `motion_plan.stop_intent`, horizon timings, and `release_policy`. A deterministic longitudinal controller turns the nearest target into mutually exclusive throttle or brake and owns rate limiting and safety clamps.

Release is explicitly `scripted_stage_release_v1`: the runtime starts a separate release clip from the captured Stop pose after the brief zero-speed confirmation. The model does not claim to learn the transition between clips.
