# Stop-sign temporal model

The active V0 planner learns one causal RGB/telemetry sequence contract for a straight stop-sign approach. Every usable sample carries the fine phase `accelerate`, `cruise_approach`, `decelerate`, `stop_hold`, or `release`. A processed trip is admitted for expert training only when its metadata contains a successful `stopSignGoal` / `stopSignOutcome` pair for `stop-sign:temporal-v1`.

## Inputs and outputs

- Inputs are RGB frames at `[-20, -15, -10, -5, 0]` and current-speed telemetry at the same offsets. All offsets are causal; zero is the current sample.
- Primary outputs are `future_speed_mps` and `stop_intent` at `[250, 500, 1000, 2000, 3000, 5000]` ms.
- `expert_throttle`, `expert_brake`, and physical `actual_brake_pressure` remain low-weight auxiliary diagnostics. They are not the controller interface.
- Oracle sign/stop-line distances are supervision and scoring metadata only. They are intentionally absent from the checked-in model inputs.

Training uses inverse-frequency phase sampling, so each observed phase has equal sampler mass. Train/validation isolation is enforced by quantized physical stop-sign pose; frames or attempts from the same stop location cannot appear in both splits.

## Controller handoff

The model server returns `motion_plan.future_speed_mps`, `motion_plan.stop_intent`, horizon timings, and `release_policy`. A deterministic longitudinal controller turns the nearest target into mutually exclusive throttle or brake and owns rate limiting and safety clamps.

V0 release is explicitly `scripted_dwell_release_v0`: the runtime holds the vehicle for the configured dwell (about five seconds) and releases only after that state machine completes. The model does not decide when the dwell is complete. Learning release timing is a later contract version, not a silent behavior change.
