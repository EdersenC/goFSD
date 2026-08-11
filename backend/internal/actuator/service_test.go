package actuator

import (
	"errors"
	"os"
	"strings"
	"testing"
	"time"

	"awesomeProject/internal/control"
	"awesomeProject/internal/parkingcontrol"
)

type fakeController struct {
	applied []controlState
	err     error
}

func (f *fakeController) Apply(next controlState) error {
	f.applied = append(f.applied, next)
	return f.err
}

func (f *fakeController) Close() {}

func TestStepAppliesNormalizedInputs(t *testing.T) {
	cfg := DefaultConfig()
	cfg.StaleTimeout = time.Second
	svc := NewService(cfg, "")
	svc.controller = &fakeController{}
	svc.ready = true
	svc.supported = true
	base := time.Date(2026, 4, 22, 12, 0, 0, 0, time.UTC)
	svc.lastCmd = &commandEnvelope{
		controlState: controlState{
			Steer:    0.6,
			Throttle: 0.7,
			Brake:    0.2,
		},
		Enabled:    true,
		InputMode:  InputModeNormalized,
		ReceivedAt: base.Format(time.RFC3339Nano),
	}

	if err := svc.step(base.Add(10 * time.Millisecond)); err != nil {
		t.Fatalf("step returned error: %v", err)
	}
	if svc.applied.Steer != 0.6 {
		t.Fatalf("unexpected steer: %f", svc.applied.Steer)
	}
	if diff := svc.applied.Throttle - 0.7; diff < -1e-9 || diff > 1e-9 {
		t.Fatalf("unexpected throttle: %f", svc.applied.Throttle)
	}
	if svc.applied.Brake != 0 {
		t.Fatalf("unexpected brake: %f", svc.applied.Brake)
	}
}

func TestStepAppliesLiveActuatorTuning(t *testing.T) {
	cfg := DefaultConfig()
	cfg.StaleTimeout = time.Second
	svc := NewService(cfg, "")
	svc.controller = &fakeController{}
	svc.ready = true
	svc.supported = true
	tuning := cfg.Tuning()
	tuning.SteeringGain = 1.5
	tuning.ThrottleGain = 2.0
	tuning.ThrottleFloor = 0.35
	if _, err := svc.ApplyTuning(tuning); err != nil {
		t.Fatalf("ApplyTuning returned error: %v", err)
	}

	base := time.Date(2026, 4, 22, 12, 0, 0, 0, time.UTC)
	svc.lastCmd = &commandEnvelope{
		controlState: controlState{
			Steer:    0.4,
			Throttle: 0.1,
			Brake:    0.3,
		},
		Enabled:    true,
		InputMode:  InputModeNormalized,
		ReceivedAt: base.Format(time.RFC3339Nano),
	}

	if err := svc.step(base.Add(10 * time.Millisecond)); err != nil {
		t.Fatalf("step returned error: %v", err)
	}
	if diff := svc.applied.Steer - 0.6; diff < -1e-9 || diff > 1e-9 {
		t.Fatalf("unexpected tuned steer: %f", svc.applied.Steer)
	}
	if diff := svc.applied.Throttle - 0.35; diff < -1e-9 || diff > 1e-9 {
		t.Fatalf("unexpected tuned throttle after weak brake threshold: %f", svc.applied.Throttle)
	}
	if diff := svc.applied.Brake - 0.0; diff < -1e-9 || diff > 1e-9 {
		t.Fatalf("unexpected brake: %f", svc.applied.Brake)
	}
}

func TestStepAppliesSpeedLimitBrakeThresholdAndReverseLockout(t *testing.T) {
	now := time.Date(2026, 4, 22, 12, 0, 0, 0, time.UTC)
	store := control.NewStore(control.WithNowFunc(func() time.Time { return now }))
	store.UpdateTelemetry(control.TelemetryUpdate{CurrentSpeed: 5.5, TimestampMs: now.UnixMilli()})
	cfg := DefaultConfig()
	cfg.StaleTimeout = time.Second
	cfg.SpeedLimitKPH = 17.0
	cfg.OverspeedBrakeMarginKPH = 2.0
	cfg.OverspeedBrake = 0.25
	cfg.ModelBrakeThreshold = 0.55
	cfg.ReverseLockoutSpeedKPH = 1.0
	svc := NewService(cfg, "", store)
	svc.controller = &fakeController{}
	svc.ready = true
	svc.supported = true
	svc.lastCmd = &commandEnvelope{
		controlState: controlState{
			Throttle: 0.8,
			Brake:    0.4,
		},
		Enabled:    true,
		InputMode:  InputModeNormalized,
		ReceivedAt: now.Format(time.RFC3339Nano),
	}

	if err := svc.step(now.Add(10 * time.Millisecond)); err != nil {
		t.Fatalf("step returned error: %v", err)
	}
	if svc.applied.Throttle != 0 {
		t.Fatalf("expected speed limiter to cut throttle, got=%+v", svc.applied)
	}
	if svc.applied.Brake != 0.25 {
		t.Fatalf("expected overspeed brake after weak model brake was ignored, got=%+v", svc.applied)
	}
	if !svc.target.Safety.BrakeThresholdApplied || !svc.target.Safety.SpeedLimitActive || !svc.target.Safety.OverspeedBrakeApplied {
		t.Fatalf("expected safety debug flags, got=%+v", svc.target.Safety)
	}

	now = now.Add(100 * time.Millisecond)
	store.UpdateTelemetry(control.TelemetryUpdate{CurrentSpeed: 0.1, TimestampMs: now.UnixMilli()})
	svc.lastCmd = &commandEnvelope{
		controlState: controlState{
			Brake: 0.9,
		},
		Enabled:    true,
		InputMode:  InputModeNormalized,
		ReceivedAt: now.Format(time.RFC3339Nano),
	}
	if err := svc.step(now.Add(10 * time.Millisecond)); err != nil {
		t.Fatalf("second step returned error: %v", err)
	}
	if svc.applied.Brake != 0 || !svc.applied.Handbrake {
		t.Fatalf("expected reverse lockout to translate brake into a handbrake hold near stop, got=%+v", svc.applied)
	}
	if !svc.target.Safety.ReverseLockoutApplied || !svc.target.Safety.LowSpeedBrakeHold {
		t.Fatalf("expected reverse lockout debug flag, got=%+v", svc.target.Safety)
	}

	now = now.Add(100 * time.Millisecond)
	store.UpdateTelemetry(control.TelemetryUpdate{CurrentSpeed: 0.1, TimestampMs: now.UnixMilli()})
	svc.lastCmd = &commandEnvelope{
		controlState: controlState{
			Throttle: 0.8,
		},
		Enabled:    true,
		InputMode:  InputModeNormalized,
		ReceivedAt: now.Format(time.RFC3339Nano),
	}
	if err := svc.step(now.Add(10 * time.Millisecond)); err != nil {
		t.Fatalf("forward release step returned error: %v", err)
	}
	if svc.applied.Handbrake || svc.applied.Brake != 0 || svc.applied.Throttle <= 0 {
		t.Fatalf("expected forward throttle to release the automatic handbrake hold, got=%+v", svc.applied)
	}
}

func TestResolveTargetTimesOutToNeutralByDefault(t *testing.T) {
	now := time.Date(2026, 4, 22, 12, 0, 0, 0, time.UTC)
	target, stale, enabled, timedOut := resolveTarget(&commandEnvelope{
		controlState: controlState{Steer: 0.2, Throttle: 0.4, Brake: 0.2},
		Enabled:      true,
		InputMode:    InputModeNormalized,
		ReceivedAt:   now.Add(-time.Second).Format(time.RFC3339Nano),
	}, 100*time.Millisecond, now)

	if target != (controlState{}) {
		t.Fatalf("expected neutral target, got=%+v", target)
	}
	if !stale || enabled || !timedOut {
		t.Fatalf("unexpected timeout flags stale=%v enabled=%v timedOut=%v", stale, enabled, timedOut)
	}
}

func TestDisabledSafetyHoldPersistsUntilExplicitRelease(t *testing.T) {
	cfg := DefaultConfig()
	cfg.StaleTimeout = 250 * time.Millisecond
	now := time.Date(2026, 8, 5, 12, 0, 0, 0, time.UTC)
	svc := NewService(cfg, "")
	svc.controller = &fakeController{}
	svc.ready = true
	svc.supported = true
	svc.nowFunc = func() time.Time { return now }
	enabled := false

	if _, err := svc.Submit(CommandRequest{Enabled: &enabled, Handbrake: true}); err != nil {
		t.Fatalf("submit safety hold: %v", err)
	}
	if err := svc.step(now.Add(cfg.StaleTimeout + time.Second)); err != nil {
		t.Fatalf("step beyond stale timeout: %v", err)
	}
	if !svc.applied.Handbrake || svc.applied.TimedOut || svc.applied.Enabled {
		t.Fatalf("expected disabled safety hold to remain latched, got=%+v", svc.applied)
	}

	now = now.Add(cfg.StaleTimeout + time.Second)
	if _, err := svc.Submit(CommandRequest{Enabled: &enabled, Handbrake: false}); err != nil {
		t.Fatalf("submit explicit release: %v", err)
	}
	if err := svc.step(now.Add(cfg.StaleTimeout + time.Second)); err != nil {
		t.Fatalf("step after explicit release: %v", err)
	}
	if svc.applied.Handbrake || svc.applied.TimedOut || svc.applied.Enabled {
		t.Fatalf("expected disabled neutral command to remain neutral, got=%+v", svc.applied)
	}
}

func TestSubmitDefaultsToNormalizedInputMode(t *testing.T) {
	svc := NewService(DefaultConfig(), "")
	svc.controller = &fakeController{}
	svc.ready = true
	svc.supported = true

	state, err := svc.Submit(CommandRequest{Steer: 0.2, Throttle: 0.4, BrakePressureAvg: 0.3})
	if err != nil {
		t.Fatalf("Submit returned error: %v", err)
	}
	if state.LastCommand == nil {
		t.Fatal("expected last command")
	}
	if state.LastCommand.InputMode != InputModeNormalized {
		t.Fatalf("expected normalized input mode, got=%s", state.LastCommand.InputMode)
	}
	if state.LastCommand.Brake != 0.3 {
		t.Fatalf("expected brake to be stored on command, got=%+v", state.LastCommand)
	}
}

func TestAppliedStateCarriesExactSubmittedCommandID(t *testing.T) {
	cfg := DefaultConfig()
	now := time.Date(2026, 8, 5, 12, 0, 0, 0, time.UTC)
	svc := NewService(cfg, "")
	svc.controller = &fakeController{}
	svc.ready = true
	svc.supported = true
	svc.nowFunc = func() time.Time { return now }
	enabled := false

	queued, err := svc.Submit(CommandRequest{Enabled: &enabled, Handbrake: true})
	if err != nil {
		t.Fatalf("Submit returned error: %v", err)
	}
	if queued.LastCommandID <= 0 || queued.Applied.CommandID == queued.LastCommandID {
		t.Fatalf("expected Submit receipt before controller apply, got=%+v", queued)
	}
	if err := svc.step(now.Add(10 * time.Millisecond)); err != nil {
		t.Fatalf("step returned error: %v", err)
	}
	applied := svc.State()
	if applied.Applied.CommandID != queued.LastCommandID || applied.LastApplyAttemptedCommandID != queued.LastCommandID {
		t.Fatalf("expected exact command receipt to reach controller apply state, queued=%d state=%+v", queued.LastCommandID, applied)
	}
	if !applied.Applied.Handbrake || applied.LastApplySucceededAt == "" || applied.LastApplyError != "" {
		t.Fatalf("expected successful applied safety hold, got=%+v", applied)
	}
}

func TestCloseClearsLastCommandSoRestartCannotReplayDrive(t *testing.T) {
	svc := NewService(DefaultConfig(), "")
	svc.controller = &fakeController{}
	svc.ready = true
	svc.supported = true
	svc.lastCmd = &commandEnvelope{
		controlState: controlState{Steer: 0.5, Throttle: 0.8},
		CommandID:    7,
		Enabled:      true,
		ReceivedAt:   time.Now().UTC().Format(time.RFC3339Nano),
	}
	svc.nextCommandID = 7

	if err := svc.Close(); err != nil {
		t.Fatalf("Close returned error: %v", err)
	}
	state := svc.State()
	if state.LastCommand != nil || state.LastCommandID != 0 || state.Applied.CommandID != 0 || state.Target.CommandID != 0 {
		t.Fatalf("expected close to clear replayable command and applied receipts, got=%+v", state)
	}
	if svc.nextCommandID != 7 {
		t.Fatalf("expected command IDs to remain monotonic across restart, got=%d", svc.nextCommandID)
	}
}

func TestSubmitRejectsInvalidInputMode(t *testing.T) {
	svc := NewService(DefaultConfig(), "")
	svc.controller = &fakeController{}
	svc.ready = true
	svc.supported = true

	_, err := svc.Submit(CommandRequest{InputMode: "model_raw"})
	if err == nil || !errors.Is(err, ErrInvalidInputMode) {
		t.Fatalf("expected invalid input mode error, got=%v", err)
	}
}

func TestStepReportsControllerError(t *testing.T) {
	controllerErr := errors.New("controller failed")
	svc := NewService(DefaultConfig(), "")
	svc.controller = &fakeController{err: controllerErr}
	svc.ready = true
	svc.supported = true
	base := time.Date(2026, 4, 22, 12, 0, 0, 0, time.UTC)
	svc.nowFunc = func() time.Time { return base }
	queued, submitErr := svc.Submit(CommandRequest{
		Steer:            0.2,
		BrakePressureAvg: 0.1,
		InputMode:        InputModeNormalized,
	})
	if submitErr != nil {
		t.Fatalf("Submit returned error: %v", submitErr)
	}

	err := svc.step(base.Add(10 * time.Millisecond))
	if !errors.Is(err, controllerErr) {
		t.Fatalf("expected controller error, got=%v", err)
	}
	if svc.lastApplyError == "" {
		t.Fatal("expected last apply error to be recorded")
	}
	state := svc.State()
	if state.LastApplyAttemptedCommandID != queued.LastCommandID {
		t.Fatalf("expected failed apply to identify command %d, got=%+v", queued.LastCommandID, state)
	}
	if state.Applied.CommandID == queued.LastCommandID {
		t.Fatalf("failed controller apply must not advance applied command ID, got=%+v", state.Applied)
	}
}

func TestResolveServiceThrottleBrakeConflictLetsBrakeWin(t *testing.T) {
	resolved, changed := resolveServiceThrottleBrakeConflict(controlState{
		Steer:    0.1,
		Throttle: 0.30,
		Brake:    0.60,
	})
	if !changed {
		t.Fatalf("expected conflict resolution to trigger")
	}
	if resolved.Throttle != 0 || resolved.Brake != 0.60 {
		t.Fatalf("unexpected resolved state: %+v", resolved)
	}
}

func TestResolveServiceThrottleBrakeConflictAlwaysLetsBrakeWin(t *testing.T) {
	resolved, changed := resolveServiceThrottleBrakeConflict(controlState{
		Steer:    0.1,
		Throttle: 0.80,
		Brake:    0.30,
	})
	if !changed {
		t.Fatalf("expected conflict resolution to trigger")
	}
	if resolved.Throttle != 0 || resolved.Brake != 0.30 {
		t.Fatalf("unexpected resolved state: %+v", resolved)
	}
}

func TestApplyResetAndSaveTuning(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/train_config.toml"
	content := []byte("[backend.actuator]\nsteering_gain = 1.0\nthrottle_gain = 1.0\nthrottle_floor = 0.0\n")
	if err := os.WriteFile(path, content, 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	svc := NewService(DefaultConfig(), path)
	next := DefaultConfig().Tuning()
	next.SteeringGain = 1.4
	next.ThrottleGain = 1.8
	next.ThrottleFloor = 0.22
	state, err := svc.ApplyTuning(next)
	if err != nil {
		t.Fatalf("ApplyTuning returned error: %v", err)
	}
	if state.Live != next {
		t.Fatalf("unexpected live tuning: %+v", state.Live)
	}

	reset := svc.ResetTuning()
	if reset.Live.SteeringGain != DefaultConfig().SteeringGain {
		t.Fatalf("unexpected reset tuning: %+v", reset.Live)
	}

	if _, err := svc.ApplyTuning(next); err != nil {
		t.Fatalf("second ApplyTuning returned error: %v", err)
	}
	saved, err := svc.SaveTuning()
	if err != nil {
		t.Fatalf("SaveTuning returned error: %v", err)
	}
	if saved.Saved != next {
		t.Fatalf("unexpected saved tuning: %+v", saved.Saved)
	}

	loaded, err := LoadConfig(path)
	if err != nil {
		t.Fatalf("LoadConfig returned error: %v", err)
	}
	if loaded.Tuning() != next {
		t.Fatalf("unexpected persisted config: got=%+v want=%+v", loaded.Tuning(), next)
	}
}

func TestLoadParkingControllerConfigKeepsCheckedInProfileFailClosed(t *testing.T) {
	path := t.TempDir() + "/train_config.toml"
	content := []byte(`[backend.actuator]
tick_hz = 60

[backend.parking_controller]
calibration_verified = false
calibration_profile_id = ''
vehicle_model_hash = 0
game_build = ''
adapter_version = ''
steering_convention = 'positive_wheel_is_positive_xinput'
steering_profile = [
  { wheel_steer = -1.0, command = -0.8 },
  { wheel_steer = 0.0, command = 0.0 },
  { wheel_steer = 1.0, command = 0.8 },
]
straight_approach_only = true
speed_kp = 0.5
plan_timeout = '350ms'
telemetry_timeout = '200ms'
estimated_actuation_latency = '40ms'
expected_horizon_dt_ms = [50, 100, 150, 200, 250, 300]
`)
	if err := os.WriteFile(path, content, 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	cfg, err := LoadConfig(path)
	if err != nil {
		t.Fatalf("LoadConfig returned error: %v", err)
	}
	if cfg.ParkingCalibration.Verified || cfg.ParkingCalibration.VehicleModelHash != 0 {
		t.Fatalf("expected the placeholder calibration to remain fail-closed, got=%+v", cfg.ParkingCalibration)
	}
	if cfg.TickHz != 60 || !cfg.ParkingController.StraightApproachOnly || cfg.ParkingController.SpeedKp != 0.5 || cfg.ParkingController.SteeringProfile[0].Command != -0.8 {
		t.Fatalf("unexpected parsed parking controller config: %+v", cfg)
	}
	if cfg.ParkingPlanTimeout != 350*time.Millisecond || cfg.ParkingTelemetryTimeout != 200*time.Millisecond || cfg.ParkingEstimatedActuationLatency != 40*time.Millisecond {
		t.Fatalf("unexpected parking timing config: %+v", cfg)
	}
}

func TestLoadParkingControllerConfigRejectsUntraceableVerifiedProfile(t *testing.T) {
	path := t.TempDir() + "/train_config.toml"
	content := []byte(`[backend.parking_controller]
calibration_verified = true
calibration_profile_id = 'parking-v1'
vehicle_model_hash = 123
steering_convention = 'positive_wheel_is_positive_xinput'
`)
	if err := os.WriteFile(path, content, 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	if _, err := LoadConfig(path); err == nil || !strings.Contains(err.Error(), "game_build") {
		t.Fatalf("expected verified profile without provenance to be rejected, got=%v", err)
	}
}

func TestParkingSetpointPlanRunsCalibratedFeedbackController(t *testing.T) {
	base := time.Date(2026, 8, 5, 12, 0, 0, 0, time.UTC)
	now := base
	store := control.NewStore(control.WithNowFunc(func() time.Time { return now }))
	updateParkingTelemetry(store, base, 0.25, 0.10, testParkingVehicleModelHash)
	svc := newReadyParkingService(testParkingConfig(), store, &now)
	armParkingInference(t, svc)

	state, err := svc.SubmitParkingSetpointPlan(testParkingPlan(base))
	if err != nil {
		t.Fatalf("SubmitParkingSetpointPlan returned error: %v", err)
	}
	if state.ParkingController.LastPlanID != 1 || state.ParkingController.Owner != OwnerParkingInference {
		t.Fatalf("expected accepted plan receipt and ownership, got=%+v", state.ParkingController)
	}

	stepAt := base.Add(20 * time.Millisecond)
	if err := svc.step(stepAt); err != nil {
		t.Fatalf("parking step returned error: %v", err)
	}
	applied := svc.State()
	if applied.Applied.PlanID != 1 || applied.Target.Parking == nil || applied.Target.Parking.FailSafe {
		t.Fatalf("expected plan 1 to reach the parking controller, got=%+v", applied)
	}
	if applied.Target.Parking.SelectedSetpoint == nil || applied.Target.Parking.MeasuredWheelSteer == nil {
		t.Fatalf("expected selected setpoint and physical feedback in trace, got=%+v", applied.Target.Parking)
	}
	if applied.Applied.Throttle <= 0 || applied.Applied.Brake != 0 {
		t.Fatalf("expected mutually exclusive positive speed effort, got=%+v", applied.Applied)
	}
	if applied.Applied.Steer >= 0 || applied.Applied.Steer < -1 {
		t.Fatalf("expected straight-mode feedback to center positive measured steering, got=%+v", applied.Applied)
	}
	if applied.Target.Parking.ControllerOutput.SteeringFeedForward != 0 {
		t.Fatalf("expected straight mode to ignore model steering feed-forward, got=%+v", applied.Target.Parking)
	}
	if applied.ParkingController.LastPlanAppliedID != 1 || applied.LastApplyAttemptedPlanID != 1 {
		t.Fatalf("expected exact plan receipt through controller apply, got=%+v", applied)
	}
}

func TestParkingSetpointPlanFailsClosedWhenPlanExpires(t *testing.T) {
	base := time.Date(2026, 8, 5, 12, 0, 0, 0, time.UTC)
	tests := []struct {
		name              string
		speedMPS          float64
		wantBrake         bool
		wantHandbrakeHold bool
	}{
		{name: "moving uses service brake", speedMPS: 0.8, wantBrake: true},
		{name: "confirmed stop uses handbrake", speedMPS: 0.05, wantHandbrakeHold: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			now := base
			store := control.NewStore(control.WithNowFunc(func() time.Time { return now }))
			updateParkingTelemetry(store, base, test.speedMPS, 0, testParkingVehicleModelHash)
			cfg := testParkingConfig()
			cfg.ParkingTelemetryTimeout = time.Second
			svc := newReadyParkingService(cfg, store, &now)
			armParkingInference(t, svc)
			if _, err := svc.SubmitParkingSetpointPlan(testParkingPlan(base)); err != nil {
				t.Fatalf("SubmitParkingSetpointPlan returned error: %v", err)
			}

			stepAt := base.Add(cfg.ParkingPlanTimeout + time.Millisecond)
			if err := svc.step(stepAt); err != nil {
				t.Fatalf("parking fail-safe step returned error: %v", err)
			}
			state := svc.State()
			if state.Target.Parking == nil || !state.Target.Parking.FailSafe || !state.Applied.TimedOut {
				t.Fatalf("expected explicit parking fail-safe trace, got=%+v", state)
			}
			if test.wantBrake && (state.Applied.Brake <= 0 || state.Applied.Handbrake || state.Applied.Throttle != 0) {
				t.Fatalf("expected service-brake fail-safe while moving, got=%+v", state.Applied)
			}
			if test.wantHandbrakeHold && (!state.Applied.Handbrake || state.Applied.Brake != 0 || state.Applied.Throttle != 0) {
				t.Fatalf("expected handbrake fail-safe only at confirmed low speed, got=%+v", state.Applied)
			}
		})
	}
}

func TestParkingSetpointPlanRequiresVerifiedCalibration(t *testing.T) {
	base := time.Date(2026, 8, 5, 12, 0, 0, 0, time.UTC)
	now := base
	store := control.NewStore(control.WithNowFunc(func() time.Time { return now }))
	updateParkingTelemetry(store, base, 0.2, 0, testParkingVehicleModelHash)
	cfg := testParkingConfig()
	cfg.ParkingCalibration.Verified = false
	svc := newReadyParkingService(cfg, store, &now)
	armParkingInference(t, svc)

	_, err := svc.SubmitParkingSetpointPlan(testParkingPlan(base))
	if !errors.Is(err, ErrParkingControllerNotReady) {
		t.Fatalf("expected unverified calibration to block setpoint plans, got=%v", err)
	}
}

func TestParkingSetpointPlanRejectsWrongHorizonTiming(t *testing.T) {
	base := time.Date(2026, 8, 5, 12, 0, 0, 0, time.UTC)
	now := base
	store := control.NewStore(control.WithNowFunc(func() time.Time { return now }))
	updateParkingTelemetry(store, base, 0.2, 0, testParkingVehicleModelHash)
	svc := newReadyParkingService(testParkingConfig(), store, &now)
	armParkingInference(t, svc)
	plan := testParkingPlan(base)
	plan.Points[1].DtMs = 110

	if _, err := svc.SubmitParkingSetpointPlan(plan); err == nil {
		t.Fatal("expected mismatched model/controller timing to be rejected")
	}
}

func TestParkingInferenceOwnershipRejectsCompetingDriveCommand(t *testing.T) {
	base := time.Date(2026, 8, 5, 12, 0, 0, 0, time.UTC)
	now := base
	store := control.NewStore(control.WithNowFunc(func() time.Time { return now }))
	updateParkingTelemetry(store, base, 0.2, 0, testParkingVehicleModelHash)
	svc := newReadyParkingService(testParkingConfig(), store, &now)
	armParkingInference(t, svc)

	enabled := true
	_, err := svc.Submit(CommandRequest{Enabled: &enabled, Throttle: 0.5, Owner: OwnerCalibration})
	if !errors.Is(err, ErrParkingSessionOwned) {
		t.Fatalf("expected active parking session ownership error, got=%v", err)
	}

	disabled := false
	if _, err := svc.Submit(CommandRequest{Enabled: &disabled, Handbrake: true, Owner: OwnerCalibration}); err != nil {
		t.Fatalf("expected disabled safety preemption to remain available, got=%v", err)
	}
}

func TestParkingControllerFailsClosedOnVehicleCalibrationMismatch(t *testing.T) {
	base := time.Date(2026, 8, 5, 12, 0, 0, 0, time.UTC)
	now := base
	store := control.NewStore(control.WithNowFunc(func() time.Time { return now }))
	updateParkingTelemetry(store, base, 0.8, 0, testParkingVehicleModelHash+1)
	svc := newReadyParkingService(testParkingConfig(), store, &now)
	armParkingInference(t, svc)
	if _, err := svc.SubmitParkingSetpointPlan(testParkingPlan(base)); err != nil {
		t.Fatalf("SubmitParkingSetpointPlan returned error: %v", err)
	}

	if err := svc.step(base.Add(20 * time.Millisecond)); err != nil {
		t.Fatalf("parking fail-safe step returned error: %v", err)
	}
	state := svc.State()
	if state.Target.Parking == nil || !state.Target.Parking.FailSafe || state.Applied.Brake <= 0 || state.Applied.Throttle != 0 {
		t.Fatalf("expected vehicle mismatch to fail closed with service brake, got=%+v", state)
	}
}

func TestParkingArmAppliesSafeStopUntilFirstPlan(t *testing.T) {
	base := time.Date(2026, 8, 5, 12, 0, 0, 0, time.UTC)
	now := base
	store := control.NewStore(control.WithNowFunc(func() time.Time { return now }))
	updateParkingTelemetry(store, base, 0.8, 0, testParkingVehicleModelHash)
	svc := newReadyParkingService(testParkingConfig(), store, &now)
	armParkingInference(t, svc)

	if err := svc.step(base.Add(20 * time.Millisecond)); err != nil {
		t.Fatalf("apply awaiting-plan stop: %v", err)
	}
	state := svc.State()
	if state.Applied.Brake <= 0 || state.Applied.Handbrake || state.Target.Parking == nil || state.Target.Parking.PlanState != "awaiting-plan" {
		t.Fatalf("expected service brake while awaiting the first plan at speed, got=%+v", state)
	}

	now = base.Add(40 * time.Millisecond)
	updateParkingTelemetry(store, now, 0.05, 0, testParkingVehicleModelHash)
	if err := svc.step(now); err != nil {
		t.Fatalf("apply awaiting-plan handbrake: %v", err)
	}
	state = svc.State()
	if !state.Applied.Handbrake || state.Applied.Brake != 0 || state.Applied.Throttle != 0 {
		t.Fatalf("expected handbrake only after fresh telemetry confirms low speed, got=%+v", state.Applied)
	}
}

func TestParkingSafetyStopTransitionsFromServiceBrakeToHandbrake(t *testing.T) {
	base := time.Date(2026, 8, 5, 12, 0, 0, 0, time.UTC)
	now := base
	store := control.NewStore(control.WithNowFunc(func() time.Time { return now }))
	updateParkingTelemetry(store, base, 0.8, 0, testParkingVehicleModelHash)
	svc := newReadyParkingService(testParkingConfig(), store, &now)
	armParkingInference(t, svc)

	queued, err := svc.RequestParkingSafetyStop()
	if err != nil {
		t.Fatalf("RequestParkingSafetyStop returned error: %v", err)
	}
	if queued.LastCommandID <= 0 || !queued.ParkingController.Stopping {
		t.Fatalf("expected trackable actuator-owned stop request, got=%+v", queued)
	}
	if err := svc.step(base.Add(20 * time.Millisecond)); err != nil {
		t.Fatalf("apply moving safety stop: %v", err)
	}
	state := svc.State()
	if state.Applied.CommandID != queued.LastCommandID || state.Applied.Brake <= 0 || state.Applied.Handbrake {
		t.Fatalf("expected service brake while moving, got=%+v", state.Applied)
	}

	now = base.Add(40 * time.Millisecond)
	updateParkingTelemetry(store, now, 0.05, 0, testParkingVehicleModelHash)
	if err := svc.step(now); err != nil {
		t.Fatalf("apply stopped safety hold: %v", err)
	}
	state = svc.State()
	if !state.Applied.Handbrake || state.Applied.Brake != 0 || !state.ParkingController.Stopping {
		t.Fatalf("expected persistent handbrake hold after confirmed stop, got=%+v", state)
	}
	if _, err := svc.SubmitParkingSetpointPlan(testParkingPlan(now)); !errors.Is(err, ErrParkingSessionOwned) {
		t.Fatalf("expected safety stop to reject later plans, got=%v", err)
	}
}

func TestParkingSetpointPlanRejectsActiveControllerApplyFault(t *testing.T) {
	base := time.Date(2026, 8, 5, 12, 0, 0, 0, time.UTC)
	now := base
	store := control.NewStore(control.WithNowFunc(func() time.Time { return now }))
	updateParkingTelemetry(store, base, 0.2, 0, testParkingVehicleModelHash)
	svc := newReadyParkingService(testParkingConfig(), store, &now)
	armParkingInference(t, svc)
	svc.lastApplyError = "virtual controller write failed"

	_, err := svc.SubmitParkingSetpointPlan(testParkingPlan(base))
	if err == nil || !strings.Contains(err.Error(), "apply fault") {
		t.Fatalf("expected active controller apply fault to reject the plan, got=%v", err)
	}
}

func TestParkingControllerRejectsFreshReceiptOfStaleSourceTelemetry(t *testing.T) {
	base := time.Date(2026, 8, 5, 12, 0, 0, 0, time.UTC)
	now := base.Add(300 * time.Millisecond)
	store := control.NewStore(control.WithNowFunc(func() time.Time { return now }))
	updateParkingTelemetry(store, base, 0.2, 0, testParkingVehicleModelHash)
	svc := newReadyParkingService(testParkingConfig(), store, &now)
	armParkingInference(t, svc)
	if _, err := svc.SubmitParkingSetpointPlan(testParkingPlan(now)); err != nil {
		t.Fatalf("SubmitParkingSetpointPlan returned error: %v", err)
	}

	if err := svc.step(now.Add(20 * time.Millisecond)); err != nil {
		t.Fatalf("parking fail-safe step returned error: %v", err)
	}
	state := svc.State()
	if state.Target.Parking == nil || !state.Target.Parking.FailSafe || !strings.Contains(state.Target.Parking.Fault, "source is stale") {
		t.Fatalf("expected stale source timestamp to fail closed despite a fresh receipt, got=%+v", state)
	}
	if state.Applied.Brake <= 0 || state.Applied.Handbrake {
		t.Fatalf("expected unknown-speed source timing failure to use service brake, got=%+v", state.Applied)
	}
}

const testParkingVehicleModelHash int64 = 123456

func testParkingConfig() Config {
	cfg := DefaultConfig()
	cfg.SpeedLimitKPH = 0
	cfg.ParkingCalibration = ParkingCalibration{
		Verified:           true,
		ProfileID:          "test-profile-v1",
		VehicleModelHash:   testParkingVehicleModelHash,
		GameBuild:          "test-build",
		AdapterVersion:     "test-adapter-v1",
		SteeringConvention: "positive_wheel_is_positive_xinput",
	}
	return cfg
}

func newReadyParkingService(cfg Config, store *control.Store, now *time.Time) *Service {
	svc := NewService(cfg, "", store)
	svc.controller = &fakeController{}
	svc.ready = true
	svc.supported = true
	svc.nowFunc = func() time.Time { return *now }
	return svc
}

func armParkingInference(t *testing.T, svc *Service) {
	t.Helper()
	enabled := true
	if _, err := svc.Submit(CommandRequest{
		Enabled:   &enabled,
		InputMode: InputModeNormalized,
		Owner:     OwnerParkingInference,
	}); err != nil {
		t.Fatalf("arm parking inference: %v", err)
	}
}

func testParkingPlan(sampledAt time.Time) parkingcontrol.Plan {
	dtMs := []int{50, 100, 150, 200, 250, 300}
	points := make([]parkingcontrol.Setpoint, len(dtMs))
	for index, offset := range dtMs {
		points[index] = parkingcontrol.Setpoint{
			DtMs:                        offset,
			DesiredWheelSteerNormalized: 0.45,
			DesiredSpeedMPS:             1.0,
			StopProbability:             0.05,
		}
	}
	return parkingcontrol.Plan{
		Contract:    parkingcontrol.ParkingSetpointContractV1,
		SampledAtS:  timeToSeconds(sampledAt),
		ReceivedAtS: timeToSeconds(sampledAt),
		Points:      points,
		Direction:   parkingcontrol.ParkingDirectionForward,
	}
}

func updateParkingTelemetry(store *control.Store, at time.Time, speedMPS, steering float64, vehicleModelHash int64) {
	store.UpdateTelemetry(control.TelemetryUpdate{
		CurrentSpeed:     speedMPS,
		Steering:         steering,
		VehicleExists:    true,
		IsInVehicle:      true,
		VehicleModelHash: vehicleModelHash,
		TimestampMs:      at.UnixMilli(),
	})
}
