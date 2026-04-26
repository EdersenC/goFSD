package actuator

import (
	"errors"
	"os"
	"testing"
	"time"
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
	if diff := svc.applied.Throttle - 0.5; diff < -1e-9 || diff > 1e-9 {
		t.Fatalf("unexpected throttle: %f", svc.applied.Throttle)
	}
	if svc.applied.Brake != 0.2 {
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
	if _, err := svc.ApplyTuning(Tuning{
		SteeringGain:  1.5,
		ThrottleGain:  2.0,
		ThrottleFloor: 0.35,
	}); err != nil {
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
	if diff := svc.applied.Throttle - 0.05; diff < -1e-9 || diff > 1e-9 {
		t.Fatalf("unexpected tuned throttle after brake conflict: %f", svc.applied.Throttle)
	}
	if diff := svc.applied.Brake - 0.3; diff < -1e-9 || diff > 1e-9 {
		t.Fatalf("unexpected brake: %f", svc.applied.Brake)
	}
}

func TestResolveTargetTimesOutToNeutralByDefault(t *testing.T) {
	now := time.Date(2026, 4, 22, 12, 0, 0, 0, time.UTC)
	target, stale, enabled, timedOut := resolveTarget(&commandEnvelope{
		controlState: controlState{Steer: 0.2, Throttle: 0.4, Brake: 0.2},
		Enabled:      true,
		InputMode:    InputModeNormalized,
		ReceivedAt:   now.Add(-time.Second).Format(time.RFC3339Nano),
	}, 100*time.Millisecond, false, now)

	if target != (controlState{}) {
		t.Fatalf("expected neutral target, got=%+v", target)
	}
	if !stale || enabled || !timedOut {
		t.Fatalf("unexpected timeout flags stale=%v enabled=%v timedOut=%v", stale, enabled, timedOut)
	}
}

func TestResolveTargetCanHoldLastCommandWhenConfigured(t *testing.T) {
	now := time.Date(2026, 4, 22, 12, 0, 0, 0, time.UTC)
	target, stale, enabled, timedOut := resolveTarget(&commandEnvelope{
		controlState: controlState{Steer: 0.2, Throttle: 0.4, Brake: 0.2},
		Enabled:      true,
		InputMode:    InputModeNormalized,
		ReceivedAt:   now.Add(-time.Second).Format(time.RFC3339Nano),
	}, 100*time.Millisecond, true, now)

	if target.Throttle != 0.4 || target.Brake != 0.2 {
		t.Fatalf("expected held throttle, got=%+v", target)
	}
	if !stale || !enabled || timedOut {
		t.Fatalf("unexpected hold flags stale=%v enabled=%v timedOut=%v", stale, enabled, timedOut)
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
	svc.lastCmd = &commandEnvelope{
		controlState: controlState{Steer: 0.2, Brake: 0.1},
		Enabled:      true,
		InputMode:    InputModeNormalized,
		ReceivedAt:   base.Format(time.RFC3339Nano),
	}

	err := svc.step(base.Add(10 * time.Millisecond))
	if !errors.Is(err, controllerErr) {
		t.Fatalf("expected controller error, got=%v", err)
	}
	if svc.lastApplyError == "" {
		t.Fatal("expected last apply error to be recorded")
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

func TestApplyResetAndSaveTuning(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/train_config.toml"
	content := []byte("[backend.actuator]\nsteering_gain = 1.0\nthrottle_gain = 1.0\nthrottle_floor = 0.0\n")
	if err := os.WriteFile(path, content, 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	svc := NewService(DefaultConfig(), path)
	next := Tuning{
		SteeringGain:  1.4,
		ThrottleGain:  1.8,
		ThrottleFloor: 0.22,
	}
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
	if loaded.SteeringGain != next.SteeringGain || loaded.ThrottleGain != next.ThrottleGain || loaded.ThrottleFloor != next.ThrottleFloor {
		t.Fatalf("unexpected persisted config: %+v", loaded)
	}
}
