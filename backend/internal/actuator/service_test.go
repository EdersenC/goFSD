package actuator

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
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

func TestStepAppliesDeadzoneScaleAndBrakePriority(t *testing.T) {
	cfg := DefaultConfig()
	cfg.StaleTimeout = time.Second
	svc := NewService(cfg, "")
	svc.controller = &fakeController{}
	svc.ready = true
	svc.supported = true
	svc.lastTickAt = time.Date(2026, 4, 16, 12, 0, 0, 0, time.UTC)
	svc.lastCmd = &commandEnvelope{
		controlState: controlState{
			Steer:    0.6,
			Throttle: 0.7,
			Brake:    0.3,
		},
		Enabled:    true,
		ReceivedAt: svc.lastTickAt.Format(time.RFC3339Nano),
	}

	if err := svc.step(svc.lastTickAt.Add(time.Second)); err != nil {
		t.Fatalf("step returned error: %v", err)
	}

	wantSteer := 0.6
	if got := svc.target.Steer; got != wantSteer {
		t.Fatalf("unexpected target steer: got=%f want=%f", got, wantSteer)
	}
	if got := svc.applied.Steer; got != wantSteer {
		t.Fatalf("unexpected steer: got=%f want=%f", got, wantSteer)
	}
	if svc.applied.Throttle != 0 {
		t.Fatalf("expected throttle to be zero when brake is present, got=%f", svc.applied.Throttle)
	}
	wantBrake := 0.3 * cfg.BrakeInputGain
	if svc.applied.Brake != wantBrake {
		t.Fatalf("unexpected brake: got=%f want=%f", svc.applied.Brake, wantBrake)
	}
}

func TestResolveTargetAppliesRawModelScalingBeforeControllerGains(t *testing.T) {
	now := time.Date(2026, 4, 16, 12, 0, 0, 0, time.UTC)
	tuning := DefaultConfig().Tuning()
	tuning.ModelSteerScale = 8
	tuning.ModelAccelScale = 4
	tuning.SteerInputGain = 3
	tuning.ThrottleInputGain = 2

	target, stale, enabled, timedOut := resolveTarget(&commandEnvelope{
		controlState: controlState{
			Steer:    0.05,
			Throttle: 0.1,
			Brake:    0.0,
		},
		InputMode:  InputModeModelRaw,
		Enabled:    true,
		ReceivedAt: now.Format(time.RFC3339Nano),
	}, tuning, time.Second, true, now)

	if stale {
		t.Fatal("expected target to stay fresh")
	}
	if timedOut {
		t.Fatal("expected target to stay active")
	}
	if !enabled {
		t.Fatal("expected target to stay enabled")
	}
	if target.Steer <= 0.4 {
		t.Fatalf("expected raw steer scaling and gain to produce a visible target, got=%f", target.Steer)
	}
	if target.Throttle <= 0.7 {
		t.Fatalf("expected raw accel scaling and gain to produce a visible throttle target, got=%f", target.Throttle)
	}
}

func TestStepReturnsToNeutralWhenCommandGoesStale(t *testing.T) {
	cfg := DefaultConfig()
	cfg.StaleTimeout = 100 * time.Millisecond
	cfg.SteerRatePerSecond = 10
	cfg.ThrottleRatePerSec = 10
	cfg.BrakeRatePerSecond = 10
	svc := NewService(cfg, "")
	svc.controller = &fakeController{}
	svc.ready = true
	svc.supported = true
	base := time.Date(2026, 4, 16, 12, 0, 0, 0, time.UTC)
	svc.lastTickAt = base
	svc.lastCmd = &commandEnvelope{
		controlState: controlState{
			Steer:    0.8,
			Throttle: 0.5,
		},
		Enabled:    true,
		ReceivedAt: base.Format(time.RFC3339Nano),
	}

	if err := svc.step(base.Add(50 * time.Millisecond)); err != nil {
		t.Fatalf("warm step returned error: %v", err)
	}
	if svc.applied.Stale {
		t.Fatal("expected fresh command before timeout")
	}

	if err := svc.step(base.Add(250 * time.Millisecond)); err != nil {
		t.Fatalf("stale step returned error: %v", err)
	}
	if !svc.applied.Stale {
		t.Fatal("expected command to become stale")
	}
	if !svc.applied.TimedOut {
		t.Fatal("expected stale command to enter timeout neutralization")
	}
	if svc.applied.Throttle >= 0.5 {
		t.Fatalf("expected throttle to ramp down after timeout, got=%f", svc.applied.Throttle)
	}
}

func TestStepHoldsLastCommandWhenConfigured(t *testing.T) {
	cfg := DefaultConfig()
	cfg.HoldLastCommand = true
	cfg.StaleTimeout = 100 * time.Millisecond
	cfg.SteerRatePerSecond = 10
	cfg.ThrottleRatePerSec = 10
	cfg.BrakeRatePerSecond = 10
	svc := NewService(cfg, "")
	svc.controller = &fakeController{}
	svc.ready = true
	svc.supported = true
	base := time.Date(2026, 4, 16, 12, 0, 0, 0, time.UTC)
	svc.lastTickAt = base
	svc.lastCmd = &commandEnvelope{
		controlState: controlState{
			Steer:    0.8,
			Throttle: 0.5,
		},
		Enabled:    true,
		ReceivedAt: base.Format(time.RFC3339Nano),
	}

	if err := svc.step(base.Add(50 * time.Millisecond)); err != nil {
		t.Fatalf("warm step returned error: %v", err)
	}
	if err := svc.step(base.Add(250 * time.Millisecond)); err != nil {
		t.Fatalf("stale hold step returned error: %v", err)
	}
	if !svc.applied.Stale {
		t.Fatal("expected command to be marked stale while still held")
	}
	if svc.applied.TimedOut {
		t.Fatal("expected legacy hold mode to avoid timeout neutralization")
	}
	if svc.applied.Throttle != 0.5 {
		t.Fatalf("expected throttle to stay held at 0.5, got=%f", svc.applied.Throttle)
	}
	if !svc.applied.Enabled {
		t.Fatal("expected held command to remain enabled")
	}
}

func TestDefaultConfigTimesOutToNeutralInsteadOfHoldingForever(t *testing.T) {
	cfg := DefaultConfig()
	if cfg.HoldLastCommand {
		t.Fatal("expected timeout-to-neutral to be the default actuator behavior")
	}
}

func TestStepKeepsHoldingReachedTargetBetweenUpdates(t *testing.T) {
	cfg := DefaultConfig()
	cfg.StaleTimeout = time.Second
	cfg.SteerRatePerSecond = 10
	cfg.ThrottleRatePerSec = 10
	cfg.BrakeRatePerSecond = 10
	svc := NewService(cfg, "")
	svc.controller = &fakeController{}
	svc.ready = true
	svc.supported = true
	base := time.Date(2026, 4, 16, 12, 0, 0, 0, time.UTC)
	svc.lastTickAt = base
	svc.lastCmd = &commandEnvelope{
		controlState: controlState{
			Steer:    0.3,
			Throttle: 0.2,
		},
		Enabled:    true,
		ReceivedAt: base.Format(time.RFC3339Nano),
	}

	if err := svc.step(base.Add(100 * time.Millisecond)); err != nil {
		t.Fatalf("first step returned error: %v", err)
	}
	firstApplied := svc.applied
	if !firstApplied.Holding {
		t.Fatal("expected target to be held once reached")
	}

	if err := svc.step(base.Add(200 * time.Millisecond)); err != nil {
		t.Fatalf("second step returned error: %v", err)
	}
	if !controlStateEqual(svc.applied.controlState, firstApplied.controlState) {
		t.Fatalf("expected controller state to remain latched between updates, got=%+v want=%+v", svc.applied.controlState, firstApplied.controlState)
	}
	if svc.applied.Holding != firstApplied.Holding || svc.applied.TimedOut != firstApplied.TimedOut || svc.applied.Enabled != firstApplied.Enabled || svc.applied.Stale != firstApplied.Stale {
		t.Fatalf("expected latched flags to remain stable between updates, got=%+v want=%+v", svc.applied, firstApplied)
	}
}

func TestSubmitDefaultsEnabledToTrue(t *testing.T) {
	cfg := DefaultConfig()
	cfg.ModelSteerScale = 0.5
	cfg.ModelAccelScale = 0.25
	svc := NewService(cfg, "")
	svc.supported = true
	svc.ready = true
	svc.controller = &fakeController{}

	state, err := svc.Submit(CommandRequest{Steer: 2, Throttle: 1.5, Brake: -1})
	if err != nil {
		t.Fatalf("submit returned error: %v", err)
	}
	if state.LastCommand == nil {
		t.Fatal("expected last command to be recorded")
	}
	if !state.LastCommand.Enabled {
		t.Fatal("expected enabled to default to true")
	}
	if state.LastCommand.Steer != 2 {
		t.Fatalf("expected raw steer request to remain unchanged, got=%f", state.LastCommand.Steer)
	}
	if state.LastCommand.Throttle != 1.5 {
		t.Fatalf("expected raw throttle request to remain unchanged, got=%f", state.LastCommand.Throttle)
	}
	if state.LastCommand.Brake != -1 {
		t.Fatalf("expected raw brake request to remain unchanged, got=%f", state.LastCommand.Brake)
	}
	if state.LastCommand.InputMode != InputModeModelRaw {
		t.Fatalf("expected default input mode %q, got=%q", InputModeModelRaw, state.LastCommand.InputMode)
	}
}

func TestSubmitNormalizedModePreservesExistingContract(t *testing.T) {
	cfg := DefaultConfig()
	cfg.ModelSteerScale = 8
	cfg.ModelAccelScale = 4
	svc := NewService(cfg, "")
	svc.supported = true
	svc.ready = true
	svc.controller = &fakeController{}

	state, err := svc.Submit(CommandRequest{
		Steer:     2,
		Throttle:  1.5,
		Brake:     -1,
		InputMode: InputModeNormalized,
	})
	if err != nil {
		t.Fatalf("submit returned error: %v", err)
	}
	if state.LastCommand == nil {
		t.Fatal("expected last command to be recorded")
	}
	if state.LastCommand.Steer != 2 {
		t.Fatalf("expected normalized steer request to remain unchanged, got=%f", state.LastCommand.Steer)
	}
	if state.LastCommand.Throttle != 1.5 {
		t.Fatalf("expected normalized throttle request to remain unchanged, got=%f", state.LastCommand.Throttle)
	}
	if state.LastCommand.Brake != -1 {
		t.Fatalf("expected normalized brake request to remain unchanged, got=%f", state.LastCommand.Brake)
	}
	if state.LastCommand.InputMode != InputModeNormalized {
		t.Fatalf("expected input mode %q, got=%q", InputModeNormalized, state.LastCommand.InputMode)
	}
}

func TestSubmitRejectsUnknownInputMode(t *testing.T) {
	cfg := DefaultConfig()
	svc := NewService(cfg, "")
	svc.supported = true
	svc.ready = true
	svc.controller = &fakeController{}

	if _, err := svc.Submit(CommandRequest{InputMode: "mystery"}); err == nil {
		t.Fatal("expected invalid input mode error")
	} else if !strings.Contains(err.Error(), "invalid actuator input mode") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestStepDoesNotAdvanceAppliedOnControllerError(t *testing.T) {
	cfg := DefaultConfig()
	cfg.StaleTimeout = time.Second
	controllerErr := errors.New("controller update failed")
	svc := NewService(cfg, "")
	svc.controller = &fakeController{err: controllerErr}
	svc.ready = true
	svc.supported = true
	svc.lastTickAt = time.Date(2026, 4, 16, 12, 0, 0, 0, time.UTC)
	svc.applied = AppliedState{
		controlState: controlState{Steer: 0.1},
		Enabled:      true,
		Stale:        false,
		UpdatedAt:    svc.lastTickAt.Format(time.RFC3339Nano),
	}
	svc.lastCmd = &commandEnvelope{
		controlState: controlState{
			Steer: 1.0,
		},
		Enabled:    true,
		ReceivedAt: svc.lastTickAt.Format(time.RFC3339Nano),
	}

	err := svc.step(svc.lastTickAt.Add(100 * time.Millisecond))
	if !errors.Is(err, controllerErr) {
		t.Fatalf("expected controller error, got=%v", err)
	}
	if svc.applied.Steer != 0.1 {
		t.Fatalf("expected applied steer to remain at last successful value, got=%f", svc.applied.Steer)
	}
	if svc.target.Steer == 0 {
		t.Fatalf("expected target to be updated despite controller error, got=%f", svc.target.Steer)
	}
	if svc.lastApplyError == "" {
		t.Fatal("expected lastApplyError to be recorded")
	}
	if svc.lastApplySucceededAt != "" {
		t.Fatalf("expected no success timestamp after failed apply, got=%q", svc.lastApplySucceededAt)
	}
}

func TestApplyTuningChangesLiveValues(t *testing.T) {
	cfg := DefaultConfig()
	svc := NewService(cfg, "")

	next := Tuning{
		SteerDeadzone:      0.03,
		MaxSteerScale:      0.9,
		SteerInputGain:     5.0,
		ThrottleInputGain:  1.8,
		BrakeInputGain:     2.2,
		ModelSteerScale:    6.5,
		ModelAccelScale:    3.1,
		SteerRatePerSecond: 3.5,
		ThrottleRatePerSec: 1.5,
		BrakeRatePerSecond: 4.5,
	}

	state, err := svc.ApplyTuning(next)
	if err != nil {
		t.Fatalf("ApplyTuning returned error: %v", err)
	}
	if state.Live.SteerInputGain != 5.0 {
		t.Fatalf("expected live steer gain to update, got=%f", state.Live.SteerInputGain)
	}
	if state.Live.ModelSteerScale != 6.5 {
		t.Fatalf("expected live model steer scale to update, got=%f", state.Live.ModelSteerScale)
	}
	if state.Saved.SteerInputGain == state.Live.SteerInputGain {
		t.Fatal("expected saved tuning to remain unchanged until SaveTuning")
	}
}

func TestResetTuningRestoresSavedValues(t *testing.T) {
	cfg := DefaultConfig()
	svc := NewService(cfg, "")

	_, err := svc.ApplyTuning(Tuning{
		SteerDeadzone:      0.04,
		MaxSteerScale:      0.95,
		SteerInputGain:     6.0,
		ThrottleInputGain:  2.0,
		BrakeInputGain:     2.0,
		ModelSteerScale:    8.0,
		ModelAccelScale:    4.0,
		SteerRatePerSecond: 2.0,
		ThrottleRatePerSec: 2.0,
		BrakeRatePerSecond: 2.0,
	})
	if err != nil {
		t.Fatalf("ApplyTuning returned error: %v", err)
	}

	state := svc.ResetTuning()
	if state.Live.SteerInputGain != cfg.SteerInputGain {
		t.Fatalf("expected reset steer gain=%f, got=%f", cfg.SteerInputGain, state.Live.SteerInputGain)
	}
}

func TestSaveTuningWritesConfigAndPreservesUnrelatedSections(t *testing.T) {
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "train_config.toml")
	raw := `
[dataset]
data_root = "S:\\fsd_fivem_data"

[backend.inference]
fps = 30

[backend.actuator]
steer_deadzone = 0.02
max_steer_scale = 1.0
steer_input_gain = 4.0
throttle_input_gain = 2.0
brake_input_gain = 2.0
model_steer_scale = 8.0
model_accel_scale = 4.0
steer_rate_per_second = 2.5
throttle_rate_per_second = 1.3
brake_rate_per_second = 3.0
`
	if err := os.WriteFile(configPath, []byte(strings.TrimSpace(raw)), 0644); err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}

	cfg := DefaultConfig()
	svc := NewService(cfg, configPath)
	_, err := svc.ApplyTuning(Tuning{
		SteerDeadzone:      0.05,
		MaxSteerScale:      0.75,
		SteerInputGain:     7.0,
		ThrottleInputGain:  1.6,
		BrakeInputGain:     1.4,
		ModelSteerScale:    9.0,
		ModelAccelScale:    4.5,
		SteerRatePerSecond: 4.0,
		ThrottleRatePerSec: 1.7,
		BrakeRatePerSecond: 4.2,
	})
	if err != nil {
		t.Fatalf("ApplyTuning returned error: %v", err)
	}

	state, err := svc.SaveTuning()
	if err != nil {
		t.Fatalf("SaveTuning returned error: %v", err)
	}
	if state.Saved.SteerInputGain != 7.0 {
		t.Fatalf("expected saved steer gain to update, got=%f", state.Saved.SteerInputGain)
	}

	updated, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("ReadFile returned error: %v", err)
	}
	contents := string(updated)
	for _, fragment := range []string{
		"[dataset]",
		`data_root = 'S:\fsd_fivem_data'`,
		"[backend.inference]",
		"steer_input_gain = 7.0",
		"model_steer_scale = 9.0",
		"model_accel_scale = 4.5",
		"steer_deadzone = 0.05",
		"brake_rate_per_second = 4.2",
	} {
		if !strings.Contains(contents, fragment) {
			t.Fatalf("expected config to contain %q, got:\n%s", fragment, contents)
		}
	}
	if strings.Contains(contents, "hold_last_command") {
		t.Fatalf("expected SaveTuning to avoid re-introducing legacy hold_last_command, got:\n%s", contents)
	}
}
