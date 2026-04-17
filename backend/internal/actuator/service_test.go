package actuator

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

type fakeController struct {
	applied []controlState
}

func (f *fakeController) Apply(next controlState) error {
	f.applied = append(f.applied, next)
	return nil
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

	wantSteer := cfg.MaxSteerScale
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
	if svc.applied.Throttle >= 0.5 {
		t.Fatalf("expected throttle to ramp down after timeout, got=%f", svc.applied.Throttle)
	}
}

func TestSubmitDefaultsEnabledToTrue(t *testing.T) {
	cfg := DefaultConfig()
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
	if state.LastCommand.Steer != 1 {
		t.Fatalf("expected steer to clamp to 1, got=%f", state.LastCommand.Steer)
	}
	if state.LastCommand.Throttle != 1 {
		t.Fatalf("expected throttle to clamp to 1, got=%f", state.LastCommand.Throttle)
	}
	if state.LastCommand.Brake != 0 {
		t.Fatalf("expected brake to clamp to 0, got=%f", state.LastCommand.Brake)
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
		"steer_deadzone = 0.05",
		"brake_rate_per_second = 4.2",
	} {
		if !strings.Contains(contents, fragment) {
			t.Fatalf("expected config to contain %q, got:\n%s", fragment, contents)
		}
	}
}
