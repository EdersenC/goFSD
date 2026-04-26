package translation

import (
	"os"
	"strings"
	"testing"
	"time"

	"awesomeProject/internal/actuator"
)

type fakeActuator struct {
	commands []actuator.CommandRequest
	err      error
}

func (f *fakeActuator) Submit(req actuator.CommandRequest) (actuator.State, error) {
	f.commands = append(f.commands, req)
	return actuator.State{}, f.err
}

func TestSubmitTransitionsThroughLaunchTrackAndHold(t *testing.T) {
	sink := &fakeActuator{}
	svc := NewService(DefaultConfig(), "", sink)
	svc.nowFunc = func() time.Time { return time.Date(2026, 4, 22, 12, 0, 0, 0, time.UTC) }

	launch := Sample{
		Sequence:     1,
		Telemetry:    Telemetry{CurrentSpeedMPS: 0.2},
		Lateral:      LateralIntent{HeadingErrorDeg: 10, Confidence: 0.8},
		Longitudinal: LongitudinalIntent{TargetSpeedMPS: 6, TargetAccelMPS2: 0.6, Confidence: 0.8},
		Motion:       MotionIntent{Intent: MotionIntentMove, Confidence: 0.9},
	}
	state, err := svc.Submit(launch)
	if err != nil {
		t.Fatalf("Submit launch returned error: %v", err)
	}
	if state.Mode != ModeLaunch {
		t.Fatalf("expected launch mode, got=%s", state.Mode)
	}
	if state.LastTrace == nil || state.LastTrace.Command.Throttle <= 0 {
		t.Fatalf("expected launch throttle, trace=%+v", state.LastTrace)
	}

	track := launch
	track.Sequence = 2
	track.Telemetry.CurrentSpeedMPS = 4.2
	state, err = svc.Submit(track)
	if err != nil {
		t.Fatalf("Submit track returned error: %v", err)
	}
	if state.Mode != ModeTrack {
		t.Fatalf("expected track mode, got=%s", state.Mode)
	}

	hold := track
	hold.Sequence = 3
	hold.Longitudinal.TargetSpeedMPS = 2.0
	hold.Longitudinal.TargetAccelMPS2 = 0.0
	hold.Motion = MotionIntent{Intent: MotionIntentHold, Confidence: 0.1}
	hold.Telemetry.CurrentSpeedMPS = 2.0
	state, err = svc.Submit(hold)
	if err != nil {
		t.Fatalf("Submit hold returned error: %v", err)
	}
	if state.Mode != ModeHold {
		t.Fatalf("expected hold mode, got=%s", state.Mode)
	}
	if state.LastTrace.Command.Throttle != 0 {
		t.Fatalf("expected hold command to be neutral, got=%+v", state.LastTrace.Command)
	}
}

func TestSubmitUsesStopIntentToCoast(t *testing.T) {
	sink := &fakeActuator{}
	svc := NewService(DefaultConfig(), "", sink)
	state, err := svc.Submit(Sample{
		Sequence:     1,
		Telemetry:    Telemetry{CurrentSpeedMPS: 3.0},
		Lateral:      LateralIntent{HeadingErrorDeg: 0, Confidence: 1},
		Longitudinal: LongitudinalIntent{TargetSpeedMPS: 0, TargetAccelMPS2: -0.2, Confidence: 1},
		Motion:       MotionIntent{Intent: MotionIntentStop, Confidence: 1},
	})
	if err != nil {
		t.Fatalf("Submit returned error: %v", err)
	}
	if state.Mode != ModeHold {
		t.Fatalf("expected hold mode, got=%s", state.Mode)
	}
	if state.LastTrace == nil {
		t.Fatal("expected trace")
	}
	if state.LastTrace.Command.Throttle != 0 {
		t.Fatalf("expected explicit stop to coast neutral, trace=%+v", state.LastTrace)
	}
	if !strings.Contains(strings.Join(state.LastTrace.Reasons, ","), "explicit_stop_coast") {
		t.Fatalf("expected explicit stop coast reason, got=%v", state.LastTrace.Reasons)
	}
}

func TestSubmitHoldsThrottleAcrossSmallPredictionDip(t *testing.T) {
	sink := &fakeActuator{}
	svc := NewService(DefaultConfig(), "", sink)
	now := time.Date(2026, 4, 22, 12, 0, 0, 0, time.UTC)
	svc.nowFunc = func() time.Time { return now }

	first, err := svc.Submit(Sample{
		Sequence:     1,
		Telemetry:    Telemetry{CurrentSpeedMPS: 3.2},
		Lateral:      LateralIntent{HeadingErrorDeg: 0, Confidence: 1},
		Longitudinal: LongitudinalIntent{TargetSpeedMPS: 4.0, TargetAccelMPS2: 0.25, Confidence: 1},
		Motion:       MotionIntent{Intent: MotionIntentMove, Confidence: 1},
	})
	if err != nil {
		t.Fatalf("Submit first returned error: %v", err)
	}
	if first.LastTrace == nil || first.LastTrace.Command.Throttle <= 0 {
		t.Fatalf("expected initial throttle, trace=%+v", first.LastTrace)
	}

	now = now.Add(75 * time.Millisecond)
	state, err := svc.Submit(Sample{
		Sequence:     1,
		Telemetry:    Telemetry{CurrentSpeedMPS: 3.5},
		Lateral:      LateralIntent{HeadingErrorDeg: 0, Confidence: 1},
		Longitudinal: LongitudinalIntent{TargetSpeedMPS: 3.5, TargetAccelMPS2: -0.2, Confidence: 1},
		Motion:       MotionIntent{Intent: MotionIntentMove, Confidence: 1},
	})
	if err != nil {
		t.Fatalf("Submit returned error: %v", err)
	}
	if state.Mode != ModeTrack {
		t.Fatalf("expected track mode, got=%s", state.Mode)
	}
	if state.LastTrace == nil {
		t.Fatal("expected trace")
	}
	if state.LastTrace.Command.Throttle <= 0 {
		t.Fatalf("expected throttle hold, trace=%+v", state.LastTrace)
	}
	if !strings.Contains(strings.Join(state.LastTrace.Reasons, ","), "throttle_hold") {
		t.Fatalf("expected throttle hold reason, got=%v", state.LastTrace.Reasons)
	}
}

func TestSubmitCoastsOnOverspeedWithoutBrake(t *testing.T) {
	sink := &fakeActuator{}
	svc := NewService(DefaultConfig(), "", sink)
	state, err := svc.Submit(Sample{
		Sequence:     1,
		Telemetry:    Telemetry{CurrentSpeedMPS: 6.5},
		Lateral:      LateralIntent{HeadingErrorDeg: 0, Confidence: 1},
		Longitudinal: LongitudinalIntent{TargetSpeedMPS: 4.0, TargetAccelMPS2: -0.2, Confidence: 1},
		Motion:       MotionIntent{Intent: MotionIntentMove, Confidence: 1},
	})
	if err != nil {
		t.Fatalf("Submit returned error: %v", err)
	}
	if state.Mode != ModeTrack {
		t.Fatalf("expected track mode, got=%s", state.Mode)
	}
	if state.LastTrace == nil {
		t.Fatal("expected trace")
	}
	if state.LastTrace.Command.Throttle != 0 {
		t.Fatalf("expected overspeed coast, trace=%+v", state.LastTrace)
	}
	if !strings.Contains(strings.Join(state.LastTrace.Reasons, ","), "coast_instead_of_brake") {
		t.Fatalf("expected coast reason, got=%v", state.LastTrace.Reasons)
	}
}

func TestSubmitRampsThrottleUpWhenDemandJumps(t *testing.T) {
	sink := &fakeActuator{}
	cfg := DefaultConfig()
	cfg.ThrottleGain = 10
	cfg.ThrottleRampUpPerSecond = 0.5
	svc := NewService(cfg, "", sink)
	now := time.Date(2026, 4, 22, 12, 0, 0, 0, time.UTC)
	svc.nowFunc = func() time.Time { return now }

	first, err := svc.Submit(Sample{
		Sequence:     1,
		Telemetry:    Telemetry{CurrentSpeedMPS: 1.0},
		Lateral:      LateralIntent{HeadingErrorDeg: 0, Confidence: 1},
		Longitudinal: LongitudinalIntent{TargetSpeedMPS: 6.0, TargetAccelMPS2: 1.0, Confidence: 1},
		Motion:       MotionIntent{Intent: MotionIntentMove, Confidence: 1},
	})
	if err != nil {
		t.Fatalf("Submit first returned error: %v", err)
	}
	if first.LastTrace == nil {
		t.Fatal("expected first trace")
	}
	if first.LastTrace.Command.Throttle <= 0 || first.LastTrace.Command.Throttle >= 1 {
		t.Fatalf("expected ramp-limited throttle on first step, trace=%+v", first.LastTrace)
	}
	if !strings.Contains(strings.Join(first.LastTrace.Reasons, ","), "throttle_ramp") {
		t.Fatalf("expected throttle ramp reason, got=%v", first.LastTrace.Reasons)
	}

	now = now.Add(600 * time.Millisecond)
	second, err := svc.Submit(Sample{
		Sequence:     2,
		Telemetry:    Telemetry{CurrentSpeedMPS: 1.2},
		Lateral:      LateralIntent{HeadingErrorDeg: 0, Confidence: 1},
		Longitudinal: LongitudinalIntent{TargetSpeedMPS: 6.0, TargetAccelMPS2: 1.0, Confidence: 1},
		Motion:       MotionIntent{Intent: MotionIntentMove, Confidence: 1},
	})
	if err != nil {
		t.Fatalf("Submit second returned error: %v", err)
	}
	if second.LastTrace == nil {
		t.Fatal("expected second trace")
	}
	if second.LastTrace.Command.Throttle <= first.LastTrace.Command.Throttle {
		t.Fatalf("expected throttle to ramp upward, first=%+v second=%+v", first.LastTrace.Command, second.LastTrace.Command)
	}
}

func TestSubmitGatesOnLowConfidence(t *testing.T) {
	sink := &fakeActuator{}
	svc := NewService(DefaultConfig(), "", sink)
	state, err := svc.Submit(Sample{
		Sequence:     1,
		Telemetry:    Telemetry{CurrentSpeedMPS: 2.0},
		Lateral:      LateralIntent{HeadingErrorDeg: 20, Confidence: 0.01},
		Longitudinal: LongitudinalIntent{TargetSpeedMPS: 6, TargetAccelMPS2: 0.4, Confidence: 0.01},
		Motion:       MotionIntent{Intent: MotionIntentMove, Confidence: 1.0},
	})
	if err != nil {
		t.Fatalf("Submit returned error: %v", err)
	}
	if state.LastTrace == nil {
		t.Fatal("expected trace")
	}
	if state.LastTrace.Command.Steer != 0 {
		t.Fatalf("expected steer to be gated off, got=%f", state.LastTrace.Command.Steer)
	}
	if state.LastTrace.Command.Throttle != 0 {
		t.Fatalf("expected throttle to be gated off, got=%f", state.LastTrace.Command.Throttle)
	}
	if !strings.Contains(strings.Join(state.LastTrace.Reasons, ","), "confidence_low") {
		t.Fatalf("expected low confidence reasons, got=%v", state.LastTrace.Reasons)
	}
}

func TestApplyResetAndSaveTuning(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/train_config.toml"
	content := []byte("[backend.translation]\nheading_deadband_deg = 2.5\n")
	if err := os.WriteFile(path, content, 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	svc := NewService(DefaultConfig(), path, &fakeActuator{})
	next := DefaultConfig().Tuning()
	next.TargetAccelGain = 0.9
	state, err := svc.ApplyTuning(next)
	if err != nil {
		t.Fatalf("ApplyTuning returned error: %v", err)
	}
	if state.Live.TargetAccelGain != 0.9 {
		t.Fatalf("expected live target accel gain update, got=%f", state.Live.TargetAccelGain)
	}
	reset := svc.ResetTuning()
	if reset.Live.TargetAccelGain == 0.9 {
		t.Fatalf("expected reset to restore saved tuning")
	}

	_, err = svc.ApplyTuning(next)
	if err != nil {
		t.Fatalf("ApplyTuning second returned error: %v", err)
	}
	saved, err := svc.SaveTuning()
	if err != nil {
		t.Fatalf("SaveTuning returned error: %v", err)
	}
	if saved.Saved.TargetAccelGain != 0.9 {
		t.Fatalf("expected saved target accel gain update, got=%f", saved.Saved.TargetAccelGain)
	}
}
