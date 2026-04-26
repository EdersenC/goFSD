package actuator

import "testing"

func TestProcessActuatorCommandClampsNegativeThrottleAndSteeringRange(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Profile = "debug_raw"
	applyActuatorProfile(&cfg, cfg.Profile)

	final, debug := ProcessActuatorCommand(ControlCommand{
		Steering:         1.4,
		Throttle:         -0.2,
		BrakePressureAvg: 1.6,
	}, &ProcessorState{}, cfg)

	if final.Steering != 1.0 || final.Throttle != 0.0 || final.BrakePressureAvg != 1.0 {
		t.Fatalf("unexpected final command: %+v", final)
	}
	if !debug.ClampApplied {
		t.Fatalf("expected clamp to be applied, got %+v", debug)
	}
}

func TestProcessActuatorCommandAppliesDeadzoneRateLimitAndSmoothing(t *testing.T) {
	cfg := DefaultConfig()
	state := &ProcessorState{
		PrevSteering: 0.50,
		PrevThrottle: 0.50,
		PrevBrake:    0.10,
	}

	final, debug := ProcessActuatorCommand(ControlCommand{
		Steering:         0.01,
		Throttle:         0.90,
		BrakePressureAvg: 0.00,
	}, state, cfg)

	if !debug.DeadzoneApplied {
		t.Fatalf("expected steering deadzone to be applied, got %+v", debug)
	}
	if !debug.RateLimitApplied {
		t.Fatalf("expected rate limit to be applied, got %+v", debug)
	}
	if !debug.SmoothingApplied {
		t.Fatalf("expected smoothing to be applied, got %+v", debug)
	}
	if final.Steering >= 0.50 {
		t.Fatalf("expected steering to move back toward zero, got %+v", final)
	}
	if final.Throttle <= 0.50 || final.Throttle >= 0.54 {
		t.Fatalf("expected smoothed throttle near 0.508, got %+v", final)
	}
	if final.BrakePressureAvg >= 0.10 {
		t.Fatalf("expected brake to smooth toward zero, got %+v", final)
	}
}

func TestProcessActuatorCommandDebugRawBypassesDeadzoneRateLimitAndSmoothing(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Profile = "debug_raw"
	applyActuatorProfile(&cfg, cfg.Profile)
	state := &ProcessorState{
		PrevSteering: 0.25,
		PrevThrottle: 0.25,
		PrevBrake:    0.25,
	}

	final, debug := ProcessActuatorCommand(ControlCommand{
		Steering:         0.01,
		Throttle:         0.80,
		BrakePressureAvg: 0.40,
	}, state, cfg)

	if final.Steering != 0.01 || final.Throttle != 0.80 || final.BrakePressureAvg != 0.40 {
		t.Fatalf("unexpected debug_raw final command: %+v", final)
	}
	if debug.ConflictApplied || debug.DeadzoneApplied || debug.RateLimitApplied || debug.SmoothingApplied {
		t.Fatalf("debug_raw should bypass processing, got %+v", debug)
	}
}

func TestApplyFallbackDecayDecaysThrottleFasterThanSteering(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Profile = "debug_raw"
	applyActuatorProfile(&cfg, cfg.Profile)
	cfg.FallbackDecayEnabled = true
	cfg.FallbackSteeringDecay = 0.85
	cfg.FallbackThrottleDecay = 0.65
	cfg.FallbackBrakeDecay = 0.65
	state := &ProcessorState{
		PrevSteering: 0.80,
		PrevThrottle: 0.80,
		PrevBrake:    0.60,
	}

	final, debug := ApplyFallbackDecay(state, cfg)

	if !debug.FallbackApplied {
		t.Fatalf("expected fallback flag, got %+v", debug)
	}
	if final.Steering != 0.68 || final.Throttle != 0.52 || final.BrakePressureAvg != 0.39 {
		t.Fatalf("unexpected fallback command: %+v", final)
	}
	if state.Stats.FallbackCommands != 1 {
		t.Fatalf("expected fallback counter increment, got %+v", state.Stats)
	}
}

func TestProcessActuatorCommandPreservesBrakeAxisForFinalActuatorMapping(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Profile = "debug_raw"
	applyActuatorProfile(&cfg, cfg.Profile)

	final, debug := ProcessActuatorCommand(ControlCommand{
		Steering:         0.0,
		Throttle:         0.30,
		BrakePressureAvg: 0.60,
	}, &ProcessorState{}, cfg)

	if debug.ConflictApplied {
		t.Fatalf("expected processor to leave throttle/brake arbitration to final mapping, got %+v", debug)
	}
	if final.Throttle != 0.30 || final.BrakePressureAvg != 0.60 {
		t.Fatalf("expected processor to preserve throttle/brake values, got %+v", final)
	}
}
