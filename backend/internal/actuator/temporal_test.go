package actuator

import (
	"math"
	"testing"
	"time"

	"awesomeProject/internal/control"
)

func TestSamplePredictionHorizonInterpolatesDtMs(t *testing.T) {
	plan, err := NormalizePredictionHorizon(PredictionHorizon{
		InputTimestampS:    10,
		ReceivedTimestampS: 10.05,
		Confidence:         1,
		Points: []FuturePoint{
			{DtMs: 100, Steer: floatPtr(0.0), Throttle: floatPtr(0.2), X: floatPtr(0.0), Y: floatPtr(4.0)},
			{DtMs: 200, Steer: floatPtr(1.0), Throttle: floatPtr(0.4), X: floatPtr(10.0), Y: floatPtr(8.0)},
		},
	}, 10.05)
	if err != nil {
		t.Fatalf("NormalizePredictionHorizon returned error: %v", err)
	}

	point, clamped := SamplePredictionHorizon(plan, 150)
	if clamped {
		t.Fatalf("did not expect interpolation to clamp")
	}
	if point.DtMs != 150 {
		t.Fatalf("unexpected dt_ms: %d", point.DtMs)
	}
	if point.Steer == nil || math.Abs(*point.Steer-0.5) > 1e-9 {
		t.Fatalf("unexpected interpolated steer: %+v", point.Steer)
	}
	if point.Throttle == nil || math.Abs(*point.Throttle-0.3) > 1e-9 {
		t.Fatalf("unexpected interpolated throttle: %+v", point.Throttle)
	}
	if point.X == nil || math.Abs(*point.X-5.0) > 1e-9 {
		t.Fatalf("unexpected interpolated x: %+v", point.X)
	}
}

func TestTemporalTargetDtCalculation(t *testing.T) {
	plan := PredictionHorizon{InputTimestampS: 1.0}
	now := time.Unix(1, 100*int64(time.Millisecond)).UTC()
	ageMs := ComputePlanAgeMs(plan, now)
	lookaheadMs := SpeedLookaheadMs(4)
	targetDtMs := ComputeTargetDtMs(ageMs, 50, lookaheadMs)

	if math.Abs(ageMs-100) > 1e-6 {
		t.Fatalf("unexpected plan age: %f", ageMs)
	}
	if math.Abs(lookaheadMs-350) > 1e-9 {
		t.Fatalf("unexpected lookahead: %f", lookaheadMs)
	}
	if math.Abs(targetDtMs-500) > 1e-6 {
		t.Fatalf("unexpected target dt: %f", targetDtMs)
	}
}

func TestPlanAgeUsesInputTimestampNotReceivedTimestamp(t *testing.T) {
	plan := PredictionHorizon{InputTimestampS: 10.0, ReceivedTimestampS: 99.0}
	ageMs := ComputePlanAgeMsFromSeconds(plan, 10.25)
	if math.Abs(ageMs-250) > 1e-9 {
		t.Fatalf("expected plan age to use now_s - input_timestamp_s, got=%f", ageMs)
	}
}

func TestSpeedLookaheadClamps(t *testing.T) {
	if got := SpeedLookaheadMs(0); got != 250 {
		t.Fatalf("unexpected zero-speed lookahead: %f", got)
	}
	if got := SpeedLookaheadMs(10); got != 500 {
		t.Fatalf("unexpected mid-speed lookahead: %f", got)
	}
	if got := SpeedLookaheadMs(100); got != 850 {
		t.Fatalf("unexpected high-speed lookahead: %f", got)
	}
}

func TestTemporalPlanBufferExpiredFallback(t *testing.T) {
	cfg := DefaultConfig()
	now := time.Unix(100, 0).UTC()
	buffer := TemporalPlanBuffer{
		LastControls: ControlCommand{
			Steering: 0.8,
			Throttle: 0.7,
		},
	}
	plan := PredictionHorizon{
		InputTimestampS:    timeToSeconds(now.Add(-900 * time.Millisecond)),
		ReceivedTimestampS: timeToSeconds(now.Add(-850 * time.Millisecond)),
		Confidence:         1,
		Source:             "test",
		Points: []FuturePoint{
			{DtMs: 100, Steer: floatPtr(1), Throttle: floatPtr(1)},
		},
	}
	if err := buffer.Accept(plan, now); err != nil {
		t.Fatalf("Accept returned error: %v", err)
	}
	buffer.LastControls = ControlCommand{Steering: 0.8, Throttle: 0.7}

	command, trace := buffer.Resolve(now, 10, cfg)
	if trace.PlanState != PlanStateExpired || !trace.FallbackApplied {
		t.Fatalf("expected expired fallback, got trace=%+v", trace)
	}
	if math.Abs(command.Steering) > 0.25+1e-9 {
		t.Fatalf("expected expired fallback to limit steering, got=%+v", command)
	}
	if command.Throttle != 0 {
		t.Fatalf("expected expired fallback to cut throttle when braking, got=%+v", command)
	}
	if command.BrakePressureAvg < 0.15 {
		t.Fatalf("expected light brake at high speed, got=%+v", command)
	}
}

func TestTemporalPlanBufferInvalidTelemetryFallback(t *testing.T) {
	cfg := DefaultConfig()
	now := time.Unix(100, 0).UTC()
	buffer := TemporalPlanBuffer{
		LastControls: ControlCommand{Steering: 0.8, Throttle: 0.7},
	}
	plan := PredictionHorizon{
		InputTimestampS:    timeToSeconds(now),
		ReceivedTimestampS: timeToSeconds(now),
		Confidence:         1,
		Source:             "test",
		Points: []FuturePoint{
			{DtMs: 100, Steer: floatPtr(1), Throttle: floatPtr(1)},
		},
	}
	if err := buffer.Accept(plan, now); err != nil {
		t.Fatalf("Accept returned error: %v", err)
	}

	command, trace := buffer.ActuatorTick(timeToSeconds(now), control.ActuatorEgoState{
		TimestampS:    timeToSeconds(now),
		SpeedMPS:      10,
		Valid:         false,
		InvalidReason: "speed is invalid",
	}, cfg)
	if !trace.FallbackApplied || trace.ControlMode != "fallback" || trace.TelemetryValid {
		t.Fatalf("expected invalid telemetry fallback, got command=%+v trace=%+v", command, trace)
	}
	if trace.TelemetryInvalidReason != "speed is invalid" || trace.ValidationError != "speed is invalid" {
		t.Fatalf("expected invalid reason to be logged, got=%+v", trace)
	}
	if math.Abs(command.Steering) > 0.25+1e-9 || command.BrakePressureAvg < 0.15 {
		t.Fatalf("expected safe fallback controls, got=%+v", command)
	}
}

func TestTemporalActuatorLogsTelemetryAge(t *testing.T) {
	cfg := DefaultConfig()
	nowS := 100.0
	buffer := TemporalPlanBuffer{}
	if err := buffer.Accept(PredictionHorizon{
		InputTimestampS:    nowS,
		ReceivedTimestampS: nowS,
		Confidence:         1,
		Points: []FuturePoint{
			{DtMs: 500, Steer: floatPtr(0.2), Throttle: floatPtr(0.3)},
		},
	}, time.Unix(100, 0)); err != nil {
		t.Fatalf("Accept returned error: %v", err)
	}

	_, trace := buffer.ActuatorTick(nowS, control.ActuatorEgoState{
		TimestampS: nowS - 0.120,
		SpeedMPS:   4,
		Valid:      true,
	}, cfg)
	if math.Abs(trace.TelemetryAgeMs-120) > 1e-6 {
		t.Fatalf("unexpected telemetry age: %+v", trace)
	}
}

func TestControlSmootherRateLimits(t *testing.T) {
	cfg := DefaultConfig()
	smoother := NewControlSmoother(cfg)

	command := smoother.Smooth(ControlCommand{Steering: 1, Throttle: 1}, ControlCommand{})
	if math.Abs(command.Steering-cfg.TemporalSteeringMaxDelta) > 1e-9 {
		t.Fatalf("unexpected steering rate limit: %+v", command)
	}
	if math.Abs(command.Throttle-cfg.TemporalThrottleMaxDelta) > 1e-9 {
		t.Fatalf("unexpected throttle rate limit: %+v", command)
	}

	command = smoother.Smooth(ControlCommand{BrakePressureAvg: 1}, ControlCommand{})
	if math.Abs(command.BrakePressureAvg-cfg.TemporalBrakeMaxDelta) > 1e-9 {
		t.Fatalf("unexpected brake rate limit: %+v", command)
	}
}

func TestControlSmootherThrottleBrakeExclusivity(t *testing.T) {
	command := NewControlSmoother(DefaultConfig()).Smooth(ControlCommand{
		Throttle:         0.8,
		BrakePressureAvg: 0.7,
	}, ControlCommand{})

	if command.Throttle != 0 {
		t.Fatalf("expected brake priority to suppress throttle, got=%+v", command)
	}
	if command.BrakePressureAvg <= 0 {
		t.Fatalf("expected brake to remain active, got=%+v", command)
	}
}

func TestLegacyPredictionAdapterRepeatsImmediateControl(t *testing.T) {
	desiredSpeed := 6.5
	plan := LegacyPredictionAdapter(ControlCommand{
		Steering:         0.2,
		Throttle:         0.3,
		BrakePressureAvg: 0.4,
	}, 10, 10.05, &desiredSpeed, "legacy-test")

	normalized, err := NormalizePredictionHorizon(plan, 10.05)
	if err != nil {
		t.Fatalf("NormalizePredictionHorizon returned error: %v", err)
	}
	if len(normalized.Points) != len(DefaultPredictionHorizonDtMs) {
		t.Fatalf("unexpected point count: %d", len(normalized.Points))
	}
	for index, point := range normalized.Points {
		if point.DtMs != DefaultPredictionHorizonDtMs[index] {
			t.Fatalf("unexpected dt at %d: %d", index, point.DtMs)
		}
		if point.Steer == nil || *point.Steer != 0.2 || point.Throttle == nil || *point.Throttle != 0.3 || point.Brake == nil || *point.Brake != 0.4 {
			t.Fatalf("unexpected repeated control at %d: %+v", index, point)
		}
		if point.DesiredSpeedMPS == nil || *point.DesiredSpeedMPS != desiredSpeed {
			t.Fatalf("unexpected repeated desired speed at %d: %+v", index, point)
		}
	}
}
