package stopsigncontrol

import (
	"encoding/json"
	"errors"
	"math"
	"strings"
	"testing"
)

func validPlan() Plan {
	return Plan{
		Contract:    StopSignMotionPlanContractV1,
		SampledAtS:  100,
		ReceivedAtS: 100.025,
		Direction:   StopSignDirectionForward,
		Points: []Setpoint{
			{DtMs: 100, DesiredWheelSteerNormalized: -1, DesiredSpeedMPS: 0, StopProbability: 0},
			{DtMs: 300, DesiredWheelSteerNormalized: 1, DesiredSpeedMPS: 2, StopProbability: 0.8},
		},
	}
}

func TestValidatePlanAcceptsExactForwardV1Contract(t *testing.T) {
	if err := ValidatePlan(validPlan()); err != nil {
		t.Fatalf("ValidatePlan returned error: %v", err)
	}
}

func TestValidatePlanRejectsContractTimestampOrderingAndPointBounds(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*Plan)
	}{
		{name: "wrong contract", mutate: func(plan *Plan) { plan.Contract = "stop_sign_motion_plan_v2" }},
		{name: "wrong direction", mutate: func(plan *Plan) { plan.Direction = "reverse" }},
		{name: "nonfinite sampled time", mutate: func(plan *Plan) { plan.SampledAtS = math.NaN() }},
		{name: "nonpositive sampled time", mutate: func(plan *Plan) { plan.SampledAtS = 0 }},
		{name: "nonfinite received time", mutate: func(plan *Plan) { plan.ReceivedAtS = math.Inf(1) }},
		{name: "receipt before sample", mutate: func(plan *Plan) { plan.ReceivedAtS = plan.SampledAtS - 0.001 }},
		{name: "empty points", mutate: func(plan *Plan) { plan.Points = nil }},
		{name: "nonpositive first dt", mutate: func(plan *Plan) { plan.Points[0].DtMs = 0 }},
		{name: "duplicate dt", mutate: func(plan *Plan) { plan.Points[1].DtMs = plan.Points[0].DtMs }},
		{name: "decreasing dt", mutate: func(plan *Plan) { plan.Points[1].DtMs = plan.Points[0].DtMs - 1 }},
		{name: "nonfinite steer", mutate: func(plan *Plan) { plan.Points[0].DesiredWheelSteerNormalized = math.NaN() }},
		{name: "steer out of range", mutate: func(plan *Plan) { plan.Points[0].DesiredWheelSteerNormalized = 1.01 }},
		{name: "negative speed", mutate: func(plan *Plan) { plan.Points[0].DesiredSpeedMPS = -0.01 }},
		{name: "speed above contract maximum", mutate: func(plan *Plan) { plan.Points[0].DesiredSpeedMPS = StopSignMotionPlanMaxSpeedMPS + 0.001 }},
		{name: "nonfinite stop probability", mutate: func(plan *Plan) { plan.Points[0].StopProbability = math.Inf(-1) }},
		{name: "stop probability out of range", mutate: func(plan *Plan) { plan.Points[0].StopProbability = 1.01 }},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			plan := validPlan()
			tc.mutate(&plan)
			if err := ValidatePlan(plan); err == nil || !errors.Is(err, ErrInvalidPlan) {
				t.Fatalf("expected ErrInvalidPlan, got=%v", err)
			}
		})
	}
}

func TestPlanSamplerInterpolatesByAgeAndClampsEndpoints(t *testing.T) {
	sampler, err := NewPlanSampler(validPlan())
	if err != nil {
		t.Fatalf("NewPlanSampler returned error: %v", err)
	}

	tests := []struct {
		name string
		age  float64
		want Setpoint
	}{
		{name: "before horizon", age: 0, want: validPlan().Points[0]},
		{name: "first point", age: 100, want: validPlan().Points[0]},
		{name: "interpolated", age: 200, want: Setpoint{DtMs: 200, DesiredWheelSteerNormalized: 0, DesiredSpeedMPS: 1, StopProbability: 0.4}},
		{name: "last point", age: 300, want: validPlan().Points[1]},
		{name: "after horizon", age: 500, want: validPlan().Points[1]},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := sampler.SampleByAge(tc.age)
			if err != nil {
				t.Fatalf("SampleByAge returned error: %v", err)
			}
			if got.DtMs != tc.want.DtMs ||
				math.Abs(got.DesiredWheelSteerNormalized-tc.want.DesiredWheelSteerNormalized) > 1e-12 ||
				math.Abs(got.DesiredSpeedMPS-tc.want.DesiredSpeedMPS) > 1e-12 ||
				math.Abs(got.StopProbability-tc.want.StopProbability) > 1e-12 {
				t.Fatalf("unexpected sample: got=%+v want=%+v", got, tc.want)
			}
		})
	}
}

func TestPlanSamplerOwnsPlanCopy(t *testing.T) {
	plan := validPlan()
	sampler, err := NewPlanSampler(plan)
	if err != nil {
		t.Fatalf("NewPlanSampler returned error: %v", err)
	}
	plan.Points[0].DesiredSpeedMPS = StopSignMotionPlanMaxSpeedMPS

	got, err := sampler.SampleByAge(0)
	if err != nil {
		t.Fatalf("SampleByAge returned error: %v", err)
	}
	if got.DesiredSpeedMPS != 0 {
		t.Fatalf("expected sampler to retain an immutable point copy, got=%+v", got)
	}
}

func TestPlanSamplerRejectsInvalidAgeAndUninitializedSampler(t *testing.T) {
	sampler, err := NewPlanSampler(validPlan())
	if err != nil {
		t.Fatalf("NewPlanSampler returned error: %v", err)
	}
	for _, age := range []float64{-1, math.NaN(), math.Inf(1)} {
		if _, err := sampler.SampleByAge(age); err == nil || !errors.Is(err, ErrInvalidSampleAge) {
			t.Fatalf("expected age=%v to return ErrInvalidSampleAge, got=%v", age, err)
		}
	}
	var nilSampler *PlanSampler
	if _, err := nilSampler.SampleByAge(0); err == nil || !errors.Is(err, ErrInvalidSampleAge) {
		t.Fatalf("expected nil sampler to return ErrInvalidSampleAge, got=%v", err)
	}
}

func TestPlanJSONUsesVersionedSnakeCaseWireNames(t *testing.T) {
	encoded, err := json.Marshal(validPlan())
	if err != nil {
		t.Fatalf("json.Marshal returned error: %v", err)
	}
	wire := string(encoded)
	for _, field := range []string{
		`"sampled_at_s"`,
		`"received_at_s"`,
		`"dt_ms"`,
		`"desired_wheel_steer_normalized"`,
		`"desired_speed_mps"`,
		`"stop_probability"`,
	} {
		if !strings.Contains(wire, field) {
			t.Fatalf("expected JSON field %s in %s", field, wire)
		}
	}
}
