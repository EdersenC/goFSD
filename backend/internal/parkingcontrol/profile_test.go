package parkingcontrol

import (
	"errors"
	"math"
	"testing"
)

func TestSteeringProfileInterpolatesCalibratedSegments(t *testing.T) {
	profile, err := newSteeringProfile([]SteeringCalibrationPoint{
		{WheelSteer: -1, Command: -0.9},
		{WheelSteer: -0.25, Command: -0.1},
		{WheelSteer: 0, Command: 0.05},
		{WheelSteer: 1, Command: 0.8},
	})
	if err != nil {
		t.Fatalf("newSteeringProfile returned error: %v", err)
	}
	if got := profile.commandFor(-0.625); math.Abs(got-(-0.5)) > 1e-9 {
		t.Fatalf("unexpected left-segment interpolation: got=%f want=-0.5", got)
	}
	if got := profile.commandFor(0.5); math.Abs(got-0.425) > 1e-9 {
		t.Fatalf("unexpected right-segment interpolation: got=%f want=0.425", got)
	}
}

func TestSteeringProfileRejectsUnsafeCalibration(t *testing.T) {
	tests := []struct {
		name   string
		points []SteeringCalibrationPoint
	}{
		{name: "too few", points: []SteeringCalibrationPoint{{WheelSteer: -1, Command: -1}}},
		{name: "incomplete range", points: []SteeringCalibrationPoint{{WheelSteer: -0.5, Command: -1}, {WheelSteer: 1, Command: 1}}},
		{name: "unordered wheel", points: []SteeringCalibrationPoint{{WheelSteer: -1, Command: -1}, {WheelSteer: -1, Command: 0}, {WheelSteer: 1, Command: 1}}},
		{name: "nonmonotonic command", points: []SteeringCalibrationPoint{{WheelSteer: -1, Command: -1}, {WheelSteer: 0, Command: 0.2}, {WheelSteer: 1, Command: 0.1}}},
		{name: "nonfinite", points: []SteeringCalibrationPoint{{WheelSteer: -1, Command: -1}, {WheelSteer: 1, Command: math.NaN()}}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := newSteeringProfile(tc.points)
			if err == nil || !errors.Is(err, ErrInvalidConfig) {
				t.Fatalf("expected invalid calibration error, got=%v", err)
			}
		})
	}
}
