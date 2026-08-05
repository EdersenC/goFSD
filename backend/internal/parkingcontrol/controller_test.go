package parkingcontrol

import (
	"errors"
	"math"
	"reflect"
	"testing"
	"time"
)

func testConfig() Config {
	return Config{
		SteeringProfile: []SteeringCalibrationPoint{
			{WheelSteer: -1, Command: -0.8},
			{WheelSteer: 0, Command: 0},
			{WheelSteer: 0.5, Command: 0.4},
			{WheelSteer: 1, Command: 0.9},
		},
		SteeringFeedbackGain:            0.5,
		SteeringSlewPerSecond:           2,
		SpeedKp:                         0.4,
		SpeedKi:                         0.2,
		SpeedIntegralMin:                -2,
		SpeedIntegralMax:                2,
		MaxThrottleEffort:               0.8,
		MaxBrakeEffort:                  0.9,
		LongitudinalSlewPerSecond:       1,
		StopProbabilityThreshold:        0.7,
		StopProbabilityReleaseThreshold: 0.5,
		HoldSpeedMPS:                    0.15,
		StopBrakeEffort:                 0.8,
		MaxDesiredSpeedMPS:              5,
		MaxDT:                           500 * time.Millisecond,
	}
}

func TestControllerComposesSteeringFeedforwardFeedbackAndDTSlew(t *testing.T) {
	controller, err := New(testConfig())
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	input := Input{
		DesiredWheelSteer:  0.5,
		MeasuredWheelSteer: 0.3,
		DesiredSpeedMPS:    1,
		CurrentSpeedMPS:    1,
		DT:                 100 * time.Millisecond,
	}
	first, err := controller.Step(input)
	if err != nil {
		t.Fatalf("Step returned error: %v", err)
	}
	if math.Abs(first.Steering-0.2) > 1e-9 {
		t.Fatalf("expected 0.2 dt-scaled first steering step toward 0.5, got=%+v", first)
	}
	input.DT = 150 * time.Millisecond
	second, err := controller.Step(input)
	if err != nil {
		t.Fatalf("second Step returned error: %v", err)
	}
	if math.Abs(second.Steering-0.5) > 1e-9 {
		t.Fatalf("expected second dt-scaled step to reach calibrated target, got=%+v", second)
	}
}

func TestControllerUsesOneSignedMutuallyExclusiveLongitudinalEffort(t *testing.T) {
	controller, err := New(testConfig())
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	drive, err := controller.Step(Input{DesiredSpeedMPS: 4, DT: 100 * time.Millisecond})
	if err != nil {
		t.Fatalf("drive Step returned error: %v", err)
	}
	if math.Abs(drive.LongitudinalEffort-0.1) > 1e-9 {
		t.Fatalf("expected dt-scaled throttle effort, got=%+v", drive)
	}
	throttle, brake := drive.ThrottleBrake()
	if throttle <= 0 || brake != 0 {
		t.Fatalf("expected throttle-only split, throttle=%f brake=%f", throttle, brake)
	}

	braking, err := controller.Step(Input{DesiredSpeedMPS: 0, CurrentSpeedMPS: 2, DT: 100 * time.Millisecond})
	if err != nil {
		t.Fatalf("braking Step returned error: %v", err)
	}
	if math.Abs(braking.LongitudinalEffort-(-0.1)) > 1e-9 {
		t.Fatalf("expected throttle cut and dt-scaled brake effort, got=%+v", braking)
	}
	throttle, brake = braking.ThrottleBrake()
	if throttle != 0 || brake <= 0 {
		t.Fatalf("expected brake-only split, throttle=%f brake=%f", throttle, brake)
	}
}

func TestStopProbabilityBrakesThenRequestsHoldAtLowSpeed(t *testing.T) {
	controller, err := New(testConfig())
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	moving, err := controller.Step(Input{
		DesiredWheelSteer: 0.5,
		DesiredSpeedMPS:   2,
		CurrentSpeedMPS:   1,
		StopProbability:   0.7,
		DT:                100 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("moving stop Step returned error: %v", err)
	}
	if moving.Hold || moving.LongitudinalEffort >= 0 {
		t.Fatalf("expected controlled braking above hold speed, got=%+v", moving)
	}

	settled, err := controller.Step(Input{
		DesiredWheelSteer: 0.5,
		DesiredSpeedMPS:   2,
		CurrentSpeedMPS:   0.1,
		StopProbability:   0.6,
		DT:                100 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("settled stop Step returned error: %v", err)
	}
	if !settled.Hold || settled.LongitudinalEffort != 0 {
		t.Fatalf("expected explicit hold with neutral pedal effort, got=%+v", settled)
	}
}

func TestStopProbabilityHysteresisPreventsBrakeChatter(t *testing.T) {
	controller, err := New(testConfig())
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	entered, err := controller.Step(Input{
		DesiredSpeedMPS: 2,
		CurrentSpeedMPS: 1,
		StopProbability: testConfig().StopProbabilityThreshold,
		DT:              100 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("stop entry Step returned error: %v", err)
	}
	if entered.Hold || entered.LongitudinalEffort >= 0 || !controller.stopLatched {
		t.Fatalf("expected stop latch to enter with service braking, got=%+v", entered)
	}

	chatter, err := controller.Step(Input{
		DesiredSpeedMPS: 2,
		CurrentSpeedMPS: 1,
		StopProbability: 0.6,
		DT:              100 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("hysteresis Step returned error: %v", err)
	}
	if chatter.LongitudinalEffort >= 0 || !controller.stopLatched {
		t.Fatalf("expected stop latch to survive probability chatter, got=%+v", chatter)
	}

	released, err := controller.Step(Input{
		DesiredSpeedMPS: 2,
		CurrentSpeedMPS: 1,
		StopProbability: testConfig().StopProbabilityReleaseThreshold,
		DT:              100 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("stop release Step returned error: %v", err)
	}
	if released.Hold || released.LongitudinalEffort <= 0 || controller.stopLatched {
		t.Fatalf("expected release threshold to resume speed tracking, got=%+v", released)
	}
}

func TestSteeringPIAccumulatesAndRejectsWindupAtFinalLimit(t *testing.T) {
	config := testConfig()
	config.SteeringFeedbackGain = 0
	config.SteeringKi = 1
	config.SteeringIntegralMin = -1
	config.SteeringIntegralMax = 1
	config.SteeringSlewPerSecond = 100
	controller, err := New(config)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	input := Input{
		DesiredWheelSteer:  0,
		MeasuredWheelSteer: -0.5,
		DT:                 100 * time.Millisecond,
	}
	first, err := controller.Step(input)
	if err != nil {
		t.Fatalf("first Step returned error: %v", err)
	}
	second, err := controller.Step(input)
	if err != nil {
		t.Fatalf("second Step returned error: %v", err)
	}
	if math.Abs(first.Steering-0.05) > 1e-9 || math.Abs(second.Steering-0.1) > 1e-9 {
		t.Fatalf("expected steering integral accumulation, first=%+v second=%+v", first, second)
	}

	controller.Reset()
	saturated, err := controller.Step(Input{
		DesiredWheelSteer:  1,
		MeasuredWheelSteer: -1,
		DT:                 100 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("saturated Step returned error: %v", err)
	}
	if saturated.Steering != 1 || controller.steeringIntegral != 0 {
		t.Fatalf("expected final-limit saturation to reject integral windup, output=%+v integral=%f", saturated, controller.steeringIntegral)
	}
}

func TestSteeringIntegralResetsOnHoldInvalidInputAndReset(t *testing.T) {
	config := testConfig()
	config.SteeringKi = 1
	config.SteeringIntegralMin = -1
	config.SteeringIntegralMax = 1
	config.SteeringSlewPerSecond = 100
	controller, err := New(config)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	accumulate := Input{DesiredWheelSteer: 0, MeasuredWheelSteer: -0.5, DT: 100 * time.Millisecond}
	if _, err := controller.Step(accumulate); err != nil {
		t.Fatalf("accumulation Step returned error: %v", err)
	}
	if controller.steeringIntegral == 0 {
		t.Fatal("expected nonzero steering integral before reset checks")
	}

	if output, err := controller.Step(Input{StopProbability: config.StopProbabilityThreshold, DT: 100 * time.Millisecond}); err != nil || !output.Hold {
		t.Fatalf("expected low-speed hold, output=%+v err=%v", output, err)
	}
	if controller.steeringIntegral != 0 {
		t.Fatalf("expected hold to reset steering integral, got=%f", controller.steeringIntegral)
	}

	if _, err := controller.Step(Input{StopProbability: config.StopProbabilityReleaseThreshold, DT: 100 * time.Millisecond}); err != nil {
		t.Fatalf("stop release Step returned error: %v", err)
	}
	if _, err := controller.Step(accumulate); err != nil {
		t.Fatalf("second accumulation Step returned error: %v", err)
	}
	if _, err := controller.Step(Input{DesiredWheelSteer: math.NaN(), DT: 100 * time.Millisecond}); err == nil {
		t.Fatal("expected invalid input error")
	}
	if controller.steeringIntegral != 0 || controller.stopLatched {
		t.Fatalf("expected invalid input to reset controller state, integral=%f stopLatched=%t", controller.steeringIntegral, controller.stopLatched)
	}

	if _, err := controller.Step(accumulate); err != nil {
		t.Fatalf("third accumulation Step returned error: %v", err)
	}
	controller.Reset()
	if controller.steeringIntegral != 0 || controller.stopLatched {
		t.Fatalf("expected Reset to clear controller state, integral=%f stopLatched=%t", controller.steeringIntegral, controller.stopLatched)
	}
}

func TestInvalidInputReturnsFailSafeHoldAndResetsHistory(t *testing.T) {
	controller, err := New(testConfig())
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	valid := Input{DesiredWheelSteer: 1, DesiredSpeedMPS: 3, DT: 100 * time.Millisecond}
	if _, err := controller.Step(valid); err != nil {
		t.Fatalf("initial Step returned error: %v", err)
	}

	failSafe, err := controller.Step(Input{DesiredWheelSteer: math.NaN(), DT: 100 * time.Millisecond})
	if err == nil || !errors.Is(err, ErrInvalidInput) {
		t.Fatalf("expected invalid input error, got=%v", err)
	}
	if failSafe != FailSafeOutput() {
		t.Fatalf("expected explicit fail-safe hold, got=%+v", failSafe)
	}

	afterReset, err := controller.Step(valid)
	if err != nil {
		t.Fatalf("Step after reset returned error: %v", err)
	}
	fresh, _ := New(testConfig())
	freshOutput, _ := fresh.Step(valid)
	if afterReset != freshOutput {
		t.Fatalf("expected invalid input to clear dynamic history, reset=%+v fresh=%+v", afterReset, freshOutput)
	}
}

func TestControllerRejectsUnsafeDT(t *testing.T) {
	controller, err := New(testConfig())
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	for _, dt := range []time.Duration{0, -time.Millisecond, testConfig().MaxDT + time.Nanosecond} {
		output, err := controller.Step(Input{DT: dt})
		if err == nil || !errors.Is(err, ErrInvalidInput) || !output.Hold {
			t.Fatalf("expected dt=%s to fail safe, output=%+v err=%v", dt, output, err)
		}
	}
}

func TestNewClonesCalibrationAndRejectsInvalidConfiguration(t *testing.T) {
	config := testConfig()
	controller, err := New(config)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	config.SteeringProfile[2].Command = 1
	output, err := controller.Step(Input{DesiredWheelSteer: 0.5, DT: 500 * time.Millisecond})
	if err != nil {
		t.Fatalf("Step returned error: %v", err)
	}
	if math.Abs(output.Steering-0.65) > 1e-9 {
		t.Fatalf("expected cloned 0.4 feedforward plus 0.25 feedback, got=%+v", output)
	}

	tests := []struct {
		name   string
		mutate func(*Config)
	}{
		{name: "zero PI gains", mutate: func(c *Config) { c.SpeedKp, c.SpeedKi = 0, 0 }},
		{name: "zero steering slew", mutate: func(c *Config) { c.SteeringSlewPerSecond = 0 }},
		{name: "effort above normalized range", mutate: func(c *Config) { c.MaxThrottleEffort = 1.1 }},
		{name: "stop brake exceeds brake bound", mutate: func(c *Config) { c.StopBrakeEffort = c.MaxBrakeEffort + 0.01 }},
		{name: "invalid integral bounds", mutate: func(c *Config) { c.SpeedIntegralMin = 0.1 }},
		{name: "negative steering ki", mutate: func(c *Config) { c.SteeringKi = -0.1 }},
		{name: "invalid steering integral bounds", mutate: func(c *Config) { c.SteeringIntegralMax = -0.1 }},
		{name: "invalid stop threshold", mutate: func(c *Config) { c.StopProbabilityThreshold = 0 }},
		{name: "release threshold equals enter", mutate: func(c *Config) { c.StopProbabilityReleaseThreshold = c.StopProbabilityThreshold }},
		{name: "negative release threshold", mutate: func(c *Config) { c.StopProbabilityReleaseThreshold = -0.1 }},
		{name: "invalid maximum dt", mutate: func(c *Config) { c.MaxDT = 0 }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			candidate := testConfig()
			tc.mutate(&candidate)
			_, err := New(candidate)
			if err == nil || !errors.Is(err, ErrInvalidConfig) {
				t.Fatalf("expected invalid config error, got=%v", err)
			}
		})
	}
}

func TestControllerConfigUsesSnakeCaseJSONAndTOMLNames(t *testing.T) {
	tests := []struct {
		typeOf reflect.Type
		field  string
		name   string
	}{
		{typeOf: reflect.TypeOf(Config{}), field: "SteeringProfile", name: "steering_profile"},
		{typeOf: reflect.TypeOf(Config{}), field: "StopProbabilityReleaseThreshold", name: "stop_probability_release_threshold"},
		{typeOf: reflect.TypeOf(SteeringCalibrationPoint{}), field: "WheelSteer", name: "wheel_steer"},
		{typeOf: reflect.TypeOf(SteeringCalibrationPoint{}), field: "Command", name: "command"},
	}

	for _, tc := range tests {
		field, ok := tc.typeOf.FieldByName(tc.field)
		if !ok {
			t.Fatalf("missing public field %s", tc.field)
		}
		if got := field.Tag.Get("json"); got != tc.name {
			t.Fatalf("unexpected JSON tag for %s: got=%q want=%q", tc.field, got, tc.name)
		}
		if got := field.Tag.Get("toml"); got != tc.name {
			t.Fatalf("unexpected TOML tag for %s: got=%q want=%q", tc.field, got, tc.name)
		}
	}
}
