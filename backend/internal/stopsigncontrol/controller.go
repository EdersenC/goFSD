// Package stopsigncontrol turns stop-sign motion plans and measured vehicle
// feedback into bounded normalized control commands. Controller is stateful and
// must be driven serially with the actual elapsed time between calls.
package stopsigncontrol

import (
	"errors"
	"fmt"
	"time"
)

var ErrInvalidConfig = errors.New("invalid stop-sign feedback controller configuration")
var ErrInvalidInput = errors.New("invalid stop-sign feedback controller input")

// Config defines the calibrated steering map and closed-loop safety bounds.
type Config struct {
	SteeringProfile                 []SteeringCalibrationPoint `json:"steering_profile" toml:"steering_profile"`
	StraightApproachOnly            bool                       `json:"straight_approach_only" toml:"straight_approach_only"`
	SteeringFeedbackGain            float64                    `json:"steering_feedback_gain" toml:"steering_feedback_gain"`
	SteeringKi                      float64                    `json:"steering_ki" toml:"steering_ki"`
	SteeringIntegralMin             float64                    `json:"steering_integral_min" toml:"steering_integral_min"`
	SteeringIntegralMax             float64                    `json:"steering_integral_max" toml:"steering_integral_max"`
	SteeringSlewPerSecond           float64                    `json:"steering_slew_per_second" toml:"steering_slew_per_second"`
	SpeedKp                         float64                    `json:"speed_kp" toml:"speed_kp"`
	SpeedKi                         float64                    `json:"speed_ki" toml:"speed_ki"`
	SpeedIntegralMin                float64                    `json:"speed_integral_min" toml:"speed_integral_min"`
	SpeedIntegralMax                float64                    `json:"speed_integral_max" toml:"speed_integral_max"`
	MaxThrottleEffort               float64                    `json:"max_throttle_effort" toml:"max_throttle_effort"`
	MaxBrakeEffort                  float64                    `json:"max_brake_effort" toml:"max_brake_effort"`
	LongitudinalSlewPerSecond       float64                    `json:"longitudinal_slew_per_second" toml:"longitudinal_slew_per_second"`
	StopProbabilityThreshold        float64                    `json:"stop_probability_threshold" toml:"stop_probability_threshold"`
	StopProbabilityReleaseThreshold float64                    `json:"stop_probability_release_threshold" toml:"stop_probability_release_threshold"`
	HoldSpeedMPS                    float64                    `json:"hold_speed_mps" toml:"hold_speed_mps"`
	StopBrakeEffort                 float64                    `json:"stop_brake_effort" toml:"stop_brake_effort"`
	MaxDesiredSpeedMPS              float64                    `json:"max_desired_speed_mps" toml:"max_desired_speed_mps"`
	MaxDT                           time.Duration              `json:"max_dt" toml:"max_dt"`
}

// Input contains planner setpoints, measured physical feedback, and the actual
// elapsed control interval.
type Input struct {
	DesiredWheelSteer  float64
	DesiredSpeedMPS    float64
	StopProbability    float64
	MeasuredWheelSteer float64
	CurrentSpeedMPS    float64
	DT                 time.Duration
}

// Output uses one signed longitudinal effort so throttle and brake cannot be
// requested simultaneously. Positive effort is throttle; negative is brake.
type Output struct {
	Steering                   float64 `json:"steering"`
	LongitudinalEffort         float64 `json:"longitudinal_effort"`
	Hold                       bool    `json:"hold"`
	StopLatched                bool    `json:"stop_latched"`
	SteeringFeedForward        float64 `json:"steering_feed_forward"`
	SteeringError              float64 `json:"steering_error"`
	SteeringProportional       float64 `json:"steering_proportional"`
	SteeringIntegralCorrection float64 `json:"steering_integral_correction"`
	SpeedError                 float64 `json:"speed_error"`
	SpeedProportional          float64 `json:"speed_proportional"`
	SpeedIntegralCorrection    float64 `json:"speed_integral_correction"`
}

// ThrottleBrake converts signed effort to mutually exclusive actuator values.
func (o Output) ThrottleBrake() (throttle, brake float64) {
	if o.LongitudinalEffort > 0 {
		return o.LongitudinalEffort, 0
	}
	if o.LongitudinalEffort < 0 {
		return 0, -o.LongitudinalEffort
	}
	return 0, 0
}

// FailSafeOutput is the command returned for malformed or unsafe input.
func FailSafeOutput() Output {
	return Output{Hold: true}
}

type Controller struct {
	config           Config
	steeringProfile  steeringProfile
	speedController  boundedPI
	steeringIntegral float64
	previousSteering float64
	previousEffort   float64
	stopLatched      bool
}

func New(config Config) (*Controller, error) {
	profile, err := newSteeringProfile(config.SteeringProfile)
	if err != nil {
		return nil, err
	}
	if err := validateConfig(config); err != nil {
		return nil, err
	}
	return &Controller{
		config:          cloneConfig(config),
		steeringProfile: profile,
		speedController: boundedPI{
			kp:          config.SpeedKp,
			ki:          config.SpeedKi,
			integralMin: config.SpeedIntegralMin,
			integralMax: config.SpeedIntegralMax,
			outputMin:   -config.MaxBrakeEffort,
			outputMax:   config.MaxThrottleEffort,
		},
	}, nil
}

// Step advances the feedback controller once. Invalid external input resets
// accumulated state and returns an explicit fail-safe hold alongside the error.
func (c *Controller) Step(input Input) (Output, error) {
	if err := c.validateInput(input); err != nil {
		c.Reset()
		return FailSafeOutput(), err
	}

	dtSeconds := input.DT.Seconds()
	c.updateStopLatch(input.StopProbability)
	hold := c.stopLatched && input.CurrentSpeedMPS <= c.config.HoldSpeedMPS
	if hold {
		c.speedController.reset()
		c.steeringIntegral = 0
		c.previousEffort = 0
		c.previousSteering = slew(
			c.previousSteering,
			0,
			c.config.SteeringSlewPerSecond*dtSeconds,
		)
		return Output{
			Steering:    c.previousSteering,
			Hold:        true,
			StopLatched: true,
		}, nil
	}

	steeringTarget, steeringTrace := c.steeringTarget(input, dtSeconds)
	effortTarget := c.longitudinalTarget(input, dtSeconds)
	c.previousEffort = slewLongitudinal(
		c.previousEffort,
		effortTarget,
		c.config.LongitudinalSlewPerSecond*dtSeconds,
	)
	c.previousSteering = slew(
		c.previousSteering,
		steeringTarget,
		c.config.SteeringSlewPerSecond*dtSeconds,
	)
	return Output{
		Steering:                   c.previousSteering,
		LongitudinalEffort:         c.previousEffort,
		Hold:                       hold,
		StopLatched:                c.stopLatched,
		SteeringFeedForward:        steeringTrace.feedForward,
		SteeringError:              steeringTrace.err,
		SteeringProportional:       steeringTrace.proportional,
		SteeringIntegralCorrection: steeringTrace.integralCorrection,
		SpeedError:                 input.DesiredSpeedMPS - input.CurrentSpeedMPS,
		SpeedProportional:          c.config.SpeedKp * (input.DesiredSpeedMPS - input.CurrentSpeedMPS),
		SpeedIntegralCorrection:    c.config.SpeedKi * c.speedController.integral,
	}, nil
}

func (c *Controller) longitudinalTarget(input Input, dtSeconds float64) float64 {
	if c.stopLatched {
		c.speedController.reset()
		return -c.config.StopBrakeEffort
	}
	speedError := input.DesiredSpeedMPS - input.CurrentSpeedMPS
	return c.speedController.step(speedError, dtSeconds)
}

// Reset clears all integral and output history.
func (c *Controller) Reset() {
	c.speedController.reset()
	c.steeringIntegral = 0
	c.previousSteering = 0
	c.previousEffort = 0
	c.stopLatched = false
}

type steeringControlTrace struct {
	feedForward        float64
	err                float64
	proportional       float64
	integralCorrection float64
}

func (c *Controller) steeringTarget(input Input, dtSeconds float64) (float64, steeringControlTrace) {
	desiredWheelSteer := input.DesiredWheelSteer
	if c.config.StraightApproachOnly {
		desiredWheelSteer = 0
	}
	err := desiredWheelSteer - input.MeasuredWheelSteer
	feedForward := c.steeringProfile.commandFor(desiredWheelSteer)
	candidateIntegral := clamp(
		c.steeringIntegral+err*dtSeconds,
		c.config.SteeringIntegralMin,
		c.config.SteeringIntegralMax,
	)
	unsaturated := feedForward +
		c.config.SteeringFeedbackGain*err +
		c.config.SteeringKi*candidateIntegral
	pushesUpperLimit := unsaturated > 1 && err > 0
	pushesLowerLimit := unsaturated < -1 && err < 0
	if !pushesUpperLimit && !pushesLowerLimit {
		c.steeringIntegral = candidateIntegral
	}
	return clamp(unsaturated, -1, 1), steeringControlTrace{
		feedForward:        feedForward,
		err:                err,
		proportional:       c.config.SteeringFeedbackGain * err,
		integralCorrection: c.config.SteeringKi * c.steeringIntegral,
	}
}

func (c *Controller) updateStopLatch(stopProbability float64) {
	if c.stopLatched {
		if stopProbability <= c.config.StopProbabilityReleaseThreshold {
			c.stopLatched = false
		}
		return
	}
	if stopProbability >= c.config.StopProbabilityThreshold {
		c.stopLatched = true
	}
}

func (c *Controller) validateInput(input Input) error {
	values := []struct {
		name  string
		value float64
		min   float64
		max   float64
	}{
		{name: "desired wheel steer", value: input.DesiredWheelSteer, min: -1, max: 1},
		{name: "measured wheel steer", value: input.MeasuredWheelSteer, min: -1, max: 1},
		{name: "desired speed", value: input.DesiredSpeedMPS, min: 0, max: c.config.MaxDesiredSpeedMPS},
		{name: "stop probability", value: input.StopProbability, min: 0, max: 1},
	}
	if !finite(input.CurrentSpeedMPS) || input.CurrentSpeedMPS < 0 {
		return inputError("current speed %.6f must be finite and >= 0", input.CurrentSpeedMPS)
	}
	for _, item := range values {
		if !finite(item.value) || item.value < item.min || item.value > item.max {
			return inputError("%s %.6f is outside [%.6f, %.6f]", item.name, item.value, item.min, item.max)
		}
	}
	if input.DT <= 0 || input.DT > c.config.MaxDT {
		return inputError("dt %s must be within (0, %s]", input.DT, c.config.MaxDT)
	}
	return nil
}

func validateConfig(config Config) error {
	finiteNonNegative := []struct {
		name  string
		value float64
	}{
		{name: "steering feedback gain", value: config.SteeringFeedbackGain},
		{name: "steering ki", value: config.SteeringKi},
		{name: "speed kp", value: config.SpeedKp},
		{name: "speed ki", value: config.SpeedKi},
		{name: "hold speed", value: config.HoldSpeedMPS},
	}
	for _, item := range finiteNonNegative {
		if !finite(item.value) || item.value < 0 {
			return configError("%s must be finite and >= 0", item.name)
		}
	}
	if config.SpeedKp == 0 && config.SpeedKi == 0 {
		return configError("at least one speed PI gain must be > 0")
	}
	positive := []struct {
		name  string
		value float64
	}{
		{name: "steering slew", value: config.SteeringSlewPerSecond},
		{name: "longitudinal slew", value: config.LongitudinalSlewPerSecond},
		{name: "maximum desired speed", value: config.MaxDesiredSpeedMPS},
	}
	for _, item := range positive {
		if !finite(item.value) || item.value <= 0 {
			return configError("%s must be finite and > 0", item.name)
		}
	}
	normalizedPositive := []struct {
		name  string
		value float64
	}{
		{name: "maximum throttle effort", value: config.MaxThrottleEffort},
		{name: "maximum brake effort", value: config.MaxBrakeEffort},
		{name: "stop brake effort", value: config.StopBrakeEffort},
	}
	for _, item := range normalizedPositive {
		if !finite(item.value) || item.value <= 0 || item.value > 1 {
			return configError("%s must be within (0, 1]", item.name)
		}
	}
	if config.StopBrakeEffort > config.MaxBrakeEffort {
		return configError("stop brake effort must not exceed maximum brake effort")
	}
	if !finite(config.SpeedIntegralMin) || !finite(config.SpeedIntegralMax) || config.SpeedIntegralMin > 0 || config.SpeedIntegralMax < 0 || config.SpeedIntegralMin > config.SpeedIntegralMax {
		return configError("speed integral bounds must be finite, ordered, and contain zero")
	}
	if !finite(config.SteeringIntegralMin) || !finite(config.SteeringIntegralMax) || config.SteeringIntegralMin > 0 || config.SteeringIntegralMax < 0 || config.SteeringIntegralMin > config.SteeringIntegralMax {
		return configError("steering integral bounds must be finite, ordered, and contain zero")
	}
	if !finite(config.StopProbabilityThreshold) || config.StopProbabilityThreshold <= 0 || config.StopProbabilityThreshold > 1 {
		return configError("stop probability threshold must be within (0, 1]")
	}
	if !finite(config.StopProbabilityReleaseThreshold) || config.StopProbabilityReleaseThreshold < 0 || config.StopProbabilityReleaseThreshold >= config.StopProbabilityThreshold {
		return configError("stop probability release threshold must be within [0, stop probability threshold)")
	}
	if config.MaxDT <= 0 {
		return configError("maximum dt must be > 0")
	}
	return nil
}

func cloneConfig(config Config) Config {
	config.SteeringProfile = append([]SteeringCalibrationPoint(nil), config.SteeringProfile...)
	return config
}

func slew(current, target, maximumDelta float64) float64 {
	return current + clamp(target-current, -maximumDelta, maximumDelta)
}

func slewLongitudinal(current, target, maximumDelta float64) float64 {
	if current > 0 && target <= 0 || current < 0 && target >= 0 {
		current = 0
	}
	return slew(current, target, maximumDelta)
}

func configError(format string, args ...any) error {
	return fmt.Errorf("%w: %s", ErrInvalidConfig, fmt.Sprintf(format, args...))
}

func inputError(format string, args ...any) error {
	return fmt.Errorf("%w: %s", ErrInvalidInput, fmt.Sprintf(format, args...))
}
