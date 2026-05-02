package translation

import (
	"errors"
	"fmt"
	"os"
	"strings"

	toml "github.com/pelletier/go-toml/v2"
)

const (
	defaultHeadingDeadbandDeg           = 2.5
	defaultHeadingFullLockDeg           = 45.0
	defaultLowSpeedSteerGain            = 1.35
	defaultHighSpeedSteerGain           = 0.75
	defaultSteerGainFadeSpeedMPS        = 5.0
	defaultSteerResponseBlend           = 0.55
	defaultSteerCommandRatePerSecond    = 4.0
	defaultSteerDeadzone                = 0.02
	defaultMaxSteerScale                = 1.0
	defaultSteerOutputGain              = 1.0
	defaultThrottleGain                 = 1.0
	defaultBrakeGain                    = 1.0
	defaultBrakeActivationThreshold     = 0.18
	defaultOverspeedBrakeMarginMPS      = 0.75
	defaultBrakeReleaseMarginMPS        = 0.20
	defaultBrakeEnterHoldSeconds        = 0.18
	defaultThrottleHoldMin              = 0.12
	defaultThrottleHoldSeconds          = 2.0
	defaultThrottleRampUpPerSecond      = 1.20
	defaultThrottleDecayPerSecond       = 0.90
	defaultTargetSpeedErrorGain         = 0.5
	defaultTargetAccelGain              = 0.25
	defaultLongitudinalDeadband         = 0.05
	defaultLaunchThrottleMin            = 0.22
	defaultLaunchSpeedThreshold         = 1.0
	defaultLaunchTargetSpeedMargin      = 0.75
	defaultMaxTargetSpeedKPH            = 17.0
	defaultMotionConfidenceOnThreshold  = 0.60
	defaultMotionConfidenceOffThreshold = 0.40
	defaultMotionHoldSpeedMax           = 4.0
	defaultMinLateralConfidence         = 0.05
	defaultMinLongitudinalConfidence    = 0.05
)

type Config struct {
	HeadingDeadbandDeg           float64
	HeadingFullLockDeg           float64
	LowSpeedSteerGain            float64
	HighSpeedSteerGain           float64
	SteerGainFadeSpeedMPS        float64
	SteerResponseBlend           float64
	SteerCommandRatePerSecond    float64
	SteerDeadzone                float64
	MaxSteerScale                float64
	SteerOutputGain              float64
	ThrottleGain                 float64
	BrakeGain                    float64
	BrakeActivationThreshold     float64
	OverspeedBrakeMarginMPS      float64
	BrakeReleaseMarginMPS        float64
	BrakeEnterHoldSeconds        float64
	ThrottleHoldSeconds          float64
	ThrottleHoldMin              float64
	ThrottleRampUpPerSecond      float64
	ThrottleDecayPerSecond       float64
	TargetSpeedErrorGain         float64
	TargetAccelGain              float64
	LongitudinalDeadband         float64
	LaunchThrottleMin            float64
	LaunchSpeedThreshold         float64
	LaunchTargetSpeedMargin      float64
	MaxTargetSpeedKPH            float64
	MotionConfidenceOnThreshold  float64
	MotionConfidenceOffThreshold float64
	MotionHoldSpeedMax           float64
	MinLateralConfidence         float64
	MinLongitudinalConfidence    float64
}

type Tuning struct {
	HeadingDeadbandDeg           float64 `json:"headingDeadbandDeg"`
	HeadingFullLockDeg           float64 `json:"headingFullLockDeg"`
	LowSpeedSteerGain            float64 `json:"lowSpeedSteerGain"`
	HighSpeedSteerGain           float64 `json:"highSpeedSteerGain"`
	SteerGainFadeSpeedMPS        float64 `json:"steerGainFadeSpeedMps"`
	SteerResponseBlend           float64 `json:"steerResponseBlend"`
	SteerCommandRatePerSecond    float64 `json:"steerCommandRatePerSecond"`
	SteerDeadzone                float64 `json:"steerDeadzone"`
	MaxSteerScale                float64 `json:"maxSteerScale"`
	SteerOutputGain              float64 `json:"steerOutputGain"`
	ThrottleGain                 float64 `json:"throttleGain"`
	BrakeGain                    float64 `json:"brakeGain"`
	BrakeActivationThreshold     float64 `json:"brakeActivationThreshold"`
	OverspeedBrakeMarginMPS      float64 `json:"overspeedBrakeMarginMps"`
	BrakeReleaseMarginMPS        float64 `json:"brakeReleaseMarginMps"`
	BrakeEnterHoldSeconds        float64 `json:"brakeEnterHoldSeconds"`
	ThrottleHoldSeconds          float64 `json:"throttleHoldSeconds"`
	ThrottleHoldMin              float64 `json:"throttleHoldMin"`
	ThrottleRampUpPerSecond      float64 `json:"throttleRampUpPerSecond"`
	ThrottleDecayPerSecond       float64 `json:"throttleDecayPerSecond"`
	TargetSpeedErrorGain         float64 `json:"targetSpeedErrorGain"`
	TargetAccelGain              float64 `json:"targetAccelGain"`
	LongitudinalDeadband         float64 `json:"longitudinalDeadband"`
	LaunchThrottleMin            float64 `json:"launchThrottleMin"`
	LaunchSpeedThreshold         float64 `json:"launchSpeedThreshold"`
	LaunchTargetSpeedMargin      float64 `json:"launchTargetSpeedMargin"`
	MaxTargetSpeedKPH            float64 `json:"maxTargetSpeedKph"`
	MotionConfidenceOnThreshold  float64 `json:"motionConfidenceOnThreshold"`
	MotionConfidenceOffThreshold float64 `json:"motionConfidenceOffThreshold"`
	MotionHoldSpeedMax           float64 `json:"motionHoldSpeedMax"`
	MinLateralConfidence         float64 `json:"minLateralConfidence"`
	MinLongitudinalConfidence    float64 `json:"minLongitudinalConfidence"`
}

type configFile struct {
	Backend backendSection `toml:"backend"`
}

type backendSection struct {
	Translation translationSection `toml:"translation"`
}

type translationSection struct {
	HeadingDeadbandDeg           *float64 `toml:"heading_deadband_deg"`
	HeadingFullLockDeg           *float64 `toml:"heading_full_lock_deg"`
	LowSpeedSteerGain            *float64 `toml:"low_speed_steer_gain"`
	HighSpeedSteerGain           *float64 `toml:"high_speed_steer_gain"`
	SteerGainFadeSpeedMPS        *float64 `toml:"steer_gain_fade_speed_mps"`
	SteerResponseBlend           *float64 `toml:"steer_response_blend"`
	SteerCommandRatePerSecond    *float64 `toml:"steer_command_rate_per_second"`
	SteerDeadzone                *float64 `toml:"steer_deadzone"`
	MaxSteerScale                *float64 `toml:"max_steer_scale"`
	SteerOutputGain              *float64 `toml:"steer_output_gain"`
	ThrottleGain                 *float64 `toml:"throttle_gain"`
	BrakeGain                    *float64 `toml:"brake_gain"`
	BrakeActivationThreshold     *float64 `toml:"brake_activation_threshold"`
	OverspeedBrakeMarginMPS      *float64 `toml:"overspeed_brake_margin_mps"`
	BrakeReleaseMarginMPS        *float64 `toml:"brake_release_margin_mps"`
	BrakeEnterHoldSeconds        *float64 `toml:"brake_enter_hold_seconds"`
	ThrottleHoldSeconds          *float64 `toml:"throttle_hold_seconds"`
	ThrottleHoldMin              *float64 `toml:"throttle_hold_min"`
	ThrottleRampUpPerSecond      *float64 `toml:"throttle_ramp_up_per_second"`
	ThrottleDecayPerSecond       *float64 `toml:"throttle_decay_per_second"`
	TargetSpeedErrorGain         *float64 `toml:"target_speed_error_gain"`
	TargetAccelGain              *float64 `toml:"target_accel_gain"`
	LongitudinalDeadband         *float64 `toml:"longitudinal_deadband"`
	LaunchThrottleMin            *float64 `toml:"launch_throttle_min"`
	LaunchSpeedThreshold         *float64 `toml:"launch_speed_threshold"`
	LaunchTargetSpeedMargin      *float64 `toml:"launch_target_speed_margin"`
	MaxTargetSpeedKPH            *float64 `toml:"max_target_speed_kph"`
	MotionConfidenceOnThreshold  *float64 `toml:"motion_confidence_on_threshold"`
	MotionConfidenceOffThreshold *float64 `toml:"motion_confidence_off_threshold"`
	MotionHoldSpeedMax           *float64 `toml:"motion_hold_speed_max"`
	MinLateralConfidence         *float64 `toml:"min_lateral_confidence"`
	MinLongitudinalConfidence    *float64 `toml:"min_longitudinal_confidence"`
}

func DefaultConfig() Config {
	return Config{
		HeadingDeadbandDeg:           defaultHeadingDeadbandDeg,
		HeadingFullLockDeg:           defaultHeadingFullLockDeg,
		LowSpeedSteerGain:            defaultLowSpeedSteerGain,
		HighSpeedSteerGain:           defaultHighSpeedSteerGain,
		SteerGainFadeSpeedMPS:        defaultSteerGainFadeSpeedMPS,
		SteerResponseBlend:           defaultSteerResponseBlend,
		SteerCommandRatePerSecond:    defaultSteerCommandRatePerSecond,
		SteerDeadzone:                defaultSteerDeadzone,
		MaxSteerScale:                defaultMaxSteerScale,
		SteerOutputGain:              defaultSteerOutputGain,
		ThrottleGain:                 defaultThrottleGain,
		BrakeGain:                    defaultBrakeGain,
		BrakeActivationThreshold:     defaultBrakeActivationThreshold,
		OverspeedBrakeMarginMPS:      defaultOverspeedBrakeMarginMPS,
		BrakeReleaseMarginMPS:        defaultBrakeReleaseMarginMPS,
		BrakeEnterHoldSeconds:        defaultBrakeEnterHoldSeconds,
		ThrottleHoldSeconds:          defaultThrottleHoldSeconds,
		ThrottleHoldMin:              defaultThrottleHoldMin,
		ThrottleRampUpPerSecond:      defaultThrottleRampUpPerSecond,
		ThrottleDecayPerSecond:       defaultThrottleDecayPerSecond,
		TargetSpeedErrorGain:         defaultTargetSpeedErrorGain,
		TargetAccelGain:              defaultTargetAccelGain,
		LongitudinalDeadband:         defaultLongitudinalDeadband,
		LaunchThrottleMin:            defaultLaunchThrottleMin,
		LaunchSpeedThreshold:         defaultLaunchSpeedThreshold,
		LaunchTargetSpeedMargin:      defaultLaunchTargetSpeedMargin,
		MaxTargetSpeedKPH:            defaultMaxTargetSpeedKPH,
		MotionConfidenceOnThreshold:  defaultMotionConfidenceOnThreshold,
		MotionConfidenceOffThreshold: defaultMotionConfidenceOffThreshold,
		MotionHoldSpeedMax:           defaultMotionHoldSpeedMax,
		MinLateralConfidence:         defaultMinLateralConfidence,
		MinLongitudinalConfidence:    defaultMinLongitudinalConfidence,
	}
}

func LoadConfig(path string) (Config, error) {
	cfg := DefaultConfig()
	if strings.TrimSpace(path) == "" {
		return cfg, nil
	}

	raw, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return cfg, nil
		}
		return Config{}, err
	}

	var parsed configFile
	if err := toml.Unmarshal(raw, &parsed); err != nil {
		return Config{}, err
	}

	section := parsed.Backend.Translation
	if section.HeadingDeadbandDeg != nil {
		cfg.HeadingDeadbandDeg = *section.HeadingDeadbandDeg
	}
	if section.HeadingFullLockDeg != nil {
		cfg.HeadingFullLockDeg = *section.HeadingFullLockDeg
	}
	if section.LowSpeedSteerGain != nil {
		cfg.LowSpeedSteerGain = *section.LowSpeedSteerGain
	}
	if section.HighSpeedSteerGain != nil {
		cfg.HighSpeedSteerGain = *section.HighSpeedSteerGain
	}
	if section.SteerGainFadeSpeedMPS != nil {
		cfg.SteerGainFadeSpeedMPS = *section.SteerGainFadeSpeedMPS
	}
	if section.SteerResponseBlend != nil {
		cfg.SteerResponseBlend = *section.SteerResponseBlend
	}
	if section.SteerCommandRatePerSecond != nil {
		cfg.SteerCommandRatePerSecond = *section.SteerCommandRatePerSecond
	}
	if section.SteerDeadzone != nil {
		cfg.SteerDeadzone = *section.SteerDeadzone
	}
	if section.MaxSteerScale != nil {
		cfg.MaxSteerScale = *section.MaxSteerScale
	}
	if section.SteerOutputGain != nil {
		cfg.SteerOutputGain = *section.SteerOutputGain
	}
	if section.ThrottleGain != nil {
		cfg.ThrottleGain = *section.ThrottleGain
	}
	if section.BrakeGain != nil {
		cfg.BrakeGain = *section.BrakeGain
	}
	if section.BrakeActivationThreshold != nil {
		cfg.BrakeActivationThreshold = *section.BrakeActivationThreshold
	}
	if section.OverspeedBrakeMarginMPS != nil {
		cfg.OverspeedBrakeMarginMPS = *section.OverspeedBrakeMarginMPS
	}
	if section.BrakeReleaseMarginMPS != nil {
		cfg.BrakeReleaseMarginMPS = *section.BrakeReleaseMarginMPS
	}
	if section.BrakeEnterHoldSeconds != nil {
		cfg.BrakeEnterHoldSeconds = *section.BrakeEnterHoldSeconds
	}
	if section.ThrottleHoldSeconds != nil {
		cfg.ThrottleHoldSeconds = *section.ThrottleHoldSeconds
	}
	if section.ThrottleHoldMin != nil {
		cfg.ThrottleHoldMin = *section.ThrottleHoldMin
	}
	if section.ThrottleRampUpPerSecond != nil {
		cfg.ThrottleRampUpPerSecond = *section.ThrottleRampUpPerSecond
	}
	if section.ThrottleDecayPerSecond != nil {
		cfg.ThrottleDecayPerSecond = *section.ThrottleDecayPerSecond
	}
	if section.TargetSpeedErrorGain != nil {
		cfg.TargetSpeedErrorGain = *section.TargetSpeedErrorGain
	}
	if section.TargetAccelGain != nil {
		cfg.TargetAccelGain = *section.TargetAccelGain
	}
	if section.LongitudinalDeadband != nil {
		cfg.LongitudinalDeadband = *section.LongitudinalDeadband
	}
	if section.LaunchThrottleMin != nil {
		cfg.LaunchThrottleMin = *section.LaunchThrottleMin
	}
	if section.LaunchSpeedThreshold != nil {
		cfg.LaunchSpeedThreshold = *section.LaunchSpeedThreshold
	}
	if section.LaunchTargetSpeedMargin != nil {
		cfg.LaunchTargetSpeedMargin = *section.LaunchTargetSpeedMargin
	}
	if section.MaxTargetSpeedKPH != nil {
		cfg.MaxTargetSpeedKPH = *section.MaxTargetSpeedKPH
	}
	if section.MotionConfidenceOnThreshold != nil {
		cfg.MotionConfidenceOnThreshold = *section.MotionConfidenceOnThreshold
	}
	if section.MotionConfidenceOffThreshold != nil {
		cfg.MotionConfidenceOffThreshold = *section.MotionConfidenceOffThreshold
	}
	if section.MotionHoldSpeedMax != nil {
		cfg.MotionHoldSpeedMax = *section.MotionHoldSpeedMax
	}
	if section.MinLateralConfidence != nil {
		cfg.MinLateralConfidence = *section.MinLateralConfidence
	}
	if section.MinLongitudinalConfidence != nil {
		cfg.MinLongitudinalConfidence = *section.MinLongitudinalConfidence
	}

	if err := ValidateTuning(cfg.Tuning()); err != nil {
		return Config{}, err
	}
	return cfg, nil
}

func (c Config) Tuning() Tuning {
	return Tuning{
		HeadingDeadbandDeg:           c.HeadingDeadbandDeg,
		HeadingFullLockDeg:           c.HeadingFullLockDeg,
		LowSpeedSteerGain:            c.LowSpeedSteerGain,
		HighSpeedSteerGain:           c.HighSpeedSteerGain,
		SteerGainFadeSpeedMPS:        c.SteerGainFadeSpeedMPS,
		SteerResponseBlend:           c.SteerResponseBlend,
		SteerCommandRatePerSecond:    c.SteerCommandRatePerSecond,
		SteerDeadzone:                c.SteerDeadzone,
		MaxSteerScale:                c.MaxSteerScale,
		SteerOutputGain:              c.SteerOutputGain,
		ThrottleGain:                 c.ThrottleGain,
		BrakeGain:                    c.BrakeGain,
		BrakeActivationThreshold:     c.BrakeActivationThreshold,
		OverspeedBrakeMarginMPS:      c.OverspeedBrakeMarginMPS,
		BrakeReleaseMarginMPS:        c.BrakeReleaseMarginMPS,
		BrakeEnterHoldSeconds:        c.BrakeEnterHoldSeconds,
		ThrottleHoldSeconds:          c.ThrottleHoldSeconds,
		ThrottleHoldMin:              c.ThrottleHoldMin,
		ThrottleRampUpPerSecond:      c.ThrottleRampUpPerSecond,
		ThrottleDecayPerSecond:       c.ThrottleDecayPerSecond,
		TargetSpeedErrorGain:         c.TargetSpeedErrorGain,
		TargetAccelGain:              c.TargetAccelGain,
		LongitudinalDeadband:         c.LongitudinalDeadband,
		LaunchThrottleMin:            c.LaunchThrottleMin,
		LaunchSpeedThreshold:         c.LaunchSpeedThreshold,
		LaunchTargetSpeedMargin:      c.LaunchTargetSpeedMargin,
		MaxTargetSpeedKPH:            c.MaxTargetSpeedKPH,
		MotionConfidenceOnThreshold:  c.MotionConfidenceOnThreshold,
		MotionConfidenceOffThreshold: c.MotionConfidenceOffThreshold,
		MotionHoldSpeedMax:           c.MotionHoldSpeedMax,
		MinLateralConfidence:         c.MinLateralConfidence,
		MinLongitudinalConfidence:    c.MinLongitudinalConfidence,
	}
}

func (c *Config) ApplyTuning(tuning Tuning) {
	c.HeadingDeadbandDeg = tuning.HeadingDeadbandDeg
	c.HeadingFullLockDeg = tuning.HeadingFullLockDeg
	c.LowSpeedSteerGain = tuning.LowSpeedSteerGain
	c.HighSpeedSteerGain = tuning.HighSpeedSteerGain
	c.SteerGainFadeSpeedMPS = tuning.SteerGainFadeSpeedMPS
	c.SteerResponseBlend = tuning.SteerResponseBlend
	c.SteerCommandRatePerSecond = tuning.SteerCommandRatePerSecond
	c.SteerDeadzone = tuning.SteerDeadzone
	c.MaxSteerScale = tuning.MaxSteerScale
	c.SteerOutputGain = tuning.SteerOutputGain
	c.ThrottleGain = tuning.ThrottleGain
	c.BrakeGain = tuning.BrakeGain
	c.BrakeActivationThreshold = tuning.BrakeActivationThreshold
	c.OverspeedBrakeMarginMPS = tuning.OverspeedBrakeMarginMPS
	c.BrakeReleaseMarginMPS = tuning.BrakeReleaseMarginMPS
	c.BrakeEnterHoldSeconds = tuning.BrakeEnterHoldSeconds
	c.ThrottleHoldSeconds = tuning.ThrottleHoldSeconds
	c.ThrottleHoldMin = tuning.ThrottleHoldMin
	c.ThrottleRampUpPerSecond = tuning.ThrottleRampUpPerSecond
	c.ThrottleDecayPerSecond = tuning.ThrottleDecayPerSecond
	c.TargetSpeedErrorGain = tuning.TargetSpeedErrorGain
	c.TargetAccelGain = tuning.TargetAccelGain
	c.LongitudinalDeadband = tuning.LongitudinalDeadband
	c.LaunchThrottleMin = tuning.LaunchThrottleMin
	c.LaunchSpeedThreshold = tuning.LaunchSpeedThreshold
	c.LaunchTargetSpeedMargin = tuning.LaunchTargetSpeedMargin
	c.MaxTargetSpeedKPH = tuning.MaxTargetSpeedKPH
	c.MotionConfidenceOnThreshold = tuning.MotionConfidenceOnThreshold
	c.MotionConfidenceOffThreshold = tuning.MotionConfidenceOffThreshold
	c.MotionHoldSpeedMax = tuning.MotionHoldSpeedMax
	c.MinLateralConfidence = tuning.MinLateralConfidence
	c.MinLongitudinalConfidence = tuning.MinLongitudinalConfidence
}

func ValidateTuning(tuning Tuning) error {
	if tuning.HeadingDeadbandDeg < 0 {
		return fmt.Errorf("backend translation heading_deadband_deg must be >= 0")
	}
	if tuning.HeadingFullLockDeg <= 0 {
		return fmt.Errorf("backend translation heading_full_lock_deg must be > 0")
	}
	if tuning.LowSpeedSteerGain < 0 {
		return fmt.Errorf("backend translation low_speed_steer_gain must be >= 0")
	}
	if tuning.HighSpeedSteerGain < 0 {
		return fmt.Errorf("backend translation high_speed_steer_gain must be >= 0")
	}
	if tuning.SteerGainFadeSpeedMPS <= 0 {
		return fmt.Errorf("backend translation steer_gain_fade_speed_mps must be > 0")
	}
	if tuning.SteerResponseBlend < 0 || tuning.SteerResponseBlend > 1 {
		return fmt.Errorf("backend translation steer_response_blend must be in [0,1]")
	}
	if tuning.SteerCommandRatePerSecond < 0 {
		return fmt.Errorf("backend translation steer_command_rate_per_second must be >= 0")
	}
	if tuning.SteerDeadzone < 0 || tuning.SteerDeadzone >= 1 {
		return fmt.Errorf("backend translation steer_deadzone must be in [0,1)")
	}
	if tuning.MaxSteerScale <= 0 || tuning.MaxSteerScale > 1 {
		return fmt.Errorf("backend translation max_steer_scale must be in (0,1]")
	}
	if tuning.SteerOutputGain <= 0 {
		return fmt.Errorf("backend translation steer_output_gain must be > 0")
	}
	if tuning.ThrottleGain <= 0 {
		return fmt.Errorf("backend translation throttle_gain must be > 0")
	}
	if tuning.BrakeGain <= 0 {
		return fmt.Errorf("backend translation brake_gain must be > 0")
	}
	if tuning.BrakeActivationThreshold < 0 || tuning.BrakeActivationThreshold > 1 {
		return fmt.Errorf("backend translation brake_activation_threshold must be in [0,1]")
	}
	if tuning.OverspeedBrakeMarginMPS < 0 {
		return fmt.Errorf("backend translation overspeed_brake_margin_mps must be >= 0")
	}
	if tuning.BrakeReleaseMarginMPS < 0 {
		return fmt.Errorf("backend translation brake_release_margin_mps must be >= 0")
	}
	if tuning.BrakeReleaseMarginMPS > tuning.OverspeedBrakeMarginMPS {
		return fmt.Errorf("backend translation brake_release_margin_mps must be <= overspeed_brake_margin_mps")
	}
	if tuning.BrakeEnterHoldSeconds < 0 {
		return fmt.Errorf("backend translation brake_enter_hold_seconds must be >= 0")
	}
	if tuning.ThrottleHoldSeconds < 0 {
		return fmt.Errorf("backend translation throttle_hold_seconds must be >= 0")
	}
	if tuning.ThrottleHoldMin < 0 || tuning.ThrottleHoldMin > 1 {
		return fmt.Errorf("backend translation throttle_hold_min must be in [0,1]")
	}
	if tuning.ThrottleRampUpPerSecond < 0 {
		return fmt.Errorf("backend translation throttle_ramp_up_per_second must be >= 0")
	}
	if tuning.ThrottleDecayPerSecond < 0 {
		return fmt.Errorf("backend translation throttle_decay_per_second must be >= 0")
	}
	if tuning.TargetSpeedErrorGain < 0 {
		return fmt.Errorf("backend translation target_speed_error_gain must be >= 0")
	}
	if tuning.TargetAccelGain < 0 {
		return fmt.Errorf("backend translation target_accel_gain must be >= 0")
	}
	if tuning.LongitudinalDeadband < 0 {
		return fmt.Errorf("backend translation longitudinal_deadband must be >= 0")
	}
	if tuning.LaunchThrottleMin < 0 || tuning.LaunchThrottleMin > 1 {
		return fmt.Errorf("backend translation launch_throttle_min must be in [0,1]")
	}
	if tuning.LaunchSpeedThreshold < 0 {
		return fmt.Errorf("backend translation launch_speed_threshold must be >= 0")
	}
	if tuning.LaunchTargetSpeedMargin < 0 {
		return fmt.Errorf("backend translation launch_target_speed_margin must be >= 0")
	}
	if tuning.MaxTargetSpeedKPH < 0 {
		return fmt.Errorf("backend translation max_target_speed_kph must be >= 0")
	}
	if tuning.MotionConfidenceOnThreshold < 0 || tuning.MotionConfidenceOnThreshold > 1 {
		return fmt.Errorf("backend translation motion_confidence_on_threshold must be in [0,1]")
	}
	if tuning.MotionConfidenceOffThreshold < 0 || tuning.MotionConfidenceOffThreshold > 1 {
		return fmt.Errorf("backend translation motion_confidence_off_threshold must be in [0,1]")
	}
	if tuning.MotionConfidenceOffThreshold > tuning.MotionConfidenceOnThreshold {
		return fmt.Errorf("backend translation motion_confidence_off_threshold must be <= motion_confidence_on_threshold")
	}
	if tuning.MotionHoldSpeedMax < 0 {
		return fmt.Errorf("backend translation motion_hold_speed_max must be >= 0")
	}
	if tuning.MinLateralConfidence < 0 || tuning.MinLateralConfidence > 1 {
		return fmt.Errorf("backend translation min_lateral_confidence must be in [0,1]")
	}
	if tuning.MinLongitudinalConfidence < 0 || tuning.MinLongitudinalConfidence > 1 {
		return fmt.Errorf("backend translation min_longitudinal_confidence must be in [0,1]")
	}
	return nil
}

func SaveTuning(path string, tuning Tuning) error {
	if strings.TrimSpace(path) == "" {
		return fmt.Errorf("translation config path is not available")
	}
	if err := ValidateTuning(tuning); err != nil {
		return err
	}

	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	var parsed map[string]any
	if err := toml.Unmarshal(raw, &parsed); err != nil {
		return err
	}

	backend := ensureMap(parsed, "backend")
	section := ensureMap(backend, "translation")
	section["heading_deadband_deg"] = tuning.HeadingDeadbandDeg
	section["heading_full_lock_deg"] = tuning.HeadingFullLockDeg
	section["low_speed_steer_gain"] = tuning.LowSpeedSteerGain
	section["high_speed_steer_gain"] = tuning.HighSpeedSteerGain
	section["steer_gain_fade_speed_mps"] = tuning.SteerGainFadeSpeedMPS
	section["steer_response_blend"] = tuning.SteerResponseBlend
	section["steer_command_rate_per_second"] = tuning.SteerCommandRatePerSecond
	section["steer_deadzone"] = tuning.SteerDeadzone
	section["max_steer_scale"] = tuning.MaxSteerScale
	section["steer_output_gain"] = tuning.SteerOutputGain
	section["throttle_gain"] = tuning.ThrottleGain
	section["brake_gain"] = tuning.BrakeGain
	section["brake_activation_threshold"] = tuning.BrakeActivationThreshold
	section["overspeed_brake_margin_mps"] = tuning.OverspeedBrakeMarginMPS
	section["brake_release_margin_mps"] = tuning.BrakeReleaseMarginMPS
	section["brake_enter_hold_seconds"] = tuning.BrakeEnterHoldSeconds
	section["throttle_hold_seconds"] = tuning.ThrottleHoldSeconds
	section["throttle_hold_min"] = tuning.ThrottleHoldMin
	section["throttle_ramp_up_per_second"] = tuning.ThrottleRampUpPerSecond
	section["throttle_decay_per_second"] = tuning.ThrottleDecayPerSecond
	section["target_speed_error_gain"] = tuning.TargetSpeedErrorGain
	section["target_accel_gain"] = tuning.TargetAccelGain
	section["longitudinal_deadband"] = tuning.LongitudinalDeadband
	section["launch_throttle_min"] = tuning.LaunchThrottleMin
	section["launch_speed_threshold"] = tuning.LaunchSpeedThreshold
	section["launch_target_speed_margin"] = tuning.LaunchTargetSpeedMargin
	section["max_target_speed_kph"] = tuning.MaxTargetSpeedKPH
	section["motion_confidence_on_threshold"] = tuning.MotionConfidenceOnThreshold
	section["motion_confidence_off_threshold"] = tuning.MotionConfidenceOffThreshold
	section["motion_hold_speed_max"] = tuning.MotionHoldSpeedMax
	section["min_lateral_confidence"] = tuning.MinLateralConfidence
	section["min_longitudinal_confidence"] = tuning.MinLongitudinalConfidence

	encoded, err := toml.Marshal(parsed)
	if err != nil {
		return err
	}
	return os.WriteFile(path, encoded, 0o644)
}

func ensureMap(target map[string]any, key string) map[string]any {
	if existing, ok := target[key]; ok {
		if typed, ok := existing.(map[string]any); ok {
			return typed
		}
	}
	next := map[string]any{}
	target[key] = next
	return next
}
