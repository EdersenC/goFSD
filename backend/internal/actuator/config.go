package actuator

import (
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	toml "github.com/pelletier/go-toml/v2"
)

const (
	defaultTickHz                  = 60
	defaultActuatorURL             = "http://127.0.0.1:8080"
	defaultActuatorHTTPTimeout     = 500 * time.Millisecond
	defaultStaleTimeout            = 250 * time.Millisecond
	defaultActuatorProfile         = "smooth"
	defaultSteeringDeadzone        = 0.02
	defaultThrottleDeadzone        = 0.03
	defaultBrakeDeadzone           = 0.03
	defaultSteeringMaxDelta        = 0.06
	defaultThrottleMaxDelta        = 0.04
	defaultBrakeMaxDelta           = 0.04
	defaultSteeringAlpha           = 0.30
	defaultThrottleAlpha           = 0.20
	defaultBrakeAlpha              = 0.20
	defaultFallbackDecay           = true
	defaultFallbackSteerDecay      = 0.85
	defaultFallbackThrottleDecay   = 0.65
	defaultFallbackBrakeDecay      = 0.65
	defaultRecentCommandWindow     = 64
	defaultSteeringGain            = 1.0
	defaultThrottleGain            = 1.0
	defaultThrottleFloor           = 0.0
	defaultSpeedLimitKPH           = 17.0
	defaultOverspeedBrakeMarginKPH = 2.0
	defaultOverspeedBrake          = 0.25
	defaultModelBrakeThreshold     = 0.55
	defaultReverseLockoutSpeedKPH  = 1.0
	defaultTemporalEnabled         = false
	defaultTemporalLatency         = 50 * time.Millisecond
	defaultTemporalFreshPlanAge    = 150 * time.Millisecond
	defaultTemporalUsablePlanAge   = 400 * time.Millisecond
	defaultTemporalStalePlanAge    = 800 * time.Millisecond
	defaultTemporalSteeringDelta   = 0.08
	defaultTemporalThrottleDelta   = 0.05
	defaultTemporalBrakeDelta      = 0.08
)

type Config struct {
	TickHz                            int
	StaleTimeout                      time.Duration
	URL                               string
	RequestTimeout                    time.Duration
	Profile                           string
	SteeringDeadzone                  float64
	ThrottleDeadzone                  float64
	BrakeDeadzone                     float64
	SteeringMaxDelta                  float64
	ThrottleMaxDelta                  float64
	BrakeMaxDelta                     float64
	SteeringAlpha                     float64
	ThrottleAlpha                     float64
	BrakeAlpha                        float64
	FallbackDecayEnabled              bool
	FallbackSteeringDecay             float64
	FallbackThrottleDecay             float64
	FallbackBrakeDecay                float64
	RecentCommandWindow               int
	SteeringGain                      float64
	ThrottleGain                      float64
	ThrottleFloor                     float64
	SpeedLimitKPH                     float64
	OverspeedBrakeMarginKPH           float64
	OverspeedBrake                    float64
	ModelBrakeThreshold               float64
	ReverseLockoutSpeedKPH            float64
	TemporalHorizonActuatorEnabled    bool
	TemporalEstimatedActuationLatency time.Duration
	TemporalFreshPlanAge              time.Duration
	TemporalUsablePlanAge             time.Duration
	TemporalStalePlanAge              time.Duration
	TemporalSteeringMaxDelta          float64
	TemporalThrottleMaxDelta          float64
	TemporalBrakeMaxDelta             float64
}

type Tuning struct {
	SteeringGain            float64 `json:"steeringGain"`
	ThrottleGain            float64 `json:"throttleGain"`
	ThrottleFloor           float64 `json:"throttleFloor"`
	SpeedLimitKPH           float64 `json:"speedLimitKph"`
	OverspeedBrakeMarginKPH float64 `json:"overspeedBrakeMarginKph"`
	OverspeedBrake          float64 `json:"overspeedBrake"`
	ModelBrakeThreshold     float64 `json:"modelBrakeThreshold"`
	ReverseLockoutSpeedKPH  float64 `json:"reverseLockoutSpeedKph"`
}

type TuningState struct {
	Live          Tuning `json:"live"`
	Saved         Tuning `json:"saved"`
	ConfigPath    string `json:"configPath,omitempty"`
	SaveSupported bool   `json:"saveSupported"`
}

type configFile struct {
	Backend backendSection `toml:"backend"`
}

type backendSection struct {
	Actuator actuatorSection `toml:"actuator"`
}

type actuatorSection struct {
	TickHz                            int      `toml:"tick_hz"`
	StaleTimeout                      string   `toml:"stale_timeout"`
	URL                               string   `toml:"url"`
	RequestTimeout                    string   `toml:"request_timeout"`
	Profile                           string   `toml:"profile"`
	SteeringDeadzone                  *float64 `toml:"steering_deadzone"`
	ThrottleDeadzone                  *float64 `toml:"throttle_deadzone"`
	BrakeDeadzone                     *float64 `toml:"brake_deadzone"`
	SteeringMaxDelta                  *float64 `toml:"steering_max_delta"`
	ThrottleMaxDelta                  *float64 `toml:"throttle_max_delta"`
	BrakeMaxDelta                     *float64 `toml:"brake_max_delta"`
	SteeringAlpha                     *float64 `toml:"steering_alpha"`
	ThrottleAlpha                     *float64 `toml:"throttle_alpha"`
	BrakeAlpha                        *float64 `toml:"brake_alpha"`
	FallbackDecayEnabled              *bool    `toml:"fallback_decay_enabled"`
	FallbackSteeringDecay             *float64 `toml:"fallback_steering_decay"`
	FallbackThrottleDecay             *float64 `toml:"fallback_throttle_decay"`
	FallbackBrakeDecay                *float64 `toml:"fallback_brake_decay"`
	RecentCommandWindow               *int     `toml:"recent_command_window"`
	SteeringGain                      *float64 `toml:"steering_gain"`
	ThrottleGain                      *float64 `toml:"throttle_gain"`
	ThrottleFloor                     *float64 `toml:"throttle_floor"`
	SpeedLimitKPH                     *float64 `toml:"speed_limit_kph"`
	OverspeedBrakeMarginKPH           *float64 `toml:"overspeed_brake_margin_kph"`
	OverspeedBrake                    *float64 `toml:"overspeed_brake"`
	ModelBrakeThreshold               *float64 `toml:"model_brake_threshold"`
	ReverseLockoutSpeedKPH            *float64 `toml:"reverse_lockout_speed_kph"`
	TemporalHorizonActuatorEnabled    *bool    `toml:"temporal_horizon_actuator_enabled"`
	TemporalEstimatedActuationLatency string   `toml:"temporal_estimated_actuation_latency"`
	TemporalFreshPlanAge              string   `toml:"temporal_fresh_plan_age"`
	TemporalUsablePlanAge             string   `toml:"temporal_usable_plan_age"`
	TemporalStalePlanAge              string   `toml:"temporal_stale_plan_age"`
	TemporalSteeringMaxDelta          *float64 `toml:"temporal_steering_max_delta"`
	TemporalThrottleMaxDelta          *float64 `toml:"temporal_throttle_max_delta"`
	TemporalBrakeMaxDelta             *float64 `toml:"temporal_brake_max_delta"`
}

func DefaultConfig() Config {
	return Config{
		TickHz:                            defaultTickHz,
		StaleTimeout:                      defaultStaleTimeout,
		URL:                               defaultActuatorURL,
		RequestTimeout:                    defaultActuatorHTTPTimeout,
		Profile:                           defaultActuatorProfile,
		SteeringDeadzone:                  defaultSteeringDeadzone,
		ThrottleDeadzone:                  defaultThrottleDeadzone,
		BrakeDeadzone:                     defaultBrakeDeadzone,
		SteeringMaxDelta:                  defaultSteeringMaxDelta,
		ThrottleMaxDelta:                  defaultThrottleMaxDelta,
		BrakeMaxDelta:                     defaultBrakeMaxDelta,
		SteeringAlpha:                     defaultSteeringAlpha,
		ThrottleAlpha:                     defaultThrottleAlpha,
		BrakeAlpha:                        defaultBrakeAlpha,
		FallbackDecayEnabled:              defaultFallbackDecay,
		FallbackSteeringDecay:             defaultFallbackSteerDecay,
		FallbackThrottleDecay:             defaultFallbackThrottleDecay,
		FallbackBrakeDecay:                defaultFallbackBrakeDecay,
		RecentCommandWindow:               defaultRecentCommandWindow,
		SteeringGain:                      defaultSteeringGain,
		ThrottleGain:                      defaultThrottleGain,
		ThrottleFloor:                     defaultThrottleFloor,
		SpeedLimitKPH:                     defaultSpeedLimitKPH,
		OverspeedBrakeMarginKPH:           defaultOverspeedBrakeMarginKPH,
		OverspeedBrake:                    defaultOverspeedBrake,
		ModelBrakeThreshold:               defaultModelBrakeThreshold,
		ReverseLockoutSpeedKPH:            defaultReverseLockoutSpeedKPH,
		TemporalHorizonActuatorEnabled:    defaultTemporalEnabled,
		TemporalEstimatedActuationLatency: defaultTemporalLatency,
		TemporalFreshPlanAge:              defaultTemporalFreshPlanAge,
		TemporalUsablePlanAge:             defaultTemporalUsablePlanAge,
		TemporalStalePlanAge:              defaultTemporalStalePlanAge,
		TemporalSteeringMaxDelta:          defaultTemporalSteeringDelta,
		TemporalThrottleMaxDelta:          defaultTemporalThrottleDelta,
		TemporalBrakeMaxDelta:             defaultTemporalBrakeDelta,
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

	section := parsed.Backend.Actuator
	if section.TickHz > 0 {
		cfg.TickHz = section.TickHz
	}
	if value := strings.TrimSpace(section.Profile); value != "" {
		cfg.Profile = value
	}
	if value := strings.TrimSpace(section.URL); value != "" {
		cfg.URL = strings.TrimRight(value, "/")
	}
	if value := strings.TrimSpace(section.StaleTimeout); value != "" {
		duration, err := time.ParseDuration(value)
		if err != nil {
			return Config{}, fmt.Errorf("invalid backend.actuator.stale_timeout: %w", err)
		}
		cfg.StaleTimeout = duration
	}
	if value := strings.TrimSpace(section.RequestTimeout); value != "" {
		duration, err := time.ParseDuration(value)
		if err != nil {
			return Config{}, fmt.Errorf("invalid backend.actuator.request_timeout: %w", err)
		}
		cfg.RequestTimeout = duration
	}
	applyActuatorProfile(&cfg, cfg.Profile)
	if section.SteeringDeadzone != nil {
		cfg.SteeringDeadzone = *section.SteeringDeadzone
	}
	if section.ThrottleDeadzone != nil {
		cfg.ThrottleDeadzone = *section.ThrottleDeadzone
	}
	if section.BrakeDeadzone != nil {
		cfg.BrakeDeadzone = *section.BrakeDeadzone
	}
	if section.SteeringMaxDelta != nil {
		cfg.SteeringMaxDelta = *section.SteeringMaxDelta
	}
	if section.ThrottleMaxDelta != nil {
		cfg.ThrottleMaxDelta = *section.ThrottleMaxDelta
	}
	if section.BrakeMaxDelta != nil {
		cfg.BrakeMaxDelta = *section.BrakeMaxDelta
	}
	if section.SteeringAlpha != nil {
		cfg.SteeringAlpha = *section.SteeringAlpha
	}
	if section.ThrottleAlpha != nil {
		cfg.ThrottleAlpha = *section.ThrottleAlpha
	}
	if section.BrakeAlpha != nil {
		cfg.BrakeAlpha = *section.BrakeAlpha
	}
	if section.FallbackDecayEnabled != nil {
		cfg.FallbackDecayEnabled = *section.FallbackDecayEnabled
	}
	if section.FallbackSteeringDecay != nil {
		cfg.FallbackSteeringDecay = *section.FallbackSteeringDecay
	}
	if section.FallbackThrottleDecay != nil {
		cfg.FallbackThrottleDecay = *section.FallbackThrottleDecay
	}
	if section.FallbackBrakeDecay != nil {
		cfg.FallbackBrakeDecay = *section.FallbackBrakeDecay
	}
	if section.RecentCommandWindow != nil {
		cfg.RecentCommandWindow = *section.RecentCommandWindow
	}
	if section.SteeringGain != nil {
		cfg.SteeringGain = *section.SteeringGain
	}
	if section.ThrottleGain != nil {
		cfg.ThrottleGain = *section.ThrottleGain
	}
	if section.ThrottleFloor != nil {
		cfg.ThrottleFloor = *section.ThrottleFloor
	}
	if section.SpeedLimitKPH != nil {
		cfg.SpeedLimitKPH = *section.SpeedLimitKPH
	}
	if section.OverspeedBrakeMarginKPH != nil {
		cfg.OverspeedBrakeMarginKPH = *section.OverspeedBrakeMarginKPH
	}
	if section.OverspeedBrake != nil {
		cfg.OverspeedBrake = *section.OverspeedBrake
	}
	if section.ModelBrakeThreshold != nil {
		cfg.ModelBrakeThreshold = *section.ModelBrakeThreshold
	}
	if section.ReverseLockoutSpeedKPH != nil {
		cfg.ReverseLockoutSpeedKPH = *section.ReverseLockoutSpeedKPH
	}
	if section.TemporalHorizonActuatorEnabled != nil {
		cfg.TemporalHorizonActuatorEnabled = *section.TemporalHorizonActuatorEnabled
	}
	if value := strings.TrimSpace(section.TemporalEstimatedActuationLatency); value != "" {
		duration, err := time.ParseDuration(value)
		if err != nil {
			return Config{}, fmt.Errorf("invalid backend.actuator.temporal_estimated_actuation_latency: %w", err)
		}
		cfg.TemporalEstimatedActuationLatency = duration
	}
	if value := strings.TrimSpace(section.TemporalFreshPlanAge); value != "" {
		duration, err := time.ParseDuration(value)
		if err != nil {
			return Config{}, fmt.Errorf("invalid backend.actuator.temporal_fresh_plan_age: %w", err)
		}
		cfg.TemporalFreshPlanAge = duration
	}
	if value := strings.TrimSpace(section.TemporalUsablePlanAge); value != "" {
		duration, err := time.ParseDuration(value)
		if err != nil {
			return Config{}, fmt.Errorf("invalid backend.actuator.temporal_usable_plan_age: %w", err)
		}
		cfg.TemporalUsablePlanAge = duration
	}
	if value := strings.TrimSpace(section.TemporalStalePlanAge); value != "" {
		duration, err := time.ParseDuration(value)
		if err != nil {
			return Config{}, fmt.Errorf("invalid backend.actuator.temporal_stale_plan_age: %w", err)
		}
		cfg.TemporalStalePlanAge = duration
	}
	if section.TemporalSteeringMaxDelta != nil {
		cfg.TemporalSteeringMaxDelta = *section.TemporalSteeringMaxDelta
	}
	if section.TemporalThrottleMaxDelta != nil {
		cfg.TemporalThrottleMaxDelta = *section.TemporalThrottleMaxDelta
	}
	if section.TemporalBrakeMaxDelta != nil {
		cfg.TemporalBrakeMaxDelta = *section.TemporalBrakeMaxDelta
	}

	if cfg.TickHz < 1 {
		return Config{}, fmt.Errorf("backend actuator tick_hz must be > 0")
	}
	if cfg.StaleTimeout <= 0 {
		return Config{}, fmt.Errorf("backend actuator stale_timeout must be > 0")
	}
	if cfg.RequestTimeout <= 0 {
		return Config{}, fmt.Errorf("backend actuator request_timeout must be > 0")
	}
	if err := validateProcessorConfig(cfg); err != nil {
		return Config{}, err
	}

	return cfg, nil
}

func applyActuatorProfile(cfg *Config, rawProfile string) {
	profile := strings.ToLower(strings.TrimSpace(rawProfile))
	switch profile {
	case "", "smooth":
		cfg.Profile = "smooth"
		cfg.SteeringDeadzone = 0.02
		cfg.ThrottleDeadzone = 0.03
		cfg.BrakeDeadzone = 0.03
		cfg.SteeringMaxDelta = 0.06
		cfg.ThrottleMaxDelta = 0.04
		cfg.BrakeMaxDelta = 0.04
		cfg.SteeringAlpha = 0.30
		cfg.ThrottleAlpha = 0.20
		cfg.BrakeAlpha = 0.20
	case "responsive":
		cfg.Profile = "responsive"
		cfg.SteeringDeadzone = 0.02
		cfg.ThrottleDeadzone = 0.03
		cfg.BrakeDeadzone = 0.03
		cfg.SteeringMaxDelta = 0.08
		cfg.ThrottleMaxDelta = 0.05
		cfg.BrakeMaxDelta = 0.05
		cfg.SteeringAlpha = 0.35
		cfg.ThrottleAlpha = 0.25
		cfg.BrakeAlpha = 0.25
	case "debug_raw":
		cfg.Profile = "debug_raw"
		cfg.SteeringDeadzone = 0
		cfg.ThrottleDeadzone = 0
		cfg.BrakeDeadzone = 0
		cfg.SteeringMaxDelta = 0
		cfg.ThrottleMaxDelta = 0
		cfg.BrakeMaxDelta = 0
		cfg.SteeringAlpha = 1
		cfg.ThrottleAlpha = 1
		cfg.BrakeAlpha = 1
	default:
		cfg.Profile = profile
	}
}

func validateProcessorConfig(cfg Config) error {
	switch cfg.Profile {
	case "smooth", "responsive", "debug_raw":
	default:
		return fmt.Errorf("backend actuator profile must be one of smooth, responsive, debug_raw")
	}
	for name, value := range map[string]float64{
		"steering_deadzone":           cfg.SteeringDeadzone,
		"throttle_deadzone":           cfg.ThrottleDeadzone,
		"brake_deadzone":              cfg.BrakeDeadzone,
		"steering_max_delta":          cfg.SteeringMaxDelta,
		"throttle_max_delta":          cfg.ThrottleMaxDelta,
		"brake_max_delta":             cfg.BrakeMaxDelta,
		"fallback_steering_decay":     cfg.FallbackSteeringDecay,
		"fallback_throttle_decay":     cfg.FallbackThrottleDecay,
		"fallback_brake_decay":        cfg.FallbackBrakeDecay,
		"steering_gain":               cfg.SteeringGain,
		"throttle_gain":               cfg.ThrottleGain,
		"throttle_floor":              cfg.ThrottleFloor,
		"speed_limit_kph":             cfg.SpeedLimitKPH,
		"overspeed_brake_margin_kph":  cfg.OverspeedBrakeMarginKPH,
		"overspeed_brake":             cfg.OverspeedBrake,
		"model_brake_threshold":       cfg.ModelBrakeThreshold,
		"reverse_lockout_speed_kph":   cfg.ReverseLockoutSpeedKPH,
		"temporal_steering_max_delta": cfg.TemporalSteeringMaxDelta,
		"temporal_throttle_max_delta": cfg.TemporalThrottleMaxDelta,
		"temporal_brake_max_delta":    cfg.TemporalBrakeMaxDelta,
	} {
		if value < 0 {
			return fmt.Errorf("backend actuator %s must be >= 0", name)
		}
	}
	for name, value := range map[string]float64{
		"steering_alpha": cfg.SteeringAlpha,
		"throttle_alpha": cfg.ThrottleAlpha,
		"brake_alpha":    cfg.BrakeAlpha,
	} {
		if value < 0 || value > 1 {
			return fmt.Errorf("backend actuator %s must be in [0,1]", name)
		}
	}
	if cfg.SteeringDeadzone >= 1 || cfg.ThrottleDeadzone >= 1 || cfg.BrakeDeadzone >= 1 {
		return fmt.Errorf("backend actuator deadzones must be < 1")
	}
	if cfg.FallbackSteeringDecay > 1 || cfg.FallbackThrottleDecay > 1 || cfg.FallbackBrakeDecay > 1 {
		return fmt.Errorf("backend actuator fallback decay values must be <= 1")
	}
	if cfg.SteeringGain == 0 {
		return fmt.Errorf("backend actuator steering_gain must be > 0")
	}
	if cfg.ThrottleGain == 0 {
		return fmt.Errorf("backend actuator throttle_gain must be > 0")
	}
	if cfg.ThrottleFloor > 1 {
		return fmt.Errorf("backend actuator throttle_floor must be in [0,1]")
	}
	if cfg.OverspeedBrake > 1 {
		return fmt.Errorf("backend actuator overspeed_brake must be in [0,1]")
	}
	if cfg.ModelBrakeThreshold <= 0.5 || cfg.ModelBrakeThreshold > 1 {
		return fmt.Errorf("backend actuator model_brake_threshold must be in (0.5, 1]")
	}
	if cfg.RecentCommandWindow < 1 {
		return fmt.Errorf("backend actuator recent_command_window must be > 0")
	}
	if cfg.TemporalEstimatedActuationLatency < 0 {
		return fmt.Errorf("backend actuator temporal_estimated_actuation_latency must be >= 0")
	}
	if cfg.TemporalFreshPlanAge <= 0 || cfg.TemporalUsablePlanAge <= 0 || cfg.TemporalStalePlanAge <= 0 {
		return fmt.Errorf("backend actuator temporal plan age thresholds must be > 0")
	}
	if cfg.TemporalFreshPlanAge > cfg.TemporalUsablePlanAge || cfg.TemporalUsablePlanAge > cfg.TemporalStalePlanAge {
		return fmt.Errorf("backend actuator temporal plan ages must satisfy fresh <= usable <= stale")
	}
	return nil
}

func (c Config) Tuning() Tuning {
	return Tuning{
		SteeringGain:            c.SteeringGain,
		ThrottleGain:            c.ThrottleGain,
		ThrottleFloor:           c.ThrottleFloor,
		SpeedLimitKPH:           c.SpeedLimitKPH,
		OverspeedBrakeMarginKPH: c.OverspeedBrakeMarginKPH,
		OverspeedBrake:          c.OverspeedBrake,
		ModelBrakeThreshold:     c.ModelBrakeThreshold,
		ReverseLockoutSpeedKPH:  c.ReverseLockoutSpeedKPH,
	}
}

func (c *Config) ApplyTuning(tuning Tuning) {
	c.SteeringGain = tuning.SteeringGain
	c.ThrottleGain = tuning.ThrottleGain
	c.ThrottleFloor = tuning.ThrottleFloor
	c.SpeedLimitKPH = tuning.SpeedLimitKPH
	c.OverspeedBrakeMarginKPH = tuning.OverspeedBrakeMarginKPH
	c.OverspeedBrake = tuning.OverspeedBrake
	c.ModelBrakeThreshold = tuning.ModelBrakeThreshold
	c.ReverseLockoutSpeedKPH = tuning.ReverseLockoutSpeedKPH
}

func ValidateTuning(tuning Tuning) error {
	if tuning.SteeringGain <= 0 {
		return fmt.Errorf("backend actuator steering_gain must be > 0")
	}
	if tuning.ThrottleGain <= 0 {
		return fmt.Errorf("backend actuator throttle_gain must be > 0")
	}
	if tuning.ThrottleFloor < 0 || tuning.ThrottleFloor > 1 {
		return fmt.Errorf("backend actuator throttle_floor must be in [0,1]")
	}
	if tuning.SpeedLimitKPH < 0 {
		return fmt.Errorf("backend actuator speed_limit_kph must be >= 0")
	}
	if tuning.OverspeedBrakeMarginKPH < 0 {
		return fmt.Errorf("backend actuator overspeed_brake_margin_kph must be >= 0")
	}
	if tuning.OverspeedBrake < 0 || tuning.OverspeedBrake > 1 {
		return fmt.Errorf("backend actuator overspeed_brake must be in [0,1]")
	}
	if tuning.ModelBrakeThreshold <= 0.5 || tuning.ModelBrakeThreshold > 1 {
		return fmt.Errorf("backend actuator model_brake_threshold must be in (0.5, 1]")
	}
	if tuning.ReverseLockoutSpeedKPH < 0 {
		return fmt.Errorf("backend actuator reverse_lockout_speed_kph must be >= 0")
	}
	return nil
}

func SaveTuning(path string, tuning Tuning) error {
	if strings.TrimSpace(path) == "" {
		return fmt.Errorf("actuator config path is not available")
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
	section := ensureMap(backend, "actuator")
	section["steering_gain"] = tuning.SteeringGain
	section["throttle_gain"] = tuning.ThrottleGain
	section["throttle_floor"] = tuning.ThrottleFloor
	section["speed_limit_kph"] = tuning.SpeedLimitKPH
	section["overspeed_brake_margin_kph"] = tuning.OverspeedBrakeMarginKPH
	section["overspeed_brake"] = tuning.OverspeedBrake
	section["model_brake_threshold"] = tuning.ModelBrakeThreshold
	section["reverse_lockout_speed_kph"] = tuning.ReverseLockoutSpeedKPH

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
