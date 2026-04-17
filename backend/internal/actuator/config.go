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
	defaultTickHz              = 60
	defaultSteerDeadzone       = 0.02
	defaultMaxSteerScale       = 1.0
	defaultSteerInputGain      = 4.0
	defaultThrottleInputGain   = 2.0
	defaultBrakeInputGain      = 2.0
	defaultSteerRatePerSecond  = 6.0
	defaultThrottleRatePerSec  = 4.0
	defaultBrakeRatePerSecond  = 6.0
	defaultActuatorURL         = "http://127.0.0.1:8080"
	defaultActuatorHTTPTimeout = 500 * time.Millisecond
	defaultStaleTimeout        = 250 * time.Millisecond
)

type Config struct {
	TickHz             int
	StaleTimeout       time.Duration
	SteerDeadzone      float64
	MaxSteerScale      float64
	SteerInputGain     float64
	ThrottleInputGain  float64
	BrakeInputGain     float64
	SteerRatePerSecond float64
	ThrottleRatePerSec float64
	BrakeRatePerSecond float64
	URL                string
	RequestTimeout     time.Duration
}

type Tuning struct {
	SteerDeadzone      float64 `json:"steerDeadzone"`
	MaxSteerScale      float64 `json:"maxSteerScale"`
	SteerInputGain     float64 `json:"steerInputGain"`
	ThrottleInputGain  float64 `json:"throttleInputGain"`
	BrakeInputGain     float64 `json:"brakeInputGain"`
	SteerRatePerSecond float64 `json:"steerRatePerSecond"`
	ThrottleRatePerSec float64 `json:"throttleRatePerSecond"`
	BrakeRatePerSecond float64 `json:"brakeRatePerSecond"`
}

type configFile struct {
	Backend backendSection `toml:"backend"`
}

type backendSection struct {
	Actuator actuatorSection `toml:"actuator"`
}

type actuatorSection struct {
	TickHz             int     `toml:"tick_hz"`
	StaleTimeout       string  `toml:"stale_timeout"`
	SteerDeadzone      float64 `toml:"steer_deadzone"`
	MaxSteerScale      float64 `toml:"max_steer_scale"`
	SteerInputGain     float64 `toml:"steer_input_gain"`
	ThrottleInputGain  float64 `toml:"throttle_input_gain"`
	BrakeInputGain     float64 `toml:"brake_input_gain"`
	SteerRatePerSecond float64 `toml:"steer_rate_per_second"`
	ThrottleRatePerSec float64 `toml:"throttle_rate_per_second"`
	BrakeRatePerSecond float64 `toml:"brake_rate_per_second"`
	URL                string  `toml:"url"`
	RequestTimeout     string  `toml:"request_timeout"`
}

func DefaultConfig() Config {
	return Config{
		TickHz:             defaultTickHz,
		StaleTimeout:       defaultStaleTimeout,
		SteerDeadzone:      defaultSteerDeadzone,
		MaxSteerScale:      defaultMaxSteerScale,
		SteerInputGain:     defaultSteerInputGain,
		ThrottleInputGain:  defaultThrottleInputGain,
		BrakeInputGain:     defaultBrakeInputGain,
		SteerRatePerSecond: defaultSteerRatePerSecond,
		ThrottleRatePerSec: defaultThrottleRatePerSec,
		BrakeRatePerSecond: defaultBrakeRatePerSecond,
		URL:                defaultActuatorURL,
		RequestTimeout:     defaultActuatorHTTPTimeout,
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
	if section.SteerDeadzone > 0 {
		cfg.SteerDeadzone = section.SteerDeadzone
	}
	if section.MaxSteerScale > 0 {
		cfg.MaxSteerScale = section.MaxSteerScale
	}
	if section.SteerInputGain > 0 {
		cfg.SteerInputGain = section.SteerInputGain
	}
	if section.ThrottleInputGain > 0 {
		cfg.ThrottleInputGain = section.ThrottleInputGain
	}
	if section.BrakeInputGain > 0 {
		cfg.BrakeInputGain = section.BrakeInputGain
	}
	if section.SteerRatePerSecond > 0 {
		cfg.SteerRatePerSecond = section.SteerRatePerSecond
	}
	if section.ThrottleRatePerSec > 0 {
		cfg.ThrottleRatePerSec = section.ThrottleRatePerSec
	}
	if section.BrakeRatePerSecond > 0 {
		cfg.BrakeRatePerSecond = section.BrakeRatePerSecond
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

	if cfg.TickHz < 1 {
		return Config{}, fmt.Errorf("backend actuator tick_hz must be > 0")
	}
	if cfg.StaleTimeout <= 0 {
		return Config{}, fmt.Errorf("backend actuator stale_timeout must be > 0")
	}
	if err := ValidateTuning(cfg.Tuning()); err != nil {
		return Config{}, err
	}
	if cfg.RequestTimeout <= 0 {
		return Config{}, fmt.Errorf("backend actuator request_timeout must be > 0")
	}

	return cfg, nil
}

func (c Config) Tuning() Tuning {
	return Tuning{
		SteerDeadzone:      c.SteerDeadzone,
		MaxSteerScale:      c.MaxSteerScale,
		SteerInputGain:     c.SteerInputGain,
		ThrottleInputGain:  c.ThrottleInputGain,
		BrakeInputGain:     c.BrakeInputGain,
		SteerRatePerSecond: c.SteerRatePerSecond,
		ThrottleRatePerSec: c.ThrottleRatePerSec,
		BrakeRatePerSecond: c.BrakeRatePerSecond,
	}
}

func (c *Config) ApplyTuning(tuning Tuning) {
	c.SteerDeadzone = tuning.SteerDeadzone
	c.MaxSteerScale = tuning.MaxSteerScale
	c.SteerInputGain = tuning.SteerInputGain
	c.ThrottleInputGain = tuning.ThrottleInputGain
	c.BrakeInputGain = tuning.BrakeInputGain
	c.SteerRatePerSecond = tuning.SteerRatePerSecond
	c.ThrottleRatePerSec = tuning.ThrottleRatePerSec
	c.BrakeRatePerSecond = tuning.BrakeRatePerSecond
}

func ValidateTuning(tuning Tuning) error {
	if tuning.SteerDeadzone < 0 || tuning.SteerDeadzone >= 1 {
		return fmt.Errorf("backend actuator steer_deadzone must be in [0,1)")
	}
	if tuning.MaxSteerScale <= 0 || tuning.MaxSteerScale > 1 {
		return fmt.Errorf("backend actuator max_steer_scale must be in (0,1]")
	}
	if tuning.SteerInputGain <= 0 {
		return fmt.Errorf("backend actuator steer_input_gain must be > 0")
	}
	if tuning.ThrottleInputGain <= 0 {
		return fmt.Errorf("backend actuator throttle_input_gain must be > 0")
	}
	if tuning.BrakeInputGain <= 0 {
		return fmt.Errorf("backend actuator brake_input_gain must be > 0")
	}
	if tuning.SteerRatePerSecond <= 0 {
		return fmt.Errorf("backend actuator steer_rate_per_second must be > 0")
	}
	if tuning.ThrottleRatePerSec <= 0 {
		return fmt.Errorf("backend actuator throttle_rate_per_second must be > 0")
	}
	if tuning.BrakeRatePerSecond <= 0 {
		return fmt.Errorf("backend actuator brake_rate_per_second must be > 0")
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
	actuatorSection := ensureMap(backend, "actuator")
	actuatorSection["steer_deadzone"] = tuning.SteerDeadzone
	actuatorSection["max_steer_scale"] = tuning.MaxSteerScale
	actuatorSection["steer_input_gain"] = tuning.SteerInputGain
	actuatorSection["throttle_input_gain"] = tuning.ThrottleInputGain
	actuatorSection["brake_input_gain"] = tuning.BrakeInputGain
	actuatorSection["steer_rate_per_second"] = tuning.SteerRatePerSecond
	actuatorSection["throttle_rate_per_second"] = tuning.ThrottleRatePerSec
	actuatorSection["brake_rate_per_second"] = tuning.BrakeRatePerSecond

	encoded, err := toml.Marshal(parsed)
	if err != nil {
		return err
	}

	return os.WriteFile(path, encoded, 0644)
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
