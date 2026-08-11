package actuator

import (
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	"awesomeProject/internal/stopsigncontrol"

	toml "github.com/pelletier/go-toml/v2"
)

const (
	defaultTickHz                   = 60
	defaultActuatorURL              = "http://127.0.0.1:8080"
	defaultActuatorHTTPTimeout      = 500 * time.Millisecond
	defaultStaleTimeout             = 250 * time.Millisecond
	defaultSteeringGain             = 1.0
	defaultThrottleGain             = 1.0
	defaultThrottleFloor            = 0.0
	defaultSpeedLimitKPH            = 17.0
	defaultOverspeedBrakeMarginKPH  = 2.0
	defaultOverspeedBrake           = 0.25
	defaultModelBrakeThreshold      = 0.55
	defaultReverseLockoutSpeedKPH   = 1.0
	defaultStopSignPlanTimeout      = 400 * time.Millisecond
	defaultStopSignTelemetryTimeout = 250 * time.Millisecond
	defaultStopSignLatency          = 50 * time.Millisecond
)

type StopSignCalibration struct {
	Verified           bool   `json:"verified"`
	ProfileID          string `json:"profileId"`
	VehicleModelHash   int64  `json:"vehicleModelHash"`
	GameBuild          string `json:"gameBuild"`
	AdapterVersion     string `json:"adapterVersion"`
	SteeringConvention string `json:"steeringConvention"`
}

type Config struct {
	TickHz                            int
	StaleTimeout                      time.Duration
	URL                               string
	RequestTimeout                    time.Duration
	SteeringGain                      float64
	ThrottleGain                      float64
	ThrottleFloor                     float64
	SpeedLimitKPH                     float64
	OverspeedBrakeMarginKPH           float64
	OverspeedBrake                    float64
	ModelBrakeThreshold               float64
	ReverseLockoutSpeedKPH            float64
	StopSignController                stopsigncontrol.Config
	StopSignCalibration               StopSignCalibration
	StopSignPlanTimeout               time.Duration
	StopSignTelemetryTimeout          time.Duration
	StopSignEstimatedActuationLatency time.Duration
	StopSignExpectedHorizonDtMs       []int
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
	Actuator           actuatorSection           `toml:"actuator"`
	StopSignController stopSignControllerSection `toml:"stop_sign_controller"`
}

type stopSignControllerSection struct {
	CalibrationVerified             *bool                                      `toml:"calibration_verified"`
	CalibrationProfileID            string                                     `toml:"calibration_profile_id"`
	VehicleModelHash                *int64                                     `toml:"vehicle_model_hash"`
	GameBuild                       string                                     `toml:"game_build"`
	AdapterVersion                  string                                     `toml:"adapter_version"`
	SteeringConvention              string                                     `toml:"steering_convention"`
	SteeringProfile                 []stopsigncontrol.SteeringCalibrationPoint `toml:"steering_profile"`
	StraightApproachOnly            *bool                                      `toml:"straight_approach_only"`
	SteeringFeedbackGain            *float64                                   `toml:"steering_feedback_gain"`
	SteeringKi                      *float64                                   `toml:"steering_ki"`
	SteeringIntegralMin             *float64                                   `toml:"steering_integral_min"`
	SteeringIntegralMax             *float64                                   `toml:"steering_integral_max"`
	SteeringSlewPerSecond           *float64                                   `toml:"steering_slew_per_second"`
	SpeedKp                         *float64                                   `toml:"speed_kp"`
	SpeedKi                         *float64                                   `toml:"speed_ki"`
	SpeedIntegralMin                *float64                                   `toml:"speed_integral_min"`
	SpeedIntegralMax                *float64                                   `toml:"speed_integral_max"`
	MaxThrottleEffort               *float64                                   `toml:"max_throttle_effort"`
	MaxBrakeEffort                  *float64                                   `toml:"max_brake_effort"`
	LongitudinalSlewPerSecond       *float64                                   `toml:"longitudinal_slew_per_second"`
	StopProbabilityThreshold        *float64                                   `toml:"stop_probability_threshold"`
	StopProbabilityReleaseThreshold *float64                                   `toml:"stop_probability_release_threshold"`
	HoldSpeedMPS                    *float64                                   `toml:"hold_speed_mps"`
	StopBrakeEffort                 *float64                                   `toml:"stop_brake_effort"`
	MaxDesiredSpeedMPS              *float64                                   `toml:"max_desired_speed_mps"`
	MaxDT                           string                                     `toml:"max_dt"`
	PlanTimeout                     string                                     `toml:"plan_timeout"`
	TelemetryTimeout                string                                     `toml:"telemetry_timeout"`
	EstimatedActuationLatency       string                                     `toml:"estimated_actuation_latency"`
	ExpectedHorizonDtMs             []int                                      `toml:"expected_horizon_dt_ms"`
}

type actuatorSection struct {
	TickHz                  int      `toml:"tick_hz"`
	StaleTimeout            string   `toml:"stale_timeout"`
	URL                     string   `toml:"url"`
	RequestTimeout          string   `toml:"request_timeout"`
	SteeringGain            *float64 `toml:"steering_gain"`
	ThrottleGain            *float64 `toml:"throttle_gain"`
	ThrottleFloor           *float64 `toml:"throttle_floor"`
	SpeedLimitKPH           *float64 `toml:"speed_limit_kph"`
	OverspeedBrakeMarginKPH *float64 `toml:"overspeed_brake_margin_kph"`
	OverspeedBrake          *float64 `toml:"overspeed_brake"`
	ModelBrakeThreshold     *float64 `toml:"model_brake_threshold"`
	ReverseLockoutSpeedKPH  *float64 `toml:"reverse_lockout_speed_kph"`
}

func DefaultConfig() Config {
	return Config{
		TickHz:                            defaultTickHz,
		StaleTimeout:                      defaultStaleTimeout,
		URL:                               defaultActuatorURL,
		RequestTimeout:                    defaultActuatorHTTPTimeout,
		SteeringGain:                      defaultSteeringGain,
		ThrottleGain:                      defaultThrottleGain,
		ThrottleFloor:                     defaultThrottleFloor,
		SpeedLimitKPH:                     defaultSpeedLimitKPH,
		OverspeedBrakeMarginKPH:           defaultOverspeedBrakeMarginKPH,
		OverspeedBrake:                    defaultOverspeedBrake,
		ModelBrakeThreshold:               defaultModelBrakeThreshold,
		ReverseLockoutSpeedKPH:            defaultReverseLockoutSpeedKPH,
		StopSignController:                defaultStopSignControllerConfig(),
		StopSignPlanTimeout:               defaultStopSignPlanTimeout,
		StopSignTelemetryTimeout:          defaultStopSignTelemetryTimeout,
		StopSignEstimatedActuationLatency: defaultStopSignLatency,
		StopSignExpectedHorizonDtMs:       []int{250, 500, 1000, 2000, 3000, 5000},
	}
}

func defaultStopSignControllerConfig() stopsigncontrol.Config {
	return stopsigncontrol.Config{
		SteeringProfile: []stopsigncontrol.SteeringCalibrationPoint{
			{WheelSteer: -1, Command: -1},
			{WheelSteer: 0, Command: 0},
			{WheelSteer: 1, Command: 1},
		},
		StraightApproachOnly:            true,
		SteeringFeedbackGain:            0.35,
		SteeringKi:                      0.10,
		SteeringIntegralMin:             -0.5,
		SteeringIntegralMax:             0.5,
		SteeringSlewPerSecond:           4.0,
		SpeedKp:                         0.45,
		SpeedKi:                         0.12,
		SpeedIntegralMin:                -2.0,
		SpeedIntegralMax:                2.0,
		MaxThrottleEffort:               0.65,
		MaxBrakeEffort:                  0.70,
		LongitudinalSlewPerSecond:       2.0,
		StopProbabilityThreshold:        0.65,
		StopProbabilityReleaseThreshold: 0.35,
		HoldSpeedMPS:                    0.15,
		StopBrakeEffort:                 0.45,
		MaxDesiredSpeedMPS:              stopsigncontrol.StopSignMotionPlanMaxSpeedMPS,
		MaxDT:                           250 * time.Millisecond,
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
	if err := applyStopSignControllerSection(&cfg, parsed.Backend.StopSignController); err != nil {
		return Config{}, err
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
	if err := validateActuatorSafetyConfig(cfg); err != nil {
		return Config{}, err
	}
	if err := validateStopSignControllerConfig(cfg); err != nil {
		return Config{}, err
	}

	return cfg, nil
}

func applyStopSignControllerSection(cfg *Config, section stopSignControllerSection) error {
	controller := &cfg.StopSignController
	if section.CalibrationVerified != nil {
		cfg.StopSignCalibration.Verified = *section.CalibrationVerified
	}
	if value := strings.TrimSpace(section.CalibrationProfileID); value != "" {
		cfg.StopSignCalibration.ProfileID = value
	}
	if section.VehicleModelHash != nil {
		cfg.StopSignCalibration.VehicleModelHash = *section.VehicleModelHash
	}
	if value := strings.TrimSpace(section.GameBuild); value != "" {
		cfg.StopSignCalibration.GameBuild = value
	}
	if value := strings.TrimSpace(section.AdapterVersion); value != "" {
		cfg.StopSignCalibration.AdapterVersion = value
	}
	if value := strings.TrimSpace(section.SteeringConvention); value != "" {
		cfg.StopSignCalibration.SteeringConvention = value
	}
	if len(section.SteeringProfile) > 0 {
		controller.SteeringProfile = append([]stopsigncontrol.SteeringCalibrationPoint(nil), section.SteeringProfile...)
	}
	if section.StraightApproachOnly != nil {
		controller.StraightApproachOnly = *section.StraightApproachOnly
	}
	assignFloat := func(destination *float64, value *float64) {
		if value != nil {
			*destination = *value
		}
	}
	assignFloat(&controller.SteeringFeedbackGain, section.SteeringFeedbackGain)
	assignFloat(&controller.SteeringKi, section.SteeringKi)
	assignFloat(&controller.SteeringIntegralMin, section.SteeringIntegralMin)
	assignFloat(&controller.SteeringIntegralMax, section.SteeringIntegralMax)
	assignFloat(&controller.SteeringSlewPerSecond, section.SteeringSlewPerSecond)
	assignFloat(&controller.SpeedKp, section.SpeedKp)
	assignFloat(&controller.SpeedKi, section.SpeedKi)
	assignFloat(&controller.SpeedIntegralMin, section.SpeedIntegralMin)
	assignFloat(&controller.SpeedIntegralMax, section.SpeedIntegralMax)
	assignFloat(&controller.MaxThrottleEffort, section.MaxThrottleEffort)
	assignFloat(&controller.MaxBrakeEffort, section.MaxBrakeEffort)
	assignFloat(&controller.LongitudinalSlewPerSecond, section.LongitudinalSlewPerSecond)
	assignFloat(&controller.StopProbabilityThreshold, section.StopProbabilityThreshold)
	assignFloat(&controller.StopProbabilityReleaseThreshold, section.StopProbabilityReleaseThreshold)
	assignFloat(&controller.HoldSpeedMPS, section.HoldSpeedMPS)
	assignFloat(&controller.StopBrakeEffort, section.StopBrakeEffort)
	assignFloat(&controller.MaxDesiredSpeedMPS, section.MaxDesiredSpeedMPS)

	var err error
	if controller.MaxDT, err = parseOptionalDuration("backend.stop_sign_controller.max_dt", section.MaxDT, controller.MaxDT); err != nil {
		return err
	}
	if cfg.StopSignPlanTimeout, err = parseOptionalDuration("backend.stop_sign_controller.plan_timeout", section.PlanTimeout, cfg.StopSignPlanTimeout); err != nil {
		return err
	}
	if cfg.StopSignTelemetryTimeout, err = parseOptionalDuration("backend.stop_sign_controller.telemetry_timeout", section.TelemetryTimeout, cfg.StopSignTelemetryTimeout); err != nil {
		return err
	}
	if cfg.StopSignEstimatedActuationLatency, err = parseOptionalDuration("backend.stop_sign_controller.estimated_actuation_latency", section.EstimatedActuationLatency, cfg.StopSignEstimatedActuationLatency); err != nil {
		return err
	}
	if len(section.ExpectedHorizonDtMs) > 0 {
		cfg.StopSignExpectedHorizonDtMs = append([]int(nil), section.ExpectedHorizonDtMs...)
	}
	return nil
}

func parseOptionalDuration(label, raw string, fallback time.Duration) (time.Duration, error) {
	value := strings.TrimSpace(raw)
	if value == "" {
		return fallback, nil
	}
	parsed, err := time.ParseDuration(value)
	if err != nil {
		return 0, fmt.Errorf("invalid %s: %w", label, err)
	}
	return parsed, nil
}

func validateStopSignControllerConfig(cfg Config) error {
	if _, err := stopsigncontrol.New(cfg.StopSignController); err != nil {
		return fmt.Errorf("backend stop-sign controller: %w", err)
	}
	if cfg.StopSignPlanTimeout <= 0 {
		return fmt.Errorf("backend stop-sign controller plan_timeout must be > 0")
	}
	if cfg.StopSignTelemetryTimeout <= 0 {
		return fmt.Errorf("backend stop-sign controller telemetry_timeout must be > 0")
	}
	if cfg.StopSignEstimatedActuationLatency < 0 {
		return fmt.Errorf("backend stop-sign controller estimated_actuation_latency must be >= 0")
	}
	if len(cfg.StopSignExpectedHorizonDtMs) == 0 {
		return fmt.Errorf("backend stop-sign controller expected_horizon_dt_ms must not be empty")
	}
	for index, value := range cfg.StopSignExpectedHorizonDtMs {
		if value <= 0 || index > 0 && value <= cfg.StopSignExpectedHorizonDtMs[index-1] {
			return fmt.Errorf("backend stop-sign controller expected_horizon_dt_ms must be positive and strictly increasing")
		}
	}
	if !cfg.StopSignCalibration.Verified {
		return nil
	}
	if strings.TrimSpace(cfg.StopSignCalibration.ProfileID) == "" ||
		cfg.StopSignCalibration.VehicleModelHash == 0 ||
		strings.TrimSpace(cfg.StopSignCalibration.GameBuild) == "" ||
		strings.TrimSpace(cfg.StopSignCalibration.AdapterVersion) == "" {
		return fmt.Errorf("verified backend stop-sign calibration requires profile_id, vehicle_model_hash, game_build, and adapter_version")
	}
	if cfg.StopSignCalibration.SteeringConvention != "positive_wheel_is_positive_xinput" {
		return fmt.Errorf("verified backend stop-sign calibration steering_convention must be positive_wheel_is_positive_xinput")
	}
	return nil
}

func validateActuatorSafetyConfig(cfg Config) error {
	return ValidateTuning(cfg.Tuning())
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
