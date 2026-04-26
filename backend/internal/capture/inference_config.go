package capture

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const (
	defaultInferenceConfigRelativePath = "fsd_trainer/train_config.toml"
	defaultPlannerFormat               = "temporal_telemetry_gru_v1"
	defaultHorizonMode                 = "weighted_short_horizon"
	defaultPredictionTimeout           = 250 * time.Millisecond
	defaultAlignmentTolerance          = 125 * time.Millisecond
	defaultTargetSpeedErrorGain        = 0.5
	defaultTargetSpeedDeadband         = 0.05
	defaultDeltaSpeedTrimGain          = 0.25
	defaultDeltaSpeedDeadband          = 0.02
	defaultHeadingErrorDeadbandDeg     = 2.5
	defaultHeadingErrorFullLockDeg     = 45.0
	defaultLowSpeedSteerGain           = 1.35
	defaultHighSpeedSteerGain          = 0.75
	defaultSteerGainFadeSpeedMPS       = 5.0
	defaultSteerResponseBlend          = 0.55
	defaultLaunchThrottleMin           = 0.22
	defaultLaunchSpeedThreshold        = 1.0
	defaultLaunchTargetSpeedMargin     = 0.75
	defaultMaxTargetSpeedKPH           = 17.0
	defaultMoveIntentThreshold         = 0.55
	defaultMoveIntentOnThreshold       = 0.60
	defaultMoveIntentOffThreshold      = 0.40
	defaultMoveIntentHoldSpeedMax      = 4.0
	defaultSteerCommandRatePerSecond   = 4.0
)

type InferenceConfig struct {
	ConfigPath                      string
	PlannerFormat                   string
	ModelServerURL                  string
	ModelDevice                     string
	SourceID                        string
	AutoLoad                        bool
	FPS                             int
	WindowSize                      int
	FrameStride                     int
	DispatchStride                  int
	FrameWidth                      int
	FrameHeight                     int
	RequestTimeout                  time.Duration
	PredictionTimeout               time.Duration
	JPEGQuality                     int
	ImageOffsets                    []int
	TelemetryOffsets                []int
	FutureSteps                     int
	TelemetryFeatureNames           []string
	ControlOutputNames              []string
	AuxOutputNames                  []string
	HorizonMode                     string
	HorizonControlWeights           []float64
	AlignmentTolerance              time.Duration
	TelemetryNormalizationEnabled   bool
	TelemetryNormalizationStatsPath string
	TargetSpeedErrorGain            float64
	TargetSpeedDeadband             float64
	DeltaSpeedTrimGain              float64
	DeltaSpeedDeadband              float64
	HeadingErrorDeadbandDeg         float64
	HeadingErrorFullLockDeg         float64
	LowSpeedSteerGain               float64
	HighSpeedSteerGain              float64
	SteerGainFadeSpeedMPS           float64
	SteerResponseBlend              float64
	LaunchThrottleMin               float64
	LaunchSpeedThreshold            float64
	LaunchTargetSpeedMargin         float64
	MaxTargetSpeedKPH               float64
	MoveIntentThreshold             float64
	MoveIntentOnThreshold           float64
	MoveIntentOffThreshold          float64
	MoveIntentHoldSpeedMax          float64
	SteerCommandRatePerSec          float64
}

type backendSection struct {
	Inference backendInferenceSection `toml:"inference"`
}

type backendInferenceSection struct {
	PlannerFormat                   string    `toml:"planner_format"`
	ModelServerURL                  string    `toml:"model_server_url"`
	ModelDevice                     string    `toml:"model_device"`
	SourceID                        string    `toml:"source_id"`
	AutoLoad                        *bool     `toml:"auto_load"`
	FPS                             int       `toml:"fps"`
	DispatchStride                  *int      `toml:"dispatch_stride"`
	FrameWidth                      int       `toml:"frame_width"`
	FrameHeight                     int       `toml:"frame_height"`
	RequestTimeout                  string    `toml:"request_timeout"`
	PredictionTimeout               string    `toml:"inference_timeout"`
	JPEGQuality                     int       `toml:"jpeg_quality"`
	HorizonMode                     string    `toml:"horizon_mode"`
	HorizonControlWeights           []float64 `toml:"horizon_control_weights"`
	AlignmentTolerance              string    `toml:"alignment_tolerance"`
	TelemetryNormalizationEnabled   *bool     `toml:"telemetry_normalization_enabled"`
	TelemetryNormalizationStatsPath string    `toml:"telemetry_normalization_stats_path"`
	TargetSpeedErrorGain            *float64  `toml:"target_speed_error_gain"`
	TargetSpeedDeadband             *float64  `toml:"target_speed_deadband"`
	DeltaSpeedTrimGain              *float64  `toml:"delta_speed_trim_gain"`
	DeltaSpeedDeadband              *float64  `toml:"delta_speed_deadband"`
	HeadingErrorDeadbandDeg         *float64  `toml:"heading_error_deadband_deg"`
	HeadingErrorFullLockDeg         *float64  `toml:"heading_error_full_lock_deg"`
	LowSpeedSteerGain               *float64  `toml:"low_speed_steer_gain"`
	HighSpeedSteerGain              *float64  `toml:"high_speed_steer_gain"`
	SteerGainFadeSpeedMPS           *float64  `toml:"steer_gain_fade_speed_mps"`
	SteerResponseBlend              *float64  `toml:"steer_response_blend"`
	LaunchThrottleMin               *float64  `toml:"launch_throttle_min"`
	LaunchSpeedThreshold            *float64  `toml:"launch_speed_threshold"`
	LaunchTargetSpeedMargin         *float64  `toml:"launch_target_speed_margin"`
	MaxTargetSpeedKPH               *float64  `toml:"max_target_speed_kph"`
	MoveIntentThreshold             *float64  `toml:"move_intent_threshold"`
	MoveIntentOnThreshold           *float64  `toml:"move_intent_on_threshold"`
	MoveIntentOffThreshold          *float64  `toml:"move_intent_off_threshold"`
	MoveIntentHoldSpeedMax          *float64  `toml:"move_intent_hold_speed_max"`
	SteerCommandRatePerSec          *float64  `toml:"steer_command_rate_per_second"`
}

func DefaultInferenceConfig() InferenceConfig {
	url := strings.TrimRight(strings.TrimSpace(envOrDefault("INFERENCE_MODEL_SERVER_URL", defaultInferenceModelServerURL)), "/")
	if url == "" {
		url = defaultInferenceModelServerURL
	}
	sourceID := strings.TrimSpace(envOrDefault("INFERENCE_SOURCE_ID", defaultInferenceSourceID))
	if sourceID == "" {
		sourceID = defaultInferenceSourceID
	}
	modelDevice := strings.TrimSpace(envOrDefault("INFERENCE_MODEL_DEVICE", "cuda"))
	if modelDevice == "" {
		modelDevice = "cuda"
	}
	return InferenceConfig{
		PlannerFormat:           defaultPlannerFormat,
		ModelServerURL:          url,
		ModelDevice:             strings.ToLower(modelDevice),
		SourceID:                sourceID,
		AutoLoad:                parseBoolEnv("INFERENCE_AUTO_LOAD_MODEL", true),
		FPS:                     defaultInferenceFPS,
		WindowSize:              defaultInferenceWindowSize,
		FrameStride:             defaultInferenceStride,
		DispatchStride:          defaultInferenceStride,
		FrameWidth:              defaultInferenceWidth,
		FrameHeight:             defaultInferenceHeight,
		RequestTimeout:          parseDurationEnv("INFERENCE_REQUEST_TIMEOUT", defaultInferenceRequestTimeout),
		PredictionTimeout:       defaultPredictionTimeout,
		JPEGQuality:             defaultInferenceJPEGQuality,
		ImageOffsets:            []int{-8, -6, -4, -2, 0},
		TelemetryOffsets:        []int{-8, -7, -6, -5, -4, -3, -2, -1, 0},
		FutureSteps:             6,
		TelemetryFeatureNames:   []string{"current_speed", "yaw_sin", "yaw_cos", "yaw_rate", "steering", "acceleration"},
		ControlOutputNames:      []string{"steering", "acceleration", "brakePressureAvg"},
		AuxOutputNames:          []string{"future_speed", "future_yaw_delta", "future_yaw_rate"},
		HorizonMode:             defaultHorizonMode,
		HorizonControlWeights:   []float64{0.60, 0.30, 0.10},
		AlignmentTolerance:      defaultAlignmentTolerance,
		TargetSpeedErrorGain:    defaultTargetSpeedErrorGain,
		TargetSpeedDeadband:     defaultTargetSpeedDeadband,
		DeltaSpeedTrimGain:      defaultDeltaSpeedTrimGain,
		DeltaSpeedDeadband:      defaultDeltaSpeedDeadband,
		HeadingErrorDeadbandDeg: defaultHeadingErrorDeadbandDeg,
		HeadingErrorFullLockDeg: defaultHeadingErrorFullLockDeg,
		LowSpeedSteerGain:       defaultLowSpeedSteerGain,
		HighSpeedSteerGain:      defaultHighSpeedSteerGain,
		SteerGainFadeSpeedMPS:   defaultSteerGainFadeSpeedMPS,
		SteerResponseBlend:      defaultSteerResponseBlend,
		LaunchThrottleMin:       defaultLaunchThrottleMin,
		LaunchSpeedThreshold:    defaultLaunchSpeedThreshold,
		LaunchTargetSpeedMargin: defaultLaunchTargetSpeedMargin,
		MaxTargetSpeedKPH:       defaultMaxTargetSpeedKPH,
		MoveIntentThreshold:     defaultMoveIntentThreshold,
		MoveIntentOnThreshold:   defaultMoveIntentOnThreshold,
		MoveIntentOffThreshold:  defaultMoveIntentOffThreshold,
		MoveIntentHoldSpeedMax:  defaultMoveIntentHoldSpeedMax,
		SteerCommandRatePerSec:  defaultSteerCommandRatePerSecond,
	}
}

func ResolveInferenceConfigPath(explicitPath string) (string, error) {
	if explicitPath != "" {
		abs, err := filepath.Abs(explicitPath)
		if err != nil {
			return "", err
		}
		return abs, nil
	}

	if envPath := strings.TrimSpace(os.Getenv("FSD_CONFIG_PATH")); envPath != "" {
		abs, err := filepath.Abs(envPath)
		if err != nil {
			return "", err
		}
		return abs, nil
	}

	cwd, err := os.Getwd()
	if err != nil {
		return "", err
	}
	candidates := []string{
		filepath.Join(cwd, "../", defaultInferenceConfigRelativePath),
		filepath.Join(cwd, defaultInferenceConfigRelativePath),
	}
	for _, candidate := range candidates {
		if _, err := os.Stat(candidate); err == nil {
			abs, absErr := filepath.Abs(candidate)
			if absErr != nil {
				return "", absErr
			}
			return abs, nil
		}
	}
	return "", fmt.Errorf("inference config not found; checked %s", strings.Join(candidates, ", "))
}

func LoadInferenceConfig(path string) (InferenceConfig, error) {
	cfg := DefaultInferenceConfig()
	datasetConfig := DefaultDatasetConfig()

	parsed, _, err := loadTrainConfigFile(path)
	if err != nil {
		return InferenceConfig{}, err
	}
	if parsed.Dataset.WindowStride != nil {
		return InferenceConfig{}, fmt.Errorf("%s", legacyDatasetWindowStrideMigrationNotice)
	}
	if parsed.Dataset.WindowSize != nil {
		datasetConfig.WindowSize = *parsed.Dataset.WindowSize
	}
	if parsed.Dataset.FrameStride != nil {
		datasetConfig.FrameStride = *parsed.Dataset.FrameStride
	}
	if len(parsed.Dataset.ImageOffsets) > 0 {
		datasetConfig.ImageOffsets = append([]int(nil), parsed.Dataset.ImageOffsets...)
	} else {
		datasetConfig.ImageOffsets = deriveLegacyImageOffsets(datasetConfig.WindowSize, datasetConfig.FrameStride)
	}
	if len(parsed.Dataset.TelemetryOffsets) > 0 {
		datasetConfig.TelemetryOffsets = append([]int(nil), parsed.Dataset.TelemetryOffsets...)
	}
	if len(parsed.Dataset.FutureOffsets) > 0 {
		datasetConfig.FutureOffsets = append([]int(nil), parsed.Dataset.FutureOffsets...)
	}
	if len(parsed.Dataset.TelemetryFeatureNames) > 0 {
		datasetConfig.TelemetryFeatureNames = append([]string(nil), parsed.Dataset.TelemetryFeatureNames...)
	}
	if len(parsed.Dataset.ControlTargetNames) > 0 {
		datasetConfig.ControlTargetNames = append([]string(nil), parsed.Dataset.ControlTargetNames...)
	}
	if len(parsed.Dataset.AuxTargetNames) > 0 {
		datasetConfig.AuxTargetNames = append([]string(nil), parsed.Dataset.AuxTargetNames...)
	}
	cfg.WindowSize = datasetConfig.WindowSize
	cfg.FrameStride = datasetConfig.FrameStride
	cfg.ImageOffsets = append([]int(nil), datasetConfig.ImageOffsets...)
	cfg.TelemetryOffsets = append([]int(nil), datasetConfig.TelemetryOffsets...)
	cfg.FutureSteps = len(datasetConfig.FutureOffsets)
	cfg.TelemetryFeatureNames = append([]string(nil), datasetConfig.TelemetryFeatureNames...)
	cfg.ControlOutputNames = append([]string(nil), datasetConfig.ControlTargetNames...)
	cfg.AuxOutputNames = append([]string(nil), datasetConfig.AuxTargetNames...)

	section := parsed.Backend.Inference
	if strings.TrimSpace(path) != "" {
		if _, statErr := os.Stat(path); statErr == nil {
			cfg.ConfigPath = path
		}
	}
	if value := strings.TrimSpace(section.PlannerFormat); value != "" {
		cfg.PlannerFormat = value
	}
	if value := strings.TrimRight(strings.TrimSpace(section.ModelServerURL), "/"); value != "" {
		cfg.ModelServerURL = value
	}
	if value := strings.TrimSpace(section.ModelDevice); value != "" {
		cfg.ModelDevice = strings.ToLower(value)
	}
	if value := strings.TrimSpace(section.SourceID); value != "" {
		cfg.SourceID = value
	}
	if section.AutoLoad != nil {
		cfg.AutoLoad = *section.AutoLoad
	}
	if section.FPS > 0 {
		cfg.FPS = section.FPS
	}
	if section.DispatchStride != nil {
		cfg.DispatchStride = *section.DispatchStride
	}
	if section.FrameWidth > 0 {
		cfg.FrameWidth = section.FrameWidth
	}
	if section.FrameHeight > 0 {
		cfg.FrameHeight = section.FrameHeight
	}
	if strings.TrimSpace(section.RequestTimeout) != "" {
		duration, err := time.ParseDuration(strings.TrimSpace(section.RequestTimeout))
		if err != nil {
			return InferenceConfig{}, fmt.Errorf("invalid backend.inference.request_timeout: %w", err)
		}
		cfg.RequestTimeout = duration
	}
	if strings.TrimSpace(section.PredictionTimeout) != "" {
		duration, err := time.ParseDuration(strings.TrimSpace(section.PredictionTimeout))
		if err != nil {
			return InferenceConfig{}, fmt.Errorf("invalid backend.inference.inference_timeout: %w", err)
		}
		cfg.PredictionTimeout = duration
	}
	if section.JPEGQuality > 0 {
		cfg.JPEGQuality = section.JPEGQuality
	}
	if value := strings.TrimSpace(section.HorizonMode); value != "" {
		cfg.HorizonMode = value
	}
	if len(section.HorizonControlWeights) > 0 {
		cfg.HorizonControlWeights = append([]float64(nil), section.HorizonControlWeights...)
	}
	if strings.TrimSpace(section.AlignmentTolerance) != "" {
		duration, err := time.ParseDuration(strings.TrimSpace(section.AlignmentTolerance))
		if err != nil {
			return InferenceConfig{}, fmt.Errorf("invalid backend.inference.alignment_tolerance: %w", err)
		}
		cfg.AlignmentTolerance = duration
	}
	if section.TelemetryNormalizationEnabled != nil {
		cfg.TelemetryNormalizationEnabled = *section.TelemetryNormalizationEnabled
	}
	if value := strings.TrimSpace(section.TelemetryNormalizationStatsPath); value != "" {
		cfg.TelemetryNormalizationStatsPath = value
	}
	if section.TargetSpeedErrorGain != nil {
		cfg.TargetSpeedErrorGain = *section.TargetSpeedErrorGain
	}
	if section.TargetSpeedDeadband != nil {
		cfg.TargetSpeedDeadband = *section.TargetSpeedDeadband
	}
	if section.DeltaSpeedTrimGain != nil {
		cfg.DeltaSpeedTrimGain = *section.DeltaSpeedTrimGain
	}
	if section.DeltaSpeedDeadband != nil {
		cfg.DeltaSpeedDeadband = *section.DeltaSpeedDeadband
	}
	if section.HeadingErrorDeadbandDeg != nil {
		cfg.HeadingErrorDeadbandDeg = *section.HeadingErrorDeadbandDeg
	}
	if section.HeadingErrorFullLockDeg != nil {
		cfg.HeadingErrorFullLockDeg = *section.HeadingErrorFullLockDeg
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
	if section.MoveIntentThreshold != nil {
		cfg.MoveIntentThreshold = *section.MoveIntentThreshold
	}
	if section.MoveIntentOnThreshold != nil {
		cfg.MoveIntentOnThreshold = *section.MoveIntentOnThreshold
	} else if section.MoveIntentThreshold != nil {
		cfg.MoveIntentOnThreshold = *section.MoveIntentThreshold
	}
	if section.MoveIntentOffThreshold != nil {
		cfg.MoveIntentOffThreshold = *section.MoveIntentOffThreshold
	} else if section.MoveIntentThreshold != nil {
		cfg.MoveIntentOffThreshold = clamp(*section.MoveIntentThreshold-0.15, 0.0, 1.0)
	}
	if section.MoveIntentHoldSpeedMax != nil {
		cfg.MoveIntentHoldSpeedMax = *section.MoveIntentHoldSpeedMax
	}
	if section.SteerCommandRatePerSec != nil {
		cfg.SteerCommandRatePerSec = *section.SteerCommandRatePerSec
	}

	if strings.TrimSpace(cfg.ModelDevice) == "" {
		return InferenceConfig{}, fmt.Errorf("backend inference model_device must not be empty")
	}
	if cfg.FPS < 1 {
		return InferenceConfig{}, fmt.Errorf("backend inference fps must be > 0")
	}
	if err := validateDatasetConfig("dataset", DatasetConfig{
		ImageWidth:                   datasetConfig.ImageWidth,
		ImageHeight:                  datasetConfig.ImageHeight,
		WindowSize:                   cfg.WindowSize,
		FrameStride:                  cfg.FrameStride,
		SampleStride:                 datasetConfig.SampleStride,
		ImageOffsets:                 append([]int(nil), cfg.ImageOffsets...),
		TelemetryOffsets:             append([]int(nil), cfg.TelemetryOffsets...),
		FutureOffsets:                append([]int(nil), datasetConfig.FutureOffsets...),
		TelemetryFeatureNames:        append([]string(nil), cfg.TelemetryFeatureNames...),
		ControlTargetNames:           append([]string(nil), cfg.ControlOutputNames...),
		AuxTargetNames:               append([]string(nil), cfg.AuxOutputNames...),
		LabelTolerance:               datasetConfig.LabelTolerance,
		DeltaSpeedClip:               datasetConfig.DeltaSpeedClip,
		DeltaSpeedNormalize:          datasetConfig.DeltaSpeedNormalize,
		SyncFlashBrightnessThreshold: datasetConfig.SyncFlashBrightnessThreshold,
		SyncFlashFrameLimit:          datasetConfig.SyncFlashFrameLimit,
	}); err != nil {
		return InferenceConfig{}, err
	}
	if cfg.DispatchStride < 1 {
		return InferenceConfig{}, fmt.Errorf("backend inference dispatch_stride must be > 0")
	}
	if cfg.PlannerFormat != defaultPlannerFormat {
		return InferenceConfig{}, fmt.Errorf("backend inference planner_format must be %q", defaultPlannerFormat)
	}
	if cfg.PredictionTimeout <= 0 {
		return InferenceConfig{}, fmt.Errorf("backend inference inference_timeout must be > 0")
	}
	if cfg.AlignmentTolerance <= 0 {
		return InferenceConfig{}, fmt.Errorf("backend inference alignment_tolerance must be > 0")
	}
	if len(cfg.ImageOffsets) != cfg.WindowSize {
		return InferenceConfig{}, fmt.Errorf("backend inference image_offsets length must match window_size")
	}
	if len(cfg.TelemetryOffsets) < 1 || cfg.TelemetryOffsets[len(cfg.TelemetryOffsets)-1] != 0 {
		return InferenceConfig{}, fmt.Errorf("backend inference telemetry_offsets must end at 0")
	}
	if cfg.FutureSteps < 1 {
		return InferenceConfig{}, fmt.Errorf("backend inference future_steps must be > 0")
	}
	if len(cfg.ControlOutputNames) < 2 || cfg.ControlOutputNames[0] != "steering" || cfg.ControlOutputNames[1] != "acceleration" {
		return InferenceConfig{}, fmt.Errorf("backend inference control_output_names must start with [steering, acceleration]")
	}
	if len(cfg.ControlOutputNames) > 3 {
		return InferenceConfig{}, fmt.Errorf("backend inference control_output_names must have 2 or 3 entries")
	}
	if len(cfg.ControlOutputNames) == 3 && cfg.ControlOutputNames[2] != "brakePressureAvg" {
		return InferenceConfig{}, fmt.Errorf("backend inference third control_output_name must be brakePressureAvg when present")
	}
	switch cfg.HorizonMode {
	case "weighted_short_horizon", "t_plus_1_only":
	default:
		return InferenceConfig{}, fmt.Errorf("backend inference horizon_mode must be weighted_short_horizon or t_plus_1_only")
	}
	if cfg.HorizonMode == "weighted_short_horizon" {
		if len(cfg.HorizonControlWeights) != 3 {
			return InferenceConfig{}, fmt.Errorf("backend inference horizon_control_weights must have three entries")
		}
	}
	for _, value := range cfg.HorizonControlWeights {
		if value < 0 {
			return InferenceConfig{}, fmt.Errorf("backend inference horizon_control_weights must be >= 0")
		}
	}
	if cfg.TelemetryNormalizationEnabled && strings.TrimSpace(cfg.TelemetryNormalizationStatsPath) == "" {
		return InferenceConfig{}, fmt.Errorf("backend inference telemetry_normalization_stats_path is required when telemetry normalization is enabled")
	}
	if cfg.FrameWidth < 1 || cfg.FrameHeight < 1 {
		return InferenceConfig{}, fmt.Errorf("backend inference frame dimensions must be > 0")
	}
	if cfg.RequestTimeout <= 0 {
		return InferenceConfig{}, fmt.Errorf("backend inference request_timeout must be > 0")
	}
	if cfg.JPEGQuality < 1 || cfg.JPEGQuality > 100 {
		return InferenceConfig{}, fmt.Errorf("backend inference jpeg_quality must be between 1 and 100")
	}
	if cfg.TargetSpeedErrorGain < 0 {
		return InferenceConfig{}, fmt.Errorf("backend inference target_speed_error_gain must be >= 0")
	}
	if cfg.TargetSpeedDeadband < 0 {
		return InferenceConfig{}, fmt.Errorf("backend inference target_speed_deadband must be >= 0")
	}
	if cfg.DeltaSpeedTrimGain < 0 {
		return InferenceConfig{}, fmt.Errorf("backend inference delta_speed_trim_gain must be >= 0")
	}
	if cfg.DeltaSpeedDeadband < 0 {
		return InferenceConfig{}, fmt.Errorf("backend inference delta_speed_deadband must be >= 0")
	}
	if cfg.HeadingErrorDeadbandDeg < 0 {
		return InferenceConfig{}, fmt.Errorf("backend inference heading_error_deadband_deg must be >= 0")
	}
	if cfg.HeadingErrorFullLockDeg <= 0 {
		return InferenceConfig{}, fmt.Errorf("backend inference heading_error_full_lock_deg must be > 0")
	}
	if cfg.LowSpeedSteerGain < 0 {
		return InferenceConfig{}, fmt.Errorf("backend inference low_speed_steer_gain must be >= 0")
	}
	if cfg.HighSpeedSteerGain < 0 {
		return InferenceConfig{}, fmt.Errorf("backend inference high_speed_steer_gain must be >= 0")
	}
	if cfg.SteerGainFadeSpeedMPS <= 0 {
		return InferenceConfig{}, fmt.Errorf("backend inference steer_gain_fade_speed_mps must be > 0")
	}
	if cfg.SteerResponseBlend < 0 || cfg.SteerResponseBlend > 1 {
		return InferenceConfig{}, fmt.Errorf("backend inference steer_response_blend must be in [0,1]")
	}
	if cfg.LaunchThrottleMin < 0 || cfg.LaunchThrottleMin > 1 {
		return InferenceConfig{}, fmt.Errorf("backend inference launch_throttle_min must be in [0,1]")
	}
	if cfg.LaunchSpeedThreshold < 0 {
		return InferenceConfig{}, fmt.Errorf("backend inference launch_speed_threshold must be >= 0")
	}
	if cfg.LaunchTargetSpeedMargin < 0 {
		return InferenceConfig{}, fmt.Errorf("backend inference launch_target_speed_margin must be >= 0")
	}
	if cfg.MaxTargetSpeedKPH < 0 {
		return InferenceConfig{}, fmt.Errorf("backend inference max_target_speed_kph must be >= 0")
	}
	if cfg.MoveIntentThreshold < 0 || cfg.MoveIntentThreshold > 1 {
		return InferenceConfig{}, fmt.Errorf("backend inference move_intent_threshold must be in [0,1]")
	}
	if cfg.MoveIntentOnThreshold < 0 || cfg.MoveIntentOnThreshold > 1 {
		return InferenceConfig{}, fmt.Errorf("backend inference move_intent_on_threshold must be in [0,1]")
	}
	if cfg.MoveIntentOffThreshold < 0 || cfg.MoveIntentOffThreshold > 1 {
		return InferenceConfig{}, fmt.Errorf("backend inference move_intent_off_threshold must be in [0,1]")
	}
	if cfg.MoveIntentOffThreshold > cfg.MoveIntentOnThreshold {
		return InferenceConfig{}, fmt.Errorf("backend inference move_intent_off_threshold must be <= move_intent_on_threshold")
	}
	if cfg.MoveIntentHoldSpeedMax < 0 {
		return InferenceConfig{}, fmt.Errorf("backend inference move_intent_hold_speed_max must be >= 0")
	}
	if cfg.SteerCommandRatePerSec < 0 {
		return InferenceConfig{}, fmt.Errorf("backend inference steer_command_rate_per_second must be >= 0")
	}

	return cfg, nil
}
