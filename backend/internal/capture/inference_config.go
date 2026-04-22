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
	ConfigPath              string
	ModelServerURL          string
	ModelDevice             string
	SourceID                string
	AutoLoad                bool
	FPS                     int
	WindowSize              int
	FrameStride             int
	DispatchStride          int
	FrameWidth              int
	FrameHeight             int
	RequestTimeout          time.Duration
	JPEGQuality             int
	TargetSpeedErrorGain    float64
	TargetSpeedDeadband     float64
	DeltaSpeedTrimGain      float64
	DeltaSpeedDeadband      float64
	HeadingErrorDeadbandDeg float64
	HeadingErrorFullLockDeg float64
	LowSpeedSteerGain       float64
	HighSpeedSteerGain      float64
	SteerGainFadeSpeedMPS   float64
	SteerResponseBlend      float64
	LaunchThrottleMin       float64
	LaunchSpeedThreshold    float64
	LaunchTargetSpeedMargin float64
	MaxTargetSpeedKPH       float64
	MoveIntentThreshold     float64
	MoveIntentOnThreshold   float64
	MoveIntentOffThreshold  float64
	MoveIntentHoldSpeedMax  float64
	SteerCommandRatePerSec  float64
}

type backendSection struct {
	Inference backendInferenceSection `toml:"inference"`
}

type backendInferenceSection struct {
	ModelServerURL          string   `toml:"model_server_url"`
	ModelDevice             string   `toml:"model_device"`
	SourceID                string   `toml:"source_id"`
	AutoLoad                *bool    `toml:"auto_load"`
	FPS                     int      `toml:"fps"`
	DispatchStride          *int     `toml:"dispatch_stride"`
	FrameWidth              int      `toml:"frame_width"`
	FrameHeight             int      `toml:"frame_height"`
	RequestTimeout          string   `toml:"request_timeout"`
	JPEGQuality             int      `toml:"jpeg_quality"`
	TargetSpeedErrorGain    *float64 `toml:"target_speed_error_gain"`
	TargetSpeedDeadband     *float64 `toml:"target_speed_deadband"`
	DeltaSpeedTrimGain      *float64 `toml:"delta_speed_trim_gain"`
	DeltaSpeedDeadband      *float64 `toml:"delta_speed_deadband"`
	HeadingErrorDeadbandDeg *float64 `toml:"heading_error_deadband_deg"`
	HeadingErrorFullLockDeg *float64 `toml:"heading_error_full_lock_deg"`
	LowSpeedSteerGain       *float64 `toml:"low_speed_steer_gain"`
	HighSpeedSteerGain      *float64 `toml:"high_speed_steer_gain"`
	SteerGainFadeSpeedMPS   *float64 `toml:"steer_gain_fade_speed_mps"`
	SteerResponseBlend      *float64 `toml:"steer_response_blend"`
	LaunchThrottleMin       *float64 `toml:"launch_throttle_min"`
	LaunchSpeedThreshold    *float64 `toml:"launch_speed_threshold"`
	LaunchTargetSpeedMargin *float64 `toml:"launch_target_speed_margin"`
	MaxTargetSpeedKPH       *float64 `toml:"max_target_speed_kph"`
	MoveIntentThreshold     *float64 `toml:"move_intent_threshold"`
	MoveIntentOnThreshold   *float64 `toml:"move_intent_on_threshold"`
	MoveIntentOffThreshold  *float64 `toml:"move_intent_off_threshold"`
	MoveIntentHoldSpeedMax  *float64 `toml:"move_intent_hold_speed_max"`
	SteerCommandRatePerSec  *float64 `toml:"steer_command_rate_per_second"`
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
		JPEGQuality:             defaultInferenceJPEGQuality,
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
	datasetConfig, err := LoadDatasetConfig(path)
	if err != nil {
		return InferenceConfig{}, err
	}
	cfg.WindowSize = datasetConfig.WindowSize
	cfg.FrameStride = datasetConfig.FrameStride

	parsed, _, err := loadTrainConfigFile(path)
	if err != nil {
		return InferenceConfig{}, err
	}

	section := parsed.Backend.Inference
	if strings.TrimSpace(path) != "" {
		if _, statErr := os.Stat(path); statErr == nil {
			cfg.ConfigPath = path
		}
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
	if section.JPEGQuality > 0 {
		cfg.JPEGQuality = section.JPEGQuality
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
		SampleStride:                 defaultDatasetSampleStride,
		LabelTolerance:               defaultDatasetLabelTolerance,
		DeltaSpeedClip:               defaultDatasetDeltaSpeedClip,
		DeltaSpeedNormalize:          defaultDatasetDeltaSpeedNormalize,
		SyncFlashBrightnessThreshold: defaultSyncFlashBrightnessThreshold,
		SyncFlashFrameLimit:          defaultDatasetSyncFlashFrameLimit,
	}); err != nil {
		return InferenceConfig{}, err
	}
	if cfg.DispatchStride < 1 {
		return InferenceConfig{}, fmt.Errorf("backend inference dispatch_stride must be > 0")
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
