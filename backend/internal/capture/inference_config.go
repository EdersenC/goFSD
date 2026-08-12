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
	defaultPlannerFormat               = "temporal_stop_sign_v1"
	defaultControlContract             = "stop_sign_motion_plan_v1"
	defaultPredictionTimeout           = 250 * time.Millisecond
	defaultAlignmentTolerance          = 125 * time.Millisecond
	defaultMaxFrameTelemetrySkew       = 75 * time.Millisecond
)

type InferenceConfig struct {
	ConfigPath                      string
	PlannerFormat                   string
	ControlContract                 string
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
	FutureOffsets                   []int
	FutureSteps                     int
	ControlHorizonDtMs              []int
	TelemetrySampleInterval         time.Duration
	TelemetryFeatureNames           []string
	ControlOutputNames              []string
	AuxOutputNames                  []string
	AlignmentTolerance              time.Duration
	MaxFrameTelemetrySkew           time.Duration
	TelemetryNormalizationEnabled   bool
	TelemetryNormalizationStatsPath string
}

type backendSection struct {
	Inference backendInferenceSection `toml:"inference"`
}

type backendInferenceSection struct {
	PlannerFormat                   string `toml:"planner_format"`
	ControlContract                 string `toml:"control_contract"`
	ModelServerURL                  string `toml:"model_server_url"`
	ModelDevice                     string `toml:"model_device"`
	SourceID                        string `toml:"source_id"`
	AutoLoad                        *bool  `toml:"auto_load"`
	FPS                             int    `toml:"fps"`
	DispatchStride                  *int   `toml:"dispatch_stride"`
	FrameWidth                      int    `toml:"frame_width"`
	FrameHeight                     int    `toml:"frame_height"`
	RequestTimeout                  string `toml:"request_timeout"`
	PredictionTimeout               string `toml:"inference_timeout"`
	JPEGQuality                     int    `toml:"jpeg_quality"`
	AlignmentTolerance              string `toml:"alignment_tolerance"`
	MaxFrameTelemetrySkew           string `toml:"max_frame_telemetry_skew"`
	TelemetryNormalizationEnabled   *bool  `toml:"telemetry_normalization_enabled"`
	TelemetryNormalizationStatsPath string `toml:"telemetry_normalization_stats_path"`
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
		ControlContract:         defaultControlContract,
		ModelServerURL:          url,
		ModelDevice:             strings.ToLower(modelDevice),
		SourceID:                sourceID,
		AutoLoad:                parseBoolEnv("INFERENCE_AUTO_LOAD_MODEL", true),
		FPS:                     defaultInferenceFPS,
		WindowSize:              5,
		FrameStride:             defaultInferenceStride,
		DispatchStride:          defaultInferenceStride,
		FrameWidth:              defaultInferenceWidth,
		FrameHeight:             defaultInferenceHeight,
		RequestTimeout:          parseDurationEnv("INFERENCE_REQUEST_TIMEOUT", defaultInferenceRequestTimeout),
		PredictionTimeout:       defaultPredictionTimeout,
		JPEGQuality:             defaultInferenceJPEGQuality,
		ImageOffsets:            []int{-20, -15, -10, -5, 0},
		TelemetryOffsets:        []int{-20, -15, -10, -5, 0},
		FutureOffsets:           []int{2, 5, 10, 20},
		FutureSteps:             4,
		ControlHorizonDtMs:      []int{100, 250, 500, 1000},
		TelemetrySampleInterval: defaultTelemetrySampleInterval,
		TelemetryFeatureNames:   []string{"current_speed"},
		ControlOutputNames:      []string{"future_speed_mps", "stop_intent"},
		AuxOutputNames:          []string{"expert_throttle", "expert_brake", "actual_brake_pressure"},
		AlignmentTolerance:      defaultAlignmentTolerance,
		MaxFrameTelemetrySkew:   defaultMaxFrameTelemetrySkew,
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
	if parsed.Dataset.TelemetrySampleIntervalMs != nil {
		datasetConfig.TelemetrySampleInterval = time.Duration(*parsed.Dataset.TelemetrySampleIntervalMs) * time.Millisecond
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
	cfg.FutureOffsets = append([]int(nil), datasetConfig.FutureOffsets...)
	cfg.FutureSteps = len(datasetConfig.FutureOffsets)
	cfg.TelemetrySampleInterval = datasetConfig.TelemetrySampleInterval
	cfg.ControlHorizonDtMs = deriveControlHorizonDtMs(datasetConfig.FutureOffsets, datasetConfig.TelemetrySampleInterval)
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
	if value := strings.TrimSpace(section.ControlContract); value != "" {
		cfg.ControlContract = value
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
	if strings.TrimSpace(section.AlignmentTolerance) != "" {
		duration, err := time.ParseDuration(strings.TrimSpace(section.AlignmentTolerance))
		if err != nil {
			return InferenceConfig{}, fmt.Errorf("invalid backend.inference.alignment_tolerance: %w", err)
		}
		cfg.AlignmentTolerance = duration
	}
	if strings.TrimSpace(section.MaxFrameTelemetrySkew) != "" {
		duration, err := time.ParseDuration(strings.TrimSpace(section.MaxFrameTelemetrySkew))
		if err != nil {
			return InferenceConfig{}, fmt.Errorf("invalid backend.inference.max_frame_telemetry_skew: %w", err)
		}
		cfg.MaxFrameTelemetrySkew = duration
	}
	if section.TelemetryNormalizationEnabled != nil {
		cfg.TelemetryNormalizationEnabled = *section.TelemetryNormalizationEnabled
	}
	if value := strings.TrimSpace(section.TelemetryNormalizationStatsPath); value != "" {
		cfg.TelemetryNormalizationStatsPath = value
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
		TelemetrySampleInterval:      cfg.TelemetrySampleInterval,
		ImageOffsets:                 append([]int(nil), cfg.ImageOffsets...),
		TelemetryOffsets:             append([]int(nil), cfg.TelemetryOffsets...),
		FutureOffsets:                append([]int(nil), datasetConfig.FutureOffsets...),
		TelemetryFeatureNames:        append([]string(nil), cfg.TelemetryFeatureNames...),
		ControlTargetNames:           append([]string(nil), cfg.ControlOutputNames...),
		AuxTargetNames:               append([]string(nil), cfg.AuxOutputNames...),
		LabelTolerance:               datasetConfig.LabelTolerance,
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
	if cfg.ControlContract != defaultControlContract {
		return InferenceConfig{}, fmt.Errorf("backend inference control_contract must be %q", defaultControlContract)
	}
	if cfg.PredictionTimeout <= 0 {
		return InferenceConfig{}, fmt.Errorf("backend inference inference_timeout must be > 0")
	}
	if cfg.AlignmentTolerance <= 0 {
		return InferenceConfig{}, fmt.Errorf("backend inference alignment_tolerance must be > 0")
	}
	if cfg.MaxFrameTelemetrySkew <= 0 {
		return InferenceConfig{}, fmt.Errorf("backend inference max_frame_telemetry_skew must be > 0")
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
	if len(cfg.FutureOffsets) != cfg.FutureSteps {
		return InferenceConfig{}, fmt.Errorf("backend inference future_offsets length must match future_steps")
	}
	expectedControlNames := []string{"future_speed_mps", "stop_intent"}
	if err := validateExactStrings("backend inference control_output_names", cfg.ControlOutputNames, expectedControlNames); err != nil {
		return InferenceConfig{}, err
	}
	expectedHorizon := deriveControlHorizonDtMs(cfg.FutureOffsets, cfg.TelemetrySampleInterval)
	if err := validateExactInts("backend inference control_horizon_dt_ms", cfg.ControlHorizonDtMs, expectedHorizon); err != nil {
		return InferenceConfig{}, err
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
	return cfg, nil
}

func deriveControlHorizonDtMs(futureOffsets []int, sampleInterval time.Duration) []int {
	intervalMs := int(sampleInterval / time.Millisecond)
	out := make([]int, 0, len(futureOffsets))
	for _, offset := range futureOffsets {
		out = append(out, offset*intervalMs)
	}
	return out
}

func validateExactStrings(label string, actual, expected []string) error {
	if len(actual) != len(expected) {
		return fmt.Errorf("%s differ: got=%v want=%v", label, actual, expected)
	}
	for index := range expected {
		if strings.TrimSpace(actual[index]) != expected[index] {
			return fmt.Errorf("%s differ: got=%v want=%v", label, actual, expected)
		}
	}
	return nil
}

func validateExactInts(label string, actual, expected []int) error {
	if len(actual) != len(expected) {
		return fmt.Errorf("%s differ: got=%v want=%v", label, actual, expected)
	}
	for index := range expected {
		if actual[index] != expected[index] {
			return fmt.Errorf("%s differ: got=%v want=%v", label, actual, expected)
		}
	}
	return nil
}
