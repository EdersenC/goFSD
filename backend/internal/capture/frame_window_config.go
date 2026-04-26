package capture

import (
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	toml "github.com/pelletier/go-toml/v2"
)

const (
	defaultDatasetWindowSize                 = 3
	defaultDatasetFrameStride                = 2
	defaultDatasetSampleStride               = 2
	defaultDatasetImageWidth                 = 224
	defaultDatasetImageHeight                = 224
	defaultDatasetLabelTolerance             = 100 * time.Millisecond
	defaultDatasetDeltaSpeedClip             = 2.0
	defaultDatasetDeltaSpeedNormalize        = true
	defaultSyncFlashBrightnessThreshold      = 245.0
	defaultDatasetSyncFlashFrameLimit        = 90
	missingDatasetFrameStrideMessage         = "dataset.frame_stride must be configured explicitly"
	missingDatasetSampleStrideMessage        = "dataset.sample_stride must be configured explicitly"
	legacyDatasetWindowStrideMigrationNotice = "dataset.window_stride is no longer supported; use dataset.frame_stride and dataset.sample_stride"
)

type DatasetConfig struct {
	ImageWidth                   int
	ImageHeight                  int
	WindowSize                   int
	FrameStride                  int
	SampleStride                 int
	ImageOffsets                 []int
	TelemetryOffsets             []int
	FutureOffsets                []int
	TelemetryFeatureNames        []string
	ControlTargetNames           []string
	AuxTargetNames               []string
	LabelTolerance               time.Duration
	DeltaSpeedClip               float64
	DeltaSpeedNormalize          bool
	SyncFlashBrightnessThreshold float64
	SyncFlashFrameLimit          int
}

type trainConfigFile struct {
	Dataset datasetSection `toml:"dataset"`
	Backend backendSection `toml:"backend"`
}

type datasetSection struct {
	ImageWidth                   *int     `toml:"image_width"`
	ImageHeight                  *int     `toml:"image_height"`
	WindowSize                   *int     `toml:"window_size"`
	WindowStride                 *int     `toml:"window_stride"`
	FrameStride                  *int     `toml:"frame_stride"`
	SampleStride                 *int     `toml:"sample_stride"`
	ImageOffsets                 []int    `toml:"image_offsets"`
	TelemetryOffsets             []int    `toml:"telemetry_offsets"`
	FutureOffsets                []int    `toml:"future_offsets"`
	TelemetryFeatureNames        []string `toml:"telemetry_feature_names"`
	ControlTargetNames           []string `toml:"control_target_names"`
	AuxTargetNames               []string `toml:"aux_target_names"`
	LabelTolerance               string   `toml:"label_tolerance"`
	DeltaSpeedClip               *float64 `toml:"delta_speed_clip"`
	DeltaSpeedNormalize          *bool    `toml:"delta_speed_normalize"`
	SyncFlashBrightnessThreshold *float64 `toml:"sync_flash_brightness_threshold"`
	SyncFlashFrameLimit          *int     `toml:"sync_flash_frame_limit"`
}

func DefaultDatasetConfig() DatasetConfig {
	return DatasetConfig{
		ImageWidth:                   defaultDatasetImageWidth,
		ImageHeight:                  defaultDatasetImageHeight,
		WindowSize:                   defaultDatasetWindowSize,
		FrameStride:                  defaultDatasetFrameStride,
		SampleStride:                 defaultDatasetSampleStride,
		ImageOffsets:                 []int{-4, -2, 0},
		TelemetryOffsets:             []int{-2, -1, 0},
		FutureOffsets:                []int{1},
		TelemetryFeatureNames:        []string{"current_speed", "yaw_sin", "yaw_cos", "yaw_rate", "steering", "acceleration"},
		ControlTargetNames:           []string{"steering", "acceleration", "brakePressureAvg"},
		AuxTargetNames:               []string{"future_speed", "future_yaw_delta", "future_yaw_rate"},
		LabelTolerance:               defaultDatasetLabelTolerance,
		DeltaSpeedClip:               defaultDatasetDeltaSpeedClip,
		DeltaSpeedNormalize:          defaultDatasetDeltaSpeedNormalize,
		SyncFlashBrightnessThreshold: defaultSyncFlashBrightnessThreshold,
		SyncFlashFrameLimit:          defaultDatasetSyncFlashFrameLimit,
	}
}

func LoadDatasetConfig(path string) (DatasetConfig, error) {
	cfg := DefaultDatasetConfig()

	parsed, exists, err := loadTrainConfigFile(path)
	if err != nil {
		return DatasetConfig{}, err
	}

	if parsed.Dataset.WindowStride != nil {
		return DatasetConfig{}, fmt.Errorf("%s", legacyDatasetWindowStrideMigrationNotice)
	}

	if parsed.Dataset.ImageWidth != nil {
		cfg.ImageWidth = *parsed.Dataset.ImageWidth
	}
	if parsed.Dataset.ImageHeight != nil {
		cfg.ImageHeight = *parsed.Dataset.ImageHeight
	}
	if parsed.Dataset.WindowSize != nil {
		cfg.WindowSize = *parsed.Dataset.WindowSize
	}
	if parsed.Dataset.FrameStride != nil {
		cfg.FrameStride = *parsed.Dataset.FrameStride
	} else if exists {
		return DatasetConfig{}, fmt.Errorf("%s", missingDatasetFrameStrideMessage)
	}
	if parsed.Dataset.SampleStride != nil {
		cfg.SampleStride = *parsed.Dataset.SampleStride
	} else if exists {
		return DatasetConfig{}, fmt.Errorf("%s", missingDatasetSampleStrideMessage)
	}
	if len(parsed.Dataset.ImageOffsets) > 0 {
		cfg.ImageOffsets = append([]int(nil), parsed.Dataset.ImageOffsets...)
	}
	if len(parsed.Dataset.TelemetryOffsets) > 0 {
		cfg.TelemetryOffsets = append([]int(nil), parsed.Dataset.TelemetryOffsets...)
	}
	if len(parsed.Dataset.FutureOffsets) > 0 {
		cfg.FutureOffsets = append([]int(nil), parsed.Dataset.FutureOffsets...)
	}
	if len(parsed.Dataset.TelemetryFeatureNames) > 0 {
		cfg.TelemetryFeatureNames = append([]string(nil), parsed.Dataset.TelemetryFeatureNames...)
	}
	if len(parsed.Dataset.ControlTargetNames) > 0 {
		cfg.ControlTargetNames = append([]string(nil), parsed.Dataset.ControlTargetNames...)
	}
	if len(parsed.Dataset.AuxTargetNames) > 0 {
		cfg.AuxTargetNames = append([]string(nil), parsed.Dataset.AuxTargetNames...)
	}
	if len(cfg.ImageOffsets) == 0 || len(parsed.Dataset.ImageOffsets) == 0 {
		cfg.ImageOffsets = deriveLegacyImageOffsets(cfg.WindowSize, cfg.FrameStride)
	}
	if len(cfg.TelemetryOffsets) == 0 || len(parsed.Dataset.TelemetryOffsets) == 0 {
		cfg.TelemetryOffsets = deriveTelemetryOffsets(len(cfg.TelemetryOffsets))
	}
	if len(cfg.FutureOffsets) == 0 {
		cfg.FutureOffsets = []int{1}
	}
	if strings.TrimSpace(parsed.Dataset.LabelTolerance) != "" {
		labelTolerance, err := time.ParseDuration(strings.TrimSpace(parsed.Dataset.LabelTolerance))
		if err != nil {
			return DatasetConfig{}, fmt.Errorf("invalid dataset.label_tolerance: %w", err)
		}
		cfg.LabelTolerance = labelTolerance
	}
	if parsed.Dataset.DeltaSpeedClip != nil {
		cfg.DeltaSpeedClip = *parsed.Dataset.DeltaSpeedClip
	}
	if parsed.Dataset.DeltaSpeedNormalize != nil {
		cfg.DeltaSpeedNormalize = *parsed.Dataset.DeltaSpeedNormalize
	}
	if parsed.Dataset.SyncFlashBrightnessThreshold != nil {
		cfg.SyncFlashBrightnessThreshold = *parsed.Dataset.SyncFlashBrightnessThreshold
	}
	if parsed.Dataset.SyncFlashFrameLimit != nil {
		cfg.SyncFlashFrameLimit = *parsed.Dataset.SyncFlashFrameLimit
	}

	if err := validateDatasetConfig("dataset", cfg); err != nil {
		return DatasetConfig{}, err
	}

	return cfg, nil
}

func loadTrainConfigFile(path string) (trainConfigFile, bool, error) {
	var parsed trainConfigFile
	if strings.TrimSpace(path) == "" {
		return parsed, false, nil
	}

	raw, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return parsed, false, nil
		}
		return trainConfigFile{}, false, err
	}

	if err := toml.Unmarshal(raw, &parsed); err != nil {
		return trainConfigFile{}, false, err
	}

	return parsed, true, nil
}

func validateDatasetConfig(prefix string, cfg DatasetConfig) error {
	if cfg.ImageWidth < 1 {
		return fmt.Errorf("%s image_width must be > 0", prefix)
	}
	if cfg.ImageHeight < 1 {
		return fmt.Errorf("%s image_height must be > 0", prefix)
	}
	if cfg.WindowSize < 1 || cfg.WindowSize%2 == 0 {
		return fmt.Errorf("%s window_size must be a positive odd number", prefix)
	}
	if len(cfg.ImageOffsets) < 1 {
		return fmt.Errorf("%s image_offsets must contain at least one entry", prefix)
	}
	if len(cfg.TelemetryOffsets) < 1 {
		return fmt.Errorf("%s telemetry_offsets must contain at least one entry", prefix)
	}
	if len(cfg.FutureOffsets) < 1 {
		return fmt.Errorf("%s future_offsets must contain at least one entry", prefix)
	}
	if len(cfg.ImageOffsets) != cfg.WindowSize {
		return fmt.Errorf("%s image_offsets length must match window_size", prefix)
	}
	if cfg.FrameStride < 1 {
		return fmt.Errorf("%s frame_stride must be > 0", prefix)
	}
	if cfg.SampleStride < 1 {
		return fmt.Errorf("%s sample_stride must be > 0", prefix)
	}
	if cfg.LabelTolerance <= 0 {
		return fmt.Errorf("%s label_tolerance must be > 0", prefix)
	}
	if cfg.DeltaSpeedClip <= 0 {
		return fmt.Errorf("%s delta_speed_clip must be > 0", prefix)
	}
	if cfg.SyncFlashBrightnessThreshold <= 0 {
		return fmt.Errorf("%s sync_flash_brightness_threshold must be > 0", prefix)
	}
	if cfg.SyncFlashFrameLimit < 1 {
		return fmt.Errorf("%s sync_flash_frame_limit must be > 0", prefix)
	}
	if cfg.TelemetryOffsets[len(cfg.TelemetryOffsets)-1] != 0 {
		return fmt.Errorf("%s telemetry_offsets must end at 0", prefix)
	}
	if len(cfg.TelemetryFeatureNames) < 1 {
		return fmt.Errorf("%s telemetry_feature_names must not be empty", prefix)
	}
	if len(cfg.ControlTargetNames) < 1 {
		return fmt.Errorf("%s control_target_names must not be empty", prefix)
	}
	if len(cfg.AuxTargetNames) < 1 {
		return fmt.Errorf("%s aux_target_names must not be empty", prefix)
	}
	return nil
}

func deriveLegacyImageOffsets(windowSize int, frameStride int) []int {
	if windowSize < 1 {
		return nil
	}
	offsets := make([]int, 0, windowSize)
	start := -((windowSize - 1) * frameStride)
	for index := 0; index < windowSize; index++ {
		offsets = append(offsets, start+(index*frameStride))
	}
	return offsets
}

func deriveTelemetryOffsets(length int) []int {
	if length < 1 {
		length = 3
	}
	offsets := make([]int, 0, length)
	start := -(length - 1)
	for index := 0; index < length; index++ {
		offsets = append(offsets, start+index)
	}
	return offsets
}
