package capture

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	toml "github.com/pelletier/go-toml/v2"
)

const (
	defaultInferenceConfigRelativePath = "fsd_trainer/train_config.toml"
)

type InferenceConfig struct {
	ConfigPath     string
	ModelServerURL string
	ModelDevice    string
	SourceID       string
	AutoLoad       bool
	FPS            int
	WindowSize     int
	WindowStride   int
	FrameWidth     int
	FrameHeight    int
	RequestTimeout time.Duration
	JPEGQuality    int
}

type backendConfigFile struct {
	Backend backendSection `toml:"backend"`
}

type backendSection struct {
	Inference backendInferenceSection `toml:"inference"`
}

type backendInferenceSection struct {
	ModelServerURL string `toml:"model_server_url"`
	ModelDevice    string `toml:"model_device"`
	SourceID       string `toml:"source_id"`
	AutoLoad       *bool  `toml:"auto_load"`
	FPS            int    `toml:"fps"`
	WindowSize     int    `toml:"window_size"`
	WindowStride   int    `toml:"window_stride"`
	FrameWidth     int    `toml:"frame_width"`
	FrameHeight    int    `toml:"frame_height"`
	RequestTimeout string `toml:"request_timeout"`
	JPEGQuality    int    `toml:"jpeg_quality"`
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
		ModelServerURL: url,
		ModelDevice:    strings.ToLower(modelDevice),
		SourceID:       sourceID,
		AutoLoad:       parseBoolEnv("INFERENCE_AUTO_LOAD_MODEL", true),
		FPS:            defaultInferenceFPS,
		WindowSize:     defaultInferenceWindowSize,
		WindowStride:   defaultInferenceStride,
		FrameWidth:     defaultInferenceWidth,
		FrameHeight:    defaultInferenceHeight,
		RequestTimeout: parseDurationEnv("INFERENCE_REQUEST_TIMEOUT", defaultInferenceRequestTimeout),
		JPEGQuality:    defaultInferenceJPEGQuality,
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
	if strings.TrimSpace(path) == "" {
		return cfg, nil
	}

	raw, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return cfg, nil
		}
		return InferenceConfig{}, err
	}

	var parsed backendConfigFile
	if err := toml.Unmarshal(raw, &parsed); err != nil {
		return InferenceConfig{}, err
	}

	section := parsed.Backend.Inference
	cfg.ConfigPath = path
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
	if section.WindowSize > 0 {
		cfg.WindowSize = section.WindowSize
	}
	if section.WindowStride > 0 {
		cfg.WindowStride = section.WindowStride
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

	if strings.TrimSpace(cfg.ModelDevice) == "" {
		return InferenceConfig{}, fmt.Errorf("backend inference model_device must not be empty")
	}
	if cfg.FPS < 1 {
		return InferenceConfig{}, fmt.Errorf("backend inference fps must be > 0")
	}
	if cfg.WindowSize < 1 || cfg.WindowSize%2 == 0 {
		return InferenceConfig{}, fmt.Errorf("backend inference window_size must be a positive odd number")
	}
	if cfg.WindowStride < 1 {
		return InferenceConfig{}, fmt.Errorf("backend inference window_stride must be > 0")
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
