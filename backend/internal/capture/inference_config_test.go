package capture

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestLoadInferenceConfigReadsBackendInferenceSection(t *testing.T) {
	t.Setenv("INFERENCE_MODEL_SERVER_URL", "http://env.example")
	t.Setenv("INFERENCE_SOURCE_ID", "monitor-env")
	t.Setenv("INFERENCE_AUTO_LOAD_MODEL", "false")
	t.Setenv("INFERENCE_REQUEST_TIMEOUT", "3s")

	dir := t.TempDir()
	path := filepath.Join(dir, "train_config.toml")
	content := []byte(`
[backend.inference]
model_server_url = "http://127.0.0.1:9090"
model_device = "cuda"
source_id = "monitor-7"
auto_load = true
fps = 24
window_size = 5
window_stride = 2
frame_width = 480
frame_height = 480
request_timeout = "7s"
jpeg_quality = 82
`)
	if err := os.WriteFile(path, content, 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	cfg, err := LoadInferenceConfig(path)
	if err != nil {
		t.Fatalf("LoadInferenceConfig returned error: %v", err)
	}
	if cfg.ModelServerURL != "http://127.0.0.1:9090" {
		t.Fatalf("unexpected model server url: %s", cfg.ModelServerURL)
	}
	if cfg.ModelDevice != "cuda" {
		t.Fatalf("unexpected model device: %s", cfg.ModelDevice)
	}
	if cfg.SourceID != "monitor-7" {
		t.Fatalf("unexpected source id: %s", cfg.SourceID)
	}
	if !cfg.AutoLoad || cfg.FPS != 24 || cfg.WindowSize != 5 || cfg.WindowStride != 2 {
		t.Fatalf("unexpected config values: %+v", cfg)
	}
	if cfg.FrameWidth != 480 || cfg.FrameHeight != 480 {
		t.Fatalf("unexpected frame size: %dx%d", cfg.FrameWidth, cfg.FrameHeight)
	}
	if cfg.RequestTimeout != 7*time.Second {
		t.Fatalf("unexpected timeout: %s", cfg.RequestTimeout)
	}
	if cfg.JPEGQuality != 82 {
		t.Fatalf("unexpected jpeg quality: %d", cfg.JPEGQuality)
	}
}

func TestLoadInferenceConfigFallsBackToDefaultsWhenFileMissing(t *testing.T) {
	cfg, err := LoadInferenceConfig(filepath.Join(t.TempDir(), "missing.toml"))
	if err != nil {
		t.Fatalf("LoadInferenceConfig returned error: %v", err)
	}
	if cfg.ModelDevice != "cuda" {
		t.Fatalf("unexpected default model device: %s", cfg.ModelDevice)
	}
	if cfg.FrameWidth != defaultInferenceWidth || cfg.FrameHeight != defaultInferenceHeight {
		t.Fatalf("unexpected default frame size: %dx%d", cfg.FrameWidth, cfg.FrameHeight)
	}
}
