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
[dataset]
image_width = 320
image_height = 180
window_size = 5
frame_stride = 2
sample_stride = 10
label_tolerance = "120ms"
sync_flash_brightness_threshold = 250.0
sync_flash_frame_limit = 45

[backend.inference]
model_server_url = "http://127.0.0.1:9090"
model_device = "cuda"
source_id = "monitor-7"
auto_load = true
fps = 24
dispatch_stride = 3
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
	if !cfg.AutoLoad || cfg.FPS != 24 || cfg.WindowSize != 5 || cfg.FrameStride != 2 || cfg.DispatchStride != 3 {
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
	if cfg.DispatchStride != defaultInferenceStride {
		t.Fatalf("unexpected default dispatch stride: %d", cfg.DispatchStride)
	}
	if cfg.FrameWidth != defaultInferenceWidth || cfg.FrameHeight != defaultInferenceHeight {
		t.Fatalf("unexpected default frame size: %dx%d", cfg.FrameWidth, cfg.FrameHeight)
	}
}

func TestLoadDatasetConfigReadsDatasetSection(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "train_config.toml")
	content := []byte(`
[dataset]
image_width = 640
image_height = 360
window_size = 7
frame_stride = 3
sample_stride = 9
label_tolerance = "75ms"
delta_speed_clip = 1.5
delta_speed_normalize = false
sync_flash_brightness_threshold = 200.5
sync_flash_frame_limit = 25
`)
	if err := os.WriteFile(path, content, 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	cfg, err := LoadDatasetConfig(path)
	if err != nil {
		t.Fatalf("LoadDatasetConfig returned error: %v", err)
	}
	if cfg.WindowSize != 7 || cfg.FrameStride != 3 || cfg.SampleStride != 9 {
		t.Fatalf("unexpected dataset config: %+v", cfg)
	}
	if cfg.ImageWidth != 640 || cfg.ImageHeight != 360 {
		t.Fatalf("unexpected image size: %dx%d", cfg.ImageWidth, cfg.ImageHeight)
	}
	if cfg.LabelTolerance != 75*time.Millisecond {
		t.Fatalf("unexpected label tolerance: %s", cfg.LabelTolerance)
	}
	if cfg.DeltaSpeedClip != 1.5 || cfg.DeltaSpeedNormalize {
		t.Fatalf("unexpected delta-speed transform config: %+v", cfg)
	}
	if cfg.SyncFlashBrightnessThreshold != 200.5 || cfg.SyncFlashFrameLimit != 25 {
		t.Fatalf("unexpected sync-flash config: %+v", cfg)
	}
}

func TestLoadDatasetConfigRejectsLegacyWindowStride(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "train_config.toml")
	content := []byte(`
[dataset]
window_size = 5
window_stride = 2
`)
	if err := os.WriteFile(path, content, 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	if _, err := LoadDatasetConfig(path); err == nil {
		t.Fatalf("expected legacy window_stride config to fail")
	}
}

func TestLoadDatasetConfigRejectsMissingSampleStride(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "train_config.toml")
	content := []byte(`
[dataset]
window_size = 5
frame_stride = 2
`)
	if err := os.WriteFile(path, content, 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	if _, err := LoadDatasetConfig(path); err == nil {
		t.Fatalf("expected missing sample_stride config to fail")
	}
}

func TestLoadDatasetConfigRejectsInvalidImageSize(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "train_config.toml")
	content := []byte(`
[dataset]
image_width = 0
image_height = 480
window_size = 5
frame_stride = 2
sample_stride = 10
`)
	if err := os.WriteFile(path, content, 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	if _, err := LoadDatasetConfig(path); err == nil {
		t.Fatalf("expected invalid image size config to fail")
	}
}
