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
image_offsets = [-8, -6, -4, -2, 0]
frame_stride = 2
sample_stride = 10
telemetry_offsets = [-8, -7, -6, -5, -4, -3, -2, -1, 0]
future_offsets = [1, 2, 3, 4, 5, 6]
telemetry_feature_names = ["current_speed", "yaw_sin", "yaw_cos", "yaw_rate", "steering", "acceleration"]
control_target_names = ["steering", "acceleration", "brakePressureAvg"]
aux_target_names = ["future_speed", "future_speed_delta", "future_yaw_delta", "future_yaw_rate"]
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
low_speed_steer_gain = 1.4
high_speed_steer_gain = 0.8
steer_gain_fade_speed_mps = 6.5
steer_response_blend = 0.6
max_target_speed_kph = 17.0
steer_command_rate_per_second = 6.0
throttle_hold_seconds = 2.3
throttle_hold_min = 0.09
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
	if cfg.LowSpeedSteerGain != 1.4 || cfg.HighSpeedSteerGain != 0.8 {
		t.Fatalf("unexpected steer gains: low=%f high=%f", cfg.LowSpeedSteerGain, cfg.HighSpeedSteerGain)
	}
	if cfg.SteerGainFadeSpeedMPS != 6.5 {
		t.Fatalf("unexpected steer gain fade speed: %f", cfg.SteerGainFadeSpeedMPS)
	}
	if cfg.SteerResponseBlend != 0.6 {
		t.Fatalf("unexpected steer response blend: %f", cfg.SteerResponseBlend)
	}
	if cfg.MaxTargetSpeedKPH != 17.0 {
		t.Fatalf("unexpected max target speed kph: %f", cfg.MaxTargetSpeedKPH)
	}
	if cfg.SteerCommandRatePerSec != 6.0 {
		t.Fatalf("unexpected steer command rate: %f", cfg.SteerCommandRatePerSec)
	}
	if cfg.ThrottleHoldSeconds != 2.3 {
		t.Fatalf("unexpected throttle hold seconds: %f", cfg.ThrottleHoldSeconds)
	}
	if cfg.ThrottleHoldMin != 0.09 {
		t.Fatalf("unexpected throttle hold min: %f", cfg.ThrottleHoldMin)
	}
}

func TestLoadInferenceConfigFallsBackToDefaultsWhenFileMissing(t *testing.T) {
	cfg, err := LoadInferenceConfig("")
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
	if cfg.MaxTargetSpeedKPH != defaultMaxTargetSpeedKPH {
		t.Fatalf("unexpected default max target speed kph: %f", cfg.MaxTargetSpeedKPH)
	}
	if cfg.LowSpeedSteerGain != defaultLowSpeedSteerGain || cfg.HighSpeedSteerGain != defaultHighSpeedSteerGain {
		t.Fatalf("unexpected default steer gains: low=%f high=%f", cfg.LowSpeedSteerGain, cfg.HighSpeedSteerGain)
	}
	if cfg.SteerGainFadeSpeedMPS != defaultSteerGainFadeSpeedMPS {
		t.Fatalf("unexpected default steer gain fade speed: %f", cfg.SteerGainFadeSpeedMPS)
	}
	if cfg.SteerResponseBlend != defaultSteerResponseBlend {
		t.Fatalf("unexpected default steer response blend: %f", cfg.SteerResponseBlend)
	}
	if cfg.SteerCommandRatePerSec != defaultSteerCommandRatePerSecond {
		t.Fatalf("unexpected default steer command rate: %f", cfg.SteerCommandRatePerSec)
	}
	if cfg.ThrottleHoldSeconds != defaultThrottleHoldSeconds {
		t.Fatalf("unexpected default throttle hold seconds: %f", cfg.ThrottleHoldSeconds)
	}
	if cfg.ThrottleHoldMin != defaultThrottleHoldMin {
		t.Fatalf("unexpected default throttle hold min: %f", cfg.ThrottleHoldMin)
	}
	if cfg.PredictionTimeout != defaultPredictionTimeout {
		t.Fatalf("unexpected default inference timeout: %s", cfg.PredictionTimeout)
	}
	if cfg.AlignmentTolerance != defaultAlignmentTolerance {
		t.Fatalf("unexpected default alignment tolerance: %s", cfg.AlignmentTolerance)
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
future_speed_delta_clip = 1.5
future_speed_delta_normalize = false
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
	if cfg.FutureSpeedDeltaClip != 1.5 || cfg.FutureSpeedDeltaNormalize {
		t.Fatalf("unexpected future-speed-delta transform config: %+v", cfg)
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
