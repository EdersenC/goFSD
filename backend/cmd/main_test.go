package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	datasetproc "awesomeProject/internal/dataset"
)

func TestRunReportRunsWritesDatasetReport(t *testing.T) {
	root := t.TempDir()
	runDir := filepath.Join(root, "run-001")
	tripDir := filepath.Join(runDir, "scene-a_default", "trip-000")

	writeCommandTripMetadata(t, tripDir, map[string]any{
		"runId":        "run-001",
		"sceneId":      "scene-a",
		"sceneVariant": "default",
		"tripIndex":    0,
	})
	writeCommandJSONFile(t, filepath.Join(tripDir, "processing.json"), datasetproc.ProcessingStatus{
		State:      "completed",
		FrameCount: 1,
	})
	writeCommandDatasetJSONL(t, filepath.Join(tripDir, "dataset.jsonl"), []datasetproc.DatasetSample{
		{Label: map[string]any{"Steering": 0.1, "future_yaw_delta": 15.0, "future_horizon_seconds": 0.2, "delta_speed": 0.0, "delta_speed_target": 0.0, "future_speed": 5.0, "future_speed_target": 5.0, "currentSpeed": 5.0, "isStopped": 0}},
	})

	configPath := writeCLIConfig(t)
	t.Setenv("FSD_CONFIG_PATH", configPath)

	if err := runReportRuns([]string{"-root", root}); err != nil {
		t.Fatalf("runReportRuns: %v", err)
	}

	if _, err := os.Stat(filepath.Join(runDir, "dataset_report.json")); err != nil {
		t.Fatalf("dataset report not written: %v", err)
	}
}

func TestRunProcessRunsWritesDatasetReport(t *testing.T) {
	root := t.TempDir()
	runDir := filepath.Join(root, "run-002")
	tripDir := filepath.Join(runDir, "scene-a_default", "trip-000")
	framesDir := filepath.Join(tripDir, "frames")
	if err := os.MkdirAll(framesDir, 0o755); err != nil {
		t.Fatalf("mkdir frames dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(framesDir, "000001.jpg"), []byte("x"), 0o644); err != nil {
		t.Fatalf("write frame: %v", err)
	}
	writeCommandTripMetadata(t, tripDir, map[string]any{
		"runId":        "run-002",
		"sceneId":      "scene-a",
		"sceneVariant": "default",
		"tripIndex":    0,
	})

	configPath := writeCLIConfig(t)
	t.Setenv("FSD_CONFIG_PATH", configPath)

	if err := runProcessRuns([]string{"-root", root, "-workers", "1"}); err != nil {
		t.Fatalf("runProcessRuns: %v", err)
	}

	if _, err := os.Stat(filepath.Join(runDir, "dataset_report.json")); err != nil {
		t.Fatalf("dataset report not written: %v", err)
	}
}

func writeCLIConfig(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "train_config.toml")
	body := []byte(`
[dataset]
image_width = 224
image_height = 224
window_size = 3
frame_stride = 2
sample_stride = 2
label_tolerance = "100ms"
delta_speed_clip = 2.0
delta_speed_normalize = true
sync_flash_brightness_threshold = 245.0
sync_flash_frame_limit = 90
`)
	if err := os.WriteFile(path, body, 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	return path
}

func writeCommandTripMetadata(t *testing.T, tripDir string, metadata map[string]any) {
	t.Helper()
	if err := os.MkdirAll(tripDir, 0o755); err != nil {
		t.Fatalf("mkdir trip dir: %v", err)
	}
	writeCommandJSONFile(t, filepath.Join(tripDir, "metadata.json"), metadata)
}

func writeCommandDatasetJSONL(t *testing.T, path string, samples []datasetproc.DatasetSample) {
	t.Helper()
	file, err := os.Create(path)
	if err != nil {
		t.Fatalf("create dataset jsonl: %v", err)
	}
	enc := json.NewEncoder(file)
	for _, sample := range samples {
		if err := enc.Encode(sample); err != nil {
			_ = file.Close()
			t.Fatalf("encode dataset sample: %v", err)
		}
	}
	if err := file.Close(); err != nil {
		t.Fatalf("close dataset jsonl: %v", err)
	}
}

func writeCommandJSONFile(t *testing.T, path string, value any) {
	t.Helper()
	body, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("marshal json file: %v", err)
	}
	if err := os.WriteFile(path, append(body, '\n'), 0o644); err != nil {
		t.Fatalf("write json file: %v", err)
	}
}
