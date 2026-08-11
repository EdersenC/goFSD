package main

import (
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"awesomeProject/internal/capture"
	datasetproc "awesomeProject/internal/dataset"
)

func TestBackendListenAddressDefaultsToLoopback(t *testing.T) {
	tests := []struct {
		name string
		host string
		port string
		want string
	}{
		{name: "defaults", want: "127.0.0.1:8080"},
		{name: "custom port", port: "9090", want: "127.0.0.1:9090"},
		{name: "explicit remote host", host: "0.0.0.0", port: "8081", want: "0.0.0.0:8081"},
		{name: "IPv6", host: "::1", want: "[::1]:8080"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := backendListenAddress(test.host, test.port); got != test.want {
				t.Fatalf("unexpected listen address: got=%q want=%q", got, test.want)
			}
		})
	}
}

func TestBackendHealthResponseIdentifiesParkingLab(t *testing.T) {
	health := backendHealthResponse()
	if health["status"] != "ok" || health["service"] != "stop-sign-lab-backend" {
		t.Fatalf("unexpected backend health identity: %+v", health)
	}
}

func TestWriteJSONDisablesBrowserCaching(t *testing.T) {
	recorder := httptest.NewRecorder()
	writeJSON(recorder, http.StatusOK, map[string]bool{"fivemConnected": true})

	if got := recorder.Header().Get("Cache-Control"); got != "no-store" {
		t.Fatalf("unexpected JSON cache policy: got=%q want=%q", got, "no-store")
	}
	if !strings.Contains(recorder.Body.String(), `"fivemConnected":true`) {
		t.Fatalf("unexpected JSON body: %s", recorder.Body.String())
	}
}

func TestRunBackendReturnsListenerFailure(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("reserve listener: %v", err)
	}
	defer listener.Close()

	port := listener.Addr().(*net.TCPAddr).Port
	t.Setenv("HOST", "127.0.0.1")
	t.Setenv("PORT", strconv.Itoa(port))

	err = runBackend([]string{"serve"}, io.Discard)
	if err == nil || !strings.Contains(err.Error(), "failed to bind capture API") {
		t.Fatalf("unexpected backend startup result: %v", err)
	}
}

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
		{
			Label: datasetproc.GroupedLabel{
				Control: datasetproc.GroupedLabelControl{
					Steering: 0.1,
				},
				Aux: datasetproc.GroupedLabelAux{
					FutureYawDelta:         15.0,
					FutureHorizonSeconds:   0.2,
					FutureSpeedDelta:       0.0,
					FutureSpeedDeltaTarget: 0.0,
					FutureSpeed:            5.0,
					FutureSpeedTarget:      5.0,
				},
			},
			TelemetryHistory: []datasetproc.GroupedTelemetryItem{
				{
					Aux: datasetproc.GroupedTelemetryAux{
						CurrentSpeed: 5.0,
						IsStopped:    0,
					},
				},
			},
		},
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
	datasetConfig, err := capture.LoadDatasetConfig(configPath)
	if err != nil {
		t.Fatalf("LoadDatasetConfig: %v", err)
	}
	processor := datasetproc.NewProcessor(datasetProcessorOptions(datasetConfig)...)
	writeCommandJSONFile(t, filepath.Join(tripDir, "processing.json"), datasetproc.ProcessingStatus{
		State:                     "completed",
		ConfigFingerprint:         processor.ConfigFingerprint(),
		ImageWidth:                datasetConfig.ImageWidth,
		ImageHeight:               datasetConfig.ImageHeight,
		ImageOffsets:              append([]int(nil), datasetConfig.ImageOffsets...),
		TelemetryOffsets:          append([]int(nil), datasetConfig.TelemetryOffsets...),
		FutureOffsets:             append([]int(nil), datasetConfig.FutureOffsets...),
		TelemetrySampleIntervalMs: float64(datasetConfig.TelemetrySampleInterval) / float64(time.Millisecond),
		FrameCount:                1,
		SampleCount:               0,
	})
	writeCommandDatasetJSONL(t, filepath.Join(tripDir, "dataset.jsonl"), nil)

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
future_speed_delta_clip = 2.0
future_speed_delta_normalize = true
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
