package dataset

import (
	"context"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestAttachImagePaths(t *testing.T) {
	frames := []VideoFrame{{Index: 0, PTS: 0.0}, {Index: 1, PTS: 0.1}, {Index: 2, PTS: 0.2}}
	got := AttachImagePaths(frames, "frames")
	want := []string{"frames/000001.jpg", "frames/000002.jpg", "frames/000003.jpg"}
	for i := range want {
		if got[i].ImagePath != want[i] {
			t.Fatalf("unexpected image path at %d: got=%q want=%q", i, got[i].ImagePath, want[i])
		}
	}
}

func TestLoadRunTripRecord(t *testing.T) {
	tmp := t.TempDir()
	path := filepath.Join(tmp, "run.jsonl")
	rows := []runTripRecord{
		{RunID: "run-a", TripIndex: 0},
		{RunID: "run-a", TripIndex: 1, VehicleData: []map[string]any{{"time": 1200.0, "Steering": 0.25}}},
	}
	file, err := os.Create(path)
	if err != nil {
		t.Fatalf("create run file: %v", err)
	}
	enc := json.NewEncoder(file)
	for _, row := range rows {
		if err := enc.Encode(row); err != nil {
			t.Fatalf("encode row: %v", err)
		}
	}
	_ = file.Close()

	record, err := loadRunTripRecord(path, "run-a", 1)
	if err != nil {
		t.Fatalf("loadRunTripRecord: %v", err)
	}
	if record.TripIndex != 1 || len(record.VehicleData) != 1 {
		t.Fatalf("unexpected record: %+v", record)
	}
}

func TestWaitForTripReadinessWaitsForManifestRow(t *testing.T) {
	tmp := t.TempDir()
	sceneDir := filepath.Join(tmp, "run-a", "scene-a_default")
	tripDir := filepath.Join(sceneDir, "trip-000")
	if err := os.MkdirAll(tripDir, 0o755); err != nil {
		t.Fatalf("mkdir trip dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(tripDir, "video.mkv"), []byte("video"), 0o644); err != nil {
		t.Fatalf("write video: %v", err)
	}

	go func() {
		time.Sleep(50 * time.Millisecond)
		writeJSONFile(t, filepath.Join(tripDir, "metadata.json"), tripMetadata{
			RunID:     "run-a",
			TripIndex: 0,
		})

		runFile := filepath.Join(sceneDir, "run.jsonl")
		file, err := os.Create(runFile)
		if err != nil {
			t.Errorf("create run manifest: %v", err)
			return
		}
		defer file.Close()

		if err := json.NewEncoder(file).Encode(runTripRecord{
			RunID:     "run-a",
			TripIndex: 0,
		}); err != nil {
			t.Errorf("encode run record: %v", err)
		}
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()
	if err := WaitForTripReadiness(ctx, tripDir, 10*time.Millisecond); err != nil {
		t.Fatalf("WaitForTripReadiness: %v", err)
	}
}

func TestBuildDatasetSamplesUsesNearestRawLabel(t *testing.T) {
	rawFrames := make([]VideoFrame, 0, 13)
	labels := make([]timedLabel, 0, 12)
	for index := 0; index < 13; index++ {
		rawFrames = append(rawFrames, VideoFrame{
			Index: index,
			PTS:   float64(index) * 0.1,
		})
		if index < 12 {
			labels = append(labels, timedLabel{
				RelativeSeconds: float64(index) * 0.1,
				Label: map[string]any{
					"time":              float64(index) * 100.0,
					"Steering":          float64(index) * 0.1,
					"currentSpeed":      float64(index),
					"acceleration":      float64(index) * 0.05,
					"yaw":               10.0 + (float64(index) * 2.0),
					"yawRate":           float64(index) * 0.5,
					"routeForwardDelta": float64(index) * 0.1,
				},
			})
		}
	}
	frames := AttachImagePaths(rawFrames, "frames")

	samples := buildDatasetSamples(frames, labels, 0.0, 3, 2, 4, 100*time.Millisecond, 2.0, true)
	if len(samples) != 1 {
		t.Fatalf("unexpected sample count: got=%d want=1", len(samples))
	}
	if samples[0].AnchorVideoPTS != 0.4 {
		t.Fatalf("unexpected anchor pts: %+v", samples[0])
	}
	if samples[0].AnchorGameTime != 0.4 {
		t.Fatalf("unexpected anchor game time: %+v", samples[0])
	}
	if !reflect.DeepEqual(samples[0].FramePaths, []string{"frames/000001.jpg", "frames/000003.jpg", "frames/000005.jpg"}) {
		t.Fatalf("unexpected frame paths: %+v", samples[0].FramePaths)
	}
	flatLabel := flattenedLabel(samples[0].Label)
	if flatLabel["Steering"] != 0.4 {
		t.Fatalf("unexpected label payload: %+v", flatLabel)
	}
	if flatLabel["acceleration"] != nil {
		t.Fatalf("expected raw acceleration to be dropped, label=%+v", flatLabel)
	}
	if flatLabel["future_speed_delta"] != 2.0 {
		t.Fatalf("unexpected future_speed_delta: %+v", flatLabel)
	}
	if flatLabel["future_speed_delta_target"] != 1.0 {
		t.Fatalf("unexpected future_speed_delta_target: %+v", flatLabel)
	}
	if flatLabel["future_speed"] != 8.0 {
		t.Fatalf("unexpected future_speed: %+v", flatLabel)
	}
	if flatLabel["future_speed_target"] != 8.0 {
		t.Fatalf("unexpected future_speed_target: %+v", flatLabel)
	}
	if flatLabel["future_yaw_delta"] != 8.0 {
		t.Fatalf("unexpected future_yaw_delta: %+v", flatLabel)
	}
	if math.Abs(flatLabel["future_horizon_seconds"].(float64)-0.4) > 1e-6 {
		t.Fatalf("unexpected future_horizon_seconds: %+v", flatLabel)
	}
	if flatLabel["yaw_rate"] != 4.0 {
		t.Fatalf("unexpected yaw_rate: %+v", flatLabel)
	}
	if flatLabel["routeForwardDelta"] != 0.4 {
		t.Fatalf("unexpected routeForwardDelta: %+v", flatLabel)
	}
	if samples[0].Label.Control.Steering == nil || samples[0].Label.Aux.FutureYawDelta == nil {
		t.Fatalf("expected grouped label sections to be populated: %+v", samples[0].Label)
	}
	history := sampleTelemetryWindow(t, samples[0].TelemetryHistory, "telemetry_history")
	if len(history) != 5 {
		t.Fatalf("unexpected telemetry history length: got=%d", len(history))
	}
	if history[0]["time"] != 0.0 || history[4]["time"] != 400.0 {
		t.Fatalf("unexpected telemetry history window: %+v", history)
	}
	if history[4]["acceleration"] != 0.2 {
		t.Fatalf("expected telemetry history to keep raw acceleration: %+v", history[0])
	}
	future := sampleTelemetryWindow(t, samples[0].TelemetryFuture, "telemetry_future")
	if len(future) != defaultFutureTelemetryCount {
		t.Fatalf("unexpected telemetry future length: got=%d", len(future))
	}
	if future[0]["time"] != 500.0 || future[len(future)-1]["time"] != 1000.0 {
		t.Fatalf("unexpected telemetry future window: %+v", future)
	}
	if future[0]["acceleration"] != 0.25 {
		t.Fatalf("expected telemetry future to keep raw acceleration: %+v", future[0])
	}
}

func TestBuildDatasetSamplesSupportsConfigurableWindowSize(t *testing.T) {
	rawFrames := make([]VideoFrame, 0, 17)
	labels := make([]timedLabel, 0, 17)
	for index := 0; index < 17; index++ {
		rawFrames = append(rawFrames, VideoFrame{Index: index, PTS: float64(index) * 0.1})
		labels = append(labels, timedLabel{
			RelativeSeconds: float64(index) * 0.1,
			Label: map[string]any{
				"time":              float64(index) * 100.0,
				"Steering":          0.2,
				"currentSpeed":      6.0 + float64(index),
				"yaw":               32.0 + float64(index),
				"yawRate":           0.5 + (0.1 * float64(index)),
				"routeForwardDelta": 0.5 + (0.05 * float64(index)),
			},
		})
	}
	frames := AttachImagePaths(rawFrames, "frames")

	samples := buildDatasetSamples(frames, labels, 0.0, 5, 2, 8, 100*time.Millisecond, 2.0, true)
	if len(samples) != 1 {
		t.Fatalf("unexpected sample count: got=%d want=1", len(samples))
	}
	if !reflect.DeepEqual(samples[0].FramePaths, []string{
		"frames/000001.jpg",
		"frames/000003.jpg",
		"frames/000005.jpg",
		"frames/000007.jpg",
		"frames/000009.jpg",
	}) {
		t.Fatalf("unexpected frame paths: %+v", samples[0].FramePaths)
	}
	history := sampleTelemetryWindow(t, samples[0].TelemetryHistory, "telemetry_history")
	if len(history) != 9 {
		t.Fatalf("unexpected telemetry history length: got=%d", len(history))
	}
	if history[0]["time"] != 0.0 || history[len(history)-1]["time"] != 800.0 {
		t.Fatalf("unexpected telemetry history window: %+v", history)
	}
}

func TestBuildDatasetSamplesUsesIndependentSampleStride(t *testing.T) {
	rawFrames := make([]VideoFrame, 0, 41)
	labels := make([]timedLabel, 0, 41)
	for index := 0; index <= 40; index++ {
		rawFrames = append(rawFrames, VideoFrame{
			Index: index,
			PTS:   float64(index) * 0.1,
		})
		labels = append(labels, timedLabel{
			RelativeSeconds: float64(index) * 0.1,
			Label: map[string]any{
				"time":              float64(index) * 100.0,
				"Steering":          0.1 + (0.01 * float64(index)),
				"currentSpeed":      4.0 + float64(index),
				"yaw":               10.0 + float64(index),
				"yawRate":           0.25 + (0.05 * float64(index)),
				"routeForwardDelta": 0.25 + (0.01 * float64(index)),
			},
		})
	}
	frames := AttachImagePaths(rawFrames, "frames")

	samples := buildDatasetSamples(frames, labels, 0.0, 5, 2, 10, 100*time.Millisecond, 2.0, true)
	if len(samples) != 3 {
		t.Fatalf("unexpected sample count: got=%d want=3", len(samples))
	}
	if !reflect.DeepEqual(samples[0].FramePaths, []string{
		"frames/000003.jpg",
		"frames/000005.jpg",
		"frames/000007.jpg",
		"frames/000009.jpg",
		"frames/000011.jpg",
	}) {
		t.Fatalf("unexpected first sample frame paths: %+v", samples[0].FramePaths)
	}
	if !reflect.DeepEqual(samples[1].FramePaths, []string{
		"frames/000013.jpg",
		"frames/000015.jpg",
		"frames/000017.jpg",
		"frames/000019.jpg",
		"frames/000021.jpg",
	}) {
		t.Fatalf("unexpected second sample frame paths: %+v", samples[1].FramePaths)
	}
}

func TestBuildDatasetSamplesSkipsIncompleteTelemetryWindows(t *testing.T) {
	frames := AttachImagePaths([]VideoFrame{
		{Index: 0, PTS: 0.0},
		{Index: 1, PTS: 0.1},
		{Index: 2, PTS: 0.2},
		{Index: 3, PTS: 0.3},
		{Index: 4, PTS: 0.4},
		{Index: 5, PTS: 0.5},
		{Index: 6, PTS: 0.6},
	}, "frames")
	labels := []timedLabel{
		{RelativeSeconds: 0.4, Label: map[string]any{"time": 400.0, "Steering": 0.1, "currentSpeed": 1.0, "acceleration": 0.2, "yaw": 10.0, "yawRate": 0.25, "routeForwardDelta": 0.1}},
		{RelativeSeconds: 0.5, Label: map[string]any{"time": 500.0, "Steering": 0.2, "currentSpeed": 2.0, "acceleration": 0.3, "yaw": 14.0, "yawRate": 0.5, "routeForwardDelta": 0.2}},
		{RelativeSeconds: 0.6, Label: map[string]any{"time": 600.0, "Steering": 0.2, "currentSpeed": 3.0, "acceleration": 0.4, "yaw": 18.0, "yawRate": 0.75, "routeForwardDelta": 0.3}},
		{RelativeSeconds: 0.7, Label: map[string]any{"time": 700.0, "Steering": 0.2, "currentSpeed": 4.0, "acceleration": 0.5, "yaw": 22.0, "yawRate": 1.0, "routeForwardDelta": 0.4}},
		{RelativeSeconds: 0.8, Label: map[string]any{"time": 800.0, "Steering": 0.2, "currentSpeed": 5.0, "acceleration": 0.6, "yaw": 26.0, "yawRate": 1.25, "routeForwardDelta": 0.5}},
		{RelativeSeconds: 0.9, Label: map[string]any{"time": 900.0, "Steering": 0.2, "currentSpeed": 6.0, "acceleration": 0.7, "yaw": 30.0, "yawRate": 1.5, "routeForwardDelta": 0.6}},
		{RelativeSeconds: 1.0, Label: map[string]any{"time": 1000.0, "Steering": 0.2, "currentSpeed": 7.0, "acceleration": 0.8, "yaw": 34.0, "yawRate": 1.75, "routeForwardDelta": 0.7}},
	}

	samples, stats := buildDatasetSamplesWithStats(frames, labels, 0.0, 3, 2, 4, 100*time.Millisecond, 2.0, true)
	if len(samples) != 0 {
		t.Fatalf("unexpected sample count: got=%d want=0", len(samples))
	}
	if stats.IncompleteTelemetryHistoryCount != 1 {
		t.Fatalf("unexpected telemetry history skip stats: %+v", stats)
	}
	if reasons := stats.zeroSampleReasons(); reasons["incomplete_telemetry_history"] != 1 {
		t.Fatalf("unexpected zero sample reasons: %+v", reasons)
	}
}

func TestDetectSyncFlashPTS(t *testing.T) {
	tmp := t.TempDir()
	framesDir := filepath.Join(tmp, "frames")
	if err := os.MkdirAll(framesDir, 0o755); err != nil {
		t.Fatalf("mkdir frames: %v", err)
	}

	brightnesses := []uint8{30, 40, 255, 80}
	frames := make([]VideoFrame, 0, len(brightnesses))
	for i, brightness := range brightnesses {
		path := filepath.Join(framesDir, formatFrameName(i+1))
		writeJPEG(t, path, brightness)
		frames = append(frames, VideoFrame{Index: i, PTS: float64(i) * 0.1})
	}
	frames = AttachImagePaths(frames, "frames")

	pts, err := detectSyncFlashPTS(tmp, frames, 10, 245)
	if err != nil {
		t.Fatalf("detectSyncFlashPTS: %v", err)
	}
	if pts != 0.2 {
		t.Fatalf("unexpected sync pts: got=%v want=0.2", pts)
	}
}

func TestProbeVideoFramesParsesFFprobeJSON(t *testing.T) {
	processor := NewProcessor(WithCommandFactory(func(_ context.Context, _ string, _ ...string) *exec.Cmd {
		cmd := exec.Command(os.Args[0], "-test.run=TestHelperProcessFFprobe", "--")
		cmd.Env = append(os.Environ(), "GO_WANT_HELPER_PROCESS=ffprobe")
		return cmd
	}))

	frames, err := processor.ProbeVideoFrames(context.Background(), "video.mkv")
	if err != nil {
		t.Fatalf("ProbeVideoFrames: %v", err)
	}
	if len(frames) != 2 {
		t.Fatalf("unexpected frame count: got=%d want=2", len(frames))
	}
	if frames[1].PTS != 0.1 {
		t.Fatalf("unexpected frame payload: %+v", frames[1])
	}
}

func TestQueueWritesConfiguredImageSizeToStatus(t *testing.T) {
	tripDir := t.TempDir()
	processor := NewProcessor(WithImageSize(320, 180))

	statusPath, err := processor.Queue(tripDir)
	if err != nil {
		t.Fatalf("Queue: %v", err)
	}

	status, err := ReadStatusFile(statusPath)
	if err != nil {
		t.Fatalf("ReadStatusFile: %v", err)
	}
	if status.ImageWidth != 320 || status.ImageHeight != 180 {
		t.Fatalf("unexpected status image size: %+v", status)
	}
}

func TestExtractFramesResizesToConfiguredImageSize(t *testing.T) {
	tmp := t.TempDir()
	framesDir := filepath.Join(tmp, "frames")

	err := extractFrames(
		context.Background(),
		func(_ context.Context, _ string, args ...string) *exec.Cmd {
			cmdArgs := append([]string{"-test.run=TestHelperProcessFFmpeg", "--"}, args...)
			cmd := exec.Command(os.Args[0], cmdArgs...)
			cmd.Env = append(os.Environ(), "GO_WANT_HELPER_PROCESS=ffmpeg")
			return cmd
		},
		"ffmpeg",
		"video.mkv",
		framesDir,
		320,
		180,
	)
	if err != nil {
		t.Fatalf("extractFrames: %v", err)
	}

	file, err := os.Open(filepath.Join(framesDir, "000001.jpg"))
	if err != nil {
		t.Fatalf("open extracted frame: %v", err)
	}
	defer file.Close()

	cfg, _, err := image.DecodeConfig(file)
	if err != nil {
		t.Fatalf("decode extracted frame: %v", err)
	}
	if cfg.Width != 320 || cfg.Height != 180 {
		t.Fatalf("unexpected extracted frame size: %dx%d", cfg.Width, cfg.Height)
	}
}

func TestShouldSkipProcessing(t *testing.T) {
	tmp := t.TempDir()
	if shouldSkipProcessing(tmp) {
		t.Fatalf("expected empty trip dir not to be skipped")
	}

	if err := os.WriteFile(filepath.Join(tmp, "dataset.jsonl"), []byte("{}\n"), 0o644); err != nil {
		t.Fatalf("write dataset.jsonl: %v", err)
	}
	if !shouldSkipProcessing(tmp) {
		t.Fatalf("expected dataset.jsonl to trigger skip")
	}

	if err := os.Remove(filepath.Join(tmp, "dataset.jsonl")); err != nil {
		t.Fatalf("remove dataset.jsonl: %v", err)
	}
	framesDir := filepath.Join(tmp, "frames")
	if err := os.MkdirAll(framesDir, 0o755); err != nil {
		t.Fatalf("mkdir frames: %v", err)
	}
	if err := os.WriteFile(filepath.Join(framesDir, "000001.jpg"), []byte("x"), 0o644); err != nil {
		t.Fatalf("write frame: %v", err)
	}
	if !shouldSkipProcessing(tmp) {
		t.Fatalf("expected non-empty frames dir to trigger skip")
	}
}

func TestShouldSkipDatasetOnlyProcessing(t *testing.T) {
	tmp := t.TempDir()
	if shouldSkipDatasetOnlyProcessing(tmp) {
		t.Fatalf("expected missing dataset.jsonl not to be skipped")
	}

	if err := os.WriteFile(filepath.Join(tmp, "dataset.jsonl"), []byte("{}\n"), 0o644); err != nil {
		t.Fatalf("write dataset.jsonl: %v", err)
	}
	if !shouldSkipDatasetOnlyProcessing(tmp) {
		t.Fatalf("expected dataset.jsonl to trigger dataset-only skip")
	}
}

func TestSmoothedFutureSpeedUsesAvailableNeighbors(t *testing.T) {
	labels := []timedLabel{
		{RelativeSeconds: 0.2, Label: map[string]any{"currentSpeed": 4.0}},
		{RelativeSeconds: 0.4, Label: map[string]any{"currentSpeed": 6.0}},
		{RelativeSeconds: 0.6, Label: map[string]any{"currentSpeed": 8.0}},
		{RelativeSeconds: 0.8, Label: map[string]any{"currentSpeed": 10.0}},
	}

	got, ok := smoothedFutureSpeed(labels, 1, 2)
	if !ok {
		t.Fatal("expected smoothedFutureSpeed to succeed")
	}
	if got != 7.0 {
		t.Fatalf("unexpected smoothed future speed: got=%v want=7.0", got)
	}
}

func TestSmoothedFutureYawWrapsAcrossZeroDegrees(t *testing.T) {
	labels := []timedLabel{
		{RelativeSeconds: 0.2, Label: map[string]any{"yaw": 359.0}},
		{RelativeSeconds: 0.4, Label: map[string]any{"yaw": 1.0}},
		{RelativeSeconds: 0.6, Label: map[string]any{"yaw": 3.0}},
	}

	got, ok := smoothedFutureYaw(labels, 1, 1)
	if !ok {
		t.Fatal("expected smoothedFutureYaw to succeed")
	}
	if math.Abs(got-1.0) > 1e-6 {
		t.Fatalf("unexpected smoothed future yaw: got=%v want=1.0", got)
	}
}

func TestWrapHeadingDeltaDegreesWrapsAcrossZero(t *testing.T) {
	if got := wrapHeadingDeltaDegrees(359.0 - 1.0); math.Abs(got+2.0) > 1e-6 {
		t.Fatalf("unexpected wrapped heading delta: got=%v want=-2.0", got)
	}
	if got := wrapHeadingDeltaDegrees(1.0 - 359.0); math.Abs(got-2.0) > 1e-6 {
		t.Fatalf("unexpected wrapped heading delta: got=%v want=2.0", got)
	}
}

func TestSmoothedYawRateUsesAvailableNeighbors(t *testing.T) {
	labels := []timedLabel{
		{RelativeSeconds: 0.2, Label: map[string]any{"yawRate": -3.0}},
		{RelativeSeconds: 0.4, Label: map[string]any{"yawRate": 0.0}},
		{RelativeSeconds: 0.6, Label: map[string]any{"yawRate": 6.0}},
	}

	got, ok := smoothedYawRate(labels, 1, 1)
	if !ok {
		t.Fatal("expected smoothedYawRate to succeed")
	}
	if math.Abs(got-1.0) > 1e-6 {
		t.Fatalf("unexpected smoothed yaw rate: got=%v want=1.0", got)
	}
}

func TestSmoothedYawRateDerivesFromYawWhenRawFieldIsMissing(t *testing.T) {
	labels := []timedLabel{
		{RelativeSeconds: 0.2, Label: map[string]any{"yaw": 359.0}},
		{RelativeSeconds: 0.4, Label: map[string]any{"yaw": 1.0}},
		{RelativeSeconds: 0.6, Label: map[string]any{"yaw": 3.0}},
	}

	got, ok := smoothedYawRate(labels, 1, 0)
	if !ok {
		t.Fatal("expected smoothedYawRate to derive from yaw")
	}
	want := degreesToRadians(4.0) / 0.4
	if math.Abs(got-want) > 1e-6 {
		t.Fatalf("unexpected derived yaw rate: got=%v want=%v", got, want)
	}
}

func TestBuildDatasetSamplesWithStatsTracksMissingFutureYawTargets(t *testing.T) {
	rawFrames := make([]VideoFrame, 0, 13)
	labels := make([]timedLabel, 0, 12)
	for index := 0; index < 13; index++ {
		rawFrames = append(rawFrames, VideoFrame{Index: index, PTS: float64(index) * 0.1})
		if index < 12 {
			labels = append(labels, timedLabel{
				RelativeSeconds: float64(index) * 0.1,
				Label: map[string]any{
					"time":              float64(index) * 100.0,
					"Steering":          0.1,
					"currentSpeed":      3.0 + float64(index),
					"acceleration":      0.1,
					"routeForwardDelta": 0.2,
				},
			})
		}
	}
	frames := AttachImagePaths(rawFrames, "frames")

	samples, stats := buildDatasetSamplesWithStats(frames, labels, 0.0, 3, 2, 4, 100*time.Millisecond, 2.0, true)
	if len(samples) != 0 {
		t.Fatalf("expected no samples, got=%d", len(samples))
	}
	if stats.MissingFutureYawTargetCount != 1 {
		t.Fatalf("unexpected missing future-yaw count: %+v", stats)
	}
	if stats.GeneratedSampleCount != 0 {
		t.Fatalf("unexpected generated sample count: %+v", stats)
	}
	reasons := stats.zeroSampleReasons()
	if reasons["missing_future_yaw_target"] != 1 {
		t.Fatalf("unexpected zero sample reasons: %+v", reasons)
	}
}

func TestBuildDatasetSamplesWithStatsTracksIncompleteFrameHistory(t *testing.T) {
	frames := AttachImagePaths([]VideoFrame{
		{Index: 0, PTS: 0.0},
		{Index: 1, PTS: 0.1},
		{Index: 2, PTS: 0.2},
		{Index: 3, PTS: 0.3},
		{Index: 4, PTS: 0.4},
		{Index: 5, PTS: 0.5},
		{Index: 6, PTS: 0.6},
	}, "frames")
	labels := make([]timedLabel, 0, 12)
	for index := 0; index < 12; index++ {
		labels = append(labels, timedLabel{
			RelativeSeconds: float64(index) * 0.1,
			Label: map[string]any{
				"time":              float64(index) * 100.0,
				"Steering":          0.1,
				"currentSpeed":      1.0 + float64(index),
				"acceleration":      0.1,
				"yaw":               10.0 + float64(index),
				"yawRate":           0.5,
				"routeForwardDelta": 0.1,
			},
		})
	}

	samples, stats := buildDatasetSamplesWithStats(frames, labels, 0.0, 3, 2, 10, 100*time.Millisecond, 2.0, true)
	if len(samples) != 0 {
		t.Fatalf("expected no samples, got=%d", len(samples))
	}
	if stats.IncompleteFrameHistoryCount != 1 {
		t.Fatalf("unexpected incomplete frame-history count: %+v", stats)
	}
	if reasons := stats.zeroSampleReasons(); reasons["incomplete_frame_history"] != 1 {
		t.Fatalf("unexpected zero sample reasons: %+v", reasons)
	}
}

func TestBuildDatasetSamplesWithStatsTracksIncompleteTelemetryFuture(t *testing.T) {
	rawFrames := make([]VideoFrame, 0, 13)
	labels := make([]timedLabel, 0, 9)
	for index := 0; index < 13; index++ {
		rawFrames = append(rawFrames, VideoFrame{Index: index, PTS: float64(index) * 0.1})
		if index < 9 {
			labels = append(labels, timedLabel{
				RelativeSeconds: float64(index) * 0.1,
				Label: map[string]any{
					"time":              float64(index) * 100.0,
					"Steering":          0.1,
					"currentSpeed":      2.0 + float64(index),
					"acceleration":      0.1,
					"yaw":               10.0 + float64(index),
					"yawRate":           0.5,
					"routeForwardDelta": 0.1,
				},
			})
		}
	}
	frames := AttachImagePaths(rawFrames, "frames")

	samples, stats := buildDatasetSamplesWithStats(frames, labels, 0.0, 3, 2, 8, 100*time.Millisecond, 2.0, true)
	if len(samples) != 0 {
		t.Fatalf("expected no samples, got=%d", len(samples))
	}
	if stats.IncompleteTelemetryFutureCount != 1 {
		t.Fatalf("unexpected incomplete telemetry-future count: %+v", stats)
	}
	if reasons := stats.zeroSampleReasons(); reasons["incomplete_telemetry_future"] != 1 {
		t.Fatalf("unexpected zero sample reasons: %+v", reasons)
	}
}

func TestBuildTrainingLabelDropsAccelerationAndAddsFutureTargets(t *testing.T) {
	current := map[string]any{
		"time":              400.0,
		"Steering":          0.1,
		"currentSpeed":      4.0,
		"acceleration":      0.25,
		"yaw":               12.0,
		"yawRate":           0.5,
		"routeForwardDelta": 0.25,
	}
	future := map[string]any{
		"time":         1400.0,
		"Steering":     0.2,
		"currentSpeed": 6.5,
		"acceleration": 0.5,
		"yaw":          18.0,
		"yawRate":      1.25,
	}

	derived, ok := buildTrainingLabel(current, future, 5.25, 18.0, 1.25, 0.75, 1.0, 2.0, true)
	if !ok {
		t.Fatal("expected training label to be derived")
	}
	flatDerived := flattenedLabel(derived)
	if flatDerived["acceleration"] != nil {
		t.Fatalf("expected acceleration to be removed, label=%+v", flatDerived)
	}
	if flatDerived["currentSpeed"] != nil {
		t.Fatalf("expected currentSpeed to be omitted from derived label: %+v", flatDerived)
	}
	if flatDerived["future_speed_delta"] != 1.25 {
		t.Fatalf("unexpected future_speed_delta in derived label: %+v", flatDerived)
	}
	if flatDerived["future_speed_delta_target"] != 0.625 {
		t.Fatalf("unexpected future_speed_delta_target in derived label: %+v", flatDerived)
	}
	if flatDerived["future_speed"] != 6.5 {
		t.Fatalf("unexpected future_speed in derived label: %+v", flatDerived)
	}
	if flatDerived["future_speed_target"] != 5.25 {
		t.Fatalf("unexpected future_speed_target in derived label: %+v", flatDerived)
	}
	if flatDerived["routeForwardDelta"] != 0.75 {
		t.Fatalf("unexpected routeForwardDelta in derived label: %+v", flatDerived)
	}
	if flatDerived["future_yaw_delta"] != 6.0 {
		t.Fatalf("unexpected future_yaw_delta in derived label: %+v", flatDerived)
	}
	if flatDerived["future_horizon_seconds"] != 1.0 {
		t.Fatalf("unexpected future_horizon_seconds in derived label: %+v", flatDerived)
	}
	if flatDerived["yaw_rate"] != 1.25 {
		t.Fatalf("unexpected yaw_rate in derived label: %+v", flatDerived)
	}
	if derived.Control.Steering == nil || derived.Aux.FutureSpeedDelta == nil {
		t.Fatalf("expected grouped derived label sections: %+v", derived)
	}
}

func TestResolvedRouteForwardDeltaFallsBackToCoordsGpsAndYaw(t *testing.T) {
	value, ok := resolvedRouteForwardDelta(map[string]any{
		"coords": []any{0.0, 0.0, 0.0},
		"gps":    []any{0.0, 5.0, 0.0},
		"yaw":    0.0,
	})
	if !ok {
		t.Fatal("expected derived routeForwardDelta")
	}
	if math.Abs(value-5.0) > 1e-6 {
		t.Fatalf("unexpected derived routeForwardDelta: got=%f want=5.0", value)
	}
}

func TestThinStoppedSamplesKeepsBurstThenSparseSamples(t *testing.T) {
	samples := []DatasetSample{
		stoppedSample(0.0, false, 0.1),
		stoppedSample(1.0, true, 0.2),
		stoppedSample(1.5, 1.0, 0.3),
		stoppedSample(2.0, true, 0.4),
		stoppedSample(2.5, 1.0, 0.5),
		stoppedSample(3.1, true, 0.6),
		stoppedSample(3.2, false, 0.7),
	}

	filtered := thinStoppedSamples(samples, 3, 2.0)
	if len(filtered) != 5 {
		t.Fatalf("unexpected filtered count: got=%d filtered=%+v", len(filtered), filtered)
	}

	gotTimes := make([]float64, 0, len(filtered))
	for _, sample := range filtered {
		gotTimes = append(gotTimes, sample.AnchorGameTime)
	}
	wantTimes := []float64{0.0, 1.0, 1.5, 2.0, 3.2}
	if !reflect.DeepEqual(gotTimes, wantTimes) {
		t.Fatalf("unexpected kept sample times: got=%v want=%v", gotTimes, wantTimes)
	}
}

func TestThinStoppedSamplesKeepsLaterStoppedSampleAfterSpacing(t *testing.T) {
	samples := []DatasetSample{
		stoppedSample(10.0, true, nil),
		stoppedSample(10.5, true, nil),
		stoppedSample(11.0, true, nil),
		stoppedSample(13.1, true, nil),
		stoppedSample(13.2, false, nil),
	}

	filtered := thinStoppedSamples(samples, 3, 2.0)
	if len(filtered) != 5 {
		t.Fatalf("expected later stopped sample after spacing to be kept, filtered=%+v", filtered)
	}
}

func TestProcessTripDatasetOnlyThinsStoppedTail(t *testing.T) {
	tmp := t.TempDir()
	tripDir := filepath.Join(tmp, "trip-000")
	framesDir := filepath.Join(tripDir, "frames")
	if err := os.MkdirAll(framesDir, 0o755); err != nil {
		t.Fatalf("mkdir frames: %v", err)
	}

	for i := 1; i <= 15; i++ {
		brightness := uint8(80)
		if i == 1 {
			brightness = 255
		}
		writeJPEG(t, filepath.Join(framesDir, formatFrameName(i)), brightness)
	}

	metadataBody := `{"runId":"run-a","sceneId":"scene-a","sceneVariant":"default","tripIndex":0,"syncTime":0}`
	if err := os.WriteFile(filepath.Join(tripDir, "metadata.json"), []byte(metadataBody), 0o644); err != nil {
		t.Fatalf("write metadata: %v", err)
	}
	if err := os.WriteFile(filepath.Join(tripDir, "video.mkv"), []byte("stub"), 0o644); err != nil {
		t.Fatalf("write video: %v", err)
	}
	vehicleData := make([]map[string]any, 0, 15)
	stoppedFlags := []any{false, false, false, false, false, false, true, true, true, true, false, false, false, false, false}
	for index := 0; index < 15; index++ {
		vehicleData = append(vehicleData, map[string]any{
			"time":              float64(index) * 100.0,
			"Steering":          0.1 + (0.01 * float64(index)),
			"currentSpeed":      8.0 - (0.3 * float64(index)),
			"isStopped":         stoppedFlags[index],
			"yaw":               10.0 + float64(index),
			"yawRate":           0.1,
			"routeForwardDelta": 0.25 - (0.02 * float64(index)),
		})
	}
	runRecord := runTripRecord{
		RunID:       "run-a",
		TripIndex:   0,
		VehicleData: vehicleData,
	}
	runPayload, err := json.Marshal(runRecord)
	if err != nil {
		t.Fatalf("marshal run record: %v", err)
	}
	if err := os.WriteFile(filepath.Join(tmp, "run.jsonl"), append(runPayload, '\n'), 0o644); err != nil {
		t.Fatalf("write run.jsonl: %v", err)
	}

	processor := NewProcessor(
		WithForce(true),
		WithDatasetOnly(true),
		WithSamplingConfig(3, 2, 2),
		WithCommandFactory(func(_ context.Context, name string, _ ...string) *exec.Cmd {
			if name == "ffprobe" {
				cmd := exec.Command(os.Args[0], "-test.run=TestHelperProcessFFprobeStoppedTail", "--")
				cmd.Env = append(os.Environ(), "GO_WANT_HELPER_PROCESS=ffprobe_stopped_tail")
				return cmd
			}
			return exec.Command("definitely-missing-ffmpeg-command")
		}),
	)

	if err := processor.ProcessTrip(context.Background(), tripDir); err != nil {
		t.Fatalf("ProcessTrip stopped-tail dataset-only: %v", err)
	}

	body, err := os.ReadFile(filepath.Join(tripDir, "dataset.jsonl"))
	if err != nil {
		t.Fatalf("read dataset.jsonl: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(body)), "\n")
	if len(lines) != 3 {
		t.Fatalf("unexpected dataset line count after thinning: got=%d body=%s", len(lines), string(body))
	}

	var samples []DatasetSample
	for _, line := range lines {
		var sample DatasetSample
		if err := json.Unmarshal([]byte(line), &sample); err != nil {
			t.Fatalf("parse dataset sample: %v", err)
		}
		samples = append(samples, sample)
	}

	gotStopped := []any{
		sampleCurrentTelemetryValue(samples[0], "isStopped"),
		sampleCurrentTelemetryValue(samples[1], "isStopped"),
		sampleCurrentTelemetryValue(samples[2], "isStopped"),
	}
	wantStopped := []any{false, true, true}
	if !reflect.DeepEqual(gotStopped, wantStopped) {
		t.Fatalf("unexpected stopped labels after thinning: got=%v want=%v", gotStopped, wantStopped)
	}
}

func TestProcessTripDatasetOnlyRewritesDatasetWithoutFFmpeg(t *testing.T) {
	tmp := t.TempDir()
	tripDir := filepath.Join(tmp, "trip-000")
	framesDir := filepath.Join(tripDir, "frames")
	if err := os.MkdirAll(framesDir, 0o755); err != nil {
		t.Fatalf("mkdir frames: %v", err)
	}

	for index := 1; index <= 13; index++ {
		brightness := uint8(80)
		if index == 1 {
			brightness = 255
		}
		writeJPEG(t, filepath.Join(framesDir, fmt.Sprintf("%06d.jpg", index)), brightness)
	}

	metadataBody := `{"runId":"run-a","sceneId":"scene-a","sceneVariant":"default","tripIndex":0,"syncTime":0}`
	if err := os.WriteFile(filepath.Join(tripDir, "metadata.json"), []byte(metadataBody), 0o644); err != nil {
		t.Fatalf("write metadata: %v", err)
	}
	if err := os.WriteFile(filepath.Join(tripDir, "video.mkv"), []byte("stub"), 0o644); err != nil {
		t.Fatalf("write video: %v", err)
	}
	vehicleData := make([]map[string]any, 0, 12)
	for index := 0; index < 12; index++ {
		vehicleData = append(vehicleData, map[string]any{
			"time":              float64(index) * 100.0,
			"Steering":          float64(index) * 0.1,
			"currentSpeed":      float64(index),
			"acceleration":      float64(index) * 0.05,
			"yaw":               10.0 + (float64(index) * 2.0),
			"yawRate":           float64(index) * 0.5,
			"routeForwardDelta": float64(index) * 0.1,
		})
	}
	runRecord := runTripRecord{
		RunID:       "run-a",
		TripIndex:   0,
		VehicleData: vehicleData,
	}
	runPayload, err := json.Marshal(runRecord)
	if err != nil {
		t.Fatalf("marshal run record: %v", err)
	}
	if err := os.WriteFile(filepath.Join(tmp, "run.jsonl"), append(runPayload, '\n'), 0o644); err != nil {
		t.Fatalf("write run.jsonl: %v", err)
	}
	if err := os.WriteFile(filepath.Join(tripDir, "dataset.jsonl"), []byte("{\"stale\":true}\n"), 0o644); err != nil {
		t.Fatalf("write stale dataset: %v", err)
	}

	processor := NewProcessor(
		WithForce(true),
		WithDatasetOnly(true),
		WithSamplingConfig(3, 2, 2),
		WithCommandFactory(func(_ context.Context, name string, _ ...string) *exec.Cmd {
			if name == "ffprobe" {
				cmd := exec.Command(os.Args[0], "-test.run=TestHelperProcessFFprobeDatasetOnly", "--")
				cmd.Env = append(os.Environ(), "GO_WANT_HELPER_PROCESS=ffprobe_dataset_only")
				return cmd
			}
			return exec.Command("definitely-missing-ffmpeg-command")
		}),
	)

	if err := processor.ProcessTrip(context.Background(), tripDir); err != nil {
		t.Fatalf("ProcessTrip dataset-only: %v", err)
	}

	body, err := os.ReadFile(filepath.Join(tripDir, "dataset.jsonl"))
	if err != nil {
		t.Fatalf("read dataset.jsonl: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(body)), "\n")
	if len(lines) != 1 {
		t.Fatalf("unexpected dataset line count: got=%d body=%s", len(lines), string(body))
	}
	var sample DatasetSample
	if err := json.Unmarshal([]byte(lines[0]), &sample); err != nil {
		t.Fatalf("parse dataset sample: %v", err)
	}
	flatLabel := flattenedLabel(sample.Label)
	if flatLabel["future_speed_delta"] != 2.0 {
		t.Fatalf("unexpected future_speed_delta in rewritten dataset: %+v", flatLabel)
	}
	if flatLabel["future_speed_delta_target"] != 1.0 {
		t.Fatalf("unexpected future_speed_delta_target in rewritten dataset: %+v", flatLabel)
	}
	if flatLabel["future_speed"] != 6.0 {
		t.Fatalf("unexpected future_speed in rewritten dataset: %+v", flatLabel)
	}
	if flatLabel["future_speed_target"] != 6.0 {
		t.Fatalf("unexpected future_speed_target in rewritten dataset: %+v", flatLabel)
	}
	if math.Abs(flatLabel["future_yaw_delta"].(float64)-4.0) > 1e-6 {
		t.Fatalf("unexpected future_yaw_delta in rewritten dataset: %+v", flatLabel)
	}
	if math.Abs(flatLabel["future_horizon_seconds"].(float64)-0.2) > 1e-6 {
		t.Fatalf("unexpected future_horizon_seconds in rewritten dataset: %+v", flatLabel)
	}
	if flatLabel["acceleration"] != nil {
		t.Fatalf("expected rewritten dataset to omit acceleration: %+v", flatLabel)
	}
	if flatLabel["currentSpeed"] != nil {
		t.Fatalf("expected rewritten dataset to omit redundant currentSpeed: %+v", flatLabel)
	}
	if flatLabel["isStopped"] != nil {
		t.Fatalf("expected rewritten dataset to omit redundant isStopped: %+v", flatLabel)
	}
	if len(sample.TelemetryHistory) != 5 {
		t.Fatalf("expected telemetry_history to be serialized with 5 entries: %+v", sample)
	}
	if sample.TelemetryHistory[0].Control.Acceleration != 0.0 {
		t.Fatalf("expected serialized telemetry history to keep acceleration in control: %+v", sample.TelemetryHistory[0])
	}
	if len(sample.TelemetryFuture) != defaultFutureTelemetryCount {
		t.Fatalf("expected telemetry_future to be serialized with 6 entries: %+v", sample)
	}
	if sample.TelemetryFuture[0].Control.Acceleration != 0.25 {
		t.Fatalf("expected serialized telemetry future to keep acceleration in control: %+v", sample.TelemetryFuture[0])
	}
	if sample.TelemetryHistory[0].Aux.CurrentSpeed != 0.0 {
		t.Fatalf("expected serialized telemetry history to keep aux fields together: %+v", sample.TelemetryHistory[0])
	}
	if strings.Index(lines[0], "\"control\"") == -1 || strings.Index(lines[0], "\"aux\"") == -1 || strings.Index(lines[0], "\"raw\"") == -1 {
		t.Fatalf("expected grouped sections in serialized JSON: %s", lines[0])
	}
	if strings.Index(lines[0], "\"control\"") > strings.Index(lines[0], "\"aux\"") || strings.Index(lines[0], "\"aux\"") > strings.Index(lines[0], "\"raw\"") {
		t.Fatalf("expected control, aux, raw ordering in serialized JSON: %s", lines[0])
	}
}

func TestHelperProcessFFprobe(t *testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") != "ffprobe" {
		return
	}
	_, _ = os.Stdout.Write([]byte(`{"frames":[{"pts_time":"0.0"},{"pts_time":"0.1"}]}`))
	os.Exit(0)
}

func TestHelperProcessFFmpeg(t *testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") != "ffmpeg" {
		return
	}

	args := helperArgsAfterDoubleDash(os.Args)
	scaleWidth, scaleHeight, err := helperScaleArgs(args)
	if err != nil {
		_, _ = os.Stderr.WriteString(err.Error())
		os.Exit(1)
	}
	outputPattern := args[len(args)-1]
	outputPath := strings.Replace(outputPattern, "%06d", "000001", 1)
	if err := writeJPEGFile(outputPath, scaleWidth, scaleHeight, 200); err != nil {
		_, _ = os.Stderr.WriteString(err.Error())
		os.Exit(1)
	}
	os.Exit(0)
}

func TestHelperProcessFFprobeDatasetOnly(t *testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") != "ffprobe_dataset_only" {
		return
	}
	_, _ = os.Stdout.Write([]byte(`{"frames":[{"pts_time":"0.0"},{"pts_time":"0.1"},{"pts_time":"0.2"},{"pts_time":"0.3"},{"pts_time":"0.4"},{"pts_time":"0.5"},{"pts_time":"0.6"},{"pts_time":"0.7"},{"pts_time":"0.8"},{"pts_time":"0.9"},{"pts_time":"1.0"},{"pts_time":"1.1"},{"pts_time":"1.2"}]}`))
	os.Exit(0)
}

func TestHelperProcessFFprobeStoppedTail(t *testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") != "ffprobe_stopped_tail" {
		return
	}
	_, _ = os.Stdout.Write([]byte(`{"frames":[{"pts_time":"0.0"},{"pts_time":"0.1"},{"pts_time":"0.2"},{"pts_time":"0.3"},{"pts_time":"0.4"},{"pts_time":"0.5"},{"pts_time":"0.6"},{"pts_time":"0.7"},{"pts_time":"0.8"},{"pts_time":"0.9"},{"pts_time":"1.0"},{"pts_time":"1.1"},{"pts_time":"1.2"},{"pts_time":"1.3"},{"pts_time":"1.4"}]}`))
	os.Exit(0)
}

func writeJPEG(t *testing.T, path string, brightness uint8) {
	t.Helper()
	if err := writeJPEGFile(path, 8, 8, brightness); err != nil {
		t.Fatalf("encode jpeg: %v", err)
	}
}

func writeJPEGFile(path string, width int, height int, brightness uint8) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	img := image.NewRGBA(image.Rect(0, 0, width, height))
	fill := color.RGBA{R: brightness, G: brightness, B: brightness, A: 255}
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			img.Set(x, y, fill)
		}
	}
	return jpeg.Encode(file, img, nil)
}

func helperArgsAfterDoubleDash(args []string) []string {
	for i, arg := range args {
		if arg == "--" {
			return args[i+1:]
		}
	}
	return nil
}

func sampleTelemetryWindow(t *testing.T, window []GroupedTelemetryItem, field string) []map[string]any {
	t.Helper()
	if window == nil {
		t.Fatalf("expected %s to be populated", field)
	}
	flattened := make([]map[string]any, 0, len(window))
	for _, item := range window {
		flattened = append(flattened, flattenGroupedTelemetry(item))
	}
	return flattened
}

func flattenedLabel(label GroupedLabel) map[string]any {
	return flattenGroupedLabel(label)
}

func stoppedSample(anchorGameTime float64, isStopped any, steering any) DatasetSample {
	sample := DatasetSample{
		AnchorGameTime: anchorGameTime,
		TelemetryHistory: []GroupedTelemetryItem{
			{
				Aux: GroupedTelemetryAux{IsStopped: isStopped},
			},
		},
	}
	if steering != nil {
		sample.Label.Control.Steering = steering
	}
	return sample
}

func helperScaleArgs(args []string) (int, int, error) {
	for i := 0; i < len(args)-1; i++ {
		if args[i] != "-vf" {
			continue
		}
		filter := strings.TrimSpace(args[i+1])
		if !strings.HasPrefix(filter, "scale=") {
			return 0, 0, fmt.Errorf("unexpected ffmpeg filter: %s", filter)
		}
		sizeParts := strings.Split(strings.TrimPrefix(filter, "scale="), ":")
		if len(sizeParts) != 2 {
			return 0, 0, fmt.Errorf("unexpected scale filter format: %s", filter)
		}
		width, err := strconv.Atoi(sizeParts[0])
		if err != nil {
			return 0, 0, fmt.Errorf("invalid scale width: %w", err)
		}
		height, err := strconv.Atoi(sizeParts[1])
		if err != nil {
			return 0, 0, fmt.Errorf("invalid scale height: %w", err)
		}
		return width, height, nil
	}
	return 0, 0, fmt.Errorf("missing -vf scale filter")
}

func formatFrameName(index int) string {
	return fmt.Sprintf("%06d.jpg", index)
}
