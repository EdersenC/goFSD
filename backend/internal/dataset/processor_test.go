package dataset

import (
	"context"
	"encoding/json"
	"errors"
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
	"sync"
	"testing"
	"time"
)

func TestAttachImagePaths(t *testing.T) {
	frames := []VideoFrame{{Index: 0, PTS: 0.0}, {Index: 2, PTS: 0.1}, {Index: 3, PTS: 0.2}}
	got := AttachImagePaths(frames, "frames")
	want := []string{"frames/000001.jpg", "frames/000003.jpg", "frames/000004.jpg"}
	for i := range want {
		if got[i].ImagePath != want[i] {
			t.Fatalf("unexpected image path at %d: got=%q want=%q", i, got[i].ImagePath, want[i])
		}
	}
}

func TestBuildFrameWindowAtOffsetsRejectsMissingSourceFrame(t *testing.T) {
	frames := AttachImagePaths([]VideoFrame{
		{Index: 0},
		{Index: 2},
		{Index: 3},
	}, "frames")
	if window, ok := buildFrameWindowAtOffsets(frames, 2, []int{-2, 0}); ok || window != nil {
		t.Fatalf("missing source frame must not compact the timeline: %v", window)
	}
}

func TestBuildFrameWindowAtOffsetsUsesExactNonUniformOffsets(t *testing.T) {
	frames := make([]VideoFrame, 11)
	for index := range frames {
		frames[index] = VideoFrame{Index: index, ImagePath: fmt.Sprintf("frames/%06d.jpg", index+1)}
	}

	window, ok := buildFrameWindowAtOffsets(frames, 10, []int{-10, -7, -3, -1, 0})
	if !ok {
		t.Fatal("expected a complete nonuniform frame window")
	}
	want := []string{
		"frames/000001.jpg",
		"frames/000004.jpg",
		"frames/000008.jpg",
		"frames/000010.jpg",
		"frames/000011.jpg",
	}
	if !reflect.DeepEqual(window, want) {
		t.Fatalf("unexpected nonuniform frame window: got=%v want=%v", window, want)
	}
}

func TestProcessorFingerprintUsesImageOffsetsAndIgnoresExecutionMode(t *testing.T) {
	offsets := []int{-8, -5, -3, -1, 0}
	full := NewProcessor(WithImageOffsets(offsets))
	datasetOnly := NewProcessor(WithImageOffsets(offsets), WithDatasetOnly(true))
	if full.ConfigFingerprint() != datasetOnly.ConfigFingerprint() {
		t.Fatal("dataset-only execution must not invalidate equivalent published outputs")
	}

	differentTimeline := NewProcessor(WithImageOffsets([]int{-8, -6, -4, -2, 0}))
	if full.ConfigFingerprint() == differentTimeline.ConfigFingerprint() {
		t.Fatal("different image offsets must invalidate prior outputs")
	}
	offsets[0] = -99
	if !reflect.DeepEqual(full.imageOffsets, []int{-8, -5, -3, -1, 0}) {
		t.Fatal("WithImageOffsets must defensively copy caller-owned configuration")
	}
	status := full.newProcessingStatus("queued")
	full.imageOffsets[0] = -77
	full.telemetryOffsets[0] = -77
	full.futureOffsets[0] = 77
	if !reflect.DeepEqual(status.ImageOffsets, []int{-8, -5, -3, -1, 0}) ||
		status.TelemetryOffsets[0] == -77 ||
		status.FutureOffsets[0] == 77 {
		t.Fatalf("processing status must own defensive timeline copies: %+v", status)
	}
}

func TestWriteStatusFileNeverExposesPartialJSONToConcurrentReaders(t *testing.T) {
	path := filepath.Join(t.TempDir(), "processing.json")
	if err := writeStatusFile(path, ProcessingStatus{State: "queued"}); err != nil {
		t.Fatalf("write initial status: %v", err)
	}
	done := make(chan struct{})
	readErrors := make(chan error, 1)
	var readers sync.WaitGroup
	for reader := 0; reader < 4; reader++ {
		readers.Add(1)
		go func() {
			defer readers.Done()
			for {
				select {
				case <-done:
					return
				default:
				}
				status, err := ReadStatusFile(path)
				if err != nil || (status.State != "queued" && status.State != "running") {
					select {
					case readErrors <- fmt.Errorf("read atomic status: state=%q err=%v", status.State, err):
					default:
					}
					return
				}
			}
		}()
	}
	for index := 0; index < 64; index++ {
		status := ProcessingStatus{State: "running", FrameCount: index, SampleCount: index / 2}
		if err := writeStatusFile(path, status); err != nil {
			close(done)
			readers.Wait()
			t.Fatalf("write status %d: %v", index, err)
		}
	}
	close(done)
	readers.Wait()
	select {
	case err := <-readErrors:
		t.Fatal(err)
	default:
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

	samples := buildDatasetSamples(frames, labels, 0.0, 3, 2, 4, 100*time.Millisecond, testTelemetryOffsets(3, 2), defaultFutureOffsets(), 100*time.Millisecond)
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
	if flatLabel["future_speed"] != 8.0 {
		t.Fatalf("unexpected future_speed: %+v", flatLabel)
	}
	if flatLabel["future_speed_target"] != 8.0 {
		t.Fatalf("unexpected future_speed_target: %+v", flatLabel)
	}
	if math.Abs(flatLabel["future_horizon_seconds"].(float64)-0.4) > 1e-6 {
		t.Fatalf("unexpected future_horizon_seconds: %+v", flatLabel)
	}
	if flatLabel["routeForwardDelta"] != 0.4 {
		t.Fatalf("unexpected routeForwardDelta: %+v", flatLabel)
	}
	if samples[0].Label.Control.Steering == nil || samples[0].Label.Aux.FutureSpeedTarget == nil {
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

func TestBuildTelemetryFutureUsesTimestampSlotsInsteadOfAdjacentRows(t *testing.T) {
	labels := []timedLabel{
		{RelativeSeconds: 0.400, Label: map[string]any{"time": 400.0}},
		{RelativeSeconds: 0.421, Label: map[string]any{"time": 421.0}},
		{RelativeSeconds: 0.449, Label: map[string]any{"time": 449.0}},
		{RelativeSeconds: 0.503, Label: map[string]any{"time": 503.0}},
		{RelativeSeconds: 0.548, Label: map[string]any{"time": 548.0}},
		{RelativeSeconds: 0.601, Label: map[string]any{"time": 601.0}},
		{RelativeSeconds: 0.652, Label: map[string]any{"time": 652.0}},
		{RelativeSeconds: 0.698, Label: map[string]any{"time": 698.0}},
	}

	future, ok := buildTelemetryFuture(
		labels,
		0,
		defaultFutureOffsets(),
		50*time.Millisecond,
		25*time.Millisecond,
	)
	if !ok {
		t.Fatal("expected a complete timestamp-aligned future horizon")
	}
	got := sampleTelemetryWindow(t, future, "telemetry_future")
	wantTimes := []float64{449, 503, 548, 601, 652, 698}
	for index, want := range wantTimes {
		if got[index]["time"] != want {
			t.Fatalf("unexpected horizon slot %d: got=%v want_time=%v", index, got[index], want)
		}
	}
	if got[0]["time"] == 421.0 {
		t.Fatalf("future horizon used the adjacent row instead of the +50ms slot: %+v", got)
	}
}

func TestBuildTelemetryFutureRejectsMissingTimestampSlot(t *testing.T) {
	labels := []timedLabel{
		{RelativeSeconds: 0.400, Label: map[string]any{"time": 400.0}},
		{RelativeSeconds: 0.500, Label: map[string]any{"time": 500.0}},
		{RelativeSeconds: 0.550, Label: map[string]any{"time": 550.0}},
		{RelativeSeconds: 0.600, Label: map[string]any{"time": 600.0}},
		{RelativeSeconds: 0.650, Label: map[string]any{"time": 650.0}},
		{RelativeSeconds: 0.700, Label: map[string]any{"time": 700.0}},
	}

	if future, ok := buildTelemetryFuture(
		labels,
		0,
		defaultFutureOffsets(),
		50*time.Millisecond,
		25*time.Millisecond,
	); ok || future != nil {
		t.Fatalf("expected the missing +50ms slot to reject the horizon, got=%+v", future)
	}
}

func TestBuildTelemetryFutureKeepsDenseRowsForConfiguredOffsets(t *testing.T) {
	labels := make([]timedLabel, 0, 7)
	for offset := 0; offset <= 6; offset++ {
		labels = append(labels, timedLabel{
			RelativeSeconds: float64(offset) * 0.05,
			Label:           map[string]any{"time": float64(offset * 50)},
		})
	}

	future, ok := buildTelemetryFuture(
		labels,
		0,
		[]int{1, 3, 6},
		50*time.Millisecond,
		25*time.Millisecond,
	)
	if !ok {
		t.Fatal("expected a complete dense future horizon")
	}
	if len(future) != 6 {
		t.Fatalf("future rows must remain indexable by configured offsets: got=%d want=6", len(future))
	}
}

func TestBuildTelemetryHistoryUsesTimestampSlotsInsteadOfAdjacentRows(t *testing.T) {
	labels := []timedLabel{
		{RelativeSeconds: 0.249, Label: map[string]any{"time": 249.0}},
		{RelativeSeconds: 0.301, Label: map[string]any{"time": 301.0}},
		{RelativeSeconds: 0.329, Label: map[string]any{"time": 329.0}},
		{RelativeSeconds: 0.351, Label: map[string]any{"time": 351.0}},
		{RelativeSeconds: 0.400, Label: map[string]any{"time": 400.0}},
	}

	history, ok := buildTelemetryHistory(
		labels,
		4,
		[]int{-3, -2, -1, 0},
		50*time.Millisecond,
		25*time.Millisecond,
	)
	if !ok {
		t.Fatal("expected a complete timestamp-aligned telemetry history")
	}
	got := sampleTelemetryWindow(t, history, "telemetry_history")
	wantTimes := []float64{249, 301, 351, 400}
	for index, want := range wantTimes {
		if got[index]["time"] != want {
			t.Fatalf("unexpected history slot %d: got=%v want_time=%v", index, got[index], want)
		}
	}
	if got[2]["time"] == 329.0 {
		t.Fatalf("telemetry history used the adjacent row instead of the -50ms slot: %+v", got)
	}
}

func TestBuildTelemetryHistoryRejectsMissingTimestampSlot(t *testing.T) {
	labels := []timedLabel{
		{RelativeSeconds: 0.250, Label: map[string]any{"time": 250.0}},
		{RelativeSeconds: 0.300, Label: map[string]any{"time": 300.0}},
		{RelativeSeconds: 0.400, Label: map[string]any{"time": 400.0}},
	}

	if history, ok := buildTelemetryHistory(
		labels,
		2,
		[]int{-3, -2, -1, 0},
		50*time.Millisecond,
		25*time.Millisecond,
	); ok || history != nil {
		t.Fatalf("expected the missing -50ms slot to reject the history, got=%+v", history)
	}
}

func TestBuildTelemetryHistoryKeepsDenseRowsForConfiguredOffsets(t *testing.T) {
	labels := make([]timedLabel, 0, 5)
	for offset := 0; offset <= 4; offset++ {
		labels = append(labels, timedLabel{
			RelativeSeconds: float64(offset) * 0.05,
			Label:           map[string]any{"time": float64(offset * 50)},
		})
	}

	history, ok := buildTelemetryHistory(
		labels,
		4,
		[]int{-4, -2, 0},
		50*time.Millisecond,
		25*time.Millisecond,
	)
	if !ok {
		t.Fatal("expected a complete dense telemetry history")
	}
	if len(history) != 5 {
		t.Fatalf("history rows must remain indexable by configured offsets: got=%d want=5", len(history))
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

	samples := buildDatasetSamples(frames, labels, 0.0, 5, 2, 8, 100*time.Millisecond, testTelemetryOffsets(5, 2), defaultFutureOffsets(), 100*time.Millisecond)
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

	samples := buildDatasetSamples(frames, labels, 0.0, 5, 2, 10, 100*time.Millisecond, testTelemetryOffsets(5, 2), defaultFutureOffsets(), 100*time.Millisecond)
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

	samples, stats := buildDatasetSamplesWithStats(frames, labels, 0.0, 3, 2, 4, 100*time.Millisecond, testTelemetryOffsets(3, 2), defaultFutureOffsets(), 100*time.Millisecond)
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

func TestQueueWritesConfiguredInterpretationMetadataToStatus(t *testing.T) {
	tripDir := t.TempDir()
	imageOffsets := []int{-8, -5, -3, -1, 0}
	telemetryOffsets := []int{-4, -2, 0}
	futureOffsets := []int{1, 3, 6}
	processor := NewProcessor(
		WithImageSize(320, 180),
		WithImageOffsets(imageOffsets),
		WithTelemetryTimelineConfig(telemetryOffsets, futureOffsets, 75*time.Millisecond),
	)

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
	if !reflect.DeepEqual(status.ImageOffsets, imageOffsets) ||
		!reflect.DeepEqual(status.TelemetryOffsets, telemetryOffsets) ||
		!reflect.DeepEqual(status.FutureOffsets, futureOffsets) ||
		status.TelemetrySampleIntervalMs != 75 {
		t.Fatalf("unexpected status timeline interpretation: %+v", status)
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

func TestProcessorSkipRequiresCompleteOutputsAndMatchingFingerprint(t *testing.T) {
	tripDir := t.TempDir()
	processor := NewProcessor()

	if processor.shouldSkipTrip(tripDir) {
		t.Fatal("empty trip must not be treated as complete")
	}
	if err := os.WriteFile(filepath.Join(tripDir, "dataset.jsonl"), []byte("{}\n"), 0o644); err != nil {
		t.Fatalf("write partial dataset: %v", err)
	}
	if processor.shouldSkipTrip(tripDir) {
		t.Fatal("dataset without frames and status must not be treated as complete")
	}
	framesDir := filepath.Join(tripDir, "frames")
	if err := os.MkdirAll(framesDir, 0o755); err != nil {
		t.Fatalf("mkdir frames: %v", err)
	}
	if err := os.WriteFile(filepath.Join(framesDir, "000001.jpg"), []byte("partial"), 0o644); err != nil {
		t.Fatalf("write partial frame: %v", err)
	}
	if processor.shouldSkipTrip(tripDir) {
		t.Fatal("partial outputs without a completed status must not be treated as complete")
	}

	writeCompletedProcessingFixture(t, tripDir, processor, 1, 1)
	if !processor.shouldSkipTrip(tripDir) {
		t.Fatal("complete outputs with a matching fingerprint should be skipped")
	}
	if !processor.TripOutputsCurrent(tripDir) {
		t.Fatal("public readiness check disagrees with processor skip rule")
	}
	status, err := ReadStatusFile(filepath.Join(tripDir, "processing.json"))
	if err != nil {
		t.Fatalf("read complete status: %v", err)
	}
	status.ImageOffsets = nil
	if err := writeStatusFile(filepath.Join(tripDir, "processing.json"), status); err != nil {
		t.Fatalf("write opaque complete status: %v", err)
	}
	if processor.shouldSkipTrip(tripDir) {
		t.Fatal("same-fingerprint status without explicit timeline metadata must not be skipped")
	}
	writeCompletedProcessingFixture(t, tripDir, processor, 1, 1)
	status, err = ReadStatusFile(filepath.Join(tripDir, "processing.json"))
	if err != nil {
		t.Fatalf("read status before dimension drift: %v", err)
	}
	status.ImageWidth = 0
	if err := writeStatusFile(filepath.Join(tripDir, "processing.json"), status); err != nil {
		t.Fatalf("write status without explicit dimensions: %v", err)
	}
	if processor.TripOutputsCurrent(tripDir) {
		t.Fatal("status without explicit image dimensions must not be current")
	}
	writeCompletedProcessingFixture(t, tripDir, processor, 1, 1)

	changedProcessor := NewProcessor(WithImageSize(320, 180))
	if changedProcessor.shouldSkipTrip(tripDir) {
		t.Fatal("changed processing config must invalidate prior outputs")
	}
}

func TestTripOutputsCurrentRequiresExplicitCountsAndCompletePublishedFiles(t *testing.T) {
	processor := NewProcessor()
	tests := []struct {
		name   string
		mutate func(t *testing.T, tripDir string)
	}{
		{
			name: "missing sample count",
			mutate: func(t *testing.T, tripDir string) {
				statusPath := filepath.Join(tripDir, "processing.json")
				body, err := os.ReadFile(statusPath)
				if err != nil {
					t.Fatalf("read status: %v", err)
				}
				var status map[string]any
				if err := json.Unmarshal(body, &status); err != nil {
					t.Fatalf("decode status: %v", err)
				}
				delete(status, "sampleCount")
				writeJSONFile(t, statusPath, status)
			},
		},
		{
			name: "truncated dataset",
			mutate: func(t *testing.T, tripDir string) {
				if err := os.WriteFile(filepath.Join(tripDir, "dataset.jsonl"), nil, 0o644); err != nil {
					t.Fatalf("truncate dataset: %v", err)
				}
			},
		},
		{
			name: "invalid dataset suffix",
			mutate: func(t *testing.T, tripDir string) {
				file, err := os.OpenFile(filepath.Join(tripDir, "dataset.jsonl"), os.O_APPEND|os.O_WRONLY, 0)
				if err != nil {
					t.Fatalf("open dataset: %v", err)
				}
				if _, err := file.WriteString("not-json\n"); err != nil {
					_ = file.Close()
					t.Fatalf("append invalid dataset row: %v", err)
				}
				if err := file.Close(); err != nil {
					t.Fatalf("close dataset: %v", err)
				}
			},
		},
		{
			name: "gapped frames",
			mutate: func(t *testing.T, tripDir string) {
				if err := os.Rename(
					filepath.Join(tripDir, "frames", "000001.jpg"),
					filepath.Join(tripDir, "frames", "000002.jpg"),
				); err != nil {
					t.Fatalf("gap frames: %v", err)
				}
			},
		},
		{
			name: "zero byte frame",
			mutate: func(t *testing.T, tripDir string) {
				if err := os.WriteFile(filepath.Join(tripDir, "frames", "000001.jpg"), nil, 0o644); err != nil {
					t.Fatalf("truncate frame: %v", err)
				}
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tripDir := t.TempDir()
			writeCompletedProcessingFixture(t, tripDir, processor, 1, 1)
			test.mutate(t, tripDir)
			if processor.TripOutputsCurrent(tripDir) {
				t.Fatal("incomplete published output was reported current")
			}
		})
	}
}

func TestTripOutputsCurrentAcceptsExplicitCompletedZeroSampleOutputs(t *testing.T) {
	processor := NewProcessor()
	tripDir := t.TempDir()
	writeCompletedProcessingFixture(t, tripDir, processor, 0, 0)

	if !processor.TripOutputsCurrent(tripDir) {
		t.Fatal("completed zero-sample output with an explicit empty dataset and frame directory must be current")
	}
}

func TestTripOutputsCurrentAcceptsCompleteInspectionFramesBeyondTrainingReferences(t *testing.T) {
	processor := NewProcessor()
	tripDir := t.TempDir()
	writeCompletedProcessingFixture(t, tripDir, processor, 3, 1)

	datasetPath := filepath.Join(tripDir, "dataset.jsonl")
	if err := os.WriteFile(datasetPath, []byte(`{"frame_paths":["frames/000002.jpg"]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write sparse training references: %v", err)
	}
	if !processor.TripOutputsCurrent(tripDir) {
		t.Fatal("complete stop-sign inspection frames may exceed the subset referenced by training rows")
	}

	if err := os.Remove(filepath.Join(tripDir, "frames", "000003.jpg")); err != nil {
		t.Fatalf("remove unreferenced inspection frame: %v", err)
	}
	if processor.TripOutputsCurrent(tripDir) {
		t.Fatal("a missing inspection frame must still invalidate the published output")
	}
}

func TestValidateFrameSetRejectsTruncationAndDimensionDrift(t *testing.T) {
	tripDir := t.TempDir()
	framesDir := filepath.Join(tripDir, "frames")
	if err := os.MkdirAll(framesDir, 0o755); err != nil {
		t.Fatalf("mkdir frames: %v", err)
	}
	frames := AttachImagePaths([]VideoFrame{{Index: 0}, {Index: 1}}, "frames")
	if err := writeJPEGFile(filepath.Join(framesDir, "000001.jpg"), 8, 8, 100); err != nil {
		t.Fatalf("write first frame: %v", err)
	}
	if err := validateFrameSet(tripDir, frames, 8, 8); err == nil || !strings.Contains(err.Error(), "incomplete") {
		t.Fatalf("expected truncated frame set rejection, got %v", err)
	}

	if err := writeJPEGFile(filepath.Join(framesDir, "000002.jpg"), 4, 4, 100); err != nil {
		t.Fatalf("write second frame: %v", err)
	}
	if err := validateFrameSet(tripDir, frames, 8, 8); err == nil || !strings.Contains(err.Error(), "dimensions") {
		t.Fatalf("expected frame dimension rejection, got %v", err)
	}
}

func TestProcessTripFailureDoesNotPublishStagedOutputs(t *testing.T) {
	tmp := t.TempDir()
	tripDir := filepath.Join(tmp, "trip-000")
	if err := os.MkdirAll(tripDir, 0o755); err != nil {
		t.Fatalf("mkdir trip: %v", err)
	}
	if err := os.WriteFile(filepath.Join(tripDir, "video.mkv"), []byte("stub"), 0o644); err != nil {
		t.Fatalf("write video: %v", err)
	}
	writeJSONFile(t, filepath.Join(tripDir, "metadata.json"), tripMetadata{RunID: "run-a", TripIndex: 0})
	writeJSONLinesFile(t, filepath.Join(tmp, "run.jsonl"), []runTripRecord{{RunID: "run-a", TripIndex: 0}})

	processor := NewProcessor(
		WithCommandFactory(func(_ context.Context, name string, args ...string) *exec.Cmd {
			if name == "ffprobe" {
				cmd := exec.Command(os.Args[0], "-test.run=TestHelperProcessFFprobe", "--")
				cmd.Env = append(os.Environ(), "GO_WANT_HELPER_PROCESS=ffprobe")
				return cmd
			}
			cmdArgs := append([]string{"-test.run=TestHelperProcessFFmpeg", "--"}, args...)
			cmd := exec.Command(os.Args[0], cmdArgs...)
			cmd.Env = append(os.Environ(), "GO_WANT_HELPER_PROCESS=ffmpeg")
			return cmd
		}),
	)

	err := processor.ProcessTrip(context.Background(), tripDir)
	if err == nil {
		t.Fatalf("expected staged processing to fail before publish, got %v", err)
	}
	if pathExists(filepath.Join(tripDir, "frames")) {
		t.Fatal("failed processing must not publish staged frames")
	}
	if pathExists(filepath.Join(tripDir, "dataset.jsonl")) {
		t.Fatal("failed processing must not publish a staged dataset")
	}
	workspaces, globErr := filepath.Glob(filepath.Join(tripDir, processingWorkspacePrefix+"*"))
	if globErr != nil {
		t.Fatalf("glob processing workspaces: %v", globErr)
	}
	if len(workspaces) != 0 {
		t.Fatalf("failed processing workspace was not cleaned up: %v", workspaces)
	}
	status, statusErr := ReadStatusFile(filepath.Join(tripDir, "processing.json"))
	if statusErr != nil {
		t.Fatalf("read failed status: %v", statusErr)
	}
	if status.State != "failed" || status.ConfigFingerprint != processor.ConfigFingerprint() {
		t.Fatalf("unexpected failed status: %+v", status)
	}
}

func TestPromotionPrevalidatesEveryStagedOutputBeforeBackup(t *testing.T) {
	tripDir := t.TempDir()
	workspace := processingWorkspaceForRoot(filepath.Join(tripDir, processingWorkspacePrefix+"prevalidate"))
	if err := os.MkdirAll(workspace.framesDir, 0o755); err != nil {
		t.Fatalf("mkdir staged frames: %v", err)
	}
	writeGenerationFixture(t, filepath.Join(tripDir, "frames"), "old")
	if err := os.WriteFile(filepath.Join(tripDir, "dataset.jsonl"), []byte("old\n"), 0o644); err != nil {
		t.Fatalf("write old dataset: %v", err)
	}

	err := promoteProcessingWorkspace(tripDir, workspace, true)
	if err == nil || !strings.Contains(err.Error(), "staged processing output is missing") {
		t.Fatalf("expected missing staged dataset failure, got %v", err)
	}
	assertGenerationFixture(t, filepath.Join(tripDir, "frames"), "old")
	assertFileContent(t, filepath.Join(tripDir, "dataset.jsonl"), "old\n")
	if pathExists(filepath.Join(workspace.root, "previous-frames")) {
		t.Fatal("prevalidation failure must not move an existing final into the workspace")
	}
}

func TestPromotionCrashPhasesRollbackToOneCompleteGeneration(t *testing.T) {
	for successfulRenames := 0; successfulRenames <= 4; successfulRenames++ {
		t.Run(fmt.Sprintf("after-%d-renames", successfulRenames), func(t *testing.T) {
			tripDir, workspace := createPromotionFixture(t)
			calls := 0
			injectedRename := func(source string, target string) error {
				if calls == successfulRenames && successfulRenames < 4 {
					return errors.New("injected promotion interruption")
				}
				calls++
				return os.Rename(source, target)
			}
			err := promoteProcessingWorkspaceWithRename(tripDir, workspace, true, injectedRename)
			if successfulRenames < 4 && err == nil {
				t.Fatal("expected injected promotion interruption")
			}
			if successfulRenames == 4 && err != nil {
				t.Fatalf("complete promotion: %v", err)
			}
			if err := rollbackProcessingWorkspace(tripDir, workspace); err != nil {
				t.Fatalf("rollback crash phase: %v", err)
			}
			assertGenerationFixture(t, filepath.Join(tripDir, "frames"), "old")
			assertFileContent(t, filepath.Join(tripDir, "dataset.jsonl"), "old\n")
		})
	}
}

func TestPromotionPreservesBackupWhenRestoreFails(t *testing.T) {
	tripDir, workspace := createPromotionFixture(t)
	calls := 0
	err := promoteProcessingWorkspaceWithRename(tripDir, workspace, true, func(source string, target string) error {
		if calls == 3 {
			return errors.New("injected dataset publish failure")
		}
		calls++
		return os.Rename(source, target)
	})
	if err == nil {
		t.Fatal("expected injected publish failure")
	}

	err = rollbackProcessingWorkspaceWithFS(
		tripDir,
		workspace,
		func(source string, target string) error {
			if filepath.Base(source) == "previous-frames" {
				return errors.New("injected frame restore failure")
			}
			return os.Rename(source, target)
		},
		os.RemoveAll,
	)
	if err == nil {
		t.Fatal("expected injected restore failure")
	}
	if !pathExists(filepath.Join(workspace.root, "previous-frames")) {
		t.Fatal("failed rollback must preserve the only previous-generation backup")
	}
	if err := rollbackProcessingWorkspace(tripDir, workspace); err != nil {
		t.Fatalf("retry rollback: %v", err)
	}
	assertGenerationFixture(t, filepath.Join(tripDir, "frames"), "old")
	assertFileContent(t, filepath.Join(tripDir, "dataset.jsonl"), "old\n")
}

func TestPromotionRecoveryHandlesCrashBetweenRenameAndJournalUpdate(t *testing.T) {
	tests := []struct {
		name       string
		prepare    func(t *testing.T, tripDir string, workspace processingWorkspace, operation *promotionJournalOperation)
		wantTarget bool
	}{
		{
			name: "backup moved",
			prepare: func(t *testing.T, tripDir string, workspace processingWorkspace, operation *promotionJournalOperation) {
				operation.HadPrior = true
				operation.BackupStarted = true
				mustRename(t, filepath.Join(tripDir, "dataset.jsonl"), filepath.Join(workspace.root, "previous-dataset.jsonl"))
			},
			wantTarget: true,
		},
		{
			name: "published over prior",
			prepare: func(t *testing.T, tripDir string, workspace processingWorkspace, operation *promotionJournalOperation) {
				operation.HadPrior = true
				operation.BackupStarted = true
				operation.BackupDone = true
				operation.PublishStarted = true
				mustRename(t, filepath.Join(tripDir, "dataset.jsonl"), filepath.Join(workspace.root, "previous-dataset.jsonl"))
				mustRename(t, workspace.datasetPath, filepath.Join(tripDir, "dataset.jsonl"))
			},
			wantTarget: true,
		},
		{
			name: "published without prior",
			prepare: func(t *testing.T, tripDir string, workspace processingWorkspace, operation *promotionJournalOperation) {
				operation.PublishStarted = true
				if err := os.Remove(filepath.Join(tripDir, "dataset.jsonl")); err != nil {
					t.Fatalf("remove prior dataset: %v", err)
				}
				mustRename(t, workspace.datasetPath, filepath.Join(tripDir, "dataset.jsonl"))
			},
			wantTarget: false,
		},
		{
			name: "restore moved",
			prepare: func(t *testing.T, tripDir string, workspace processingWorkspace, operation *promotionJournalOperation) {
				operation.HadPrior = true
				operation.BackupStarted = true
				operation.BackupDone = true
				operation.PublishStarted = true
				operation.PublishDone = true
				operation.RestoreStarted = true
				mustRename(t, filepath.Join(tripDir, "dataset.jsonl"), filepath.Join(workspace.root, "previous-dataset.jsonl"))
				mustRename(t, workspace.datasetPath, filepath.Join(tripDir, "dataset.jsonl"))
				if err := os.Remove(filepath.Join(tripDir, "dataset.jsonl")); err != nil {
					t.Fatalf("remove published dataset: %v", err)
				}
				mustRename(t, filepath.Join(workspace.root, "previous-dataset.jsonl"), filepath.Join(tripDir, "dataset.jsonl"))
			},
			wantTarget: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tripDir := t.TempDir()
			if err := os.WriteFile(filepath.Join(tripDir, "dataset.jsonl"), []byte("old\n"), 0o644); err != nil {
				t.Fatalf("write old dataset: %v", err)
			}
			workspace, err := newProcessingWorkspace(tripDir)
			if err != nil {
				t.Fatalf("new workspace: %v", err)
			}
			if err := os.WriteFile(workspace.datasetPath, []byte("new\n"), 0o644); err != nil {
				t.Fatalf("write staged dataset: %v", err)
			}
			journal := newPromotionJournal(false)
			journal.Phase = "promoting"
			test.prepare(t, tripDir, workspace, &journal.Operations[0])
			if err := writePromotionJournal(workspace, journal); err != nil {
				t.Fatalf("write crash journal: %v", err)
			}
			if err := rollbackProcessingWorkspace(tripDir, workspace); err != nil {
				t.Fatalf("recover crash window: %v", err)
			}
			target := filepath.Join(tripDir, "dataset.jsonl")
			if !test.wantTarget {
				if pathExists(target) {
					t.Fatal("new no-prior output survived rollback")
				}
				return
			}
			assertFileContent(t, target, "old\n")
		})
	}
}

func TestProcessorQueueRecoversStalePromotionBeforeDirectBatchWork(t *testing.T) {
	tripDir, workspace := createPromotionFixture(t)
	if err := promoteProcessingWorkspace(tripDir, workspace, true); err != nil {
		t.Fatalf("publish simulated crashed promotion: %v", err)
	}
	writeJSONFile(t, filepath.Join(tripDir, "processing.json"), ProcessingStatus{State: "running"})

	statusPath, err := NewProcessor(WithForce(true)).Queue(tripDir)
	if err != nil {
		t.Fatalf("Queue after stale promotion: %v", err)
	}
	assertGenerationFixture(t, filepath.Join(tripDir, "frames"), "old")
	assertFileContent(t, filepath.Join(tripDir, "dataset.jsonl"), "old\n")
	if pathExists(workspace.root) {
		t.Fatal("successfully recovered promotion workspace was not removed")
	}
	status, err := ReadStatusFile(statusPath)
	if err != nil || status.State != "queued" {
		t.Fatalf("direct batch recovery did not leave queued status: %+v err=%v", status, err)
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

	samples, stats := buildDatasetSamplesWithStats(frames, labels, 0.0, 3, 2, 10, 100*time.Millisecond, testTelemetryOffsets(3, 2), defaultFutureOffsets(), 100*time.Millisecond)
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

	samples, stats := buildDatasetSamplesWithStats(frames, labels, 0.0, 3, 2, 8, 100*time.Millisecond, testTelemetryOffsets(3, 2), defaultFutureOffsets(), 100*time.Millisecond)
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

	derived, ok := buildTrainingLabel(current, future, 5.25, 0.75, 1.0)
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
	if flatDerived["future_speed"] != 6.5 {
		t.Fatalf("unexpected future_speed in derived label: %+v", flatDerived)
	}
	if flatDerived["future_speed_target"] != 5.25 {
		t.Fatalf("unexpected future_speed_target in derived label: %+v", flatDerived)
	}
	if flatDerived["routeForwardDelta"] != 0.75 {
		t.Fatalf("unexpected routeForwardDelta in derived label: %+v", flatDerived)
	}
	if flatDerived["future_horizon_seconds"] != 1.0 {
		t.Fatalf("unexpected future_horizon_seconds in derived label: %+v", flatDerived)
	}
	if derived.Control.Steering == nil || derived.Aux.FutureSpeedTarget == nil {
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
		WithImageSize(8, 8),
		WithSamplingConfig(3, 2, 2),
		WithTelemetryTimelineConfig(testTelemetryOffsets(3, 2), defaultFutureOffsets(), 100*time.Millisecond),
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
		WithImageSize(8, 8),
		WithSamplingConfig(3, 2, 2),
		WithTelemetryTimelineConfig(testTelemetryOffsets(3, 2), defaultFutureOffsets(), 100*time.Millisecond),
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
	if flatLabel["future_speed"] != 6.0 {
		t.Fatalf("unexpected future_speed in rewritten dataset: %+v", flatLabel)
	}
	if flatLabel["future_speed_target"] != 6.0 {
		t.Fatalf("unexpected future_speed_target in rewritten dataset: %+v", flatLabel)
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

func TestPruneUnreferencedJPEGFramesKeepsOnlyDatasetFramesAndLeavesVideo(t *testing.T) {
	tripDir := t.TempDir()
	framesDir := filepath.Join(tripDir, "frames")
	if err := os.MkdirAll(framesDir, 0o755); err != nil {
		t.Fatalf("mkdir frames: %v", err)
	}
	for index := 1; index <= 5; index++ {
		if err := os.WriteFile(filepath.Join(framesDir, formatFrameName(index)), []byte("jpeg"), 0o644); err != nil {
			t.Fatalf("write staged frame: %v", err)
		}
	}
	videoPath := filepath.Join(tripDir, "video.mkv")
	if err := os.WriteFile(videoPath, []byte("video-stays"), 0o644); err != nil {
		t.Fatalf("write video: %v", err)
	}
	samples := []DatasetSample{{FramePaths: []string{
		"frames/000001.jpg",
		"frames/000003.jpg",
		"frames/000005.jpg",
	}}}

	count, err := pruneUnreferencedJPEGFrames(framesDir, samples)
	if err != nil {
		t.Fatalf("prune staged frames: %v", err)
	}
	if count != 3 {
		t.Fatalf("unexpected retained count: got=%d want=3", count)
	}
	entries, err := os.ReadDir(framesDir)
	if err != nil {
		t.Fatalf("read pruned frames: %v", err)
	}
	gotNames := make([]string, 0, len(entries))
	for _, entry := range entries {
		gotNames = append(gotNames, entry.Name())
	}
	wantNames := []string{"000001.jpg", "000003.jpg", "000005.jpg"}
	if !reflect.DeepEqual(gotNames, wantNames) {
		t.Fatalf("unexpected retained frames: got=%v want=%v", gotNames, wantNames)
	}
	video, err := os.ReadFile(videoPath)
	if err != nil || string(video) != "video-stays" {
		t.Fatalf("video was changed by frame pruning: body=%q err=%v", video, err)
	}
}

func writeCompletedProcessingFixture(
	t *testing.T,
	tripDir string,
	processor *Processor,
	frameCount int,
	sampleCount int,
) {
	t.Helper()
	framesDir := filepath.Join(tripDir, "frames")
	if err := os.RemoveAll(framesDir); err != nil {
		t.Fatalf("clear fixture frames: %v", err)
	}
	if err := os.MkdirAll(framesDir, 0o755); err != nil {
		t.Fatalf("mkdir fixture frames: %v", err)
	}
	for index := 1; index <= frameCount; index++ {
		if err := os.WriteFile(filepath.Join(framesDir, formatFrameName(index)), []byte("frame"), 0o644); err != nil {
			t.Fatalf("write fixture frame: %v", err)
		}
	}
	datasetFile, err := os.Create(filepath.Join(tripDir, "dataset.jsonl"))
	if err != nil {
		t.Fatalf("create fixture dataset: %v", err)
	}
	framePaths := make([]string, 0, frameCount)
	for index := 1; index <= frameCount; index++ {
		framePaths = append(framePaths, "frames/"+formatFrameName(index))
	}
	encoder := json.NewEncoder(datasetFile)
	for index := 0; index < sampleCount; index++ {
		if err := encoder.Encode(map[string]any{"sample": index, "frame_paths": framePaths}); err != nil {
			_ = datasetFile.Close()
			t.Fatalf("write fixture dataset: %v", err)
		}
	}
	if err := datasetFile.Close(); err != nil {
		t.Fatalf("close fixture dataset: %v", err)
	}
	writeJSONFile(t, filepath.Join(tripDir, "processing.json"), ProcessingStatus{
		State:                     "completed",
		ConfigFingerprint:         processor.ConfigFingerprint(),
		CompletedAt:               time.Now().Format(time.RFC3339),
		FramesDir:                 "frames",
		DatasetFile:               "dataset.jsonl",
		ImageWidth:                processor.imageWidth,
		ImageHeight:               processor.imageHeight,
		ImageOffsets:              append([]int(nil), processor.imageOffsets...),
		TelemetryOffsets:          append([]int(nil), processor.telemetryOffsets...),
		FutureOffsets:             append([]int(nil), processor.futureOffsets...),
		TelemetrySampleIntervalMs: float64(processor.telemetrySampleInterval) / float64(time.Millisecond),
		FrameCount:                frameCount,
		SampleCount:               sampleCount,
	})
}

func createPromotionFixture(t *testing.T) (string, processingWorkspace) {
	t.Helper()
	tripDir := t.TempDir()
	writeGenerationFixture(t, filepath.Join(tripDir, "frames"), "old")
	if err := os.WriteFile(filepath.Join(tripDir, "dataset.jsonl"), []byte("old\n"), 0o644); err != nil {
		t.Fatalf("write old dataset: %v", err)
	}
	workspace, err := newProcessingWorkspace(tripDir)
	if err != nil {
		t.Fatalf("new processing workspace: %v", err)
	}
	writeGenerationFixture(t, workspace.framesDir, "new")
	if err := os.WriteFile(workspace.datasetPath, []byte("new\n"), 0o644); err != nil {
		t.Fatalf("write staged dataset: %v", err)
	}
	return tripDir, workspace
}

func writeGenerationFixture(t *testing.T, dir string, generation string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("mkdir generation fixture: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "generation.txt"), []byte(generation), 0o644); err != nil {
		t.Fatalf("write generation fixture: %v", err)
	}
}

func assertGenerationFixture(t *testing.T, dir string, want string) {
	t.Helper()
	assertFileContent(t, filepath.Join(dir, "generation.txt"), want)
}

func assertFileContent(t *testing.T, path string, want string) {
	t.Helper()
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	if string(body) != want {
		t.Fatalf("unexpected %s content: got=%q want=%q", path, string(body), want)
	}
}

func mustRename(t *testing.T, source string, target string) {
	t.Helper()
	if err := os.Rename(source, target); err != nil {
		t.Fatalf("rename %s to %s: %v", source, target, err)
	}
}

func writeJSONLinesFile(t *testing.T, path string, rows []runTripRecord) {
	t.Helper()
	file, err := os.Create(path)
	if err != nil {
		t.Fatalf("create JSONL file: %v", err)
	}
	encoder := json.NewEncoder(file)
	for _, row := range rows {
		if err := encoder.Encode(row); err != nil {
			_ = file.Close()
			t.Fatalf("encode JSONL row: %v", err)
		}
	}
	if err := file.Close(); err != nil {
		t.Fatalf("close JSONL file: %v", err)
	}
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

func testTelemetryOffsets(windowSize int, frameStride int) []int {
	firstOffset := -((windowSize - 1) * frameStride)
	offsets := make([]int, 0, -firstOffset+1)
	for offset := firstOffset; offset <= 0; offset++ {
		offsets = append(offsets, offset)
	}
	return offsets
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
