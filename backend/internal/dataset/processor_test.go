package dataset

import (
	"context"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
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

func TestBuildDatasetSamplesUsesNearestRawLabel(t *testing.T) {
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
		{RelativeSeconds: 0.19, Label: map[string]any{"time": 190.0, "Steering": 0.12, "currentSpeed": 4.5, "acceleration": 0.25}},
		{RelativeSeconds: 0.41, Label: map[string]any{"time": 410.0, "Steering": 0.32, "currentSpeed": 6.0, "acceleration": 0.5}},
	}

	samples := buildDatasetSamples(frames, labels, 0.0, 3, 2, 2, 100*time.Millisecond, 2.0, true)
	if len(samples) != 1 {
		t.Fatalf("unexpected sample count: got=%d want=1", len(samples))
	}
	if samples[0].AnchorVideoPTS != 0.2 {
		t.Fatalf("unexpected anchor pts: %+v", samples[0])
	}
	if samples[0].AnchorGameTime != 0.19 {
		t.Fatalf("unexpected anchor game time: %+v", samples[0])
	}
	if !reflect.DeepEqual(samples[0].FramePaths, []string{"frames/000001.jpg", "frames/000003.jpg", "frames/000005.jpg"}) {
		t.Fatalf("unexpected frame paths: %+v", samples[0].FramePaths)
	}
	if samples[0].Label["Steering"] != 0.12 {
		t.Fatalf("unexpected label payload: %+v", samples[0].Label)
	}
	if samples[0].Label["acceleration"] != nil {
		t.Fatalf("expected raw acceleration to be dropped, label=%+v", samples[0].Label)
	}
	if samples[0].Label["delta_speed"] != 0.75 {
		t.Fatalf("unexpected delta_speed: %+v", samples[0].Label)
	}
	if samples[0].Label["delta_speed_target"] != 0.375 {
		t.Fatalf("unexpected delta_speed_target: %+v", samples[0].Label)
	}
	if samples[0].Label["future_speed"] != 6.0 {
		t.Fatalf("unexpected future_speed: %+v", samples[0].Label)
	}
	if samples[0].Label["future_speed_target"] != 5.25 {
		t.Fatalf("unexpected future_speed_target: %+v", samples[0].Label)
	}
	if samples[0].Label["future_steer"] != 0.32 {
		t.Fatalf("unexpected future_steer: %+v", samples[0].Label)
	}
}

func TestBuildDatasetSamplesSupportsConfigurableWindowSize(t *testing.T) {
	frames := AttachImagePaths([]VideoFrame{
		{Index: 0, PTS: 0.0},
		{Index: 1, PTS: 0.1},
		{Index: 2, PTS: 0.2},
		{Index: 3, PTS: 0.3},
		{Index: 4, PTS: 0.4},
		{Index: 5, PTS: 0.5},
		{Index: 6, PTS: 0.6},
		{Index: 7, PTS: 0.7},
		{Index: 8, PTS: 0.8},
		{Index: 9, PTS: 0.9},
		{Index: 10, PTS: 1.0},
	}, "frames")
	labels := []timedLabel{
		{RelativeSeconds: 0.41, Label: map[string]any{"time": 410.0, "Steering": 0.32, "currentSpeed": 6.0}},
		{RelativeSeconds: 0.61, Label: map[string]any{"time": 610.0, "Steering": 0.22, "currentSpeed": 7.0}},
	}

	samples := buildDatasetSamples(frames, labels, 0.0, 5, 2, 2, 100*time.Millisecond, 2.0, true)
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
}

func TestBuildDatasetSamplesUsesIndependentSampleStride(t *testing.T) {
	rawFrames := make([]VideoFrame, 0, 29)
	for index := 0; index <= 28; index++ {
		rawFrames = append(rawFrames, VideoFrame{
			Index: index,
			PTS:   float64(index) * 0.1,
		})
	}
	frames := AttachImagePaths(rawFrames, "frames")
	labels := []timedLabel{
		{RelativeSeconds: 0.4, Label: map[string]any{"time": 400.0, "Steering": 0.1, "currentSpeed": 4.0}},
		{RelativeSeconds: 1.4, Label: map[string]any{"time": 1400.0, "Steering": 0.2, "currentSpeed": 6.0}},
		{RelativeSeconds: 2.4, Label: map[string]any{"time": 2400.0, "Steering": 0.3, "currentSpeed": 9.0}},
	}

	samples := buildDatasetSamples(frames, labels, 0.0, 5, 2, 10, 100*time.Millisecond, 2.0, true)
	if len(samples) != 2 {
		t.Fatalf("unexpected sample count: got=%d want=2", len(samples))
	}
	if !reflect.DeepEqual(samples[0].FramePaths, []string{
		"frames/000001.jpg",
		"frames/000003.jpg",
		"frames/000005.jpg",
		"frames/000007.jpg",
		"frames/000009.jpg",
	}) {
		t.Fatalf("unexpected first sample frame paths: %+v", samples[0].FramePaths)
	}
	if !reflect.DeepEqual(samples[1].FramePaths, []string{
		"frames/000011.jpg",
		"frames/000013.jpg",
		"frames/000015.jpg",
		"frames/000017.jpg",
		"frames/000019.jpg",
	}) {
		t.Fatalf("unexpected second sample frame paths: %+v", samples[1].FramePaths)
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

func TestBuildTrainingLabelDropsAccelerationAndAddsFutureTargets(t *testing.T) {
	current := map[string]any{
		"time":         400.0,
		"Steering":     0.1,
		"currentSpeed": 4.0,
		"acceleration": 0.25,
	}
	future := map[string]any{
		"time":         1400.0,
		"Steering":     0.2,
		"currentSpeed": 6.5,
		"acceleration": 0.5,
	}

	derived, ok := buildTrainingLabel(current, future, 5.25, 2.0, true)
	if !ok {
		t.Fatal("expected training label to be derived")
	}
	if derived["acceleration"] != nil {
		t.Fatalf("expected acceleration to be removed, label=%+v", derived)
	}
	if derived["currentSpeed"] != 4.0 {
		t.Fatalf("unexpected currentSpeed in derived label: %+v", derived)
	}
	if derived["delta_speed"] != 1.25 {
		t.Fatalf("unexpected delta_speed in derived label: %+v", derived)
	}
	if derived["delta_speed_target"] != 0.625 {
		t.Fatalf("unexpected delta_speed_target in derived label: %+v", derived)
	}
	if derived["future_speed"] != 6.5 {
		t.Fatalf("unexpected future_speed in derived label: %+v", derived)
	}
	if derived["future_speed_target"] != 5.25 {
		t.Fatalf("unexpected future_speed_target in derived label: %+v", derived)
	}
	if derived["future_steer"] != 0.2 {
		t.Fatalf("unexpected future_steer in derived label: %+v", derived)
	}
}

func TestThinStoppedSamplesKeepsBurstThenSparseSamples(t *testing.T) {
	samples := []DatasetSample{
		{AnchorGameTime: 0.0, Label: map[string]any{"isStopped": false, "Steering": 0.1}},
		{AnchorGameTime: 1.0, Label: map[string]any{"isStopped": true, "Steering": 0.2}},
		{AnchorGameTime: 1.5, Label: map[string]any{"isStopped": 1.0, "Steering": 0.3}},
		{AnchorGameTime: 2.0, Label: map[string]any{"isStopped": true, "Steering": 0.4}},
		{AnchorGameTime: 2.5, Label: map[string]any{"isStopped": 1.0, "Steering": 0.5}},
		{AnchorGameTime: 3.1, Label: map[string]any{"isStopped": true, "Steering": 0.6}},
		{AnchorGameTime: 3.2, Label: map[string]any{"isStopped": false, "Steering": 0.7}},
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
		{AnchorGameTime: 10.0, Label: map[string]any{"isStopped": true}},
		{AnchorGameTime: 10.5, Label: map[string]any{"isStopped": true}},
		{AnchorGameTime: 11.0, Label: map[string]any{"isStopped": true}},
		{AnchorGameTime: 13.1, Label: map[string]any{"isStopped": true}},
		{AnchorGameTime: 13.2, Label: map[string]any{"isStopped": false}},
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
	runBody := `{"runId":"run-a","tripIndex":0,"vehicleData":[{"time":200,"Steering":0.1,"currentSpeed":8.0,"isStopped":false},{"time":400,"Steering":0.2,"currentSpeed":7.0,"isStopped":true},{"time":600,"Steering":0.2,"currentSpeed":6.0,"isStopped":true},{"time":800,"Steering":0.2,"currentSpeed":5.0,"isStopped":1},{"time":1000,"Steering":0.2,"currentSpeed":4.0,"isStopped":true},{"time":1200,"Steering":0.2,"currentSpeed":3.0,"isStopped":true},{"time":1400,"Steering":0.3,"currentSpeed":2.0,"isStopped":false}]}`
	if err := os.WriteFile(filepath.Join(tmp, "run.jsonl"), []byte(runBody+"\n"), 0o644); err != nil {
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
	if len(lines) != 4 {
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
		samples[0].Label["isStopped"],
		samples[1].Label["isStopped"],
		samples[2].Label["isStopped"],
		samples[3].Label["isStopped"],
	}
	wantStopped := []any{false, true, true, 1.0}
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

	writeJPEG(t, filepath.Join(framesDir, "000001.jpg"), 255)
	writeJPEG(t, filepath.Join(framesDir, "000002.jpg"), 80)
	writeJPEG(t, filepath.Join(framesDir, "000003.jpg"), 80)
	writeJPEG(t, filepath.Join(framesDir, "000004.jpg"), 80)
	writeJPEG(t, filepath.Join(framesDir, "000005.jpg"), 80)
	writeJPEG(t, filepath.Join(framesDir, "000006.jpg"), 80)
	writeJPEG(t, filepath.Join(framesDir, "000007.jpg"), 80)

	metadataBody := `{"runId":"run-a","sceneId":"scene-a","sceneVariant":"default","tripIndex":0,"syncTime":0}`
	if err := os.WriteFile(filepath.Join(tripDir, "metadata.json"), []byte(metadataBody), 0o644); err != nil {
		t.Fatalf("write metadata: %v", err)
	}
	if err := os.WriteFile(filepath.Join(tripDir, "video.mkv"), []byte("stub"), 0o644); err != nil {
		t.Fatalf("write video: %v", err)
	}
	runBody := `{"runId":"run-a","tripIndex":0,"vehicleData":[{"time":200,"Steering":0.1,"currentSpeed":4.0,"acceleration":0.2},{"time":400,"Steering":0.2,"currentSpeed":6.0,"acceleration":0.4}]}`
	if err := os.WriteFile(filepath.Join(tmp, "run.jsonl"), []byte(runBody+"\n"), 0o644); err != nil {
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
	if sample.Label["delta_speed"] != 1.0 {
		t.Fatalf("unexpected delta_speed in rewritten dataset: %+v", sample.Label)
	}
	if sample.Label["delta_speed_target"] != 0.5 {
		t.Fatalf("unexpected delta_speed_target in rewritten dataset: %+v", sample.Label)
	}
	if sample.Label["future_speed"] != 6.0 {
		t.Fatalf("unexpected future_speed in rewritten dataset: %+v", sample.Label)
	}
	if sample.Label["future_speed_target"] != 5.0 {
		t.Fatalf("unexpected future_speed_target in rewritten dataset: %+v", sample.Label)
	}
	if sample.Label["future_steer"] != 0.2 {
		t.Fatalf("unexpected future_steer in rewritten dataset: %+v", sample.Label)
	}
	if sample.Label["acceleration"] != nil {
		t.Fatalf("expected rewritten dataset to omit acceleration: %+v", sample.Label)
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
	_, _ = os.Stdout.Write([]byte(`{"frames":[{"pts_time":"0.0"},{"pts_time":"0.1"},{"pts_time":"0.2"},{"pts_time":"0.3"},{"pts_time":"0.4"},{"pts_time":"0.5"},{"pts_time":"0.6"}]}`))
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
