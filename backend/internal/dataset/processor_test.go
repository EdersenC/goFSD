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
	}, "frames")
	labels := []timedLabel{
		{RelativeSeconds: 0.19, Label: map[string]any{"time": 190.0, "Steering": 0.12}},
		{RelativeSeconds: 0.41, Label: map[string]any{"time": 410.0, "Steering": 0.32}},
	}

	samples := buildDatasetSamples(frames, labels, 0.0, 3, 2, 100*time.Millisecond)
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

func TestHelperProcessFFprobe(t *testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") != "ffprobe" {
		return
	}
	_, _ = os.Stdout.Write([]byte(`{"frames":[{"pts_time":"0.0"},{"pts_time":"0.1"}]}`))
	os.Exit(0)
}

func writeJPEG(t *testing.T, path string, brightness uint8) {
	t.Helper()
	file, err := os.Create(path)
	if err != nil {
		t.Fatalf("create jpeg: %v", err)
	}
	defer file.Close()

	img := image.NewRGBA(image.Rect(0, 0, 8, 8))
	fill := color.RGBA{R: brightness, G: brightness, B: brightness, A: 255}
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			img.Set(x, y, fill)
		}
	}
	if err := jpeg.Encode(file, img, nil); err != nil {
		t.Fatalf("encode jpeg: %v", err)
	}
}

func formatFrameName(index int) string {
	return fmt.Sprintf("%06d.jpg", index)
}
