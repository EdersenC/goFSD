package capture

import (
	"encoding/json"
	"image"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"awesomeProject/internal/control"
)

func TestShouldDispatchInferenceFrame(t *testing.T) {
	cases := []struct {
		frameIndex int
		want       bool
	}{
		{frameIndex: 0, want: false},
		{frameIndex: 3, want: false},
		{frameIndex: 4, want: true},
		{frameIndex: 5, want: false},
		{frameIndex: 6, want: true},
	}

	for _, tc := range cases {
		got := shouldDispatchInferenceFrame(tc.frameIndex, 3, 2, 2)
		if got != tc.want {
			t.Fatalf("unexpected dispatch decision for frame %d: got=%v want=%v", tc.frameIndex, got, tc.want)
		}
	}
}

func TestShouldDispatchInferenceFrameUsesIndependentDispatchStride(t *testing.T) {
	cases := []struct {
		frameIndex int
		want       bool
	}{
		{frameIndex: 4, want: true},
		{frameIndex: 5, want: true},
		{frameIndex: 6, want: true},
	}

	for _, tc := range cases {
		got := shouldDispatchInferenceFrame(tc.frameIndex, 3, 2, 1)
		if got != tc.want {
			t.Fatalf("unexpected dispatch decision for frame %d: got=%v want=%v", tc.frameIndex, got, tc.want)
		}
	}
}

func TestBuildPredictionWindowUsesEveryOtherFrame(t *testing.T) {
	now := time.Unix(1710000000, 0).UTC()
	buffer := []bufferedInferenceFrame{
		{index: 0, capturedAt: now, image: image.NewRGBA(image.Rect(0, 0, 1, 1))},
		{index: 1, capturedAt: now.Add(10 * time.Millisecond), image: image.NewRGBA(image.Rect(0, 0, 1, 1))},
		{index: 2, capturedAt: now.Add(20 * time.Millisecond), image: image.NewRGBA(image.Rect(0, 0, 1, 1))},
		{index: 3, capturedAt: now.Add(30 * time.Millisecond), image: image.NewRGBA(image.Rect(0, 0, 1, 1))},
		{index: 4, capturedAt: now.Add(40 * time.Millisecond), image: image.NewRGBA(image.Rect(0, 0, 1, 1))},
	}

	window := buildPredictionWindow(buffer, 7, 3, 2)
	if window == nil {
		t.Fatal("expected prediction window")
	}
	if window.frameIndex != 4 {
		t.Fatalf("unexpected frame index: got=%d want=4", window.frameIndex)
	}
	if window.sequenceNumber != 7 {
		t.Fatalf("unexpected sequence number: got=%d want=7", window.sequenceNumber)
	}
	wantIndices := []int{0, 2, 4}
	for idx, want := range wantIndices {
		if window.frameIndices[idx] != want {
			t.Fatalf("unexpected frame index at %d: got=%d want=%d", idx, window.frameIndices[idx], want)
		}
	}
}

func TestCloneInferenceStatusCopiesPrediction(t *testing.T) {
	status := InferenceStatus{
		State: "running",
		LastPrediction: &InferencePrediction{
			Steering:           0.25,
			DeltaSpeed:         -0.1,
			WindowFrameIndices: []int{0, 2, 4},
		},
	}

	cloned := cloneInferenceStatus(status)
	cloned.LastPrediction.WindowFrameIndices[0] = 99
	if status.LastPrediction.WindowFrameIndices[0] != 0 {
		t.Fatalf("expected original prediction indices to remain unchanged, got=%v", status.LastPrediction.WindowFrameIndices)
	}
}

func TestBuildInferenceFFmpegArgsUsesConfiguredFrameSize(t *testing.T) {
	cfg := DefaultInferenceConfig()
	cfg.FrameWidth = 480
	cfg.FrameHeight = 480
	cfg.FPS = 30
	spec := captureSpec{backend: "ddagrab", inputFormat: "lavfi", input: "ddagrab=output_idx=1:framerate=30:video_size=1920x1080:offset_x=0:offset_y=0"}
	args := buildInferenceFFmpegArgs(spec, cfg)
	if !containsInferenceArg(args, "hwdownload,format=bgra,scale=480:480:flags=lanczos,fps=30,format=rgb24") {
		t.Fatalf("expected configured inference filter, args=%v", args)
	}
}

func TestBuildPredictionWindowUsesConfiguredStride(t *testing.T) {
	now := time.Unix(1710000000, 0).UTC()
	buffer := []bufferedInferenceFrame{
		{index: 0, capturedAt: now, image: image.NewRGBA(image.Rect(0, 0, 1, 1))},
		{index: 1, capturedAt: now.Add(10 * time.Millisecond), image: image.NewRGBA(image.Rect(0, 0, 1, 1))},
		{index: 2, capturedAt: now.Add(20 * time.Millisecond), image: image.NewRGBA(image.Rect(0, 0, 1, 1))},
		{index: 3, capturedAt: now.Add(30 * time.Millisecond), image: image.NewRGBA(image.Rect(0, 0, 1, 1))},
		{index: 4, capturedAt: now.Add(40 * time.Millisecond), image: image.NewRGBA(image.Rect(0, 0, 1, 1))},
		{index: 5, capturedAt: now.Add(50 * time.Millisecond), image: image.NewRGBA(image.Rect(0, 0, 1, 1))},
		{index: 6, capturedAt: now.Add(60 * time.Millisecond), image: image.NewRGBA(image.Rect(0, 0, 1, 1))},
	}

	window := buildPredictionWindow(buffer, 2, 3, 3)
	if window == nil {
		t.Fatal("expected prediction window")
	}
	wantIndices := []int{0, 3, 6}
	for idx, want := range wantIndices {
		if window.frameIndices[idx] != want {
			t.Fatalf("unexpected frame index at %d: got=%d want=%d", idx, window.frameIndices[idx], want)
		}
	}
}

func containsInferenceArg(args []string, want string) bool {
	for _, arg := range args {
		if arg == want {
			return true
		}
	}
	return false
}

func TestSourceIDFromNameDoesNotPanicOnShortHash(t *testing.T) {
	got := sourceIDFromName("tiny")
	if got == "" {
		t.Fatal("expected non-empty source id")
	}
}

func TestInferencerModelsProxiesPythonServer(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/models" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"models": []map[string]any{{
				"label":  "run-1 - epoch 006 (best)",
				"path":   "C:/models/run-1/epoch-006.pt",
				"runId":  "run-1",
				"epoch":  6,
				"isBest": true,
			}},
		})
	}))
	defer server.Close()

	cfg := DefaultInferenceConfig()
	cfg.ModelServerURL = server.URL
	inferencer := NewInferencer(cfg, nil)

	models, err := inferencer.Models(t.Context(), "")
	if err != nil {
		t.Fatalf("Models returned error: %v", err)
	}
	if len(models) != 1 {
		t.Fatalf("unexpected model count: got=%d want=1", len(models))
	}
	if models[0].Path != "C:/models/run-1/epoch-006.pt" {
		t.Fatalf("unexpected model path: %s", models[0].Path)
	}
	if !models[0].IsBest {
		t.Fatal("expected model to be marked best")
	}
}

func TestRequestPredictionReadsControlOutputs(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/predict" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if r.Method != http.MethodPost {
			t.Fatalf("unexpected method: %s", r.Method)
		}
		var payload map[string]any
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("decode payload: %v", err)
		}
		if payload["current_speed"] != 4.5 {
			t.Fatalf("expected current_speed=4.5, got=%v", payload["current_speed"])
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"checkpoint": "C:/models/run-1/epoch-006.pt",
			"device":     "cuda",
			"control_outputs": map[string]any{
				"steer":       0.125,
				"delta_speed": -0.375,
			},
		})
	}))
	defer server.Close()

	cfg := DefaultInferenceConfig()
	cfg.ModelServerURL = server.URL
	store := control.NewStore(control.WithNowFunc(func() time.Time { return time.Unix(1710000000, 0).UTC() }))
	store.UpdateTelemetry(control.TelemetryUpdate{CurrentSpeed: 4.5, TimestampMs: 123})
	inferencer := NewInferencer(cfg, store)
	now := time.Unix(1710000000, 0).UTC()
	inferencer.nowFunc = func() time.Time { return now }

	window := predictionWindow{
		frames: []*image.RGBA{
			image.NewRGBA(image.Rect(0, 0, 2, 2)),
			image.NewRGBA(image.Rect(0, 0, 2, 2)),
			image.NewRGBA(image.Rect(0, 0, 2, 2)),
		},
		frameIndex:     42,
		frameIndices:   []int{38, 40, 42},
		capturedAt:     now.Add(-33 * time.Millisecond),
		sequenceNumber: 9,
	}

	prediction, err := inferencer.requestPrediction(t.Context(), server.URL, window)
	if err != nil {
		t.Fatalf("requestPrediction returned error: %v", err)
	}
	if prediction.Steering != 0.125 {
		t.Fatalf("unexpected steering: got=%f want=0.125", prediction.Steering)
	}
	if prediction.DeltaSpeed != -0.375 {
		t.Fatalf("unexpected delta speed: got=%f want=-0.375", prediction.DeltaSpeed)
	}
	if prediction.CurrentSpeed != 4.5 {
		t.Fatalf("unexpected current speed: got=%f want=4.5", prediction.CurrentSpeed)
	}
	if prediction.Sequence != 9 {
		t.Fatalf("unexpected sequence: got=%d want=9", prediction.Sequence)
	}
	if len(prediction.WindowFrameHashes) != 3 {
		t.Fatalf("unexpected frame hash count: got=%d want=3", len(prediction.WindowFrameHashes))
	}
}

func TestRequestPredictionFailsWhenResponseOmitsControls(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"checkpoint": "C:/models/run-1/epoch-006.pt",
			"device":     "cuda",
			"outputs": map[string]any{
				"route_xy": []float64{0.1, 0.2},
			},
		})
	}))
	defer server.Close()

	cfg := DefaultInferenceConfig()
	cfg.ModelServerURL = server.URL
	store := control.NewStore()
	store.UpdateTelemetry(control.TelemetryUpdate{CurrentSpeed: 1.0, TimestampMs: 123})
	inferencer := NewInferencer(cfg, store)

	window := predictionWindow{
		frames: []*image.RGBA{
			image.NewRGBA(image.Rect(0, 0, 2, 2)),
			image.NewRGBA(image.Rect(0, 0, 2, 2)),
			image.NewRGBA(image.Rect(0, 0, 2, 2)),
		},
		frameIndex:     2,
		frameIndices:   []int{0, 1, 2},
		capturedAt:     time.Unix(1710000000, 0).UTC(),
		sequenceNumber: 1,
	}

	_, err := inferencer.requestPrediction(t.Context(), server.URL, window)
	if err == nil {
		t.Fatal("expected requestPrediction to fail when control outputs are missing")
	}
	if !strings.Contains(err.Error(), "missing control outputs") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRequestPredictionFailsWhenTelemetryIsUnavailable(t *testing.T) {
	cfg := DefaultInferenceConfig()
	inferencer := NewInferencer(cfg, control.NewStore())
	now := time.Unix(1710000000, 0).UTC()
	inferencer.nowFunc = func() time.Time { return now }

	window := predictionWindow{
		frames: []*image.RGBA{
			image.NewRGBA(image.Rect(0, 0, 2, 2)),
			image.NewRGBA(image.Rect(0, 0, 2, 2)),
			image.NewRGBA(image.Rect(0, 0, 2, 2)),
		},
		frameIndex:     2,
		frameIndices:   []int{0, 1, 2},
		capturedAt:     now,
		sequenceNumber: 1,
	}

	_, err := inferencer.requestPrediction(t.Context(), "http://127.0.0.1:1", window)
	if err == nil {
		t.Fatal("expected requestPrediction to fail without telemetry")
	}
	if !strings.Contains(err.Error(), "currentSpeed telemetry is unavailable") {
		t.Fatalf("unexpected error: %v", err)
	}
}
