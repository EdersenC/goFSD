package capture

import (
	"bytes"
	"encoding/json"
	"image"
	"io"
	"net/http"
	"testing"
	"time"

	"awesomeProject/internal/actuator"
	"awesomeProject/internal/control"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func TestShouldDispatchInferenceFrame(t *testing.T) {
	offsets := []int{-4, -2, 0}
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
		got := shouldDispatchInferenceFrame(tc.frameIndex, offsets, 2)
		if got != tc.want {
			t.Fatalf("unexpected dispatch decision for frame %d: got=%v want=%v", tc.frameIndex, got, tc.want)
		}
	}
}

func TestBuildPredictionWindowUsesImageOffsets(t *testing.T) {
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

	window := buildPredictionWindow(buffer, 2, []int{-6, -3, 0})
	if window == nil {
		t.Fatal("expected prediction window")
	}
	wantIndices := []int{0, 3, 6}
	for idx, want := range wantIndices {
		if window.frameIndices[idx] != want {
			t.Fatalf("unexpected frame index at %d: got=%d want=%d", idx, window.frameIndices[idx], want)
		}
	}
	if len(window.frameTimes) != 3 || !window.frameTimes[0].Equal(now) {
		t.Fatalf("unexpected frame times: %+v", window.frameTimes)
	}
}

func TestCloneInferenceStatusCopiesPrediction(t *testing.T) {
	status := InferenceStatus{
		State: "running",
		LastPrediction: &InferencePrediction{
			WindowFrameIndices:            []int{0, 2, 4},
			SelectedTelemetryOffsets:      []int{-2, -1, 0},
			SelectedTelemetryTimestampsMs: []int64{10, 20, 30},
			RawPredControls:               [][]float64{{0.1, 0.2}},
		},
	}

	cloned := cloneInferenceStatus(status)
	cloned.LastPrediction.WindowFrameIndices[0] = 99
	cloned.LastPrediction.RawPredControls[0][0] = 9
	if status.LastPrediction.WindowFrameIndices[0] != 0 {
		t.Fatalf("expected original prediction indices to remain unchanged, got=%v", status.LastPrediction.WindowFrameIndices)
	}
	if status.LastPrediction.RawPredControls[0][0] != 0.1 {
		t.Fatalf("expected original prediction controls to remain unchanged, got=%v", status.LastPrediction.RawPredControls)
	}
}

func TestBuildInferenceFFmpegArgsUsesConfiguredFrameSize(t *testing.T) {
	cfg := DefaultInferenceConfig()
	cfg.FrameWidth = 480
	cfg.FrameHeight = 480
	cfg.FPS = 30
	spec := captureSpec{backend: "ddagrab", inputFormat: "lavfi", input: "ddagrab=output_idx=1:framerate=30:video_size=1920x1080:offset_x=0:offset_y=0"}
	args := buildInferenceFFmpegArgs(spec, cfg)
	found := false
	for _, arg := range args {
		if arg == "hwdownload,format=bgra,scale=480:480:flags=lanczos,fps=30,format=rgb24" {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected configured inference filter, args=%v", args)
	}
}

func TestInferencerModelsProxiesPythonServer(t *testing.T) {
	cfg := DefaultInferenceConfig()
	cfg.ModelServerURL = "http://planner.local"
	inferencer := NewInferencer(cfg, actuator.DefaultConfig(), nil)
	inferencer.httpClient = &http.Client{
		Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			body, _ := json.Marshal(map[string]any{
				"models": []map[string]any{{
					"label":  "run-1 - epoch 006 (best)",
					"path":   "C:/models/run-1/epoch-006.pt",
					"runId":  "run-1",
					"epoch":  6,
					"isBest": true,
				}},
			})
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body:       io.NopCloser(bytes.NewReader(body)),
			}, nil
		}),
	}

	models, err := inferencer.Models(t.Context(), "")
	if err != nil {
		t.Fatalf("Models returned error: %v", err)
	}
	if len(models) != 1 || models[0].Path != "C:/models/run-1/epoch-006.pt" {
		t.Fatalf("unexpected models: %+v", models)
	}
}

func TestRequestPredictionBuildsPlannerCommand(t *testing.T) {
	cfg := DefaultInferenceConfig()
	cfg.ModelServerURL = "http://planner.local"
	cfg.ImageOffsets = []int{-4, -2, 0}
	cfg.WindowSize = len(cfg.ImageOffsets)
	cfg.TelemetryOffsets = []int{-2, -1, 0}
	cfg.FutureSteps = 6
	cfg.HorizonControlWeights = []float64{0.60, 0.30, 0.10}
	cfg.PredictionTimeout = time.Second
	nowValue := time.UnixMilli(1000).UTC()
	store := control.NewStore(control.WithNowFunc(func() time.Time { return nowValue }))
	store.UpdateTelemetry(control.TelemetryUpdate{CurrentSpeed: 4.0, CurrentYaw: 10.0, YawRate: 0.1, Steering: 0.1, Acceleration: 0.2, TimestampMs: 1000})
	nowValue = time.UnixMilli(1033).UTC()
	store.UpdateTelemetry(control.TelemetryUpdate{CurrentSpeed: 4.5, CurrentYaw: 11.0, YawRate: 0.2, Steering: 0.2, Acceleration: 0.3, TimestampMs: 1033})
	nowValue = time.UnixMilli(1066).UTC()
	store.UpdateTelemetry(control.TelemetryUpdate{CurrentSpeed: 5.0, CurrentYaw: 12.0, YawRate: 0.3, Steering: 0.3, Acceleration: 0.4, TimestampMs: 1066})
	inferencer := NewInferencer(cfg, actuator.DefaultConfig(), store)
	now := time.UnixMilli(1066).UTC()
	inferencer.nowFunc = func() time.Time { return now }
	inferencer.httpClient = &http.Client{
		Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			var payload map[string]any
			if err := json.NewDecoder(req.Body).Decode(&payload); err != nil {
				t.Fatalf("decode payload: %v", err)
			}
			telemetry, ok := payload["telemetry"].([]any)
			if !ok || len(telemetry) != 1 {
				t.Fatalf("expected telemetry batch, got=%T %+v", payload["telemetry"], payload["telemetry"])
			}
			body, _ := json.Marshal(map[string]any{
				"checkpoint":           "C:/models/run-1/epoch-006.pt",
				"device":               "cuda",
				"planner_format":       "temporal_telemetry_gru_v1",
				"control_target_names": []string{"steering", "acceleration"},
				"pred_controls": [][][]float64{{
					{0.50, 0.40},
					{0.30, 0.20},
					{0.10, -0.10},
					{0.00, 0.00},
					{0.00, 0.00},
					{0.00, 0.00},
				}},
				"pred_aux": [][][]float64{{
					{5, 1, 0.1},
					{6, 2, 0.2},
					{7, 3, 0.3},
					{8, 4, 0.4},
					{9, 5, 0.5},
					{10, 6, 0.6},
				}},
			})
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body:       io.NopCloser(bytes.NewReader(body)),
			}, nil
		}),
	}

	window := predictionWindow{
		frames: []*image.RGBA{
			image.NewRGBA(image.Rect(0, 0, 2, 2)),
			image.NewRGBA(image.Rect(0, 0, 2, 2)),
			image.NewRGBA(image.Rect(0, 0, 2, 2)),
		},
		frameIndex:     4,
		frameIndices:   []int{0, 2, 4},
		frameTimes:     []time.Time{time.UnixMilli(1000), time.UnixMilli(1033), time.UnixMilli(1066)},
		capturedAt:     now,
		sequenceNumber: 9,
	}

	prediction, command, err := inferencer.requestPrediction(t.Context(), cfg.ModelServerURL, window)
	if err != nil {
		t.Fatalf("requestPrediction returned error: %v", err)
	}
	if prediction == nil {
		t.Fatal("expected prediction")
	}
	if got := prediction.CollapsedCommand.Steering; got <= 0.3 || got >= 0.5 {
		t.Fatalf("unexpected collapsed steering: %+v", prediction.CollapsedCommand)
	}
	if prediction.PostProcessedCommand.Throttle < 0 {
		t.Fatalf("expected non-negative throttle after processing, got=%+v", prediction.PostProcessedCommand)
	}
	if command.InputMode != actuator.InputModeNormalized {
		t.Fatalf("expected normalized command, got=%+v", command)
	}
	if len(prediction.RawPredControls) != 6 || len(prediction.RawPredAux) != 6 {
		t.Fatalf("unexpected raw planner outputs: %+v", prediction)
	}
}

func TestBuildPlannerSelectionFailsWhenTelemetryIsUnavailable(t *testing.T) {
	inferencer := NewInferencer(DefaultInferenceConfig(), actuator.DefaultConfig(), control.NewStore())
	inferencer.nowFunc = func() time.Time { return time.Unix(1710000000, 0).UTC() }
	_, err := inferencer.buildPlannerSelection(predictionWindow{capturedAt: time.Unix(1710000000, 0).UTC()})
	if err == nil || err.Error() != "planner telemetry is unavailable" {
		t.Fatalf("expected telemetry unavailable error, got=%v", err)
	}
}

func TestFindAnchorTelemetryIndexPrefersBackendReceiveTimestamp(t *testing.T) {
	history := []control.RuntimeTelemetry{
		{TimestampMs: 1000, ReceivedAtMs: 5000},
		{TimestampMs: 1033, ReceivedAtMs: 5067},
		{TimestampMs: 1066, ReceivedAtMs: 5134},
	}

	index, err := findAnchorTelemetryIndex(history, 5135, 75*time.Millisecond)
	if err != nil {
		t.Fatalf("findAnchorTelemetryIndex returned error: %v", err)
	}
	if index != 2 {
		t.Fatalf("unexpected aligned index: got=%d want=2", index)
	}
}

func TestCollapsePlannerCommandIncludesBrakePressureAvgWhenPresent(t *testing.T) {
	cfg := DefaultInferenceConfig()
	cfg.HorizonMode = "weighted_short_horizon"
	cfg.HorizonControlWeights = []float64{0.60, 0.30, 0.10}

	command, err := collapsePlannerCommand([][]float64{
		{0.50, 0.40, 0.20},
		{0.30, 0.20, 0.40},
		{0.10, 0.10, 0.80},
	}, []string{"steering", "acceleration", "brakePressureAvg"}, cfg)
	if err != nil {
		t.Fatalf("collapsePlannerCommand returned error: %v", err)
	}
	if diff := command.BrakePressureAvg - 0.32; diff < -1e-9 || diff > 1e-9 {
		t.Fatalf("unexpected collapsed brake: %+v", command)
	}
	if diff := command.Throttle - 0.31; diff < -1e-9 || diff > 1e-9 {
		t.Fatalf("unexpected collapsed throttle: %+v", command)
	}
}
