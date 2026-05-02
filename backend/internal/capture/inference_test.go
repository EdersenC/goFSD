package capture

import (
	"bytes"
	"encoding/json"
	"image"
	"io"
	"math"
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
					{5, 0.0, 1, 0.1},
					{6, 1.0, 2, 0.2},
					{7, 2.0, 3, 0.3},
					{8, 3.0, 4, 0.4},
					{9, 4.0, 5, 0.5},
					{10, 5.0, 6, 0.6},
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
	if prediction.PredictionHorizon == nil {
		t.Fatal("expected temporal prediction horizon to be recorded")
	}
	if len(prediction.PredictionHorizon.Points) != 6 || prediction.PredictionHorizon.Points[0].DtMs != 100 {
		t.Fatalf("unexpected temporal horizon bins: %+v", prediction.PredictionHorizon.Points)
	}
	if point := prediction.PredictionHorizon.Points[0]; point.DesiredSpeedMPS == nil || math.Abs(*point.DesiredSpeedMPS-5) > 1e-9 {
		t.Fatalf("expected future_speed aux to populate desired speed, got=%+v", point)
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

func TestBuildPlannerSelectionRecordsFrameTelemetrySkew(t *testing.T) {
	cfg := DefaultInferenceConfig()
	cfg.TelemetryOffsets = []int{0}
	cfg.TelemetryFeatureNames = []string{"current_speed"}
	cfg.MaxFrameTelemetrySkew = 75 * time.Millisecond
	nowValue := time.UnixMilli(1000).UTC()
	store := control.NewStore(control.WithNowFunc(func() time.Time { return nowValue }))
	store.UpdateTelemetry(control.TelemetryUpdate{CurrentSpeed: 4.0, TimestampMs: 1000})
	inferencer := NewInferencer(cfg, actuator.DefaultConfig(), store)
	inferencer.nowFunc = func() time.Time { return time.UnixMilli(1100).UTC() }

	selection, err := inferencer.buildPlannerSelection(predictionWindow{
		frameIndex:     7,
		frameTimes:     []time.Time{time.UnixMilli(1100).UTC()},
		capturedAt:     time.UnixMilli(1100).UTC(),
		sequenceNumber: 3,
	})
	if err != nil {
		t.Fatalf("buildPlannerSelection returned error: %v", err)
	}
	if selection.frameTelemetryAligned {
		t.Fatalf("expected frame/telemetry skew to exceed threshold, got=%+v", selection)
	}
	if math.Abs(selection.frameTelemetrySkewMs-100) > 1e-9 {
		t.Fatalf("unexpected skew: %+v", selection)
	}
	if selection.frameID == nil || *selection.frameID != 7 {
		t.Fatalf("expected frame id to be captured, got=%+v", selection.frameID)
	}
}

func TestBuildPredictionAdaptsLegacyImmediateControlToTemporalHorizon(t *testing.T) {
	cfg := DefaultInferenceConfig()
	cfg.FutureSteps = 6
	actuatorCfg := actuator.DefaultConfig()
	actuatorCfg.TemporalHorizonActuatorEnabled = true
	inferencer := NewInferencer(cfg, actuatorCfg, control.NewStore())
	now := time.UnixMilli(2000).UTC()
	inferencer.nowFunc = func() time.Time { return now }

	prediction, _, err := inferencer.buildPrediction(pythonPredictResponse{
		PlannerFormat:      "temporal_telemetry_gru_v1",
		ControlTargetNames: []string{"steering", "acceleration", "brakePressureAvg"},
		PredControls: [][][]float64{{
			{0.2, 0.4, 0.1},
		}},
	}, "http://planner.local", predictionWindow{
		frameIndex:     1,
		frameIndices:   []int{1},
		frameTimes:     []time.Time{time.UnixMilli(1900).UTC()},
		capturedAt:     time.UnixMilli(1900).UTC(),
		sequenceNumber: 1,
	}, plannerSelection{
		selectedTelemetry: []control.RuntimeTelemetry{{CurrentSpeed: 3}},
		telemetryTimesMs:  []int64{1900},
		frameShape:        []int{1, 1, 3, cfg.FrameHeight, cfg.FrameWidth},
		telemetryShape:    []int{1, 1, len(cfg.TelemetryFeatureNames)},
	}, nil)
	if err != nil {
		t.Fatalf("buildPrediction returned error: %v", err)
	}
	if prediction.PredictionHorizon == nil {
		t.Fatal("expected legacy immediate output to produce a temporal horizon")
	}
	if len(prediction.PredictionHorizon.Points) != len(actuator.DefaultPredictionHorizonDtMs) {
		t.Fatalf("unexpected adapted horizon length: %+v", prediction.PredictionHorizon.Points)
	}
	first := prediction.PredictionHorizon.Points[0]
	if first.Steer == nil || *first.Steer != 0.2 || first.Throttle == nil || *first.Throttle != 0.4 || first.Brake == nil || *first.Brake != 0.1 {
		t.Fatalf("unexpected adapted point: %+v", first)
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

func TestCollapsePlannerCommandConvertsNegativeThrottleToBrakePressureWhenPresent(t *testing.T) {
	cfg := DefaultInferenceConfig()
	cfg.HorizonMode = "weighted_short_horizon"
	cfg.HorizonControlWeights = []float64{0.60, 0.30, 0.10}

	command, err := collapsePlannerCommand([][]float64{
		{0.40, -0.50, 0.10},
		{0.20, -0.30, 0.80},
		{-0.10, -0.10, 0.20},
	}, []string{"steering", "acceleration", "brakePressureAvg"}, cfg)
	if err != nil {
		t.Fatalf("collapsePlannerCommand returned error: %v", err)
	}
	if math.Abs(command.Throttle-0.0) > 1e-9 {
		t.Fatalf("expected no throttle when model requests reverse, got=%+v", command.Throttle)
	}
	if command.BrakePressureAvg < 0.5-1e-9 || command.BrakePressureAvg > 0.5+1e-9 {
		t.Fatalf("expected brake from max abs(reverse demand) to dominate, got=%+v", command)
	}
}

func TestStabilizeThrottleCommandHoldsOnDemandDrop(t *testing.T) {
	cfg := DefaultInferenceConfig()
	cfg.ThrottleHoldSeconds = 2
	inf := NewInferencer(cfg, actuator.DefaultConfig(), control.NewStore())

	now := time.UnixMilli(1000).UTC()
	inf.nowFunc = func() time.Time { return now }
	value, held := inf.stabilizeThrottleCommand(0.6, 8, now)
	if held {
		t.Fatalf("expected hold disabled on first throttle demand, got=%v value=%f", held, value)
	}
	if math.Abs(value-0.6) > 1e-9 {
		t.Fatalf("unexpected throttled value: %f", value)
	}

	now = now.Add(500 * time.Millisecond)
	value, held = inf.stabilizeThrottleCommand(0.3, 8, now)
	if !held {
		t.Fatalf("expected throttle hold after demand drop, got=%v value=%f", held, value)
	}
	if math.Abs(value-0.6) > 1e-9 {
		t.Fatalf("expected held throttle to remain at previous peak, got=%f", value)
	}

	now = now.Add(2500 * time.Millisecond)
	value, held = inf.stabilizeThrottleCommand(0.3, 8, now)
	if held {
		t.Fatalf("expected hold to expire, got=%v value=%f", held, value)
	}
	if math.Abs(value-0.3) > 1e-9 {
		t.Fatalf("unexpected throttled value after hold window: %f", value)
	}
}

func TestStabilizeThrottleCommandEnforcesMinWhenBelowFloor(t *testing.T) {
	cfg := DefaultInferenceConfig()
	cfg.ThrottleHoldMin = 0.12
	inf := NewInferencer(cfg, actuator.DefaultConfig(), control.NewStore())

	value, held := inf.stabilizeThrottleCommand(0.05, 0, time.UnixMilli(1200).UTC())
	if held {
		t.Fatalf("expected no hold when applying low throttle for first time")
	}
	if math.Abs(value-0.12) > 1e-9 {
		t.Fatalf("expected throttle floor to apply below hold min, got=%f", value)
	}
}
