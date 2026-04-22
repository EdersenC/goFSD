package capture

import (
	"encoding/json"
	"image"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"awesomeProject/internal/actuator"
	"awesomeProject/internal/control"
)

type fakeActuatorSubmitter struct {
	commands []actuator.CommandRequest
	err      error
}

func (f *fakeActuatorSubmitter) Submit(req actuator.CommandRequest) (actuator.State, error) {
	f.commands = append(f.commands, req)
	return actuator.State{}, f.err
}

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
			FutureYawDelta:     15.0,
			FutureYaw:          35.0,
			CurrentYaw:         20.0,
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
		if payload["route_forward_delta"] != 0.75 {
			t.Fatalf("expected route_forward_delta=0.75, got=%v", payload["route_forward_delta"])
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"checkpoint": "C:/models/run-1/epoch-006.pt",
			"device":     "cuda",
			"control_outputs": map[string]any{
				"future_yaw_delta": 33.0,
				"future_speed":     5.25,
				"delta_speed":      -0.375,
				"move_intent_prob": 0.8,
			},
		})
	}))
	defer server.Close()

	cfg := DefaultInferenceConfig()
	cfg.ModelServerURL = server.URL
	store := control.NewStore(control.WithNowFunc(func() time.Time { return time.Unix(1710000000, 0).UTC() }))
	store.UpdateTelemetry(control.TelemetryUpdate{
		CurrentSpeed:      4.5,
		CurrentYaw:        12.0,
		RouteForwardDelta: 0.75,
		TimestampMs:       123,
	})
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
	if prediction.FutureYawDelta != 33.0 {
		t.Fatalf("unexpected future yaw delta: got=%f want=33.0", prediction.FutureYawDelta)
	}
	if prediction.FutureYaw != 45.0 {
		t.Fatalf("unexpected future yaw: got=%f want=45.0", prediction.FutureYaw)
	}
	if prediction.FutureSpeed != 5.25 {
		t.Fatalf("unexpected future speed: got=%f want=5.25", prediction.FutureSpeed)
	}
	if prediction.DeltaSpeed != -0.375 {
		t.Fatalf("unexpected delta speed: got=%f want=-0.375", prediction.DeltaSpeed)
	}
	if prediction.CurrentSpeed != 4.5 {
		t.Fatalf("unexpected current speed: got=%f want=4.5", prediction.CurrentSpeed)
	}
	if prediction.CurrentYaw != 12.0 {
		t.Fatalf("unexpected current yaw: got=%f want=12.0", prediction.CurrentYaw)
	}
	if prediction.RouteForwardDelta != 0.75 {
		t.Fatalf("unexpected route forward delta: got=%f want=0.75", prediction.RouteForwardDelta)
	}
	if !prediction.HasMoveIntent || prediction.MoveIntentProb != 0.8 {
		t.Fatalf("unexpected move intent: %+v", prediction)
	}
	if prediction.Sequence != 9 {
		t.Fatalf("unexpected sequence: got=%d want=9", prediction.Sequence)
	}
	if len(prediction.WindowFrameHashes) != 3 {
		t.Fatalf("unexpected frame hash count: got=%d want=3", len(prediction.WindowFrameHashes))
	}
}

func TestCommandFromPredictionUsesFutureSpeedWithDeltaTrim(t *testing.T) {
	cfg := DefaultInferenceConfig()
	inferencer := NewInferencer(cfg, nil)
	prediction := &InferencePrediction{
		FutureYawDelta:   15.0,
		FutureYaw:        30.0,
		FutureSpeed:      8.0,
		DeltaSpeed:       -0.4,
		CurrentSpeed:     4.0,
		CurrentYaw:       15.0,
		ControlSemantics: "target_speed",
		MoveIntentProb:   0.9,
		HasMoveIntent:    true,
		Sequence:         11,
	}

	command, err := inferencer.commandFromPrediction(prediction)
	if err != nil {
		t.Fatalf("commandFromPrediction returned error: %v", err)
	}
	if prediction.CommandPreview == nil {
		t.Fatal("expected command preview to be populated")
	}
	if command.InputMode != actuator.InputModeNormalized {
		t.Fatalf("unexpected input mode: got=%s want=%s", command.InputMode, actuator.InputModeNormalized)
	}
	if command.Throttle <= 0 {
		t.Fatalf("expected throttle > 0, got=%f", command.Throttle)
	}
	if command.Brake != 0 {
		t.Fatalf("expected brake=0, got=%f", command.Brake)
	}
	if prediction.HeadingErrorDeg != 15.0 {
		t.Fatalf("unexpected heading error: got=%f want=15.0", prediction.HeadingErrorDeg)
	}
	if prediction.LongitudinalMode != "future_speed" {
		t.Fatalf("unexpected longitudinal mode: %s", prediction.LongitudinalMode)
	}
}

func TestCommandFromPredictionSuppressesPositiveThrottleWhenMoveIntentIsLow(t *testing.T) {
	cfg := DefaultInferenceConfig()
	inferencer := NewInferencer(cfg, nil)
	prediction := &InferencePrediction{
		FutureYawDelta:   15.0,
		FutureYaw:        30.0,
		FutureSpeed:      8.0,
		DeltaSpeed:       0.2,
		CurrentSpeed:     0.0,
		CurrentYaw:       15.0,
		ControlSemantics: "target_speed",
		MoveIntentProb:   0.2,
		HasMoveIntent:    true,
	}

	command, err := inferencer.commandFromPrediction(prediction)
	if err != nil {
		t.Fatalf("commandFromPrediction returned error: %v", err)
	}
	if command.Throttle != 0 {
		t.Fatalf("expected throttle to be gated off, got=%f", command.Throttle)
	}
	if command.Brake != 0 {
		t.Fatalf("expected brake=0, got=%f", command.Brake)
	}
}

func TestCommandFromPredictionKeepsLowSpeedThrottleAliveAcrossMoveIntentHysteresisBand(t *testing.T) {
	cfg := DefaultInferenceConfig()
	inferencer := NewInferencer(cfg, nil)

	first := &InferencePrediction{
		FutureYawDelta:   0.0,
		FutureSpeed:      4.0,
		DeltaSpeed:       0.1,
		CurrentSpeed:     0.0,
		CurrentYaw:       0.0,
		ControlSemantics: "target_speed",
		MoveIntentProb:   0.8,
		HasMoveIntent:    true,
	}
	firstCommand, err := inferencer.commandFromPrediction(first)
	if err != nil {
		t.Fatalf("first commandFromPrediction returned error: %v", err)
	}
	if firstCommand.Throttle <= 0 {
		t.Fatalf("expected first throttle > 0, got=%f", firstCommand.Throttle)
	}
	if !first.MoveIntentActive {
		t.Fatal("expected move intent to latch active on high probability")
	}

	second := &InferencePrediction{
		FutureYawDelta:   0.0,
		FutureSpeed:      3.0,
		DeltaSpeed:       0.05,
		CurrentSpeed:     2.0,
		CurrentYaw:       0.0,
		ControlSemantics: "target_speed",
		MoveIntentProb:   0.5,
		HasMoveIntent:    true,
	}
	secondCommand, err := inferencer.commandFromPrediction(second)
	if err != nil {
		t.Fatalf("second commandFromPrediction returned error: %v", err)
	}
	if secondCommand.Throttle <= 0 {
		t.Fatalf("expected second throttle > 0 while intent stays latched, got=%f", secondCommand.Throttle)
	}
	if !second.MoveIntentActive {
		t.Fatal("expected move intent to remain active inside hysteresis band")
	}
}

func TestCommandFromPredictionDropsLowSpeedThrottleAfterMoveIntentTurnsOff(t *testing.T) {
	cfg := DefaultInferenceConfig()
	inferencer := NewInferencer(cfg, nil)

	_, err := inferencer.commandFromPrediction(&InferencePrediction{
		FutureYawDelta:   0.0,
		FutureSpeed:      4.0,
		CurrentSpeed:     0.0,
		CurrentYaw:       0.0,
		ControlSemantics: "target_speed",
		MoveIntentProb:   0.9,
		HasMoveIntent:    true,
	})
	if err != nil {
		t.Fatalf("warmup commandFromPrediction returned error: %v", err)
	}

	prediction := &InferencePrediction{
		FutureYawDelta:   0.0,
		FutureSpeed:      3.0,
		CurrentSpeed:     1.0,
		CurrentYaw:       0.0,
		ControlSemantics: "target_speed",
		MoveIntentProb:   0.2,
		HasMoveIntent:    true,
	}
	command, err := inferencer.commandFromPrediction(prediction)
	if err != nil {
		t.Fatalf("commandFromPrediction returned error: %v", err)
	}
	if command.Throttle != 0 {
		t.Fatalf("expected throttle to be gated off after move intent turns off, got=%f", command.Throttle)
	}
	if prediction.MoveIntentActive {
		t.Fatal("expected move intent to be inactive below off threshold")
	}
}

func TestCommandFromPredictionDoesNotGateThrottleAboveMoveIntentHoldSpeed(t *testing.T) {
	cfg := DefaultInferenceConfig()
	inferencer := NewInferencer(cfg, nil)
	prediction := &InferencePrediction{
		FutureYawDelta:   0.0,
		FutureSpeed:      6.5,
		CurrentSpeed:     cfg.MoveIntentHoldSpeedMax + 0.5,
		CurrentYaw:       0.0,
		ControlSemantics: "target_speed",
		MoveIntentProb:   0.1,
		HasMoveIntent:    true,
	}
	command, err := inferencer.commandFromPrediction(prediction)
	if err != nil {
		t.Fatalf("commandFromPrediction returned error: %v", err)
	}
	if command.Throttle <= 0 {
		t.Fatalf("expected throttle above hold-speed ceiling even with low move intent, got=%f", command.Throttle)
	}
}

func TestCommandFromPredictionCapsFutureSpeedTargetAtMaxTargetSpeed(t *testing.T) {
	cfg := DefaultInferenceConfig()
	inferencer := NewInferencer(cfg, nil)
	prediction := &InferencePrediction{
		FutureYawDelta:   0.0,
		FutureSpeed:      10.0,
		CurrentSpeed:     4.0,
		CurrentYaw:       0.0,
		ControlSemantics: "target_speed",
	}

	command, err := inferencer.commandFromPrediction(prediction)
	if err != nil {
		t.Fatalf("commandFromPrediction returned error: %v", err)
	}
	wantCap := cfg.MaxTargetSpeedKPH / 3.6
	if math.Abs(prediction.TargetSpeed-wantCap) > 1e-9 {
		t.Fatalf("expected target speed to be capped at %f m/s, got=%f", wantCap, prediction.TargetSpeed)
	}
	if command.Throttle <= 0 {
		t.Fatalf("expected capped target to still allow throttle below the cap, got=%f", command.Throttle)
	}
}

func TestCommandFromPredictionSuppressesPositiveDeltaSpeedAboveMaxTargetSpeed(t *testing.T) {
	cfg := DefaultInferenceConfig()
	inferencer := NewInferencer(cfg, nil)
	prediction := &InferencePrediction{
		FutureYawDelta:   0.0,
		DeltaSpeed:       0.5,
		CurrentSpeed:     cfg.MaxTargetSpeedKPH/3.6 + 0.2,
		CurrentYaw:       0.0,
		ControlSemantics: "speed_delta",
	}

	command, err := inferencer.commandFromPrediction(prediction)
	if err != nil {
		t.Fatalf("commandFromPrediction returned error: %v", err)
	}
	if command.Throttle != 0 {
		t.Fatalf("expected positive throttle to be suppressed above max target speed, got=%f", command.Throttle)
	}
}

func TestPreviewSteerCommandIsStrongerAtLowSpeedThanHighSpeed(t *testing.T) {
	cfg := DefaultInferenceConfig()
	cfg.HeadingErrorDeadbandDeg = 0
	cfg.HeadingErrorFullLockDeg = 40
	inferencer := NewInferencer(cfg, nil)

	lowSpeedSteer := inferencer.previewSteerCommand(&InferencePrediction{
		FutureYawDelta: 9.0,
		CurrentSpeed:   0.0,
	})
	highSpeedSteer := inferencer.previewSteerCommand(&InferencePrediction{
		FutureYawDelta: 9.0,
		CurrentSpeed:   cfg.SteerGainFadeSpeedMPS,
	})

	if lowSpeedSteer <= highSpeedSteer {
		t.Fatalf("expected low-speed steer to exceed high-speed steer, low=%f high=%f", lowSpeedSteer, highSpeedSteer)
	}
}

func TestShapeSteerCommandBlendsTowardNewTurnDemand(t *testing.T) {
	cfg := DefaultInferenceConfig()
	cfg.SteerResponseBlend = 0.25
	cfg.SteerCommandRatePerSec = 100.0
	cfg.FPS = 20
	cfg.DispatchStride = 1
	inferencer := NewInferencer(cfg, nil)

	first := inferencer.shapeSteerCommand(1.0)
	if first != 1.0 {
		t.Fatalf("expected first shaped steer to initialize at 1.0, got=%f", first)
	}
	second := inferencer.shapeSteerCommand(-1.0)
	if math.Abs(second-0.5) > 1e-9 {
		t.Fatalf("expected blended steer to settle at 0.5, got=%f", second)
	}
}

func TestCommandFromPredictionRateLimitsSteerCommandAcrossTicks(t *testing.T) {
	cfg := DefaultInferenceConfig()
	cfg.FPS = 20
	cfg.DispatchStride = 2
	cfg.HeadingErrorDeadbandDeg = 0
	cfg.HeadingErrorFullLockDeg = 10
	cfg.SteerCommandRatePerSec = 2.0
	inferencer := NewInferencer(cfg, nil)

	firstPrediction := &InferencePrediction{
		FutureYawDelta:   10.0,
		DeltaSpeed:       0.0,
		CurrentYaw:       0.0,
		CurrentSpeed:     3.0,
		ControlSemantics: "speed_delta",
	}
	firstCommand, err := inferencer.commandFromPrediction(firstPrediction)
	if err != nil {
		t.Fatalf("first commandFromPrediction returned error: %v", err)
	}
	if math.Abs(firstCommand.Steer-0.99) > 1e-9 {
		t.Fatalf("expected first steer to initialize near 0.99, got=%f", firstCommand.Steer)
	}

	secondPrediction := &InferencePrediction{
		FutureYawDelta:   -10.0,
		DeltaSpeed:       0.0,
		CurrentYaw:       0.0,
		CurrentSpeed:     3.0,
		ControlSemantics: "speed_delta",
	}
	secondCommand, err := inferencer.commandFromPrediction(secondPrediction)
	if err != nil {
		t.Fatalf("second commandFromPrediction returned error: %v", err)
	}
	if math.Abs(secondCommand.Steer-0.79) > 1e-9 {
		t.Fatalf("expected steer rate limiting to step down to 0.79, got=%f", secondCommand.Steer)
	}
}

func TestCommandFromPredictionFallsBackToDeltaSpeed(t *testing.T) {
	cfg := DefaultInferenceConfig()
	inferencer := NewInferencer(cfg, nil)
	prediction := &InferencePrediction{
		FutureYawDelta:   -20.0,
		FutureYaw:        350.0,
		DeltaSpeed:       -0.5,
		CurrentSpeed:     4.0,
		CurrentYaw:       10.0,
		ControlSemantics: "speed_delta",
	}

	command, err := inferencer.commandFromPrediction(prediction)
	if err != nil {
		t.Fatalf("commandFromPrediction returned error: %v", err)
	}
	if command.Brake <= 0 {
		t.Fatalf("expected brake > 0, got=%f", command.Brake)
	}
	if command.Throttle != 0 {
		t.Fatalf("expected throttle=0, got=%f", command.Throttle)
	}
	if prediction.LongitudinalMode != "delta_speed" {
		t.Fatalf("unexpected longitudinal mode: %s", prediction.LongitudinalMode)
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
	store.UpdateTelemetry(control.TelemetryUpdate{CurrentSpeed: 1.0, CurrentYaw: 5.0, TimestampMs: 123})
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
	if !strings.Contains(err.Error(), "missing future_yaw_delta output") {
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
	if !strings.Contains(err.Error(), "control telemetry is unavailable") {
		t.Fatalf("unexpected error: %v", err)
	}
}
