package capture

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"image"
	"io"
	"math"
	"net/http"
	"os/exec"
	"reflect"
	"strings"
	"testing"
	"time"

	"awesomeProject/internal/actuator"
	"awesomeProject/internal/control"
	"awesomeProject/internal/stopsigncontrol"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func inferenceJSONResponse(statusCode int, payload any) *http.Response {
	body, _ := json.Marshal(payload)
	return &http.Response{
		StatusCode: statusCode,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(bytes.NewReader(body)),
	}
}

type recordingInferenceActuator struct {
	commands      []actuator.CommandRequest
	plans         []stopsigncontrol.Plan
	err           error
	state         actuator.State
	nextCommandID int64
	skipApply     bool
	applyFault    string
	onSubmit      func(actuator.CommandRequest)
}

type statefulInferenceActuator struct {
	recordingInferenceActuator
	state actuator.State
}

const testStopSignModelHash int64 = 424242

func (a *statefulInferenceActuator) State() actuator.State {
	state := a.recordingInferenceActuator.State()
	state.Supported = a.state.Supported
	state.Ready = a.state.Ready
	state.Platform = a.state.Platform
	state.StopSignController = a.state.StopSignController
	if a.state.LastError != "" {
		state.LastError = a.state.LastError
	}
	if a.state.LastApplyError != "" {
		state.LastApplyError = a.state.LastApplyError
	}
	return state
}

func (a *recordingInferenceActuator) Submit(req actuator.CommandRequest) (actuator.State, error) {
	a.commands = append(a.commands, req)
	if a.onSubmit != nil {
		a.onSubmit(req)
	}
	if a.err != nil {
		return a.state, a.err
	}
	a.nextCommandID++
	a.state.Supported = true
	a.state.Ready = true
	a.state.Platform = "windows"
	a.state.LastCommandID = a.nextCommandID
	if a.skipApply {
		return a.state, nil
	}
	a.state.LastApplyAttemptedCommandID = a.nextCommandID
	if a.applyFault != "" {
		a.state.LastApplyError = a.applyFault
		return a.state, nil
	}
	appliedAt := time.UnixMilli(req.TimestampMs).UTC()
	if req.TimestampMs == 0 {
		appliedAt = time.Unix(1, 0).UTC()
	}
	enabled := req.Enabled == nil || *req.Enabled
	if req.Owner == actuator.OwnerStopSignInference && enabled {
		a.state.StopSignController.Owner = actuator.OwnerStopSignInference
		a.state.StopSignController.Stopping = false
	}
	if !enabled {
		a.state.StopSignController.Owner = ""
		a.state.StopSignController.Stopping = false
	}
	a.state.Applied.CommandID = a.nextCommandID
	a.state.Applied.Enabled = enabled
	a.state.Applied.Handbrake = req.Handbrake || req.Owner == actuator.OwnerStopSignInference && enabled
	a.state.Applied.Steer = req.Steer
	a.state.Applied.Throttle = req.Throttle
	a.state.Applied.Brake = req.BrakePressureAvg
	a.state.LastApplyError = ""
	a.state.LastApplySucceededAt = appliedAt.Format(time.RFC3339Nano)
	return a.state, nil
}

func (a *recordingInferenceActuator) RequestStopSignSafetyStop() (actuator.State, error) {
	enabled := true
	state, err := a.Submit(actuator.CommandRequest{
		Enabled:   &enabled,
		InputMode: actuator.InputModeNormalized,
		Owner:     actuator.OwnerStopSignInference,
	})
	if err != nil {
		return state, err
	}
	a.state.StopSignController.Owner = actuator.OwnerStopSignInference
	a.state.StopSignController.Stopping = true
	return a.state, nil
}

func (a *recordingInferenceActuator) State() actuator.State {
	return a.state
}

func (a *recordingInferenceActuator) SubmitStopSignMotionPlan(plan stopsigncontrol.Plan) (actuator.State, error) {
	a.plans = append(a.plans, plan)
	if a.err != nil {
		return a.state, a.err
	}
	a.state.StopSignController.LastPlanID++
	return a.state, nil
}

func assertStopSignSafetyStopRequested(t *testing.T, actuatorSink *recordingInferenceActuator) {
	t.Helper()
	if len(actuatorSink.commands) != 1 {
		t.Fatalf("expected exactly one actuator-owned stopSign safety stop, got=%+v", actuatorSink.commands)
	}
	command := actuatorSink.commands[0]
	if command.Enabled == nil || !*command.Enabled || command.Owner != actuator.OwnerStopSignInference ||
		command.Steer != 0 || command.Throttle != 0 || command.BrakePressureAvg != 0 || command.Handbrake {
		t.Fatalf("expected a neutral enabled ownership command with no direct handbrake, got=%+v", command)
	}
	if !actuatorSink.state.StopSignController.Stopping {
		t.Fatalf("expected the actuator-owned speed-aware stop state, got=%+v", actuatorSink.state.StopSignController)
	}
}

func floatPtr(value float64) *float64 {
	return &value
}

func validStopSignTelemetryUpdate(now time.Time, longitudinal float64) control.TelemetryUpdate {
	zero := 0.0
	onGround := true
	egoStop := &control.StopSignPose{X: 0, Y: 0, Z: 0, Heading: 0}
	return control.TelemetryUpdate{
		VehicleExists:              true,
		IsInVehicle:                true,
		VehicleModelHash:           testStopSignModelHash,
		PositionX:                  &zero,
		PositionY:                  floatPtr(longitudinal),
		PositionZ:                  &zero,
		VelocityX:                  &zero,
		VelocityY:                  &zero,
		VelocityZ:                  &zero,
		PitchDeg:                   &zero,
		RollDeg:                    &zero,
		OnGround:                   &onGround,
		StopSignTargetConfigured:   true,
		StopSignEgoStopPose:        egoStop,
		StopSignLongitudinalErrorM: longitudinal,
		StopSignPhase:              control.StopSignPhaseAccelerate,
		TimestampMs:                now.UnixMilli(),
	}
}

func bindInferenceToCurrentStopSignTarget(t *testing.T, inferencer *Inferencer, store *control.Store) {
	t.Helper()
	telemetry, _ := store.LatestTelemetrySnapshot()
	if telemetry == nil {
		t.Fatal("expected stopSign telemetry")
	}
	target, err := stopSignTargetFromTelemetry(*telemetry)
	if err != nil {
		t.Fatalf("derive stopSign target: %v", err)
	}
	inferencer.stopSignTarget = &target
}

func validStopSignRuntimeTelemetry() control.RuntimeTelemetry {
	zero := 0.0
	onGround := true
	return control.RuntimeTelemetry{
		CurrentYaw:    0,
		VelocityX:     &zero,
		VelocityY:     &zero,
		PitchDeg:      &zero,
		RollDeg:       &zero,
		OnGround:      &onGround,
		StopSignPhase: control.StopSignPhaseAccelerate,
	}
}

func validStopSignModelStatus(cfg InferenceConfig) stopSignModelStatus {
	inputs := make(map[string]stopSignModelInputSpec, len(requiredStopSignStateInputs))
	for _, name := range requiredStopSignStateInputs {
		inputs[name] = stopSignModelInputSpec{Enabled: true}
	}
	return stopSignModelStatus{
		Loaded:                    true,
		Checkpoint:                "C:/models/stopSign/epoch-001.pt",
		PlannerFormat:             cfg.PlannerFormat,
		PlannerFormatVersion:      requiredStopSignPlannerFormatVersion,
		ControlContract:           validStopSignControlContract(),
		ControlHorizonDtMs:        append([]int(nil), cfg.ControlHorizonDtMs...),
		TelemetrySampleIntervalMs: int(cfg.TelemetrySampleInterval.Milliseconds()),
		Direction:                 stopsigncontrol.StopSignDirectionForward,
		ImageSize:                 stopSignModelImageSize{Width: cfg.FrameWidth, Height: cfg.FrameHeight},
		FrameWindow: stopSignModelFrameWindow{
			Size:          cfg.WindowSize,
			FrameStride:   cfg.FrameStride,
			InputChannels: cfg.WindowSize * 3,
		},
		ImageOffsets:       append([]int(nil), cfg.ImageOffsets...),
		TelemetryOffsets:   append([]int(nil), cfg.TelemetryOffsets...),
		FutureOffsets:      append([]int(nil), cfg.FutureOffsets...),
		TelemetryFeatures:  append([]string(nil), cfg.TelemetryFeatureNames...),
		ControlTargetNames: append([]string(nil), cfg.ControlOutputNames...),
		AuxTargetNames:     append([]string(nil), cfg.AuxOutputNames...),
		StateInputs:        inputs,
	}
}

func validStopSignControlContract() stopSignControlContract {
	return stopSignControlContract{
		Name:      stopsigncontrol.StopSignMotionPlanContractV1,
		Version:   1,
		Direction: stopsigncontrol.StopSignDirectionForward,
		Targets:   append([]string(nil), requiredStopSignControlHeads...),
		OutputActivations: map[string]string{
			"future_speed_mps": "sigmoid",
			"stop_intent":      "sigmoid",
		},
		OutputRanges: map[string][]float64{
			"future_speed_mps": {0, stopsigncontrol.StopSignMotionPlanMaxSpeedMPS},
			"stop_intent":      {0, 1},
		},
	}
}

func validStopSignPredictResponse(cfg InferenceConfig, checkpoint string, sampledAtS float64) pythonPredictResponse {
	controls := make([][]float64, cfg.FutureSteps)
	for index := range controls {
		controls[index] = []float64{1.0, 0.05}
	}
	return pythonPredictResponse{
		Checkpoint:                checkpoint,
		Device:                    "cuda",
		PlannerFormat:             cfg.PlannerFormat,
		PlannerFormatVersion:      requiredStopSignPlannerFormatVersion,
		ControlContract:           validStopSignControlContract(),
		ControlHorizonDtMs:        append([]int(nil), cfg.ControlHorizonDtMs...),
		TelemetrySampleIntervalMs: int(cfg.TelemetrySampleInterval.Milliseconds()),
		SampledAtS:                sampledAtS,
		Direction:                 stopsigncontrol.StopSignDirectionForward,
		ImageOffsets:              append([]int(nil), cfg.ImageOffsets...),
		TelemetryOffsets:          append([]int(nil), cfg.TelemetryOffsets...),
		FutureOffsets:             append([]int(nil), cfg.FutureOffsets...),
		TelemetryFeatureNames:     append([]string(nil), cfg.TelemetryFeatureNames...),
		ControlTargetNames:        append([]string(nil), cfg.ControlOutputNames...),
		AuxTargetNames:            append([]string(nil), cfg.AuxOutputNames...),
		StateInputs:               validStopSignModelStatus(cfg).StateInputs,
		PredControls:              [][][]float64{controls},
	}
}

func readyStopSignActuatorState(cfg InferenceConfig) actuator.State {
	return actuator.State{
		Supported: true,
		Ready:     true,
		Platform:  "windows",
		StopSignController: actuator.StopSignControllerState{
			Ready:               true,
			Contract:            cfg.ControlContract,
			ExpectedHorizonDtMs: append([]int(nil), cfg.ControlHorizonDtMs...),
			Calibration: actuator.StopSignCalibration{
				Verified:         true,
				VehicleModelHash: testStopSignModelHash,
			},
		},
	}
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

func TestResolveInferenceMonitorAutoSelectsFiveMWindowMonitor(t *testing.T) {
	sources := []Source{
		{ID: "monitor-1", CaptureType: "monitor", OffsetX: 0, OffsetY: 0, Width: 1920, Height: 1080},
		{ID: "monitor-2", CaptureType: "monitor", OffsetX: 1920, OffsetY: 0, Width: 1920, Height: 1080},
		{ID: "window-fivem", Name: "FiveM by Cfx.re", CaptureType: "window", OffsetX: 2140, OffsetY: 90, Width: 1280, Height: 720},
	}

	monitor, ok := resolveInferenceMonitor(sources, "auto")
	if !ok {
		t.Fatal("expected an auto-selected inference monitor")
	}
	if monitor.ID != "monitor-2" {
		t.Fatalf("unexpected monitor: got=%s want=monitor-2", monitor.ID)
	}

	explicit, ok := resolveInferenceMonitor(sources, "monitor-1")
	if !ok || explicit.ID != "monitor-1" {
		t.Fatalf("explicit source selection changed: %+v ok=%t", explicit, ok)
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
		if arg == "fps=30,showinfo=checksum=0,hwdownload,format=bgra,scale=480:480:flags=lanczos,format=rgb24" {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected configured inference filter, args=%v", args)
	}
	if !strings.Contains(strings.Join(args, " "), "-loglevel info -nostats") {
		t.Fatalf("expected showinfo log output without progress stats, args=%v", args)
	}
}

func TestParseInferenceFrameTimingReadsShowinfoPTS(t *testing.T) {
	observedAt := time.Date(2026, 8, 5, 12, 0, 0, 0, time.UTC)
	line := "[Parsed_showinfo_5 @ 000001] n:  12 pts:     20 pts_time:0.666667 duration:1"
	timing, matched, err := parseInferenceFrameTiming(line, observedAt)
	if err != nil {
		t.Fatalf("parse timing: %v", err)
	}
	if !matched || timing.index != 12 || timing.pts != 666667*time.Microsecond || !timing.observedAt.Equal(observedAt) {
		t.Fatalf("unexpected timing: matched=%v timing=%+v", matched, timing)
	}

	if _, matched, err := parseInferenceFrameTiming("[Parsed_showinfo_5] config in time_base: 1/30", observedAt); matched || err != nil {
		t.Fatalf("expected showinfo configuration line to be ignored, matched=%v err=%v", matched, err)
	}
	if _, matched, err := parseInferenceFrameTiming("[Parsed_showinfo_5] n: bad pts: 0 pts_time:NOPTS", observedAt); !matched || err == nil {
		t.Fatalf("expected malformed frame timing to fail closed, matched=%v err=%v", matched, err)
	}
}

func TestInferenceFrameClockUsesPTSInsteadOfMetadataArrival(t *testing.T) {
	clock := newInferenceFrameClock(30)
	anchor := time.Date(2026, 8, 5, 12, 0, 0, 0, time.UTC)
	first, err := clock.resolve(0, inferenceFrameTiming{index: 0, pts: 250 * time.Millisecond, observedAt: anchor})
	if err != nil {
		t.Fatalf("resolve first frame: %v", err)
	}
	if !first.Equal(anchor) {
		t.Fatalf("unexpected first timestamp: got=%s want=%s", first, anchor)
	}

	lateArrival := anchor.Add(800 * time.Millisecond)
	second, err := clock.resolve(1, inferenceFrameTiming{
		index:      1,
		pts:        283333333 * time.Nanosecond,
		observedAt: lateArrival,
	})
	if err != nil {
		t.Fatalf("resolve delayed second frame: %v", err)
	}
	want := anchor.Add(33333333 * time.Nanosecond)
	if !second.Equal(want) {
		t.Fatalf("expected PTS-derived timestamp, got=%s want=%s", second, want)
	}
	if second.Equal(lateArrival) {
		t.Fatal("frame timestamp must not use delayed metadata or stdout arrival time")
	}
}

func TestInferenceFrameClockRejectsMismatchedMetadata(t *testing.T) {
	anchor := time.Date(2026, 8, 5, 12, 0, 0, 0, time.UTC)
	tests := []struct {
		name   string
		timing inferenceFrameTiming
		want   string
	}{
		{
			name:   "missing frame index",
			timing: inferenceFrameTiming{index: 2, pts: 33333333 * time.Nanosecond, observedAt: anchor.Add(time.Second)},
			want:   "index mismatch",
		},
		{
			name:   "non-increasing pts",
			timing: inferenceFrameTiming{index: 1, pts: 0, observedAt: anchor.Add(time.Second)},
			want:   "not increasing",
		},
		{
			name:   "wrong cadence",
			timing: inferenceFrameTiming{index: 1, pts: 100 * time.Millisecond, observedAt: anchor.Add(time.Second)},
			want:   "cadence mismatch",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			clock := newInferenceFrameClock(30)
			if _, err := clock.resolve(0, inferenceFrameTiming{index: 0, pts: 0, observedAt: anchor}); err != nil {
				t.Fatalf("resolve first frame: %v", err)
			}
			if _, err := clock.resolve(1, tc.timing); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("expected %q error, got=%v", tc.want, err)
			}
		})
	}
}

func TestConsumeInferenceStderrPublishesOrderedFrameTiming(t *testing.T) {
	inferencer := NewInferencer(DefaultInferenceConfig(), actuator.DefaultConfig(), nil)
	observedAt := time.Date(2026, 8, 5, 12, 0, 0, 0, time.UTC)
	inferencer.nowFunc = func() time.Time { return observedAt }
	session := &inferenceSession{
		ctx: context.Background(),
		stderr: io.NopCloser(strings.NewReader(strings.Join([]string{
			"Input #0, lavfi, from 'ddagrab':",
			"[Parsed_showinfo_5] n: 0 pts: 0 pts_time:0 duration:1",
			"[Parsed_showinfo_5] n: 1 pts: 1 pts_time:0.0333333 duration:1",
		}, "\n"))),
		frameTimings: make(chan inferenceFrameTimingEvent, 2),
	}

	inferencer.consumeInferenceStderr(session)
	first, ok := <-session.frameTimings
	if !ok || first.err != nil || first.timing.index != 0 {
		t.Fatalf("unexpected first timing event: ok=%v event=%+v", ok, first)
	}
	second, ok := <-session.frameTimings
	if !ok || second.err != nil || second.timing.index != 1 {
		t.Fatalf("unexpected second timing event: ok=%v event=%+v", ok, second)
	}
	if _, ok := <-session.frameTimings; ok {
		t.Fatal("expected timing stream to close with FFmpeg stderr")
	}
}

func TestCompleteInferenceFrameWithoutTimingMetadataTripsSafetyStop(t *testing.T) {
	cfg := DefaultInferenceConfig()
	cfg.FrameWidth = 1
	cfg.FrameHeight = 1
	actuatorSink := &recordingInferenceActuator{}
	inferencer := NewInferencer(cfg, actuator.DefaultConfig(), nil, actuatorSink)
	loopCtx, cancel := context.WithCancel(context.Background())
	timingStream := make(chan inferenceFrameTimingEvent)
	close(timingStream)
	session := &inferenceSession{
		ctx:          loopCtx,
		cancel:       cancel,
		stdout:       io.NopCloser(bytes.NewReader([]byte{0, 0, 0})),
		frameTimings: timingStream,
	}
	inferencer.active = session
	inferencer.status.State = "running"
	inferencer.status.Active = true

	inferencer.consumeInferenceFrames(loopCtx, session)
	status := inferencer.Status()
	if status.State != "error" || !strings.Contains(status.LastError, "timing metadata ended before frame 0") {
		t.Fatalf("expected missing timing metadata to be terminal, got=%+v", status)
	}
	if len(actuatorSink.commands) != 1 || !actuatorSink.state.StopSignController.Stopping {
		t.Fatalf("expected missing timing metadata to issue one stopSign safety stop, commands=%+v state=%+v", actuatorSink.commands, actuatorSink.state)
	}
}

func TestInferenceStartRejectsUnsafeStopSignStateBeforeCaptureSetup(t *testing.T) {
	now := time.Date(2026, 8, 5, 12, 0, 0, 0, time.UTC)
	store := control.NewStore(control.WithNowFunc(func() time.Time { return now }))
	store.UpdateTelemetry(control.TelemetryUpdate{
		VehicleExists:              true,
		IsInVehicle:                true,
		StopSignTargetConfigured:   false,
		StopSignLongitudinalErrorM: -12,
		TimestampMs:                now.UnixMilli(),
	})
	inferencer := NewInferencer(DefaultInferenceConfig(), actuator.DefaultConfig(), store)
	inferencer.nowFunc = func() time.Time { return now }
	discoveryCalled := false
	inferencer.discover = func(context.Context) ([]Source, error) {
		discoveryCalled = true
		return nil, nil
	}

	_, err := inferencer.Start(t.Context(), InferenceStartRequest{})
	if err == nil || !errors.Is(err, ErrInferenceStartFailed) || !errors.Is(err, ErrStopSignInferencePrecondition) || !strings.Contains(err.Error(), "stop-sign target is not configured") {
		t.Fatalf("expected useful stop-sign precondition error, got=%v", err)
	}
	if discoveryCalled {
		t.Fatal("expected stopSign preconditions to fail before capture source discovery")
	}
}

func TestStopSignInferenceRequiresEgoStopPose(t *testing.T) {
	now := time.Date(2026, 8, 5, 12, 0, 0, 0, time.UTC)
	store := control.NewStore(control.WithNowFunc(func() time.Time { return now }))
	update := validStopSignTelemetryUpdate(now, -12)
	update.StopSignEgoStopPose = nil
	store.UpdateTelemetry(update)
	inferencer := NewInferencer(DefaultInferenceConfig(), actuator.DefaultConfig(), store)
	inferencer.nowFunc = func() time.Time { return now }

	err := inferencer.validateStopSignInferenceStart()
	if err == nil || !errors.Is(err, ErrStopSignInferencePrecondition) || !strings.Contains(err.Error(), "stop-sign target is not configured") {
		t.Fatalf("expected missing ego stop pose precondition, got=%v", err)
	}
}

func TestInferenceLifecycleRejectsModelChangesWhileRunning(t *testing.T) {
	inferencer := NewInferencer(DefaultInferenceConfig(), actuator.DefaultConfig(), nil)
	inferencer.active = &inferenceSession{}
	httpCalled := false
	inferencer.httpClient = &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		httpCalled = true
		return nil, errors.New("unexpected request")
	})}
	discoveryCalled := false
	inferencer.discover = func(context.Context) ([]Source, error) {
		discoveryCalled = true
		return nil, nil
	}

	if _, err := inferencer.Start(t.Context(), InferenceStartRequest{}); !errors.Is(err, ErrInferenceAlreadyRunning) {
		t.Fatalf("expected second Start to reject before side effects, got=%v", err)
	}
	if _, err := inferencer.LoadModel(t.Context(), InferenceModelLoadRequest{Checkpoint: "replacement.pt"}); !errors.Is(err, ErrInferenceAlreadyRunning) {
		t.Fatalf("expected model load to reject while inference is active, got=%v", err)
	}
	if discoveryCalled || httpCalled {
		t.Fatalf("expected no discovery/model HTTP side effects, discovery=%v http=%v", discoveryCalled, httpCalled)
	}
}

func TestInferenceStatusPreservesLoadedModelAcrossRefreshes(t *testing.T) {
	inferencer := NewInferencer(DefaultInferenceConfig(), actuator.DefaultConfig(), nil)
	inferencer.loadedCheckpoint = `S:\models\stopSign\epoch-012.pt`
	inferencer.loadedModelDevice = "cuda"

	status := inferencer.Status()
	if status.LoadedCheckpoint != inferencer.loadedCheckpoint || status.LoadedModelDevice != "cuda" {
		t.Fatalf("loaded model was not exposed in inference status: %+v", status)
	}
}

func TestInferenceStatusReportsAuthoritativeCalibrationPreflight(t *testing.T) {
	now := time.Date(2026, 8, 9, 12, 0, 0, 0, time.UTC)
	store := control.NewStore(control.WithNowFunc(func() time.Time { return now }))
	store.UpdateTelemetry(validStopSignTelemetryUpdate(now, -12))
	cfg := DefaultInferenceConfig()
	unverified := readyStopSignActuatorState(cfg)
	unverified.StopSignController.Ready = false
	unverified.StopSignController.Calibration.Verified = false
	actuatorSink := &statefulInferenceActuator{state: unverified}
	inferencer := NewInferencer(cfg, actuator.DefaultConfig(), store, actuatorSink)
	inferencer.nowFunc = func() time.Time { return now }

	blocked := inferencer.Status()
	if !blocked.ActuatorReady || blocked.ControllerReady || blocked.CalibrationVerified || blocked.SafetyReady {
		t.Fatalf("expected fail-closed unverified calibration preflight, got=%+v", blocked)
	}
	if !strings.Contains(blocked.SafetyBlocker, "verified vehicle calibration") {
		t.Fatalf("expected actionable calibration blocker, got=%q", blocked.SafetyBlocker)
	}

	verified := readyStopSignActuatorState(cfg)
	verified.StopSignController.Calibration.ProfileID = "test-vehicle-v1"
	actuatorSink.state = verified
	ready := inferencer.Status()
	if !ready.ActuatorReady || !ready.ControllerReady || !ready.CalibrationVerified || !ready.SafetyReady {
		t.Fatalf("expected every local inference preflight to pass, got=%+v", ready)
	}
	if ready.CalibrationID != "test-vehicle-v1" || ready.SafetyBlocker != "" {
		t.Fatalf("unexpected verified preflight detail: %+v", ready)
	}
}

func TestLoadModelVerifiesStopSignCompatibilityBeforeClaimingReady(t *testing.T) {
	cfg := DefaultInferenceConfig()
	cfg.ModelServerURL = "http://planner.local"
	modelStatus := validStopSignModelStatus(cfg)
	modelStatus.Device = "cuda:0"
	requests := make([]string, 0, 2)
	inferencer := NewInferencer(cfg, actuator.DefaultConfig(), nil)
	inferencer.httpClient = &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		requests = append(requests, req.Method+" "+req.URL.Path)
		switch req.URL.Path {
		case "/model/load":
			return inferenceJSONResponse(http.StatusOK, map[string]any{
				"status":     "loaded",
				"checkpoint": modelStatus.Checkpoint,
				"device":     modelStatus.Device,
			}), nil
		case "/model":
			return inferenceJSONResponse(http.StatusOK, modelStatus), nil
		default:
			return inferenceJSONResponse(http.StatusNotFound, map[string]string{"error": "not found"}), nil
		}
	})}

	response, err := inferencer.LoadModel(t.Context(), InferenceModelLoadRequest{Checkpoint: modelStatus.Checkpoint})
	if err != nil {
		t.Fatalf("LoadModel returned error: %v", err)
	}
	if !reflect.DeepEqual(requests, []string{"POST /model/load", "GET /model"}) {
		t.Fatalf("model load was not followed by compatibility verification: %v", requests)
	}
	if response["checkpoint"] != modelStatus.Checkpoint || response["device"] != modelStatus.Device {
		t.Fatalf("unexpected authoritative model response: %+v", response)
	}
	status := inferencer.Status()
	if status.LoadedCheckpoint != modelStatus.Checkpoint || status.LoadedModelDevice != modelStatus.Device {
		t.Fatalf("verified model was not retained in status: %+v", status)
	}
}

func TestLoadModelClearsStaleReadyStateWhenCompatibilityFails(t *testing.T) {
	cfg := DefaultInferenceConfig()
	cfg.ModelServerURL = "http://planner.local"
	incompatible := validStopSignModelStatus(cfg)
	incompatible.Direction = "reverse"
	inferencer := NewInferencer(cfg, actuator.DefaultConfig(), nil)
	inferencer.loadedCheckpoint = "C:/models/old.pt"
	inferencer.loadedModelDevice = "cuda"
	inferencer.httpClient = &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		if req.URL.Path == "/model/load" {
			return inferenceJSONResponse(http.StatusOK, map[string]any{
				"status":     "loaded",
				"checkpoint": incompatible.Checkpoint,
				"device":     "cuda",
			}), nil
		}
		return inferenceJSONResponse(http.StatusOK, incompatible), nil
	})}

	_, err := inferencer.LoadModel(t.Context(), InferenceModelLoadRequest{Checkpoint: incompatible.Checkpoint})
	if err == nil || !errors.Is(err, ErrStopSignModelIncompatible) {
		t.Fatalf("expected incompatible checkpoint rejection, got=%v", err)
	}
	status := inferencer.Status()
	if status.LoadedCheckpoint != "" || status.LoadedModelDevice != "" {
		t.Fatalf("failed model load left a stale ready checkpoint: %+v", status)
	}
}

func TestInferenceStartRequiresReadyActuatorBeforeSourceDiscovery(t *testing.T) {
	now := time.Date(2026, 8, 5, 12, 0, 0, 0, time.UTC)
	cfg := DefaultInferenceConfig()
	tests := []struct {
		name        string
		state       actuator.State
		wantBlocked bool
	}{
		{
			name:        "unsupported",
			state:       actuator.State{Supported: false, Ready: false, Platform: "linux"},
			wantBlocked: true,
		},
		{
			name:        "not ready",
			state:       actuator.State{Supported: true, Ready: false, Platform: "windows", LastError: "ViGEmBus unavailable"},
			wantBlocked: true,
		},
		{
			name:        "active apply fault",
			state:       actuator.State{Supported: true, Ready: true, Platform: "windows", LastApplyError: "controller write failed"},
			wantBlocked: true,
		},
		{
			name:  "ready",
			state: readyStopSignActuatorState(cfg),
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			store := control.NewStore(control.WithNowFunc(func() time.Time { return now }))
			store.UpdateTelemetry(validStopSignTelemetryUpdate(now, -12))
			actuatorSink := &statefulInferenceActuator{state: tc.state}
			inferencer := NewInferencer(cfg, actuator.DefaultConfig(), store, actuatorSink)
			inferencer.nowFunc = func() time.Time { return now }
			discoveryErr := errors.New("source discovery reached")
			discoveryCalled := false
			inferencer.discover = func(context.Context) ([]Source, error) {
				discoveryCalled = true
				return nil, discoveryErr
			}

			_, err := inferencer.Start(t.Context(), InferenceStartRequest{})
			if tc.wantBlocked {
				if err == nil || !errors.Is(err, ErrInferenceActuatorUnavailable) {
					t.Fatalf("expected actuator readiness error, got=%v", err)
				}
				if discoveryCalled {
					t.Fatal("expected readiness failure before source discovery")
				}
				return
			}
			if !errors.Is(err, discoveryErr) || !discoveryCalled {
				t.Fatalf("expected ready actuator to proceed to source discovery, err=%v called=%v", err, discoveryCalled)
			}
		})
	}
}

func TestInferenceStartCancellationAfterArmingDoesNotSpawnCapture(t *testing.T) {
	now := time.Date(2026, 8, 9, 12, 0, 0, 0, time.UTC)
	cfg := DefaultInferenceConfig()
	cfg.AutoLoad = false
	store := control.NewStore(control.WithNowFunc(func() time.Time { return now }))
	store.UpdateTelemetry(validStopSignTelemetryUpdate(now, -12))
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	actuatorSink := &recordingInferenceActuator{state: readyStopSignActuatorState(cfg)}
	submissions := 0
	actuatorSink.onSubmit = func(actuator.CommandRequest) {
		submissions++
		if submissions == 1 {
			cancel()
		}
	}
	inferencer := NewInferencer(cfg, actuator.DefaultConfig(), store, actuatorSink)
	inferencer.nowFunc = func() time.Time { return now }
	inferencer.discover = func(context.Context) ([]Source, error) {
		return []Source{{ID: "monitor-1", CaptureType: "monitor", Width: 1920, Height: 1080}}, nil
	}
	inferencer.probe = func(context.Context, string, string) (bool, error) {
		return true, nil
	}
	inferencer.httpClient = &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		return inferenceJSONResponse(http.StatusOK, validStopSignModelStatus(cfg)), nil
	})}
	inferencer.newCommand = func(context.Context, string, ...string) *exec.Cmd {
		return exec.Command("definitely-missing-inference-test-command")
	}

	_, err := inferencer.Start(ctx, InferenceStartRequest{})
	if err == nil || !errors.Is(err, ErrInferenceStartFailed) || !errors.Is(err, context.Canceled) {
		t.Fatalf("expected request cancellation before capture spawn, got=%v", err)
	}
	if len(actuatorSink.commands) != 2 {
		t.Fatalf("expected one arm followed by one restored safety hold, got=%+v", actuatorSink.commands)
	}
	if !actuatorSink.state.StopSignController.Stopping {
		t.Fatalf("expected canceled start to restore the stopSign safety hold, got=%+v", actuatorSink.state.StopSignController)
	}
	if inferencer.active != nil {
		t.Fatal("canceled inference start must not install an active capture session")
	}
	if status := inferencer.Status(); status.State != "idle" || status.Active || status.LastError != "" {
		t.Fatalf("canceled inference start must restore its prior idle status, got=%+v", status)
	}
}

func TestStopSignInferenceRejectsActuatorWithDifferentSetpointTiming(t *testing.T) {
	cfg := DefaultInferenceConfig()
	state := readyStopSignActuatorState(cfg)
	state.StopSignController.ExpectedHorizonDtMs = []int{100, 250, 500, 1001}
	actuatorSink := &statefulInferenceActuator{state: state}
	inferencer := NewInferencer(cfg, actuator.DefaultConfig(), nil, actuatorSink)
	if err := inferencer.validateInferenceActuatorReady(); err == nil || !errors.Is(err, ErrInferenceActuatorUnavailable) || !strings.Contains(err.Error(), "actuator control horizon timing") {
		t.Fatalf("expected mismatched setpoint timing to fail closed, got=%v", err)
	}
}

func TestStopSignInferenceRejectsVehicleDifferentFromCalibrationProfile(t *testing.T) {
	now := time.Date(2026, 8, 5, 12, 0, 0, 0, time.UTC)
	cfg := DefaultInferenceConfig()
	store := control.NewStore(control.WithNowFunc(func() time.Time { return now }))
	telemetry := validStopSignTelemetryUpdate(now, -12)
	telemetry.VehicleModelHash = testStopSignModelHash + 1
	store.UpdateTelemetry(telemetry)
	actuatorSink := &statefulInferenceActuator{state: readyStopSignActuatorState(cfg)}
	inferencer := NewInferencer(cfg, actuator.DefaultConfig(), store, actuatorSink)

	err := inferencer.validateInferenceActuatorReady()
	if err == nil || !errors.Is(err, ErrInferenceActuatorUnavailable) || !strings.Contains(err.Error(), "does not match calibrated hash") {
		t.Fatalf("expected current vehicle identity mismatch to fail before arming, got=%v", err)
	}
}

func TestStopSignInferenceRejectsFreshReceiptOfStaleSourceTelemetry(t *testing.T) {
	base := time.Date(2026, 8, 5, 12, 0, 0, 0, time.UTC)
	now := base.Add(defaultTelemetryStaleAfter + time.Millisecond)
	store := control.NewStore(control.WithNowFunc(func() time.Time { return now }))
	store.UpdateTelemetry(validStopSignTelemetryUpdate(base, -12))
	inferencer := NewInferencer(DefaultInferenceConfig(), actuator.DefaultConfig(), store)
	inferencer.nowFunc = func() time.Time { return now }

	err := inferencer.validateStopSignInferenceStart()
	if err == nil || !strings.Contains(err.Error(), "source telemetry is stale") {
		t.Fatalf("expected stale source timestamp to fail despite a fresh backend receipt, got=%v", err)
	}
}

func TestStopSignInferenceArmRequiresAppliedNeutralCommand(t *testing.T) {
	actuatorSink := &recordingInferenceActuator{skipApply: true}
	inferencer := NewInferencer(DefaultInferenceConfig(), actuator.DefaultConfig(), nil, actuatorSink)
	inferencer.actuatorConfirmTimeout = 5 * time.Millisecond
	inferencer.actuatorConfirmInterval = time.Millisecond

	err := inferencer.armActuatorForStopSignInference()
	if err == nil || !strings.Contains(err.Error(), "timed out") {
		t.Fatalf("expected unapplied neutral arm to fail closed, got=%v", err)
	}
	if len(actuatorSink.commands) != 1 {
		t.Fatalf("expected one neutral arm command, got=%+v", actuatorSink.commands)
	}
	command := actuatorSink.commands[0]
	if command.Enabled == nil || !*command.Enabled || command.Handbrake || command.Steer != 0 || command.Throttle != 0 || command.BrakePressureAvg != 0 {
		t.Fatalf("expected enabled neutral arm request, got=%+v", command)
	}
}

func TestValidateStopSignStartEnvelopeMatchesForwardCurriculumBounds(t *testing.T) {
	tests := []struct {
		name      string
		telemetry control.RuntimeTelemetry
		wantError string
	}{
		{
			name: "nearest curriculum boundary",
			telemetry: control.RuntimeTelemetry{
				StopSignLongitudinalErrorM: stopSignStartLongitudinalMaxM,
				StopSignLateralErrorM:      stopSignStartLateralLimitM,
				StopSignHeadingErrorDeg:    stopSignStartHeadingLimitDeg,
			},
		},
		{
			name: "furthest curriculum boundary",
			telemetry: control.RuntimeTelemetry{
				StopSignLongitudinalErrorM: stopSignStartLongitudinalMinM,
				StopSignLateralErrorM:      -stopSignStartLateralLimitM,
				StopSignHeadingErrorDeg:    -stopSignStartHeadingLimitDeg,
			},
		},
		{
			name: "too close",
			telemetry: control.RuntimeTelemetry{
				StopSignLongitudinalErrorM: stopSignStartLongitudinalMaxM + 0.01,
			},
			wantError: "longitudinal offset",
		},
		{
			name: "too far",
			telemetry: control.RuntimeTelemetry{
				StopSignLongitudinalErrorM: stopSignStartLongitudinalMinM - 0.01,
			},
			wantError: "longitudinal offset",
		},
		{
			name: "outside lateral envelope",
			telemetry: control.RuntimeTelemetry{
				StopSignLongitudinalErrorM: -12,
				StopSignLateralErrorM:      stopSignStartLateralLimitM + 0.01,
			},
			wantError: "lateral offset",
		},
		{
			name: "outside heading envelope",
			telemetry: control.RuntimeTelemetry{
				StopSignLongitudinalErrorM: -12,
				StopSignHeadingErrorDeg:    stopSignStartHeadingLimitDeg + 0.01,
			},
			wantError: "heading error",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := validateStopSignStartEnvelope(tc.telemetry)
			if tc.wantError == "" {
				if err != nil {
					t.Fatalf("expected valid curriculum pose, got=%v", err)
				}
				return
			}
			if err == nil || !errors.Is(err, ErrStopSignInferencePrecondition) || !strings.Contains(err.Error(), tc.wantError) {
				t.Fatalf("expected %q precondition error, got=%v", tc.wantError, err)
			}
		})
	}
}

func TestValidateStopSignInferenceStartRequiresFreshValidEgoTelemetry(t *testing.T) {
	now := time.Date(2026, 8, 5, 12, 0, 0, 0, time.UTC)
	invalidStore := control.NewStore(control.WithNowFunc(func() time.Time { return now }))
	invalidUpdate := validStopSignTelemetryUpdate(now, -12)
	invalidUpdate.VehicleExists = false
	invalidUpdate.IsInVehicle = false
	invalidStore.UpdateTelemetry(invalidUpdate)
	invalidInferencer := NewInferencer(DefaultInferenceConfig(), actuator.DefaultConfig(), invalidStore)
	invalidInferencer.nowFunc = func() time.Time { return now }
	if err := invalidInferencer.validateStopSignInferenceStart(); err == nil || !strings.Contains(err.Error(), "vehicle does not exist") {
		t.Fatalf("expected invalid ego telemetry error, got=%v", err)
	}

	validStore := control.NewStore(control.WithNowFunc(func() time.Time { return now }))
	validStore.UpdateTelemetry(validStopSignTelemetryUpdate(now, -12))
	staleInferencer := NewInferencer(DefaultInferenceConfig(), actuator.DefaultConfig(), validStore)
	staleInferencer.nowFunc = func() time.Time { return now.Add(defaultTelemetryStaleAfter + time.Millisecond) }
	if err := staleInferencer.validateStopSignInferenceStart(); err == nil || !strings.Contains(err.Error(), "telemetry is stale") {
		t.Fatalf("expected stale ego telemetry error, got=%v", err)
	}
}

func TestStopSignTargetLossDisablesActuatorWithoutFallbackDecay(t *testing.T) {
	now := time.Date(2026, 8, 5, 12, 0, 0, 0, time.UTC)
	store := control.NewStore(control.WithNowFunc(func() time.Time { return now }))
	store.UpdateTelemetry(validStopSignTelemetryUpdate(now, -12))

	actuatorSink := &recordingInferenceActuator{}
	inferencer := NewInferencer(DefaultInferenceConfig(), actuator.DefaultConfig(), store, actuatorSink)
	inferencer.nowFunc = func() time.Time { return now }
	bindInferenceToCurrentStopSignTarget(t, inferencer, store)

	now = now.Add(50 * time.Millisecond)
	lostTarget := validStopSignTelemetryUpdate(now, -12)
	lostTarget.StopSignTargetConfigured = false
	lostTarget.StopSignEgoStopPose = nil
	store.UpdateTelemetry(lostTarget)
	cause := inferencer.validateActiveStopSignInference()
	if cause == nil || !errors.Is(cause, ErrStopSignInferencePrecondition) {
		t.Fatalf("expected target-loss safety error, got=%v", cause)
	}

	inferencer.handlePredictionFailure(predictionWindow{sequenceNumber: 17}, cause)
	assertStopSignSafetyStopRequested(t, actuatorSink)

	now = now.Add(50 * time.Millisecond)
	store.UpdateTelemetry(validStopSignTelemetryUpdate(now, -12))
	latched := inferencer.validateActiveStopSignInference()
	if latched == nil || !strings.Contains(latched.Error(), "interlock is latched") {
		t.Fatalf("expected restored target to remain latched until restart, got=%v", latched)
	}
	inferencer.handlePredictionFailure(predictionWindow{sequenceNumber: 18}, latched)
	if len(actuatorSink.commands) != 1 {
		t.Fatalf("expected the latched failure to suppress duplicate safety commands, got=%+v", actuatorSink.commands)
	}
}

func TestPredictionErrorHardDisablesInsteadOfDecayingControls(t *testing.T) {
	actuatorSink := &recordingInferenceActuator{}
	inferencer := NewInferencer(DefaultInferenceConfig(), actuator.DefaultConfig(), nil, actuatorSink)
	inferencer.nowFunc = func() time.Time { return time.UnixMilli(2000).UTC() }

	inferencer.handlePredictionFailure(predictionWindow{sequenceNumber: 23}, errors.New("planner request timed out"))
	assertStopSignSafetyStopRequested(t, actuatorSink)
	if !inferencer.stopSignSafetyTripped {
		t.Fatal("expected generic prediction error to latch the stopSign safety interlock")
	}
}

func TestStopSignSuccessNeutralizesDriveAndSurfacesSucceededState(t *testing.T) {
	actuatorSink := &recordingInferenceActuator{}
	inferencer := NewInferencer(DefaultInferenceConfig(), actuator.DefaultConfig(), nil, actuatorSink)
	inferencer.nowFunc = func() time.Time { return time.UnixMilli(3000).UTC() }

	inferencer.handlePredictionFailure(predictionWindow{sequenceNumber: 24}, ErrStopSignInferenceComplete)
	assertStopSignSafetyStopRequested(t, actuatorSink)
	status := inferencer.Status()
	if status.State != "succeeded" || status.LastError != "" || !inferencer.stopSignCompleted {
		t.Fatalf("expected a latched succeeded state, got status=%+v completed=%v", status, inferencer.stopSignCompleted)
	}
	if err := inferencer.validateActiveStopSignInference(); !errors.Is(err, ErrStopSignInferenceComplete) {
		t.Fatalf("expected completed session to remain terminal until restart, got=%v", err)
	}
}

func TestStopSignTerminalTransitionHoldsOnceAndCancelsActiveSession(t *testing.T) {
	actuatorSink := &recordingInferenceActuator{}
	inferencer := NewInferencer(DefaultInferenceConfig(), actuator.DefaultConfig(), nil, actuatorSink)
	inferencer.nowFunc = func() time.Time { return time.UnixMilli(4000).UTC() }
	loopCtx, cancel := context.WithCancel(context.Background())
	session := &inferenceSession{cancel: cancel}
	inferencer.active = session
	inferencer.status.State = "running"
	inferencer.status.Active = true
	cause := errors.New("stopSign target disappeared")

	inferencer.handleSessionPredictionFailure(session, predictionWindow{sequenceNumber: 25}, cause)
	select {
	case <-loopCtx.Done():
	default:
		t.Fatal("expected terminal safety transition to cancel the active session")
	}
	status := inferencer.Status()
	if status.State != "error" || status.LastError != cause.Error() || status.StoppedAt != "" || !status.Active {
		t.Fatalf("expected terminal reason with cleanup still active, got=%+v", status)
	}
	assertStopSignSafetyStopRequested(t, actuatorSink)

	inferencer.handleSessionPredictionFailure(session, predictionWindow{sequenceNumber: 26}, errors.New("duplicate failure"))
	if len(actuatorSink.commands) != 1 || inferencer.Status().LastError != cause.Error() {
		t.Fatalf("expected duplicate failure to preserve the first terminal result, status=%+v commands=%+v", inferencer.Status(), actuatorSink.commands)
	}
}

func TestStopSignSuccessDoesNotMaskSafetyHoldFailureAfterCleanup(t *testing.T) {
	actuatorSink := &recordingInferenceActuator{err: errors.New("controller disconnected")}
	inferencer := NewInferencer(DefaultInferenceConfig(), actuator.DefaultConfig(), nil, actuatorSink)
	loopCtx, cancel := context.WithCancel(context.Background())
	session := &inferenceSession{cancel: cancel}
	inferencer.active = session
	inferencer.status.State = "running"
	inferencer.status.Active = true

	inferencer.handleSessionPredictionFailure(session, predictionWindow{sequenceNumber: 27}, ErrStopSignInferenceComplete)
	select {
	case <-loopCtx.Done():
	default:
		t.Fatal("expected failed success hold to cancel the active session")
	}
	inferencer.finishInferenceSession(session, nil)
	status := inferencer.Status()
	if status.State != "error" || status.Active || !strings.Contains(status.LastError, "controller disconnected") || status.StoppedAt == "" {
		t.Fatalf("expected cleanup to preserve the failed physical hold as an error, got=%+v", status)
	}
}

func TestStopSignSuccessFailsWhenAppliedHoldCannotBeConfirmed(t *testing.T) {
	actuatorSink := &recordingInferenceActuator{
		skipApply:     true,
		nextCommandID: 9,
		state: actuator.State{
			Supported:            true,
			Ready:                true,
			LastApplySucceededAt: time.Unix(1, 0).UTC().Format(time.RFC3339Nano),
		},
	}
	actuatorSink.state.Applied.CommandID = 9
	actuatorSink.state.Applied.Handbrake = true
	inferencer := NewInferencer(DefaultInferenceConfig(), actuator.DefaultConfig(), nil, actuatorSink)
	inferencer.actuatorConfirmTimeout = 5 * time.Millisecond
	inferencer.actuatorConfirmInterval = time.Millisecond

	inferencer.handlePredictionFailure(predictionWindow{sequenceNumber: 28}, ErrStopSignInferenceComplete)
	status := inferencer.Status()
	if status.State != "error" || !strings.Contains(status.LastError, "timed out") || !inferencer.stopSignHoldFailed {
		t.Fatalf("expected an old matching applied state to fail exact-command confirmation, got=%+v", status)
	}
	if len(actuatorSink.commands) != 1 {
		t.Fatalf("expected exactly one submitted hold while confirmation timed out, got=%+v", actuatorSink.commands)
	}
}

func TestStopSignSuccessFailsWhenControllerApplyFailsAsynchronously(t *testing.T) {
	actuatorSink := &recordingInferenceActuator{applyFault: "virtual controller write failed"}
	inferencer := NewInferencer(DefaultInferenceConfig(), actuator.DefaultConfig(), nil, actuatorSink)

	inferencer.handlePredictionFailure(predictionWindow{sequenceNumber: 29}, ErrStopSignInferenceComplete)
	status := inferencer.Status()
	if status.State != "error" || !strings.Contains(status.LastError, "controller apply failed") || !strings.Contains(status.LastError, actuatorSink.applyFault) {
		t.Fatalf("expected asynchronous controller failure to override success, got=%+v", status)
	}
}

func TestLiveFrameStreamTerminationAlwaysLatchesOneErrorHold(t *testing.T) {
	tests := []struct {
		name   string
		stream string
	}{
		{name: "eof"},
		{name: "partial frame", stream: "x"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := DefaultInferenceConfig()
			cfg.FrameWidth = 1
			cfg.FrameHeight = 1
			actuatorSink := &recordingInferenceActuator{}
			inferencer := NewInferencer(cfg, actuator.DefaultConfig(), nil, actuatorSink)
			loopCtx, cancel := context.WithCancel(context.Background())
			session := &inferenceSession{
				cancel: cancel,
				stdout: io.NopCloser(strings.NewReader(tc.stream)),
			}
			inferencer.active = session
			inferencer.status.State = "running"
			inferencer.status.Active = true

			inferencer.consumeInferenceFrames(loopCtx, session)
			inferencer.handleInferenceProcessExit(session, nil)
			inferencer.finishInferenceSession(session, nil)

			status := inferencer.Status()
			if status.State != "error" || status.Active || !strings.Contains(status.LastError, "frame stream ended unexpectedly") {
				t.Fatalf("expected live frame stream termination to remain a terminal error, got=%+v", status)
			}
			assertStopSignSafetyStopRequested(t, actuatorSink)
		})
	}
}

func TestUnexpectedInferenceProcessExitAlwaysLatchesOneErrorHold(t *testing.T) {
	tests := []struct {
		name       string
		processErr error
	}{
		{name: "clean exit"},
		{name: "nonzero exit", processErr: errors.New("exit status 2")},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			actuatorSink := &recordingInferenceActuator{}
			inferencer := NewInferencer(DefaultInferenceConfig(), actuator.DefaultConfig(), nil, actuatorSink)
			loopCtx, cancel := context.WithCancel(context.Background())
			session := &inferenceSession{cancel: cancel}
			inferencer.active = session
			inferencer.status.State = "running"
			inferencer.status.Active = true

			inferencer.handleInferenceProcessExit(session, tc.processErr)
			inferencer.handleInferenceProcessExit(session, tc.processErr)
			inferencer.finishInferenceSession(session, tc.processErr)

			status := inferencer.Status()
			if status.State != "error" || status.Active || !strings.Contains(status.LastError, "capture process exited unexpectedly") {
				t.Fatalf("expected unexpected process exit to remain terminal, got=%+v", status)
			}
			if tc.processErr != nil && !strings.Contains(status.LastError, tc.processErr.Error()) {
				t.Fatalf("expected process error reason to be preserved, got=%+v", status)
			}
			if len(actuatorSink.commands) != 1 {
				t.Fatalf("expected duplicate exit signals to produce one hold, got=%+v", actuatorSink.commands)
			}
			select {
			case <-loopCtx.Done():
			default:
				t.Fatal("expected unexpected process exit to cancel the live session")
			}
		})
	}
}

func TestStopSignEvaluationDeadlineUsesTerminalSafetyStop(t *testing.T) {
	actuatorSink := &recordingInferenceActuator{}
	inferencer := NewInferencer(DefaultInferenceConfig(), actuator.DefaultConfig(), nil, actuatorSink)
	inferencer.stopSignEvaluationLimit = 10 * time.Millisecond
	loopCtx, cancel := context.WithCancel(context.Background())
	defer cancel()
	session := &inferenceSession{cancel: cancel}
	inferencer.active = session
	inferencer.status.State = "running"
	inferencer.status.Active = true

	go inferencer.monitorStopSignEvaluationDeadline(loopCtx, session)
	select {
	case <-loopCtx.Done():
	case <-time.After(time.Second):
		t.Fatal("stopSign evaluation deadline did not cancel inference")
	}
	status := inferencer.Status()
	if status.State != "error" || !strings.Contains(status.LastError, ErrStopSignInferenceDeadlineExceeded.Error()) || status.StoppedAt != "" {
		t.Fatalf("expected deadline reason while process cleanup remains active, got=%+v", status)
	}
	assertStopSignSafetyStopRequested(t, actuatorSink)
}

func TestInferenceStopLatchesSafetyHoldBeforeCancel(t *testing.T) {
	actuatorSink := &recordingInferenceActuator{}
	inferencer := NewInferencer(DefaultInferenceConfig(), actuator.DefaultConfig(), nil, actuatorSink)
	loopCtx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	done <- nil
	inferencer.active = &inferenceSession{cancel: cancel, done: done}
	inferencer.status.State = "running"
	inferencer.status.Active = true

	if _, err := inferencer.Stop(t.Context()); err != nil {
		t.Fatalf("Stop returned error: %v", err)
	}
	select {
	case <-loopCtx.Done():
	default:
		t.Fatal("expected Stop to cancel capture after applying the hold")
	}
	assertStopSignSafetyStopRequested(t, actuatorSink)
}

func TestStopSignOperatingStateAcceptsBehaviorPhasesAndRejectsHazards(t *testing.T) {
	tests := []struct {
		name      string
		mutate    func(*control.RuntimeTelemetry)
		wantError string
	}{
		{name: "safe accelerate state"},
		{
			name: "decelerate phase",
			mutate: func(telemetry *control.RuntimeTelemetry) {
				telemetry.StopSignPhase = control.StopSignPhaseDecelerate
			},
		},
		{
			name: "stop hold phase",
			mutate: func(telemetry *control.RuntimeTelemetry) {
				telemetry.StopSignPhase = control.StopSignPhaseStopHold
			},
		},
		{
			name: "failed phase",
			mutate: func(telemetry *control.RuntimeTelemetry) {
				telemetry.StopSignPhase = "failed"
			},
			wantError: "phase is not safe",
		},
		{
			name: "collision",
			mutate: func(telemetry *control.RuntimeTelemetry) {
				telemetry.CollisionState = "collision"
			},
			wantError: "collision detected",
		},
		{
			name: "off ground",
			mutate: func(telemetry *control.RuntimeTelemetry) {
				offGround := false
				telemetry.OnGround = &offGround
			},
			wantError: "left the ground",
		},
		{
			name: "pitch",
			mutate: func(telemetry *control.RuntimeTelemetry) {
				telemetry.PitchDeg = floatPtr(stopSignMaximumTiltDeg + 0.1)
			},
			wantError: "tilt limit",
		},
		{
			name: "roll",
			mutate: func(telemetry *control.RuntimeTelemetry) {
				telemetry.RollDeg = floatPtr(-stopSignMaximumTiltDeg - 0.1)
			},
			wantError: "tilt limit",
		},
		{
			name: "reverse motion",
			mutate: func(telemetry *control.RuntimeTelemetry) {
				telemetry.VelocityY = floatPtr(stopSignReverseSpeedLimitMPS - 0.01)
			},
			wantError: "reverse motion detected",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			telemetry := validStopSignRuntimeTelemetry()
			if tc.mutate != nil {
				tc.mutate(&telemetry)
			}
			err := validateStopSignOperatingState(telemetry)
			if tc.wantError == "" {
				if err != nil {
					t.Fatalf("expected safe state, got=%v", err)
				}
				return
			}
			if err == nil || !errors.Is(err, ErrStopSignInferencePrecondition) || !strings.Contains(err.Error(), tc.wantError) {
				t.Fatalf("expected %q safety error, got=%v", tc.wantError, err)
			}
		})
	}
}

func TestStopSignOperatingStateRecognizesCompletedEvaluation(t *testing.T) {
	telemetry := validStopSignRuntimeTelemetry()
	telemetry.StopSignPhase = "complete"
	if err := validateStopSignOperatingState(telemetry); !errors.Is(err, ErrStopSignInferenceComplete) {
		t.Fatalf("expected succeeded evaluation terminal, got=%v", err)
	}
}

func TestActiveStopSignInferenceBindsTargetIdentity(t *testing.T) {
	now := time.Date(2026, 8, 5, 12, 0, 0, 0, time.UTC)
	store := control.NewStore(control.WithNowFunc(func() time.Time { return now }))
	store.UpdateTelemetry(validStopSignTelemetryUpdate(now, -12))
	actuatorSink := &recordingInferenceActuator{}
	inferencer := NewInferencer(DefaultInferenceConfig(), actuator.DefaultConfig(), store, actuatorSink)
	inferencer.nowFunc = func() time.Time { return now }
	bindInferenceToCurrentStopSignTarget(t, inferencer, store)

	now = now.Add(50 * time.Millisecond)
	store.UpdateTelemetry(validStopSignTelemetryUpdate(now, -10))
	if err := inferencer.validateActiveStopSignInference(); err != nil {
		t.Fatalf("expected motion relative to the same target to remain valid, got=%v", err)
	}

	now = now.Add(50 * time.Millisecond)
	replaced := validStopSignTelemetryUpdate(now, -15)
	replaced.StopSignEgoStopPose = &control.StopSignPose{X: 0, Y: 1, Z: 0, Heading: 0}
	store.UpdateTelemetry(replaced)
	cause := inferencer.validateActiveStopSignInference()
	if cause == nil || !strings.Contains(cause.Error(), "target changed") {
		t.Fatalf("expected target replacement error, got=%v", cause)
	}
	inferencer.handlePredictionFailure(predictionWindow{sequenceNumber: 31}, cause)
	if !inferencer.stopSignSafetyTripped {
		t.Fatalf("expected target replacement to latch-disable actuation, commands=%+v", actuatorSink.commands)
	}
	assertStopSignSafetyStopRequested(t, actuatorSink)
}

func TestValidateStopSignModelStatusRequiresVisionOnlyStateContract(t *testing.T) {
	cfg := DefaultInferenceConfig()
	if err := validateStopSignModelStatus(validStopSignModelStatus(cfg), cfg); err != nil {
		t.Fatalf("expected stopSign-compatible model status, got=%v", err)
	}

	missingStopIntent := validStopSignModelStatus(cfg)
	missingStopIntent.ControlTargetNames = []string{"future_speed_mps"}
	if err := validateStopSignModelStatus(missingStopIntent, cfg); err == nil || !strings.Contains(err.Error(), "stop_intent") {
		t.Fatalf("expected missing stop-intent head rejection, got=%v", err)
	}

	unloaded := validStopSignModelStatus(cfg)
	unloaded.Loaded = false
	if err := validateStopSignModelStatus(unloaded, cfg); err == nil || !errors.Is(err, ErrStopSignModelIncompatible) {
		t.Fatalf("expected unloaded model rejection, got=%v", err)
	}

	reorderedFeatures := validStopSignModelStatus(cfg)
	reorderedFeatures.TelemetryFeatures = []string{"yaw_sin"}
	if err := validateStopSignModelStatus(reorderedFeatures, cfg); err == nil || !strings.Contains(err.Error(), "telemetry features") {
		t.Fatalf("expected reordered telemetry contract rejection, got=%v", err)
	}

	unexpectedInput := validStopSignModelStatus(cfg)
	unexpectedInput.StateInputs["route_forward_delta"] = stopSignModelInputSpec{Enabled: true}
	if err := validateStopSignModelStatus(unexpectedInput, cfg); err == nil || !strings.Contains(err.Error(), "unexpected state input") {
		t.Fatalf("expected extra enabled state input rejection, got=%v", err)
	}
}

func TestPredictionModelMustRemainBoundToSessionCheckpoint(t *testing.T) {
	cfg := DefaultInferenceConfig()
	inferencer := NewInferencer(cfg, actuator.DefaultConfig(), nil)
	inferencer.stopSignCheckpoint = "C:/models/stopSign/epoch-001.pt"
	compatible := validStopSignPredictResponse(cfg, inferencer.stopSignCheckpoint, 1)
	if err := inferencer.validateStopSignPredictionModel(compatible); err != nil {
		t.Fatalf("expected bound prediction model, got=%v", err)
	}

	hotSwapped := compatible
	hotSwapped.Checkpoint = "C:/models/legacy-driving/epoch-099.pt"
	if err := inferencer.validateStopSignPredictionModel(hotSwapped); err == nil || !errors.Is(err, ErrStopSignModelIncompatible) {
		t.Fatalf("expected hot-swapped prediction checkpoint rejection, got=%v", err)
	}

	legacyContract := compatible
	legacyContract.ControlContract.Name = "direct_controller_v0"
	if err := inferencer.validateStopSignPredictionModel(legacyContract); err == nil || !strings.Contains(err.Error(), stopsigncontrol.StopSignMotionPlanContractV1) {
		t.Fatalf("expected legacy control contract rejection, got=%v", err)
	}

	legacyPlannerVersion := compatible
	legacyPlannerVersion.PlannerFormatVersion = 2
	if err := inferencer.validateStopSignPredictionModel(legacyPlannerVersion); err == nil || !strings.Contains(err.Error(), "planner format version") {
		t.Fatalf("expected legacy planner version rejection, got=%v", err)
	}

	wrongDirection := compatible
	wrongDirection.Direction = "reverse"
	if err := inferencer.validateStopSignPredictionModel(wrongDirection); err == nil || !strings.Contains(err.Error(), "direction") {
		t.Fatalf("expected reverse direction rejection, got=%v", err)
	}

	wrongContractDirection := compatible
	wrongContractDirection.ControlContract.Direction = "reverse"
	if err := inferencer.validateStopSignPredictionModel(wrongContractDirection); err == nil || !strings.Contains(err.Error(), "control contract direction") {
		t.Fatalf("expected nested reverse contract direction rejection, got=%v", err)
	}

	wrongTiming := compatible
	wrongTiming.ControlHorizonDtMs = []int{100, 250, 500, 1001}
	if err := inferencer.validateStopSignPredictionModel(wrongTiming); err == nil || !strings.Contains(err.Error(), "control horizon timing") {
		t.Fatalf("expected noncanonical setpoint timing rejection, got=%v", err)
	}

	wrongSampleInterval := compatible
	wrongSampleInterval.TelemetrySampleIntervalMs = 51
	if err := inferencer.validateStopSignPredictionModel(wrongSampleInterval); err == nil || !strings.Contains(err.Error(), "telemetry sample interval") {
		t.Fatalf("expected mismatched telemetry cadence rejection, got=%v", err)
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

func TestInferencerModelsClearsCheckpointUnloadedBehindBackend(t *testing.T) {
	cfg := DefaultInferenceConfig()
	cfg.ModelServerURL = "http://planner.local"
	inferencer := NewInferencer(cfg, actuator.DefaultConfig(), nil)
	inferencer.loadedCheckpoint = "C:/models/stale.pt"
	inferencer.loadedModelDevice = "cuda"
	inferencer.httpClient = &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		switch req.URL.Path {
		case "/models":
			return inferenceJSONResponse(http.StatusOK, map[string]any{
				"models": []map[string]any{{
					"label":  "run-1 - epoch 006 (best)",
					"path":   "C:/models/run-1/epoch-006.pt",
					"isBest": true,
				}},
			}), nil
		case "/model":
			return inferenceJSONResponse(http.StatusOK, map[string]any{"loaded": false}), nil
		default:
			return inferenceJSONResponse(http.StatusNotFound, map[string]string{"error": "not found"}), nil
		}
	})}

	models, err := inferencer.Models(t.Context(), "")
	if err != nil || len(models) != 1 {
		t.Fatalf("expected catalog discovery to succeed, models=%+v err=%v", models, err)
	}
	status := inferencer.Status()
	if status.LoadedCheckpoint != "" || status.LoadedModelDevice != "" {
		t.Fatalf("remote unload left a stale loaded model in backend status: %+v", status)
	}
}

func TestRequestPredictionBuildsPhysicalSetpointPlan(t *testing.T) {
	cfg := DefaultInferenceConfig()
	cfg.ModelServerURL = "http://planner.local"
	cfg.ImageOffsets = []int{-4, -2, 0}
	cfg.WindowSize = len(cfg.ImageOffsets)
	cfg.TelemetryOffsets = []int{-2, -1, 0}
	cfg.FutureSteps = 4
	cfg.PredictionTimeout = time.Second
	nowValue := time.UnixMilli(966).UTC()
	store := control.NewStore(control.WithNowFunc(func() time.Time { return nowValue }))
	store.UpdateTelemetry(control.TelemetryUpdate{CurrentSpeed: 4.0, CurrentYaw: 10.0, YawRate: 0.1, Steering: 0.1, Acceleration: 0.2, TimestampMs: 966})
	nowValue = time.UnixMilli(1016).UTC()
	store.UpdateTelemetry(control.TelemetryUpdate{CurrentSpeed: 4.5, CurrentYaw: 11.0, YawRate: 0.2, Steering: 0.2, Acceleration: 0.3, TimestampMs: 1016})
	nowValue = time.UnixMilli(1066).UTC()
	current := validStopSignTelemetryUpdate(nowValue, -1.2)
	current.CurrentSpeed = 5.0
	current.CurrentYaw = 12.0
	current.YawRate = 0.3
	current.Steering = 0.3
	current.Acceleration = 0.4
	current.RouteDirectionKeepStraight = 1
	current.RouteDirectionCode = 1
	current.RouteDirectionDistanceM = 8.5
	current.RouteForwardDelta = 0.25
	current.RouteHeadingError = -3.5
	current.RouteDistance = 7.25
	current.LeadVehicleDistance = 12.0
	current.HasLeadVehicle = true
	current.StopSignLateralErrorM = 0.4
	current.StopSignHeadingErrorDeg = -6.5
	current.StopSignDistanceM = 1.3
	current.StopSignAttemptIndex = 2
	current.StopSignAttemptCount = 8
	store.UpdateTelemetry(current)
	inferencer := NewInferencer(cfg, actuator.DefaultConfig(), store)
	now := time.UnixMilli(1066).UTC()
	inferencer.nowFunc = func() time.Time { return now }
	bindInferenceToCurrentStopSignTarget(t, inferencer, store)
	inferencer.stopSignCheckpoint = "C:/models/run-1/epoch-006.pt"
	inferencer.httpClient = &http.Client{
		Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			var payload map[string]any
			if err := json.NewDecoder(req.Body).Decode(&payload); err != nil {
				t.Fatalf("decode payload: %v", err)
			}
			if got := payload["planner_format"]; got != cfg.PlannerFormat {
				t.Fatalf("unexpected planner contract: got=%#v want=%q", got, cfg.PlannerFormat)
			}
			if got := payload["control_contract"]; got != stopsigncontrol.StopSignMotionPlanContractV1 {
				t.Fatalf("unexpected control contract: got=%#v want=%q", got, stopsigncontrol.StopSignMotionPlanContractV1)
			}
			sampledAtS, ok := payload["sampled_at_s"].(float64)
			if !ok || math.Abs(sampledAtS-1.066) > 1e-9 {
				t.Fatalf("unexpected sampled_at_s: got=%#v want=1.066", payload["sampled_at_s"])
			}
			for _, forbidden := range []string{
				"currentSpeed",
				"routeForwardDelta",
				"routeHeadingError",
				"routeDistance",
				"stopSignLongitudinalError",
				"stopSignLateralError",
				"stopSignHeadingError",
				"stopSignDistance",
			} {
				if _, ok := payload[forbidden]; ok {
					t.Fatalf("inference payload must not expose scalar state input %q: %+v", forbidden, payload)
				}
			}
			telemetry, ok := payload["telemetry"].([]any)
			if !ok || len(telemetry) != 1 {
				t.Fatalf("expected telemetry batch, got=%T %+v", payload["telemetry"], payload["telemetry"])
			}
			response := validStopSignPredictResponse(cfg, "C:/models/run-1/epoch-006.pt", sampledAtS)
			response.PredControls = [][][]float64{{
				{1.40, 0.05},
				{1.20, 0.05},
				{0.00, 0.90},
				{0.00, 1.00},
			}}
			response.PredAux = [][][]float64{{
				{0.2, 0.0, 0.0},
				{0.2, 0.0, 0.0},
				{0.0, 1.0, 1.0},
				{0.0, 1.0, 1.0},
			}}
			body, _ := json.Marshal(response)
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

	prediction, err := inferencer.requestPrediction(t.Context(), cfg.ModelServerURL, window)
	if err != nil {
		t.Fatalf("requestPrediction returned error: %v", err)
	}
	if prediction == nil {
		t.Fatal("expected prediction")
	}
	if len(prediction.RawPredControls) != 4 || len(prediction.RawPredAux) != 4 {
		t.Fatalf("unexpected raw planner outputs: %+v", prediction)
	}
	if prediction.ControlContract.Name != stopsigncontrol.StopSignMotionPlanContractV1 || prediction.ControlContract.Version != 1 {
		t.Fatalf("unexpected prediction control contract: %+v", prediction.ControlContract)
	}
	if prediction.SetpointPlan == nil {
		t.Fatal("expected physical stopSign setpoint plan")
	}
	plan := prediction.SetpointPlan
	if plan.Contract != stopsigncontrol.StopSignMotionPlanContractV1 || plan.Direction != stopsigncontrol.StopSignDirectionForward {
		t.Fatalf("unexpected setpoint plan identity: %+v", plan)
	}
	if math.Abs(plan.SampledAtS-1.066) > 1e-9 || len(plan.Points) != 4 {
		t.Fatalf("unexpected setpoint plan timestamp/length: %+v", plan)
	}
	for index, point := range plan.Points {
		if point.DtMs != cfg.ControlHorizonDtMs[index] {
			t.Fatalf("unexpected point timing at %d: got=%d want=%d", index, point.DtMs, cfg.ControlHorizonDtMs[index])
		}
	}
	first := plan.Points[0]
	if first.DesiredWheelSteerNormalized != 0 || math.Abs(first.DesiredSpeedMPS-1.4) > 1e-9 || math.Abs(first.StopProbability-0.05) > 1e-9 {
		t.Fatalf("unexpected first physical setpoint: %+v", first)
	}
}

func TestPredictionResponseIsDiscardedWhenStopSignHazardAppearsDuringRequest(t *testing.T) {
	cfg := DefaultInferenceConfig()
	cfg.ModelServerURL = "http://planner.local"
	cfg.ImageOffsets = []int{0}
	cfg.WindowSize = 1
	cfg.TelemetryOffsets = []int{0}
	cfg.FutureOffsets = []int{1}
	cfg.FutureSteps = 1
	cfg.ControlHorizonDtMs = []int{50}
	now := time.UnixMilli(1000).UTC()
	store := control.NewStore(control.WithNowFunc(func() time.Time { return now }))
	store.UpdateTelemetry(validStopSignTelemetryUpdate(now, -12))
	actuatorSink := &recordingInferenceActuator{}
	inferencer := NewInferencer(cfg, actuator.DefaultConfig(), store, actuatorSink)
	inferencer.nowFunc = func() time.Time { return now }
	bindInferenceToCurrentStopSignTarget(t, inferencer, store)
	inferencer.stopSignCheckpoint = "C:/models/stopSign/epoch-001.pt"
	loopCtx, cancel := context.WithCancel(context.Background())
	session := &inferenceSession{cancel: cancel}
	inferencer.active = session
	inferencer.status.State = "running"
	inferencer.status.Active = true
	inferencer.httpClient = &http.Client{
		Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			var payload map[string]any
			if err := json.NewDecoder(req.Body).Decode(&payload); err != nil {
				t.Fatalf("decode payload: %v", err)
			}
			sampledAtS, ok := payload["sampled_at_s"].(float64)
			if !ok {
				t.Fatalf("expected sampled_at_s request field, got=%#v", payload["sampled_at_s"])
			}
			now = now.Add(10 * time.Millisecond)
			hazard := validStopSignTelemetryUpdate(now, -12)
			hazard.CollisionState = "vehicle"
			store.UpdateTelemetry(hazard)
			response := validStopSignPredictResponse(cfg, inferencer.stopSignCheckpoint, sampledAtS)
			body, _ := json.Marshal(response)
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body:       io.NopCloser(bytes.NewReader(body)),
			}, nil
		}),
	}

	inferencer.processPredictionWindow(loopCtx, session, cfg.ModelServerURL, predictionWindow{
		frames:         []*image.RGBA{image.NewRGBA(image.Rect(0, 0, 2, 2))},
		frameIndex:     1,
		frameIndices:   []int{1},
		frameTimes:     []time.Time{now},
		capturedAt:     now,
		sequenceNumber: 41,
	})

	assertStopSignSafetyStopRequested(t, actuatorSink)
	if len(actuatorSink.plans) != 0 {
		t.Fatalf("expected the stale setpoint plan to be discarded, got=%+v", actuatorSink.plans)
	}
	select {
	case <-loopCtx.Done():
	default:
		t.Fatal("expected request-time hazard to cancel the active inference session")
	}
}

func TestBuildPlannerSelectionFailsWhenTelemetryIsUnavailable(t *testing.T) {
	inferencer := NewInferencer(DefaultInferenceConfig(), actuator.DefaultConfig(), control.NewStore())
	inferencer.nowFunc = func() time.Time { return time.Unix(1710000000, 0).UTC() }
	_, err := inferencer.buildPlannerSelection(predictionWindow{capturedAt: time.Unix(1710000000, 0).UTC()})
	if err == nil || !errors.Is(err, ErrStopSignInferencePrecondition) || !strings.Contains(err.Error(), "FiveM telemetry is unavailable") {
		t.Fatalf("expected telemetry unavailable error, got=%v", err)
	}
}

func TestBuildPlannerSelectionEnforcesFrameTelemetrySkewLimit(t *testing.T) {
	cfg := DefaultInferenceConfig()
	cfg.TelemetryOffsets = []int{0}
	cfg.TelemetryFeatureNames = []string{"current_speed"}
	cfg.MaxFrameTelemetrySkew = 75 * time.Millisecond
	nowValue := time.UnixMilli(1000).UTC()
	store := control.NewStore(control.WithNowFunc(func() time.Time { return nowValue }))
	update := validStopSignTelemetryUpdate(nowValue, -12)
	update.CurrentSpeed = 4.0
	store.UpdateTelemetry(update)
	inferencer := NewInferencer(cfg, actuator.DefaultConfig(), store)
	bindInferenceToCurrentStopSignTarget(t, inferencer, store)

	inferencer.nowFunc = func() time.Time { return time.UnixMilli(1075).UTC() }
	selection, err := inferencer.buildPlannerSelection(predictionWindow{
		frameIndex:     7,
		frameTimes:     []time.Time{time.UnixMilli(1075).UTC()},
		capturedAt:     time.UnixMilli(1075).UTC(),
		sequenceNumber: 3,
	})
	if err != nil {
		t.Fatalf("expected exact skew boundary to remain valid: %v", err)
	}
	if !selection.frameTelemetryAligned || math.Abs(selection.frameTelemetrySkewMs-75) > 1e-9 {
		t.Fatalf("expected aligned selection at exact boundary, got=%+v", selection)
	}

	inferencer.nowFunc = func() time.Time { return time.UnixMilli(1076).UTC() }
	_, err = inferencer.buildPlannerSelection(predictionWindow{
		frameIndex:     8,
		frameTimes:     []time.Time{time.UnixMilli(1076).UTC()},
		capturedAt:     time.UnixMilli(1076).UTC(),
		sequenceNumber: 4,
	})
	if err == nil || !errors.Is(err, ErrStopSignInferencePrecondition) || !strings.Contains(err.Error(), "skew exceeded") {
		t.Fatalf("expected skew beyond the configured maximum to fail closed, got=%v", err)
	}
}

func TestBuildPredictionRejectsLegacyImmediateOutputShape(t *testing.T) {
	cfg := DefaultInferenceConfig()
	inferencer := NewInferencer(cfg, actuator.DefaultConfig(), control.NewStore())
	inferencer.nowFunc = func() time.Time { return time.UnixMilli(2000).UTC() }
	response := validStopSignPredictResponse(cfg, "C:/models/stopSign/epoch-001.pt", 1.9)
	response.PredControls = [][][]float64{{
		{0.4, 0.1},
	}}

	_, err := inferencer.buildPrediction(response, "http://planner.local", predictionWindow{
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
	if err == nil || !strings.Contains(err.Error(), "pred_controls horizon mismatch") {
		t.Fatalf("expected legacy one-step output to be rejected, got=%v", err)
	}
}

func TestBuildPredictionRejectsOutOfRangePhysicalSetpoints(t *testing.T) {
	cfg := DefaultInferenceConfig()
	inferencer := NewInferencer(cfg, actuator.DefaultConfig(), control.NewStore())
	inferencer.nowFunc = func() time.Time { return time.UnixMilli(2000).UTC() }
	window := predictionWindow{
		frameIndex:     1,
		frameIndices:   []int{1},
		frameTimes:     []time.Time{time.UnixMilli(1900).UTC()},
		capturedAt:     time.UnixMilli(1900).UTC(),
		sequenceNumber: 1,
	}
	selection := plannerSelection{
		selectedTelemetry: []control.RuntimeTelemetry{{CurrentSpeed: 3}},
		telemetryTimesMs:  []int64{1900},
		frameShape:        []int{1, 1, 3, cfg.FrameHeight, cfg.FrameWidth},
		telemetryShape:    []int{1, 1, len(cfg.TelemetryFeatureNames)},
	}
	tests := []struct {
		name       string
		column     int
		value      float64
		wantDetail string
	}{
		{name: "future speed", column: 0, value: stopsigncontrol.StopSignMotionPlanMaxSpeedMPS + 0.01, wantDetail: "desired_speed_mps"},
		{name: "stop intent", column: 1, value: -0.01, wantDetail: "stop_probability"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			response := validStopSignPredictResponse(cfg, "C:/models/stopSign/epoch-001.pt", 1.9)
			response.PredControls[0][2][tc.column] = tc.value
			_, err := inferencer.buildPrediction(response, "http://planner.local", window, selection, nil)
			if err == nil || !errors.Is(err, stopsigncontrol.ErrInvalidPlan) || !strings.Contains(err.Error(), tc.wantDetail) {
				t.Fatalf("expected out-of-range %s rejection, got=%v", tc.wantDetail, err)
			}
		})
	}
}

func TestBuildPredictionRequiresSampleTimestampEcho(t *testing.T) {
	cfg := DefaultInferenceConfig()
	inferencer := NewInferencer(cfg, actuator.DefaultConfig(), control.NewStore())
	inferencer.nowFunc = func() time.Time { return time.UnixMilli(2000).UTC() }
	response := validStopSignPredictResponse(cfg, "C:/models/stopSign/epoch-001.pt", 1.901)

	_, err := inferencer.buildPrediction(response, "http://planner.local", predictionWindow{
		frameTimes: []time.Time{time.UnixMilli(1900).UTC()},
		capturedAt: time.UnixMilli(1900).UTC(),
	}, plannerSelection{telemetryTimesMs: []int64{1900}}, nil)
	if err == nil || !strings.Contains(err.Error(), "sampled_at_s did not echo") {
		t.Fatalf("expected mismatched observation timestamp rejection, got=%v", err)
	}
}

func TestFindAnchorTelemetryIndexPrefersSourceTimestamp(t *testing.T) {
	history := []control.RuntimeTelemetry{
		{TimestampMs: 1000, ReceivedAtMs: 5000},
		{TimestampMs: 1033, ReceivedAtMs: 5067},
		{TimestampMs: 1066, ReceivedAtMs: 5134},
	}

	index, err := findAnchorTelemetryIndex(history, 1067, 75*time.Millisecond)
	if err != nil {
		t.Fatalf("findAnchorTelemetryIndex returned error: %v", err)
	}
	if index != 2 {
		t.Fatalf("unexpected aligned index: got=%d want=2", index)
	}
}

func TestSelectTelemetryAtOffsetsUsesConfiguredSourceTimestamps(t *testing.T) {
	history := []control.RuntimeTelemetry{
		{TimestampMs: 850, CurrentSpeed: 0.5},
		{TimestampMs: 900, CurrentSpeed: 1.0},
		{TimestampMs: 926, CurrentSpeed: 99.0},
		{TimestampMs: 951, CurrentSpeed: 2.0},
		{TimestampMs: 978, CurrentSpeed: 99.0},
		{TimestampMs: 1000, CurrentSpeed: 3.0},
	}

	selected, timestamps, err := selectTelemetryAtOffsets(
		history,
		len(history)-1,
		[]int{-2, -1, 0},
		50*time.Millisecond,
		125*time.Millisecond,
	)
	if err != nil {
		t.Fatalf("selectTelemetryAtOffsets returned error: %v", err)
	}
	wantTimestamps := []int64{900, 951, 1000}
	wantSpeeds := []float64{1, 2, 3}
	if !reflect.DeepEqual(timestamps, wantTimestamps) {
		t.Fatalf("unexpected source timestamps: got=%v want=%v", timestamps, wantTimestamps)
	}
	for index, sample := range selected {
		if sample.CurrentSpeed != wantSpeeds[index] {
			t.Fatalf("unexpected sample at offset %d: got speed=%f want=%f", index, sample.CurrentSpeed, wantSpeeds[index])
		}
	}
}

func TestSelectTelemetryAtOffsetsFailsWhenTimestampSlotExceedsHalfIntervalTolerance(t *testing.T) {
	history := []control.RuntimeTelemetry{
		{TimestampMs: 900},
		{TimestampMs: 924},
		{TimestampMs: 1000},
	}

	_, _, err := selectTelemetryAtOffsets(
		history,
		len(history)-1,
		[]int{-2, -1, 0},
		50*time.Millisecond,
		125*time.Millisecond,
	)
	if err == nil || !strings.Contains(err.Error(), "offset -1") || !strings.Contains(err.Error(), "tolerance=25ms") {
		t.Fatalf("expected missing 50ms source slot to fail closed at the bounded tolerance, got=%v", err)
	}
}
