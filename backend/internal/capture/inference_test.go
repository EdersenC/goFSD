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
	"strings"
	"testing"
	"time"

	"awesomeProject/internal/actuator"
	"awesomeProject/internal/control"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

type recordingInferenceActuator struct {
	commands      []actuator.CommandRequest
	err           error
	state         actuator.State
	nextCommandID int64
	skipApply     bool
	applyFault    string
}

type statefulInferenceActuator struct {
	recordingInferenceActuator
	state actuator.State
}

func (a *statefulInferenceActuator) State() actuator.State {
	state := a.recordingInferenceActuator.State()
	state.Supported = a.state.Supported
	state.Ready = a.state.Ready
	state.Platform = a.state.Platform
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
	a.state.Applied.CommandID = a.nextCommandID
	a.state.Applied.Enabled = enabled
	a.state.Applied.Handbrake = req.Handbrake
	a.state.Applied.Steer = req.Steer
	a.state.Applied.Throttle = req.Throttle
	a.state.Applied.Brake = req.BrakePressureAvg
	a.state.LastApplyError = ""
	a.state.LastApplySucceededAt = appliedAt.Format(time.RFC3339Nano)
	return a.state, nil
}

func (a *recordingInferenceActuator) State() actuator.State {
	return a.state
}

func validParkingTelemetryUpdate(now time.Time, longitudinal float64) control.TelemetryUpdate {
	zero := 0.0
	onGround := true
	return control.TelemetryUpdate{
		VehicleExists:            true,
		IsInVehicle:              true,
		PositionX:                &zero,
		PositionY:                floatPtr(longitudinal),
		PositionZ:                &zero,
		VelocityX:                &zero,
		VelocityY:                &zero,
		VelocityZ:                &zero,
		PitchDeg:                 &zero,
		RollDeg:                  &zero,
		OnGround:                 &onGround,
		ParkingTargetConfigured:  true,
		ParkingLongitudinalError: longitudinal,
		ParkingPhase:             "ready",
		TimestampMs:              now.UnixMilli(),
	}
}

func bindInferenceToCurrentParkingTarget(t *testing.T, inferencer *Inferencer, store *control.Store) {
	t.Helper()
	telemetry, _ := store.LatestTelemetrySnapshot()
	if telemetry == nil {
		t.Fatal("expected parking telemetry")
	}
	target, err := parkingTargetFromTelemetry(*telemetry)
	if err != nil {
		t.Fatalf("derive parking target: %v", err)
	}
	inferencer.parkingTarget = &target
}

func validParkingRuntimeTelemetry() control.RuntimeTelemetry {
	zero := 0.0
	onGround := true
	return control.RuntimeTelemetry{
		CurrentYaw:   0,
		VelocityX:    &zero,
		VelocityY:    &zero,
		PitchDeg:     &zero,
		RollDeg:      &zero,
		OnGround:     &onGround,
		ParkingPhase: "ready",
	}
}

func validParkingModelStatus(cfg InferenceConfig) parkingModelStatus {
	inputs := make(map[string]parkingModelInputSpec, len(requiredParkingStateInputs))
	for _, name := range requiredParkingStateInputs {
		inputs[name] = parkingModelInputSpec{Enabled: true}
	}
	return parkingModelStatus{
		Loaded:        true,
		Checkpoint:    "C:/models/parking/epoch-001.pt",
		PlannerFormat: cfg.PlannerFormat,
		ImageSize:     parkingModelImageSize{Width: cfg.FrameWidth, Height: cfg.FrameHeight},
		FrameWindow: parkingModelFrameWindow{
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

func TestInferenceStartRejectsUnsafeParkingStateBeforeCaptureSetup(t *testing.T) {
	now := time.Date(2026, 8, 5, 12, 0, 0, 0, time.UTC)
	store := control.NewStore(control.WithNowFunc(func() time.Time { return now }))
	store.UpdateTelemetry(control.TelemetryUpdate{
		VehicleExists:            true,
		IsInVehicle:              true,
		ParkingTargetConfigured:  false,
		ParkingLongitudinalError: -12,
		TimestampMs:              now.UnixMilli(),
	})
	inferencer := NewInferencer(DefaultInferenceConfig(), actuator.DefaultConfig(), store)
	inferencer.nowFunc = func() time.Time { return now }
	discoveryCalled := false
	inferencer.discover = func(context.Context) ([]Source, error) {
		discoveryCalled = true
		return nil, nil
	}

	_, err := inferencer.Start(t.Context(), InferenceStartRequest{})
	if err == nil || !errors.Is(err, ErrInferenceStartFailed) || !errors.Is(err, ErrParkingInferencePrecondition) || !strings.Contains(err.Error(), "parking target is not configured") {
		t.Fatalf("expected useful parking precondition error, got=%v", err)
	}
	if discoveryCalled {
		t.Fatal("expected parking preconditions to fail before capture source discovery")
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

func TestInferenceStartRequiresReadyActuatorBeforeSourceDiscovery(t *testing.T) {
	now := time.Date(2026, 8, 5, 12, 0, 0, 0, time.UTC)
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
			state: actuator.State{Supported: true, Ready: true, Platform: "windows"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			store := control.NewStore(control.WithNowFunc(func() time.Time { return now }))
			store.UpdateTelemetry(validParkingTelemetryUpdate(now, -12))
			actuatorSink := &statefulInferenceActuator{state: tc.state}
			inferencer := NewInferencer(DefaultInferenceConfig(), actuator.DefaultConfig(), store, actuatorSink)
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

func TestParkingInferenceRejectsTemporalActuationUntilOffsetsAreTimedExactly(t *testing.T) {
	actuatorCfg := actuator.DefaultConfig()
	actuatorCfg.TemporalHorizonActuatorEnabled = true
	actuatorSink := &statefulInferenceActuator{state: actuator.State{Supported: true, Ready: true, Platform: "windows"}}
	inferencer := NewInferencer(DefaultInferenceConfig(), actuatorCfg, nil, actuatorSink)
	if err := inferencer.validateInferenceActuatorReady(); err == nil || !errors.Is(err, ErrInferenceActuatorUnavailable) || !strings.Contains(err.Error(), "future offsets") {
		t.Fatalf("expected temporal parking actuation to fail closed, got=%v", err)
	}
}

func TestParkingInferenceArmRequiresAppliedNeutralCommand(t *testing.T) {
	actuatorSink := &recordingInferenceActuator{skipApply: true}
	inferencer := NewInferencer(DefaultInferenceConfig(), actuator.DefaultConfig(), nil, actuatorSink)
	inferencer.actuatorConfirmTimeout = 5 * time.Millisecond
	inferencer.actuatorConfirmInterval = time.Millisecond

	err := inferencer.armActuatorForParkingInference()
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

func TestValidateParkingStartEnvelopeMatchesForwardCurriculumBounds(t *testing.T) {
	tests := []struct {
		name      string
		telemetry control.RuntimeTelemetry
		wantError string
	}{
		{
			name: "nearest curriculum boundary",
			telemetry: control.RuntimeTelemetry{
				ParkingLongitudinalError: parkingStartLongitudinalMaxM,
				ParkingLateralError:      parkingStartLateralLimitM,
				ParkingHeadingError:      parkingStartHeadingLimitDeg,
			},
		},
		{
			name: "furthest curriculum boundary",
			telemetry: control.RuntimeTelemetry{
				ParkingLongitudinalError: parkingStartLongitudinalMinM,
				ParkingLateralError:      -parkingStartLateralLimitM,
				ParkingHeadingError:      -parkingStartHeadingLimitDeg,
			},
		},
		{
			name: "too close",
			telemetry: control.RuntimeTelemetry{
				ParkingLongitudinalError: parkingStartLongitudinalMaxM + 0.01,
			},
			wantError: "longitudinal offset",
		},
		{
			name: "too far",
			telemetry: control.RuntimeTelemetry{
				ParkingLongitudinalError: parkingStartLongitudinalMinM - 0.01,
			},
			wantError: "longitudinal offset",
		},
		{
			name: "outside lateral envelope",
			telemetry: control.RuntimeTelemetry{
				ParkingLongitudinalError: -12,
				ParkingLateralError:      parkingStartLateralLimitM + 0.01,
			},
			wantError: "lateral offset",
		},
		{
			name: "outside heading envelope",
			telemetry: control.RuntimeTelemetry{
				ParkingLongitudinalError: -12,
				ParkingHeadingError:      parkingStartHeadingLimitDeg + 0.01,
			},
			wantError: "heading error",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := validateParkingStartEnvelope(tc.telemetry)
			if tc.wantError == "" {
				if err != nil {
					t.Fatalf("expected valid curriculum pose, got=%v", err)
				}
				return
			}
			if err == nil || !errors.Is(err, ErrParkingInferencePrecondition) || !strings.Contains(err.Error(), tc.wantError) {
				t.Fatalf("expected %q precondition error, got=%v", tc.wantError, err)
			}
		})
	}
}

func TestValidateParkingInferenceStartRequiresFreshValidEgoTelemetry(t *testing.T) {
	now := time.Date(2026, 8, 5, 12, 0, 0, 0, time.UTC)
	invalidStore := control.NewStore(control.WithNowFunc(func() time.Time { return now }))
	invalidStore.UpdateTelemetry(control.TelemetryUpdate{
		VehicleExists:            false,
		IsInVehicle:              false,
		ParkingTargetConfigured:  true,
		ParkingLongitudinalError: -12,
		TimestampMs:              now.UnixMilli(),
	})
	invalidInferencer := NewInferencer(DefaultInferenceConfig(), actuator.DefaultConfig(), invalidStore)
	invalidInferencer.nowFunc = func() time.Time { return now }
	if err := invalidInferencer.validateParkingInferenceStart(); err == nil || !strings.Contains(err.Error(), "vehicle does not exist") {
		t.Fatalf("expected invalid ego telemetry error, got=%v", err)
	}

	validStore := control.NewStore(control.WithNowFunc(func() time.Time { return now }))
	validStore.UpdateTelemetry(control.TelemetryUpdate{
		VehicleExists:            true,
		IsInVehicle:              true,
		ParkingTargetConfigured:  true,
		ParkingLongitudinalError: -12,
		TimestampMs:              now.UnixMilli(),
	})
	staleInferencer := NewInferencer(DefaultInferenceConfig(), actuator.DefaultConfig(), validStore)
	staleInferencer.nowFunc = func() time.Time { return now.Add(defaultTelemetryStaleAfter + time.Millisecond) }
	if err := staleInferencer.validateParkingInferenceStart(); err == nil || !strings.Contains(err.Error(), "telemetry is stale") {
		t.Fatalf("expected stale ego telemetry error, got=%v", err)
	}
}

func TestParkingTargetLossDisablesActuatorWithoutFallbackDecay(t *testing.T) {
	now := time.Date(2026, 8, 5, 12, 0, 0, 0, time.UTC)
	store := control.NewStore(control.WithNowFunc(func() time.Time { return now }))
	store.UpdateTelemetry(validParkingTelemetryUpdate(now, -12))

	actuatorSink := &recordingInferenceActuator{}
	actuatorCfg := actuator.DefaultConfig()
	actuatorCfg.TemporalHorizonActuatorEnabled = true
	inferencer := NewInferencer(DefaultInferenceConfig(), actuatorCfg, store, actuatorSink)
	inferencer.nowFunc = func() time.Time { return now }
	bindInferenceToCurrentParkingTarget(t, inferencer, store)
	inferencer.processorState.PrevSteering = 0.8
	inferencer.processorState.PrevThrottle = 0.9
	inferencer.processorState.PrevBrake = 0.7

	now = now.Add(50 * time.Millisecond)
	lostTarget := validParkingTelemetryUpdate(now, -12)
	lostTarget.ParkingTargetConfigured = false
	store.UpdateTelemetry(lostTarget)
	cause := inferencer.validateActiveParkingInference()
	if cause == nil || !errors.Is(cause, ErrParkingInferencePrecondition) {
		t.Fatalf("expected target-loss safety error, got=%v", cause)
	}

	inferencer.handlePredictionFailure(predictionWindow{sequenceNumber: 17}, cause)
	if len(actuatorSink.commands) != 1 {
		t.Fatalf("expected exactly one fail-safe actuator command, got=%+v", actuatorSink.commands)
	}
	command := actuatorSink.commands[0]
	if command.Enabled == nil || *command.Enabled || command.Steer != 0 || command.Throttle != 0 || command.BrakePressureAvg != 0 || !command.Handbrake {
		t.Fatalf("expected disabled drive controls plus a safety handbrake hold after target loss, got=%+v", command)
	}
	if inferencer.processorState.PrevSteering != 0 || inferencer.processorState.PrevThrottle != 0 || inferencer.processorState.PrevBrake != 0 {
		t.Fatalf("expected fallback history to be cleared, got=%+v", inferencer.processorState)
	}

	now = now.Add(50 * time.Millisecond)
	store.UpdateTelemetry(validParkingTelemetryUpdate(now, -12))
	latched := inferencer.validateActiveParkingInference()
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
	inferencer.processorState.PrevSteering = -0.7
	inferencer.processorState.PrevThrottle = 0.8
	inferencer.processorState.PrevBrake = 0.6

	inferencer.handlePredictionFailure(predictionWindow{sequenceNumber: 23}, errors.New("planner request timed out"))
	if len(actuatorSink.commands) != 1 {
		t.Fatalf("expected one fail-stop command, got=%+v", actuatorSink.commands)
	}
	command := actuatorSink.commands[0]
	if command.Enabled == nil || *command.Enabled || command.Steer != 0 || command.Throttle != 0 || command.BrakePressureAvg != 0 || !command.Handbrake {
		t.Fatalf("expected generic prediction error to disable drive controls and hold the handbrake, got=%+v", command)
	}
	if !inferencer.parkingSafetyTripped {
		t.Fatal("expected generic prediction error to latch the parking safety interlock")
	}
}

func TestParkingSuccessNeutralizesDriveAndSurfacesSucceededState(t *testing.T) {
	actuatorSink := &recordingInferenceActuator{}
	inferencer := NewInferencer(DefaultInferenceConfig(), actuator.DefaultConfig(), nil, actuatorSink)
	inferencer.nowFunc = func() time.Time { return time.UnixMilli(3000).UTC() }

	inferencer.handlePredictionFailure(predictionWindow{sequenceNumber: 24}, ErrParkingInferenceComplete)
	if len(actuatorSink.commands) != 1 {
		t.Fatalf("expected one terminal safety command, got=%+v", actuatorSink.commands)
	}
	command := actuatorSink.commands[0]
	if command.Enabled == nil || *command.Enabled || command.Steer != 0 || command.Throttle != 0 || command.BrakePressureAvg != 0 || !command.Handbrake {
		t.Fatalf("expected parking success to disable drive controls and hold the handbrake, got=%+v", command)
	}
	status := inferencer.Status()
	if status.State != "succeeded" || status.LastError != "" || !inferencer.parkingCompleted {
		t.Fatalf("expected a latched succeeded state, got status=%+v completed=%v", status, inferencer.parkingCompleted)
	}
	if err := inferencer.validateActiveParkingInference(); !errors.Is(err, ErrParkingInferenceComplete) {
		t.Fatalf("expected completed session to remain terminal until restart, got=%v", err)
	}
}

func TestParkingTerminalTransitionHoldsOnceAndCancelsActiveSession(t *testing.T) {
	actuatorSink := &recordingInferenceActuator{}
	inferencer := NewInferencer(DefaultInferenceConfig(), actuator.DefaultConfig(), nil, actuatorSink)
	inferencer.nowFunc = func() time.Time { return time.UnixMilli(4000).UTC() }
	loopCtx, cancel := context.WithCancel(context.Background())
	session := &inferenceSession{cancel: cancel}
	inferencer.active = session
	inferencer.status.State = "running"
	inferencer.status.Active = true
	cause := errors.New("parking target disappeared")

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
	if len(actuatorSink.commands) != 1 || actuatorSink.commands[0].Enabled == nil || *actuatorSink.commands[0].Enabled || !actuatorSink.commands[0].Handbrake {
		t.Fatalf("expected one disabled handbrake command, got=%+v", actuatorSink.commands)
	}

	inferencer.handleSessionPredictionFailure(session, predictionWindow{sequenceNumber: 26}, errors.New("duplicate failure"))
	if len(actuatorSink.commands) != 1 || inferencer.Status().LastError != cause.Error() {
		t.Fatalf("expected duplicate failure to preserve the first terminal result, status=%+v commands=%+v", inferencer.Status(), actuatorSink.commands)
	}
}

func TestParkingSuccessDoesNotMaskSafetyHoldFailureAfterCleanup(t *testing.T) {
	actuatorSink := &recordingInferenceActuator{err: errors.New("controller disconnected")}
	inferencer := NewInferencer(DefaultInferenceConfig(), actuator.DefaultConfig(), nil, actuatorSink)
	loopCtx, cancel := context.WithCancel(context.Background())
	session := &inferenceSession{cancel: cancel}
	inferencer.active = session
	inferencer.status.State = "running"
	inferencer.status.Active = true

	inferencer.handleSessionPredictionFailure(session, predictionWindow{sequenceNumber: 27}, ErrParkingInferenceComplete)
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

func TestParkingSuccessFailsWhenAppliedHoldCannotBeConfirmed(t *testing.T) {
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

	inferencer.handlePredictionFailure(predictionWindow{sequenceNumber: 28}, ErrParkingInferenceComplete)
	status := inferencer.Status()
	if status.State != "error" || !strings.Contains(status.LastError, "timed out") || !inferencer.parkingHoldFailed {
		t.Fatalf("expected an old matching applied state to fail exact-command confirmation, got=%+v", status)
	}
	if len(actuatorSink.commands) != 1 {
		t.Fatalf("expected exactly one submitted hold while confirmation timed out, got=%+v", actuatorSink.commands)
	}
}

func TestParkingSuccessFailsWhenControllerApplyFailsAsynchronously(t *testing.T) {
	actuatorSink := &recordingInferenceActuator{applyFault: "virtual controller write failed"}
	inferencer := NewInferencer(DefaultInferenceConfig(), actuator.DefaultConfig(), nil, actuatorSink)

	inferencer.handlePredictionFailure(predictionWindow{sequenceNumber: 29}, ErrParkingInferenceComplete)
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
			if len(actuatorSink.commands) != 1 || actuatorSink.commands[0].Enabled == nil || *actuatorSink.commands[0].Enabled || !actuatorSink.commands[0].Handbrake {
				t.Fatalf("expected exactly one persistent error hold, got=%+v", actuatorSink.commands)
			}
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

func TestParkingEvaluationDeadlineUsesTerminalSafetyStop(t *testing.T) {
	actuatorSink := &recordingInferenceActuator{}
	inferencer := NewInferencer(DefaultInferenceConfig(), actuator.DefaultConfig(), nil, actuatorSink)
	inferencer.parkingEvaluationLimit = 10 * time.Millisecond
	loopCtx, cancel := context.WithCancel(context.Background())
	defer cancel()
	session := &inferenceSession{cancel: cancel}
	inferencer.active = session
	inferencer.status.State = "running"
	inferencer.status.Active = true

	go inferencer.monitorParkingEvaluationDeadline(loopCtx, session)
	select {
	case <-loopCtx.Done():
	case <-time.After(time.Second):
		t.Fatal("parking evaluation deadline did not cancel inference")
	}
	status := inferencer.Status()
	if status.State != "error" || !strings.Contains(status.LastError, ErrParkingInferenceDeadlineExceeded.Error()) || status.StoppedAt != "" {
		t.Fatalf("expected deadline reason while process cleanup remains active, got=%+v", status)
	}
	if len(actuatorSink.commands) != 1 || actuatorSink.commands[0].Enabled == nil || *actuatorSink.commands[0].Enabled || !actuatorSink.commands[0].Handbrake {
		t.Fatalf("expected deadline to issue one latched safety hold, got=%+v", actuatorSink.commands)
	}
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
	if len(actuatorSink.commands) != 1 || actuatorSink.commands[0].Enabled == nil || *actuatorSink.commands[0].Enabled || !actuatorSink.commands[0].Handbrake {
		t.Fatalf("expected Stop to issue one disabled handbrake command, got=%+v", actuatorSink.commands)
	}
}

func TestParkingOperatingStateRejectsExpertPhasesAndVehicleHazards(t *testing.T) {
	tests := []struct {
		name      string
		mutate    func(*control.RuntimeTelemetry)
		wantError string
	}{
		{name: "safe ready state"},
		{
			name: "expert collection phase",
			mutate: func(telemetry *control.RuntimeTelemetry) {
				telemetry.ParkingPhase = "parking"
			},
			wantError: "phase must be ready",
		},
		{
			name: "evaluation settling phase",
			mutate: func(telemetry *control.RuntimeTelemetry) {
				telemetry.ParkingPhase = "settling"
				telemetry.ParkingAttemptCount = 0
			},
		},
		{
			name: "expert settling phase",
			mutate: func(telemetry *control.RuntimeTelemetry) {
				telemetry.ParkingPhase = "settling"
				telemetry.ParkingAttemptCount = 1
			},
			wantError: "phase must be ready",
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
				telemetry.PitchDeg = floatPtr(parkingMaximumTiltDeg + 0.1)
			},
			wantError: "tilt limit",
		},
		{
			name: "roll",
			mutate: func(telemetry *control.RuntimeTelemetry) {
				telemetry.RollDeg = floatPtr(-parkingMaximumTiltDeg - 0.1)
			},
			wantError: "tilt limit",
		},
		{
			name: "reverse motion",
			mutate: func(telemetry *control.RuntimeTelemetry) {
				telemetry.VelocityY = floatPtr(parkingReverseSpeedLimitMPS - 0.01)
			},
			wantError: "reverse motion detected",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			telemetry := validParkingRuntimeTelemetry()
			if tc.mutate != nil {
				tc.mutate(&telemetry)
			}
			err := validateParkingOperatingState(telemetry)
			if tc.wantError == "" {
				if err != nil {
					t.Fatalf("expected safe state, got=%v", err)
				}
				return
			}
			if err == nil || !errors.Is(err, ErrParkingInferencePrecondition) || !strings.Contains(err.Error(), tc.wantError) {
				t.Fatalf("expected %q safety error, got=%v", tc.wantError, err)
			}
		})
	}
}

func TestParkingOperatingStateRecognizesSucceededEvaluation(t *testing.T) {
	telemetry := validParkingRuntimeTelemetry()
	telemetry.ParkingPhase = "succeeded"
	telemetry.ParkingParked = true
	if err := validateParkingOperatingState(telemetry); !errors.Is(err, ErrParkingInferenceComplete) {
		t.Fatalf("expected succeeded evaluation terminal, got=%v", err)
	}

	telemetry.ParkingParked = false
	if err := validateParkingOperatingState(telemetry); err == nil || !errors.Is(err, ErrParkingInferencePrecondition) {
		t.Fatalf("expected inconsistent succeeded telemetry to fail closed, got=%v", err)
	}
}

func TestActiveParkingInferenceBindsTargetIdentity(t *testing.T) {
	now := time.Date(2026, 8, 5, 12, 0, 0, 0, time.UTC)
	store := control.NewStore(control.WithNowFunc(func() time.Time { return now }))
	store.UpdateTelemetry(validParkingTelemetryUpdate(now, -12))
	actuatorSink := &recordingInferenceActuator{}
	inferencer := NewInferencer(DefaultInferenceConfig(), actuator.DefaultConfig(), store, actuatorSink)
	inferencer.nowFunc = func() time.Time { return now }
	bindInferenceToCurrentParkingTarget(t, inferencer, store)

	now = now.Add(50 * time.Millisecond)
	store.UpdateTelemetry(validParkingTelemetryUpdate(now, -10))
	if err := inferencer.validateActiveParkingInference(); err != nil {
		t.Fatalf("expected motion relative to the same target to remain valid, got=%v", err)
	}

	now = now.Add(50 * time.Millisecond)
	replaced := validParkingTelemetryUpdate(now, -15)
	replaced.PositionY = floatPtr(-10)
	store.UpdateTelemetry(replaced)
	cause := inferencer.validateActiveParkingInference()
	if cause == nil || !strings.Contains(cause.Error(), "target changed") {
		t.Fatalf("expected target replacement error, got=%v", cause)
	}
	inferencer.handlePredictionFailure(predictionWindow{sequenceNumber: 31}, cause)
	if !inferencer.parkingSafetyTripped || len(actuatorSink.commands) != 1 || actuatorSink.commands[0].Enabled == nil || *actuatorSink.commands[0].Enabled {
		t.Fatalf("expected target replacement to latch-disable actuation, commands=%+v", actuatorSink.commands)
	}
}

func TestValidateParkingModelStatusRejectsTargetBlindCheckpoints(t *testing.T) {
	cfg := DefaultInferenceConfig()
	if err := validateParkingModelStatus(validParkingModelStatus(cfg), cfg); err != nil {
		t.Fatalf("expected parking-compatible model status, got=%v", err)
	}

	missingInput := validParkingModelStatus(cfg)
	missingInput.StateInputs["parking_lateral_error"] = parkingModelInputSpec{Enabled: false}
	if err := validateParkingModelStatus(missingInput, cfg); err == nil || !strings.Contains(err.Error(), "parking_lateral_error") {
		t.Fatalf("expected disabled parking input rejection, got=%v", err)
	}

	missingBrake := validParkingModelStatus(cfg)
	missingBrake.ControlTargetNames = []string{"steering", "acceleration"}
	if err := validateParkingModelStatus(missingBrake, cfg); err == nil || !strings.Contains(err.Error(), "brakePressureAvg") {
		t.Fatalf("expected missing brake head rejection, got=%v", err)
	}

	unloaded := validParkingModelStatus(cfg)
	unloaded.Loaded = false
	if err := validateParkingModelStatus(unloaded, cfg); err == nil || !errors.Is(err, ErrParkingModelIncompatible) {
		t.Fatalf("expected unloaded model rejection, got=%v", err)
	}

	reorderedFeatures := validParkingModelStatus(cfg)
	reorderedFeatures.TelemetryFeatures[0], reorderedFeatures.TelemetryFeatures[1] = reorderedFeatures.TelemetryFeatures[1], reorderedFeatures.TelemetryFeatures[0]
	if err := validateParkingModelStatus(reorderedFeatures, cfg); err == nil || !strings.Contains(err.Error(), "telemetry features") {
		t.Fatalf("expected reordered telemetry contract rejection, got=%v", err)
	}

	unexpectedInput := validParkingModelStatus(cfg)
	unexpectedInput.StateInputs["route_forward_delta"] = parkingModelInputSpec{Enabled: true}
	if err := validateParkingModelStatus(unexpectedInput, cfg); err == nil || !strings.Contains(err.Error(), "unexpected state input") {
		t.Fatalf("expected extra enabled state input rejection, got=%v", err)
	}
}

func TestPredictionModelMustRemainBoundToSessionCheckpoint(t *testing.T) {
	cfg := DefaultInferenceConfig()
	inferencer := NewInferencer(cfg, actuator.DefaultConfig(), nil)
	inferencer.parkingCheckpoint = "C:/models/parking/epoch-001.pt"
	compatible := pythonPredictResponse{
		Checkpoint:            inferencer.parkingCheckpoint,
		PlannerFormat:         cfg.PlannerFormat,
		ImageOffsets:          append([]int(nil), cfg.ImageOffsets...),
		TelemetryOffsets:      append([]int(nil), cfg.TelemetryOffsets...),
		FutureOffsets:         append([]int(nil), cfg.FutureOffsets...),
		TelemetryFeatureNames: append([]string(nil), cfg.TelemetryFeatureNames...),
		ControlTargetNames:    append([]string(nil), cfg.ControlOutputNames...),
		AuxTargetNames:        append([]string(nil), cfg.AuxOutputNames...),
		StateInputs:           validParkingModelStatus(cfg).StateInputs,
	}
	if err := inferencer.validateParkingPredictionModel(compatible); err != nil {
		t.Fatalf("expected bound prediction model, got=%v", err)
	}
	compatible.Checkpoint = "C:/models/legacy-driving/epoch-099.pt"
	if err := inferencer.validateParkingPredictionModel(compatible); err == nil || !errors.Is(err, ErrParkingModelIncompatible) {
		t.Fatalf("expected hot-swapped prediction checkpoint rejection, got=%v", err)
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
	current := validParkingTelemetryUpdate(nowValue, -1.2)
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
	current.ParkingLateralError = 0.4
	current.ParkingHeadingError = -6.5
	current.ParkingDistance = 1.3
	current.ParkingInsideBay = true
	current.ParkingAttemptIndex = 2
	current.ParkingAttemptCount = 8
	store.UpdateTelemetry(current)
	inferencer := NewInferencer(cfg, actuator.DefaultConfig(), store)
	now := time.UnixMilli(1066).UTC()
	inferencer.nowFunc = func() time.Time { return now }
	bindInferenceToCurrentParkingTarget(t, inferencer, store)
	inferencer.parkingCheckpoint = "C:/models/run-1/epoch-006.pt"
	inferencer.httpClient = &http.Client{
		Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			var payload map[string]any
			if err := json.NewDecoder(req.Body).Decode(&payload); err != nil {
				t.Fatalf("decode payload: %v", err)
			}
			expectedCurrentInputs := map[string]any{
				"currentSpeed":                  5.0,
				"routeForwardDelta":             0.25,
				"routeHeadingError":             -3.5,
				"routeDistance":                 7.25,
				"leadVehicleDistance":           12.0,
				"hasLeadVehicle":                true,
				"routeDirectionUnknown":         float64(0),
				"routeDirectionKeepStraight":    float64(1),
				"routeDirectionTurnLeft":        float64(0),
				"routeDirectionTurnRight":       float64(0),
				"routeDirectionRerouteWrongWay": float64(0),
				"routeDirectionCode":            float64(1),
				"routeDirectionDistanceM":       8.5,
				"parkingTargetConfigured":       true,
				"parkingLongitudinalError":      -1.2,
				"parkingLateralError":           0.4,
				"parkingHeadingError":           -6.5,
				"parkingDistance":               1.3,
				"parkingInsideBay":              true,
				"parkingAligned":                false,
				"parkingParked":                 false,
				"parkingAttemptIndex":           float64(2),
				"parkingAttemptCount":           float64(8),
				"parkingPhase":                  "ready",
			}
			for name, want := range expectedCurrentInputs {
				if got, ok := payload[name]; !ok || got != want {
					t.Fatalf("unexpected current telemetry input %s: got=%#v want=%#v", name, got, want)
				}
			}
			telemetry, ok := payload["telemetry"].([]any)
			if !ok || len(telemetry) != 1 {
				t.Fatalf("expected telemetry batch, got=%T %+v", payload["telemetry"], payload["telemetry"])
			}
			body, _ := json.Marshal(map[string]any{
				"checkpoint":              "C:/models/run-1/epoch-006.pt",
				"device":                  "cuda",
				"planner_format":          "temporal_telemetry_gru_v1",
				"image_offsets":           cfg.ImageOffsets,
				"telemetry_offsets":       cfg.TelemetryOffsets,
				"future_offsets":          cfg.FutureOffsets,
				"telemetry_feature_names": cfg.TelemetryFeatureNames,
				"control_target_names":    []string{"steering", "acceleration", "brakePressureAvg"},
				"aux_target_names":        cfg.AuxOutputNames,
				"state_inputs":            validParkingModelStatus(cfg).StateInputs,
				"pred_controls": [][][]float64{{
					{0.50, 0.40, 0.00},
					{0.30, 0.20, 0.00},
					{0.10, -0.10, 0.10},
					{0.00, 0.00, 0.00},
					{0.00, 0.00, 0.00},
					{0.00, 0.00, 0.00},
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

func TestPredictionResponseIsDiscardedWhenParkingHazardAppearsDuringRequest(t *testing.T) {
	cfg := DefaultInferenceConfig()
	cfg.ModelServerURL = "http://planner.local"
	cfg.ImageOffsets = []int{0}
	cfg.WindowSize = 1
	cfg.TelemetryOffsets = []int{0}
	cfg.FutureOffsets = []int{1}
	cfg.FutureSteps = 1
	now := time.UnixMilli(1000).UTC()
	store := control.NewStore(control.WithNowFunc(func() time.Time { return now }))
	store.UpdateTelemetry(validParkingTelemetryUpdate(now, -12))
	actuatorSink := &recordingInferenceActuator{}
	inferencer := NewInferencer(cfg, actuator.DefaultConfig(), store, actuatorSink)
	inferencer.nowFunc = func() time.Time { return now }
	bindInferenceToCurrentParkingTarget(t, inferencer, store)
	inferencer.parkingCheckpoint = "C:/models/parking/epoch-001.pt"
	loopCtx, cancel := context.WithCancel(context.Background())
	session := &inferenceSession{cancel: cancel}
	inferencer.active = session
	inferencer.status.State = "running"
	inferencer.status.Active = true
	inferencer.httpClient = &http.Client{
		Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			now = now.Add(10 * time.Millisecond)
			hazard := validParkingTelemetryUpdate(now, -12)
			hazard.CollisionState = "vehicle"
			store.UpdateTelemetry(hazard)
			body, _ := json.Marshal(map[string]any{
				"checkpoint":              inferencer.parkingCheckpoint,
				"device":                  "cuda",
				"planner_format":          cfg.PlannerFormat,
				"image_offsets":           cfg.ImageOffsets,
				"telemetry_offsets":       cfg.TelemetryOffsets,
				"future_offsets":          cfg.FutureOffsets,
				"telemetry_feature_names": cfg.TelemetryFeatureNames,
				"control_target_names":    cfg.ControlOutputNames,
				"aux_target_names":        cfg.AuxOutputNames,
				"state_inputs":            validParkingModelStatus(cfg).StateInputs,
				"pred_controls": [][][]float64{{
					{0.4, 0.5, 0},
				}},
			})
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

	if len(actuatorSink.commands) != 1 {
		t.Fatalf("expected exactly one safety hold and no stale enabled command, got=%+v", actuatorSink.commands)
	}
	command := actuatorSink.commands[0]
	if command.Enabled == nil || *command.Enabled || !command.Handbrake {
		t.Fatalf("expected request-time hazard to discard the prediction and latch the handbrake, got=%+v", command)
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
	if err == nil || !errors.Is(err, ErrParkingInferencePrecondition) || !strings.Contains(err.Error(), "FiveM telemetry is unavailable") {
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
	update := validParkingTelemetryUpdate(nowValue, -12)
	update.CurrentSpeed = 4.0
	store.UpdateTelemetry(update)
	inferencer := NewInferencer(cfg, actuator.DefaultConfig(), store)
	bindInferenceToCurrentParkingTarget(t, inferencer, store)

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
	if err == nil || !errors.Is(err, ErrParkingInferencePrecondition) || !strings.Contains(err.Error(), "skew exceeded") {
		t.Fatalf("expected skew beyond the configured maximum to fail closed, got=%v", err)
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
