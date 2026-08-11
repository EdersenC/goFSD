package capture

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"hash/fnv"
	"image"
	"image/jpeg"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"awesomeProject/internal/actuator"
	"awesomeProject/internal/control"
	"awesomeProject/internal/parkingcontrol"
)

const (
	defaultInferenceModelServerURL = "http://127.0.0.1:8090"
	autoInferenceSourceID          = "auto"
	defaultInferenceSourceID       = autoInferenceSourceID
	defaultInferenceFPS            = 30
	defaultInferenceWindowSize     = 3
	defaultInferenceStride         = 2
	defaultInferenceWidth          = 480
	defaultInferenceHeight         = 480
	defaultInferenceRequestTimeout = 5 * time.Second
	defaultInferenceJPEGQuality    = 90
	defaultDebugFrameDumpLimit     = 30
	defaultTelemetryStaleAfter     = 500 * time.Millisecond
	defaultParkingEvaluationLimit  = 45 * time.Second
	defaultActuatorConfirmTimeout  = 750 * time.Millisecond
	defaultActuatorConfirmInterval = 10 * time.Millisecond
	defaultFrameTimingTimeout      = 500 * time.Millisecond
	defaultModelStatusSyncTimeout  = 750 * time.Millisecond
	framePTSCadenceTolerance       = 2 * time.Millisecond
	framePTSClockLeadTolerance     = 50 * time.Millisecond
)

var inferenceShowinfoPattern = regexp.MustCompile(`\bn:\s*([0-9]+)\s+pts:\s*\S+\s+pts_time:\s*(\S+)`)

var (
	ErrInferenceAlreadyRunning      = errors.New("inference already running")
	ErrInferenceNotRunning          = errors.New("inference is not running")
	ErrInferenceStartFailed         = errors.New("failed to start inference")
	ErrInferenceActuatorUnavailable = errors.New("inference actuator is unavailable")
)

type inferenceCommandFactory func(ctx context.Context, name string, args ...string) *exec.Cmd

type InferenceStartRequest struct {
	ModelServerURL string `json:"modelServerUrl,omitempty"`
	AutoLoad       *bool  `json:"autoLoad,omitempty"`
}

type InferenceModelLoadRequest struct {
	ModelServerURL string `json:"modelServerUrl,omitempty"`
	Checkpoint     string `json:"checkpoint,omitempty"`
	Device         string `json:"device,omitempty"`
}

type InferencePrediction struct {
	Sequence                      int                       `json:"sequence"`
	FrameIndex                    int                       `json:"frameIndex"`
	SourceFPS                     int                       `json:"sourceFps"`
	InferenceHz                   int                       `json:"inferenceHz"`
	ModelServerURL                string                    `json:"modelServerUrl"`
	Checkpoint                    string                    `json:"checkpoint,omitempty"`
	ModelDevice                   string                    `json:"modelDevice,omitempty"`
	PlannerFormat                 string                    `json:"plannerFormat,omitempty"`
	PlannerFormatVersion          int                       `json:"plannerFormatVersion"`
	ControlContract               parkingControlContract    `json:"controlContract"`
	ControlHorizonDtMs            []int                     `json:"controlHorizonDtMs"`
	TelemetrySampleIntervalMs     int                       `json:"telemetrySampleIntervalMs"`
	CapturedAt                    string                    `json:"capturedAt"`
	PredictedAt                   string                    `json:"predictedAt"`
	WindowFrameIndices            []int                     `json:"windowFrameIndices"`
	WindowFrameHashes             []string                  `json:"windowFrameHashes"`
	WindowFrameTimestampsMs       []int64                   `json:"windowFrameTimestampsMs"`
	LatestFrameTimestampS         float64                   `json:"latestFrameTimestampS,omitempty"`
	TelemetryTimestampS           float64                   `json:"telemetryTimestampS,omitempty"`
	FrameID                       *int64                    `json:"frameId,omitempty"`
	CaptureLatencyMs              *float64                  `json:"captureLatencyMs,omitempty"`
	FrameTelemetrySkewMs          float64                   `json:"frameTelemetrySkewMs,omitempty"`
	FrameTelemetryAligned         bool                      `json:"frameTelemetryAligned"`
	SelectedTelemetryOffsets      []int                     `json:"selectedTelemetryOffsets"`
	SelectedTelemetryTimestampsMs []int64                   `json:"selectedTelemetryTimestampsMs"`
	ImageTensorShape              []int                     `json:"imageTensorShape"`
	TelemetryTensorShape          []int                     `json:"telemetryTensorShape"`
	PredControlsShape             []int                     `json:"predControlsShape"`
	PredAuxShape                  []int                     `json:"predAuxShape,omitempty"`
	LastTelemetry                 *control.RuntimeTelemetry `json:"lastTelemetry,omitempty"`
	RawPredControls               [][]float64               `json:"rawPredControls"`
	RawPredAux                    [][]float64               `json:"rawPredAux,omitempty"`
	RawStateInputs                map[string]any            `json:"rawStateInputs,omitempty"`
	NormalizedStateInputs         map[string]float64        `json:"normalizedStateInputs,omitempty"`
	SetpointPlan                  *parkingcontrol.Plan      `json:"setpointPlan,omitempty"`
}

type InferenceStatus struct {
	State               string               `json:"state"`
	Active              bool                 `json:"active"`
	ActuatorReady       bool                 `json:"actuatorReady"`
	ControllerReady     bool                 `json:"controllerReady"`
	CalibrationVerified bool                 `json:"calibrationVerified"`
	CalibrationID       string               `json:"calibrationId,omitempty"`
	SafetyReady         bool                 `json:"safetyReady"`
	SafetyBlocker       string               `json:"safetyBlocker,omitempty"`
	SourceID            string               `json:"sourceId,omitempty"`
	SourceFPS           int                  `json:"sourceFps"`
	InferenceHz         int                  `json:"inferenceHz"`
	WindowSize          int                  `json:"windowSize"`
	FrameStride         int                  `json:"frameStride"`
	DispatchStride      int                  `json:"dispatchStride"`
	FrameWidth          int                  `json:"frameWidth"`
	FrameHeight         int                  `json:"frameHeight"`
	ModelServerURL      string               `json:"modelServerUrl,omitempty"`
	LoadedCheckpoint    string               `json:"loadedCheckpoint,omitempty"`
	LoadedModelDevice   string               `json:"loadedModelDevice,omitempty"`
	StartedAt           string               `json:"startedAt,omitempty"`
	StoppedAt           string               `json:"stoppedAt,omitempty"`
	DebugFramesDir      string               `json:"debugFramesDir,omitempty"`
	DebugFramesSaved    int                  `json:"debugFramesSaved"`
	DebugFramesLimit    int                  `json:"debugFramesLimit"`
	LastPrediction      *InferencePrediction `json:"lastPrediction,omitempty"`
	FramesSeen          int                  `json:"framesSeen"`
	PredictionsSent     int                  `json:"predictionsSent"`
	PredictionErrors    int                  `json:"predictionErrors"`
	LastError           string               `json:"lastError,omitempty"`
}

type inferenceSession struct {
	ctx          context.Context
	cancel       context.CancelFunc
	stdin        io.WriteCloser
	stdout       io.ReadCloser
	stderr       io.ReadCloser
	cmd          *exec.Cmd
	done         chan error
	predictQ     chan predictionWindow
	frameTimings chan inferenceFrameTimingEvent
	frameDump    *debugFrameDump
}

type debugFrameDump struct {
	dir   string
	limit int
	saved int
}

type predictionWindow struct {
	frames         []*image.RGBA
	frameIndex     int
	frameIndices   []int
	frameTimes     []time.Time
	capturedAt     time.Time
	sequenceNumber int
}

type bufferedInferenceFrame struct {
	index      int
	capturedAt time.Time
	image      *image.RGBA
}

type inferenceFrameTiming struct {
	index      int
	pts        time.Duration
	observedAt time.Time
}

type inferenceFrameTimingEvent struct {
	timing inferenceFrameTiming
	err    error
}

type inferenceFrameClock struct {
	frameInterval time.Duration
	initialized   bool
	firstPTS      time.Duration
	anchorAt      time.Time
	lastIndex     int
	lastPTS       time.Duration
	lastObserved  time.Time
}

type actuatorSubmitter interface {
	Submit(req actuator.CommandRequest) (actuator.State, error)
}

type actuatorParkingPlanSubmitter interface {
	SubmitParkingSetpointPlan(plan parkingcontrol.Plan) (actuator.State, error)
}

type actuatorParkingSafetyStopper interface {
	RequestParkingSafetyStop() (actuator.State, error)
}

type actuatorStateProvider interface {
	State() actuator.State
}

type actuatorApplyExpectation struct {
	label    string
	stopping bool
}

type pythonPredictResponse struct {
	Checkpoint                string                           `json:"checkpoint"`
	Device                    string                           `json:"device"`
	PlannerFormat             string                           `json:"planner_format"`
	PlannerFormatVersion      int                              `json:"planner_format_version"`
	ControlContract           parkingControlContract           `json:"control_contract"`
	ControlHorizonDtMs        []int                            `json:"control_horizon_dt_ms"`
	TelemetrySampleIntervalMs int                              `json:"telemetry_sample_interval_ms"`
	SampledAtS                float64                          `json:"sampled_at_s"`
	Direction                 string                           `json:"direction"`
	ImageOffsets              []int                            `json:"image_offsets"`
	TelemetryOffsets          []int                            `json:"telemetry_offsets"`
	TelemetryFeatureNames     []string                         `json:"telemetry_feature_names"`
	ControlTargetNames        []string                         `json:"control_target_names"`
	AuxTargetNames            []string                         `json:"aux_target_names"`
	PredControls              [][][]float64                    `json:"pred_controls"`
	PredAux                   [][][]float64                    `json:"pred_aux"`
	FutureOffsets             []int                            `json:"future_offsets"`
	StateInputs               map[string]parkingModelInputSpec `json:"state_inputs"`
	RawStateInputs            map[string]any                   `json:"raw_state_inputs"`
	NormalizedStateInputs     map[string]float64               `json:"normalized_state_inputs"`
}

type pythonModelsResponse struct {
	Models []InferenceModelOption `json:"models"`
}

type Inferencer struct {
	mu                          sync.Mutex
	lifecycleMu                 sync.Mutex
	actuationMu                 sync.Mutex
	ffmpegBin                   string
	discover                    SourceDiscovery
	probe                       CapabilityProbe
	newCommand                  inferenceCommandFactory
	httpClient                  *http.Client
	nowFunc                     func() time.Time
	requestTimeout              time.Duration
	config                      InferenceConfig
	modelServerURL              string
	autoLoad                    bool
	loadedCheckpoint            string
	loadedModelDevice           string
	sourceID                    string
	telemetry                   *control.Store
	actuator                    actuatorSubmitter
	telemetryStaleAfter         time.Duration
	lastTelemetryWaitLog        time.Time
	telemetryNormalizer         *telemetryNormalizer
	normalizationErr            error
	parkingTarget               *parkingInferenceTarget
	parkingCheckpoint           string
	parkingSafetyTripped        bool
	parkingCompleted            bool
	parkingManualStop           bool
	parkingHoldFailed           bool
	parkingHoldConfirmed        bool
	parkingStartEnvelopePending bool
	parkingEvaluationLimit      time.Duration
	actuatorConfirmTimeout      time.Duration
	actuatorConfirmInterval     time.Duration
	status                      InferenceStatus
	active                      *inferenceSession
}

func NewInferencer(cfg InferenceConfig, _ actuator.Config, telemetry *control.Store, actuators ...actuatorSubmitter) *Inferencer {
	if cfg.ModelServerURL == "" {
		cfg = DefaultInferenceConfig()
	}
	var actuatorSink actuatorSubmitter
	if len(actuators) > 0 {
		actuatorSink = actuators[0]
	}
	inf := &Inferencer{
		ffmpegBin: envOrDefault("FFMPEG_BIN", defaultFFmpegBin),
		discover:  discoverSources,
		probe:     probeFFmpegCapability,
		newCommand: func(ctx context.Context, name string, args ...string) *exec.Cmd {
			return exec.CommandContext(ctx, name, args...)
		},
		httpClient:              &http.Client{Timeout: cfg.RequestTimeout},
		nowFunc:                 time.Now,
		requestTimeout:          cfg.RequestTimeout,
		config:                  cfg,
		modelServerURL:          cfg.ModelServerURL,
		autoLoad:                cfg.AutoLoad,
		sourceID:                cfg.SourceID,
		telemetry:               telemetry,
		actuator:                actuatorSink,
		telemetryStaleAfter:     defaultTelemetryStaleAfter,
		parkingEvaluationLimit:  defaultParkingEvaluationLimit,
		actuatorConfirmTimeout:  defaultActuatorConfirmTimeout,
		actuatorConfirmInterval: defaultActuatorConfirmInterval,
		status: InferenceStatus{
			State:          "idle",
			SourceFPS:      cfg.FPS,
			InferenceHz:    cfg.FPS / cfg.DispatchStride,
			WindowSize:     cfg.WindowSize,
			FrameStride:    cfg.FrameStride,
			DispatchStride: cfg.DispatchStride,
			FrameWidth:     cfg.FrameWidth,
			FrameHeight:    cfg.FrameHeight,
			SourceID:       cfg.SourceID,
		},
	}
	if cfg.TelemetryNormalizationEnabled {
		inf.telemetryNormalizer, inf.normalizationErr = loadTelemetryNormalizer(cfg.TelemetryNormalizationStatsPath)
	}
	return inf
}

func (i *Inferencer) Status() InferenceStatus {
	i.mu.Lock()
	status := cloneInferenceStatus(i.status)
	status.LoadedCheckpoint = i.loadedCheckpoint
	status.LoadedModelDevice = i.loadedModelDevice
	i.mu.Unlock()

	i.populateInferencePreflight(&status)
	return status
}

func (i *Inferencer) populateInferencePreflight(status *InferenceStatus) {
	if status == nil {
		return
	}

	if provider, ok := i.actuator.(actuatorStateProvider); ok {
		state := provider.State()
		status.ActuatorReady = state.Supported && state.Ready
		status.ControllerReady = state.ParkingController.Ready
		status.CalibrationVerified = state.ParkingController.Calibration.Verified
		status.CalibrationID = strings.TrimSpace(state.ParkingController.Calibration.ProfileID)
	}

	if status.Active {
		status.SafetyBlocker = "a self-driving test is already active"
		return
	}
	if err := i.validateParkingInferenceStart(); err != nil {
		status.SafetyBlocker = inferencePreflightMessage(err)
		return
	}
	if err := i.validateInferenceActuatorReady(); err != nil {
		status.SafetyBlocker = inferencePreflightMessage(err)
		return
	}
	status.SafetyReady = true
}

func inferencePreflightMessage(err error) string {
	message := strings.TrimSpace(err.Error())
	for _, prefix := range []string{
		ErrParkingInferencePrecondition.Error() + ":",
		ErrInferenceActuatorUnavailable.Error() + ":",
	} {
		message = strings.TrimSpace(strings.TrimPrefix(message, prefix))
	}
	return message
}

func (i *Inferencer) Start(ctx context.Context, req InferenceStartRequest) (InferenceStatus, error) {
	i.lifecycleMu.Lock()
	defer i.lifecycleMu.Unlock()
	i.mu.Lock()
	alreadyRunning := i.active != nil
	i.mu.Unlock()
	if alreadyRunning {
		return InferenceStatus{}, ErrInferenceAlreadyRunning
	}

	modelServerURL := strings.TrimRight(strings.TrimSpace(req.ModelServerURL), "/")
	if modelServerURL == "" {
		modelServerURL = i.modelServerURL
	}
	autoLoad := i.autoLoad
	if req.AutoLoad != nil {
		autoLoad = *req.AutoLoad
	}
	parkingTarget, err := i.parkingInferenceStartTarget()
	if err != nil {
		return InferenceStatus{}, fmt.Errorf("%w: %w", ErrInferenceStartFailed, err)
	}
	if err := i.validateInferenceActuatorReady(); err != nil {
		return InferenceStatus{}, fmt.Errorf("%w: %w", ErrInferenceStartFailed, err)
	}

	sources, err := i.discover(ctx)
	if err != nil {
		return InferenceStatus{}, err
	}

	monitor, ok := resolveInferenceMonitor(sources, i.sourceID)
	if !ok {
		return InferenceStatus{}, ErrSourceNotFound
	}

	supportsDDAGrab, err := i.probe(ctx, i.ffmpegBin, "ddagrab")
	if err != nil {
		return InferenceStatus{}, fmt.Errorf("%w: %v", ErrInferenceStartFailed, err)
	}
	if !supportsDDAGrab {
		return InferenceStatus{}, ErrUnsupportedFFmpeg
	}
	if i.normalizationErr != nil {
		return InferenceStatus{}, fmt.Errorf("%w: %v", ErrInferenceStartFailed, i.normalizationErr)
	}

	if autoLoad {
		if err := i.loadRemoteModel(ctx, modelServerURL); err != nil {
			return InferenceStatus{}, fmt.Errorf("%w: %v", ErrInferenceStartFailed, err)
		}
	}
	modelStatus, err := i.validateLoadedParkingModel(ctx, modelServerURL)
	if err != nil {
		return InferenceStatus{}, fmt.Errorf("%w: %w", ErrInferenceStartFailed, err)
	}
	parkingTarget, err = i.parkingInferenceStartTarget()
	if err != nil {
		return InferenceStatus{}, fmt.Errorf("%w: %w", ErrInferenceStartFailed, err)
	}
	if err := inferenceStartRequestError(ctx); err != nil {
		return InferenceStatus{}, err
	}

	i.mu.Lock()
	if i.active != nil {
		i.mu.Unlock()
		return InferenceStatus{}, ErrInferenceAlreadyRunning
	}

	previousStatus := cloneInferenceStatus(i.status)
	startedAt := i.nowFunc().UTC()
	status := InferenceStatus{
		State:            "starting",
		SourceID:         monitor.ID,
		SourceFPS:        i.config.FPS,
		InferenceHz:      i.config.FPS / i.config.DispatchStride,
		WindowSize:       i.config.WindowSize,
		FrameStride:      i.config.FrameStride,
		DispatchStride:   i.config.DispatchStride,
		FrameWidth:       i.config.FrameWidth,
		FrameHeight:      i.config.FrameHeight,
		ModelServerURL:   modelServerURL,
		StartedAt:        startedAt.Format(time.RFC3339Nano),
		DebugFramesLimit: defaultDebugFrameDumpLimit,
	}
	i.status = status
	i.mu.Unlock()

	spec := monitorCaptureSpec(monitor)
	loopCtx, cancel := context.WithCancel(context.Background())
	detachRequestCancellation := context.AfterFunc(ctx, cancel)
	defer detachRequestCancellation()
	args := buildInferenceFFmpegArgs(spec, i.config)
	cmd := i.newCommand(loopCtx, i.ffmpegBin, args...)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		cancel()
		i.setInferenceError(err)
		return InferenceStatus{}, fmt.Errorf("%w: %v", ErrInferenceStartFailed, err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		cancel()
		i.setInferenceError(err)
		return InferenceStatus{}, fmt.Errorf("%w: %v", ErrInferenceStartFailed, err)
	}
	stdin, err := cmd.StdinPipe()
	if err != nil {
		cancel()
		i.setInferenceError(err)
		return InferenceStatus{}, fmt.Errorf("%w: %v", ErrInferenceStartFailed, err)
	}
	if err := inferenceStartRequestError(ctx); err != nil {
		return InferenceStatus{}, i.rollbackCanceledInferenceStart(previousStatus, cancel, false, err)
	}
	if err := i.armActuatorForParkingInference(); err != nil {
		cancel()
		if holdErr := i.submitParkingSafetyHold(0); holdErr != nil {
			err = fmt.Errorf("%w; failed to establish the actuator-owned parking safety stop: %v", err, holdErr)
		}
		i.setInferenceError(err)
		return InferenceStatus{}, fmt.Errorf("%w: failed to establish parking actuator ownership: %v", ErrInferenceStartFailed, err)
	}
	if err := inferenceStartRequestError(ctx); err != nil {
		return InferenceStatus{}, i.rollbackCanceledInferenceStart(previousStatus, cancel, true, err)
	}
	if err := cmd.Start(); err != nil {
		if requestErr := inferenceStartRequestError(ctx); requestErr != nil {
			return InferenceStatus{}, i.rollbackCanceledInferenceStart(previousStatus, cancel, true, requestErr)
		}
		cancel()
		if holdErr := i.submitParkingSafetyHold(0); holdErr != nil {
			err = fmt.Errorf("%w; failed to restore the parking safety hold: %v", err, holdErr)
		}
		i.setInferenceError(err)
		return InferenceStatus{}, fmt.Errorf("%w: %v", ErrInferenceStartFailed, err)
	}
	detached := detachRequestCancellation()
	requestErr := inferenceStartRequestError(ctx)
	if !detached || requestErr != nil {
		if requestErr == nil {
			requestErr = fmt.Errorf("%w: request ended while inference capture was starting", ErrInferenceStartFailed)
		}
		cancel()
		if cmd.Process != nil {
			_ = cmd.Process.Kill()
		}
		_ = stdin.Close()
		_ = cmd.Wait()
		return InferenceStatus{}, i.rollbackCanceledInferenceStart(previousStatus, cancel, true, requestErr)
	}

	frameDump, err := newDebugFrameDump(i.nowFunc())
	if err != nil {
		i.mu.Lock()
		i.status.LastError = fmt.Sprintf("failed to prepare debug frame dump: %v", err)
		i.mu.Unlock()
	}

	session := &inferenceSession{
		ctx:          loopCtx,
		cancel:       cancel,
		stdin:        stdin,
		stdout:       stdout,
		stderr:       stderr,
		cmd:          cmd,
		done:         make(chan error, 1),
		predictQ:     make(chan predictionWindow, 1),
		frameTimings: make(chan inferenceFrameTimingEvent, inferenceFrameTimingBufferSize(i.config.FPS)),
		frameDump:    frameDump,
	}

	i.mu.Lock()
	i.active = session
	i.status.Active = true
	i.parkingTarget = &parkingTarget
	i.parkingCheckpoint = strings.TrimSpace(modelStatus.Checkpoint)
	i.parkingSafetyTripped = false
	i.parkingCompleted = false
	i.parkingManualStop = false
	i.parkingHoldFailed = false
	i.parkingHoldConfirmed = false
	i.parkingStartEnvelopePending = true
	i.status.State = "running"
	i.status.SourceID = monitor.ID
	i.status.ModelServerURL = modelServerURL
	if frameDump != nil {
		i.status.DebugFramesDir = frameDump.dir
		i.status.DebugFramesLimit = frameDump.limit
		i.status.DebugFramesSaved = frameDump.saved
	}
	status = cloneInferenceStatus(i.status)
	i.mu.Unlock()

	go i.consumeInferenceStderr(session)
	go i.runPredictionWorker(loopCtx, session, modelServerURL)
	go i.consumeInferenceFrames(loopCtx, session)
	go i.monitorParkingEvaluationDeadline(loopCtx, session)
	go i.waitForInference(session)

	return status, nil
}

func inferenceStartRequestError(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("%w: request ended before inference startup committed: %w", ErrInferenceStartFailed, err)
	}
	return nil
}

func (i *Inferencer) rollbackCanceledInferenceStart(previousStatus InferenceStatus, cancel context.CancelFunc, armed bool, cause error) error {
	cancel()
	if armed {
		if holdErr := i.submitParkingSafetyHold(0); holdErr != nil {
			combined := fmt.Errorf("%w; failed to restore the parking safety hold: %v", cause, holdErr)
			i.setInferenceError(combined)
			return combined
		}
	}

	i.mu.Lock()
	if i.active == nil {
		i.status = cloneInferenceStatus(previousStatus)
	}
	i.mu.Unlock()
	return cause
}

func resolveInferenceMonitor(sources []Source, configuredSourceID string) (Source, bool) {
	configuredSourceID = strings.TrimSpace(configuredSourceID)
	if configuredSourceID != "" && !strings.EqualFold(configuredSourceID, autoInferenceSourceID) {
		return monitorByID(sources, configuredSourceID)
	}

	if window, ok := preferredWindowSource(sources); ok {
		if monitor, found := bestMonitorForWindow(sources, window); found {
			return monitor, true
		}
	}
	if window, ok := anyWindowSource(sources); ok {
		if monitor, found := bestMonitorForWindow(sources, window); found {
			return monitor, true
		}
	}
	return fallbackMonitorSource(sources)
}

func (i *Inferencer) Models(ctx context.Context, modelServerURL string) ([]InferenceModelOption, error) {
	modelServerURL = strings.TrimRight(strings.TrimSpace(modelServerURL), "/")
	if modelServerURL == "" {
		modelServerURL = i.modelServerURL
	}
	requestCtx, cancel := context.WithTimeout(ctx, i.requestTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(requestCtx, http.MethodGet, modelServerURL+"/models", nil)
	if err != nil {
		return nil, err
	}
	resp, err := i.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		payload, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("python model list failed: status=%d body=%s", resp.StatusCode, strings.TrimSpace(string(payload)))
	}
	var parsed pythonModelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return nil, err
	}
	i.syncRemoteLoadedModel(ctx, modelServerURL)
	return parsed.Models, nil
}

func (i *Inferencer) LoadModel(ctx context.Context, req InferenceModelLoadRequest) (map[string]any, error) {
	i.lifecycleMu.Lock()
	defer i.lifecycleMu.Unlock()
	i.mu.Lock()
	alreadyRunning := i.active != nil
	i.mu.Unlock()
	if alreadyRunning {
		return nil, ErrInferenceAlreadyRunning
	}

	modelServerURL := strings.TrimRight(strings.TrimSpace(req.ModelServerURL), "/")
	if modelServerURL == "" {
		modelServerURL = i.modelServerURL
	}
	payload := map[string]any{}
	if i.config.ConfigPath != "" {
		payload["config"] = i.config.ConfigPath
	}
	if checkpoint := strings.TrimSpace(req.Checkpoint); checkpoint != "" {
		payload["checkpoint"] = checkpoint
	}
	device := strings.TrimSpace(req.Device)
	if device == "" {
		device = i.config.ModelDevice
	}
	if device != "" {
		payload["device"] = device
	}
	parsed, err := i.postRemoteModelLoad(ctx, modelServerURL, payload)
	if err != nil {
		i.clearLoadedModel()
		return nil, err
	}
	modelStatus, err := i.validateLoadedParkingModel(ctx, modelServerURL)
	if err != nil {
		i.clearLoadedModel()
		return nil, fmt.Errorf("loaded checkpoint failed the parking compatibility check: %w", err)
	}

	resolvedDevice := strings.TrimSpace(modelStatus.Device)
	if resolvedDevice == "" {
		if value, ok := parsed["device"].(string); ok {
			resolvedDevice = strings.TrimSpace(value)
		}
	}
	if resolvedDevice == "" {
		resolvedDevice = device
	}
	i.setLoadedModel(modelServerURL, modelStatus.Checkpoint, resolvedDevice)
	parsed["status"] = "loaded"
	parsed["checkpoint"] = strings.TrimSpace(modelStatus.Checkpoint)
	if resolvedDevice != "" {
		parsed["device"] = resolvedDevice
	}
	return parsed, nil
}

func (i *Inferencer) syncRemoteLoadedModel(ctx context.Context, modelServerURL string) {
	syncCtx, cancel := context.WithTimeout(ctx, defaultModelStatusSyncTimeout)
	defer cancel()
	status, err := i.fetchParkingModelStatus(syncCtx, modelServerURL)
	if err != nil {
		return
	}
	if err := validateParkingModelStatus(status, i.config); err != nil {
		i.clearLoadedModel()
		return
	}
	i.setLoadedModel(modelServerURL, status.Checkpoint, status.Device)
}

func (i *Inferencer) setLoadedModel(modelServerURL string, checkpoint string, device string) {
	i.mu.Lock()
	defer i.mu.Unlock()
	i.modelServerURL = modelServerURL
	i.loadedCheckpoint = strings.TrimSpace(checkpoint)
	i.loadedModelDevice = strings.TrimSpace(device)
	if i.loadedModelDevice == "" {
		i.loadedModelDevice = strings.TrimSpace(i.config.ModelDevice)
	}
	if i.loadedCheckpoint == "" {
		i.loadedModelDevice = ""
	}
}

func (i *Inferencer) clearLoadedModel() {
	i.mu.Lock()
	defer i.mu.Unlock()
	i.loadedCheckpoint = ""
	i.loadedModelDevice = ""
}

func (i *Inferencer) Stop(ctx context.Context) (InferenceStatus, error) {
	i.lifecycleMu.Lock()
	defer i.lifecycleMu.Unlock()

	i.mu.Lock()
	session := i.active
	if session == nil {
		i.mu.Unlock()
		return InferenceStatus{}, ErrInferenceNotRunning
	}
	i.mu.Unlock()

	i.actuationMu.Lock()
	i.mu.Lock()
	if i.active != session {
		i.mu.Unlock()
		i.actuationMu.Unlock()
		return i.Status(), nil
	}
	shouldHold := !i.parkingSafetyTripped
	if shouldHold {
		i.parkingSafetyTripped = true
		i.parkingManualStop = true
		i.status.State = "stopping"
		i.status.StoppedAt = ""
	}
	i.mu.Unlock()

	var holdErr error
	if shouldHold {
		holdErr = i.submitParkingSafetyHoldLocked(0)
		if holdErr != nil {
			i.mu.Lock()
			i.parkingHoldFailed = true
			i.status.State = "error"
			i.status.LastError = fmt.Sprintf("failed to apply parking safety hold while stopping: %v", holdErr)
			i.mu.Unlock()
		}
	}
	i.actuationMu.Unlock()
	if session.cancel != nil {
		session.cancel()
	}
	if session.stdin != nil {
		_, _ = io.WriteString(session.stdin, "q\n")
		_ = session.stdin.Close()
	}

	timeout := 5 * time.Second
	if deadline, ok := ctx.Deadline(); ok {
		remaining := time.Until(deadline)
		if remaining > 0 && remaining < timeout {
			timeout = remaining
		}
	}

	select {
	case err := <-session.done:
		if err != nil && !isExpectedExitErr(err) {
			return InferenceStatus{}, err
		}
	case <-time.After(timeout):
		if session.cmd.Process != nil {
			_ = session.cmd.Process.Kill()
		}
		select {
		case err := <-session.done:
			if err != nil && !isExpectedExitErr(err) {
				return InferenceStatus{}, err
			}
		case <-time.After(2 * time.Second):
			return InferenceStatus{}, fmt.Errorf("%w: process did not exit", ErrStopFailed)
		}
	case <-ctx.Done():
		return InferenceStatus{}, ctx.Err()
	}

	if holdErr != nil {
		return i.Status(), fmt.Errorf("%w: failed to apply parking safety hold: %v", ErrStopFailed, holdErr)
	}
	return i.Status(), nil
}

func (i *Inferencer) waitForInference(session *inferenceSession) {
	err := session.cmd.Wait()
	i.handleInferenceProcessExit(session, err)
	i.finishInferenceSession(session, err)
	session.done <- err
}

func (i *Inferencer) finishInferenceSession(session *inferenceSession, err error) {
	i.actuationMu.Lock()
	defer i.actuationMu.Unlock()
	i.mu.Lock()
	defer i.mu.Unlock()
	active := i.active
	if active != nil && active == session {
		completed := i.parkingCompleted
		safetyTripped := i.parkingSafetyTripped
		manualStop := i.parkingManualStop
		holdFailed := i.parkingHoldFailed
		holdConfirmed := i.parkingHoldConfirmed
		i.active = nil
		i.status.Active = false
		i.parkingTarget = nil
		i.parkingCheckpoint = ""
		i.parkingStartEnvelopePending = false
		i.status.StoppedAt = i.nowFunc().UTC().Format(time.RFC3339Nano)
		switch {
		case holdFailed:
			i.status.State = "error"
			if i.status.LastError == "" {
				i.status.LastError = "parking inference stopped because the safety hold failed"
			}
		case completed && !holdConfirmed:
			i.status.State = "error"
			i.status.LastError = "parking inference stopped before the safety hold was confirmed"
		case completed:
			i.status.State = "succeeded"
			i.status.LastError = ""
		case manualStop:
			i.status.State = "idle"
			i.parkingSafetyTripped = false
			i.parkingCompleted = false
			i.parkingManualStop = false
			i.parkingHoldFailed = false
			i.parkingHoldConfirmed = false
		case safetyTripped:
			i.status.State = "error"
			if i.status.LastError == "" {
				i.status.LastError = "parking inference stopped by the safety interlock"
			}
		case err != nil:
			i.status.State = "error"
			i.status.LastError = err.Error()
		default:
			i.status.State = "idle"
		}
	}
}

func (i *Inferencer) consumeInferenceStderr(session *inferenceSession) {
	defer func() {
		_ = session.stderr.Close()
		if session.frameTimings != nil {
			close(session.frameTimings)
		}
	}()

	scanner := bufio.NewScanner(session.stderr)
	scanner.Buffer(make([]byte, 4096), 256*1024)
	diagnostics := make([]string, 0, 4)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		timing, matched, err := parseInferenceFrameTiming(line, i.nowFunc().UTC())
		if matched {
			if !publishInferenceFrameTiming(session, inferenceFrameTimingEvent{timing: timing, err: err}) {
				return
			}
			continue
		}
		if isInferenceFFmpegErrorLine(line) && len(diagnostics) < 4 {
			diagnostics = append(diagnostics, line)
		}
	}
	if err := scanner.Err(); err != nil {
		_ = publishInferenceFrameTiming(session, inferenceFrameTimingEvent{
			err: fmt.Errorf("failed to read FFmpeg frame timing metadata: %w", err),
		})
		diagnostics = append(diagnostics, err.Error())
	}

	message := strings.TrimSpace(strings.Join(diagnostics, "\n"))
	if message != "" {
		i.mu.Lock()
		if i.status.LastError == "" && i.status.State != "succeeded" {
			i.status.LastError = message
		}
		i.mu.Unlock()
	}
}

func (i *Inferencer) consumeInferenceFrames(ctx context.Context, session *inferenceSession) {
	defer func() { _ = session.stdout.Close() }()
	frameBytes := make([]byte, inferenceRawFrameBytes(i.config.FrameWidth, i.config.FrameHeight))
	buffer := make([]bufferedInferenceFrame, 0, requiredFrameCount(i.config.ImageOffsets)+1)
	frameClock := newInferenceFrameClock(i.config.FPS)
	frameIndex := 0
	sequence := 0

	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		if _, err := io.ReadFull(session.stdout, frameBytes); err != nil {
			if ctx.Err() == nil {
				i.handleSessionPredictionFailure(session, predictionWindow{}, fmt.Errorf("inference frame stream ended unexpectedly: %w", err))
			}
			return
		}

		capturedAt, err := receiveInferenceFrameTimestamp(ctx, session, frameClock, frameIndex)
		if err != nil {
			if ctx.Err() == nil {
				i.handleSessionPredictionFailure(session, predictionWindow{}, fmt.Errorf("inference frame timing failed: %w", err))
			}
			return
		}
		frame := bufferedInferenceFrame{
			index:      frameIndex,
			capturedAt: capturedAt,
			image:      rgbFrameToRGBA(frameBytes, i.config.FrameWidth, i.config.FrameHeight),
		}
		if saved, err := maybeDumpDebugFrame(session.frameDump, frame); err == nil {
			i.mu.Lock()
			i.status.DebugFramesSaved = saved
			i.mu.Unlock()
		}
		buffer = append(buffer, frame)
		if len(buffer) > requiredFrameCount(i.config.ImageOffsets)+1 {
			buffer = buffer[1:]
		}

		i.mu.Lock()
		i.status.FramesSeen++
		i.mu.Unlock()

		if shouldDispatchInferenceFrame(frameIndex, i.config.ImageOffsets, i.config.DispatchStride) {
			sequence++
			window := buildPredictionWindow(buffer, sequence, i.config.ImageOffsets)
			if window != nil {
				i.enqueuePredictionWindow(session, *window)
			}
		}

		frameIndex++
	}
}

func parseInferenceFrameTiming(line string, observedAt time.Time) (inferenceFrameTiming, bool, error) {
	match := inferenceShowinfoPattern.FindStringSubmatch(line)
	if len(match) == 0 {
		isMalformedTiming := strings.Contains(strings.ToLower(line), "showinfo") &&
			strings.Contains(line, " n:") &&
			strings.Contains(line, "pts_time:")
		if isMalformedTiming {
			return inferenceFrameTiming{}, true, fmt.Errorf("malformed FFmpeg showinfo timing line: %q", line)
		}
		return inferenceFrameTiming{}, false, nil
	}
	if observedAt.IsZero() {
		return inferenceFrameTiming{}, true, errors.New("FFmpeg showinfo timing observation has no wall-clock timestamp")
	}
	index, err := strconv.Atoi(match[1])
	if err != nil || index < 0 {
		return inferenceFrameTiming{}, true, fmt.Errorf("invalid FFmpeg showinfo frame index %q", match[1])
	}
	ptsSeconds, err := strconv.ParseFloat(match[2], 64)
	if err != nil || math.IsNaN(ptsSeconds) || math.IsInf(ptsSeconds, 0) || ptsSeconds < 0 {
		return inferenceFrameTiming{}, true, fmt.Errorf("invalid FFmpeg showinfo pts_time %q for frame %d", match[2], index)
	}
	if ptsSeconds > float64((24*time.Hour)/time.Second) {
		return inferenceFrameTiming{}, true, fmt.Errorf("FFmpeg showinfo pts_time %.6f exceeds the supported capture duration", ptsSeconds)
	}
	return inferenceFrameTiming{
		index:      index,
		pts:        time.Duration(math.Round(ptsSeconds * float64(time.Second))),
		observedAt: observedAt,
	}, true, nil
}

func publishInferenceFrameTiming(session *inferenceSession, event inferenceFrameTimingEvent) bool {
	if session == nil || session.frameTimings == nil {
		return false
	}
	if session.ctx == nil {
		session.frameTimings <- event
		return true
	}
	select {
	case session.frameTimings <- event:
		return true
	case <-session.ctx.Done():
		return false
	}
}

func receiveInferenceFrameTimestamp(
	ctx context.Context,
	session *inferenceSession,
	clock *inferenceFrameClock,
	expectedIndex int,
) (time.Time, error) {
	if session == nil || session.frameTimings == nil {
		return time.Time{}, errors.New("FFmpeg frame timing metadata stream is unavailable")
	}
	timer := time.NewTimer(defaultFrameTimingTimeout)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return time.Time{}, ctx.Err()
	case <-timer.C:
		return time.Time{}, fmt.Errorf("timed out waiting for FFmpeg timing metadata for frame %d", expectedIndex)
	case event, ok := <-session.frameTimings:
		if !ok {
			return time.Time{}, fmt.Errorf("FFmpeg timing metadata ended before frame %d", expectedIndex)
		}
		if event.err != nil {
			return time.Time{}, event.err
		}
		return clock.resolve(expectedIndex, event.timing)
	}
}

func newInferenceFrameClock(fps int) *inferenceFrameClock {
	if fps < 1 {
		return &inferenceFrameClock{}
	}
	return &inferenceFrameClock{frameInterval: time.Second / time.Duration(fps), lastIndex: -1}
}

func (c *inferenceFrameClock) resolve(expectedIndex int, timing inferenceFrameTiming) (time.Time, error) {
	if c == nil || c.frameInterval <= 0 {
		return time.Time{}, errors.New("inference frame clock has an invalid frame interval")
	}
	if expectedIndex < 0 || timing.index != expectedIndex {
		return time.Time{}, fmt.Errorf(
			"FFmpeg timing frame index mismatch: got %d, expected %d",
			timing.index,
			expectedIndex,
		)
	}
	if timing.observedAt.IsZero() {
		return time.Time{}, fmt.Errorf("FFmpeg timing for frame %d has no observation timestamp", timing.index)
	}
	if !c.initialized {
		if expectedIndex != 0 {
			return time.Time{}, fmt.Errorf("FFmpeg timing began at frame %d instead of frame 0", expectedIndex)
		}
		c.initialized = true
		c.firstPTS = timing.pts
		c.anchorAt = timing.observedAt
		c.lastIndex = timing.index
		c.lastPTS = timing.pts
		c.lastObserved = timing.observedAt
		return timing.observedAt, nil
	}
	if timing.index != c.lastIndex+1 {
		return time.Time{}, fmt.Errorf(
			"FFmpeg timing is out of order: got frame %d after frame %d",
			timing.index,
			c.lastIndex,
		)
	}
	if timing.pts <= c.lastPTS {
		return time.Time{}, fmt.Errorf(
			"FFmpeg PTS is not increasing at frame %d: got %s after %s",
			timing.index,
			timing.pts,
			c.lastPTS,
		)
	}
	if timing.observedAt.Before(c.lastObserved) {
		return time.Time{}, fmt.Errorf("FFmpeg timing observation moved backwards at frame %d", timing.index)
	}

	ptsElapsed := timing.pts - c.firstPTS
	expectedElapsed := time.Duration(timing.index) * c.frameInterval
	if durationDistance(ptsElapsed, expectedElapsed) > framePTSCadenceTolerance {
		return time.Time{}, fmt.Errorf(
			"FFmpeg PTS cadence mismatch at frame %d: got %s elapsed, expected %s",
			timing.index,
			ptsElapsed,
			expectedElapsed,
		)
	}
	capturedAt := c.anchorAt.Add(ptsElapsed)
	if capturedAt.After(timing.observedAt.Add(framePTSClockLeadTolerance)) {
		return time.Time{}, fmt.Errorf(
			"FFmpeg PTS clock leads its observation by %s at frame %d",
			capturedAt.Sub(timing.observedAt),
			timing.index,
		)
	}

	c.lastIndex = timing.index
	c.lastPTS = timing.pts
	c.lastObserved = timing.observedAt
	return capturedAt, nil
}

func durationDistance(left, right time.Duration) time.Duration {
	if left >= right {
		return left - right
	}
	return right - left
}

func inferenceFrameTimingBufferSize(fps int) int {
	if fps < 1 {
		return 2
	}
	return fps * 2
}

func isInferenceFFmpegErrorLine(line string) bool {
	lower := strings.ToLower(strings.TrimSpace(line))
	if lower == "" || strings.Contains(lower, "showinfo") {
		return false
	}
	for _, marker := range []string{"error", "failed", "invalid", "unable", "not found", "no such", "cannot", "could not", "broken pipe"} {
		if strings.Contains(lower, marker) {
			return true
		}
	}
	return false
}

func (i *Inferencer) enqueuePredictionWindow(session *inferenceSession, window predictionWindow) {
	select {
	case session.predictQ <- window:
	default:
		select {
		case <-session.predictQ:
		default:
		}
		session.predictQ <- window
	}
}

func (i *Inferencer) runPredictionWorker(ctx context.Context, session *inferenceSession, modelServerURL string) {
	for {
		select {
		case <-ctx.Done():
			return
		case window, ok := <-session.predictQ:
			if !ok {
				return
			}
			i.processPredictionWindow(ctx, session, modelServerURL, window)
		}
	}
}

func (i *Inferencer) processPredictionWindow(ctx context.Context, session *inferenceSession, modelServerURL string, window predictionWindow) {
	prediction, err := i.requestPrediction(ctx, modelServerURL, window)
	if err != nil {
		i.handleSessionPredictionFailure(session, window, err)
		return
	}
	submitted, err := i.submitActiveParkingPrediction(session, prediction)
	if err != nil {
		i.handleSessionPredictionFailure(session, window, err)
		return
	}
	if !submitted {
		return
	}

	i.mu.Lock()
	if i.active == session && !i.parkingSafetyTripped {
		i.status.LastPrediction = prediction
		i.status.PredictionsSent++
		i.status.LastError = ""
	}
	i.mu.Unlock()
	i.logPlannerDebug(prediction)
}

func (i *Inferencer) requestPrediction(ctx context.Context, modelServerURL string, window predictionWindow) (*InferencePrediction, error) {
	selection, err := i.buildPlannerSelection(window)
	if err != nil {
		return nil, err
	}
	framesBase64 := make([]string, 0, len(window.frames))
	frameHashes := make([]string, 0, len(window.frames))
	for _, frame := range window.frames {
		encoded, err := i.encodeJPEGBase64(frame)
		if err != nil {
			return nil, err
		}
		framesBase64 = append(framesBase64, encoded)
		frameHashes = append(frameHashes, hashInferencePayload(encoded))
	}

	bodyPayload := map[string]any{
		"planner_format":          i.config.PlannerFormat,
		"control_contract":        i.config.ControlContract,
		"frames_base64":           framesBase64,
		"telemetry":               selection.telemetryTensor,
		"sequence":                window.sequenceNumber,
		"sampled_at_s":            predictionInputTimestampS(window, selection),
		"image_offsets":           i.config.ImageOffsets,
		"telemetry_offsets":       i.config.TelemetryOffsets,
		"telemetry_feature_names": i.config.TelemetryFeatureNames,
		"control_output_names":    i.config.ControlOutputNames,
	}
	body, err := json.Marshal(bodyPayload)
	if err != nil {
		return nil, err
	}

	requestCtx, cancel := context.WithTimeout(ctx, i.config.PredictionTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(requestCtx, http.MethodPost, modelServerURL+"/predict", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := i.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		payload, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("python predict failed: status=%d body=%s", resp.StatusCode, strings.TrimSpace(string(payload)))
	}

	var parsed pythonPredictResponse
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return nil, err
	}
	if err := i.validateParkingPredictionModel(parsed); err != nil {
		return nil, err
	}
	prediction, err := i.buildPrediction(
		parsed,
		modelServerURL,
		window,
		selection,
		frameHashes,
	)
	if err != nil {
		return nil, err
	}
	return prediction, nil
}

func (i *Inferencer) submitActuatorPrediction(prediction *InferencePrediction) error {
	if i.actuator == nil {
		return errors.New("inference actuator is not configured")
	}
	if prediction == nil || prediction.SetpointPlan == nil {
		return errors.New("parking inference prediction has no setpoint plan")
	}
	planSink, ok := i.actuator.(actuatorParkingPlanSubmitter)
	if !ok {
		return errors.New("inference actuator does not support parking setpoint plans")
	}
	_, err := planSink.SubmitParkingSetpointPlan(*prediction.SetpointPlan)
	return err
}

func (i *Inferencer) submitActiveParkingPrediction(
	session *inferenceSession,
	prediction *InferencePrediction,
) (bool, error) {
	i.actuationMu.Lock()
	defer i.actuationMu.Unlock()

	if err := i.validateActiveParkingInference(); err != nil {
		return false, err
	}
	i.mu.Lock()
	allowed := i.active == session && !i.parkingSafetyTripped && !i.parkingCompleted && !i.parkingManualStop
	i.mu.Unlock()
	if !allowed {
		return false, nil
	}
	if err := i.submitActuatorPrediction(prediction); err != nil {
		return true, fmt.Errorf("failed to submit actuator prediction: %w", err)
	}
	return true, nil
}

func (i *Inferencer) latestTelemetryForDebug() *control.RuntimeTelemetry {
	if i.telemetry == nil {
		return nil
	}
	telemetry, _ := i.telemetry.LatestTelemetrySnapshot()
	return telemetry
}

type plannerSelection struct {
	selectedTelemetry     []control.RuntimeTelemetry
	telemetryTensor       [][][]float64
	telemetryShape        []int
	frameShape            []int
	telemetryTimesMs      []int64
	latestFrameTimestampS float64
	telemetryTimestampS   float64
	frameID               *int64
	captureLatencyMs      *float64
	frameTelemetrySkewMs  float64
	frameTelemetryAligned bool
}

func (i *Inferencer) buildPlannerSelection(window predictionWindow) (plannerSelection, error) {
	if i.normalizationErr != nil {
		return plannerSelection{}, i.normalizationErr
	}
	if err := i.validateActiveParkingInference(); err != nil {
		return plannerSelection{}, err
	}
	if i.telemetry == nil {
		return plannerSelection{}, errors.New("planner telemetry store is not configured")
	}
	history := i.telemetry.TelemetryHistorySnapshot(512)
	if len(history) == 0 {
		return plannerSelection{}, errors.New("planner telemetry is unavailable")
	}
	latest := history[len(history)-1]
	latestAge := time.Duration(i.nowFunc().UTC().UnixMilli()-telemetrySourceTimestampMs(latest)) * time.Millisecond
	if latestAge < -parkingSourceClockLeadLimit {
		return plannerSelection{}, fmt.Errorf("planner telemetry source timestamp is %s in the future", -latestAge)
	}
	if latestAge > i.telemetryStaleAfter {
		return plannerSelection{}, fmt.Errorf("planner telemetry is stale: age=%s", latestAge)
	}
	anchorMs := window.capturedAt.UnixMilli()
	anchorIndex, err := findAnchorTelemetryIndex(history, anchorMs, i.config.AlignmentTolerance)
	if err != nil {
		return plannerSelection{}, err
	}
	selected, times, err := selectTelemetryAtOffsets(
		history,
		anchorIndex,
		i.config.TelemetryOffsets,
		i.config.TelemetrySampleInterval,
		i.config.AlignmentTolerance,
	)
	if err != nil {
		return plannerSelection{}, err
	}
	features := make([][]float64, 0, len(selected))
	for index := range selected {
		vector, err := i.telemetryFeatureVector(selected, index)
		if err != nil {
			return plannerSelection{}, err
		}
		features = append(features, vector)
	}
	latestFrameTime := window.capturedAt
	for _, frameTime := range window.frameTimes {
		if frameTime.After(latestFrameTime) {
			latestFrameTime = frameTime
		}
	}
	latestFrameMs := latestFrameTime.UnixMilli()
	telemetryMs := int64(0)
	if len(times) > 0 {
		telemetryMs = times[len(times)-1]
	}
	frameSkewMs := 0.0
	aligned := telemetryMs > 0
	if telemetryMs > 0 {
		frameSkewMs = math.Abs(float64(latestFrameMs - telemetryMs))
		aligned = frameSkewMs <= durationMilliseconds(i.config.MaxFrameTelemetrySkew)
		if !aligned {
			return plannerSelection{}, parkingInferenceError(
				"frame/telemetry skew exceeded the parking safety limit: sequence=%d frame_id=%d frame_ts_ms=%d telemetry_ts_ms=%d skew_ms=%.1f threshold_ms=%.1f",
				window.sequenceNumber,
				window.frameIndex,
				latestFrameMs,
				telemetryMs,
				frameSkewMs,
				durationMilliseconds(i.config.MaxFrameTelemetrySkew),
			)
		}
	}
	captureLatencyMs := math.Max(0, float64(i.nowFunc().UTC().Sub(latestFrameTime))/float64(time.Millisecond))
	frameID := int64(window.frameIndex)
	return plannerSelection{
		selectedTelemetry:     selected,
		telemetryTensor:       [][][]float64{features},
		telemetryShape:        []int{1, len(features), len(i.config.TelemetryFeatureNames)},
		frameShape:            []int{1, len(window.frames), 3, i.config.FrameHeight, i.config.FrameWidth},
		telemetryTimesMs:      times,
		latestFrameTimestampS: timeToSeconds(latestFrameTime),
		telemetryTimestampS:   float64(telemetryMs) / 1000.0,
		frameID:               &frameID,
		captureLatencyMs:      &captureLatencyMs,
		frameTelemetrySkewMs:  frameSkewMs,
		frameTelemetryAligned: aligned,
	}, nil
}

func cloneStringMap(input map[string]string) map[string]string {
	if len(input) == 0 {
		return nil
	}
	out := make(map[string]string, len(input))
	for key, value := range input {
		out[key] = value
	}
	return out
}

func newDebugFrameDump(now time.Time) (*debugFrameDump, error) {
	root := filepath.Join(os.TempDir(), "awesomeProject-inference-frames")
	if err := os.MkdirAll(root, 0o755); err != nil {
		return nil, err
	}
	dir := filepath.Join(root, now.UTC().Format("20060102-150405.000"))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	return &debugFrameDump{
		dir:   dir,
		limit: defaultDebugFrameDumpLimit,
	}, nil
}

func maybeDumpDebugFrame(dump *debugFrameDump, frame bufferedInferenceFrame) (int, error) {
	if dump == nil || dump.saved >= dump.limit {
		if dump == nil {
			return 0, nil
		}
		return dump.saved, nil
	}

	filename := filepath.Join(dump.dir, fmt.Sprintf("frame-%04d.jpg", dump.saved))
	file, err := os.Create(filename)
	if err != nil {
		return dump.saved, err
	}
	defer file.Close()
	if err := jpeg.Encode(file, frame.image, &jpeg.Options{Quality: 95}); err != nil {
		return dump.saved, err
	}
	dump.saved++
	return dump.saved, nil
}

func (i *Inferencer) loadRemoteModel(ctx context.Context, modelServerURL string) error {
	payload := map[string]any{}
	if i.config.ConfigPath != "" {
		payload["config"] = i.config.ConfigPath
	}

	i.mu.Lock()
	loadedCheckpoint := strings.TrimSpace(i.loadedCheckpoint)
	loadedModelDevice := strings.TrimSpace(i.loadedModelDevice)
	i.mu.Unlock()

	if loadedCheckpoint != "" {
		payload["checkpoint"] = loadedCheckpoint
	}

	device := loadedModelDevice
	if device == "" {
		device = strings.TrimSpace(i.config.ModelDevice)
	}
	if device != "" {
		payload["device"] = device
	}
	parsed, err := i.postRemoteModelLoad(ctx, modelServerURL, payload)
	if err != nil {
		return err
	}

	i.mu.Lock()
	i.modelServerURL = modelServerURL
	if checkpoint, ok := parsed["checkpoint"].(string); ok && strings.TrimSpace(checkpoint) != "" {
		i.loadedCheckpoint = strings.TrimSpace(checkpoint)
	}
	if resolvedDevice, ok := parsed["device"].(string); ok && strings.TrimSpace(resolvedDevice) != "" {
		i.loadedModelDevice = strings.TrimSpace(resolvedDevice)
	}
	i.mu.Unlock()
	return nil
}

func parseBoolAny(value any) bool {
	switch typed := value.(type) {
	case bool:
		return typed
	case float64:
		return typed != 0
	case string:
		trimmed := strings.TrimSpace(typed)
		return strings.EqualFold(trimmed, "true") || trimmed == "1"
	default:
		return false
	}
}

func (i *Inferencer) postRemoteModelLoad(ctx context.Context, modelServerURL string, payload map[string]any) (map[string]any, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	requestCtx, cancel := context.WithTimeout(ctx, i.requestTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(requestCtx, http.MethodPost, modelServerURL+"/model/load", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := i.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		payload, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("python model load failed: status=%d body=%s", resp.StatusCode, strings.TrimSpace(string(payload)))
	}
	var parsed map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return nil, err
	}
	return parsed, nil
}

func (i *Inferencer) setInferenceError(err error) {
	i.mu.Lock()
	defer i.mu.Unlock()
	i.status.State = "error"
	i.status.LastError = err.Error()
	i.status.StoppedAt = i.nowFunc().UTC().Format(time.RFC3339Nano)
	log.Printf("[inference] fatal error: %v", err)
}

func (i *Inferencer) recordPredictionError(err error) {
	i.mu.Lock()
	defer i.mu.Unlock()
	i.status.PredictionErrors++
	i.status.LastError = err.Error()
	if i.status.State == "running" && !isTelemetryWaitError(err) {
		i.status.State = "error"
	}
	if isTelemetryWaitError(err) {
		now := i.nowFunc()
		if i.lastTelemetryWaitLog.IsZero() || now.Sub(i.lastTelemetryWaitLog) >= time.Second {
			i.lastTelemetryWaitLog = now
			log.Printf("[inference] waiting for telemetry: %v", err)
		}
		return
	}
	log.Printf("[inference] prediction error: %v", err)
}

func isTelemetryWaitError(err error) bool {
	if err == nil {
		return false
	}
	message := err.Error()
	return strings.Contains(message, "planner telemetry is unavailable") ||
		strings.Contains(message, "planner telemetry is stale") ||
		strings.Contains(message, "planner telemetry window is incomplete") ||
		strings.Contains(message, "planner telemetry alignment failed")
}

func buildInferenceFFmpegArgs(spec captureSpec, cfg InferenceConfig) []string {
	videoFilter := buildInferenceVideoFilter(spec, cfg)
	return []string{
		"-hide_banner",
		"-loglevel", "info",
		"-nostats",
		"-fflags", "+nobuffer",
		"-f", spec.inputFormat,
		"-i", spec.input,
		"-vf", videoFilter,
		"-pix_fmt", "rgb24",
		"-f", "rawvideo",
		"pipe:1",
	}
}

func buildInferenceVideoFilter(spec captureSpec, cfg InferenceConfig) string {
	timing := fmt.Sprintf("fps=%d,showinfo=checksum=0", cfg.FPS)
	resize := fmt.Sprintf("scale=%d:%d:flags=lanczos,format=rgb24", cfg.FrameWidth, cfg.FrameHeight)
	if spec.backend == "ddagrab" || spec.inputFormat == "lavfi" {
		return timing + ",hwdownload,format=bgra," + resize
	}
	return timing + "," + resize
}

func shouldDispatchInferenceFrame(frameIndex int, imageOffsets []int, dispatchStride int) bool {
	if len(imageOffsets) < 1 || dispatchStride < 1 {
		return false
	}
	return frameIndex >= requiredFrameCount(imageOffsets) && frameIndex%dispatchStride == 0
}

func requiredFrameCount(imageOffsets []int) int {
	if len(imageOffsets) < 1 {
		return 0
	}
	minOffset := imageOffsets[0]
	for _, offset := range imageOffsets[1:] {
		if offset < minOffset {
			minOffset = offset
		}
	}
	if minOffset > 0 {
		minOffset = 0
	}
	return -minOffset
}

func buildPredictionWindow(buffer []bufferedInferenceFrame, sequence int, imageOffsets []int) *predictionWindow {
	needed := requiredFrameCount(imageOffsets)
	if len(buffer) < needed {
		return nil
	}
	byIndex := make(map[int]bufferedInferenceFrame, len(buffer))
	for _, frame := range buffer {
		byIndex[frame.index] = frame
	}
	last := buffer[len(buffer)-1]
	windowFrames := make([]*image.RGBA, 0, len(imageOffsets))
	windowIndices := make([]int, 0, len(imageOffsets))
	windowTimes := make([]time.Time, 0, len(imageOffsets))
	for _, offset := range imageOffsets {
		frame, ok := byIndex[last.index+offset]
		if !ok {
			return nil
		}
		windowFrames = append(windowFrames, frame.image)
		windowIndices = append(windowIndices, frame.index)
		windowTimes = append(windowTimes, frame.capturedAt)
	}
	return &predictionWindow{
		frames:         windowFrames,
		frameIndex:     last.index,
		frameIndices:   windowIndices,
		frameTimes:     windowTimes,
		capturedAt:     last.capturedAt,
		sequenceNumber: sequence,
	}
}

func (i *Inferencer) encodeJPEGBase64(img image.Image) (string, error) {
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: i.config.JPEGQuality}); err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(buf.Bytes()), nil
}

func rgbFrameToRGBA(buf []byte, width int, height int) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	src := 0
	for y := 0; y < height; y++ {
		dst := y * img.Stride
		for x := 0; x < width; x++ {
			img.Pix[dst] = buf[src]
			img.Pix[dst+1] = buf[src+1]
			img.Pix[dst+2] = buf[src+2]
			img.Pix[dst+3] = 0xff
			src += 3
			dst += 4
		}
	}
	return img
}

func inferenceRawFrameBytes(width int, height int) int {
	return width * height * 3
}

func cloneInferenceStatus(status InferenceStatus) InferenceStatus {
	out := status
	if status.LastPrediction != nil {
		copyPrediction := *status.LastPrediction
		copyPrediction.WindowFrameIndices = append([]int(nil), status.LastPrediction.WindowFrameIndices...)
		copyPrediction.WindowFrameHashes = append([]string(nil), status.LastPrediction.WindowFrameHashes...)
		copyPrediction.WindowFrameTimestampsMs = append([]int64(nil), status.LastPrediction.WindowFrameTimestampsMs...)
		copyPrediction.FrameID = cloneInt64Ptr(status.LastPrediction.FrameID)
		copyPrediction.CaptureLatencyMs = cloneFloatPtr(status.LastPrediction.CaptureLatencyMs)
		copyPrediction.SelectedTelemetryOffsets = append([]int(nil), status.LastPrediction.SelectedTelemetryOffsets...)
		copyPrediction.SelectedTelemetryTimestampsMs = append([]int64(nil), status.LastPrediction.SelectedTelemetryTimestampsMs...)
		copyPrediction.ImageTensorShape = append([]int(nil), status.LastPrediction.ImageTensorShape...)
		copyPrediction.TelemetryTensorShape = append([]int(nil), status.LastPrediction.TelemetryTensorShape...)
		copyPrediction.PredControlsShape = append([]int(nil), status.LastPrediction.PredControlsShape...)
		copyPrediction.PredAuxShape = append([]int(nil), status.LastPrediction.PredAuxShape...)
		copyPrediction.ControlContract = cloneParkingControlContract(status.LastPrediction.ControlContract)
		copyPrediction.ControlHorizonDtMs = append([]int(nil), status.LastPrediction.ControlHorizonDtMs...)
		copyPrediction.RawPredControls = clone2DFloat64(status.LastPrediction.RawPredControls)
		copyPrediction.RawPredAux = clone2DFloat64(status.LastPrediction.RawPredAux)
		copyPrediction.RawStateInputs = cloneAnyMap(status.LastPrediction.RawStateInputs)
		copyPrediction.NormalizedStateInputs = cloneFloat64Map(status.LastPrediction.NormalizedStateInputs)
		if status.LastPrediction.SetpointPlan != nil {
			planCopy := *status.LastPrediction.SetpointPlan
			planCopy.Points = append([]parkingcontrol.Setpoint(nil), status.LastPrediction.SetpointPlan.Points...)
			copyPrediction.SetpointPlan = &planCopy
		}
		if status.LastPrediction.LastTelemetry != nil {
			telemetryCopy := *status.LastPrediction.LastTelemetry
			copyPrediction.LastTelemetry = &telemetryCopy
		}
		out.LastPrediction = &copyPrediction
	}
	return out
}

func hashInferencePayload(encoded string) string {
	hasher := fnv.New64a()
	_, _ = hasher.Write([]byte(encoded))
	return fmt.Sprintf("%016x", hasher.Sum64())
}

func parseBoolEnv(key string, fallback bool) bool {
	raw := strings.TrimSpace(envOrDefault(key, ""))
	if raw == "" {
		return fallback
	}
	switch strings.ToLower(raw) {
	case "1", "true", "yes", "on":
		return true
	case "0", "false", "no", "off":
		return false
	default:
		return fallback
	}
}

func (i *Inferencer) buildPrediction(
	parsed pythonPredictResponse,
	modelServerURL string,
	window predictionWindow,
	selection plannerSelection,
	frameHashes []string,
) (*InferencePrediction, error) {
	controls, err := validatePlannerTensor(
		parsed.PredControls,
		1,
		i.config.FutureSteps,
		len(i.config.ControlOutputNames),
		"pred_controls",
	)
	if err != nil {
		return nil, err
	}
	aux, auxShape, err := validateOptionalPlannerTensor(
		parsed.PredAux,
		1,
		i.config.FutureSteps,
		len(i.config.AuxOutputNames),
		"pred_aux",
	)
	if err != nil {
		return nil, err
	}
	predictedAt := i.nowFunc().UTC()
	sampledAtS := predictionInputTimestampS(window, selection)
	if !finiteParkingValue(parsed.SampledAtS) || math.Abs(parsed.SampledAtS-sampledAtS) > 1e-6 {
		return nil, fmt.Errorf(
			"prediction sampled_at_s did not echo the requested observation timestamp: got=%.9f want=%.9f",
			parsed.SampledAtS,
			sampledAtS,
		)
	}
	points := make([]parkingcontrol.Setpoint, 0, len(controls[0]))
	for index, row := range controls[0] {
		points = append(points, parkingcontrol.Setpoint{
			DtMs:                        i.config.ControlHorizonDtMs[index],
			DesiredWheelSteerNormalized: 0,
			DesiredSpeedMPS:             row[0],
			StopProbability:             row[1],
		})
	}
	plan := parkingcontrol.Plan{
		Contract:    parsed.ControlContract.Name,
		SampledAtS:  parsed.SampledAtS,
		ReceivedAtS: timeToSeconds(predictedAt),
		Points:      points,
		Direction:   parsed.Direction,
	}
	if err := parkingcontrol.ValidatePlan(plan); err != nil {
		return nil, err
	}
	frameTimesMs := make([]int64, 0, len(window.frameTimes))
	for _, item := range window.frameTimes {
		frameTimesMs = append(frameTimesMs, item.UnixMilli())
	}
	prediction := &InferencePrediction{
		Sequence:                      window.sequenceNumber,
		FrameIndex:                    window.frameIndex,
		SourceFPS:                     i.config.FPS,
		InferenceHz:                   i.config.FPS / i.config.DispatchStride,
		ModelServerURL:                modelServerURL,
		Checkpoint:                    parsed.Checkpoint,
		ModelDevice:                   parsed.Device,
		PlannerFormat:                 parsed.PlannerFormat,
		PlannerFormatVersion:          parsed.PlannerFormatVersion,
		ControlContract:               cloneParkingControlContract(parsed.ControlContract),
		ControlHorizonDtMs:            append([]int(nil), parsed.ControlHorizonDtMs...),
		TelemetrySampleIntervalMs:     parsed.TelemetrySampleIntervalMs,
		CapturedAt:                    window.capturedAt.Format(time.RFC3339Nano),
		PredictedAt:                   predictedAt.Format(time.RFC3339Nano),
		WindowFrameIndices:            append([]int(nil), window.frameIndices...),
		WindowFrameHashes:             append([]string(nil), frameHashes...),
		WindowFrameTimestampsMs:       frameTimesMs,
		LatestFrameTimestampS:         selection.latestFrameTimestampS,
		TelemetryTimestampS:           selection.telemetryTimestampS,
		FrameID:                       cloneInt64Ptr(selection.frameID),
		CaptureLatencyMs:              cloneFloatPtr(selection.captureLatencyMs),
		FrameTelemetrySkewMs:          selection.frameTelemetrySkewMs,
		FrameTelemetryAligned:         selection.frameTelemetryAligned,
		SelectedTelemetryOffsets:      append([]int(nil), i.config.TelemetryOffsets...),
		SelectedTelemetryTimestampsMs: append([]int64(nil), selection.telemetryTimesMs...),
		ImageTensorShape:              append([]int(nil), selection.frameShape...),
		TelemetryTensorShape:          append([]int(nil), selection.telemetryShape...),
		PredControlsShape:             []int{1, len(controls[0]), len(i.config.ControlOutputNames)},
		PredAuxShape:                  auxShape,
		LastTelemetry:                 cloneTelemetryPtr(i.latestTelemetryForDebug()),
		RawPredControls:               clone2DFloat64(controls[0]),
		RawPredAux:                    clone2DFloat64(aux),
		RawStateInputs:                cloneAnyMap(parsed.RawStateInputs),
		NormalizedStateInputs:         cloneFloat64Map(parsed.NormalizedStateInputs),
		SetpointPlan:                  &plan,
	}
	return prediction, nil
}

func (i *Inferencer) handlePredictionFailure(window predictionWindow, cause error) {
	i.terminateParkingInference(nil, window.sequenceNumber, cause)
}

func (i *Inferencer) handleSessionPredictionFailure(session *inferenceSession, window predictionWindow, cause error) {
	i.terminateParkingInference(session, window.sequenceNumber, cause)
}

func (i *Inferencer) handleInferenceProcessExit(session *inferenceSession, processErr error) {
	cause := errors.New("inference capture process exited unexpectedly")
	if processErr != nil {
		cause = fmt.Errorf("inference capture process exited unexpectedly: %w", processErr)
	}
	i.terminateParkingInference(session, 0, cause)
}

func (i *Inferencer) terminateParkingInference(expectedSession *inferenceSession, sequence int, cause error) {
	i.actuationMu.Lock()
	session, transitioned, succeeded := i.beginParkingTerminalTransition(expectedSession, cause)
	if !transitioned {
		i.actuationMu.Unlock()
		return
	}

	holdErr := i.submitParkingSafetyHoldLocked(sequence)
	if holdErr != nil {
		i.recordParkingSafetyHoldError(holdErr)
	} else if succeeded {
		i.recordParkingSafetyHoldConfirmed()
	}
	i.actuationMu.Unlock()
	if session != nil && session.cancel != nil {
		session.cancel()
	}
}

func (i *Inferencer) beginParkingTerminalTransition(expectedSession *inferenceSession, cause error) (*inferenceSession, bool, bool) {
	i.mu.Lock()
	defer i.mu.Unlock()
	if expectedSession != nil && i.active != expectedSession {
		return nil, false, false
	}
	if i.parkingCompleted || i.parkingSafetyTripped {
		return nil, false, false
	}

	succeeded := errors.Is(cause, ErrParkingInferenceComplete)
	i.parkingCompleted = succeeded
	i.parkingSafetyTripped = true
	i.parkingManualStop = false
	i.parkingHoldConfirmed = false
	if succeeded {
		i.status.State = "stopping"
		i.status.LastError = ""
	} else {
		if cause == nil {
			cause = errors.New("parking inference failed without a reason")
		}
		i.status.State = "error"
		i.status.LastError = cause.Error()
		i.status.PredictionErrors++
		log.Printf("[inference] parking safety stop: %v", cause)
	}
	if i.active == nil {
		i.status.StoppedAt = i.nowFunc().UTC().Format(time.RFC3339Nano)
	} else {
		i.status.StoppedAt = ""
	}
	return i.active, true, succeeded
}

func (i *Inferencer) submitParkingSafetyHold(sequence int) error {
	i.actuationMu.Lock()
	defer i.actuationMu.Unlock()
	return i.submitParkingSafetyHoldLocked(sequence)
}

func (i *Inferencer) submitParkingSafetyHoldLocked(sequence int) error {
	stopper, ok := i.actuator.(actuatorParkingSafetyStopper)
	if !ok {
		return errors.New("parking safety stop cannot be requested because the actuator does not support it")
	}
	submittedState, err := stopper.RequestParkingSafetyStop()
	if err != nil {
		return err
	}
	return i.confirmAppliedParkingStop(submittedState.LastCommandID, actuatorApplyExpectation{
		label:    "parking safety stop",
		stopping: true,
	})
}

func (i *Inferencer) armActuatorForParkingInference() error {
	i.actuationMu.Lock()
	defer i.actuationMu.Unlock()
	enabled := true
	return i.submitAndConfirmActuatorCommand(actuator.CommandRequest{
		Steer:            0,
		Throttle:         0,
		BrakePressureAvg: 0,
		InputMode:        actuator.InputModeNormalized,
		Handbrake:        false,
		Enabled:          &enabled,
		Owner:            actuator.OwnerParkingInference,
		TimestampMs:      i.nowFunc().UTC().UnixMilli(),
	}, actuatorApplyExpectation{label: "parking inference arm"})
}

func (i *Inferencer) submitAndConfirmActuatorCommand(command actuator.CommandRequest, expected actuatorApplyExpectation) error {
	submittedState, err := i.actuator.Submit(command)
	if err != nil {
		return err
	}
	return i.confirmAppliedParkingStop(submittedState.LastCommandID, expected)
}

func (i *Inferencer) confirmAppliedParkingStop(commandID int64, expected actuatorApplyExpectation) error {
	provider, ok := i.actuator.(actuatorStateProvider)
	if !ok {
		return fmt.Errorf("%s cannot be confirmed because actuator state is unavailable", expected.label)
	}
	if commandID <= 0 {
		return fmt.Errorf("%s did not receive a trackable actuator command ID", expected.label)
	}

	timeout := i.actuatorConfirmTimeout
	if timeout <= 0 {
		timeout = defaultActuatorConfirmTimeout
	}
	interval := i.actuatorConfirmInterval
	if interval <= 0 {
		interval = defaultActuatorConfirmInterval
	}
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		state := provider.State()
		if detail := strings.TrimSpace(state.LastApplyError); detail != "" && state.LastApplyAttemptedCommandID >= commandID {
			return fmt.Errorf("%s controller apply failed: %s", expected.label, detail)
		}
		if actuatorStateConfirmsCommand(state, expected, commandID) {
			return nil
		}

		select {
		case <-timer.C:
			return fmt.Errorf("timed out after %s waiting for applied %s", timeout, expected.label)
		case <-ticker.C:
		}
	}
}

func actuatorStateConfirmsCommand(state actuator.State, expected actuatorApplyExpectation, commandID int64) bool {
	if !state.Supported || !state.Ready {
		return false
	}
	if _, err := time.Parse(time.RFC3339Nano, state.LastApplySucceededAt); err != nil {
		return false
	}
	if state.ParkingController.Owner != actuator.OwnerParkingInference || state.ParkingController.Stopping != expected.stopping {
		return false
	}
	if state.Applied.CommandID != commandID || !state.Applied.Enabled || state.Applied.Steer != 0 || state.Applied.Throttle != 0 {
		return false
	}
	if state.Applied.Handbrake {
		return state.Applied.Brake == 0
	}
	return state.Applied.Brake > 0
}

func (i *Inferencer) recordParkingSafetyHoldError(err error) {
	message := fmt.Sprintf("failed to apply parking safety hold: %v", err)
	i.mu.Lock()
	i.parkingHoldFailed = true
	i.parkingHoldConfirmed = false
	i.status.State = "error"
	if i.status.LastError == "" {
		i.status.LastError = message
	} else {
		i.status.LastError += "; " + message
	}
	i.mu.Unlock()
	log.Printf("[inference] %s", message)
}

func (i *Inferencer) recordParkingSafetyHoldConfirmed() {
	i.mu.Lock()
	defer i.mu.Unlock()
	if i.parkingCompleted && !i.parkingHoldFailed {
		i.parkingHoldConfirmed = true
		i.status.State = "succeeded"
		i.status.LastError = ""
	}
}

func (i *Inferencer) monitorParkingEvaluationDeadline(ctx context.Context, session *inferenceSession) {
	timer := time.NewTimer(i.parkingEvaluationLimit)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return
	case <-timer.C:
		i.handleSessionPredictionFailure(
			session,
			predictionWindow{},
			fmt.Errorf("%w: no settled success within %s", ErrParkingInferenceDeadlineExceeded, i.parkingEvaluationLimit),
		)
	}
}

func (i *Inferencer) telemetryFeatureVector(window []control.RuntimeTelemetry, index int) ([]float64, error) {
	sample := window[index]
	yawRate := sample.YawRate
	if math.IsNaN(yawRate) || math.IsInf(yawRate, 0) {
		yawRate = deriveYawRate(window, index)
	}
	yawRadians := sample.CurrentYaw * math.Pi / 180.0
	values := map[string]float64{
		"current_speed": sample.CurrentSpeed,
		"yaw_sin":       math.Sin(yawRadians),
		"yaw_cos":       math.Cos(yawRadians),
		"yaw_rate":      yawRate,
		"steering":      sample.Steering,
		"acceleration":  sample.Acceleration,
	}
	vector := make([]float64, 0, len(i.config.TelemetryFeatureNames))
	for _, name := range i.config.TelemetryFeatureNames {
		value, ok := values[name]
		if !ok {
			return nil, fmt.Errorf("unsupported telemetry feature %q", name)
		}
		if i.telemetryNormalizer != nil {
			switch name {
			case "current_speed", "yaw_rate":
				value = i.telemetryNormalizer.Normalize(name, value)
			}
		}
		vector = append(vector, value)
	}
	return vector, nil
}

func deriveYawRate(window []control.RuntimeTelemetry, index int) float64 {
	if index <= 0 || index >= len(window) {
		return 0
	}
	current := window[index]
	previous := window[index-1]
	deltaMs := telemetrySourceTimestampMs(current) - telemetrySourceTimestampMs(previous)
	if deltaMs <= 0 {
		return 0
	}
	deltaDeg := wrapHeadingDeltaDegrees(current.CurrentYaw - previous.CurrentYaw)
	return deltaDeg / (float64(deltaMs) / 1000.0)
}

func wrapHeadingDeltaDegrees(delta float64) float64 {
	for delta > 180 {
		delta -= 360
	}
	for delta < -180 {
		delta += 360
	}
	return delta
}

func findAnchorTelemetryIndex(history []control.RuntimeTelemetry, anchorMs int64, tolerance time.Duration) (int, error) {
	bestIndex := -1
	bestDelta := tolerance.Milliseconds() + 1
	closestDelta := int64(-1)
	closestTimestamp := int64(0)
	for index := len(history) - 1; index >= 0; index-- {
		timestamp := telemetrySourceTimestampMs(history[index])
		if timestamp == 0 || timestamp > anchorMs {
			continue
		}
		delta := anchorMs - timestamp
		if closestDelta < 0 || delta < closestDelta {
			closestDelta = delta
			closestTimestamp = timestamp
		}
		if delta <= tolerance.Milliseconds() && delta < bestDelta {
			bestDelta = delta
			bestIndex = index
		}
		if delta > tolerance.Milliseconds() && bestIndex >= 0 {
			break
		}
	}
	if bestIndex < 0 {
		if closestDelta >= 0 {
			return -1, fmt.Errorf("planner telemetry alignment failed at anchor=%d nearest=%d delta=%dms tolerance=%s", anchorMs, closestTimestamp, closestDelta, tolerance)
		}
		return -1, fmt.Errorf("planner telemetry alignment failed at anchor=%d tolerance=%s", anchorMs, tolerance)
	}
	return bestIndex, nil
}

func selectTelemetryAtOffsets(
	history []control.RuntimeTelemetry,
	anchorIndex int,
	offsets []int,
	sampleInterval time.Duration,
	configuredTolerance time.Duration,
) ([]control.RuntimeTelemetry, []int64, error) {
	if anchorIndex < 0 || anchorIndex >= len(history) {
		return nil, nil, errors.New("planner telemetry anchor is outside the available history")
	}
	if len(offsets) == 0 || offsets[len(offsets)-1] != 0 {
		return nil, nil, errors.New("planner telemetry offsets must be strictly increasing and end at 0")
	}
	if sampleInterval <= 0 || sampleInterval%time.Millisecond != 0 {
		return nil, nil, errors.New("planner telemetry sample interval must be a positive whole number of milliseconds")
	}
	if configuredTolerance <= 0 {
		return nil, nil, errors.New("planner telemetry alignment tolerance must be positive")
	}
	for index := 1; index < len(offsets); index++ {
		if offsets[index] <= offsets[index-1] {
			return nil, nil, errors.New("planner telemetry offsets must be strictly increasing and end at 0")
		}
	}

	anchorTimestampMs := telemetrySourceTimestampMs(history[anchorIndex])
	if anchorTimestampMs <= 0 {
		return nil, nil, errors.New("planner telemetry anchor has no source timestamp")
	}
	tolerance := configuredTolerance
	if maximum := sampleInterval / 2; tolerance > maximum {
		tolerance = maximum
	}
	intervalMs := sampleInterval.Milliseconds()
	selected := make([]control.RuntimeTelemetry, len(offsets))
	timestamps := make([]int64, len(offsets))
	lastIndex := len(offsets) - 1
	selected[lastIndex] = history[anchorIndex]
	timestamps[lastIndex] = anchorTimestampMs

	nextIndex := anchorIndex
	nextTimestampMs := anchorTimestampMs
	for offsetIndex := lastIndex - 1; offsetIndex >= 0; offsetIndex-- {
		offset := offsets[offsetIndex]
		expectedTimestampMs := anchorTimestampMs + int64(offset)*intervalMs
		candidateIndex, candidateTimestampMs, deltaMs := closestEarlierTelemetrySample(
			history,
			nextIndex,
			nextTimestampMs,
			expectedTimestampMs,
		)
		if candidateIndex < 0 {
			return nil, nil, fmt.Errorf(
				"planner telemetry window is incomplete for offset %d at source timestamp %d",
				offset,
				expectedTimestampMs,
			)
		}
		if time.Duration(deltaMs)*time.Millisecond > tolerance {
			return nil, nil, fmt.Errorf(
				"planner telemetry timestamp alignment failed for offset %d: expected=%d nearest=%d delta=%dms tolerance=%s",
				offset,
				expectedTimestampMs,
				candidateTimestampMs,
				deltaMs,
				tolerance,
			)
		}
		selected[offsetIndex] = history[candidateIndex]
		timestamps[offsetIndex] = candidateTimestampMs
		nextIndex = candidateIndex
		nextTimestampMs = candidateTimestampMs
	}
	return selected, timestamps, nil
}

func closestEarlierTelemetrySample(
	history []control.RuntimeTelemetry,
	upperIndex int,
	upperTimestampMs int64,
	targetTimestampMs int64,
) (int, int64, int64) {
	bestIndex := -1
	bestTimestampMs := int64(0)
	bestDeltaMs := int64(0)
	for index := upperIndex - 1; index >= 0; index-- {
		timestampMs := telemetrySourceTimestampMs(history[index])
		if timestampMs <= 0 || timestampMs >= upperTimestampMs {
			continue
		}
		deltaMs := timestampMs - targetTimestampMs
		if deltaMs < 0 {
			deltaMs = -deltaMs
		}
		if bestIndex < 0 || deltaMs < bestDeltaMs {
			bestIndex = index
			bestTimestampMs = timestampMs
			bestDeltaMs = deltaMs
		}
	}
	return bestIndex, bestTimestampMs, bestDeltaMs
}

func telemetrySourceTimestampMs(sample control.RuntimeTelemetry) int64 {
	return sample.TimestampMs
}

func validatePlannerTensor(raw [][][]float64, batch, horizon, width int, name string) ([][][]float64, error) {
	if len(raw) != batch {
		return nil, fmt.Errorf("%s batch mismatch: got=%d want=%d", name, len(raw), batch)
	}
	for batchIndex, item := range raw {
		if len(item) != horizon {
			return nil, fmt.Errorf("%s horizon mismatch at batch %d: got=%d want=%d", name, batchIndex, len(item), horizon)
		}
		for horizonIndex, vector := range item {
			if len(vector) != width {
				return nil, fmt.Errorf("%s width mismatch at batch %d horizon %d: got=%d want=%d", name, batchIndex, horizonIndex, len(vector), width)
			}
		}
	}
	return raw, nil
}

func validateOptionalPlannerTensor(raw [][][]float64, batch, horizon, width int, name string) ([][]float64, []int, error) {
	if len(raw) == 0 {
		return nil, nil, nil
	}
	validated, err := validatePlannerTensor(raw, batch, horizon, width, name)
	if err != nil {
		return nil, nil, err
	}
	return validated[0], []int{batch, horizon, width}, nil
}

func clone2DFloat64(source [][]float64) [][]float64 {
	if len(source) == 0 {
		return nil
	}
	out := make([][]float64, 0, len(source))
	for _, row := range source {
		out = append(out, append([]float64(nil), row...))
	}
	return out
}

func cloneAnyMap(source map[string]any) map[string]any {
	if len(source) == 0 {
		return nil
	}
	out := make(map[string]any, len(source))
	for key, value := range source {
		out[key] = value
	}
	return out
}

func cloneFloat64Map(source map[string]float64) map[string]float64 {
	if len(source) == 0 {
		return nil
	}
	out := make(map[string]float64, len(source))
	for key, value := range source {
		out[key] = value
	}
	return out
}

func cloneTelemetryPtr(source *control.RuntimeTelemetry) *control.RuntimeTelemetry {
	if source == nil {
		return nil
	}
	copyValue := *source
	return &copyValue
}

func cloneFloatPtr(value *float64) *float64 {
	if value == nil {
		return nil
	}
	copyValue := *value
	return &copyValue
}

func cloneInt64Ptr(value *int64) *int64 {
	if value == nil {
		return nil
	}
	copyValue := *value
	return &copyValue
}

func predictionInputTimestampS(window predictionWindow, selection plannerSelection) float64 {
	latest := window.capturedAt
	for _, frameTime := range window.frameTimes {
		if frameTime.After(latest) {
			latest = frameTime
		}
	}
	for _, telemetryMs := range selection.telemetryTimesMs {
		if telemetryMs <= 0 {
			continue
		}
		telemetryTime := time.UnixMilli(telemetryMs).UTC()
		if telemetryTime.After(latest) {
			latest = telemetryTime
		}
	}
	return timeToSeconds(latest)
}

func timeToSeconds(value time.Time) float64 {
	return float64(value.UTC().UnixNano()) / float64(time.Second)
}

func durationMilliseconds(value time.Duration) float64 {
	return float64(value) / float64(time.Millisecond)
}

func (i *Inferencer) logPlannerDebug(prediction *InferencePrediction) {
	if prediction == nil || prediction.Sequence > 5 {
		return
	}
	first := parkingcontrol.Setpoint{}
	if prediction.SetpointPlan != nil && len(prediction.SetpointPlan.Points) > 0 {
		first = prediction.SetpointPlan.Points[0]
	}
	log.Printf("[inference] planner window seq=%d frameTs=%v telemetryTs=%v imageShape=%v telemetryShape=%v predShape=%v t+%dms steer=%.3f speed=%.3f stop=%.3f",
		prediction.Sequence,
		prediction.WindowFrameTimestampsMs,
		prediction.SelectedTelemetryTimestampsMs,
		prediction.ImageTensorShape,
		prediction.TelemetryTensorShape,
		prediction.PredControlsShape,
		first.DtMs,
		first.DesiredWheelSteerNormalized,
		first.DesiredSpeedMPS,
		first.StopProbability,
	)
}

func boolPtr(value bool) *bool {
	return &value
}
