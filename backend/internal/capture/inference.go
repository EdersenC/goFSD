package capture

import (
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
	"strings"
	"sync"
	"time"

	"awesomeProject/internal/actuator"
	"awesomeProject/internal/control"
)

const (
	defaultInferenceModelServerURL = "http://127.0.0.1:8090"
	defaultInferenceSourceID       = "monitor-2"
	defaultInferenceFPS            = 30
	defaultInferenceWindowSize     = 3
	defaultInferenceStride         = 2
	defaultInferenceWidth          = 480
	defaultInferenceHeight         = 480
	defaultInferenceRequestTimeout = 5 * time.Second
	defaultInferenceJPEGQuality    = 90
	defaultDebugFrameDumpLimit     = 30
	defaultTelemetryStaleAfter     = 500 * time.Millisecond
)

var (
	ErrInferenceAlreadyRunning = errors.New("inference already running")
	ErrInferenceNotRunning     = errors.New("inference is not running")
	ErrInferenceStartFailed    = errors.New("failed to start inference")
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
	CapturedAt                    string                    `json:"capturedAt"`
	PredictedAt                   string                    `json:"predictedAt"`
	WindowFrameIndices            []int                     `json:"windowFrameIndices"`
	WindowFrameHashes             []string                  `json:"windowFrameHashes"`
	WindowFrameTimestampsMs       []int64                   `json:"windowFrameTimestampsMs"`
	SelectedTelemetryOffsets      []int                     `json:"selectedTelemetryOffsets"`
	SelectedTelemetryTimestampsMs []int64                   `json:"selectedTelemetryTimestampsMs"`
	ImageTensorShape              []int                     `json:"imageTensorShape"`
	TelemetryTensorShape          []int                     `json:"telemetryTensorShape"`
	PredControlsShape             []int                     `json:"predControlsShape"`
	PredAuxShape                  []int                     `json:"predAuxShape,omitempty"`
	LastTelemetry                 *control.RuntimeTelemetry `json:"lastTelemetry,omitempty"`
	RawPredControls               [][]float64               `json:"rawPredControls"`
	RawPredAux                    [][]float64               `json:"rawPredAux,omitempty"`
	CollapsedCommand              actuator.ControlCommand   `json:"collapsedCommand"`
	PostProcessedCommand          actuator.ControlCommand   `json:"postProcessedCommand"`
	ProcessorDebug                actuator.ProcessingDebug  `json:"processorDebug"`
	ProcessorState                actuator.ProcessorState   `json:"processorState"`
	FallbackApplied               bool                      `json:"fallbackApplied"`
}

type InferenceStatus struct {
	State            string               `json:"state"`
	SourceID         string               `json:"sourceId,omitempty"`
	SourceFPS        int                  `json:"sourceFps"`
	InferenceHz      int                  `json:"inferenceHz"`
	WindowSize       int                  `json:"windowSize"`
	FrameStride      int                  `json:"frameStride"`
	DispatchStride   int                  `json:"dispatchStride"`
	FrameWidth       int                  `json:"frameWidth"`
	FrameHeight      int                  `json:"frameHeight"`
	ModelServerURL   string               `json:"modelServerUrl,omitempty"`
	StartedAt        string               `json:"startedAt,omitempty"`
	StoppedAt        string               `json:"stoppedAt,omitempty"`
	DebugFramesDir   string               `json:"debugFramesDir,omitempty"`
	DebugFramesSaved int                  `json:"debugFramesSaved"`
	DebugFramesLimit int                  `json:"debugFramesLimit"`
	LastPrediction   *InferencePrediction `json:"lastPrediction,omitempty"`
	FramesSeen       int                  `json:"framesSeen"`
	PredictionsSent  int                  `json:"predictionsSent"`
	PredictionErrors int                  `json:"predictionErrors"`
	LastError        string               `json:"lastError,omitempty"`
}

type inferenceSession struct {
	cancel    context.CancelFunc
	stdin     io.WriteCloser
	stdout    io.ReadCloser
	stderr    io.ReadCloser
	cmd       *exec.Cmd
	done      chan error
	predictQ  chan predictionWindow
	frameDump *debugFrameDump
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

type actuatorSubmitter interface {
	Submit(req actuator.CommandRequest) (actuator.State, error)
}

type pythonPredictResponse struct {
	Checkpoint         string        `json:"checkpoint"`
	Device             string        `json:"device"`
	PlannerFormat      string        `json:"planner_format"`
	ControlTargetNames []string      `json:"control_target_names"`
	PredControls       [][][]float64 `json:"pred_controls"`
	PredAux            [][][]float64 `json:"pred_aux"`
}

type pythonModelsResponse struct {
	Models []InferenceModelOption `json:"models"`
}

type Inferencer struct {
	mu                   sync.Mutex
	ffmpegBin            string
	discover             SourceDiscovery
	probe                CapabilityProbe
	newCommand           inferenceCommandFactory
	httpClient           *http.Client
	nowFunc              func() time.Time
	requestTimeout       time.Duration
	config               InferenceConfig
	actuatorConfig       actuator.Config
	modelServerURL       string
	autoLoad             bool
	loadedCheckpoint     string
	loadedModelDevice    string
	sourceID             string
	telemetry            *control.Store
	actuator             actuatorSubmitter
	telemetryStaleAfter  time.Duration
	lastTelemetryWaitLog time.Time
	processorState       actuator.ProcessorState
	telemetryNormalizer  *telemetryNormalizer
	normalizationErr     error
	status               InferenceStatus
	active               *inferenceSession
}

func NewInferencer(cfg InferenceConfig, actuatorCfg actuator.Config, telemetry *control.Store, actuators ...actuatorSubmitter) *Inferencer {
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
		httpClient:          &http.Client{Timeout: cfg.RequestTimeout},
		nowFunc:             time.Now,
		requestTimeout:      cfg.RequestTimeout,
		config:              cfg,
		actuatorConfig:      actuatorCfg,
		modelServerURL:      cfg.ModelServerURL,
		autoLoad:            cfg.AutoLoad,
		sourceID:            cfg.SourceID,
		telemetry:           telemetry,
		actuator:            actuatorSink,
		telemetryStaleAfter: defaultTelemetryStaleAfter,
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
	inf.processorState.Stats.HistoryWindow = actuatorCfg.RecentCommandWindow
	if cfg.TelemetryNormalizationEnabled {
		inf.telemetryNormalizer, inf.normalizationErr = loadTelemetryNormalizer(cfg.TelemetryNormalizationStatsPath)
	}
	return inf
}

func (i *Inferencer) Status() InferenceStatus {
	i.mu.Lock()
	defer i.mu.Unlock()
	return cloneInferenceStatus(i.status)
}

func (i *Inferencer) Start(ctx context.Context, req InferenceStartRequest) (InferenceStatus, error) {
	modelServerURL := strings.TrimRight(strings.TrimSpace(req.ModelServerURL), "/")
	if modelServerURL == "" {
		modelServerURL = i.modelServerURL
	}
	autoLoad := i.autoLoad
	if req.AutoLoad != nil {
		autoLoad = *req.AutoLoad
	}

	sources, err := i.discover(ctx)
	if err != nil {
		return InferenceStatus{}, err
	}

	monitor, ok := monitorByID(sources, i.sourceID)
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

	i.mu.Lock()
	if i.active != nil {
		i.mu.Unlock()
		return InferenceStatus{}, ErrInferenceAlreadyRunning
	}

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
	i.resetControlStateLocked()
	i.mu.Unlock()

	spec := monitorCaptureSpec(monitor)
	loopCtx, cancel := context.WithCancel(context.Background())
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
	if err := cmd.Start(); err != nil {
		cancel()
		i.setInferenceError(err)
		return InferenceStatus{}, fmt.Errorf("%w: %v", ErrInferenceStartFailed, err)
	}

	frameDump, err := newDebugFrameDump(i.nowFunc())
	if err != nil {
		i.mu.Lock()
		i.status.LastError = fmt.Sprintf("failed to prepare debug frame dump: %v", err)
		i.mu.Unlock()
	}

	session := &inferenceSession{
		cancel:    cancel,
		stdin:     stdin,
		stdout:    stdout,
		stderr:    stderr,
		cmd:       cmd,
		done:      make(chan error, 1),
		predictQ:  make(chan predictionWindow, 1),
		frameDump: frameDump,
	}

	i.mu.Lock()
	i.active = session
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

	go i.waitForInference(session)
	go i.consumeInferenceStderr(session)
	go i.runPredictionWorker(loopCtx, session, modelServerURL)
	go i.consumeInferenceFrames(loopCtx, session)

	return status, nil
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
	return parsed.Models, nil
}

func (i *Inferencer) LoadModel(ctx context.Context, req InferenceModelLoadRequest) (map[string]any, error) {
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
		return nil, err
	}

	i.mu.Lock()
	i.modelServerURL = modelServerURL
	if checkpoint, ok := parsed["checkpoint"].(string); ok {
		i.loadedCheckpoint = strings.TrimSpace(checkpoint)
	} else if checkpoint := strings.TrimSpace(req.Checkpoint); checkpoint != "" {
		i.loadedCheckpoint = checkpoint
	}
	if resolvedDevice, ok := parsed["device"].(string); ok && strings.TrimSpace(resolvedDevice) != "" {
		i.loadedModelDevice = strings.TrimSpace(resolvedDevice)
	} else if device != "" {
		i.loadedModelDevice = device
	}
	i.mu.Unlock()

	return parsed, nil
}

func (i *Inferencer) Stop(ctx context.Context) (InferenceStatus, error) {
	i.mu.Lock()
	session := i.active
	if session == nil {
		i.mu.Unlock()
		return InferenceStatus{}, ErrInferenceNotRunning
	}
	i.status.State = "stopping"
	i.mu.Unlock()

	session.cancel()
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

	return i.Status(), nil
}

func (i *Inferencer) waitForInference(session *inferenceSession) {
	err := session.cmd.Wait()
	i.mu.Lock()
	active := i.active
	if active != nil && active == session {
		i.active = nil
		i.resetControlStateLocked()
		if i.status.State == "stopping" {
			i.status.State = "idle"
			i.status.StoppedAt = i.nowFunc().UTC().Format(time.RFC3339Nano)
		} else if err != nil && !isExpectedExitErr(err) {
			i.status.State = "error"
			i.status.LastError = err.Error()
			i.status.StoppedAt = i.nowFunc().UTC().Format(time.RFC3339Nano)
		} else {
			i.status.State = "idle"
			i.status.StoppedAt = i.nowFunc().UTC().Format(time.RFC3339Nano)
		}
	}
	i.mu.Unlock()
	session.done <- err
}

func (i *Inferencer) consumeInferenceStderr(session *inferenceSession) {
	defer func() { _ = session.stderr.Close() }()
	buf, err := io.ReadAll(session.stderr)
	if err != nil {
		return
	}
	message := strings.TrimSpace(string(buf))
	if message == "" {
		return
	}
	i.mu.Lock()
	if i.status.LastError == "" {
		i.status.LastError = message
	}
	i.mu.Unlock()
}

func (i *Inferencer) consumeInferenceFrames(ctx context.Context, session *inferenceSession) {
	defer func() { _ = session.stdout.Close() }()
	frameBytes := make([]byte, inferenceRawFrameBytes(i.config.FrameWidth, i.config.FrameHeight))
	buffer := make([]bufferedInferenceFrame, 0, requiredFrameCount(i.config.ImageOffsets)+1)
	frameIndex := 0
	sequence := 0

	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		if _, err := io.ReadFull(session.stdout, frameBytes); err != nil {
			if !errors.Is(err, io.EOF) && !errors.Is(err, io.ErrUnexpectedEOF) && ctx.Err() == nil {
				i.setInferenceError(err)
			}
			return
		}

		capturedAt := i.nowFunc().UTC()
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
			prediction, command, err := i.requestPrediction(ctx, modelServerURL, window)
			if err != nil {
				i.recordPredictionError(err)
				fallbackPrediction, fallbackCommand, fallbackErr := i.buildFallbackPrediction(window, modelServerURL, err)
				if fallbackErr == nil {
					if actuatorErr := i.submitActuatorCommand(fallbackCommand); actuatorErr == nil {
						i.mu.Lock()
						i.status.LastPrediction = fallbackPrediction
						i.mu.Unlock()
					}
				}
				continue
			}
			if err := i.submitActuatorCommand(command); err != nil {
				i.recordPredictionError(err)
				continue
			}
			i.mu.Lock()
			i.status.LastPrediction = prediction
			i.status.PredictionsSent++
			i.status.LastError = ""
			i.mu.Unlock()
			i.logPlannerDebug(prediction)
		}
	}
}

func (i *Inferencer) requestPrediction(ctx context.Context, modelServerURL string, window predictionWindow) (*InferencePrediction, actuator.CommandRequest, error) {
	selection, err := i.buildPlannerSelection(window)
	if err != nil {
		return nil, actuator.CommandRequest{}, err
	}
	framesBase64 := make([]string, 0, len(window.frames))
	frameHashes := make([]string, 0, len(window.frames))
	for _, frame := range window.frames {
		encoded, err := i.encodeJPEGBase64(frame)
		if err != nil {
			return nil, actuator.CommandRequest{}, err
		}
		framesBase64 = append(framesBase64, encoded)
		frameHashes = append(frameHashes, hashInferencePayload(encoded))
	}

	bodyPayload := map[string]any{
		"planner_format":          i.config.PlannerFormat,
		"frames_base64":           framesBase64,
		"telemetry":               selection.telemetryTensor,
		"sequence":                window.sequenceNumber,
		"timestamp_ms":            i.nowFunc().UTC().UnixMilli(),
		"image_offsets":           i.config.ImageOffsets,
		"telemetry_offsets":       i.config.TelemetryOffsets,
		"telemetry_feature_names": i.config.TelemetryFeatureNames,
		"control_output_names":    i.config.ControlOutputNames,
	}
	body, err := json.Marshal(bodyPayload)
	if err != nil {
		return nil, actuator.CommandRequest{}, err
	}

	requestCtx, cancel := context.WithTimeout(ctx, i.config.PredictionTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(requestCtx, http.MethodPost, modelServerURL+"/predict", bytes.NewReader(body))
	if err != nil {
		return nil, actuator.CommandRequest{}, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := i.httpClient.Do(req)
	if err != nil {
		return nil, actuator.CommandRequest{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		payload, _ := io.ReadAll(resp.Body)
		return nil, actuator.CommandRequest{}, fmt.Errorf("python predict failed: status=%d body=%s", resp.StatusCode, strings.TrimSpace(string(payload)))
	}

	var parsed pythonPredictResponse
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return nil, actuator.CommandRequest{}, err
	}
	prediction, command, err := i.buildPrediction(
		parsed,
		modelServerURL,
		window,
		selection,
		frameHashes,
	)
	if err != nil {
		return nil, actuator.CommandRequest{}, err
	}
	return prediction, command, nil
}

func (i *Inferencer) submitActuatorCommand(command actuator.CommandRequest) error {
	if i.actuator == nil {
		return errors.New("inference actuator is not configured")
	}
	_, err := i.actuator.Submit(command)
	return err
}

func (i *Inferencer) latestTelemetryForDebug() *control.RuntimeTelemetry {
	if i.telemetry == nil {
		return nil
	}
	telemetry, _ := i.telemetry.LatestTelemetrySnapshot()
	return telemetry
}

type plannerSelection struct {
	selectedTelemetry []control.RuntimeTelemetry
	telemetryTensor   [][][]float64
	telemetryShape    []int
	frameShape        []int
	telemetryTimesMs  []int64
}

func (i *Inferencer) buildPlannerSelection(window predictionWindow) (plannerSelection, error) {
	if i.normalizationErr != nil {
		return plannerSelection{}, i.normalizationErr
	}
	if i.telemetry == nil {
		return plannerSelection{}, errors.New("planner telemetry store is not configured")
	}
	history := i.telemetry.TelemetryHistorySnapshot(512)
	if len(history) == 0 {
		return plannerSelection{}, errors.New("planner telemetry is unavailable")
	}
	latest := history[len(history)-1]
	latestAge := time.Duration(i.nowFunc().UTC().UnixMilli()-telemetryAlignmentTimestampMs(latest)) * time.Millisecond
	if latestAge > i.telemetryStaleAfter {
		return plannerSelection{}, fmt.Errorf("planner telemetry is stale: age=%s", latestAge)
	}
	anchorMs := window.capturedAt.UnixMilli()
	anchorIndex, err := findAnchorTelemetryIndex(history, anchorMs, i.config.AlignmentTolerance)
	if err != nil {
		return plannerSelection{}, err
	}
	selected := make([]control.RuntimeTelemetry, 0, len(i.config.TelemetryOffsets))
	times := make([]int64, 0, len(i.config.TelemetryOffsets))
	for _, offset := range i.config.TelemetryOffsets {
		index := anchorIndex + offset
		if index < 0 || index >= len(history) {
			return plannerSelection{}, fmt.Errorf("planner telemetry window is incomplete for offset %d", offset)
		}
		item := history[index]
		if telemetryAlignmentTimestampMs(item) > anchorMs {
			return plannerSelection{}, fmt.Errorf("planner telemetry window would use future telemetry at offset %d", offset)
		}
		selected = append(selected, item)
		times = append(times, telemetryAlignmentTimestampMs(item))
	}
	features := make([][]float64, 0, len(selected))
	for index := range selected {
		vector, err := i.telemetryFeatureVector(selected, index)
		if err != nil {
			return plannerSelection{}, err
		}
		features = append(features, vector)
	}
	return plannerSelection{
		selectedTelemetry: selected,
		telemetryTensor:   [][][]float64{features},
		telemetryShape:    []int{1, len(features), len(i.config.TelemetryFeatureNames)},
		frameShape:        []int{1, len(window.frames), 3, i.config.FrameHeight, i.config.FrameWidth},
		telemetryTimesMs:  times,
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
		"-loglevel", "error",
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
	base := fmt.Sprintf("scale=%d:%d:flags=lanczos,fps=%d", cfg.FrameWidth, cfg.FrameHeight, cfg.FPS)
	if spec.backend == "ddagrab" || spec.inputFormat == "lavfi" {
		return "hwdownload,format=bgra," + base + ",format=rgb24"
	}
	return base + ",format=rgb24"
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
		copyPrediction.SelectedTelemetryOffsets = append([]int(nil), status.LastPrediction.SelectedTelemetryOffsets...)
		copyPrediction.SelectedTelemetryTimestampsMs = append([]int64(nil), status.LastPrediction.SelectedTelemetryTimestampsMs...)
		copyPrediction.ImageTensorShape = append([]int(nil), status.LastPrediction.ImageTensorShape...)
		copyPrediction.TelemetryTensorShape = append([]int(nil), status.LastPrediction.TelemetryTensorShape...)
		copyPrediction.PredControlsShape = append([]int(nil), status.LastPrediction.PredControlsShape...)
		copyPrediction.PredAuxShape = append([]int(nil), status.LastPrediction.PredAuxShape...)
		copyPrediction.RawPredControls = clone2DFloat64(status.LastPrediction.RawPredControls)
		copyPrediction.RawPredAux = clone2DFloat64(status.LastPrediction.RawPredAux)
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
) (*InferencePrediction, actuator.CommandRequest, error) {
	controlNames, err := resolvePlannerControlNames(parsed.ControlTargetNames, i.config.ControlOutputNames, plannerTensorWidth(parsed.PredControls))
	if err != nil {
		return nil, actuator.CommandRequest{}, err
	}
	controls, err := validatePlannerTensor(parsed.PredControls, 1, i.config.FutureSteps, len(controlNames), "pred_controls")
	if err != nil {
		return nil, actuator.CommandRequest{}, err
	}
	aux, auxShape, err := validateOptionalPlannerTensor(parsed.PredAux, 1, i.config.FutureSteps, len(i.config.AuxOutputNames), "pred_aux")
	if err != nil {
		return nil, actuator.CommandRequest{}, err
	}
	rawCollapsed, err := collapsePlannerCommand(controls[0], controlNames, i.config)
	if err != nil {
		return nil, actuator.CommandRequest{}, err
	}
	finalCommand, processorDebug := actuator.ProcessActuatorCommand(rawCollapsed, &i.processorState, i.actuatorConfig)
	request := actuator.CommandRequest{
		Steer:            finalCommand.Steering,
		Throttle:         finalCommand.Throttle,
		BrakePressureAvg: finalCommand.BrakePressureAvg,
		InputMode:        actuator.InputModeNormalized,
		Handbrake:        false,
		Enabled:          boolPtr(true),
		Sequence:         int64(window.sequenceNumber),
		TimestampMs:      i.nowFunc().UTC().UnixMilli(),
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
		CapturedAt:                    window.capturedAt.Format(time.RFC3339Nano),
		PredictedAt:                   i.nowFunc().UTC().Format(time.RFC3339Nano),
		WindowFrameIndices:            append([]int(nil), window.frameIndices...),
		WindowFrameHashes:             append([]string(nil), frameHashes...),
		WindowFrameTimestampsMs:       frameTimesMs,
		SelectedTelemetryOffsets:      append([]int(nil), i.config.TelemetryOffsets...),
		SelectedTelemetryTimestampsMs: append([]int64(nil), selection.telemetryTimesMs...),
		ImageTensorShape:              append([]int(nil), selection.frameShape...),
		TelemetryTensorShape:          append([]int(nil), selection.telemetryShape...),
		PredControlsShape:             []int{1, i.config.FutureSteps, len(controlNames)},
		PredAuxShape:                  auxShape,
		LastTelemetry:                 cloneTelemetryPtr(i.latestTelemetryForDebug()),
		RawPredControls:               clone2DFloat64(controls[0]),
		RawPredAux:                    clone2DFloat64(aux),
		CollapsedCommand:              rawCollapsed,
		PostProcessedCommand:          finalCommand,
		ProcessorDebug:                processorDebug,
		ProcessorState:                i.processorState,
	}
	return prediction, request, nil
}

func (i *Inferencer) resetControlStateLocked() {
	i.processorState = actuator.ProcessorState{}
	i.processorState.Stats.HistoryWindow = i.actuatorConfig.RecentCommandWindow
}

func (i *Inferencer) buildFallbackPrediction(window predictionWindow, modelServerURL string, cause error) (*InferencePrediction, actuator.CommandRequest, error) {
	finalCommand, processorDebug := actuator.ApplyFallbackDecay(&i.processorState, i.actuatorConfig)
	request := actuator.CommandRequest{
		Steer:            finalCommand.Steering,
		Throttle:         finalCommand.Throttle,
		BrakePressureAvg: finalCommand.BrakePressureAvg,
		InputMode:        actuator.InputModeNormalized,
		Handbrake:        false,
		Enabled:          boolPtr(true),
		Sequence:         int64(window.sequenceNumber),
		TimestampMs:      i.nowFunc().UTC().UnixMilli(),
	}
	frameTimesMs := make([]int64, 0, len(window.frameTimes))
	for _, item := range window.frameTimes {
		frameTimesMs = append(frameTimesMs, item.UnixMilli())
	}
	prediction := &InferencePrediction{
		Sequence:                window.sequenceNumber,
		FrameIndex:              window.frameIndex,
		SourceFPS:               i.config.FPS,
		InferenceHz:             i.config.FPS / i.config.DispatchStride,
		ModelServerURL:          modelServerURL,
		Checkpoint:              i.loadedCheckpoint,
		ModelDevice:             i.loadedModelDevice,
		PlannerFormat:           i.config.PlannerFormat,
		CapturedAt:              window.capturedAt.Format(time.RFC3339Nano),
		PredictedAt:             i.nowFunc().UTC().Format(time.RFC3339Nano),
		WindowFrameIndices:      append([]int(nil), window.frameIndices...),
		WindowFrameTimestampsMs: frameTimesMs,
		ImageTensorShape:        []int{1, len(window.frames), 3, i.config.FrameHeight, i.config.FrameWidth},
		TelemetryTensorShape:    []int{1, len(i.config.TelemetryOffsets), len(i.config.TelemetryFeatureNames)},
		LastTelemetry:           cloneTelemetryPtr(i.latestTelemetryForDebug()),
		CollapsedCommand:        processorDebug.Raw,
		PostProcessedCommand:    finalCommand,
		ProcessorDebug:          processorDebug,
		ProcessorState:          i.processorState,
		FallbackApplied:         true,
	}
	if cause != nil {
		log.Printf("[inference] fallback applied seq=%d reason=%v steer=%.3f throttle=%.3f",
			window.sequenceNumber,
			cause,
			finalCommand.Steering,
			finalCommand.Throttle,
		)
	}
	return prediction, request, nil
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
	deltaMs := telemetryAlignmentTimestampMs(current) - telemetryAlignmentTimestampMs(previous)
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
		timestamp := telemetryAlignmentTimestampMs(history[index])
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

func telemetryAlignmentTimestampMs(sample control.RuntimeTelemetry) int64 {
	if sample.ReceivedAtMs > 0 {
		return sample.ReceivedAtMs
	}
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

func resolvePlannerControlNames(responseNames []string, configNames []string, predictedWidth int) ([]string, error) {
	if len(responseNames) > 0 {
		trimmed := make([]string, 0, len(responseNames))
		for _, name := range responseNames {
			name = strings.TrimSpace(name)
			if name == "" {
				continue
			}
			trimmed = append(trimmed, name)
		}
		if len(trimmed) < 2 || trimmed[0] != "steering" || trimmed[1] != "acceleration" {
			return nil, fmt.Errorf("planner control_target_names must start with [steering, acceleration], got=%v", trimmed)
		}
		return trimmed, nil
	}
	if len(configNames) < 2 {
		return nil, fmt.Errorf("planner control_output_names must have at least steering and acceleration")
	}
	if predictedWidth >= 2 && predictedWidth <= len(configNames) {
		return append([]string(nil), configNames[:predictedWidth]...), nil
	}
	return append([]string(nil), configNames...), nil
}

func plannerTensorWidth(predicted [][][]float64) int {
	if len(predicted) == 0 || len(predicted[0]) == 0 {
		return 0
	}
	return len(predicted[0][0])
}

func plannerControlIndex(names []string, target string) int {
	for index, name := range names {
		if strings.EqualFold(strings.TrimSpace(name), target) {
			return index
		}
	}
	return -1
}

func collapsePlannerCommand(horizon [][]float64, controlNames []string, cfg InferenceConfig) (actuator.ControlCommand, error) {
	if len(horizon) == 0 {
		return actuator.ControlCommand{}, nil
	}
	steeringIndex := plannerControlIndex(controlNames, "steering")
	throttleIndex := plannerControlIndex(controlNames, "acceleration")
	brakeIndex := plannerControlIndex(controlNames, "brakePressureAvg")
	if steeringIndex < 0 || throttleIndex < 0 {
		return actuator.ControlCommand{}, fmt.Errorf("planner controls missing steering/acceleration targets: %v", controlNames)
	}
	for _, row := range horizon {
		if steeringIndex >= len(row) || throttleIndex >= len(row) || (brakeIndex >= 0 && brakeIndex >= len(row)) {
			return actuator.ControlCommand{}, fmt.Errorf("planner control row width mismatch for targets %v", controlNames)
		}
	}
	if cfg.HorizonMode == "t_plus_1_only" {
		command := actuator.ControlCommand{
			Steering: horizon[0][steeringIndex],
			Throttle: horizon[0][throttleIndex],
		}
		if brakeIndex >= 0 {
			command.BrakePressureAvg = horizon[0][brakeIndex]
		}
		return command, nil
	}
	weights := cfg.HorizonControlWeights
	if len(horizon) < 3 {
		return actuator.ControlCommand{}, fmt.Errorf("weighted_short_horizon requires at least 3 predicted control rows, got=%d", len(horizon))
	}
	command := actuator.ControlCommand{
		Steering: (weights[0] * horizon[0][steeringIndex]) + (weights[1] * horizon[1][steeringIndex]) + (weights[2] * horizon[2][steeringIndex]),
		Throttle: (weights[0] * horizon[0][throttleIndex]) + (weights[1] * horizon[1][throttleIndex]) + (weights[2] * horizon[2][throttleIndex]),
	}
	if brakeIndex >= 0 {
		command.BrakePressureAvg = (weights[0] * horizon[0][brakeIndex]) + (weights[1] * horizon[1][brakeIndex]) + (weights[2] * horizon[2][brakeIndex])
	}
	return command, nil
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

func cloneTelemetryPtr(source *control.RuntimeTelemetry) *control.RuntimeTelemetry {
	if source == nil {
		return nil
	}
	copyValue := *source
	return &copyValue
}

func (i *Inferencer) logPlannerDebug(prediction *InferencePrediction) {
	if prediction == nil || prediction.Sequence > 5 {
		return
	}
	log.Printf("[inference] planner window seq=%d frameTs=%v telemetryTs=%v imageShape=%v telemetryShape=%v predShape=%v final steer=%.3f throttle=%.3f brake=%.3f",
		prediction.Sequence,
		prediction.WindowFrameTimestampsMs,
		prediction.SelectedTelemetryTimestampsMs,
		prediction.ImageTensorShape,
		prediction.TelemetryTensorShape,
		prediction.PredControlsShape,
		prediction.PostProcessedCommand.Steering,
		prediction.PostProcessedCommand.Throttle,
		prediction.PostProcessedCommand.BrakePressureAvg,
	)
}

func boolPtr(value bool) *bool {
	return &value
}

func clamp(value float64, minimum float64, maximum float64) float64 {
	if value < minimum {
		return minimum
	}
	if value > maximum {
		return maximum
	}
	return value
}
