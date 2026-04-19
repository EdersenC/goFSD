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
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

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
	Steering           float64           `json:"steering"`
	DeltaSpeed         float64           `json:"deltaSpeed"`
	CurrentSpeed       float64           `json:"currentSpeed"`
	TelemetryAgeMs     int64             `json:"telemetryAgeMs"`
	ControlSemantics   string            `json:"controlSemantics,omitempty"`
	CapturedAt         string            `json:"capturedAt"`
	PredictedAt        string            `json:"predictedAt"`
	FrameIndex         int               `json:"frameIndex"`
	Sequence           int               `json:"sequence"`
	SourceFPS          int               `json:"sourceFps"`
	InferenceHz        int               `json:"inferenceHz"`
	ModelServerURL     string            `json:"modelServerUrl"`
	Checkpoint         string            `json:"checkpoint,omitempty"`
	ModelDevice        string            `json:"modelDevice,omitempty"`
	ControlSources     map[string]string `json:"controlSources,omitempty"`
	WindowFrameIndices []int             `json:"windowFrameIndices"`
	WindowFrameHashes  []string          `json:"windowFrameHashes"`
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
	capturedAt     time.Time
	sequenceNumber int
}

type bufferedInferenceFrame struct {
	index      int
	capturedAt time.Time
	image      *image.RGBA
}

type pythonPredictResponse struct {
	Checkpoint           string            `json:"checkpoint"`
	Device               string            `json:"device"`
	ControlSemantics     string            `json:"control_semantics"`
	ControlTargetSources map[string]string `json:"control_target_sources"`
	Prediction           struct {
		Steering       *float64 `json:"steering"`
		LegacySteering *float64 `json:"Steering"`
		DeltaSpeed     *float64 `json:"delta_speed"`
		Acceleration   *float64 `json:"acceleration"`
	} `json:"prediction"`
	ControlOutputs struct {
		Steer      *float64 `json:"steer"`
		DeltaSpeed *float64 `json:"delta_speed"`
		Accel      *float64 `json:"accel"`
	} `json:"control_outputs"`
}

type pythonModelsResponse struct {
	Models []InferenceModelOption `json:"models"`
}

type Inferencer struct {
	mu                  sync.Mutex
	ffmpegBin           string
	discover            SourceDiscovery
	probe               CapabilityProbe
	newCommand          inferenceCommandFactory
	httpClient          *http.Client
	nowFunc             func() time.Time
	requestTimeout      time.Duration
	config              InferenceConfig
	modelServerURL      string
	autoLoad            bool
	sourceID            string
	telemetry           *control.Store
	telemetryStaleAfter time.Duration
	status              InferenceStatus
	active              *inferenceSession
}

func NewInferencer(cfg InferenceConfig, telemetry *control.Store) *Inferencer {
	if cfg.ModelServerURL == "" {
		cfg = DefaultInferenceConfig()
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
		modelServerURL:      cfg.ModelServerURL,
		autoLoad:            cfg.AutoLoad,
		sourceID:            cfg.SourceID,
		telemetry:           telemetry,
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
	return i.postRemoteModelLoad(ctx, modelServerURL, payload)
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
	buffer := make([]bufferedInferenceFrame, 0, requiredFrameCount(i.config.WindowSize, i.config.FrameStride))
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
		if len(buffer) > requiredFrameCount(i.config.WindowSize, i.config.FrameStride) {
			buffer = buffer[1:]
		}

		i.mu.Lock()
		i.status.FramesSeen++
		i.mu.Unlock()

		if shouldDispatchInferenceFrame(frameIndex, i.config.WindowSize, i.config.FrameStride, i.config.DispatchStride) {
			sequence++
			window := buildPredictionWindow(buffer, sequence, i.config.WindowSize, i.config.FrameStride)
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
		case window := <-session.predictQ:
			prediction, err := i.requestPrediction(ctx, modelServerURL, window)
			if err != nil {
				i.recordPredictionError(err)
				continue
			}
			i.mu.Lock()
			i.status.LastPrediction = prediction
			i.status.PredictionsSent++
			i.status.LastError = ""
			i.mu.Unlock()
		}
	}
}

func (i *Inferencer) requestPrediction(ctx context.Context, modelServerURL string, window predictionWindow) (*InferencePrediction, error) {
	currentSpeed, telemetryAge, err := i.liveCurrentSpeed()
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

	body, err := json.Marshal(map[string]any{
		"frames_base64": framesBase64,
		"current_speed": currentSpeed,
		"sequence":      window.sequenceNumber,
		"timestamp_ms":  i.nowFunc().UTC().UnixMilli(),
	})
	if err != nil {
		return nil, err
	}

	requestCtx, cancel := context.WithTimeout(ctx, i.requestTimeout)
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
	steering, deltaSpeed, err := parsed.controlPrediction()
	if err != nil {
		return nil, err
	}

	return &InferencePrediction{
		Steering:           steering,
		DeltaSpeed:         deltaSpeed,
		CurrentSpeed:       currentSpeed,
		TelemetryAgeMs:     telemetryAge.Milliseconds(),
		ControlSemantics:   parsed.ControlSemantics,
		CapturedAt:         window.capturedAt.Format(time.RFC3339Nano),
		PredictedAt:        i.nowFunc().UTC().Format(time.RFC3339Nano),
		FrameIndex:         window.frameIndex,
		Sequence:           window.sequenceNumber,
		SourceFPS:          i.config.FPS,
		InferenceHz:        i.config.FPS / i.config.DispatchStride,
		ModelServerURL:     modelServerURL,
		Checkpoint:         parsed.Checkpoint,
		ModelDevice:        parsed.Device,
		ControlSources:     cloneStringMap(parsed.ControlTargetSources),
		WindowFrameIndices: append([]int(nil), window.frameIndices...),
		WindowFrameHashes:  append([]string(nil), frameHashes...),
	}, nil
}

func (i *Inferencer) liveCurrentSpeed() (float64, time.Duration, error) {
	if i.telemetry == nil {
		return 0, 0, errors.New("inference telemetry store is not configured")
	}
	telemetry, updatedAt := i.telemetry.LatestTelemetrySnapshot()
	if telemetry == nil || updatedAt.IsZero() {
		return 0, 0, errors.New("live currentSpeed telemetry is unavailable")
	}
	age := i.nowFunc().Sub(updatedAt)
	if age > i.telemetryStaleAfter {
		return 0, age, fmt.Errorf("live currentSpeed telemetry is stale: age=%s", age)
	}
	return telemetry.CurrentSpeed, age, nil
}

func (r pythonPredictResponse) controlPrediction() (float64, float64, error) {
	if r.ControlOutputs.Steer != nil && r.ControlOutputs.DeltaSpeed != nil {
		return *r.ControlOutputs.Steer, *r.ControlOutputs.DeltaSpeed, nil
	}
	if r.ControlOutputs.Steer != nil && r.ControlOutputs.Accel != nil {
		return *r.ControlOutputs.Steer, *r.ControlOutputs.Accel, nil
	}
	if r.Prediction.Steering != nil && r.Prediction.DeltaSpeed != nil {
		return *r.Prediction.Steering, *r.Prediction.DeltaSpeed, nil
	}
	if r.Prediction.Steering != nil && r.Prediction.Acceleration != nil {
		return *r.Prediction.Steering, *r.Prediction.Acceleration, nil
	}
	if r.Prediction.LegacySteering != nil && r.Prediction.DeltaSpeed != nil {
		return *r.Prediction.LegacySteering, *r.Prediction.DeltaSpeed, nil
	}
	if r.Prediction.LegacySteering != nil && r.Prediction.Acceleration != nil {
		return *r.Prediction.LegacySteering, *r.Prediction.Acceleration, nil
	}
	return 0, 0, fmt.Errorf("python predict response missing control outputs")
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
	if strings.TrimSpace(i.config.ModelDevice) != "" {
		payload["device"] = i.config.ModelDevice
	}
	_, err := i.postRemoteModelLoad(ctx, modelServerURL, payload)
	return err
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
}

func (i *Inferencer) recordPredictionError(err error) {
	i.mu.Lock()
	defer i.mu.Unlock()
	i.status.PredictionErrors++
	i.status.LastError = err.Error()
	if i.status.State == "running" {
		i.status.State = "error"
	}
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

func shouldDispatchInferenceFrame(frameIndex int, windowSize int, frameStride int, dispatchStride int) bool {
	if windowSize < 1 || windowSize%2 == 0 || frameStride < 1 || dispatchStride < 1 {
		return false
	}
	return frameIndex >= requiredFrameCount(windowSize, frameStride)-1 && frameIndex%dispatchStride == 0
}

func requiredFrameCount(windowSize int, frameStride int) int {
	if windowSize < 1 || frameStride < 1 {
		return 0
	}
	return ((windowSize - 1) * frameStride) + 1
}

func buildPredictionWindow(buffer []bufferedInferenceFrame, sequence int, windowSize int, frameStride int) *predictionWindow {
	needed := requiredFrameCount(windowSize, frameStride)
	if len(buffer) < needed {
		return nil
	}
	selected := buffer[len(buffer)-needed:]
	windowFrames := make([]*image.RGBA, 0, windowSize)
	windowIndices := make([]int, 0, windowSize)
	for idx := 0; idx < len(selected); idx += frameStride {
		windowFrames = append(windowFrames, selected[idx].image)
		windowIndices = append(windowIndices, selected[idx].index)
	}
	last := selected[len(selected)-1]
	return &predictionWindow{
		frames:         windowFrames,
		frameIndex:     last.index,
		frameIndices:   windowIndices,
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
		copyPrediction.ControlSources = cloneStringMap(status.LastPrediction.ControlSources)
		copyPrediction.WindowFrameIndices = append([]int(nil), status.LastPrediction.WindowFrameIndices...)
		copyPrediction.WindowFrameHashes = append([]string(nil), status.LastPrediction.WindowFrameHashes...)
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
