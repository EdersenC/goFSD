package actuator

import (
	"context"
	"errors"
	"fmt"
	"math"
	"runtime"
	"strings"
	"sync"
	"time"
)

var ErrNotReady = errors.New("virtual controller is not ready")
var ErrInvalidInputMode = errors.New("invalid actuator input mode")

const (
	InputModeModelRaw   = "model_raw"
	InputModeNormalized = "normalized"
)

type CommandRequest struct {
	Steer       float64 `json:"steer"`
	Throttle    float64 `json:"throttle"`
	Brake       float64 `json:"brake"`
	InputMode   string  `json:"inputMode,omitempty"`
	Handbrake   bool    `json:"handbrake"`
	Enabled     *bool   `json:"enabled,omitempty"`
	Sequence    int64   `json:"sequence,omitempty"`
	TimestampMs int64   `json:"timestampMs,omitempty"`
}

type commandEnvelope struct {
	controlState
	Enabled     bool   `json:"enabled"`
	InputMode   string `json:"inputMode"`
	Sequence    int64  `json:"sequence,omitempty"`
	TimestampMs int64  `json:"timestampMs,omitempty"`
	ReceivedAt  string `json:"receivedAt"`
}

type controllerSnapshot struct {
	controlState
	Enabled   bool   `json:"enabled"`
	Stale     bool   `json:"stale"`
	Holding   bool   `json:"holding"`
	TimedOut  bool   `json:"timedOut"`
	UpdatedAt string `json:"updatedAt,omitempty"`
}

type AppliedState = controllerSnapshot

type State struct {
	Supported            bool               `json:"supported"`
	Ready                bool               `json:"ready"`
	Platform             string             `json:"platform"`
	ControllerType       string             `json:"controllerType"`
	TickHz               int                `json:"tickHz"`
	StaleTimeoutMs       int64              `json:"staleTimeoutMs"`
	LastError            string             `json:"lastError,omitempty"`
	LastCommand          *commandEnvelope   `json:"lastCommand,omitempty"`
	Target               controllerSnapshot `json:"target"`
	Applied              AppliedState       `json:"applied"`
	LastApplyError       string             `json:"lastApplyError,omitempty"`
	LastApplyAttemptedAt string             `json:"lastApplyAttemptedAt,omitempty"`
	LastApplySucceededAt string             `json:"lastApplySucceededAt,omitempty"`
}

type TuningState struct {
	Live          Tuning `json:"live"`
	Saved         Tuning `json:"saved"`
	ConfigPath    string `json:"configPath,omitempty"`
	SaveSupported bool   `json:"saveSupported"`
}

type Service struct {
	cfg Config

	mu                   sync.Mutex
	nowFunc              func() time.Time
	controller           controller
	cancel               context.CancelFunc
	done                 chan struct{}
	lastTickAt           time.Time
	started              bool
	ready                bool
	supported            bool
	lastError            string
	lastCmd              *commandEnvelope
	target               controllerSnapshot
	applied              AppliedState
	lastApplyError       string
	lastApplyAttemptedAt string
	lastApplySucceededAt string
	configPath           string
	liveTuning           Tuning
	savedTuning          Tuning
}

func NewService(cfg Config, configPath string) *Service {
	initialTuning := cfg.Tuning()
	return &Service{
		cfg:         cfg,
		nowFunc:     time.Now,
		supported:   runtime.GOOS == "windows",
		configPath:  configPath,
		liveTuning:  initialTuning,
		savedTuning: initialTuning,
		applied: AppliedState{
			controlState: controlState{},
			Enabled:      false,
			Stale:        true,
		},
		target: controllerSnapshot{
			controlState: controlState{},
			Enabled:      false,
			Stale:        true,
		},
	}
}

func (s *Service) Start() error {
	s.mu.Lock()
	if s.started {
		s.mu.Unlock()
		return nil
	}
	s.started = true
	s.done = make(chan struct{})
	s.mu.Unlock()

	ctrl, err := newController()
	if err != nil {
		s.mu.Lock()
		s.lastError = err.Error()
		s.ready = false
		s.supported = !errors.Is(err, ErrUnsupportedPlatform)
		if errors.Is(err, ErrUnsupportedPlatform) {
			close(s.done)
			s.mu.Unlock()
			return nil
		}
		close(s.done)
		s.mu.Unlock()
		return err
	}

	ctx, cancel := context.WithCancel(context.Background())

	s.mu.Lock()
	s.controller = ctrl
	s.cancel = cancel
	s.ready = true
	s.supported = true
	s.lastError = ""
	s.lastTickAt = s.nowFunc()
	s.mu.Unlock()

	go s.run(ctx)
	return nil
}

func (s *Service) Close() error {
	s.mu.Lock()
	cancel := s.cancel
	done := s.done
	ctrl := s.controller
	s.cancel = nil
	s.controller = nil
	s.ready = false
	s.started = false
	s.applied.Enabled = false
	s.applied.Stale = true
	s.applied.controlState = controlState{}
	s.target.Enabled = false
	s.target.Stale = true
	s.target.controlState = controlState{}
	s.lastApplyError = ""
	s.lastApplyAttemptedAt = ""
	s.lastApplySucceededAt = ""
	s.mu.Unlock()

	if cancel != nil {
		cancel()
	}
	if done != nil {
		<-done
	}
	if ctrl != nil {
		ctrl.Close()
	}

	return nil
}

func (s *Service) Submit(req CommandRequest) (State, error) {
	now := s.nowFunc().UTC()
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.supported {
		return s.stateLocked(now), ErrUnsupportedPlatform
	}
	if !s.ready || s.controller == nil {
		return s.stateLocked(now), ErrNotReady
	}

	cmd, err := buildCommandEnvelope(req, now)
	if err != nil {
		return s.stateLocked(now), err
	}
	s.lastCmd = &cmd
	return s.stateLocked(now), nil
}

func (s *Service) State() State {
	now := s.nowFunc().UTC()
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.stateLocked(now)
}

func (s *Service) TuningState() TuningState {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.tuningStateLocked()
}

func (s *Service) ApplyTuning(tuning Tuning) (TuningState, error) {
	if err := ValidateTuning(tuning); err != nil {
		return TuningState{}, err
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	s.liveTuning = tuning
	return s.tuningStateLocked(), nil
}

func (s *Service) ResetTuning() TuningState {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.liveTuning = s.savedTuning
	return s.tuningStateLocked()
}

func (s *Service) SaveTuning() (TuningState, error) {
	s.mu.Lock()
	tuning := s.liveTuning
	path := s.configPath
	s.mu.Unlock()

	if strings.TrimSpace(path) == "" {
		return TuningState{}, fmt.Errorf("actuator config path is not available")
	}
	if err := SaveTuning(path, tuning); err != nil {
		return TuningState{}, err
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	s.savedTuning = tuning
	return s.tuningStateLocked(), nil
}

func (s *Service) run(ctx context.Context) {
	defer close(s.done)

	interval := time.Second / time.Duration(s.cfg.TickHz)
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case tickAt := <-ticker.C:
			if err := s.step(tickAt.UTC()); err != nil {
				s.mu.Lock()
				s.lastError = err.Error()
				s.mu.Unlock()
			}
		}
	}
}

func (s *Service) step(now time.Time) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.controller == nil {
		return nil
	}

	delta := now.Sub(s.lastTickAt)
	if delta <= 0 {
		delta = time.Second / time.Duration(s.cfg.TickHz)
	}
	s.lastTickAt = now

	target := s.targetLocked(now)
	s.target = target

	nextApplied := s.applied
	switch {
	case target.TimedOut:
		nextApplied.Steer = moveTowards(s.applied.Steer, target.Steer, s.liveTuning.SteerRatePerSecond*delta.Seconds())
		nextApplied.Throttle = moveTowards(s.applied.Throttle, target.Throttle, s.liveTuning.ThrottleRatePerSec*delta.Seconds())
		nextApplied.Brake = moveTowards(s.applied.Brake, target.Brake, s.liveTuning.BrakeRatePerSecond*delta.Seconds())
		nextApplied.Handbrake = target.Handbrake
		nextApplied.Enabled = false
	case !target.Enabled:
		nextApplied.controlState = target.controlState
		nextApplied.Enabled = false
	default:
		nextApplied.Steer = moveTowards(s.applied.Steer, target.Steer, s.liveTuning.SteerRatePerSecond*delta.Seconds())
		nextApplied.Throttle = moveTowards(s.applied.Throttle, target.Throttle, s.liveTuning.ThrottleRatePerSec*delta.Seconds())
		nextApplied.Brake = moveTowards(s.applied.Brake, target.Brake, s.liveTuning.BrakeRatePerSecond*delta.Seconds())
		nextApplied.Handbrake = target.Handbrake
		nextApplied.Enabled = true
	}
	nextApplied.Stale = target.Stale
	nextApplied.TimedOut = target.TimedOut
	nextApplied.Holding = target.Enabled && !target.TimedOut && controlStateEqual(nextApplied.controlState, target.controlState)
	nextApplied.UpdatedAt = now.Format(time.RFC3339Nano)

	s.lastApplyAttemptedAt = now.Format(time.RFC3339Nano)
	if err := s.controller.Apply(nextApplied.controlState); err != nil {
		s.lastApplyError = err.Error()
		return err
	}

	s.applied = nextApplied
	s.lastApplyError = ""
	s.lastApplySucceededAt = now.Format(time.RFC3339Nano)
	return nil
}

func (s *Service) targetLocked(now time.Time) controllerSnapshot {
	target, stale, enabled, timedOut := resolveTarget(s.lastCmd, s.liveTuning, s.cfg.StaleTimeout, s.cfg.HoldLastCommand, now)
	return controllerSnapshot{
		controlState: target,
		Enabled:      enabled,
		Stale:        stale,
		Holding:      enabled && !timedOut,
		TimedOut:     timedOut,
		UpdatedAt:    now.Format(time.RFC3339Nano),
	}
}

func resolveTarget(
	cmd *commandEnvelope,
	tuning Tuning,
	staleTimeout time.Duration,
	holdLastCommand bool,
	now time.Time,
) (controlState, bool, bool, bool) {
	if cmd == nil {
		return controlState{}, true, false, false
	}

	receivedAt, err := time.Parse(time.RFC3339Nano, cmd.ReceivedAt)
	if err != nil {
		return controlState{}, true, false, false
	}
	if !cmd.Enabled {
		return controlState{}, false, false, false
	}
	isStale := now.Sub(receivedAt) > staleTimeout
	if isStale && !holdLastCommand {
		return controlState{}, true, false, true
	}

	resolved, err := commandToControlState(cmd, tuning)
	if err != nil {
		return controlState{}, true, false, false
	}
	return resolved, isStale, true, false
}

func commandToControlState(cmd *commandEnvelope, tuning Tuning) (controlState, error) {
	inputMode, err := normalizeInputMode(cmd.InputMode)
	if err != nil {
		return controlState{}, err
	}

	steer := cmd.Steer
	throttle := cmd.Throttle
	brake := cmd.Brake
	if inputMode == InputModeModelRaw {
		steer *= tuning.ModelSteerScale
		throttle *= tuning.ModelAccelScale
		brake *= tuning.ModelAccelScale
	}

	steer = clamp(steer, -1, 1)
	throttle = clamp(throttle, 0, 1)
	brake = clamp(brake, 0, 1)

	steer = clamp(steer*tuning.SteerInputGain, -1, 1)
	if math.Abs(steer) < tuning.SteerDeadzone {
		steer = 0
	}
	steer = clamp(steer*tuning.MaxSteerScale, -1, 1)

	throttle = clamp(throttle*tuning.ThrottleInputGain, 0, 1)
	brake = clamp(brake*tuning.BrakeInputGain, 0, 1)
	if brake > 0 {
		throttle = 0
	}

	return controlState{
		Steer:     steer,
		Throttle:  throttle,
		Brake:     brake,
		Handbrake: cmd.Handbrake,
	}, nil
}

func (s *Service) stateLocked(now time.Time) State {
	state := State{
		Supported:            s.supported,
		Ready:                s.ready,
		Platform:             runtime.GOOS,
		ControllerType:       "xbox360",
		TickHz:               s.cfg.TickHz,
		StaleTimeoutMs:       s.cfg.StaleTimeout.Milliseconds(),
		LastError:            s.lastError,
		Target:               s.target,
		Applied:              s.applied,
		LastApplyError:       s.lastApplyError,
		LastApplyAttemptedAt: s.lastApplyAttemptedAt,
		LastApplySucceededAt: s.lastApplySucceededAt,
	}

	if s.lastCmd != nil {
		copyCmd := *s.lastCmd
		state.LastCommand = &copyCmd
	}

	return state
}

func (s *Service) tuningStateLocked() TuningState {
	return TuningState{
		Live:          s.liveTuning,
		Saved:         s.savedTuning,
		ConfigPath:    s.configPath,
		SaveSupported: strings.TrimSpace(s.configPath) != "",
	}
}

func clamp(value float64, min float64, max float64) float64 {
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return 0
	}
	if value < min {
		return min
	}
	if value > max {
		return max
	}
	return value
}

func buildCommandEnvelope(req CommandRequest, now time.Time) (commandEnvelope, error) {
	inputMode, err := normalizeInputMode(req.InputMode)
	if err != nil {
		return commandEnvelope{}, err
	}

	cmd := commandEnvelope{
		controlState: controlState{
			Steer:     req.Steer,
			Throttle:  req.Throttle,
			Brake:     req.Brake,
			Handbrake: req.Handbrake,
		},
		Enabled:     true,
		InputMode:   inputMode,
		Sequence:    req.Sequence,
		TimestampMs: req.TimestampMs,
		ReceivedAt:  now.Format(time.RFC3339Nano),
	}
	if req.Enabled != nil {
		cmd.Enabled = *req.Enabled
	}
	return cmd, nil
}

func normalizeInputMode(raw string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "", InputModeModelRaw:
		return InputModeModelRaw, nil
	case InputModeNormalized:
		return InputModeNormalized, nil
	default:
		return "", fmt.Errorf("%w: %q", ErrInvalidInputMode, raw)
	}
}

func moveTowards(current float64, target float64, maxDelta float64) float64 {
	if maxDelta <= 0 {
		return target
	}
	delta := target - current
	if math.Abs(delta) <= maxDelta {
		return target
	}
	return current + math.Copysign(maxDelta, delta)
}

func controlStateEqual(left controlState, right controlState) bool {
	return left.Steer == right.Steer &&
		left.Throttle == right.Throttle &&
		left.Brake == right.Brake &&
		left.Handbrake == right.Handbrake
}

func (s *Service) Unsupported() bool {
	state := s.State()
	return !state.Supported
}

func (s *Service) CheckReady() error {
	state := s.State()
	if !state.Supported {
		return ErrUnsupportedPlatform
	}
	if !state.Ready {
		return fmt.Errorf("%w: %s", ErrNotReady, state.LastError)
	}
	return nil
}
