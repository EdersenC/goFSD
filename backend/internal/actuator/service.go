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

type CommandRequest struct {
	Steer       float64 `json:"steer"`
	Throttle    float64 `json:"throttle"`
	Brake       float64 `json:"brake"`
	Handbrake   bool    `json:"handbrake"`
	Enabled     *bool   `json:"enabled,omitempty"`
	Sequence    int64   `json:"sequence,omitempty"`
	TimestampMs int64   `json:"timestampMs,omitempty"`
}

type commandEnvelope struct {
	controlState
	Enabled     bool   `json:"enabled"`
	Sequence    int64  `json:"sequence,omitempty"`
	TimestampMs int64  `json:"timestampMs,omitempty"`
	ReceivedAt  string `json:"receivedAt"`
}

type AppliedState struct {
	controlState
	Enabled   bool   `json:"enabled"`
	Stale     bool   `json:"stale"`
	UpdatedAt string `json:"updatedAt,omitempty"`
}

type State struct {
	Supported      bool             `json:"supported"`
	Ready          bool             `json:"ready"`
	Platform       string           `json:"platform"`
	ControllerType string           `json:"controllerType"`
	TickHz         int              `json:"tickHz"`
	StaleTimeoutMs int64            `json:"staleTimeoutMs"`
	LastError      string           `json:"lastError,omitempty"`
	LastCommand    *commandEnvelope `json:"lastCommand,omitempty"`
	Applied        AppliedState     `json:"applied"`
}

type TuningState struct {
	Live          Tuning `json:"live"`
	Saved         Tuning `json:"saved"`
	ConfigPath    string `json:"configPath,omitempty"`
	SaveSupported bool   `json:"saveSupported"`
}

type Service struct {
	cfg Config

	mu          sync.Mutex
	nowFunc     func() time.Time
	controller  controller
	cancel      context.CancelFunc
	done        chan struct{}
	lastTickAt  time.Time
	started     bool
	ready       bool
	supported   bool
	lastError   string
	lastCmd     *commandEnvelope
	applied     AppliedState
	configPath  string
	liveTuning  Tuning
	savedTuning Tuning
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
	cmd := commandEnvelope{
		controlState: controlState{
			Steer:     clamp(req.Steer, -1, 1),
			Throttle:  clamp(req.Throttle, 0, 1),
			Brake:     clamp(req.Brake, 0, 1),
			Handbrake: req.Handbrake,
		},
		Enabled:     true,
		Sequence:    req.Sequence,
		TimestampMs: req.TimestampMs,
		ReceivedAt:  now.Format(time.RFC3339Nano),
	}
	if req.Enabled != nil {
		cmd.Enabled = *req.Enabled
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.supported {
		return s.stateLocked(now), ErrUnsupportedPlatform
	}
	if !s.ready || s.controller == nil {
		return s.stateLocked(now), ErrNotReady
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

	target, stale, enabled := s.targetLocked(now)
	if stale || !enabled {
		s.applied.controlState = target
	} else {
		s.applied.Steer = moveTowards(s.applied.Steer, target.Steer, s.liveTuning.SteerRatePerSecond*delta.Seconds())
		s.applied.Throttle = moveTowards(s.applied.Throttle, target.Throttle, s.liveTuning.ThrottleRatePerSec*delta.Seconds())
		s.applied.Brake = moveTowards(s.applied.Brake, target.Brake, s.liveTuning.BrakeRatePerSecond*delta.Seconds())
	}
	s.applied.Handbrake = target.Handbrake
	s.applied.Enabled = enabled
	s.applied.Stale = stale
	s.applied.UpdatedAt = now.Format(time.RFC3339Nano)

	return s.controller.Apply(s.applied.controlState)
}

func (s *Service) targetLocked(now time.Time) (controlState, bool, bool) {
	if s.lastCmd == nil {
		return controlState{}, true, false
	}

	receivedAt, err := time.Parse(time.RFC3339Nano, s.lastCmd.ReceivedAt)
	if err != nil {
		return controlState{}, true, false
	}
	if !s.lastCmd.Enabled {
		return controlState{}, false, false
	}
	if now.Sub(receivedAt) > s.cfg.StaleTimeout {
		return controlState{}, true, false
	}

	steer := clamp(s.lastCmd.Steer*s.liveTuning.SteerInputGain, -1, 1)
	if math.Abs(steer) < s.liveTuning.SteerDeadzone {
		steer = 0
	}
	steer = clamp(steer*s.liveTuning.MaxSteerScale, -1, 1)

	throttle := clamp(s.lastCmd.Throttle*s.liveTuning.ThrottleInputGain, 0, 1)
	brake := clamp(s.lastCmd.Brake*s.liveTuning.BrakeInputGain, 0, 1)
	if brake > 0 {
		throttle = 0
	}

	return controlState{
		Steer:     steer,
		Throttle:  throttle,
		Brake:     brake,
		Handbrake: s.lastCmd.Handbrake,
	}, false, true
}

func (s *Service) stateLocked(now time.Time) State {
	state := State{
		Supported:      s.supported,
		Ready:          s.ready,
		Platform:       runtime.GOOS,
		ControllerType: "xbox360",
		TickHz:         s.cfg.TickHz,
		StaleTimeoutMs: s.cfg.StaleTimeout.Milliseconds(),
		LastError:      s.lastError,
		Applied:        s.applied,
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
