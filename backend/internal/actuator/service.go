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

	"awesomeProject/internal/control"
)

var ErrNotReady = errors.New("virtual controller is not ready")
var ErrInvalidInputMode = errors.New("invalid actuator input mode")

const InputModeNormalized = "normalized"

type CommandRequest struct {
	Steer            float64 `json:"steer"`
	Throttle         float64 `json:"throttle"`
	BrakePressureAvg float64 `json:"brakePressureAvg"`
	InputMode        string  `json:"inputMode,omitempty"`
	Handbrake        bool    `json:"handbrake"`
	Enabled          *bool   `json:"enabled,omitempty"`
	Sequence         int64   `json:"sequence,omitempty"`
	TimestampMs      int64   `json:"timestampMs,omitempty"`
}

type telemetryProvider interface {
	LatestTelemetrySnapshot() (*control.RuntimeTelemetry, time.Time)
	LatestActuatorEgoStateSnapshot() (*control.ActuatorEgoState, time.Time)
	UpdateAppliedControls(control.AppliedControls)
}

type SafetyDebug struct {
	CurrentSpeedMPS        float64 `json:"currentSpeedMps"`
	CurrentSpeedKPH        float64 `json:"currentSpeedKph"`
	SpeedLimitKPH          float64 `json:"speedLimitKph"`
	OverspeedKPH           float64 `json:"overspeedKph,omitempty"`
	ThrottleBefore         float64 `json:"throttleBefore"`
	ThrottleAfter          float64 `json:"throttleAfter"`
	BrakeBefore            float64 `json:"brakeBefore"`
	BrakeAfter             float64 `json:"brakeAfter"`
	ModelBrakeThreshold    float64 `json:"modelBrakeThreshold"`
	ReverseLockoutSpeedKPH float64 `json:"reverseLockoutSpeedKph"`
	BrakeThresholdApplied  bool    `json:"brakeThresholdApplied"`
	SpeedLimitActive       bool    `json:"speedLimitActive"`
	OverspeedBrakeApplied  bool    `json:"overspeedBrakeApplied"`
	ReverseLockoutApplied  bool    `json:"reverseLockoutApplied"`
	TelemetryAvailable     bool    `json:"telemetryAvailable"`
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
	Enabled   bool           `json:"enabled"`
	Stale     bool           `json:"stale"`
	Holding   bool           `json:"holding"`
	TimedOut  bool           `json:"timedOut"`
	UpdatedAt string         `json:"updatedAt,omitempty"`
	Safety    SafetyDebug    `json:"safety"`
	Temporal  *TemporalDebug `json:"temporal,omitempty"`
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

type Service struct {
	cfg Config

	mu                    sync.Mutex
	nowFunc               func() time.Time
	controller            controller
	cancel                context.CancelFunc
	done                  chan struct{}
	started               bool
	ready                 bool
	supported             bool
	lastError             string
	lastCmd               *commandEnvelope
	target                controllerSnapshot
	applied               AppliedState
	lastApplyError        string
	lastApplyAttemptedAt  string
	lastApplySucceededAt  string
	configPath            string
	liveTuning            Tuning
	savedTuning           Tuning
	telemetry             telemetryProvider
	temporalBuffer        TemporalPlanBuffer
	lastTemporalLogAt     time.Time
	lastTemporalPlanState string
}

func NewService(cfg Config, configPath string, telemetry ...telemetryProvider) *Service {
	tuning := cfg.Tuning()
	service := &Service{
		cfg:         cfg,
		nowFunc:     time.Now,
		supported:   runtime.GOOS == "windows",
		configPath:  configPath,
		liveTuning:  tuning,
		savedTuning: tuning,
		applied: AppliedState{
			Enabled: false,
			Stale:   true,
		},
		target: controllerSnapshot{
			Enabled: false,
			Stale:   true,
		},
	}
	if len(telemetry) > 0 {
		service.telemetry = telemetry[0]
	}
	return service
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
	s.applied = AppliedState{Enabled: false, Stale: true}
	s.target = controllerSnapshot{Enabled: false, Stale: true}
	s.temporalBuffer = TemporalPlanBuffer{}
	s.lastTemporalLogAt = time.Time{}
	s.lastTemporalPlanState = ""
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
		return s.stateLocked(), ErrUnsupportedPlatform
	}
	if !s.ready || s.controller == nil {
		return s.stateLocked(), ErrNotReady
	}
	cmd, err := buildCommandEnvelope(req, now)
	if err != nil {
		return s.stateLocked(), err
	}
	s.lastCmd = &cmd
	if s.cfg.TemporalHorizonActuatorEnabled {
		if !cmd.Enabled {
			s.temporalBuffer = TemporalPlanBuffer{}
			return s.stateLocked(), nil
		}
		inputTimestampS := timeToSeconds(now)
		if cmd.TimestampMs > 0 {
			inputTimestampS = float64(cmd.TimestampMs) / 1000.0
		}
		plan := LegacyPredictionAdapter(ControlCommand{
			Steering:         cmd.Steer,
			Throttle:         cmd.Throttle,
			BrakePressureAvg: cmd.Brake,
		}, inputTimestampS, timeToSeconds(now), nil, "legacy-command")
		if err := s.temporalBuffer.Accept(plan, now); err != nil {
			return s.stateLocked(), err
		}
	}
	return s.stateLocked(), nil
}

func (s *Service) SubmitPredictionHorizon(plan PredictionHorizon) (State, error) {
	now := s.nowFunc().UTC()
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.supported {
		return s.stateLocked(), ErrUnsupportedPlatform
	}
	if !s.ready || s.controller == nil {
		return s.stateLocked(), ErrNotReady
	}
	if err := s.temporalBuffer.Accept(plan, now); err != nil {
		return s.stateLocked(), err
	}
	return s.stateLocked(), nil
}

func (s *Service) State() State {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.stateLocked()
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

	target := s.targetLocked(now)
	s.target = target
	nextApplied := AppliedState{
		controlState: target.controlState,
		Enabled:      target.Enabled,
		Stale:        target.Stale,
		Holding:      target.Enabled && !target.TimedOut,
		TimedOut:     target.TimedOut,
		UpdatedAt:    now.Format(time.RFC3339Nano),
	}
	if target.TimedOut {
		nextApplied.controlState = controlState{}
		nextApplied.Enabled = false
	}

	s.lastApplyAttemptedAt = now.Format(time.RFC3339Nano)
	if err := s.controller.Apply(nextApplied.controlState); err != nil {
		s.lastApplyError = err.Error()
		return err
	}
	s.applied = nextApplied
	s.recordAppliedControlsLocked(now, nextApplied.controlState)
	s.lastApplyError = ""
	s.lastApplySucceededAt = now.Format(time.RFC3339Nano)
	return nil
}

func (s *Service) targetLocked(now time.Time) controllerSnapshot {
	if s.cfg.TemporalHorizonActuatorEnabled {
		return s.temporalTargetLocked(now)
	}
	target, stale, enabled, timedOut := resolveTarget(s.lastCmd, s.cfg.StaleTimeout, now)
	var safety SafetyDebug
	target, safety = s.applyLiveTuningLocked(target)
	return controllerSnapshot{
		controlState: target,
		Enabled:      enabled,
		Stale:        stale,
		Holding:      enabled && !timedOut,
		TimedOut:     timedOut,
		UpdatedAt:    now.Format(time.RFC3339Nano),
		Safety:       safety,
	}
}

func resolveTarget(cmd *commandEnvelope, staleTimeout time.Duration, now time.Time) (controlState, bool, bool, bool) {
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
	if isStale {
		return controlState{}, true, false, true
	}
	resolved := commandToControlState(cmd)
	return resolved, isStale, true, false
}

func commandToControlState(cmd *commandEnvelope) controlState {
	return controlState{
		Steer:     clamp(cmd.Steer, -1, 1),
		Throttle:  clamp(cmd.Throttle, 0, 1),
		Brake:     clamp(cmd.Brake, 0, 1),
		Handbrake: cmd.Handbrake,
	}
}

func (s *Service) temporalTargetLocked(now time.Time) controllerSnapshot {
	egoState := control.ActuatorEgoState{
		TimestampS:    timeToSeconds(now),
		SpeedMPS:      0,
		Valid:         false,
		InvalidReason: "actuator ego telemetry unavailable",
	}
	if latest := s.latestActuatorEgoStateLocked(); latest != nil {
		egoState = *latest
	}
	command, trace := s.temporalBuffer.ActuatorTick(timeToSeconds(now), egoState, s.cfg)
	target := controlState{
		Steer:     command.Steering,
		Throttle:  command.Throttle,
		Brake:     command.BrakePressureAvg,
		Handbrake: false,
	}
	var safety SafetyDebug
	target, safety = s.applyLiveTuningLocked(target)
	trace.AppliedSteer = target.Steer
	trace.AppliedThrottle = target.Throttle
	trace.AppliedBrake = target.Brake
	s.logTemporalTraceLocked(trace, now)
	enabled := trace.PlanState != PlanStateMissing
	stale := trace.PlanState == PlanStateMissing || trace.PlanState == PlanStateStale || trace.PlanState == PlanStateExpired
	traceCopy := trace
	return controllerSnapshot{
		controlState: target,
		Enabled:      enabled,
		Stale:        stale,
		Holding:      enabled && trace.PlanState != PlanStateExpired,
		TimedOut:     false,
		UpdatedAt:    now.Format(time.RFC3339Nano),
		Safety:       safety,
		Temporal:     &traceCopy,
	}
}

func (s *Service) logTemporalTraceLocked(trace TemporalDebug, now time.Time) {
	if !trace.Enabled {
		return
	}
	stateChanged := trace.PlanState != s.lastTemporalPlanState
	periodic := s.lastTemporalLogAt.IsZero() || now.Sub(s.lastTemporalLogAt) >= time.Second
	expiredRepeat := trace.PlanState == PlanStateExpired && now.Sub(s.lastTemporalLogAt) >= 250*time.Millisecond
	if !stateChanged && !periodic && !expiredRepeat {
		return
	}
	LogTemporalDebug(trace)
	s.lastTemporalLogAt = now
	s.lastTemporalPlanState = trace.PlanState
}

func (s *Service) TuningState() TuningState {
	s.mu.Lock()
	defer s.mu.Unlock()
	return TuningState{
		Live:          s.liveTuning,
		Saved:         s.savedTuning,
		ConfigPath:    s.configPath,
		SaveSupported: strings.TrimSpace(s.configPath) != "",
	}
}

func (s *Service) ApplyTuning(tuning Tuning) (TuningState, error) {
	if err := ValidateTuning(tuning); err != nil {
		return TuningState{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.liveTuning = tuning
	return TuningState{
		Live:          s.liveTuning,
		Saved:         s.savedTuning,
		ConfigPath:    s.configPath,
		SaveSupported: strings.TrimSpace(s.configPath) != "",
	}, nil
}

func (s *Service) ResetTuning() TuningState {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.liveTuning = s.savedTuning
	return TuningState{
		Live:          s.liveTuning,
		Saved:         s.savedTuning,
		ConfigPath:    s.configPath,
		SaveSupported: strings.TrimSpace(s.configPath) != "",
	}
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
	return TuningState{
		Live:          s.liveTuning,
		Saved:         s.savedTuning,
		ConfigPath:    s.configPath,
		SaveSupported: strings.TrimSpace(s.configPath) != "",
	}, nil
}

func (s *Service) applyLiveTuningLocked(input controlState) (controlState, SafetyDebug) {
	tuning := s.liveTuning
	steer := clamp(input.Steer*s.liveTuning.SteeringGain, -1, 1)
	throttle := clamp(input.Throttle*s.liveTuning.ThrottleGain, 0, 1)
	brake := clamp(input.Brake, 0, 1)
	debug := SafetyDebug{
		SpeedLimitKPH:          tuning.SpeedLimitKPH,
		ThrottleBefore:         throttle,
		BrakeBefore:            brake,
		ModelBrakeThreshold:    tuning.ModelBrakeThreshold,
		ReverseLockoutSpeedKPH: tuning.ReverseLockoutSpeedKPH,
	}
	if throttle > 0 && throttle < s.liveTuning.ThrottleFloor {
		throttle = s.liveTuning.ThrottleFloor
	}
	if brake <= tuning.ModelBrakeThreshold {
		if brake > 0 {
			debug.BrakeThresholdApplied = true
		}
		brake = 0
	}
	if latest := s.latestTelemetryLocked(); latest != nil {
		currentSpeedMPS := math.Max(latest.CurrentSpeed, 0)
		currentSpeedKPH := currentSpeedMPS * 3.6
		debug.TelemetryAvailable = true
		debug.CurrentSpeedMPS = currentSpeedMPS
		debug.CurrentSpeedKPH = currentSpeedKPH
		if tuning.SpeedLimitKPH > 0 && currentSpeedKPH >= tuning.SpeedLimitKPH {
			debug.SpeedLimitActive = true
			debug.OverspeedKPH = currentSpeedKPH - tuning.SpeedLimitKPH
			throttle = 0
			if debug.OverspeedKPH >= tuning.OverspeedBrakeMarginKPH && tuning.OverspeedBrake > brake {
				brake = tuning.OverspeedBrake
				debug.OverspeedBrakeApplied = true
			}
		}
		if currentSpeedKPH <= tuning.ReverseLockoutSpeedKPH && brake > 0 {
			brake = 0
			debug.ReverseLockoutApplied = true
		}
	}
	resolved, _ := resolveServiceThrottleBrakeConflict(controlState{
		Steer:     steer,
		Throttle:  throttle,
		Brake:     brake,
		Handbrake: input.Handbrake,
	})
	debug.ThrottleAfter = resolved.Throttle
	debug.BrakeAfter = resolved.Brake
	return resolved, debug
}

func (s *Service) latestTelemetryLocked() *control.RuntimeTelemetry {
	if s.telemetry == nil {
		return nil
	}
	latest, _ := s.telemetry.LatestTelemetrySnapshot()
	return latest
}

func (s *Service) latestActuatorEgoStateLocked() *control.ActuatorEgoState {
	if s.telemetry == nil {
		return nil
	}
	latest, _ := s.telemetry.LatestActuatorEgoStateSnapshot()
	return latest
}

func (s *Service) recordAppliedControlsLocked(now time.Time, applied controlState) {
	if s.telemetry == nil {
		return
	}
	s.telemetry.UpdateAppliedControls(control.AppliedControls{
		Steer:      applied.Steer,
		Throttle:   applied.Throttle,
		Brake:      applied.Brake,
		TimestampS: timeToSeconds(now),
	})
}

func (s *Service) stateLocked() State {
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
			Brake:     req.BrakePressureAvg,
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
	case "", InputModeNormalized:
		return InputModeNormalized, nil
	default:
		return "", fmt.Errorf("%w: %q", ErrInvalidInputMode, raw)
	}
}

func controlStateEqual(left controlState, right controlState) bool {
	return left.Steer == right.Steer &&
		left.Throttle == right.Throttle &&
		left.Brake == right.Brake &&
		left.Handbrake == right.Handbrake
}

func resolveServiceThrottleBrakeConflict(input controlState) (controlState, bool) {
	output := input
	if output.Brake <= 0.05 || output.Throttle <= 0.05 {
		return output, false
	}
	if output.Brake >= output.Throttle {
		output.Throttle = 0
		return output, true
	}
	output.Throttle = clamp(output.Throttle-output.Brake, 0, 1)
	return output, output != input
}

func (s *Service) Unsupported() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return !s.supported
}
