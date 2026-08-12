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
	"awesomeProject/internal/stopsigncontrol"
)

var ErrNotReady = errors.New("virtual controller is not ready")
var ErrInvalidInputMode = errors.New("invalid actuator input mode")
var ErrStopSignControllerNotReady = errors.New("stop-sign motion controller is not ready")
var ErrStopSignSessionOwned = errors.New("actuator is owned by an active stop-sign inference session")

const InputModeNormalized = "normalized"

const (
	OwnerStopSignInference = "stop-sign-inference"
	OwnerCalibration       = "calibration"
	maxTelemetryFutureSkew = 50 * time.Millisecond
)

type CommandRequest struct {
	Steer            float64 `json:"steer"`
	Throttle         float64 `json:"throttle"`
	BrakePressureAvg float64 `json:"brakePressureAvg"`
	InputMode        string  `json:"inputMode,omitempty"`
	Handbrake        bool    `json:"handbrake"`
	Enabled          *bool   `json:"enabled,omitempty"`
	Sequence         int64   `json:"sequence,omitempty"`
	TimestampMs      int64   `json:"timestampMs,omitempty"`
	Owner            string  `json:"owner,omitempty"`
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
	LowSpeedBrakeHold      bool    `json:"lowSpeedBrakeHold"`
	TelemetryAvailable     bool    `json:"telemetryAvailable"`
}

type commandEnvelope struct {
	controlState
	CommandID   int64  `json:"commandId"`
	Enabled     bool   `json:"enabled"`
	InputMode   string `json:"inputMode"`
	Sequence    int64  `json:"sequence,omitempty"`
	TimestampMs int64  `json:"timestampMs,omitempty"`
	ReceivedAt  string `json:"receivedAt"`
	Owner       string `json:"owner,omitempty"`
}

type StopSignControllerState struct {
	Ready                       bool                `json:"ready"`
	Contract                    string              `json:"contract"`
	Calibration                 StopSignCalibration `json:"calibration"`
	PlanTimeoutMs               int64               `json:"planTimeoutMs"`
	TelemetryTimeoutMs          int64               `json:"telemetryTimeoutMs"`
	EstimatedActuationLatencyMs int64               `json:"estimatedActuationLatencyMs"`
	ExpectedHorizonDtMs         []int               `json:"expectedHorizonDtMs"`
	Owner                       string              `json:"owner,omitempty"`
	Stopping                    bool                `json:"stopping"`
	LastPlanID                  int64               `json:"lastPlanId,omitempty"`
	LastPlanAcceptedAt          string              `json:"lastPlanAcceptedAt,omitempty"`
	LastPlanAppliedID           int64               `json:"lastPlanAppliedId,omitempty"`
	LastPlanAppliedAt           string              `json:"lastPlanAppliedAt,omitempty"`
	LastFault                   string              `json:"lastFault,omitempty"`
}

type StopSignControlDebug struct {
	PlanID               int64                     `json:"plan_id"`
	PlanState            string                    `json:"plan_state"`
	PlanAgeMs            float64                   `json:"plan_age_ms"`
	TargetDtMs           float64                   `json:"target_dt_ms"`
	TelemetryAgeMs       float64                   `json:"telemetry_age_ms"`
	SourceTelemetryAgeMs float64                   `json:"source_telemetry_age_ms"`
	SourceReceiptSkewMs  float64                   `json:"source_receipt_skew_ms"`
	SelectedSetpoint     *stopsigncontrol.Setpoint `json:"selected_setpoint,omitempty"`
	MeasuredWheelSteer   *float64                  `json:"measured_wheel_steer,omitempty"`
	CurrentSpeedMPS      float64                   `json:"current_speed_mps"`
	ControllerOutput     stopsigncontrol.Output    `json:"controller_output"`
	RequestedThrottle    float64                   `json:"requested_throttle"`
	RequestedBrake       float64                   `json:"requested_brake"`
	AppliedSteer         float64                   `json:"applied_steer"`
	AppliedThrottle      float64                   `json:"applied_throttle"`
	AppliedBrake         float64                   `json:"applied_brake"`
	AppliedHandbrake     bool                      `json:"applied_handbrake"`
	HorizonClamped       bool                      `json:"horizon_clamped"`
	FailSafe             bool                      `json:"fail_safe"`
	Fault                string                    `json:"fault,omitempty"`
}

type controllerSnapshot struct {
	controlState
	CommandID int64                 `json:"commandId,omitempty"`
	PlanID    int64                 `json:"planId,omitempty"`
	Enabled   bool                  `json:"enabled"`
	Stale     bool                  `json:"stale"`
	Holding   bool                  `json:"holding"`
	TimedOut  bool                  `json:"timedOut"`
	UpdatedAt string                `json:"updatedAt,omitempty"`
	Safety    SafetyDebug           `json:"safety"`
	StopSign  *StopSignControlDebug `json:"stopSign,omitempty"`
}

type AppliedState = controllerSnapshot

type State struct {
	Supported                   bool                    `json:"supported"`
	Ready                       bool                    `json:"ready"`
	Platform                    string                  `json:"platform"`
	ControllerType              string                  `json:"controllerType"`
	TickHz                      int                     `json:"tickHz"`
	StaleTimeoutMs              int64                   `json:"staleTimeoutMs"`
	LastError                   string                  `json:"lastError,omitempty"`
	LastCommand                 *commandEnvelope        `json:"lastCommand,omitempty"`
	LastCommandID               int64                   `json:"lastCommandId,omitempty"`
	Target                      controllerSnapshot      `json:"target"`
	Applied                     AppliedState            `json:"applied"`
	LastApplyError              string                  `json:"lastApplyError,omitempty"`
	LastApplyAttemptedAt        string                  `json:"lastApplyAttemptedAt,omitempty"`
	LastApplyAttemptedCommandID int64                   `json:"lastApplyAttemptedCommandId,omitempty"`
	LastApplyAttemptedPlanID    int64                   `json:"lastApplyAttemptedPlanId,omitempty"`
	LastApplySucceededAt        string                  `json:"lastApplySucceededAt,omitempty"`
	StopSignController          StopSignControllerState `json:"stopSignController"`
}

type Service struct {
	cfg Config

	mu                          sync.Mutex
	nowFunc                     func() time.Time
	controller                  controller
	cancel                      context.CancelFunc
	done                        chan struct{}
	started                     bool
	ready                       bool
	supported                   bool
	lastError                   string
	lastCmd                     *commandEnvelope
	nextCommandID               int64
	target                      controllerSnapshot
	applied                     AppliedState
	lastApplyError              string
	lastApplyAttemptedAt        string
	lastApplyAttemptedCommandID int64
	lastApplyAttemptedPlanID    int64
	lastApplySucceededAt        string
	configPath                  string
	liveTuning                  Tuning
	savedTuning                 Tuning
	telemetry                   telemetryProvider
	stopSignController          *stopsigncontrol.Controller
	stopSignControllerErr       error
	stopSignPlan                *stopsigncontrol.Plan
	stopSignPlanSampler         *stopsigncontrol.PlanSampler
	stopSignPlanID              int64
	nextStopSignPlanID          int64
	stopSignPlanAcceptedAt      time.Time
	lastStopSignTickAt          time.Time
	stopSignOwner               string
	stopSignStopping            bool
	lastStopSignAppliedID       int64
	lastStopSignAppliedAt       string
	lastStopSignFault           string
}

func NewService(cfg Config, configPath string, telemetry ...telemetryProvider) *Service {
	tuning := cfg.Tuning()
	stopSignController, stopSignControllerErr := stopsigncontrol.New(cfg.StopSignController)
	service := &Service{
		cfg:                   cfg,
		nowFunc:               time.Now,
		supported:             runtime.GOOS == "windows",
		configPath:            configPath,
		liveTuning:            tuning,
		savedTuning:           tuning,
		stopSignController:    stopSignController,
		stopSignControllerErr: stopSignControllerErr,
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
	s.lastCmd = nil
	s.applied = AppliedState{Enabled: false, Stale: true}
	s.target = controllerSnapshot{Enabled: false, Stale: true}
	s.resetStopSignControlLocked()
	s.stopSignOwner = ""
	s.stopSignStopping = false
	s.stopSignPlanID = 0
	s.lastStopSignAppliedID = 0
	s.lastStopSignAppliedAt = ""
	s.lastStopSignFault = ""
	s.lastApplyError = ""
	s.lastApplyAttemptedAt = ""
	s.lastApplyAttemptedCommandID = 0
	s.lastApplyAttemptedPlanID = 0
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
	if err := s.authorizeDirectCommandLocked(cmd); err != nil {
		return s.stateLocked(), err
	}
	s.nextCommandID++
	cmd.CommandID = s.nextCommandID
	s.lastCmd = &cmd
	if cmd.Owner == OwnerStopSignInference && cmd.Enabled {
		s.resetStopSignControlLocked()
		s.stopSignOwner = cmd.Owner
		s.stopSignStopping = false
		s.lastStopSignFault = ""
	}
	if !cmd.Enabled || cmd.Owner != OwnerStopSignInference {
		s.resetStopSignControlLocked()
		s.stopSignStopping = false
		if !cmd.Enabled {
			s.stopSignOwner = ""
		}
	}
	return s.stateLocked(), nil
}

// RequestStopSignSafetyStop keeps the stopSign session under actuator ownership
// while transitioning from service brake to handbrake using fresh vehicle speed.
func (s *Service) RequestStopSignSafetyStop() (State, error) {
	now := s.nowFunc().UTC()
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.supported {
		return s.stateLocked(), ErrUnsupportedPlatform
	}
	if !s.ready || s.controller == nil {
		return s.stateLocked(), ErrNotReady
	}
	if s.stopSignOwner != "" && s.stopSignOwner != OwnerStopSignInference {
		return s.stateLocked(), fmt.Errorf("%w: owner=%q", ErrStopSignSessionOwned, s.stopSignOwner)
	}

	s.nextCommandID++
	cmd := commandEnvelope{
		CommandID:  s.nextCommandID,
		Enabled:    true,
		InputMode:  InputModeNormalized,
		ReceivedAt: now.Format(time.RFC3339Nano),
		Owner:      OwnerStopSignInference,
	}
	s.lastCmd = &cmd
	s.stopSignOwner = OwnerStopSignInference
	s.stopSignStopping = true
	s.resetStopSignControlLocked()
	s.lastStopSignFault = ""
	return s.stateLocked(), nil
}

func (s *Service) authorizeDirectCommandLocked(cmd commandEnvelope) error {
	if s.stopSignOwner != "" && cmd.Owner != s.stopSignOwner {
		if !cmd.Enabled {
			return nil
		}
		return fmt.Errorf("%w: owner=%q", ErrStopSignSessionOwned, s.stopSignOwner)
	}
	if cmd.Owner == OwnerStopSignInference && cmd.Enabled && !controlStateEqual(cmd.controlState, controlState{}) {
		return fmt.Errorf("%w: stopSign inference may submit only a neutral arm command or setpoint plan", ErrStopSignSessionOwned)
	}
	return nil
}

// SubmitStopSignMotionPlan accepts only the versioned physical-state contract
// while an explicitly armed stop-sign inference session owns the actuator.
func (s *Service) SubmitStopSignMotionPlan(plan stopsigncontrol.Plan) (State, error) {
	now := s.nowFunc().UTC()
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.supported {
		return s.stateLocked(), ErrUnsupportedPlatform
	}
	if !s.ready || s.controller == nil {
		return s.stateLocked(), ErrNotReady
	}
	if s.stopSignControllerErr != nil || s.stopSignController == nil || !s.cfg.StopSignCalibration.Verified {
		return s.stateLocked(), fmt.Errorf("%w: verified vehicle calibration is required", ErrStopSignControllerNotReady)
	}
	if s.stopSignOwner != OwnerStopSignInference || s.lastCmd == nil || !s.lastCmd.Enabled || s.lastCmd.Owner != OwnerStopSignInference {
		return s.stateLocked(), fmt.Errorf("%w: stopSign inference has not armed the actuator", ErrStopSignSessionOwned)
	}
	if s.stopSignStopping {
		return s.stateLocked(), fmt.Errorf("%w: stopSign safety stop is active", ErrStopSignSessionOwned)
	}
	if detail := strings.TrimSpace(s.lastApplyError); detail != "" {
		return s.stateLocked(), fmt.Errorf("stopSign controller apply fault is active: %s", detail)
	}
	plan.ReceivedAtS = timeToSeconds(now)
	if err := stopsigncontrol.ValidatePlan(plan); err != nil {
		return s.stateLocked(), err
	}
	if err := validateStopSignPlanTiming(plan.Points, s.cfg.StopSignExpectedHorizonDtMs); err != nil {
		return s.stateLocked(), err
	}
	if age := now.Sub(secondsToTime(plan.SampledAtS)); age < 0 || age > s.cfg.StopSignPlanTimeout {
		return s.stateLocked(), fmt.Errorf("stopSign setpoint observation age %s is outside [0, %s]", age, s.cfg.StopSignPlanTimeout)
	}
	if s.stopSignPlan != nil && plan.SampledAtS <= s.stopSignPlan.SampledAtS {
		return s.stateLocked(), fmt.Errorf("stopSign setpoint sampled_at_s %.9f did not advance beyond %.9f", plan.SampledAtS, s.stopSignPlan.SampledAtS)
	}
	sampler, err := stopsigncontrol.NewPlanSampler(plan)
	if err != nil {
		return s.stateLocked(), err
	}
	copyPlan := plan
	copyPlan.Points = append([]stopsigncontrol.Setpoint(nil), plan.Points...)
	s.nextStopSignPlanID++
	s.stopSignPlanID = s.nextStopSignPlanID
	s.stopSignPlan = &copyPlan
	s.stopSignPlanSampler = sampler
	s.stopSignPlanAcceptedAt = now
	s.lastStopSignFault = ""
	return s.stateLocked(), nil
}

func validateStopSignPlanTiming(points []stopsigncontrol.Setpoint, expected []int) error {
	if len(points) != len(expected) {
		return fmt.Errorf("stopSign setpoint horizon timing differs: got %d points want %d", len(points), len(expected))
	}
	for index := range expected {
		if points[index].DtMs != expected[index] {
			return fmt.Errorf("stopSign setpoint horizon timing differs at %d: got=%dms want=%dms", index, points[index].DtMs, expected[index])
		}
	}
	return nil
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
		CommandID:    target.CommandID,
		PlanID:       target.PlanID,
		Enabled:      target.Enabled,
		Stale:        target.Stale,
		Holding:      target.Enabled && !target.TimedOut,
		TimedOut:     target.TimedOut,
		UpdatedAt:    now.Format(time.RFC3339Nano),
	}
	if target.TimedOut && target.StopSign == nil {
		nextApplied.controlState = controlState{}
		nextApplied.Enabled = false
	}

	s.lastApplyAttemptedAt = now.Format(time.RFC3339Nano)
	s.lastApplyAttemptedCommandID = nextApplied.CommandID
	s.lastApplyAttemptedPlanID = nextApplied.PlanID
	if err := s.controller.Apply(nextApplied.controlState); err != nil {
		s.lastApplyError = err.Error()
		if nextApplied.PlanID > 0 {
			s.lastStopSignFault = err.Error()
		}
		return err
	}
	s.applied = nextApplied
	if target.StopSign != nil && target.PlanID > 0 {
		s.lastStopSignAppliedID = target.PlanID
		s.lastStopSignAppliedAt = now.Format(time.RFC3339Nano)
	}
	s.recordAppliedControlsLocked(now, nextApplied.controlState)
	s.lastApplyError = ""
	s.lastApplySucceededAt = now.Format(time.RFC3339Nano)
	return nil
}

func (s *Service) targetLocked(now time.Time) controllerSnapshot {
	if s.stopSignOwner == OwnerStopSignInference {
		if s.stopSignStopping {
			return s.stopSignStopTargetLocked(now, "stopping")
		}
		if s.stopSignPlan != nil {
			return s.stopSignTargetLocked(now)
		}
		return s.stopSignStopTargetLocked(now, "awaiting-plan")
	}
	target, stale, enabled, timedOut := resolveTarget(s.lastCmd, s.cfg.StaleTimeout, now)
	var safety SafetyDebug
	target, safety = s.applyLiveTuningLocked(target)
	return controllerSnapshot{
		controlState: target,
		CommandID:    commandID(s.lastCmd),
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
	if !cmd.Enabled {
		return controlState{Handbrake: cmd.Handbrake}, false, false, false
	}
	receivedAt, err := time.Parse(time.RFC3339Nano, cmd.ReceivedAt)
	if err != nil {
		return controlState{}, true, false, false
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

func commandID(cmd *commandEnvelope) int64 {
	if cmd == nil {
		return 0
	}
	return cmd.CommandID
}

func (s *Service) stopSignTargetLocked(now time.Time) controllerSnapshot {
	trace := StopSignControlDebug{
		PlanID:    s.stopSignPlanID,
		PlanState: "active",
	}
	if s.stopSignPlan == nil || s.stopSignPlanSampler == nil {
		return s.stopSignFailSafeTargetLocked(now, trace, "stopSign setpoint plan is unavailable")
	}

	planAge := now.Sub(secondsToTime(s.stopSignPlan.SampledAtS))
	trace.PlanAgeMs = durationMs(planAge)
	if planAge < 0 || planAge > s.cfg.StopSignPlanTimeout || now.Sub(s.stopSignPlanAcceptedAt) > s.cfg.StopSignPlanTimeout {
		return s.stopSignFailSafeTargetLocked(now, trace, fmt.Sprintf("stopSign setpoint plan is stale: age=%s", planAge))
	}

	egoState, telemetryAt := s.latestActuatorEgoStateSnapshotLocked()
	if egoState == nil {
		return s.stopSignFailSafeTargetLocked(now, trace, "actuator ego telemetry is unavailable")
	}
	trace.CurrentSpeedMPS = egoState.SpeedMPS
	trace.MeasuredWheelSteer = cloneFloatPtr(egoState.SteeringActual)
	receiptAge, sourceAge, receiptSkew, err := s.validateStopSignTelemetryTimingLocked(now, *egoState, telemetryAt)
	trace.TelemetryAgeMs = durationMs(receiptAge)
	trace.SourceTelemetryAgeMs = durationMs(sourceAge)
	trace.SourceReceiptSkewMs = durationMs(receiptSkew)
	if err != nil {
		return s.stopSignFailSafeTargetLocked(now, trace, err.Error())
	}
	if err := s.validateStopSignEgoStateLocked(*egoState); err != nil {
		return s.stopSignFailSafeTargetLocked(now, trace, err.Error())
	}

	dt := time.Second / time.Duration(s.cfg.TickHz)
	if !s.lastStopSignTickAt.IsZero() {
		dt = now.Sub(s.lastStopSignTickAt)
	}
	s.lastStopSignTickAt = now
	if dt <= 0 || dt > s.cfg.StopSignController.MaxDT {
		return s.stopSignFailSafeTargetLocked(now, trace, fmt.Sprintf("stopSign controller dt %s is outside (0, %s]", dt, s.cfg.StopSignController.MaxDT))
	}

	targetAgeMs := durationMs(planAge + s.cfg.StopSignEstimatedActuationLatency)
	trace.TargetDtMs = targetAgeMs
	firstDt := float64(s.stopSignPlan.Points[0].DtMs)
	lastDt := float64(s.stopSignPlan.Points[len(s.stopSignPlan.Points)-1].DtMs)
	trace.HorizonClamped = targetAgeMs < firstDt || targetAgeMs > lastDt
	setpoint, err := s.stopSignPlanSampler.SampleByAge(targetAgeMs)
	if err != nil {
		return s.stopSignFailSafeTargetLocked(now, trace, err.Error())
	}
	setpointCopy := setpoint
	trace.SelectedSetpoint = &setpointCopy
	output, err := s.stopSignController.Step(stopsigncontrol.Input{
		DesiredWheelSteer:  setpoint.DesiredWheelSteerNormalized,
		DesiredSpeedMPS:    setpoint.DesiredSpeedMPS,
		StopProbability:    setpoint.StopProbability,
		MeasuredWheelSteer: *egoState.SteeringActual,
		CurrentSpeedMPS:    egoState.SpeedMPS,
		DT:                 dt,
	})
	trace.ControllerOutput = output
	if err != nil {
		return s.stopSignFailSafeTargetLocked(now, trace, err.Error())
	}
	throttle, brake := output.ThrottleBrake()
	trace.RequestedThrottle = throttle
	trace.RequestedBrake = brake
	target := controlState{
		Steer:     output.Steering,
		Throttle:  throttle,
		Brake:     brake,
		Handbrake: output.Hold,
	}
	var safety SafetyDebug
	target, safety = s.applyStopSignSafetyLocked(target, egoState.SpeedMPS)
	trace.AppliedSteer = target.Steer
	trace.AppliedThrottle = target.Throttle
	trace.AppliedBrake = target.Brake
	trace.AppliedHandbrake = target.Handbrake
	traceCopy := trace
	return controllerSnapshot{
		controlState: target,
		CommandID:    commandID(s.lastCmd),
		PlanID:       s.stopSignPlanID,
		Enabled:      true,
		Stale:        false,
		Holding:      target.Handbrake,
		TimedOut:     false,
		UpdatedAt:    now.Format(time.RFC3339Nano),
		Safety:       safety,
		StopSign:     &traceCopy,
	}
}

func (s *Service) validateStopSignEgoStateLocked(ego control.ActuatorEgoState) error {
	if !ego.Valid {
		reason := strings.TrimSpace(ego.InvalidReason)
		if reason == "" {
			reason = "unknown validation failure"
		}
		return fmt.Errorf("actuator ego telemetry is invalid: %s", reason)
	}
	if ego.SteeringActual == nil || math.IsNaN(*ego.SteeringActual) || math.IsInf(*ego.SteeringActual, 0) || *ego.SteeringActual < -1 || *ego.SteeringActual > 1 {
		return errors.New("measured wheel steering is unavailable or outside [-1, 1]")
	}
	if math.IsNaN(ego.SpeedMPS) || math.IsInf(ego.SpeedMPS, 0) || ego.SpeedMPS < 0 {
		return errors.New("measured speed is invalid")
	}
	if ego.VehicleModelHash != s.cfg.StopSignCalibration.VehicleModelHash {
		return fmt.Errorf("vehicle model hash %d does not match calibrated hash %d", ego.VehicleModelHash, s.cfg.StopSignCalibration.VehicleModelHash)
	}
	return nil
}

func (s *Service) validateStopSignTelemetryTimingLocked(now time.Time, ego control.ActuatorEgoState, receivedAt time.Time) (time.Duration, time.Duration, time.Duration, error) {
	receiptAge := now.Sub(receivedAt)
	if receivedAt.IsZero() || receiptAge < 0 || receiptAge > s.cfg.StopSignTelemetryTimeout {
		return receiptAge, 0, 0, fmt.Errorf("actuator ego telemetry receipt is stale: age=%s", receiptAge)
	}
	if ego.TimestampS <= 0 || math.IsNaN(ego.TimestampS) || math.IsInf(ego.TimestampS, 0) {
		return receiptAge, 0, 0, errors.New("actuator ego telemetry source timestamp is unavailable")
	}
	sourceAt := secondsToTime(ego.TimestampS)
	sourceAge := now.Sub(sourceAt)
	receiptSkew := receivedAt.Sub(sourceAt)
	if sourceAge < -maxTelemetryFutureSkew || sourceAge > s.cfg.StopSignTelemetryTimeout {
		return receiptAge, sourceAge, receiptSkew, fmt.Errorf("actuator ego telemetry source is stale: age=%s", sourceAge)
	}
	if receiptSkew < -maxTelemetryFutureSkew || receiptSkew > s.cfg.StopSignTelemetryTimeout {
		return receiptAge, sourceAge, receiptSkew, fmt.Errorf("actuator ego telemetry source/receipt skew is invalid: skew=%s", receiptSkew)
	}
	return receiptAge, sourceAge, receiptSkew, nil
}

func (s *Service) stopSignFailSafeTargetLocked(now time.Time, trace StopSignControlDebug, fault string) controllerSnapshot {
	s.stopSignController.Reset()
	s.lastStopSignTickAt = time.Time{}
	s.lastStopSignFault = strings.TrimSpace(fault)
	trace.PlanState = "fault"
	trace.FailSafe = true
	trace.Fault = s.lastStopSignFault
	return s.stopSignStopSnapshotLocked(now, trace, true)
}

func (s *Service) stopSignStopTargetLocked(now time.Time, planState string) controllerSnapshot {
	trace := StopSignControlDebug{
		PlanID:    s.stopSignPlanID,
		PlanState: planState,
	}
	return s.stopSignStopSnapshotLocked(now, trace, false)
}

func (s *Service) stopSignStopSnapshotLocked(now time.Time, trace StopSignControlDebug, failed bool) controllerSnapshot {
	ego, receivedAt := s.latestActuatorEgoStateSnapshotLocked()
	speed := 0.0
	speedKnown := false
	if ego != nil {
		receiptAge, sourceAge, receiptSkew, timingErr := s.validateStopSignTelemetryTimingLocked(now, *ego, receivedAt)
		trace.TelemetryAgeMs = durationMs(receiptAge)
		trace.SourceTelemetryAgeMs = durationMs(sourceAge)
		trace.SourceReceiptSkewMs = durationMs(receiptSkew)
		if timingErr == nil && ego.Valid && !math.IsNaN(ego.SpeedMPS) && !math.IsInf(ego.SpeedMPS, 0) && ego.SpeedMPS >= 0 {
			speed = ego.SpeedMPS
			speedKnown = true
		}
	}
	target := controlState{}
	if speedKnown && speed <= s.cfg.StopSignController.HoldSpeedMPS {
		target.Handbrake = true
	} else {
		target.Brake = s.cfg.StopSignController.StopBrakeEffort
	}
	var safety SafetyDebug
	target, safety = s.applyStopSignSafetyLocked(target, speed)
	trace.CurrentSpeedMPS = speed
	trace.AppliedSteer = target.Steer
	trace.AppliedThrottle = target.Throttle
	trace.AppliedBrake = target.Brake
	trace.AppliedHandbrake = target.Handbrake
	traceCopy := trace
	planID := int64(0)
	if failed {
		planID = s.stopSignPlanID
	}
	return controllerSnapshot{
		controlState: target,
		CommandID:    commandID(s.lastCmd),
		PlanID:       planID,
		Enabled:      true,
		Stale:        failed,
		Holding:      target.Handbrake,
		TimedOut:     failed,
		UpdatedAt:    now.Format(time.RFC3339Nano),
		Safety:       safety,
		StopSign:     &traceCopy,
	}
}

func (s *Service) applyStopSignSafetyLocked(input controlState, currentSpeedMPS float64) (controlState, SafetyDebug) {
	target := controlState{
		Steer:     clamp(input.Steer, -1, 1),
		Throttle:  clamp(input.Throttle, 0, 1),
		Brake:     clamp(input.Brake, 0, 1),
		Handbrake: input.Handbrake,
	}
	debug := SafetyDebug{
		CurrentSpeedMPS:    currentSpeedMPS,
		CurrentSpeedKPH:    currentSpeedMPS * 3.6,
		SpeedLimitKPH:      s.cfg.SpeedLimitKPH,
		ThrottleBefore:     target.Throttle,
		BrakeBefore:        target.Brake,
		TelemetryAvailable: true,
	}
	if target.Handbrake {
		target.Throttle = 0
		target.Brake = 0
	}
	if target.Throttle > 0 && target.Brake > 0 {
		target.Throttle = 0
	}
	if s.cfg.SpeedLimitKPH > 0 && debug.CurrentSpeedKPH >= s.cfg.SpeedLimitKPH {
		debug.SpeedLimitActive = true
		debug.OverspeedKPH = debug.CurrentSpeedKPH - s.cfg.SpeedLimitKPH
		target.Throttle = 0
		if debug.OverspeedKPH >= s.cfg.OverspeedBrakeMarginKPH && s.cfg.OverspeedBrake > target.Brake {
			target.Brake = s.cfg.OverspeedBrake
			debug.OverspeedBrakeApplied = true
		}
	}
	debug.ThrottleAfter = target.Throttle
	debug.BrakeAfter = target.Brake
	return target, debug
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
	currentSpeedKPH := 0.0
	hasTelemetry := false
	if latest := s.latestTelemetryLocked(); latest != nil {
		currentSpeedMPS := math.Max(latest.CurrentSpeed, 0)
		currentSpeedKPH = currentSpeedMPS * 3.6
		hasTelemetry = true
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
	}
	resolved, _ := resolveServiceThrottleBrakeConflict(controlState{
		Steer:     steer,
		Throttle:  throttle,
		Brake:     brake,
		Handbrake: input.Handbrake,
	})
	if hasTelemetry && currentSpeedKPH <= tuning.ReverseLockoutSpeedKPH && resolved.Brake > 0 {
		resolved.Brake = 0
		resolved.Handbrake = true
		debug.ReverseLockoutApplied = true
		debug.LowSpeedBrakeHold = true
	}
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
	latest, _ := s.latestActuatorEgoStateSnapshotLocked()
	return latest
}

func (s *Service) latestActuatorEgoStateSnapshotLocked() (*control.ActuatorEgoState, time.Time) {
	if s.telemetry == nil {
		return nil, time.Time{}
	}
	return s.telemetry.LatestActuatorEgoStateSnapshot()
}

func (s *Service) resetStopSignControlLocked() {
	if s.stopSignController != nil {
		s.stopSignController.Reset()
	}
	s.stopSignPlan = nil
	s.stopSignPlanSampler = nil
	s.stopSignPlanAcceptedAt = time.Time{}
	s.lastStopSignTickAt = time.Time{}
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
		Supported:                   s.supported,
		Ready:                       s.ready,
		Platform:                    runtime.GOOS,
		ControllerType:              "xbox360",
		TickHz:                      s.cfg.TickHz,
		StaleTimeoutMs:              s.cfg.StaleTimeout.Milliseconds(),
		LastError:                   s.lastError,
		LastCommandID:               commandID(s.lastCmd),
		Target:                      s.target,
		Applied:                     s.applied,
		LastApplyError:              s.lastApplyError,
		LastApplyAttemptedAt:        s.lastApplyAttemptedAt,
		LastApplyAttemptedCommandID: s.lastApplyAttemptedCommandID,
		LastApplyAttemptedPlanID:    s.lastApplyAttemptedPlanID,
		LastApplySucceededAt:        s.lastApplySucceededAt,
		StopSignController: StopSignControllerState{
			Ready:                       s.stopSignController != nil && s.stopSignControllerErr == nil && s.cfg.StopSignCalibration.Verified,
			Contract:                    stopsigncontrol.StopSignMotionPlanContractV1,
			Calibration:                 s.cfg.StopSignCalibration,
			PlanTimeoutMs:               s.cfg.StopSignPlanTimeout.Milliseconds(),
			TelemetryTimeoutMs:          s.cfg.StopSignTelemetryTimeout.Milliseconds(),
			EstimatedActuationLatencyMs: s.cfg.StopSignEstimatedActuationLatency.Milliseconds(),
			ExpectedHorizonDtMs:         append([]int(nil), s.cfg.StopSignExpectedHorizonDtMs...),
			Owner:                       s.stopSignOwner,
			Stopping:                    s.stopSignStopping,
			LastPlanID:                  s.stopSignPlanID,
			LastPlanAppliedID:           s.lastStopSignAppliedID,
			LastPlanAppliedAt:           s.lastStopSignAppliedAt,
			LastFault:                   s.lastStopSignFault,
		},
	}
	if !s.stopSignPlanAcceptedAt.IsZero() {
		state.StopSignController.LastPlanAcceptedAt = s.stopSignPlanAcceptedAt.Format(time.RFC3339Nano)
	}
	if s.lastCmd != nil {
		copyCmd := *s.lastCmd
		state.LastCommand = &copyCmd
	}
	return state
}

func secondsToTime(value float64) time.Time {
	seconds, fractional := math.Modf(value)
	return time.Unix(int64(seconds), int64(fractional*float64(time.Second))).UTC()
}

func timeToSeconds(value time.Time) float64 {
	return float64(value.UTC().UnixNano()) / float64(time.Second)
}

func durationMs(value time.Duration) float64 {
	return float64(value) / float64(time.Millisecond)
}

func cloneFloatPtr(value *float64) *float64 {
	if value == nil {
		return nil
	}
	copyValue := *value
	return &copyValue
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
		Owner:       strings.ToLower(strings.TrimSpace(req.Owner)),
	}
	if cmd.Owner != "" && cmd.Owner != OwnerStopSignInference && cmd.Owner != OwnerCalibration {
		return commandEnvelope{}, fmt.Errorf("invalid actuator owner %q", req.Owner)
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
	output.Throttle = 0
	return output, true
}

func (s *Service) Unsupported() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return !s.supported
}
