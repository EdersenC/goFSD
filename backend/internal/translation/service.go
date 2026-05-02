package translation

import (
	"fmt"
	"log"
	"math"
	"strings"
	"sync"
	"time"

	"awesomeProject/internal/actuator"
)

const (
	ModeIdle     = "idle"
	ModeHold     = "hold"
	ModeLaunch   = "launch"
	ModeTrack    = "track"
	ModeBrake    = "brake"
	ModeFailsafe = "failsafe"

	MotionIntentHold = "hold"
	MotionIntentMove = "move"
	MotionIntentStop = "stop"
)

type actuatorSubmitter interface {
	Submit(req actuator.CommandRequest) (actuator.State, error)
}

type Telemetry struct {
	CurrentSpeedMPS   float64 `json:"currentSpeedMps"`
	CurrentYawDeg     float64 `json:"currentYawDeg"`
	RouteForwardDelta float64 `json:"routeForwardDelta"`
	TelemetryAgeMs    int64   `json:"telemetryAgeMs"`
}

type LateralIntent struct {
	HeadingErrorDeg float64 `json:"headingErrorDeg"`
	Confidence      float64 `json:"confidence"`
}

type LongitudinalIntent struct {
	TargetSpeedMPS  float64 `json:"targetSpeedMps"`
	TargetAccelMPS2 float64 `json:"targetAccelMps2"`
	Confidence      float64 `json:"confidence"`
}

type MotionIntent struct {
	Intent     string  `json:"intent"`
	Confidence float64 `json:"confidence"`
}

type Sample struct {
	Sequence       int                `json:"sequence"`
	TimestampMs    int64              `json:"timestampMs,omitempty"`
	CapturedAt     string             `json:"capturedAt,omitempty"`
	PredictedAt    string             `json:"predictedAt,omitempty"`
	Checkpoint     string             `json:"checkpoint,omitempty"`
	ModelDevice    string             `json:"modelDevice,omitempty"`
	ModelServerURL string             `json:"modelServerUrl,omitempty"`
	Telemetry      Telemetry          `json:"telemetry"`
	Lateral        LateralIntent      `json:"lateral"`
	Longitudinal   LongitudinalIntent `json:"longitudinal"`
	Motion         MotionIntent       `json:"motion"`
}

type Trace struct {
	Mode               string                  `json:"mode"`
	Reasons            []string                `json:"reasons,omitempty"`
	Sample             Sample                  `json:"sample"`
	MotionActive       bool                    `json:"motionActive"`
	HeadingCommand     float64                 `json:"headingCommand"`
	SteerGain          float64                 `json:"steerGain"`
	DesiredSteer       float64                 `json:"desiredSteer"`
	ShapedSteer        float64                 `json:"shapedSteer"`
	TargetSpeedMPS     float64                 `json:"targetSpeedMps"`
	SpeedErrorMPS      float64                 `json:"speedErrorMps"`
	SpeedContribution  float64                 `json:"speedContribution"`
	AccelContribution  float64                 `json:"accelContribution"`
	RawLongitudinal    float64                 `json:"rawLongitudinal"`
	HeldThrottle       float64                 `json:"heldThrottle"`
	BrakeLatched       bool                    `json:"brakeLatched"`
	BrakePending       bool                    `json:"brakePending"`
	SignedLongitudinal float64                 `json:"signedLongitudinal"`
	Command            actuator.CommandRequest `json:"command"`
	UpdatedAt          string                  `json:"updatedAt"`
}

type TuningState struct {
	Live          Tuning `json:"live"`
	Saved         Tuning `json:"saved"`
	ConfigPath    string `json:"configPath,omitempty"`
	SaveSupported bool   `json:"saveSupported"`
}

type State struct {
	Mode                   string                   `json:"mode"`
	LastError              string                   `json:"lastError,omitempty"`
	LastActuatorError      string                   `json:"lastActuatorError,omitempty"`
	LastActuatorAcceptedAt string                   `json:"lastActuatorAcceptedAt,omitempty"`
	LastSample             *Sample                  `json:"lastSample,omitempty"`
	LastTrace              *Trace                   `json:"lastTrace,omitempty"`
	LastCommand            *actuator.CommandRequest `json:"lastCommand,omitempty"`
	Tuning                 TuningState              `json:"tuning"`
}

type Service struct {
	actuator   actuatorSubmitter
	configPath string
	nowFunc    func() time.Time

	mu                     sync.Mutex
	liveTuning             Tuning
	savedTuning            Tuning
	lastSample             *Sample
	lastTrace              *Trace
	lastCommand            *actuator.CommandRequest
	lastError              string
	lastActuatorError      string
	lastActuatorAcceptedAt string
	lastSteerCommand       float64
	hasLastSteerCommand    bool
	lastDriveCommand       float64
	lastThrottleOutput     float64
	lastLongitudinalAt     time.Time
	brakeLatched           bool
	brakeRequestedAt       time.Time
	motionActive           bool
	motionLatched          bool
	mode                   string
	throttleHoldUntil      time.Time
	throttleHoldValue      float64
}

func NewService(cfg Config, configPath string, actuatorSink actuatorSubmitter) *Service {
	tuning := cfg.Tuning()
	return &Service{
		actuator:    actuatorSink,
		configPath:  configPath,
		nowFunc:     time.Now,
		liveTuning:  tuning,
		savedTuning: tuning,
		mode:        ModeIdle,
	}
}

func (s *Service) Submit(sample Sample) (State, error) {
	now := s.nowFunc().UTC()
	trace, err := s.translate(sample, now)
	if err != nil {
		log.Printf("[translation] translate failed seq=%d motion=%s speed=%.3f err=%v",
			sample.Sequence,
			sample.Motion.Intent,
			sample.Telemetry.CurrentSpeedMPS,
			err,
		)
		s.mu.Lock()
		s.lastError = err.Error()
		s.lastActuatorError = ""
		s.mode = ModeFailsafe
		s.lastSample = cloneSamplePtr(&sample)
		s.lastTrace = nil
		s.lastCommand = nil
		s.mu.Unlock()
		return s.State(), err
	}

	_, actuatorErr := s.actuator.Submit(trace.Command)
	if actuatorErr != nil {
		log.Printf("[translation] actuator submit failed seq=%d mode=%s steer=%.3f throttle=%.3f err=%v",
			sample.Sequence,
			trace.Mode,
			trace.Command.Steer,
			trace.Command.Throttle,
			actuatorErr,
		)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	s.mode = trace.Mode
	s.lastError = ""
	if actuatorErr != nil {
		s.lastActuatorError = actuatorErr.Error()
	} else {
		s.lastActuatorError = ""
		s.lastActuatorAcceptedAt = now.Format(time.RFC3339Nano)
	}
	s.lastSample = cloneSamplePtr(&sample)
	s.lastTrace = &trace
	cmdCopy := trace.Command
	s.lastCommand = &cmdCopy

	if actuatorErr != nil {
		return s.stateLocked(), actuatorErr
	}
	return s.stateLocked(), nil
}

func (s *Service) State() State {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.stateLocked()
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
		return TuningState{}, fmt.Errorf("translation config path is not available")
	}
	if err := SaveTuning(path, tuning); err != nil {
		return TuningState{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.savedTuning = tuning
	return s.tuningStateLocked(), nil
}

func (s *Service) translate(sample Sample, now time.Time) (Trace, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if err := validateSample(sample); err != nil {
		return Trace{}, err
	}

	reasons := make([]string, 0, 6)
	tuning := s.liveTuning
	motionIntent := strings.ToLower(strings.TrimSpace(sample.Motion.Intent))
	if motionIntent == "" {
		motionIntent = MotionIntentHold
	}
	motionConfidence := clamp(sample.Motion.Confidence, 0, 1)

	switch {
	case !s.motionLatched:
		s.motionActive = motionIntent == MotionIntentMove && motionConfidence >= tuning.MotionConfidenceOnThreshold
		s.motionLatched = true
	case motionIntent == MotionIntentStop:
		s.motionActive = false
	case motionIntent == MotionIntentMove && motionConfidence >= tuning.MotionConfidenceOnThreshold:
		s.motionActive = true
	case motionConfidence <= tuning.MotionConfidenceOffThreshold:
		s.motionActive = false
	}
	motionActive := s.motionActive

	currentSpeed := math.Max(sample.Telemetry.CurrentSpeedMPS, 0)
	targetSpeed := clamp(sample.Longitudinal.TargetSpeedMPS, 0, maxTargetSpeedMPS(tuning))
	speedError := targetSpeed - currentSpeed
	speedContribution := speedError * tuning.TargetSpeedErrorGain
	accelContribution := sample.Longitudinal.TargetAccelMPS2 * tuning.TargetAccelGain
	rawLongitudinal := clamp(speedContribution+accelContribution, -1, 1)
	if math.Abs(rawLongitudinal) < tuning.LongitudinalDeadband {
		rawLongitudinal = 0
	}

	if sample.Longitudinal.Confidence < tuning.MinLongitudinalConfidence {
		rawLongitudinal = 0
		reasons = append(reasons, "longitudinal_confidence_low")
	}
	if rawLongitudinal > 0 && currentSpeed >= maxTargetSpeedMPS(tuning) {
		rawLongitudinal = 0
		reasons = append(reasons, "max_target_speed_gate")
	}

	explicitStop := motionIntent == MotionIntentStop
	deltaSeconds := s.longitudinalDeltaSeconds(now)
	rawLongitudinalDemand := math.Max(rawLongitudinal, 0)
	driveHoldEligible := motionActive &&
		motionIntent != MotionIntentStop &&
		speedError >= -tuning.LongitudinalDeadband
	heldThrottle, throttleHeld := s.stabilizeDriveCommand(
		rawLongitudinalDemand,
		speedError,
		driveHoldEligible,
		now,
		tuning,
	)

	s.brakeLatched = false
	s.brakeRequestedAt = time.Time{}
	brakePending := false
	brakeLatched := false
	signedLongitudinal := rawLongitudinal
	switch {
	case explicitStop:
		heldThrottle = 0
		signedLongitudinal = 0
		reasons = append(reasons, "explicit_stop_coast")
	case rawLongitudinal < 0:
		if heldThrottle > 0 {
			signedLongitudinal = heldThrottle
		} else {
			signedLongitudinal = 0
			reasons = append(reasons, "coast_instead_of_brake")
		}
	default:
		signedLongitudinal = heldThrottle
	}
	if throttleHeld {
		reasons = append(reasons, "throttle_hold")
	}

	mode := ModeTrack
	switch {
	case explicitStop:
		mode = ModeHold
	case motionIntent == MotionIntentHold &&
		math.Abs(speedError) < tuning.LongitudinalDeadband &&
		math.Abs(sample.Longitudinal.TargetAccelMPS2) < tuning.LongitudinalDeadband:
		mode = ModeHold
	case currentSpeed <= tuning.LaunchSpeedThreshold && targetSpeed >= currentSpeed+tuning.LaunchTargetSpeedMargin:
		mode = ModeLaunch
	default:
		mode = ModeTrack
	}

	if mode == ModeLaunch && signedLongitudinal > 0 {
		signedLongitudinal = math.Max(signedLongitudinal, tuning.LaunchThrottleMin)
		heldThrottle = math.Max(heldThrottle, signedLongitudinal)
		reasons = append(reasons, "launch_floor")
	}
	if mode == ModeHold {
		heldThrottle = 0
		signedLongitudinal = 0
		brakePending = false
	}

	headingError := sample.Lateral.HeadingErrorDeg
	headingCommand := 0.0
	steerGain := 0.0
	if sample.Lateral.Confidence < tuning.MinLateralConfidence {
		reasons = append(reasons, "lateral_confidence_low")
	} else if math.Abs(headingError) >= tuning.HeadingDeadbandDeg {
		normalizedHeading := clamp(headingError/tuning.HeadingFullLockDeg, -1, 1)
		headingCommand = math.Copysign(math.Sqrt(math.Abs(normalizedHeading)), normalizedHeading)
		speedRatio := clamp(currentSpeed/tuning.SteerGainFadeSpeedMPS, 0, 1)
		steerGain = lerp(tuning.LowSpeedSteerGain, tuning.HighSpeedSteerGain, speedRatio)
	}
	desiredSteer := clamp(headingCommand*steerGain*tuning.SteerOutputGain, -1, 1)
	if mode == ModeHold || mode == ModeFailsafe {
		desiredSteer = 0
	}
	shapedSteer := s.shapeSteer(desiredSteer)
	if math.Abs(shapedSteer) < tuning.SteerDeadzone {
		shapedSteer = 0
	}
	shapedSteer = clamp(shapedSteer*tuning.MaxSteerScale, -1, 1)

	desiredThrottle := clamp(math.Max(signedLongitudinal, 0)*tuning.ThrottleGain, 0, 1)
	throttle := desiredThrottle
	throttleRamped := false
	if mode == ModeHold || mode == ModeFailsafe {
		throttle = 0
		s.lastThrottleOutput = 0
	} else {
		throttle, throttleRamped = s.smoothThrottleOutput(desiredThrottle, deltaSeconds, tuning)
	}
	if throttleRamped {
		reasons = append(reasons, "throttle_ramp")
	}
	s.lastLongitudinalAt = now
	s.lastDriveCommand = clamp(rawLongitudinalDemand, 0, 1)
	s.lastThrottleOutput = throttle

	command := actuator.CommandRequest{
		Steer:       shapedSteer,
		Throttle:    throttle,
		InputMode:   actuator.InputModeNormalized,
		Handbrake:   false,
		Enabled:     boolPtr(mode != ModeFailsafe),
		Sequence:    int64(sample.Sequence),
		TimestampMs: now.UnixMilli(),
	}

	trace := Trace{
		Mode:               mode,
		Reasons:            reasons,
		Sample:             sample,
		MotionActive:       motionActive,
		HeadingCommand:     headingCommand,
		SteerGain:          steerGain,
		DesiredSteer:       desiredSteer,
		ShapedSteer:        shapedSteer,
		TargetSpeedMPS:     targetSpeed,
		SpeedErrorMPS:      speedError,
		SpeedContribution:  speedContribution,
		AccelContribution:  accelContribution,
		RawLongitudinal:    rawLongitudinal,
		HeldThrottle:       heldThrottle,
		BrakeLatched:       brakeLatched,
		BrakePending:       brakePending,
		SignedLongitudinal: signedLongitudinal,
		Command:            command,
		UpdatedAt:          now.Format(time.RFC3339Nano),
	}
	return trace, nil
}

func (s *Service) longitudinalDeltaSeconds(now time.Time) float64 {
	if s.lastLongitudinalAt.IsZero() {
		return 1.0 / 15.0
	}
	delta := now.Sub(s.lastLongitudinalAt).Seconds()
	if delta <= 0 {
		return 1.0 / 60.0
	}
	return clamp(delta, 1.0/60.0, 1.0)
}

func (s *Service) stabilizeDriveCommand(rawDrive float64, speedError float64, eligible bool, now time.Time, tuning Tuning) (float64, bool) {
	rawDrive = clamp(rawDrive, 0, 1)
	if !eligible {
		s.throttleHoldUntil = time.Time{}
		s.throttleHoldValue = rawDrive
		return rawDrive, false
	}

	if rawDrive <= 0 {
		s.throttleHoldUntil = time.Time{}
		s.throttleHoldValue = 0
		return 0, false
	}

	holdWindow := time.Duration(tuning.ThrottleHoldSeconds * float64(time.Second))
	if !s.throttleHoldUntil.IsZero() && now.Before(s.throttleHoldUntil) {
		if rawDrive >= s.throttleHoldValue {
			s.throttleHoldUntil = time.Time{}
			s.throttleHoldValue = rawDrive
		} else {
			return clamp(s.throttleHoldValue, 0, 1), true
		}
	} else {
		s.throttleHoldUntil = time.Time{}
	}

	lastDemand := clamp(s.lastDriveCommand, 0, 1)
	if lastDemand > rawDrive && holdWindow > 0 {
		s.throttleHoldValue = lastDemand
		s.throttleHoldUntil = now.Add(holdWindow)
		return s.throttleHoldValue, true
	}

	stabilized := rawDrive
	if speedError > tuning.LongitudinalDeadband && stabilized > 0 && stabilized < tuning.ThrottleHoldMin {
		stabilized = tuning.ThrottleHoldMin
	}

	return clamp(stabilized, 0, 1), false
}

func (s *Service) smoothThrottleOutput(desired float64, deltaSeconds float64, tuning Tuning) (float64, bool) {
	desired = clamp(desired, 0, 1)
	current := clamp(s.lastThrottleOutput, 0, 1)
	maxDelta := tuning.ThrottleDecayPerSecond * deltaSeconds
	if desired > current {
		maxDelta = tuning.ThrottleRampUpPerSecond * deltaSeconds
	}
	next := moveTowards(current, desired, maxDelta)
	return next, math.Abs(next-desired) > 1e-6
}

func (s *Service) shapeSteer(desired float64) float64 {
	desired = clamp(desired, -1, 1)
	if !s.hasLastSteerCommand || s.liveTuning.SteerCommandRatePerSecond <= 0 {
		s.lastSteerCommand = desired
		s.hasLastSteerCommand = true
		return desired
	}
	blended := s.lastSteerCommand + (desired-s.lastSteerCommand)*s.liveTuning.SteerResponseBlend
	maxDelta := s.liveTuning.SteerCommandRatePerSecond / 15.0
	next := moveTowards(s.lastSteerCommand, blended, maxDelta)
	s.lastSteerCommand = next
	return next
}

func (s *Service) stateLocked() State {
	state := State{
		Mode:                   s.mode,
		LastError:              s.lastError,
		LastActuatorError:      s.lastActuatorError,
		LastActuatorAcceptedAt: s.lastActuatorAcceptedAt,
		Tuning:                 s.tuningStateLocked(),
	}
	if s.lastSample != nil {
		state.LastSample = cloneSamplePtr(s.lastSample)
	}
	if s.lastTrace != nil {
		trace := *s.lastTrace
		trace.Sample = cloneSample(trace.Sample)
		state.LastTrace = &trace
	}
	if s.lastCommand != nil {
		cmd := *s.lastCommand
		state.LastCommand = &cmd
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

func validateSample(sample Sample) error {
	for _, value := range []float64{
		sample.Telemetry.CurrentSpeedMPS,
		sample.Telemetry.CurrentYawDeg,
		sample.Telemetry.RouteForwardDelta,
		sample.Lateral.HeadingErrorDeg,
		sample.Lateral.Confidence,
		sample.Longitudinal.TargetSpeedMPS,
		sample.Longitudinal.TargetAccelMPS2,
		sample.Longitudinal.Confidence,
		sample.Motion.Confidence,
	} {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return fmt.Errorf("translation sample contains non-finite values")
		}
	}
	return nil
}

func maxTargetSpeedMPS(tuning Tuning) float64 {
	return math.Max(tuning.MaxTargetSpeedKPH, 0) / 3.6
}

func lerp(start float64, end float64, ratio float64) float64 {
	return start + (end-start)*ratio
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

func boolPtr(value bool) *bool {
	return &value
}

func cloneSamplePtr(sample *Sample) *Sample {
	if sample == nil {
		return nil
	}
	cloned := cloneSample(*sample)
	return &cloned
}

func cloneSample(sample Sample) Sample {
	return sample
}
