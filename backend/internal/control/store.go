package control

import (
	"errors"
	"fmt"
	"log"
	"math"
	"strings"
	"sync"
	"time"

	"awesomeProject/internal/stopsignbatch"
)

var (
	ErrInvalidCommand            = errors.New("invalid control command")
	ErrSafetyEpochRequired       = errors.New("control safety epoch is required")
	ErrSafetyEpochMismatch       = errors.New("control safety epoch is stale")
	ErrDispatchCommandIDRequired = errors.New("dispatch confirmation command id is required")
)

type CommandType string

const (
	CommandStartScene                 CommandType = "startScene"
	CommandRunAllScenes               CommandType = "runAllScenes"
	CommandEndScene                   CommandType = "endScene"
	CommandEndAllScenes               CommandType = "endAllScenes"
	CommandStartEgo                   CommandType = "startEgo"
	CommandStopEgo                    CommandType = "stopEgo"
	CommandStartStopSignBatch         CommandType = "startStopSignBatch"
	CommandSetStopSignTarget          CommandType = "setStopSignTarget"
	CommandClearStopSignTarget        CommandType = "clearStopSignTarget"
	CommandSetStopSignCatalogWaypoint CommandType = "setStopSignCatalogWaypoint"
)

const (
	StopSignPhaseAccelerate     = "accelerate"
	StopSignPhaseCruiseApproach = "cruise_approach"
	StopSignPhaseDecelerate     = "decelerate"
	StopSignPhaseStopHold       = "stop_hold"
	StopSignPhaseRelease        = "release"
)

type RuntimeStatus string

const (
	StatusIdle             RuntimeStatus = "idle"
	StatusRunningScene     RuntimeStatus = "runningScene"
	StatusRunningAllScenes RuntimeStatus = "runningAllScenes"
	StatusStopping         RuntimeStatus = "stopping"
	StatusError            RuntimeStatus = "error"
)

type Command struct {
	ID                      string             `json:"id"`
	Type                    CommandType        `json:"type"`
	SafetyEpoch             uint64             `json:"safetyEpoch"`
	SceneName               string             `json:"sceneName,omitempty"`
	PlanFingerprint         string             `json:"planFingerprint,omitempty"`
	StopSignBatchID         string             `json:"stopSignBatchId,omitempty"`
	StopSignJobs            []StopSignBatchJob `json:"stopSignJobs,omitempty"`
	StopSignCatalogPosition *WorldPosition     `json:"stopSignCatalogPosition,omitempty"`
	CreatedAt               string             `json:"createdAt"`
}

type CommandRequest struct {
	Type CommandType `json:"type"`
	// SafetyEpoch is a required optimistic precondition for commands that can begin motion.
	SafetyEpoch             *uint64            `json:"safetyEpoch,omitempty"`
	SceneName               string             `json:"sceneName,omitempty"`
	PlanFingerprint         string             `json:"planFingerprint,omitempty"`
	StopSignBatchID         string             `json:"stopSignBatchId,omitempty"`
	StopSignJobs            []StopSignBatchJob `json:"stopSignJobs,omitempty"`
	StopSignCatalogPosition *WorldPosition     `json:"stopSignCatalogPosition,omitempty"`
}

// WorldPosition identifies a GTA map location without pretending that a
// roadside prop position is a calibrated lane pose.
type WorldPosition struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
	Z float64 `json:"z"`
}

type StopSignPose = stopsignbatch.Pose
type StopSignTime = stopsignbatch.TimeOfDay
type StopSignColor = stopsignbatch.RGBColor
type StopSignVehicle = stopsignbatch.VehicleVariant
type StopSignBatchJob = stopsignbatch.Job

type StatusUpdate struct {
	Status               RuntimeStatus          `json:"status"`
	ActiveSceneName      string                 `json:"activeSceneName,omitempty"`
	LastError            string                 `json:"lastError,omitempty"`
	StopSignBatch        *StopSignBatchProgress `json:"stopSignBatch,omitempty"`
	AppliedSafetyEpoch   uint64                 `json:"appliedSafetyEpoch"`
	InFlightSafetyStarts int                    `json:"inFlightSafetyStarts"`
	SafetyStatusSequence uint64                 `json:"safetyStatusSequence,omitempty"`
}

type StopSignBatchProgress struct {
	BatchID         string `json:"batchId"`
	PlanFingerprint string `json:"planFingerprint,omitempty"`
	State           string `json:"state"`
	JobID           string `json:"jobId,omitempty"`
	JobIndex        int    `json:"jobIndex"`
	JobCount        int    `json:"jobCount"`
	CompletedJobs   int    `json:"completedJobs"`
	AttemptIndex    int    `json:"attemptIndex"`
	AttemptCount    int    `json:"attemptCount"`
	Phase           string `json:"phase,omitempty"`
	StartedAtMs     int64  `json:"startedAtMs,omitempty"`
	UpdatedAtMs     int64  `json:"updatedAtMs,omitempty"`
}

type TelemetryUpdate struct {
	CurrentSpeed                  float64       `json:"currentSpeed"`
	CurrentYaw                    float64       `json:"currentYaw"`
	YawRate                       float64       `json:"yawRate"`
	Steering                      float64       `json:"steering"`
	Acceleration                  float64       `json:"acceleration"`
	BrakePressureAvg              float64       `json:"brakePressureAvg"`
	VehicleExists                 bool          `json:"vehicleExists"`
	IsInVehicle                   bool          `json:"isInVehicle"`
	VehicleModelHash              int64         `json:"vehicleModelHash"`
	PositionX                     *float64      `json:"positionX,omitempty"`
	PositionY                     *float64      `json:"positionY,omitempty"`
	PositionZ                     *float64      `json:"positionZ,omitempty"`
	VelocityX                     *float64      `json:"velocityX,omitempty"`
	VelocityY                     *float64      `json:"velocityY,omitempty"`
	VelocityZ                     *float64      `json:"velocityZ,omitempty"`
	PitchDeg                      *float64      `json:"pitchDeg,omitempty"`
	RollDeg                       *float64      `json:"rollDeg,omitempty"`
	SteeringApplied               *float64      `json:"steeringApplied,omitempty"`
	ThrottleApplied               *float64      `json:"throttleApplied,omitempty"`
	BrakeApplied                  *float64      `json:"brakeApplied,omitempty"`
	Gear                          *int          `json:"gear,omitempty"`
	RPM                           *float64      `json:"rpm,omitempty"`
	WheelAngle                    *float64      `json:"wheelAngle,omitempty"`
	WheelSteeringFullLock         *float64      `json:"wheelSteeringFullLock,omitempty"`
	OnGround                      *bool         `json:"onGround,omitempty"`
	CollisionState                string        `json:"collisionState,omitempty"`
	RouteDirectionCode            float64       `json:"routeDirectionCode"`
	RouteDirectionDistanceM       float64       `json:"routeDirectionDistanceM"`
	RouteDirectionUnknown         float64       `json:"routeDirectionUnknown"`
	RouteDirectionKeepStraight    float64       `json:"routeDirectionKeepStraight"`
	RouteDirectionTurnLeft        float64       `json:"routeDirectionTurnLeft"`
	RouteDirectionTurnRight       float64       `json:"routeDirectionTurnRight"`
	RouteDirectionRerouteWrongWay float64       `json:"routeDirectionRerouteWrongWay"`
	RouteForwardDelta             float64       `json:"routeForwardDelta"`
	RouteHeadingError             float64       `json:"routeHeadingError"`
	RouteDistance                 float64       `json:"routeDistance"`
	LeadVehicleDistance           float64       `json:"leadVehicleDistance"`
	HasLeadVehicle                bool          `json:"hasLeadVehicle"`
	StopSignTargetConfigured      bool          `json:"stopSignTargetConfigured"`
	StopSignPose                  *StopSignPose `json:"stopSignPose,omitempty"`
	StopLinePose                  *StopSignPose `json:"stopLinePose,omitempty"`
	StopSignEgoStopPose           *StopSignPose `json:"stopSignEgoStopPose,omitempty"`
	StopSignDistanceM             float64       `json:"stopSignDistanceM"`
	StopLineDistanceM             float64       `json:"stopLineDistanceM"`
	StopSignLongitudinalErrorM    float64       `json:"stopSignLongitudinalErrorM"`
	StopSignLateralErrorM         float64       `json:"stopSignLateralErrorM"`
	StopSignHeadingErrorDeg       float64       `json:"stopSignHeadingErrorDeg"`
	StopSignDwellElapsedMS        int64         `json:"stopSignDwellElapsedMs"`
	StopSignDwellTargetMS         int64         `json:"stopSignDwellTargetMs"`
	StopSignStopped               bool          `json:"stopSignStopped"`
	StopSignAttemptIndex          int           `json:"stopSignAttemptIndex"`
	StopSignAttemptCount          int           `json:"stopSignAttemptCount"`
	StopSignPhase                 string        `json:"stopSignPhase"`
	TimestampMs                   int64         `json:"timestampMs,omitempty"`
	GameTimeMs                    int64         `json:"gameTimeMs,omitempty"`
}

type RuntimeTelemetry struct {
	CurrentSpeed                  float64       `json:"currentSpeed"`
	CurrentYaw                    float64       `json:"currentYaw"`
	YawRate                       float64       `json:"yawRate"`
	Steering                      float64       `json:"steering"`
	Acceleration                  float64       `json:"acceleration"`
	BrakePressureAvg              float64       `json:"brakePressureAvg"`
	VehicleExists                 bool          `json:"vehicleExists"`
	IsInVehicle                   bool          `json:"isInVehicle"`
	VehicleModelHash              int64         `json:"vehicleModelHash"`
	PositionX                     *float64      `json:"positionX,omitempty"`
	PositionY                     *float64      `json:"positionY,omitempty"`
	PositionZ                     *float64      `json:"positionZ,omitempty"`
	VelocityX                     *float64      `json:"velocityX,omitempty"`
	VelocityY                     *float64      `json:"velocityY,omitempty"`
	VelocityZ                     *float64      `json:"velocityZ,omitempty"`
	PitchDeg                      *float64      `json:"pitchDeg,omitempty"`
	RollDeg                       *float64      `json:"rollDeg,omitempty"`
	SteeringApplied               *float64      `json:"steeringApplied,omitempty"`
	ThrottleApplied               *float64      `json:"throttleApplied,omitempty"`
	BrakeApplied                  *float64      `json:"brakeApplied,omitempty"`
	Gear                          *int          `json:"gear,omitempty"`
	RPM                           *float64      `json:"rpm,omitempty"`
	WheelAngle                    *float64      `json:"wheelAngle,omitempty"`
	WheelSteeringFullLock         *float64      `json:"wheelSteeringFullLock,omitempty"`
	OnGround                      *bool         `json:"onGround,omitempty"`
	CollisionState                string        `json:"collisionState,omitempty"`
	RouteDirectionCode            float64       `json:"routeDirectionCode"`
	RouteDirectionDistanceM       float64       `json:"routeDirectionDistanceM"`
	RouteDirectionUnknown         float64       `json:"routeDirectionUnknown"`
	RouteDirectionKeepStraight    float64       `json:"routeDirectionKeepStraight"`
	RouteDirectionTurnLeft        float64       `json:"routeDirectionTurnLeft"`
	RouteDirectionTurnRight       float64       `json:"routeDirectionTurnRight"`
	RouteDirectionRerouteWrongWay float64       `json:"routeDirectionRerouteWrongWay"`
	RouteForwardDelta             float64       `json:"routeForwardDelta"`
	RouteHeadingError             float64       `json:"routeHeadingError"`
	RouteDistance                 float64       `json:"routeDistance"`
	LeadVehicleDistance           float64       `json:"leadVehicleDistance"`
	HasLeadVehicle                bool          `json:"hasLeadVehicle"`
	StopSignTargetConfigured      bool          `json:"stopSignTargetConfigured"`
	StopSignPose                  *StopSignPose `json:"stopSignPose,omitempty"`
	StopLinePose                  *StopSignPose `json:"stopLinePose,omitempty"`
	StopSignEgoStopPose           *StopSignPose `json:"stopSignEgoStopPose,omitempty"`
	StopSignDistanceM             float64       `json:"stopSignDistanceM"`
	StopLineDistanceM             float64       `json:"stopLineDistanceM"`
	StopSignLongitudinalErrorM    float64       `json:"stopSignLongitudinalErrorM"`
	StopSignLateralErrorM         float64       `json:"stopSignLateralErrorM"`
	StopSignHeadingErrorDeg       float64       `json:"stopSignHeadingErrorDeg"`
	StopSignDwellElapsedMS        int64         `json:"stopSignDwellElapsedMs"`
	StopSignDwellTargetMS         int64         `json:"stopSignDwellTargetMs"`
	StopSignStopped               bool          `json:"stopSignStopped"`
	StopSignAttemptIndex          int           `json:"stopSignAttemptIndex"`
	StopSignAttemptCount          int           `json:"stopSignAttemptCount"`
	StopSignPhase                 string        `json:"stopSignPhase"`
	TimestampMs                   int64         `json:"timestampMs,omitempty"`
	GameTimeMs                    int64         `json:"gameTimeMs,omitempty"`
	ReceivedAtMs                  int64         `json:"receivedAtMs,omitempty"`
	UpdatedAt                     string        `json:"updatedAt,omitempty"`
}

type SceneOption struct {
	Name  string `json:"name"`
	Label string `json:"label"`
}

type RuntimeState struct {
	Status               RuntimeStatus          `json:"status"`
	ActiveSceneName      string                 `json:"activeSceneName,omitempty"`
	LastError            string                 `json:"lastError,omitempty"`
	UpdatedAt            string                 `json:"updatedAt,omitempty"`
	FiveMConnected       bool                   `json:"fivemConnected"`
	LastPollAt           string                 `json:"lastPollAt,omitempty"`
	StopSignBatch        *StopSignBatchProgress `json:"stopSignBatch,omitempty"`
	AppliedSafetyEpoch   uint64                 `json:"appliedSafetyEpoch"`
	InFlightSafetyStarts int                    `json:"inFlightSafetyStarts"`
}

type State struct {
	SafetyEpoch     uint64                `json:"safetyEpoch"`
	Runtime         RuntimeState          `json:"runtime"`
	Telemetry       *RuntimeTelemetry     `json:"telemetry,omitempty"`
	EgoTelemetry    *EgoTelemetrySnapshot `json:"egoTelemetry,omitempty"`
	ActuatorEgo     *ActuatorEgoState     `json:"actuatorEgo,omitempty"`
	LastCommand     *Command              `json:"lastCommand,omitempty"`
	PendingCommands []Command             `json:"pendingCommands"`
	AvailableScenes []SceneOption         `json:"availableScenes"`
}

// ConsumerSessionReset is the atomic reconnect frontier returned to a control consumer.
// The consumer must apply SafetyEpoch before it resumes polling for commands.
type ConsumerSessionReset struct {
	SessionID   string `json:"sessionId"`
	SafetyEpoch uint64 `json:"safetyEpoch"`
}

type DispatchConfirmationReason string

const (
	DispatchConfirmationMissing          DispatchConfirmationReason = "missing"
	DispatchConfirmationCanceled         DispatchConfirmationReason = "canceled"
	DispatchConfirmationStaleSafetyEpoch DispatchConfirmationReason = "staleSafetyEpoch"
)

// DispatchConfirmation is the linearization result for a command about to leave the backend.
// A confirmed response carries the canonical stored command that the consumer may dispatch.
type DispatchConfirmation struct {
	CommandID string                     `json:"commandId"`
	Confirmed bool                       `json:"confirmed"`
	Reason    DispatchConfirmationReason `json:"reason,omitempty"`
	Command   *Command                   `json:"command,omitempty"`
}

type Store struct {
	mu sync.Mutex

	nowFunc func() time.Time

	safetyEpoch       uint64
	commands          []Command
	activeSessionID   string
	lastSeenCommandID string
	// Poll acknowledgements trail delivery by one request. Priority stops must stay after this cursor.
	lastDeliveredCommandID string
	canceledCommandIDs     map[string]struct{}
	lastEnqueuedCommand    *Command
	lastPollAt             time.Time
	runtimeStatus          RuntimeStatus
	activeSceneName        string
	lastError              string
	appliedSafetyEpoch     uint64
	inFlightSafetyStarts   int
	safetyStatusSequence   uint64
	stopSignBatch          *StopSignBatchProgress
	runtimeUpdatedAt       time.Time
	telemetry              *RuntimeTelemetry
	telemetryUpdatedAt     time.Time
	egoTelemetry           *EgoTelemetrySnapshot
	actuatorEgo            *ActuatorEgoState
	telemetryAdapter       *FiveMTelemetryAdapter
	lastAppliedControls    AppliedControls
	temporalHistory        TemporalHistoryBuffer
	lastTelemetryLogAt     time.Time
	availableScenes        []SceneOption
	commandHistoryLimit    int
	commandSeq             int64
	telemetryHistory       []RuntimeTelemetry
	telemetryLimit         int
}

type Option func(*Store)

func NewStore(opts ...Option) *Store {
	s := &Store{
		nowFunc:             time.Now,
		safetyEpoch:         1,
		runtimeStatus:       StatusIdle,
		commandHistoryLimit: 32,
		telemetryLimit:      512,
		telemetryAdapter:    NewFiveMTelemetryAdapter(DefaultFiveMTelemetryAdapterConfig()),
		temporalHistory:     NewTemporalHistoryBuffer(defaultTelemetryHistoryDuration, defaultTelemetryHistoryMaxEntries),
	}

	for _, opt := range opts {
		if opt != nil {
			opt(s)
		}
	}

	return s
}

func WithNowFunc(now func() time.Time) Option {
	return func(s *Store) {
		if now != nil {
			s.nowFunc = now
		}
	}
}

func (s *Store) Enqueue(req CommandRequest) (Command, error) {
	commandType := normalizeCommandType(req.Type)
	sceneName := strings.TrimSpace(req.SceneName)
	if err := validateCommand(commandType, sceneName, req.StopSignBatchID, req.PlanFingerprint, req.StopSignJobs, req.StopSignCatalogPosition); err != nil {
		return Command{}, err
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.validateSafetyEpochLocked(commandType, req.SafetyEpoch); err != nil {
		return Command{}, err
	}

	s.commandSeq++
	now := s.nowFunc()
	if isEmergencyStopCommand(commandType) {
		s.advanceSafetyEpochLocked()
		s.cancelPendingNonEmergencyCommandsLocked()
	}

	command := Command{
		ID:                      fmt.Sprintf("cmd-%d-%d", now.UTC().UnixNano(), s.commandSeq),
		Type:                    commandType,
		SafetyEpoch:             s.safetyEpoch,
		SceneName:               sceneName,
		PlanFingerprint:         strings.TrimSpace(req.PlanFingerprint),
		StopSignBatchID:         strings.TrimSpace(req.StopSignBatchID),
		StopSignJobs:            cloneStopSignBatchJobs(req.StopSignJobs),
		StopSignCatalogPosition: cloneWorldPosition(req.StopSignCatalogPosition),
		CreatedAt:               now.Format(time.RFC3339),
	}

	if isEmergencyStopCommand(command.Type) {
		s.insertEmergencyCommandLocked(command)
	} else {
		s.commands = append(s.commands, command)
	}
	s.lastEnqueuedCommand = cloneCommand(&command)
	s.trimCommandHistoryLocked()

	return command, nil
}

func (s *Store) Poll(lastSeenCommandID string) *Command {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.lastPollAt = s.nowFunc()
	s.lastSeenCommandID = strings.TrimSpace(lastSeenCommandID)

	nextIndex := s.nextCommandIndexLocked(s.lastSeenCommandID)
	if nextIndex < 0 {
		return nil
	}

	command := s.commands[nextIndex]
	s.lastDeliveredCommandID = command.ID
	return &command
}

// ConfirmDispatch decides atomically whether a previously polled command may be emitted.
// Emergency commands and confirmations serialize on the same store lock, so their lock
// acquisition order defines whether a guarded start may cross a safety frontier.
func (s *Store) ConfirmDispatch(commandID string) (DispatchConfirmation, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	commandID = strings.TrimSpace(commandID)
	if commandID == "" {
		return DispatchConfirmation{}, ErrDispatchCommandIDRequired
	}

	commandIndex := s.commandIndexLocked(commandID)
	if commandIndex < 0 {
		return rejectedDispatchConfirmation(commandID, DispatchConfirmationMissing), nil
	}

	command := s.commands[commandIndex]
	if command.ID != s.lastDeliveredCommandID {
		return rejectedDispatchConfirmation(commandID, DispatchConfirmationMissing), nil
	}
	if s.commandCanceledLocked(command.ID) {
		return rejectedDispatchConfirmation(commandID, DispatchConfirmationCanceled), nil
	}
	if requiresSafetyEpoch(command.Type) && command.SafetyEpoch != s.safetyEpoch {
		return rejectedDispatchConfirmation(commandID, DispatchConfirmationStaleSafetyEpoch), nil
	}

	return DispatchConfirmation{
		CommandID: command.ID,
		Confirmed: true,
		Command:   cloneCommand(&command),
	}, nil
}

func (s *Store) ResetConsumerSession() string {
	return s.ResetConsumerSessionWithSafetyEpoch().SessionID
}

func (s *Store) ResetConsumerSessionWithSafetyEpoch() ConsumerSessionReset {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.advanceSafetyEpochLocked()
	s.activeSessionID = fmt.Sprintf("session-%d-%d", s.nowFunc().UTC().UnixNano(), s.commandSeq+1)
	s.commands = nil
	s.lastSeenCommandID = ""
	s.lastDeliveredCommandID = ""
	s.canceledCommandIDs = nil
	s.lastEnqueuedCommand = nil
	s.runtimeStatus = StatusIdle
	s.activeSceneName = ""
	s.lastError = ""
	s.appliedSafetyEpoch = 0
	s.inFlightSafetyStarts = 0
	s.safetyStatusSequence = 0
	s.stopSignBatch = nil
	s.runtimeUpdatedAt = s.nowFunc()

	return ConsumerSessionReset{
		SessionID:   s.activeSessionID,
		SafetyEpoch: s.safetyEpoch,
	}
}

func (s *Store) UpdateStatus(update StatusUpdate) RuntimeState {
	status := normalizeStatus(update.Status)

	s.mu.Lock()
	defer s.mu.Unlock()
	if update.AppliedSafetyEpoch < s.appliedSafetyEpoch || update.AppliedSafetyEpoch > s.safetyEpoch || update.InFlightSafetyStarts < 0 {
		return s.runtimeStateLocked()
	}
	if update.AppliedSafetyEpoch == s.appliedSafetyEpoch && s.safetyStatusSequence > 0 &&
		(update.SafetyStatusSequence == 0 || update.SafetyStatusSequence <= s.safetyStatusSequence) {
		return s.runtimeStateLocked()
	}

	s.runtimeStatus = status
	s.activeSceneName = strings.TrimSpace(update.ActiveSceneName)
	s.lastError = strings.TrimSpace(update.LastError)
	s.appliedSafetyEpoch = update.AppliedSafetyEpoch
	s.inFlightSafetyStarts = update.InFlightSafetyStarts
	s.safetyStatusSequence = update.SafetyStatusSequence
	if update.StopSignBatch != nil {
		s.stopSignBatch = normalizedStopSignBatchProgress(update.StopSignBatch)
	} else if status == StatusIdle && s.stopSignBatch != nil && s.stopSignBatch.State == "running" {
		s.stopSignBatch = nil
	}
	s.runtimeUpdatedAt = s.nowFunc()

	return s.runtimeStateLocked()
}

func (s *Store) SetAvailableScenes(sceneOptions []SceneOption) []SceneOption {
	s.mu.Lock()
	defer s.mu.Unlock()

	normalized := make([]SceneOption, 0, len(sceneOptions))
	for _, option := range sceneOptions {
		name := strings.TrimSpace(option.Name)
		label := strings.TrimSpace(option.Label)
		if name == "" || label == "" {
			continue
		}
		normalized = append(normalized, SceneOption{
			Name:  name,
			Label: label,
		})
	}

	s.availableScenes = normalized
	return append([]SceneOption(nil), s.availableScenes...)
}

func (s *Store) State() State {
	s.mu.Lock()
	defer s.mu.Unlock()

	state := State{
		SafetyEpoch:     s.safetyEpoch,
		Runtime:         s.runtimeStateLocked(),
		Telemetry:       s.telemetryLocked(),
		EgoTelemetry:    s.egoTelemetryLocked(),
		ActuatorEgo:     s.actuatorEgoLocked(),
		PendingCommands: s.pendingCommandsLocked(),
		AvailableScenes: append([]SceneOption(nil), s.availableScenes...),
	}

	if s.lastEnqueuedCommand != nil {
		state.LastCommand = cloneCommand(s.lastEnqueuedCommand)
	}

	return state
}

func (s *Store) UpdateTelemetry(update TelemetryUpdate) *RuntimeTelemetry {
	s.mu.Lock()
	defer s.mu.Unlock()

	now := s.nowFunc()
	s.telemetry = &RuntimeTelemetry{
		CurrentSpeed:                  update.CurrentSpeed,
		CurrentYaw:                    update.CurrentYaw,
		YawRate:                       update.YawRate,
		Steering:                      update.Steering,
		Acceleration:                  update.Acceleration,
		BrakePressureAvg:              update.BrakePressureAvg,
		VehicleExists:                 update.VehicleExists,
		IsInVehicle:                   update.IsInVehicle,
		VehicleModelHash:              update.VehicleModelHash,
		PositionX:                     cloneFloatPtr(update.PositionX),
		PositionY:                     cloneFloatPtr(update.PositionY),
		PositionZ:                     cloneFloatPtr(update.PositionZ),
		VelocityX:                     cloneFloatPtr(update.VelocityX),
		VelocityY:                     cloneFloatPtr(update.VelocityY),
		VelocityZ:                     cloneFloatPtr(update.VelocityZ),
		PitchDeg:                      cloneFloatPtr(update.PitchDeg),
		RollDeg:                       cloneFloatPtr(update.RollDeg),
		SteeringApplied:               cloneFloatPtr(update.SteeringApplied),
		ThrottleApplied:               cloneFloatPtr(update.ThrottleApplied),
		BrakeApplied:                  cloneFloatPtr(update.BrakeApplied),
		Gear:                          cloneIntPtr(update.Gear),
		RPM:                           cloneFloatPtr(update.RPM),
		WheelAngle:                    cloneFloatPtr(update.WheelAngle),
		WheelSteeringFullLock:         cloneFloatPtr(update.WheelSteeringFullLock),
		OnGround:                      cloneBoolPtr(update.OnGround),
		CollisionState:                update.CollisionState,
		RouteDirectionCode:            update.RouteDirectionCode,
		RouteDirectionDistanceM:       update.RouteDirectionDistanceM,
		RouteDirectionUnknown:         update.RouteDirectionUnknown,
		RouteDirectionKeepStraight:    update.RouteDirectionKeepStraight,
		RouteDirectionTurnLeft:        update.RouteDirectionTurnLeft,
		RouteDirectionTurnRight:       update.RouteDirectionTurnRight,
		RouteDirectionRerouteWrongWay: update.RouteDirectionRerouteWrongWay,
		RouteForwardDelta:             update.RouteForwardDelta,
		RouteHeadingError:             update.RouteHeadingError,
		RouteDistance:                 update.RouteDistance,
		LeadVehicleDistance:           update.LeadVehicleDistance,
		HasLeadVehicle:                update.HasLeadVehicle,
		StopSignTargetConfigured:      update.StopSignTargetConfigured,
		StopSignPose:                  cloneStopSignPose(update.StopSignPose),
		StopLinePose:                  cloneStopSignPose(update.StopLinePose),
		StopSignEgoStopPose:           cloneStopSignPose(update.StopSignEgoStopPose),
		StopSignDistanceM:             update.StopSignDistanceM,
		StopLineDistanceM:             update.StopLineDistanceM,
		StopSignLongitudinalErrorM:    update.StopSignLongitudinalErrorM,
		StopSignLateralErrorM:         update.StopSignLateralErrorM,
		StopSignHeadingErrorDeg:       update.StopSignHeadingErrorDeg,
		StopSignDwellElapsedMS:        max(int64(0), update.StopSignDwellElapsedMS),
		StopSignDwellTargetMS:         max(int64(0), update.StopSignDwellTargetMS),
		StopSignStopped:               update.StopSignStopped,
		StopSignAttemptIndex:          max(0, update.StopSignAttemptIndex),
		StopSignAttemptCount:          max(0, update.StopSignAttemptCount),
		StopSignPhase:                 normalizeStopSignPhase(update.StopSignPhase),
		TimestampMs:                   update.TimestampMs,
		GameTimeMs:                    update.GameTimeMs,
		ReceivedAtMs:                  now.UnixMilli(),
		UpdatedAt:                     now.Format(time.RFC3339),
	}
	if s.telemetry.TimestampMs == 0 {
		s.telemetry.TimestampMs = now.UnixMilli()
	}
	snapshot := s.telemetryAdapter.Snapshot(update, now, s.lastAppliedControls)
	actuatorEgo := s.telemetryAdapter.ToActuatorEgoState(snapshot)
	s.egoTelemetry = &snapshot
	s.actuatorEgo = &actuatorEgo
	s.temporalHistory.Add(snapshot, actuatorEgo, s.lastAppliedControls)
	s.telemetryUpdatedAt = now
	s.telemetryHistory = append(s.telemetryHistory, cloneRuntimeTelemetry(*s.telemetry))
	if len(s.telemetryHistory) > s.telemetryLimit {
		s.telemetryHistory = append([]RuntimeTelemetry(nil), s.telemetryHistory[len(s.telemetryHistory)-s.telemetryLimit:]...)
	}
	if s.lastTelemetryLogAt.IsZero() || now.Sub(s.lastTelemetryLogAt) >= 2*time.Second {
		log.Printf("[control] telemetry recv speed=%.3f yaw=%.3f yawRate=%.3f steer=%.3f accel=%.3f brakeAvg=%.3f valid=%t invalidReason=%q applied=[steer=%.3f throttle=%.3f brake=%.3f] navCode=%.0f navDist=%.3f nav=[unknown=%.0f straight=%.0f left=%.0f right=%.0f reroute=%.0f] routeForward=%.3f routeHeading=%.3f routeDistance=%.3f hasLead=%t leadDistance=%.3f ts=%d",
			update.CurrentSpeed,
			update.CurrentYaw,
			update.YawRate,
			update.Steering,
			update.Acceleration,
			update.BrakePressureAvg,
			snapshot.Valid,
			snapshot.InvalidReason,
			s.lastAppliedControls.Steer,
			s.lastAppliedControls.Throttle,
			s.lastAppliedControls.Brake,
			update.RouteDirectionCode,
			update.RouteDirectionDistanceM,
			update.RouteDirectionUnknown,
			update.RouteDirectionKeepStraight,
			update.RouteDirectionTurnLeft,
			update.RouteDirectionTurnRight,
			update.RouteDirectionRerouteWrongWay,
			update.RouteForwardDelta,
			update.RouteHeadingError,
			update.RouteDistance,
			update.HasLeadVehicle,
			update.LeadVehicleDistance,
			s.telemetry.TimestampMs,
		)
		s.lastTelemetryLogAt = now
	}
	copyTelemetry := cloneRuntimeTelemetry(*s.telemetry)
	return &copyTelemetry
}

func (s *Store) LatestTelemetry() *RuntimeTelemetry {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.telemetryLocked()
}

func (s *Store) LatestTelemetrySnapshot() (*RuntimeTelemetry, time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.telemetryLocked(), s.telemetryUpdatedAt
}

func (s *Store) LatestEgoTelemetrySnapshot() (*EgoTelemetrySnapshot, time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.egoTelemetryLocked(), s.telemetryUpdatedAt
}

func (s *Store) LatestActuatorEgoStateSnapshot() (*ActuatorEgoState, time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.actuatorEgoLocked(), s.telemetryUpdatedAt
}

func (s *Store) UpdateAppliedControls(applied AppliedControls) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if applied.TimestampS <= 0 {
		applied.TimestampS = float64(s.nowFunc().UTC().UnixNano()) / float64(time.Second)
	}
	applied.Steer = clamp(applied.Steer, -1, 1)
	applied.Throttle = clamp(applied.Throttle, 0, 1)
	applied.Brake = clamp(applied.Brake, 0, 1)
	s.lastAppliedControls = applied
	if s.egoTelemetry != nil {
		s.egoTelemetry.SteeringApplied = finiteFloatPtr(applied.Steer)
		s.egoTelemetry.ThrottleApplied = finiteFloatPtr(applied.Throttle)
		s.egoTelemetry.BrakeApplied = finiteFloatPtr(applied.Brake)
	}
	if s.actuatorEgo != nil {
		s.actuatorEgo.LastAppliedSteer = applied.Steer
		s.actuatorEgo.LastAppliedThrottle = applied.Throttle
		s.actuatorEgo.LastAppliedBrake = applied.Brake
	}
}

func (s *Store) TelemetryHistorySnapshot(limit int) []RuntimeTelemetry {
	s.mu.Lock()
	defer s.mu.Unlock()
	if limit <= 0 || limit > len(s.telemetryHistory) {
		limit = len(s.telemetryHistory)
	}
	if limit == 0 {
		return nil
	}
	start := len(s.telemetryHistory) - limit
	out := make([]RuntimeTelemetry, limit)
	for index, telemetry := range s.telemetryHistory[start:] {
		out[index] = cloneRuntimeTelemetry(telemetry)
	}
	return out
}

func (s *Store) TemporalHistorySnapshot(limit int) []TemporalHistoryEntry {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.temporalHistory.Snapshot(limit)
}

func (s *Store) runtimeStateLocked() RuntimeState {
	runtimeState := RuntimeState{
		Status:               s.runtimeStatus,
		ActiveSceneName:      s.activeSceneName,
		LastError:            s.lastError,
		FiveMConnected:       !s.lastPollAt.IsZero() && s.nowFunc().Sub(s.lastPollAt) <= 10*time.Second,
		StopSignBatch:        cloneStopSignBatchProgress(s.stopSignBatch),
		AppliedSafetyEpoch:   s.appliedSafetyEpoch,
		InFlightSafetyStarts: s.inFlightSafetyStarts,
	}

	if !s.runtimeUpdatedAt.IsZero() {
		runtimeState.UpdatedAt = s.runtimeUpdatedAt.Format(time.RFC3339)
	}
	if !s.lastPollAt.IsZero() {
		runtimeState.LastPollAt = s.lastPollAt.Format(time.RFC3339)
	}

	return runtimeState
}

func (s *Store) pendingCommandsLocked() []Command {
	nextIndex := s.nextCommandIndexLocked(s.lastSeenCommandID)
	if nextIndex < 0 {
		return []Command{}
	}

	pending := make([]Command, 0, len(s.commands)-nextIndex)
	for _, command := range s.commands[nextIndex:] {
		if s.commandCanceledLocked(command.ID) {
			continue
		}
		pending = append(pending, *cloneCommand(&command))
	}
	return pending
}

// Canceled commands remain as cursor tombstones until history trimming removes them.
func (s *Store) cancelPendingNonEmergencyCommandsLocked() {
	// The delivered command is still cancelable until the consumer acknowledges its ID.
	startIndex := s.acknowledgedFrontierIndexLocked()
	for _, command := range s.commands[startIndex:] {
		if isEmergencyStopCommand(command.Type) {
			continue
		}
		if s.canceledCommandIDs == nil {
			s.canceledCommandIDs = make(map[string]struct{})
		}
		s.canceledCommandIDs[command.ID] = struct{}{}
	}
}

func (s *Store) acknowledgedFrontierIndexLocked() int {
	index := s.commandIndexLocked(s.lastSeenCommandID)
	if index < 0 {
		return 0
	}
	return index + 1
}

func (s *Store) insertEmergencyCommandLocked(command Command) {
	insertIndex := s.deliveryFrontierIndexLocked()
	for insertIndex < len(s.commands) {
		queued := s.commands[insertIndex]
		if s.commandCanceledLocked(queued.ID) || !isEmergencyStopCommand(queued.Type) {
			break
		}
		insertIndex++
	}

	s.commands = append(s.commands, Command{})
	copy(s.commands[insertIndex+1:], s.commands[insertIndex:])
	s.commands[insertIndex] = command
}

func (s *Store) deliveryFrontierIndexLocked() int {
	frontier := 0
	for _, commandID := range []string{s.lastSeenCommandID, s.lastDeliveredCommandID} {
		index := s.commandIndexLocked(commandID)
		if index >= 0 && index+1 > frontier {
			frontier = index + 1
		}
	}
	return frontier
}

func (s *Store) trimCommandHistoryLocked() {
	trimCount := len(s.commands) - s.commandHistoryLimit
	if trimCount <= 0 {
		return
	}

	frontier := s.deliveryFrontierIndexLocked()
	// Poll only offers a command. Keep that command until a later poll acknowledges it,
	// so dispatch confirmation cannot lose a stop (or any other command) to trimming.
	if offeredIndex := s.commandIndexLocked(s.lastDeliveredCommandID); offeredIndex >= 0 && offeredIndex < frontier {
		frontier = offeredIndex
	}
	if trimCount > frontier {
		trimCount = frontier
	}
	if trimCount == 0 {
		return
	}

	for _, command := range s.commands[:trimCount] {
		delete(s.canceledCommandIDs, command.ID)
	}
	s.commands = append([]Command(nil), s.commands[trimCount:]...)
}

func (s *Store) commandCanceledLocked(commandID string) bool {
	_, canceled := s.canceledCommandIDs[commandID]
	return canceled
}

func (s *Store) telemetryLocked() *RuntimeTelemetry {
	if s.telemetry == nil {
		return nil
	}
	copyTelemetry := cloneRuntimeTelemetry(*s.telemetry)
	return &copyTelemetry
}

func cloneRuntimeTelemetry(source RuntimeTelemetry) RuntimeTelemetry {
	clone := source
	clone.PositionX = cloneFloatPtr(source.PositionX)
	clone.PositionY = cloneFloatPtr(source.PositionY)
	clone.PositionZ = cloneFloatPtr(source.PositionZ)
	clone.VelocityX = cloneFloatPtr(source.VelocityX)
	clone.VelocityY = cloneFloatPtr(source.VelocityY)
	clone.VelocityZ = cloneFloatPtr(source.VelocityZ)
	clone.PitchDeg = cloneFloatPtr(source.PitchDeg)
	clone.RollDeg = cloneFloatPtr(source.RollDeg)
	clone.SteeringApplied = cloneFloatPtr(source.SteeringApplied)
	clone.ThrottleApplied = cloneFloatPtr(source.ThrottleApplied)
	clone.BrakeApplied = cloneFloatPtr(source.BrakeApplied)
	clone.Gear = cloneIntPtr(source.Gear)
	clone.RPM = cloneFloatPtr(source.RPM)
	clone.WheelAngle = cloneFloatPtr(source.WheelAngle)
	clone.WheelSteeringFullLock = cloneFloatPtr(source.WheelSteeringFullLock)
	clone.OnGround = cloneBoolPtr(source.OnGround)
	clone.StopSignPose = cloneStopSignPose(source.StopSignPose)
	clone.StopLinePose = cloneStopSignPose(source.StopLinePose)
	clone.StopSignEgoStopPose = cloneStopSignPose(source.StopSignEgoStopPose)
	return clone
}

func normalizedStopSignBatchProgress(source *StopSignBatchProgress) *StopSignBatchProgress {
	if source == nil {
		return nil
	}
	progress := *source
	progress.BatchID = strings.TrimSpace(progress.BatchID)
	progress.PlanFingerprint = strings.TrimSpace(progress.PlanFingerprint)
	progress.State = strings.TrimSpace(progress.State)
	progress.JobID = strings.TrimSpace(progress.JobID)
	progress.JobCount = max(0, progress.JobCount)
	progress.JobIndex = max(0, min(progress.JobIndex, progress.JobCount))
	progress.CompletedJobs = max(0, min(progress.CompletedJobs, progress.JobCount))
	progress.AttemptCount = max(0, progress.AttemptCount)
	progress.AttemptIndex = max(0, min(progress.AttemptIndex, progress.AttemptCount))
	progress.Phase = normalizeStopSignPhase(progress.Phase)
	return &progress
}

func cloneStopSignBatchProgress(source *StopSignBatchProgress) *StopSignBatchProgress {
	if source == nil {
		return nil
	}
	clone := *source
	return &clone
}

func cloneStopSignPose(source *StopSignPose) *StopSignPose {
	if source == nil {
		return nil
	}
	clone := *source
	return &clone
}

func (s *Store) egoTelemetryLocked() *EgoTelemetrySnapshot {
	if s.egoTelemetry == nil {
		return nil
	}
	copyTelemetry := cloneEgoTelemetrySnapshot(*s.egoTelemetry)
	return &copyTelemetry
}

func (s *Store) actuatorEgoLocked() *ActuatorEgoState {
	if s.actuatorEgo == nil {
		return nil
	}
	copyState := cloneActuatorEgoState(*s.actuatorEgo)
	return &copyState
}

func (s *Store) nextCommandIndexLocked(lastSeenCommandID string) int {
	if len(s.commands) == 0 {
		return -1
	}

	nextIndex := 0
	if index := s.commandIndexLocked(lastSeenCommandID); index >= 0 {
		nextIndex = index + 1
	}
	for nextIndex < len(s.commands) && s.commandCanceledLocked(s.commands[nextIndex].ID) {
		nextIndex++
	}
	if nextIndex >= len(s.commands) {
		return -1
	}
	return nextIndex
}

func (s *Store) commandIndexLocked(commandID string) int {
	if commandID == "" {
		return -1
	}
	for index, command := range s.commands {
		if command.ID == commandID {
			return index
		}
	}
	return -1
}

func validateCommand(
	commandType CommandType,
	sceneName string,
	stopSignBatchID string,
	planFingerprint string,
	stopSignJobs []StopSignBatchJob,
	stopSignCatalogPosition *WorldPosition,
) error {
	switch commandType {
	case CommandStartScene:
		if sceneName == "" {
			return fmt.Errorf("%w: sceneName is required for %s", ErrInvalidCommand, commandType)
		}
	case CommandStartStopSignBatch:
		if strings.TrimSpace(stopSignBatchID) == "" {
			return fmt.Errorf("%w: stopSignBatchId is required for %s", ErrInvalidCommand, commandType)
		}
		if !isSHA256Fingerprint(planFingerprint) {
			return fmt.Errorf("%w: planFingerprint must be a sha256 fingerprint for %s", ErrInvalidCommand, commandType)
		}
		if len(stopSignJobs) == 0 || len(stopSignJobs) > 100 {
			return fmt.Errorf("%w: stopSignJobs must contain between 1 and 100 items for %s", ErrInvalidCommand, commandType)
		}
		jobIDs := make(map[string]struct{}, len(stopSignJobs))
		for index, job := range stopSignJobs {
			jobID := strings.TrimSpace(job.ID)
			if jobID == "" || strings.TrimSpace(job.EntryID) == "" || strings.TrimSpace(job.VariationID) == "" || strings.TrimSpace(job.Seed) == "" {
				return fmt.Errorf("%w: stopSignJobs[%d] requires id, entryId, variationId, and seed", ErrInvalidCommand, index)
			}
			if _, exists := jobIDs[jobID]; exists {
				return fmt.Errorf("%w: stopSignJobs[%d].id %q is duplicated", ErrInvalidCommand, index, jobID)
			}
			jobIDs[jobID] = struct{}{}
			if err := stopsignbatch.ValidateExpandedJob(job); err != nil {
				return fmt.Errorf("%w: stopSignJobs[%d]: %v", ErrInvalidCommand, index, err)
			}
		}
	case CommandStartEgo, CommandRunAllScenes, CommandEndScene, CommandEndAllScenes, CommandStopEgo,
		CommandSetStopSignTarget, CommandClearStopSignTarget:
	case CommandSetStopSignCatalogWaypoint:
		if err := validateWorldPosition(stopSignCatalogPosition); err != nil {
			return fmt.Errorf("%w: stopSignCatalogPosition %v", ErrInvalidCommand, err)
		}
	default:
		return fmt.Errorf("%w: unsupported command type %q", ErrInvalidCommand, commandType)
	}

	return nil
}

func validateWorldPosition(position *WorldPosition) error {
	if position == nil {
		return errors.New("is required")
	}
	for label, value := range map[string]float64{"x": position.X, "y": position.Y, "z": position.Z} {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return fmt.Errorf("%s must be finite", label)
		}
	}
	if position.X < -10_000 || position.X > 10_000 || position.Y < -10_000 || position.Y > 10_000 || position.Z < -1_000 || position.Z > 3_000 {
		return errors.New("is outside the supported GTA world bounds")
	}
	return nil
}

func isSHA256Fingerprint(value string) bool {
	const prefix = "sha256:"
	normalized := strings.TrimSpace(value)
	if len(normalized) != len(prefix)+64 || !strings.HasPrefix(normalized, prefix) {
		return false
	}
	for _, character := range normalized[len(prefix):] {
		if (character < '0' || character > '9') && (character < 'a' || character > 'f') {
			return false
		}
	}
	return true
}

func cloneStopSignBatchJobs(source []StopSignBatchJob) []StopSignBatchJob {
	if len(source) == 0 {
		return nil
	}
	clone := append([]StopSignBatchJob(nil), source...)
	for index := range clone {
		if source[index].Vehicle.Color != nil {
			color := *source[index].Vehicle.Color
			clone[index].Vehicle.Color = &color
		}
	}
	return clone
}

func cloneCommand(source *Command) *Command {
	if source == nil {
		return nil
	}
	clone := *source
	clone.StopSignJobs = cloneStopSignBatchJobs(source.StopSignJobs)
	clone.StopSignCatalogPosition = cloneWorldPosition(source.StopSignCatalogPosition)
	return &clone
}

func cloneWorldPosition(source *WorldPosition) *WorldPosition {
	if source == nil {
		return nil
	}
	clone := *source
	return &clone
}

func rejectedDispatchConfirmation(commandID string, reason DispatchConfirmationReason) DispatchConfirmation {
	return DispatchConfirmation{
		CommandID: commandID,
		Confirmed: false,
		Reason:    reason,
	}
}

func isEmergencyStopCommand(commandType CommandType) bool {
	switch commandType {
	case CommandEndScene, CommandEndAllScenes, CommandStopEgo:
		return true
	default:
		return false
	}
}

func requiresSafetyEpoch(commandType CommandType) bool {
	switch commandType {
	case CommandStartScene, CommandRunAllScenes, CommandStartEgo, CommandStartStopSignBatch:
		return true
	default:
		return false
	}
}

func (s *Store) validateSafetyEpochLocked(commandType CommandType, epoch *uint64) error {
	if !requiresSafetyEpoch(commandType) {
		return nil
	}
	if epoch == nil {
		return fmt.Errorf("%w for %s", ErrSafetyEpochRequired, commandType)
	}
	if *epoch != s.safetyEpoch {
		return fmt.Errorf("%w for %s", ErrSafetyEpochMismatch, commandType)
	}
	return nil
}

func (s *Store) advanceSafetyEpochLocked() {
	if s.safetyEpoch == ^uint64(0) {
		panic("control safety invariant violated: safety epoch exhausted")
	}
	s.safetyEpoch++
}

func normalizeCommandType(commandType CommandType) CommandType {
	return CommandType(strings.TrimSpace(string(commandType)))
}

func normalizeStopSignPhase(phase string) string {
	switch strings.TrimSpace(phase) {
	case StopSignPhaseAccelerate, StopSignPhaseCruiseApproach, StopSignPhaseDecelerate,
		StopSignPhaseStopHold, StopSignPhaseRelease:
		return strings.TrimSpace(phase)
	default:
		return ""
	}
}

func normalizeStatus(status RuntimeStatus) RuntimeStatus {
	switch status {
	case StatusIdle, StatusRunningScene, StatusRunningAllScenes, StatusStopping, StatusError:
		return status
	default:
		return StatusIdle
	}
}
