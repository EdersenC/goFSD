package control

import (
	"errors"
	"fmt"
	"log"
	"strings"
	"sync"
	"time"
)

var ErrInvalidCommand = errors.New("invalid control command")

type CommandType string

const (
	CommandStartScene   CommandType = "startScene"
	CommandRunAllScenes CommandType = "runAllScenes"
	CommandEndScene     CommandType = "endScene"
	CommandEndAllScenes CommandType = "endAllScenes"
	CommandStartEgo     CommandType = "startEgo"
	CommandStopEgo      CommandType = "stopEgo"
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
	ID        string      `json:"id"`
	Type      CommandType `json:"type"`
	SceneName string      `json:"sceneName,omitempty"`
	CreatedAt string      `json:"createdAt"`
}

type CommandRequest struct {
	Type      CommandType `json:"type"`
	SceneName string      `json:"sceneName,omitempty"`
}

type StatusUpdate struct {
	Status          RuntimeStatus `json:"status"`
	ActiveSceneName string        `json:"activeSceneName,omitempty"`
	LastError       string        `json:"lastError,omitempty"`
}

type TelemetryUpdate struct {
	CurrentSpeed        float64 `json:"currentSpeed"`
	CurrentYaw          float64 `json:"currentYaw"`
	YawRate             float64 `json:"yawRate"`
	Steering            float64 `json:"steering"`
	Acceleration        float64 `json:"acceleration"`
	BrakePressureAvg    float64 `json:"brakePressureAvg"`
	RouteForwardDelta   float64 `json:"routeForwardDelta"`
	RouteHeadingError   float64 `json:"routeHeadingError"`
	RouteDistance       float64 `json:"routeDistance"`
	LeadVehicleDistance float64 `json:"leadVehicleDistance"`
	HasLeadVehicle      bool    `json:"hasLeadVehicle"`
	TimestampMs         int64   `json:"timestampMs,omitempty"`
	GameTimeMs          int64   `json:"gameTimeMs,omitempty"`
}

type RuntimeTelemetry struct {
	CurrentSpeed        float64 `json:"currentSpeed"`
	CurrentYaw          float64 `json:"currentYaw"`
	YawRate             float64 `json:"yawRate"`
	Steering            float64 `json:"steering"`
	Acceleration        float64 `json:"acceleration"`
	BrakePressureAvg    float64 `json:"brakePressureAvg"`
	RouteForwardDelta   float64 `json:"routeForwardDelta"`
	RouteHeadingError   float64 `json:"routeHeadingError"`
	RouteDistance       float64 `json:"routeDistance"`
	LeadVehicleDistance float64 `json:"leadVehicleDistance"`
	HasLeadVehicle      bool    `json:"hasLeadVehicle"`
	TimestampMs         int64   `json:"timestampMs,omitempty"`
	GameTimeMs          int64   `json:"gameTimeMs,omitempty"`
	ReceivedAtMs        int64   `json:"receivedAtMs,omitempty"`
	UpdatedAt           string  `json:"updatedAt,omitempty"`
}

type SceneOption struct {
	Name  string `json:"name"`
	Label string `json:"label"`
}

type RuntimeState struct {
	Status          RuntimeStatus `json:"status"`
	ActiveSceneName string        `json:"activeSceneName,omitempty"`
	LastError       string        `json:"lastError,omitempty"`
	UpdatedAt       string        `json:"updatedAt,omitempty"`
	FiveMConnected  bool          `json:"fivemConnected"`
	LastPollAt      string        `json:"lastPollAt,omitempty"`
}

type State struct {
	Runtime         RuntimeState      `json:"runtime"`
	Telemetry       *RuntimeTelemetry `json:"telemetry,omitempty"`
	LastCommand     *Command          `json:"lastCommand,omitempty"`
	PendingCommands []Command         `json:"pendingCommands"`
	AvailableScenes []SceneOption     `json:"availableScenes"`
}

type Store struct {
	mu sync.Mutex

	nowFunc func() time.Time

	commands            []Command
	activeSessionID     string
	lastSeenCommandID   string
	lastPollAt          time.Time
	runtimeStatus       RuntimeStatus
	activeSceneName     string
	lastError           string
	runtimeUpdatedAt    time.Time
	telemetry           *RuntimeTelemetry
	telemetryUpdatedAt  time.Time
	lastTelemetryLogAt  time.Time
	availableScenes     []SceneOption
	commandHistoryLimit int
	commandSeq          int64
	telemetryHistory    []RuntimeTelemetry
	telemetryLimit      int
}

type Option func(*Store)

func NewStore(opts ...Option) *Store {
	s := &Store{
		nowFunc:             time.Now,
		runtimeStatus:       StatusIdle,
		commandHistoryLimit: 32,
		telemetryLimit:      512,
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

	if err := validateCommand(commandType, sceneName); err != nil {
		return Command{}, err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	s.commandSeq++
	now := s.nowFunc()
	command := Command{
		ID:        fmt.Sprintf("cmd-%d-%d", now.UTC().UnixNano(), s.commandSeq),
		Type:      commandType,
		SceneName: sceneName,
		CreatedAt: now.Format(time.RFC3339),
	}

	s.commands = append(s.commands, command)
	if len(s.commands) > s.commandHistoryLimit {
		s.commands = append([]Command(nil), s.commands[len(s.commands)-s.commandHistoryLimit:]...)
	}

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
	return &command
}

func (s *Store) ResetConsumerSession() string {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.activeSessionID = fmt.Sprintf("session-%d-%d", s.nowFunc().UTC().UnixNano(), s.commandSeq+1)
	s.commands = nil
	s.lastSeenCommandID = ""
	s.runtimeStatus = StatusIdle
	s.activeSceneName = ""
	s.lastError = ""
	s.runtimeUpdatedAt = s.nowFunc()

	return s.activeSessionID
}

func (s *Store) UpdateStatus(update StatusUpdate) RuntimeState {
	status := normalizeStatus(update.Status)

	s.mu.Lock()
	defer s.mu.Unlock()

	s.runtimeStatus = status
	s.activeSceneName = strings.TrimSpace(update.ActiveSceneName)
	s.lastError = strings.TrimSpace(update.LastError)
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
		Runtime:         s.runtimeStateLocked(),
		Telemetry:       s.telemetryLocked(),
		PendingCommands: s.pendingCommandsLocked(),
		AvailableScenes: append([]SceneOption(nil), s.availableScenes...),
	}

	if len(s.commands) > 0 {
		last := s.commands[len(s.commands)-1]
		state.LastCommand = &last
	}

	return state
}

func (s *Store) UpdateTelemetry(update TelemetryUpdate) *RuntimeTelemetry {
	s.mu.Lock()
	defer s.mu.Unlock()

	now := s.nowFunc()
	s.telemetry = &RuntimeTelemetry{
		CurrentSpeed:        update.CurrentSpeed,
		CurrentYaw:          update.CurrentYaw,
		YawRate:             update.YawRate,
		Steering:            update.Steering,
		Acceleration:        update.Acceleration,
		BrakePressureAvg:    update.BrakePressureAvg,
		RouteForwardDelta:   update.RouteForwardDelta,
		RouteHeadingError:   update.RouteHeadingError,
		RouteDistance:       update.RouteDistance,
		LeadVehicleDistance: update.LeadVehicleDistance,
		HasLeadVehicle:      update.HasLeadVehicle,
		TimestampMs:         update.TimestampMs,
		GameTimeMs:          update.GameTimeMs,
		ReceivedAtMs:        now.UnixMilli(),
		UpdatedAt:           now.Format(time.RFC3339),
	}
	if s.telemetry.TimestampMs == 0 {
		s.telemetry.TimestampMs = now.UnixMilli()
	}
	s.telemetryUpdatedAt = now
	s.telemetryHistory = append(s.telemetryHistory, *s.telemetry)
	if len(s.telemetryHistory) > s.telemetryLimit {
		s.telemetryHistory = append([]RuntimeTelemetry(nil), s.telemetryHistory[len(s.telemetryHistory)-s.telemetryLimit:]...)
	}
	if s.lastTelemetryLogAt.IsZero() || now.Sub(s.lastTelemetryLogAt) >= 2*time.Second {
		log.Printf("[control] telemetry recv speed=%.3f yaw=%.3f yawRate=%.3f steer=%.3f accel=%.3f brakeAvg=%.3f routeForward=%.3f routeHeading=%.3f routeDistance=%.3f hasLead=%t leadDistance=%.3f ts=%d",
			update.CurrentSpeed,
			update.CurrentYaw,
			update.YawRate,
			update.Steering,
			update.Acceleration,
			update.BrakePressureAvg,
			update.RouteForwardDelta,
			update.RouteHeadingError,
			update.RouteDistance,
			update.HasLeadVehicle,
			update.LeadVehicleDistance,
			s.telemetry.TimestampMs,
		)
		s.lastTelemetryLogAt = now
	}
	copyTelemetry := *s.telemetry
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
	copy(out, s.telemetryHistory[start:])
	return out
}

func (s *Store) runtimeStateLocked() RuntimeState {
	runtimeState := RuntimeState{
		Status:          s.runtimeStatus,
		ActiveSceneName: s.activeSceneName,
		LastError:       s.lastError,
		FiveMConnected:  !s.lastPollAt.IsZero() && s.nowFunc().Sub(s.lastPollAt) <= 10*time.Second,
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

	pending := make([]Command, len(s.commands[nextIndex:]))
	copy(pending, s.commands[nextIndex:])
	return pending
}

func (s *Store) telemetryLocked() *RuntimeTelemetry {
	if s.telemetry == nil {
		return nil
	}
	copyTelemetry := *s.telemetry
	return &copyTelemetry
}

func (s *Store) nextCommandIndexLocked(lastSeenCommandID string) int {
	if len(s.commands) == 0 {
		return -1
	}
	if lastSeenCommandID == "" {
		return 0
	}

	for i, command := range s.commands {
		if command.ID == lastSeenCommandID {
			if i+1 >= len(s.commands) {
				return -1
			}
			return i + 1
		}
	}

	return 0
}

func validateCommand(commandType CommandType, sceneName string) error {
	switch commandType {
	case CommandStartScene:
		if sceneName == "" {
			return fmt.Errorf("%w: sceneName is required for %s", ErrInvalidCommand, commandType)
		}
	case CommandStartEgo, CommandRunAllScenes, CommandEndScene, CommandEndAllScenes, CommandStopEgo:
	default:
		return fmt.Errorf("%w: unsupported command type %q", ErrInvalidCommand, commandType)
	}

	return nil
}

func normalizeCommandType(commandType CommandType) CommandType {
	return CommandType(strings.TrimSpace(string(commandType)))
}

func normalizeStatus(status RuntimeStatus) RuntimeStatus {
	switch status {
	case StatusIdle, StatusRunningScene, StatusRunningAllScenes, StatusStopping, StatusError:
		return status
	default:
		return StatusIdle
	}
}
