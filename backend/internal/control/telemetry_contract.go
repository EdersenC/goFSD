package control

import (
	"math"
	"strings"
	"time"
)

const (
	defaultTelemetryHistoryDuration         = 1500 * time.Millisecond
	defaultTelemetryHistoryMaxEntries       = 256
	defaultMaxFrameTelemetrySkew            = 75 * time.Millisecond
	defaultMaxReasonableSpeedMPS            = 120.0
	defaultMaxReasonableAbsPosition         = 1000000.0
	defaultMaxReasonableAbsVelocityMPS      = 180.0
	defaultWheelSteeringFullLockDegrees     = 35.0
	NormalizedControlRangeDescription       = "steer [-1,1], throttle [0,1], brake [0,1]"
	FiveMWorldCoordinateConvention          = "GTA/FiveM world coordinates: X east/west, Y north/south, Z up"
	EgoRelativeWaypointCoordinateConvention = "ego-relative waypoint coordinates: x lateral right positive, y forward positive"
)

type Vec3 struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
	Z float64 `json:"z"`
}

type AppliedControls struct {
	Steer      float64 `json:"steer"`
	Throttle   float64 `json:"throttle"`
	Brake      float64 `json:"brake"`
	TimestampS float64 `json:"timestamp_s,omitempty"`
}

type EgoTelemetrySnapshot struct {
	TimestampS      float64  `json:"timestamp_s"`
	FrameID         *int64   `json:"frame_id,omitempty"`
	GameTimeS       *float64 `json:"game_time_s,omitempty"`
	WallTimeS       *float64 `json:"wall_time_s,omitempty"`
	VehicleExists   bool     `json:"vehicle_exists"`
	IsInVehicle     bool     `json:"is_in_vehicle"`
	PositionX       *float64 `json:"position_x,omitempty"`
	PositionY       *float64 `json:"position_y,omitempty"`
	PositionZ       *float64 `json:"position_z,omitempty"`
	VelocityX       *float64 `json:"velocity_x,omitempty"`
	VelocityY       *float64 `json:"velocity_y,omitempty"`
	VelocityZ       *float64 `json:"velocity_z,omitempty"`
	SpeedMPS        float64  `json:"speed_mps"`
	HeadingRad      *float64 `json:"heading_rad,omitempty"`
	YawRad          *float64 `json:"yaw_rad,omitempty"`
	YawRateRadS     *float64 `json:"yaw_rate_rad_s,omitempty"`
	PitchRad        *float64 `json:"pitch_rad,omitempty"`
	RollRad         *float64 `json:"roll_rad,omitempty"`
	SteeringActual  *float64 `json:"steering_actual,omitempty"`
	ThrottleActual  *float64 `json:"throttle_actual,omitempty"`
	BrakeActual     *float64 `json:"brake_actual,omitempty"`
	SteeringApplied *float64 `json:"steering_applied,omitempty"`
	ThrottleApplied *float64 `json:"throttle_applied,omitempty"`
	BrakeApplied    *float64 `json:"brake_applied,omitempty"`
	Gear            *int     `json:"gear,omitempty"`
	RPM             *float64 `json:"rpm,omitempty"`
	WheelAngle      *float64 `json:"wheel_angle,omitempty"`
	OnGround        *bool    `json:"on_ground,omitempty"`
	CollisionState  string   `json:"collision_state,omitempty"`
	Valid           bool     `json:"valid"`
	InvalidReason   string   `json:"invalid_reason,omitempty"`
}

type ActuatorEgoState struct {
	TimestampS          float64  `json:"timestamp_s"`
	SpeedMPS            float64  `json:"speed_mps"`
	HeadingRad          *float64 `json:"heading_rad,omitempty"`
	YawRateRadS         *float64 `json:"yaw_rate_rad_s,omitempty"`
	Position            *Vec3    `json:"position,omitempty"`
	Velocity            *Vec3    `json:"velocity,omitempty"`
	LastAppliedSteer    float64  `json:"last_applied_steer"`
	LastAppliedThrottle float64  `json:"last_applied_throttle"`
	LastAppliedBrake    float64  `json:"last_applied_brake"`
	Valid               bool     `json:"valid"`
	InvalidReason       string   `json:"invalid_reason,omitempty"`
}

type FiveMTelemetryAdapterConfig struct {
	MaxFrameTelemetrySkew       time.Duration
	WheelSteeringFullLockDeg    float64
	MaxReasonableSpeedMPS       float64
	MaxReasonableAbsPosition    float64
	MaxReasonableAbsVelocityMPS float64
	ControlRangeDescription     string
	PositionConvention          string
	WaypointConvention          string
}

type FiveMTelemetryAdapter struct {
	cfg            FiveMTelemetryAdapterConfig
	lastTimestampS float64
}

func (u TelemetryUpdate) AppliedSteerOr(fallback float64) float64 {
	if u.SteeringApplied != nil && finite(*u.SteeringApplied) {
		return *u.SteeringApplied
	}
	return fallback
}

func (u TelemetryUpdate) AppliedThrottleOr(fallback float64) float64 {
	if u.ThrottleApplied != nil && finite(*u.ThrottleApplied) {
		return *u.ThrottleApplied
	}
	return fallback
}

func (u TelemetryUpdate) AppliedBrakeOr(fallback float64) float64 {
	if u.BrakeApplied != nil && finite(*u.BrakeApplied) {
		return *u.BrakeApplied
	}
	return fallback
}

func DefaultFiveMTelemetryAdapterConfig() FiveMTelemetryAdapterConfig {
	return FiveMTelemetryAdapterConfig{
		MaxFrameTelemetrySkew:       defaultMaxFrameTelemetrySkew,
		WheelSteeringFullLockDeg:    defaultWheelSteeringFullLockDegrees,
		MaxReasonableSpeedMPS:       defaultMaxReasonableSpeedMPS,
		MaxReasonableAbsPosition:    defaultMaxReasonableAbsPosition,
		MaxReasonableAbsVelocityMPS: defaultMaxReasonableAbsVelocityMPS,
		ControlRangeDescription:     NormalizedControlRangeDescription,
		PositionConvention:          FiveMWorldCoordinateConvention,
		WaypointConvention:          EgoRelativeWaypointCoordinateConvention,
	}
}

func NewFiveMTelemetryAdapter(cfg FiveMTelemetryAdapterConfig) *FiveMTelemetryAdapter {
	if cfg.MaxFrameTelemetrySkew <= 0 {
		cfg.MaxFrameTelemetrySkew = defaultMaxFrameTelemetrySkew
	}
	if cfg.WheelSteeringFullLockDeg <= 0 {
		cfg.WheelSteeringFullLockDeg = defaultWheelSteeringFullLockDegrees
	}
	if cfg.MaxReasonableSpeedMPS <= 0 {
		cfg.MaxReasonableSpeedMPS = defaultMaxReasonableSpeedMPS
	}
	if cfg.MaxReasonableAbsPosition <= 0 {
		cfg.MaxReasonableAbsPosition = defaultMaxReasonableAbsPosition
	}
	if cfg.MaxReasonableAbsVelocityMPS <= 0 {
		cfg.MaxReasonableAbsVelocityMPS = defaultMaxReasonableAbsVelocityMPS
	}
	if strings.TrimSpace(cfg.ControlRangeDescription) == "" {
		cfg.ControlRangeDescription = NormalizedControlRangeDescription
	}
	if strings.TrimSpace(cfg.PositionConvention) == "" {
		cfg.PositionConvention = FiveMWorldCoordinateConvention
	}
	if strings.TrimSpace(cfg.WaypointConvention) == "" {
		cfg.WaypointConvention = EgoRelativeWaypointCoordinateConvention
	}
	return &FiveMTelemetryAdapter{cfg: cfg}
}

func (a *FiveMTelemetryAdapter) Snapshot(update TelemetryUpdate, receivedAt time.Time, applied AppliedControls) EgoTelemetrySnapshot {
	if a == nil {
		a = NewFiveMTelemetryAdapter(DefaultFiveMTelemetryAdapterConfig())
	}
	timestampS := 0.0
	if update.TimestampMs > 0 {
		timestampS = float64(update.TimestampMs) / 1000.0
	}
	snapshot := EgoTelemetrySnapshot{
		TimestampS:      timestampS,
		VehicleExists:   update.VehicleExists,
		IsInVehicle:     update.IsInVehicle,
		PositionX:       cloneFloatPtr(update.PositionX),
		PositionY:       cloneFloatPtr(update.PositionY),
		PositionZ:       cloneFloatPtr(update.PositionZ),
		VelocityX:       cloneFloatPtr(update.VelocityX),
		VelocityY:       cloneFloatPtr(update.VelocityY),
		VelocityZ:       cloneFloatPtr(update.VelocityZ),
		SpeedMPS:        update.CurrentSpeed,
		YawRateRadS:     finiteFloatPtr(update.YawRate),
		Gear:            cloneIntPtr(update.Gear),
		RPM:             cloneFloatPtr(update.RPM),
		WheelAngle:      cloneFloatPtr(update.WheelAngle),
		OnGround:        cloneBoolPtr(update.OnGround),
		CollisionState:  strings.TrimSpace(update.CollisionState),
		SteeringApplied: finiteFloatPtr(clamp(update.AppliedSteerOr(applied.Steer), -1, 1)),
		ThrottleApplied: finiteFloatPtr(clamp(update.AppliedThrottleOr(applied.Throttle), 0, 1)),
		BrakeApplied:    finiteFloatPtr(clamp(update.AppliedBrakeOr(applied.Brake), 0, 1)),
	}
	if timestampS > 0 {
		wallTime := timestampS
		snapshot.WallTimeS = &wallTime
	}
	if update.GameTimeMs > 0 {
		gameTimeS := float64(update.GameTimeMs) / 1000.0
		snapshot.GameTimeS = &gameTimeS
	}

	// FiveM's GetEntitySpeed is already meters/second. GetEntityHeading,
	// GetEntityPitch, and GetEntityRoll are degrees, so convert them here.
	if finite(update.CurrentYaw) {
		heading := degreesToRadians(update.CurrentYaw)
		snapshot.HeadingRad = &heading
		snapshot.YawRad = &heading
	}
	if update.PitchDeg != nil && finite(*update.PitchDeg) {
		pitch := degreesToRadians(*update.PitchDeg)
		snapshot.PitchRad = &pitch
	}
	if update.RollDeg != nil && finite(*update.RollDeg) {
		roll := degreesToRadians(*update.RollDeg)
		snapshot.RollRad = &roll
	}

	// GetVehicleWheelSteeringAngle is a physical wheel angle in degrees, not
	// a requested control. Normalize it as an actual steering proxy.
	if finite(update.Steering) {
		steeringActual := clamp(update.Steering/a.cfg.WheelSteeringFullLockDeg, -1, 1)
		snapshot.SteeringActual = &steeringActual
	}
	if finite(update.BrakePressureAvg) {
		brakeActual := clamp(update.BrakePressureAvg, 0, 1)
		snapshot.BrakeActual = &brakeActual
	}

	valid, reason := a.validate(snapshot)
	snapshot.Valid = valid
	snapshot.InvalidReason = reason
	if valid {
		a.lastTimestampS = snapshot.TimestampS
	}
	_ = receivedAt
	return snapshot
}

func (a *FiveMTelemetryAdapter) ToActuatorEgoState(snapshot EgoTelemetrySnapshot) ActuatorEgoState {
	state := ActuatorEgoState{
		TimestampS:          snapshot.TimestampS,
		SpeedMPS:            snapshot.SpeedMPS,
		HeadingRad:          cloneFloatPtr(snapshot.HeadingRad),
		YawRateRadS:         cloneFloatPtr(snapshot.YawRateRadS),
		LastAppliedSteer:    optionalFloat(snapshot.SteeringApplied, 0),
		LastAppliedThrottle: optionalFloat(snapshot.ThrottleApplied, 0),
		LastAppliedBrake:    optionalFloat(snapshot.BrakeApplied, 0),
		Valid:               snapshot.Valid,
		InvalidReason:       snapshot.InvalidReason,
	}
	if snapshot.PositionX != nil && snapshot.PositionY != nil && snapshot.PositionZ != nil {
		state.Position = &Vec3{X: *snapshot.PositionX, Y: *snapshot.PositionY, Z: *snapshot.PositionZ}
	}
	if snapshot.VelocityX != nil && snapshot.VelocityY != nil && snapshot.VelocityZ != nil {
		state.Velocity = &Vec3{X: *snapshot.VelocityX, Y: *snapshot.VelocityY, Z: *snapshot.VelocityZ}
	}
	return state
}

func (a *FiveMTelemetryAdapter) validate(snapshot EgoTelemetrySnapshot) (bool, string) {
	if snapshot.TimestampS <= 0 || !finite(snapshot.TimestampS) {
		return false, "timestamp missing"
	}
	if a.lastTimestampS > 0 && snapshot.TimestampS < a.lastTimestampS {
		return false, "timestamp moved backward"
	}
	if !snapshot.VehicleExists {
		return false, "vehicle does not exist"
	}
	if !snapshot.IsInVehicle {
		return false, "player is not in vehicle"
	}
	if !finite(snapshot.SpeedMPS) || snapshot.SpeedMPS < 0 {
		return false, "speed is invalid"
	}
	if snapshot.SpeedMPS > a.cfg.MaxReasonableSpeedMPS {
		return false, "speed is unreasonable"
	}
	if invalidVec3(snapshot.PositionX, snapshot.PositionY, snapshot.PositionZ, a.cfg.MaxReasonableAbsPosition) {
		return false, "position is unreasonable"
	}
	if invalidVec3(snapshot.VelocityX, snapshot.VelocityY, snapshot.VelocityZ, a.cfg.MaxReasonableAbsVelocityMPS) {
		return false, "velocity is unreasonable"
	}
	return true, ""
}

type TemporalHistoryEntry struct {
	TimestampS float64              `json:"timestamp_s"`
	FrameID    *int64               `json:"frame_id,omitempty"`
	Telemetry  EgoTelemetrySnapshot `json:"telemetry"`
	EgoState   ActuatorEgoState     `json:"ego_state"`
	Applied    AppliedControls      `json:"applied_controls"`
}

type CompactHistoryEntry struct {
	TimestampS float64 `json:"timestamp_s"`
	FrameID    *int64  `json:"frame_id,omitempty"`
	SpeedMPS   float64 `json:"speed_mps"`
	Steer      float64 `json:"applied_steer"`
	Throttle   float64 `json:"applied_throttle"`
	Brake      float64 `json:"applied_brake"`
	Valid      bool    `json:"valid"`
}

type TemporalHistoryBuffer struct {
	MaxDuration time.Duration
	MaxEntries  int
	entries     []TemporalHistoryEntry
}

func NewTemporalHistoryBuffer(maxDuration time.Duration, maxEntries int) TemporalHistoryBuffer {
	if maxDuration <= 0 {
		maxDuration = defaultTelemetryHistoryDuration
	}
	if maxEntries <= 0 {
		maxEntries = defaultTelemetryHistoryMaxEntries
	}
	return TemporalHistoryBuffer{MaxDuration: maxDuration, MaxEntries: maxEntries}
}

func (b *TemporalHistoryBuffer) Add(snapshot EgoTelemetrySnapshot, state ActuatorEgoState, applied AppliedControls) {
	if b.MaxDuration <= 0 {
		b.MaxDuration = defaultTelemetryHistoryDuration
	}
	if b.MaxEntries <= 0 {
		b.MaxEntries = defaultTelemetryHistoryMaxEntries
	}
	entry := TemporalHistoryEntry{
		TimestampS: snapshot.TimestampS,
		FrameID:    cloneInt64Ptr(snapshot.FrameID),
		Telemetry:  cloneEgoTelemetrySnapshot(snapshot),
		EgoState:   cloneActuatorEgoState(state),
		Applied:    applied,
	}
	b.entries = append(b.entries, entry)
	b.trim()
}

func (b *TemporalHistoryBuffer) Snapshot(limit int) []TemporalHistoryEntry {
	if limit <= 0 || limit > len(b.entries) {
		limit = len(b.entries)
	}
	if limit == 0 {
		return nil
	}
	start := len(b.entries) - limit
	out := make([]TemporalHistoryEntry, limit)
	for index := range out {
		out[index] = cloneTemporalHistoryEntry(b.entries[start+index])
	}
	return out
}

func (b *TemporalHistoryBuffer) LatestValidState() (ActuatorEgoState, bool) {
	for index := len(b.entries) - 1; index >= 0; index-- {
		if b.entries[index].EgoState.Valid {
			return cloneActuatorEgoState(b.entries[index].EgoState), true
		}
	}
	return ActuatorEgoState{}, false
}

func (b *TemporalHistoryBuffer) QueryByTimestamp(timestampS float64) (TemporalHistoryEntry, bool) {
	for index := len(b.entries) - 1; index >= 0; index-- {
		if b.entries[index].TimestampS <= timestampS {
			return cloneTemporalHistoryEntry(b.entries[index]), true
		}
	}
	return TemporalHistoryEntry{}, false
}

func (b *TemporalHistoryBuffer) ExportCompactHistory() []CompactHistoryEntry {
	out := make([]CompactHistoryEntry, 0, len(b.entries))
	for _, entry := range b.entries {
		out = append(out, CompactHistoryEntry{
			TimestampS: entry.TimestampS,
			FrameID:    cloneInt64Ptr(entry.FrameID),
			SpeedMPS:   entry.EgoState.SpeedMPS,
			Steer:      entry.Applied.Steer,
			Throttle:   entry.Applied.Throttle,
			Brake:      entry.Applied.Brake,
			Valid:      entry.EgoState.Valid,
		})
	}
	return out
}

func (b *TemporalHistoryBuffer) trim() {
	if b.MaxEntries > 0 && len(b.entries) > b.MaxEntries {
		b.entries = append([]TemporalHistoryEntry(nil), b.entries[len(b.entries)-b.MaxEntries:]...)
	}
	if b.MaxDuration <= 0 || len(b.entries) == 0 {
		return
	}
	latest := b.entries[len(b.entries)-1].TimestampS
	if latest <= 0 {
		return
	}
	cutoff := latest - b.MaxDuration.Seconds()
	start := 0
	for start < len(b.entries) && b.entries[start].TimestampS < cutoff {
		start++
	}
	if start > 0 {
		b.entries = append([]TemporalHistoryEntry(nil), b.entries[start:]...)
	}
}

func cloneTemporalHistoryEntry(entry TemporalHistoryEntry) TemporalHistoryEntry {
	return TemporalHistoryEntry{
		TimestampS: entry.TimestampS,
		FrameID:    cloneInt64Ptr(entry.FrameID),
		Telemetry:  cloneEgoTelemetrySnapshot(entry.Telemetry),
		EgoState:   cloneActuatorEgoState(entry.EgoState),
		Applied:    entry.Applied,
	}
}

func cloneEgoTelemetrySnapshot(snapshot EgoTelemetrySnapshot) EgoTelemetrySnapshot {
	out := snapshot
	out.FrameID = cloneInt64Ptr(snapshot.FrameID)
	out.GameTimeS = cloneFloatPtr(snapshot.GameTimeS)
	out.WallTimeS = cloneFloatPtr(snapshot.WallTimeS)
	out.PositionX = cloneFloatPtr(snapshot.PositionX)
	out.PositionY = cloneFloatPtr(snapshot.PositionY)
	out.PositionZ = cloneFloatPtr(snapshot.PositionZ)
	out.VelocityX = cloneFloatPtr(snapshot.VelocityX)
	out.VelocityY = cloneFloatPtr(snapshot.VelocityY)
	out.VelocityZ = cloneFloatPtr(snapshot.VelocityZ)
	out.HeadingRad = cloneFloatPtr(snapshot.HeadingRad)
	out.YawRad = cloneFloatPtr(snapshot.YawRad)
	out.YawRateRadS = cloneFloatPtr(snapshot.YawRateRadS)
	out.PitchRad = cloneFloatPtr(snapshot.PitchRad)
	out.RollRad = cloneFloatPtr(snapshot.RollRad)
	out.SteeringActual = cloneFloatPtr(snapshot.SteeringActual)
	out.ThrottleActual = cloneFloatPtr(snapshot.ThrottleActual)
	out.BrakeActual = cloneFloatPtr(snapshot.BrakeActual)
	out.SteeringApplied = cloneFloatPtr(snapshot.SteeringApplied)
	out.ThrottleApplied = cloneFloatPtr(snapshot.ThrottleApplied)
	out.BrakeApplied = cloneFloatPtr(snapshot.BrakeApplied)
	out.Gear = cloneIntPtr(snapshot.Gear)
	out.RPM = cloneFloatPtr(snapshot.RPM)
	out.WheelAngle = cloneFloatPtr(snapshot.WheelAngle)
	out.OnGround = cloneBoolPtr(snapshot.OnGround)
	return out
}

func cloneActuatorEgoState(state ActuatorEgoState) ActuatorEgoState {
	out := state
	out.HeadingRad = cloneFloatPtr(state.HeadingRad)
	out.YawRateRadS = cloneFloatPtr(state.YawRateRadS)
	if state.Position != nil {
		position := *state.Position
		out.Position = &position
	}
	if state.Velocity != nil {
		velocity := *state.Velocity
		out.Velocity = &velocity
	}
	return out
}

func invalidVec3(x, y, z *float64, maxAbs float64) bool {
	for _, value := range []*float64{x, y, z} {
		if value == nil {
			continue
		}
		if !finite(*value) || math.Abs(*value) > maxAbs {
			return true
		}
	}
	return false
}

func degreesToRadians(degrees float64) float64 {
	return degrees * math.Pi / 180.0
}

func finite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func finiteFloatPtr(value float64) *float64 {
	if !finite(value) {
		return nil
	}
	copyValue := value
	return &copyValue
}

func optionalFloat(value *float64, fallback float64) float64 {
	if value == nil || !finite(*value) {
		return fallback
	}
	return *value
}

func cloneFloatPtr(value *float64) *float64 {
	if value == nil {
		return nil
	}
	copyValue := *value
	return &copyValue
}

func cloneIntPtr(value *int) *int {
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

func cloneBoolPtr(value *bool) *bool {
	if value == nil {
		return nil
	}
	copyValue := *value
	return &copyValue
}

func clamp(value, minValue, maxValue float64) float64 {
	if value < minValue {
		return minValue
	}
	if value > maxValue {
		return maxValue
	}
	return value
}
