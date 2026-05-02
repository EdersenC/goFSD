package control

import (
	"math"
	"testing"
	"time"
)

func TestEnqueueValidatesStartSceneRequiresName(t *testing.T) {
	store := NewStore()

	if _, err := store.Enqueue(CommandRequest{Type: CommandStartScene}); err == nil {
		t.Fatal("expected validation error for empty sceneName")
	}
}

func TestPollReturnsCommandsInOrder(t *testing.T) {
	now := time.Date(2026, 4, 11, 2, 0, 0, 0, time.UTC)
	store := NewStore(WithNowFunc(func() time.Time { return now }))

	first, err := store.Enqueue(CommandRequest{Type: CommandStartScene, SceneName: "inner-city-driving:default"})
	if err != nil {
		t.Fatalf("enqueue first: %v", err)
	}
	second, err := store.Enqueue(CommandRequest{Type: CommandEndScene})
	if err != nil {
		t.Fatalf("enqueue second: %v", err)
	}

	gotFirst := store.Poll("")
	if gotFirst == nil || gotFirst.ID != first.ID {
		t.Fatalf("expected first command, got %#v", gotFirst)
	}

	gotSecond := store.Poll(first.ID)
	if gotSecond == nil || gotSecond.ID != second.ID {
		t.Fatalf("expected second command, got %#v", gotSecond)
	}

	gotNone := store.Poll(second.ID)
	if gotNone != nil {
		t.Fatalf("expected no command, got %#v", gotNone)
	}
}

func TestStateTracksPendingCommandsAndConnectivity(t *testing.T) {
	now := time.Date(2026, 4, 11, 2, 0, 0, 0, time.UTC)
	store := NewStore(WithNowFunc(func() time.Time { return now }))

	first, err := store.Enqueue(CommandRequest{Type: CommandStartScene, SceneName: "inner-city-driving:default"})
	if err != nil {
		t.Fatalf("enqueue first: %v", err)
	}
	_, err = store.Enqueue(CommandRequest{Type: CommandEndAllScenes})
	if err != nil {
		t.Fatalf("enqueue second: %v", err)
	}

	store.Poll(first.ID)
	store.UpdateStatus(StatusUpdate{
		Status:          StatusRunningScene,
		ActiveSceneName: "inner-city-driving:default",
	})

	state := store.State()
	if !state.Runtime.FiveMConnected {
		t.Fatal("expected FiveMConnected to be true")
	}
	if state.Runtime.Status != StatusRunningScene {
		t.Fatalf("expected runningScene status, got %s", state.Runtime.Status)
	}
	if state.Telemetry != nil {
		t.Fatalf("expected telemetry to be nil before updates, got %+v", state.Telemetry)
	}
	if len(state.PendingCommands) != 1 {
		t.Fatalf("expected 1 pending command, got %d", len(state.PendingCommands))
	}
	if state.PendingCommands[0].Type != CommandEndAllScenes {
		t.Fatalf("expected pending endAllScenes, got %s", state.PendingCommands[0].Type)
	}

	now = now.Add(11 * time.Second)
	state = store.State()
	if state.Runtime.FiveMConnected {
		t.Fatal("expected FiveMConnected to become false after poll timeout")
	}
}

func TestUpdateTelemetryExposesLatestSpeedSnapshot(t *testing.T) {
	now := time.Date(2026, 4, 18, 1, 2, 3, 0, time.UTC)
	store := NewStore(WithNowFunc(func() time.Time { return now }))

	telemetry := store.UpdateTelemetry(TelemetryUpdate{
		CurrentSpeed:        4.25,
		CurrentYaw:          182.5,
		RouteForwardDelta:   0.75,
		RouteHeadingError:   -8.0,
		RouteDistance:       32.0,
		LeadVehicleDistance: 18.5,
		HasLeadVehicle:      true,
		TimestampMs:         123456,
	})
	if telemetry == nil {
		t.Fatal("expected telemetry snapshot")
	}
	if telemetry.CurrentSpeed != 4.25 {
		t.Fatalf("unexpected current speed: %+v", telemetry)
	}
	if telemetry.CurrentYaw != 182.5 {
		t.Fatalf("unexpected current yaw: %+v", telemetry)
	}
	if telemetry.RouteForwardDelta != 0.75 {
		t.Fatalf("unexpected route forward delta: %+v", telemetry)
	}
	if telemetry.RouteHeadingError != -8.0 || telemetry.RouteDistance != 32.0 {
		t.Fatalf("unexpected route telemetry: %+v", telemetry)
	}
	if !telemetry.HasLeadVehicle || telemetry.LeadVehicleDistance != 18.5 {
		t.Fatalf("unexpected lead telemetry: %+v", telemetry)
	}

	state := store.State()
	if state.Telemetry == nil {
		t.Fatal("expected telemetry in state")
	}
	if state.Telemetry.CurrentSpeed != 4.25 {
		t.Fatalf("unexpected telemetry in state: %+v", state.Telemetry)
	}
	if state.Telemetry.CurrentYaw != 182.5 {
		t.Fatalf("unexpected telemetry yaw in state: %+v", state.Telemetry)
	}
	if state.Telemetry.RouteForwardDelta != 0.75 {
		t.Fatalf("unexpected telemetry route forward delta in state: %+v", state.Telemetry)
	}
	if state.Telemetry.RouteHeadingError != -8.0 || state.Telemetry.RouteDistance != 32.0 {
		t.Fatalf("unexpected route telemetry in state: %+v", state.Telemetry)
	}
	if !state.Telemetry.HasLeadVehicle || state.Telemetry.LeadVehicleDistance != 18.5 {
		t.Fatalf("unexpected lead telemetry in state: %+v", state.Telemetry)
	}
	if latest := store.LatestTelemetry(); latest == nil || latest.CurrentSpeed != 4.25 {
		t.Fatalf("unexpected latest telemetry: %+v", latest)
	}
}

func TestTelemetryNormalizationProducesEgoSnapshot(t *testing.T) {
	now := time.UnixMilli(1000).UTC()
	store := NewStore(WithNowFunc(func() time.Time { return now }))
	store.UpdateAppliedControls(AppliedControls{Steer: 0.25, Throttle: 0.50, Brake: 0.10, TimestampS: 0.9})
	positionX, positionY, positionZ := 10.0, 20.0, 3.0
	velocityX, velocityY, velocityZ := 1.0, 2.0, 0.0
	pitchDeg, rollDeg := 5.0, -2.0
	gear := 3
	rpm := 0.45
	onGround := true

	store.UpdateTelemetry(TelemetryUpdate{
		CurrentSpeed:     12.5,
		CurrentYaw:       90.0,
		YawRate:          0.4,
		Steering:         17.5,
		BrakePressureAvg: 0.3,
		VehicleExists:    true,
		IsInVehicle:      true,
		PositionX:        &positionX,
		PositionY:        &positionY,
		PositionZ:        &positionZ,
		VelocityX:        &velocityX,
		VelocityY:        &velocityY,
		VelocityZ:        &velocityZ,
		PitchDeg:         &pitchDeg,
		RollDeg:          &rollDeg,
		Gear:             &gear,
		RPM:              &rpm,
		OnGround:         &onGround,
		TimestampMs:      1000,
		GameTimeMs:       5000,
	})

	snapshot, _ := store.LatestEgoTelemetrySnapshot()
	if snapshot == nil || !snapshot.Valid {
		t.Fatalf("expected valid ego telemetry snapshot, got=%+v", snapshot)
	}
	if snapshot.SpeedMPS != 12.5 {
		t.Fatalf("unexpected speed normalization: %+v", snapshot)
	}
	if snapshot.HeadingRad == nil || math.Abs(*snapshot.HeadingRad-math.Pi/2) > 1e-9 {
		t.Fatalf("expected heading in radians, got=%+v", snapshot.HeadingRad)
	}
	if snapshot.SteeringActual == nil || math.Abs(*snapshot.SteeringActual-0.5) > 1e-9 {
		t.Fatalf("expected wheel angle to normalize to steering_actual=0.5, got=%+v", snapshot.SteeringActual)
	}
	if snapshot.BrakeActual == nil || math.Abs(*snapshot.BrakeActual-0.3) > 1e-9 {
		t.Fatalf("expected brake pressure to normalize into brake_actual, got=%+v", snapshot.BrakeActual)
	}
	if snapshot.SteeringApplied == nil || *snapshot.SteeringApplied != 0.25 {
		t.Fatalf("expected applied controls in snapshot, got=%+v", snapshot)
	}

	ego, _ := store.LatestActuatorEgoStateSnapshot()
	if ego == nil || !ego.Valid || ego.Position == nil || ego.Velocity == nil {
		t.Fatalf("expected actuator ego state with position/velocity, got=%+v", ego)
	}
	if ego.LastAppliedSteer != 0.25 || ego.LastAppliedThrottle != 0.50 || ego.LastAppliedBrake != 0.10 {
		t.Fatalf("unexpected applied controls in ego state: %+v", ego)
	}
}

func TestTelemetryTimestampMonotonicityInvalidatesBackwardUpdates(t *testing.T) {
	adapter := NewFiveMTelemetryAdapter(DefaultFiveMTelemetryAdapterConfig())
	first := adapter.Snapshot(TelemetryUpdate{
		CurrentSpeed:  5,
		VehicleExists: true,
		IsInVehicle:   true,
		TimestampMs:   2000,
	}, time.UnixMilli(2000), AppliedControls{})
	if !first.Valid {
		t.Fatalf("expected first snapshot to be valid, got=%+v", first)
	}

	second := adapter.Snapshot(TelemetryUpdate{
		CurrentSpeed:  5,
		VehicleExists: true,
		IsInVehicle:   true,
		TimestampMs:   1900,
	}, time.UnixMilli(1900), AppliedControls{})
	if second.Valid || second.InvalidReason != "timestamp moved backward" {
		t.Fatalf("expected backward timestamp invalidation, got=%+v", second)
	}
}

func TestActuatorStateAdapterConvertsSnapshot(t *testing.T) {
	adapter := NewFiveMTelemetryAdapter(DefaultFiveMTelemetryAdapterConfig())
	x, y, z := 1.0, 2.0, 3.0
	vx, vy, vz := 4.0, 5.0, 0.0
	snapshot := adapter.Snapshot(TelemetryUpdate{
		CurrentSpeed:    6.0,
		CurrentYaw:      180.0,
		YawRate:         0.25,
		VehicleExists:   true,
		IsInVehicle:     true,
		PositionX:       &x,
		PositionY:       &y,
		PositionZ:       &z,
		VelocityX:       &vx,
		VelocityY:       &vy,
		VelocityZ:       &vz,
		SteeringApplied: floatPtr(0.2),
		ThrottleApplied: floatPtr(0.3),
		BrakeApplied:    floatPtr(0.4),
		TimestampMs:     1000,
	}, time.UnixMilli(1000), AppliedControls{})

	state := adapter.ToActuatorEgoState(snapshot)
	if !state.Valid || state.Position == nil || state.Velocity == nil {
		t.Fatalf("expected valid actuator ego state, got=%+v", state)
	}
	if state.SpeedMPS != 6.0 || state.LastAppliedSteer != 0.2 || state.LastAppliedThrottle != 0.3 || state.LastAppliedBrake != 0.4 {
		t.Fatalf("unexpected actuator ego state values: %+v", state)
	}
	if state.HeadingRad == nil || math.Abs(*state.HeadingRad-math.Pi) > 1e-9 {
		t.Fatalf("expected heading conversion to radians, got=%+v", state.HeadingRad)
	}
}

func floatPtr(value float64) *float64 {
	return &value
}

func TestAppliedControlsHistoryUsesAppliedControls(t *testing.T) {
	now := time.UnixMilli(1000).UTC()
	store := NewStore(WithNowFunc(func() time.Time { return now }))
	store.UpdateAppliedControls(AppliedControls{Steer: 0.70, Throttle: 0.20, Brake: 0.05, TimestampS: 0.95})
	store.UpdateTelemetry(TelemetryUpdate{
		CurrentSpeed:  3,
		Steering:      3.5,
		VehicleExists: true,
		IsInVehicle:   true,
		TimestampMs:   1000,
	})

	history := store.TemporalHistorySnapshot(1)
	if len(history) != 1 {
		t.Fatalf("expected one history entry, got=%d", len(history))
	}
	if history[0].Applied.Steer != 0.70 || history[0].Applied.Throttle != 0.20 || history[0].Applied.Brake != 0.05 {
		t.Fatalf("expected history to keep applied controls, got=%+v", history[0])
	}
	if history[0].Telemetry.SteeringActual == nil || *history[0].Telemetry.SteeringActual == history[0].Applied.Steer {
		t.Fatalf("expected actual wheel telemetry to remain distinct from applied controls, got=%+v", history[0])
	}
}

func TestResetConsumerSessionClearsQueuedCommands(t *testing.T) {
	now := time.Date(2026, 4, 12, 1, 0, 0, 0, time.UTC)
	store := NewStore(WithNowFunc(func() time.Time { return now }))

	if _, err := store.Enqueue(CommandRequest{Type: CommandStartScene, SceneName: "inner-city-driving:default"}); err != nil {
		t.Fatalf("enqueue first: %v", err)
	}
	if _, err := store.Enqueue(CommandRequest{Type: CommandEndScene}); err != nil {
		t.Fatalf("enqueue second: %v", err)
	}

	sessionID := store.ResetConsumerSession()
	if sessionID == "" {
		t.Fatal("expected reset to return a session id")
	}

	state := store.State()
	if len(state.PendingCommands) != 0 {
		t.Fatalf("expected pending commands to be cleared, got %d", len(state.PendingCommands))
	}
	if state.Runtime.Status != StatusIdle {
		t.Fatalf("expected runtime status reset to idle, got %s", state.Runtime.Status)
	}

	if cmd := store.Poll(""); cmd != nil {
		t.Fatalf("expected no command after reset, got %#v", cmd)
	}

	next, err := store.Enqueue(CommandRequest{Type: CommandEndAllScenes})
	if err != nil {
		t.Fatalf("enqueue after reset: %v", err)
	}
	got := store.Poll("")
	if got == nil || got.ID != next.ID {
		t.Fatalf("expected new command after reset, got %#v", got)
	}
}
