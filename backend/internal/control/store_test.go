package control

import (
	"errors"
	"fmt"
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

func TestEnqueueAcceptsParkingTargetCommands(t *testing.T) {
	store := NewStore()

	for _, commandType := range []CommandType{CommandSetParkingTarget, CommandClearParkingTarget} {
		command, err := store.Enqueue(CommandRequest{Type: commandType})
		if err != nil {
			t.Fatalf("enqueue %s: %v", commandType, err)
		}
		if command.Type != commandType {
			t.Fatalf("unexpected command: got=%s want=%s", command.Type, commandType)
		}
	}
}

func TestEnqueuePrepareParkingEvaluationCarriesSeed(t *testing.T) {
	store := NewStore()

	command, err := store.Enqueue(CommandRequest{
		Type: CommandPrepareParkingEvaluation,
		Seed: "  evaluation-seed-7  ",
	})
	if err != nil {
		t.Fatalf("enqueue parking evaluation: %v", err)
	}
	if command.Type != CommandPrepareParkingEvaluation {
		t.Fatalf("unexpected command type: %s", command.Type)
	}
	if command.Seed != "evaluation-seed-7" {
		t.Fatalf("expected normalized evaluation seed, got=%q", command.Seed)
	}
}

func TestEnqueueStartParkingRunCarriesAttemptOptions(t *testing.T) {
	store := NewStore()

	command, err := store.Enqueue(CommandRequest{
		Type:         CommandStartParkingRun,
		AttemptCount: 12,
		Seed:         "  training-seed-42  ",
	})
	if err != nil {
		t.Fatalf("enqueue parking run: %v", err)
	}
	if command.AttemptCount != 12 {
		t.Fatalf("unexpected attempt count: %+v", command)
	}
	if command.Seed != "training-seed-42" {
		t.Fatalf("expected normalized seed, got=%q", command.Seed)
	}
}

func TestEnqueueStartParkingRunValidatesAttemptCount(t *testing.T) {
	for _, attemptCount := range []int{-1, 0, maximumParkingAttemptCount + 1} {
		t.Run(fmt.Sprintf("attempts_%d", attemptCount), func(t *testing.T) {
			store := NewStore()
			_, err := store.Enqueue(CommandRequest{
				Type:         CommandStartParkingRun,
				AttemptCount: attemptCount,
			})
			if !errors.Is(err, ErrInvalidCommand) {
				t.Fatalf("expected invalid command error, got=%v", err)
			}
		})
	}

	for _, attemptCount := range []int{1, maximumParkingAttemptCount} {
		store := NewStore()
		if _, err := store.Enqueue(CommandRequest{
			Type:         CommandStartParkingRun,
			AttemptCount: attemptCount,
		}); err != nil {
			t.Fatalf("expected attempt count %d to be valid: %v", attemptCount, err)
		}
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

func TestUpdateTelemetryCopiesParkingStateAcrossSnapshots(t *testing.T) {
	now := time.Date(2026, 8, 4, 12, 0, 0, 0, time.UTC)
	store := NewStore(WithNowFunc(func() time.Time { return now }))
	positionX := 42.5

	updated := store.UpdateTelemetry(TelemetryUpdate{
		PositionX:                &positionX,
		ParkingTargetConfigured:  true,
		ParkingLongitudinalError: -1.25,
		ParkingLateralError:      0.45,
		ParkingHeadingError:      -7.5,
		ParkingDistance:          1.33,
		ParkingInsideBay:         true,
		ParkingAligned:           true,
		ParkingParked:            true,
		ParkingAttemptIndex:      3,
		ParkingAttemptCount:      10,
		ParkingPhase:             "  parked  ",
	})
	assertParkingTelemetry(t, updated)
	if updated.PositionX == nil || *updated.PositionX != 42.5 {
		t.Fatalf("unexpected copied position: %+v", updated.PositionX)
	}

	*updated.PositionX = -99
	updated.ParkingPhase = "mutated"
	assertParkingTelemetry(t, store.State().Telemetry)
	assertParkingTelemetry(t, store.LatestTelemetry())

	history := store.TelemetryHistorySnapshot(1)
	if len(history) != 1 {
		t.Fatalf("expected one telemetry history entry, got=%d", len(history))
	}
	assertParkingTelemetry(t, &history[0])
	*history[0].PositionX = -100
	if freshHistory := store.TelemetryHistorySnapshot(1); freshHistory[0].PositionX == nil || *freshHistory[0].PositionX != 42.5 {
		t.Fatalf("expected history snapshots to be independent copies, got=%+v", freshHistory[0].PositionX)
	}
}

func TestUpdateTelemetryDefaultsEmptyParkingPhaseToIdle(t *testing.T) {
	store := NewStore()

	telemetry := store.UpdateTelemetry(TelemetryUpdate{ParkingPhase: "  "})
	if telemetry.ParkingPhase != parkingPhaseIdle {
		t.Fatalf("unexpected default parking phase: got=%q want=%q", telemetry.ParkingPhase, parkingPhaseIdle)
	}
}

func assertParkingTelemetry(t *testing.T, telemetry *RuntimeTelemetry) {
	t.Helper()
	if telemetry == nil {
		t.Fatal("expected parking telemetry")
	}
	if !telemetry.ParkingTargetConfigured {
		t.Fatalf("expected configured parking target: %+v", telemetry)
	}
	if telemetry.ParkingLongitudinalError != -1.25 || telemetry.ParkingLateralError != 0.45 {
		t.Fatalf("unexpected parking position error: %+v", telemetry)
	}
	if telemetry.ParkingHeadingError != -7.5 || telemetry.ParkingDistance != 1.33 {
		t.Fatalf("unexpected parking heading/distance: %+v", telemetry)
	}
	if !telemetry.ParkingInsideBay || !telemetry.ParkingAligned || !telemetry.ParkingParked {
		t.Fatalf("unexpected parking completion flags: %+v", telemetry)
	}
	if telemetry.ParkingAttemptIndex != 3 || telemetry.ParkingAttemptCount != 10 {
		t.Fatalf("unexpected parking attempt state: %+v", telemetry)
	}
	if telemetry.ParkingPhase != "parked" {
		t.Fatalf("unexpected parking phase: got=%q", telemetry.ParkingPhase)
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
	wheelAngle := 0.366519153
	wheelSteeringFullLock := 0.733038306

	store.UpdateTelemetry(TelemetryUpdate{
		CurrentSpeed:          12.5,
		CurrentYaw:            90.0,
		YawRate:               0.4,
		Steering:              0.5,
		BrakePressureAvg:      0.3,
		VehicleExists:         true,
		IsInVehicle:           true,
		PositionX:             &positionX,
		PositionY:             &positionY,
		PositionZ:             &positionZ,
		VelocityX:             &velocityX,
		VelocityY:             &velocityY,
		VelocityZ:             &velocityZ,
		PitchDeg:              &pitchDeg,
		RollDeg:               &rollDeg,
		Gear:                  &gear,
		RPM:                   &rpm,
		WheelAngle:            &wheelAngle,
		WheelSteeringFullLock: &wheelSteeringFullLock,
		OnGround:              &onGround,
		TimestampMs:           1000,
		GameTimeMs:            5000,
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
		t.Fatalf("expected normalized steering_actual=0.5, got=%+v", snapshot.SteeringActual)
	}
	if snapshot.WheelAngle == nil || *snapshot.WheelAngle != wheelAngle {
		t.Fatalf("expected raw wheel angle diagnostics, got=%+v", snapshot.WheelAngle)
	}
	if snapshot.WheelSteeringFullLock == nil || *snapshot.WheelSteeringFullLock != wheelSteeringFullLock {
		t.Fatalf("expected raw steering full-lock diagnostics, got=%+v", snapshot.WheelSteeringFullLock)
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
		Steering:      0.1,
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
