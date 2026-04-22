package control

import (
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
		CurrentSpeed:      4.25,
		CurrentYaw:        182.5,
		RouteForwardDelta: 0.75,
		TimestampMs:       123456,
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
	if latest := store.LatestTelemetry(); latest == nil || latest.CurrentSpeed != 4.25 {
		t.Fatalf("unexpected latest telemetry: %+v", latest)
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
