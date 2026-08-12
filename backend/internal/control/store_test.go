package control

import (
	"errors"
	"math"
	"testing"
	"time"

	"awesomeProject/internal/stopsignbatch"
)

func TestEnqueueValidatesStartSceneRequiresName(t *testing.T) {
	store := NewStore()

	if _, err := store.Enqueue(CommandRequest{Type: CommandStartScene}); err == nil {
		t.Fatal("expected validation error for empty sceneName")
	}
}

func TestEnqueueStartStopSignBatchCarriesJobsWithoutAliasing(t *testing.T) {
	store := NewStore()
	request := validStopSignBatchCommandRequest(currentSafetyEpochPtr(store))
	command, err := store.Enqueue(request)
	if err != nil {
		t.Fatalf("enqueue stop-sign batch: %v", err)
	}
	if command.Type != CommandStartStopSignBatch || command.StopSignBatchID != "city-stops" || len(command.StopSignJobs) != 1 {
		t.Fatalf("unexpected command: %+v", command)
	}
	request.StopSignJobs[0].Vehicle.Color.R = 0
	if command.StopSignJobs[0].Vehicle.Color == nil || command.StopSignJobs[0].Vehicle.Color.R != 255 {
		t.Fatalf("enqueue retained caller-owned job memory: %+v", command.StopSignJobs[0])
	}
	state := store.State()
	state.PendingCommands[0].StopSignJobs[0].Vehicle.Color.G = 0
	secondState := store.State()
	if secondState.PendingCommands[0].StopSignJobs[0].Vehicle.Color.G != 128 {
		t.Fatalf("state returned aliased stop-sign job memory: %+v", secondState.PendingCommands[0])
	}
}

func TestEnqueueStopSignCatalogWaypointValidatesAndClonesPosition(t *testing.T) {
	store := NewStore()
	position := &WorldPosition{X: -2335.7034, Y: 3269.0288, Z: 31.81049}
	command, err := store.Enqueue(CommandRequest{
		Type:                    CommandSetStopSignCatalogWaypoint,
		StopSignCatalogPosition: position,
	})
	if err != nil {
		t.Fatalf("enqueue stop-sign catalog waypoint: %v", err)
	}
	position.X = 0
	if command.StopSignCatalogPosition == nil || command.StopSignCatalogPosition.X != -2335.7034 {
		t.Fatalf("enqueue retained caller-owned position: %+v", command.StopSignCatalogPosition)
	}

	state := store.State()
	state.PendingCommands[0].StopSignCatalogPosition.Y = 0
	secondState := store.State()
	if got := secondState.PendingCommands[0].StopSignCatalogPosition.Y; got != 3269.0288 {
		t.Fatalf("state returned aliased catalog position: got=%f", got)
	}

	for name, invalid := range map[string]*WorldPosition{
		"missing":       nil,
		"non-finite":    {X: math.Inf(1)},
		"out-of-bounds": {X: 20_000},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := store.Enqueue(CommandRequest{
				Type:                    CommandSetStopSignCatalogWaypoint,
				StopSignCatalogPosition: invalid,
			})
			if !errors.Is(err, ErrInvalidCommand) {
				t.Fatalf("expected invalid command, got=%v", err)
			}
		})
	}
}

func TestEnqueueStopSignProbeValidatesClonesAndRequiresSafetyEpoch(t *testing.T) {
	store := NewStore()
	probe := &StopSignProbe{
		CatalogID:        "gta-v-sign-0022",
		CatalogPosition:  WorldPosition{X: -1832.1594, Y: 145.45688, Z: 77.10415},
		HeadingOffsetDeg: 180,
	}
	if _, err := store.Enqueue(CommandRequest{Type: CommandProbeStopSignTarget, StopSignProbe: probe}); !errors.Is(err, ErrSafetyEpochRequired) {
		t.Fatalf("probe without safety epoch: got=%v", err)
	}

	command, err := store.Enqueue(CommandRequest{
		Type:          CommandProbeStopSignTarget,
		SafetyEpoch:   currentSafetyEpochPtr(store),
		StopSignProbe: probe,
	})
	if err != nil {
		t.Fatalf("enqueue stop-sign probe: %v", err)
	}
	probe.CatalogPosition.X = 0
	if command.StopSignProbe == nil || command.StopSignProbe.CatalogPosition.X != -1832.1594 {
		t.Fatalf("enqueue retained caller-owned probe: %+v", command.StopSignProbe)
	}
	state := store.State()
	state.PendingCommands[0].StopSignProbe.CatalogPosition.Y = 0
	if got := store.State().PendingCommands[0].StopSignProbe.CatalogPosition.Y; got != 145.45688 {
		t.Fatalf("state returned aliased stop-sign probe: got=%f", got)
	}

	for name, invalid := range map[string]*StopSignProbe{
		"missing":         nil,
		"missing id":      {CatalogPosition: WorldPosition{X: 1, Y: 2, Z: 3}},
		"bad coordinates": {CatalogID: "bad", CatalogPosition: WorldPosition{X: math.Inf(1)}},
		"bad heading":     {CatalogID: "bad", CatalogPosition: WorldPosition{X: 1, Y: 2, Z: 3}, HeadingOffsetDeg: 90},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := store.Enqueue(CommandRequest{
				Type:          CommandProbeStopSignTarget,
				SafetyEpoch:   currentSafetyEpochPtr(store),
				StopSignProbe: invalid,
			})
			if !errors.Is(err, ErrInvalidCommand) {
				t.Fatalf("expected invalid command, got=%v", err)
			}
		})
	}
}

func TestEnqueueTeleportStopSignStartValidatesClonesAndRequiresSafetyEpoch(t *testing.T) {
	store := NewStore()
	pose := &StopSignPose{X: -1618.7, Y: -412.4, Z: 40.3, Heading: 140.24}
	if _, err := store.Enqueue(CommandRequest{Type: CommandTeleportStopSignStart, StopSignStartPose: pose}); !errors.Is(err, ErrSafetyEpochRequired) {
		t.Fatalf("teleport without safety epoch: got=%v", err)
	}

	command, err := store.Enqueue(CommandRequest{
		Type:              CommandTeleportStopSignStart,
		SafetyEpoch:       currentSafetyEpochPtr(store),
		StopSignStartPose: pose,
	})
	if err != nil {
		t.Fatalf("enqueue start teleport: %v", err)
	}
	pose.X = 0
	if command.StopSignStartPose == nil || command.StopSignStartPose.X != -1618.7 {
		t.Fatalf("enqueue retained caller-owned start pose: %+v", command.StopSignStartPose)
	}
	state := store.State()
	state.PendingCommands[0].StopSignStartPose.Y = 0
	if got := store.State().PendingCommands[0].StopSignStartPose.Y; got != -412.4 {
		t.Fatalf("state returned aliased start pose: got=%f", got)
	}

	for name, invalid := range map[string]*StopSignPose{
		"missing":          nil,
		"non-finite":       {X: math.Inf(1)},
		"out-of-bounds":    {X: 20_000},
		"bad heading low":  {Heading: -1},
		"bad heading high": {Heading: 360},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := store.Enqueue(CommandRequest{Type: CommandTeleportStopSignStart, SafetyEpoch: currentSafetyEpochPtr(store), StopSignStartPose: invalid})
			if !errors.Is(err, ErrInvalidCommand) {
				t.Fatalf("expected invalid command, got=%v", err)
			}
		})
	}
}

func TestEnqueueStartStopSignBatchValidatesIdentityAndAttempts(t *testing.T) {
	tests := []struct {
		name string
		edit func(*CommandRequest)
	}{
		{name: "missing batch id", edit: func(request *CommandRequest) { request.StopSignBatchID = "" }},
		{name: "missing fingerprint", edit: func(request *CommandRequest) { request.PlanFingerprint = "" }},
		{name: "missing jobs", edit: func(request *CommandRequest) { request.StopSignJobs = nil }},
		{name: "missing job seed", edit: func(request *CommandRequest) { request.StopSignJobs[0].Seed = "" }},
		{name: "invalid attempts", edit: func(request *CommandRequest) { request.StopSignJobs[0].AttemptCount = 0 }},
		{name: "duplicate job", edit: func(request *CommandRequest) {
			request.StopSignJobs = append(request.StopSignJobs, request.StopSignJobs[0])
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			store := NewStore()
			request := validStopSignBatchCommandRequest(currentSafetyEpochPtr(store))
			test.edit(&request)
			if _, err := store.Enqueue(request); !errors.Is(err, ErrInvalidCommand) {
				t.Fatalf("expected invalid command, got=%v", err)
			}
		})
	}
}

func TestPollReturnsCommandsInOrder(t *testing.T) {
	now := time.Date(2026, 4, 11, 2, 0, 0, 0, time.UTC)
	store := NewStore(WithNowFunc(func() time.Time { return now }))

	first, err := store.Enqueue(CommandRequest{Type: CommandSetStopSignTarget})
	if err != nil {
		t.Fatalf("enqueue first: %v", err)
	}
	second, err := store.Enqueue(CommandRequest{Type: CommandClearStopSignTarget})
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

func TestEmergencyHoldFlushesUndeliveredCommandsAndPreservesStopFIFO(t *testing.T) {
	store := NewStore()

	commandsToFlush := []CommandRequest{
		{Type: CommandSetStopSignTarget},
		{Type: CommandStartScene, SafetyEpoch: currentSafetyEpochPtr(store), SceneName: "inner-city-driving:default"},
		{Type: CommandRunAllScenes, SafetyEpoch: currentSafetyEpochPtr(store)},
		{Type: CommandStartEgo, SafetyEpoch: currentSafetyEpochPtr(store)},
		validStopSignBatchCommandRequest(currentSafetyEpochPtr(store)),
		{Type: CommandClearStopSignTarget},
	}
	for _, request := range commandsToFlush {
		mustEnqueueCommand(t, store, request)
	}

	endAll := mustEnqueueCommand(t, store, CommandRequest{Type: CommandEndAllScenes})
	stopEgo := mustEnqueueCommand(t, store, CommandRequest{Type: CommandStopEgo})

	wantOrder := []Command{endAll, stopEgo}
	pending := store.State().PendingCommands
	if len(pending) != len(wantOrder) {
		t.Fatalf("Hold must flush non-emergency pending commands, got=%+v", pending)
	}
	for index, want := range wantOrder {
		if pending[index].ID != want.ID {
			t.Fatalf("pending %d: got=%+v want=%+v", index, pending[index], want)
		}
	}

	lastSeenID := ""
	for index, want := range wantOrder {
		got := store.Poll(lastSeenID)
		if got == nil || got.ID != want.ID {
			t.Fatalf("poll %d: got=%+v want=%+v", index, got, want)
		}
		lastSeenID = got.ID
	}
	if got := store.Poll(lastSeenID); got != nil {
		t.Fatalf("non-emergency work must not execute after Hold, got=%+v", got)
	}

	state := store.State()
	if state.LastCommand == nil || state.LastCommand.ID != stopEgo.ID {
		t.Fatalf("last command must track enqueue order after priority insertion, got=%+v", state.LastCommand)
	}
}

func TestEmergencyStopInsertedAfterDeliveredUnacknowledgedStop(t *testing.T) {
	store := NewStore()
	mustEnqueueCommand(t, store, CommandRequest{Type: CommandSetStopSignTarget})
	firstStop := mustEnqueueCommand(t, store, CommandRequest{Type: CommandEndAllScenes})

	delivered := store.Poll("")
	if delivered == nil || delivered.ID != firstStop.ID {
		t.Fatalf("expected first stop to be delivered, got=%+v", delivered)
	}

	secondStop := mustEnqueueCommand(t, store, CommandRequest{Type: CommandStopEgo})
	gotSecond := store.Poll(delivered.ID)
	if gotSecond == nil || gotSecond.ID != secondStop.ID {
		t.Fatalf("second stop must remain after the delivered cursor, got=%+v want=%+v", gotSecond, secondStop)
	}
	if got := store.Poll(gotSecond.ID); got != nil {
		t.Fatalf("flushed ordinary command must not follow the stops, got=%+v", got)
	}
}

func TestEmergencyHoldCancelsDeliveredUnacknowledgedStart(t *testing.T) {
	store := NewStore()
	start := mustEnqueueCommand(t, store, CommandRequest{Type: CommandStartEgo, SafetyEpoch: currentSafetyEpochPtr(store)})

	delivered := store.Poll("")
	if delivered == nil || delivered.ID != start.ID {
		t.Fatalf("expected start to be Poll-returned, got=%+v", delivered)
	}

	stop := mustEnqueueCommand(t, store, CommandRequest{Type: CommandEndAllScenes})
	fromEmptyCursor := store.Poll("")
	if fromEmptyCursor == nil || fromEmptyCursor.ID != stop.ID {
		t.Fatalf("empty cursor must skip canceled delivered start, got=%+v want=%+v", fromEmptyCursor, stop)
	}
	fromDeliveredCursor := store.Poll(start.ID)
	if fromDeliveredCursor == nil || fromDeliveredCursor.ID != stop.ID {
		t.Fatalf("delivered start cursor must advance to stop, got=%+v want=%+v", fromDeliveredCursor, stop)
	}
}

func TestDispatchConfirmationLinearizesWithEmergencyHold(t *testing.T) {
	t.Run("confirmation before Hold may dispatch", func(t *testing.T) {
		store := NewStore()
		start := mustEnqueueCommand(t, store, CommandRequest{Type: CommandStartEgo, SafetyEpoch: currentSafetyEpochPtr(store)})
		if delivered := store.Poll(""); delivered == nil || delivered.ID != start.ID {
			t.Fatalf("expected start to be polled, got=%+v", delivered)
		}

		confirmation, err := store.ConfirmDispatch(start.ID)
		if err != nil {
			t.Fatalf("confirm start dispatch: %v", err)
		}
		if !confirmation.Confirmed || confirmation.Command == nil || confirmation.Command.ID != start.ID {
			t.Fatalf("confirmation that linearized first must permit dispatch, got=%+v", confirmation)
		}

		hold := mustEnqueueCommand(t, store, CommandRequest{Type: CommandEndAllScenes})
		if next := store.Poll(start.ID); next == nil || next.ID != hold.ID {
			t.Fatalf("confirmed start cursor must advance to the later Hold, got=%+v want=%+v", next, hold)
		}
	})

	t.Run("Hold before confirmation rejects and skips start", func(t *testing.T) {
		store := NewStore()
		start := mustEnqueueCommand(t, store, CommandRequest{Type: CommandStartEgo, SafetyEpoch: currentSafetyEpochPtr(store)})
		if delivered := store.Poll(""); delivered == nil || delivered.ID != start.ID {
			t.Fatalf("expected start to be polled, got=%+v", delivered)
		}

		hold := mustEnqueueCommand(t, store, CommandRequest{Type: CommandEndAllScenes})
		confirmation, err := store.ConfirmDispatch(start.ID)
		if err != nil {
			t.Fatalf("confirm canceled dispatch: %v", err)
		}
		if confirmation.Confirmed || confirmation.Reason != DispatchConfirmationCanceled || confirmation.Command != nil {
			t.Fatalf("Hold that linearized first must reject the canceled start, got=%+v", confirmation)
		}
		if next := store.Poll(start.ID); next == nil || next.ID != hold.ID {
			t.Fatalf("rejected start cursor must advance to the Hold, got=%+v want=%+v", next, hold)
		}
	})
}

func TestDispatchConfirmationRejectsMissingAndStaleGuardedCommands(t *testing.T) {
	store := NewStore()

	missing, err := store.ConfirmDispatch("cmd-missing")
	if err != nil {
		t.Fatalf("confirm missing command: %v", err)
	}
	if missing.Confirmed || missing.Reason != DispatchConfirmationMissing {
		t.Fatalf("missing command must be rejected, got=%+v", missing)
	}

	unoffered := mustEnqueueCommand(t, store, CommandRequest{Type: CommandSetStopSignTarget})
	unofferedConfirmation, err := store.ConfirmDispatch(unoffered.ID)
	if err != nil {
		t.Fatalf("confirm unoffered command: %v", err)
	}
	if unofferedConfirmation.Confirmed || unofferedConfirmation.Reason != DispatchConfirmationMissing {
		t.Fatalf("a command that was not offered by Poll must be rejected, got=%+v", unofferedConfirmation)
	}

	start := mustEnqueueCommand(t, store, CommandRequest{Type: CommandStartEgo, SafetyEpoch: currentSafetyEpochPtr(store)})
	if delivered := store.Poll(unoffered.ID); delivered == nil || delivered.ID != start.ID {
		t.Fatalf("expected start to be polled, got=%+v", delivered)
	}
	store.mu.Lock()
	store.advanceSafetyEpochLocked()
	store.mu.Unlock()

	stale, err := store.ConfirmDispatch(start.ID)
	if err != nil {
		t.Fatalf("confirm stale start: %v", err)
	}
	if stale.Confirmed || stale.Reason != DispatchConfirmationStaleSafetyEpoch {
		t.Fatalf("guarded start from an old safety epoch must be rejected, got=%+v", stale)
	}
}

func TestDispatchConfirmationRequiresCommandID(t *testing.T) {
	store := NewStore()
	if _, err := store.ConfirmDispatch("  "); !errors.Is(err, ErrDispatchCommandIDRequired) {
		t.Fatalf("expected missing command id error, got=%v", err)
	}
}

func TestPolledEmergencyCommandRemainsConfirmableUnderHistoryPressure(t *testing.T) {
	store := NewStore()
	store.commandHistoryLimit = 2
	stop := mustEnqueueCommand(t, store, CommandRequest{Type: CommandEndAllScenes})
	if delivered := store.Poll(""); delivered == nil || delivered.ID != stop.ID {
		t.Fatalf("expected emergency command to be offered, got=%+v", delivered)
	}

	for index := 0; index < 8; index++ {
		mustEnqueueCommand(t, store, CommandRequest{Type: CommandSetStopSignTarget})
	}
	if redelivered := store.Poll(""); redelivered == nil || redelivered.ID != stop.ID {
		t.Fatalf("failed confirmation transport must re-offer the pinned emergency command, got=%+v", redelivered)
	}

	confirmation, err := store.ConfirmDispatch(stop.ID)
	if err != nil {
		t.Fatalf("confirm offered emergency command: %v", err)
	}
	if !confirmation.Confirmed || confirmation.Command == nil || confirmation.Command.ID != stop.ID {
		t.Fatalf("history trimming removed a polled-but-unconfirmed emergency command: %+v", confirmation)
	}
}

func TestMotionStartsRequireCurrentSafetyEpoch(t *testing.T) {
	store := NewStore()

	for _, request := range guardedStartCommandRequests(nil) {
		_, err := store.Enqueue(request)
		if !errors.Is(err, ErrSafetyEpochRequired) {
			t.Fatalf("%s without epoch: got=%v want=%v", request.Type, err, ErrSafetyEpochRequired)
		}
	}

	if _, err := store.Enqueue(CommandRequest{Type: CommandSetStopSignTarget}); err != nil {
		t.Fatalf("non-start setup command must not require an epoch: %v", err)
	}
}

func TestEmergencyHoldRejectsStartsThatArriveLateAndAcceptsFreshEpoch(t *testing.T) {
	store := NewStore()
	oldEpoch := currentSafetyEpochPtr(store)

	hold := mustEnqueueCommand(t, store, CommandRequest{Type: CommandEndAllScenes})
	if store.State().SafetyEpoch <= *oldEpoch {
		t.Fatalf("Hold must advance the safety epoch: old=%d current=%d", *oldEpoch, store.State().SafetyEpoch)
	}

	for _, request := range guardedStartCommandRequests(oldEpoch) {
		_, err := store.Enqueue(request)
		if !errors.Is(err, ErrSafetyEpochMismatch) {
			t.Fatalf("late %s: got=%v want=%v", request.Type, err, ErrSafetyEpochMismatch)
		}
	}
	pending := store.State().PendingCommands
	if len(pending) != 1 || pending[0].ID != hold.ID {
		t.Fatalf("late starts must not enqueue behind Hold: %+v", pending)
	}

	freshEpoch := currentSafetyEpochPtr(store)
	for _, request := range guardedStartCommandRequests(freshEpoch) {
		if _, err := store.Enqueue(request); err != nil {
			t.Fatalf("fresh %s after Hold: %v", request.Type, err)
		}
	}
}

func TestEveryEmergencyStopAdvancesSafetyEpoch(t *testing.T) {
	store := NewStore()

	for _, commandType := range []CommandType{CommandEndScene, CommandEndAllScenes, CommandStopEgo} {
		before := store.State().SafetyEpoch
		command := mustEnqueueCommand(t, store, CommandRequest{Type: commandType})
		if after := store.State().SafetyEpoch; after != before+1 {
			t.Fatalf("%s safety epoch: got=%d want=%d", commandType, after, before+1)
		}
		if command.SafetyEpoch != before+1 {
			t.Fatalf("%s command must carry its advanced epoch: got=%d want=%d", commandType, command.SafetyEpoch, before+1)
		}
	}
}

func TestCommandsCarryAcceptedSafetyEpoch(t *testing.T) {
	store := NewStore()
	epoch := store.State().SafetyEpoch

	for _, request := range guardedStartCommandRequests(&epoch) {
		start := mustEnqueueCommand(t, store, request)
		if start.SafetyEpoch != epoch {
			t.Fatalf("%s guarded start epoch: got=%d want=%d", start.Type, start.SafetyEpoch, epoch)
		}
	}

	setup := mustEnqueueCommand(t, store, CommandRequest{Type: CommandSetStopSignTarget})
	if setup.SafetyEpoch != epoch {
		t.Fatalf("ordinary command epoch: got=%d want=%d", setup.SafetyEpoch, epoch)
	}
}

func TestStatusBarrierIgnoresDelayedLowerEpochAndSequenceUpdates(t *testing.T) {
	store := NewStore()
	mustEnqueueCommand(t, store, CommandRequest{Type: CommandEndAllScenes})
	mustEnqueueCommand(t, store, CommandRequest{Type: CommandStopEgo})
	settledEpoch := store.State().SafetyEpoch

	settled := store.UpdateStatus(StatusUpdate{
		Status:               StatusIdle,
		AppliedSafetyEpoch:   settledEpoch,
		InFlightSafetyStarts: 0,
		SafetyStatusSequence: 12,
	})
	if settled.AppliedSafetyEpoch != settledEpoch || settled.InFlightSafetyStarts != 0 || settled.Status != StatusIdle {
		t.Fatalf("expected settled consumer barrier, got=%+v", settled)
	}

	store.UpdateStatus(StatusUpdate{
		Status:               StatusRunningScene,
		ActiveSceneName:      "ego-control",
		AppliedSafetyEpoch:   settledEpoch - 2,
		InFlightSafetyStarts: 1,
		SafetyStatusSequence: 10,
	})
	store.UpdateStatus(StatusUpdate{
		Status:               StatusStopping,
		AppliedSafetyEpoch:   settledEpoch,
		InFlightSafetyStarts: 1,
		SafetyStatusSequence: 11,
	})
	store.UpdateStatus(StatusUpdate{
		Status:               StatusRunningScene,
		ActiveSceneName:      "unsequenced-stale-update",
		AppliedSafetyEpoch:   settledEpoch,
		InFlightSafetyStarts: 1,
	})

	got := store.State().Runtime
	if got.AppliedSafetyEpoch != settledEpoch || got.InFlightSafetyStarts != 0 || got.Status != StatusIdle || got.ActiveSceneName != "" {
		t.Fatalf("delayed status must not regress settled state: %+v", got)
	}
}

func TestStopSignBatchProgressUsesCanonicalPhasesAndSafetyStatusOrdering(t *testing.T) {
	store := NewStore()
	epoch := store.State().SafetyEpoch
	runtimeState := store.UpdateStatus(StatusUpdate{
		Status: StatusRunningAllScenes,
		StopSignBatch: &StopSignBatchProgress{
			BatchID: "  city-stops  ", PlanFingerprint: "  sha256:abc  ", State: " running ",
			JobID: " alta:rain ", JobIndex: 1, JobCount: 2, CompletedJobs: 1,
			AttemptIndex: 2, AttemptCount: 4, Phase: StopSignPhaseDecelerate,
			VariationProfile: &StopSignVariationProfile{
				Contract: stopsignbatch.VariationProfileContract, ConfiguredMotionVariancePct: 20,
				ChangedDimensions: []string{"target_speed"}, ChangeCount: 1, CombinationMagnitudePct: 27.735,
				TargetSpeedDeltaMPS: 5, TargetSpeedDeltaPct: 50,
			},
		},
		AppliedSafetyEpoch:   epoch,
		InFlightSafetyStarts: 1,
		SafetyStatusSequence: 4,
	})
	progress := runtimeState.StopSignBatch
	if progress == nil || progress.BatchID != "city-stops" || progress.JobID != "alta:rain" || progress.Phase != StopSignPhaseDecelerate ||
		progress.VariationProfile == nil || progress.VariationProfile.ChangeCount != 1 {
		t.Fatalf("unexpected stop-sign progress: %+v", progress)
	}

	stale := store.UpdateStatus(StatusUpdate{
		Status:               StatusIdle,
		AppliedSafetyEpoch:   epoch,
		InFlightSafetyStarts: 0,
		SafetyStatusSequence: 3,
	})
	if stale.Status != StatusRunningAllScenes || stale.StopSignBatch == nil || stale.StopSignBatch.Phase != StopSignPhaseDecelerate {
		t.Fatalf("older status sequence crossed progress frontier: %+v", stale)
	}

	reset := store.ResetConsumerSessionWithSafetyEpoch()
	if state := store.State(); state.Runtime.StopSignBatch != nil || reset.SafetyEpoch <= epoch {
		t.Fatalf("consumer reset did not clear stop-sign progress: reset=%+v state=%+v", reset, state.Runtime)
	}
}

func TestStopSignBatchProgressRejectsUnknownPhaseLabel(t *testing.T) {
	store := NewStore()
	state := store.UpdateStatus(StatusUpdate{
		Status: StatusRunningAllScenes,
		StopSignBatch: &StopSignBatchProgress{
			BatchID: "batch", State: "running", JobCount: 1, AttemptCount: 1, Phase: "approaching-ish",
		},
		AppliedSafetyEpoch:   store.State().SafetyEpoch,
		InFlightSafetyStarts: 1,
		SafetyStatusSequence: 1,
	})
	if state.StopSignBatch == nil || state.StopSignBatch.Phase != "" {
		t.Fatalf("unknown phase must not enter runtime state: %+v", state.StopSignBatch)
	}
}

func TestStatusSequenceRestartsWhenConsumerAppliesANewerEpoch(t *testing.T) {
	store := NewStore()
	oldEpoch := store.State().SafetyEpoch
	store.UpdateStatus(StatusUpdate{
		Status:               StatusRunningScene,
		AppliedSafetyEpoch:   oldEpoch,
		InFlightSafetyStarts: 1,
		SafetyStatusSequence: 100,
	})

	store.ResetConsumerSession()
	newEpoch := store.State().SafetyEpoch
	store.UpdateStatus(StatusUpdate{
		Status:               StatusStopping,
		AppliedSafetyEpoch:   oldEpoch,
		InFlightSafetyStarts: 1,
		SafetyStatusSequence: 101,
	})
	settled := store.UpdateStatus(StatusUpdate{
		Status:               StatusIdle,
		AppliedSafetyEpoch:   newEpoch,
		InFlightSafetyStarts: 0,
		SafetyStatusSequence: 1,
	})

	if settled.AppliedSafetyEpoch != newEpoch || settled.InFlightSafetyStarts != 0 || settled.Status != StatusIdle {
		t.Fatalf("new epoch must replace the prior epoch's sequence domain: %+v", settled)
	}
	store.UpdateStatus(StatusUpdate{
		Status:               StatusRunningScene,
		AppliedSafetyEpoch:   oldEpoch,
		InFlightSafetyStarts: 1,
		SafetyStatusSequence: 102,
	})
	if got := store.State().Runtime; got.AppliedSafetyEpoch != newEpoch || got.InFlightSafetyStarts != 0 || got.Status != StatusIdle {
		t.Fatalf("late pre-reset status must not regress the new consumer epoch: %+v", got)
	}
}

func TestConsumerResetInvalidatesPreviouslyIssuedSafetyEpoch(t *testing.T) {
	store := NewStore()
	oldEpoch := currentSafetyEpochPtr(store)

	store.ResetConsumerSession()
	if _, err := store.Enqueue(CommandRequest{Type: CommandStartEgo, SafetyEpoch: oldEpoch}); !errors.Is(err, ErrSafetyEpochMismatch) {
		t.Fatalf("start using pre-reset epoch: got=%v want=%v", err, ErrSafetyEpochMismatch)
	}
	if _, err := store.Enqueue(CommandRequest{Type: CommandStartEgo, SafetyEpoch: currentSafetyEpochPtr(store)}); err != nil {
		t.Fatalf("start using post-reset epoch: %v", err)
	}
}

func TestStateTracksPendingCommandsAndConnectivity(t *testing.T) {
	now := time.Date(2026, 4, 11, 2, 0, 0, 0, time.UTC)
	store := NewStore(WithNowFunc(func() time.Time { return now }))

	first, err := store.Enqueue(CommandRequest{Type: CommandSetStopSignTarget})
	if err != nil {
		t.Fatalf("enqueue first: %v", err)
	}
	_, err = store.Enqueue(CommandRequest{Type: CommandClearStopSignTarget})
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
	if state.PendingCommands[0].Type != CommandClearStopSignTarget {
		t.Fatalf("expected pending clearStopSignTarget, got %s", state.PendingCommands[0].Type)
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

	signPose := &StopSignPose{X: 10, Y: 20, Z: 3, Heading: 90}
	telemetry := store.UpdateTelemetry(TelemetryUpdate{
		CurrentSpeed:        4.25,
		CurrentYaw:          182.5,
		RouteForwardDelta:   0.75,
		RouteHeadingError:   -8.0,
		RouteDistance:       32.0,
		LeadVehicleDistance: 18.5,
		HasLeadVehicle:      true,
		TimestampMs:         123456,
		StopSignPose:        signPose,
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
	if telemetry.StopSignPose == nil || telemetry.StopSignPose.Heading != 90 {
		t.Fatalf("expected exact stop-sign pose: %+v", telemetry.StopSignPose)
	}
	signPose.X = 999
	if telemetry.StopSignPose.X != 10 {
		t.Fatal("telemetry must clone the caller's pose")
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

func TestUpdateTelemetryCopiesStopSignTemporalStateAndGeometry(t *testing.T) {
	store := NewStore()
	sign := &StopSignPose{X: 100, Y: 200, Z: 8, Heading: 90}
	line := &StopSignPose{X: 104, Y: 200, Z: 8, Heading: 90}
	egoStop := &StopSignPose{X: 106.5, Y: 200, Z: 8, Heading: 90}

	updated := store.UpdateTelemetry(TelemetryUpdate{
		StopSignTargetConfigured:      true,
		StopSignPose:                  sign,
		StopLinePose:                  line,
		StopSignEgoStopPose:           egoStop,
		StopSignDistanceM:             6.5,
		StopLineDistanceM:             0.4,
		StopSignLongitudinalErrorM:    -0.3,
		StopSignLateralErrorM:         0.1,
		StopSignHeadingErrorDeg:       -1.5,
		StopSignConfirmationElapsedMS: 200,
		StopSignConfirmationTargetMS:  250,
		StopSignStopped:               true,
		StopSignAttemptIndex:          2,
		StopSignAttemptCount:          4,
		StopSignPhase:                 StopSignPhaseStopHold,
	})
	if !updated.StopSignTargetConfigured || updated.StopSignPhase != StopSignPhaseStopHold || !updated.StopSignStopped {
		t.Fatalf("unexpected stop-sign temporal telemetry: %+v", updated)
	}
	if updated.StopSignPose == nil || updated.StopLinePose == nil || updated.StopSignEgoStopPose == nil {
		t.Fatalf("expected all canonical stop-sign poses: %+v", updated)
	}
	if updated.StopSignConfirmationElapsedMS != 200 || updated.StopSignConfirmationTargetMS != 250 {
		t.Fatalf("unexpected stop-confirmation telemetry: %+v", updated)
	}
	sign.X = -999
	updated.StopLinePose.X = -999
	state := store.State().Telemetry
	if state == nil || state.StopSignPose == nil || state.StopSignPose.X != 100 || state.StopLinePose == nil || state.StopLinePose.X != 104 {
		t.Fatalf("stop-sign poses must be independent snapshots: %+v", state)
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
		VehicleModelHash:      424242,
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
	if snapshot.VehicleModelHash != 424242 {
		t.Fatalf("expected vehicle identity in normalized telemetry, got=%d", snapshot.VehicleModelHash)
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
	if ego.SteeringActual == nil || math.Abs(*ego.SteeringActual-0.5) > 1e-9 {
		t.Fatalf("expected measured steering in actuator ego state, got=%+v", ego.SteeringActual)
	}
	if ego.VehicleModelHash != 424242 {
		t.Fatalf("expected vehicle identity in actuator ego state, got=%d", ego.VehicleModelHash)
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

	if _, err := store.Enqueue(CommandRequest{Type: CommandStartScene, SafetyEpoch: currentSafetyEpochPtr(store), SceneName: "inner-city-driving:default"}); err != nil {
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

func TestResetConsumerSessionReturnsTheAdvancedSafetyEpochAtomically(t *testing.T) {
	store := NewStore()
	previousEpoch := store.State().SafetyEpoch

	reset := store.ResetConsumerSessionWithSafetyEpoch()

	if reset.SessionID == "" {
		t.Fatal("expected reset to return a session id")
	}
	if reset.SafetyEpoch != previousEpoch+1 {
		t.Fatalf("expected reset epoch %d, got %d", previousEpoch+1, reset.SafetyEpoch)
	}
	if stateEpoch := store.State().SafetyEpoch; stateEpoch != reset.SafetyEpoch {
		t.Fatalf("reset response and state crossed a safety frontier: response=%d state=%d", reset.SafetyEpoch, stateEpoch)
	}
}

func mustEnqueueCommand(t *testing.T, store *Store, request CommandRequest) Command {
	t.Helper()
	command, err := store.Enqueue(request)
	if err != nil {
		t.Fatalf("enqueue %s: %v", request.Type, err)
	}
	return command
}

func currentSafetyEpochPtr(store *Store) *uint64 {
	epoch := store.State().SafetyEpoch
	return &epoch
}

func guardedStartCommandRequests(safetyEpoch *uint64) []CommandRequest {
	return []CommandRequest{
		{Type: CommandStartScene, SafetyEpoch: safetyEpoch, SceneName: "inner-city-driving:default"},
		{Type: CommandRunAllScenes, SafetyEpoch: safetyEpoch},
		{Type: CommandStartEgo, SafetyEpoch: safetyEpoch},
		{Type: CommandTeleportStopSignStart, SafetyEpoch: safetyEpoch, StopSignStartPose: &StopSignPose{X: 1, Y: 2, Z: 3, Heading: 90}},
		validStopSignProbeCommandRequest(safetyEpoch),
		validStopSignBatchCommandRequest(safetyEpoch),
	}
}

func validStopSignProbeCommandRequest(safetyEpoch *uint64) CommandRequest {
	return CommandRequest{
		Type:        CommandProbeStopSignTarget,
		SafetyEpoch: safetyEpoch,
		StopSignProbe: &StopSignProbe{
			CatalogID:       "gta-v-sign-0022",
			CatalogPosition: WorldPosition{X: -1832.1594, Y: 145.45688, Z: 77.10415},
		},
	}
}

func validStopSignBatchCommandRequest(safetyEpoch *uint64) CommandRequest {
	return CommandRequest{
		Type:            CommandStartStopSignBatch,
		SafetyEpoch:     safetyEpoch,
		StopSignBatchID: "city-stops",
		PlanFingerprint: "sha256:72f8a2a4127d2c515e46c6bd9f6543a2ecfdf8a16df376cf231428edd459fac5",
		StopSignJobs: []StopSignBatchJob{{
			ID:                 "alta:base",
			EntryID:            "alta",
			VariationID:        "base",
			SignPose:           StopSignPose{X: 10, Y: 20, Z: 2, Heading: 0},
			StopLinePose:       StopSignPose{X: 10, Y: 17, Z: 2, Heading: 0},
			EgoStopPose:        StopSignPose{X: 10, Y: 14.5, Z: 2, Heading: 0},
			StartPose:          StopSignPose{X: 10, Y: -25.5, Z: 2, Heading: 0},
			ExitPose:           StopSignPose{X: 10, Y: 28, Z: 2, Heading: 0},
			StopDistanceM:      3,
			EgoCenterOffsetM:   2.5,
			StartDistanceM:     40,
			ExitDistanceM:      8,
			TargetSpeedMPS:     8,
			StopConfirmationMS: 250,
			AttemptCount:       3,
			Weather:            "EXTRASUNNY",
			Time:               StopSignTime{Hour: 12, Minute: 0},
			Vehicle: StopSignVehicle{
				Model: "sultan",
				Color: &StopSignColor{R: 255, G: 128, B: 64},
			},
			Seed: "fresh:alta:base",
			VariationProfile: StopSignVariationProfile{
				Contract: stopsignbatch.VariationProfileContract,
				Baseline: true,
			},
		}},
	}
}
