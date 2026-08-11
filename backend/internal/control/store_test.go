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

	for _, commandType := range []CommandType{
		CommandSetParkingTarget,
		CommandSetParkingStart,
		CommandClearParkingTarget,
	} {
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
		Type:        CommandPrepareParkingEvaluation,
		SafetyEpoch: currentSafetyEpochPtr(store),
		Seed:        "  evaluation-seed-7  ",
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
		SafetyEpoch:  currentSafetyEpochPtr(store),
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
			SafetyEpoch:  currentSafetyEpochPtr(store),
			AttemptCount: attemptCount,
		}); err != nil {
			t.Fatalf("expected attempt count %d to be valid: %v", attemptCount, err)
		}
	}
}

func TestEnqueueStartParkingBatchRequiresPlanFingerprint(t *testing.T) {
	store := NewStore()
	request := validParkingBatchCommandRequest(currentSafetyEpochPtr(store))
	request.PlanFingerprint = ""
	if _, err := store.Enqueue(request); !errors.Is(err, ErrInvalidCommand) {
		t.Fatalf("expected missing plan fingerprint to be invalid, got=%v", err)
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

	first, err := store.Enqueue(CommandRequest{Type: CommandSetParkingTarget})
	if err != nil {
		t.Fatalf("enqueue first: %v", err)
	}
	second, err := store.Enqueue(CommandRequest{Type: CommandClearParkingTarget})
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
		{Type: CommandSetParkingTarget},
		{Type: CommandStartScene, SafetyEpoch: currentSafetyEpochPtr(store), SceneName: "inner-city-driving:default"},
		{Type: CommandRunAllScenes, SafetyEpoch: currentSafetyEpochPtr(store)},
		{Type: CommandStartEgo, SafetyEpoch: currentSafetyEpochPtr(store)},
		{Type: CommandSetParkingStart},
		{Type: CommandPrepareParkingEvaluation, SafetyEpoch: currentSafetyEpochPtr(store), Seed: "evaluation-1"},
		{Type: CommandStartParkingRun, SafetyEpoch: currentSafetyEpochPtr(store), AttemptCount: 3, Seed: "run-1"},
		validParkingBatchCommandRequest(currentSafetyEpochPtr(store)),
		{Type: CommandClearParkingTarget},
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
	mustEnqueueCommand(t, store, CommandRequest{Type: CommandSetParkingTarget})
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

	unoffered := mustEnqueueCommand(t, store, CommandRequest{Type: CommandSetParkingTarget})
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
		mustEnqueueCommand(t, store, CommandRequest{Type: CommandSetParkingTarget})
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

	if _, err := store.Enqueue(CommandRequest{Type: CommandSetParkingTarget}); err != nil {
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

	setup := mustEnqueueCommand(t, store, CommandRequest{Type: CommandSetParkingTarget})
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
		},
		AppliedSafetyEpoch:   epoch,
		InFlightSafetyStarts: 1,
		SafetyStatusSequence: 4,
	})
	progress := runtimeState.StopSignBatch
	if progress == nil || progress.BatchID != "city-stops" || progress.JobID != "alta:rain" || progress.Phase != StopSignPhaseDecelerate {
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

	first, err := store.Enqueue(CommandRequest{Type: CommandSetParkingTarget})
	if err != nil {
		t.Fatalf("enqueue first: %v", err)
	}
	_, err = store.Enqueue(CommandRequest{Type: CommandClearParkingTarget})
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
	if state.PendingCommands[0].Type != CommandClearParkingTarget {
		t.Fatalf("expected pending clearParkingTarget, got %s", state.PendingCommands[0].Type)
	}

	now = now.Add(11 * time.Second)
	state = store.State()
	if state.Runtime.FiveMConnected {
		t.Fatal("expected FiveMConnected to become false after poll timeout")
	}
}

func TestStateTracksParkingBatchProgress(t *testing.T) {
	store := NewStore()
	state := store.UpdateStatus(StatusUpdate{
		Status:          StatusRunningAllScenes,
		ActiveSceneName: "parking-batch:morning-lot",
		ParkingBatch: &ParkingBatchProgress{
			BatchID:         " morning-lot ",
			PlanFingerprint: " sha256:plan-exact ",
			State:           "running",
			JobID:           " bay-2:right ",
			JobIndex:        2,
			JobCount:        6,
			CompletedJobs:   1,
			StartedAtMs:     100,
			UpdatedAtMs:     200,
		},
	})

	if state.ParkingBatch == nil {
		t.Fatal("expected parking batch progress")
	}
	if state.ParkingBatch.BatchID != "morning-lot" || state.ParkingBatch.PlanFingerprint != "sha256:plan-exact" || state.ParkingBatch.JobID != "bay-2:right" {
		t.Fatalf("expected normalized progress, got=%+v", state.ParkingBatch)
	}
	if state.ParkingBatch.JobIndex != 2 || state.ParkingBatch.CompletedJobs != 1 {
		t.Fatalf("unexpected progress counters: %+v", state.ParkingBatch)
	}

	store.UpdateStatus(StatusUpdate{Status: StatusIdle})
	if got := store.State().Runtime.ParkingBatch; got != nil {
		t.Fatalf("idle transition should clear stale batch progress: %+v", got)
	}

	store.UpdateStatus(StatusUpdate{
		Status: StatusRunningScene,
		ParkingBatch: &ParkingBatchProgress{
			BatchID:         "morning-lot",
			PlanFingerprint: "sha256:completed-plan",
			State:           "completed",
			JobIndex:        6,
			JobCount:        6,
			CompletedJobs:   6,
		},
	})
	store.UpdateStatus(StatusUpdate{Status: StatusIdle})
	if got := store.State().Runtime.ParkingBatch; got == nil || got.State != "completed" || got.PlanFingerprint != "sha256:completed-plan" {
		t.Fatalf("terminal batch evidence must survive later idle status, got=%+v", got)
	}
}

func TestUpdateTelemetryExposesLatestSpeedSnapshot(t *testing.T) {
	now := time.Date(2026, 4, 18, 1, 2, 3, 0, time.UTC)
	store := NewStore(WithNowFunc(func() time.Time { return now }))

	targetPose := &ParkingPose{X: 10, Y: 20, Z: 3, Heading: 90}
	startPose := &ParkingPose{X: 21, Y: 20, Z: 3, Heading: 90}
	telemetry := store.UpdateTelemetry(TelemetryUpdate{
		CurrentSpeed:        4.25,
		CurrentYaw:          182.5,
		RouteForwardDelta:   0.75,
		RouteHeadingError:   -8.0,
		RouteDistance:       32.0,
		LeadVehicleDistance: 18.5,
		HasLeadVehicle:      true,
		TimestampMs:         123456,
		ParkingTargetPose:   targetPose,
		ParkingStartPose:    startPose,
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
	if telemetry.ParkingTargetPose == nil || telemetry.ParkingTargetPose.Heading != 90 {
		t.Fatalf("expected exact parking target pose: %+v", telemetry.ParkingTargetPose)
	}
	if telemetry.ParkingStartPose == nil || telemetry.ParkingStartPose.X != 21 {
		t.Fatalf("expected exact parking start pose: %+v", telemetry.ParkingStartPose)
	}
	targetPose.X = 999
	if telemetry.ParkingTargetPose.X != 10 {
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

func TestUpdateTelemetryCopiesParkingStateAcrossSnapshots(t *testing.T) {
	now := time.Date(2026, 8, 4, 12, 0, 0, 0, time.UTC)
	store := NewStore(WithNowFunc(func() time.Time { return now }))
	positionX := 42.5

	updated := store.UpdateTelemetry(TelemetryUpdate{
		PositionX:                &positionX,
		ParkingTargetConfigured:  true,
		ParkingStartConfigured:   true,
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

func TestUpdateTelemetryCopiesStopSignTemporalStateAndGeometry(t *testing.T) {
	store := NewStore()
	sign := &StopSignPose{X: 100, Y: 200, Z: 8, Heading: 90}
	line := &StopSignPose{X: 104, Y: 200, Z: 8, Heading: 90}
	egoStop := &StopSignPose{X: 106.5, Y: 200, Z: 8, Heading: 90}

	updated := store.UpdateTelemetry(TelemetryUpdate{
		StopSignTargetConfigured:   true,
		StopSignPose:               sign,
		StopLinePose:               line,
		StopSignEgoStopPose:        egoStop,
		StopSignDistanceM:          6.5,
		StopLineDistanceM:          0.4,
		StopSignLongitudinalErrorM: -0.3,
		StopSignLateralErrorM:      0.1,
		StopSignHeadingErrorDeg:    -1.5,
		StopSignDwellElapsedMS:     3200,
		StopSignDwellTargetMS:      5000,
		StopSignStopped:            true,
		StopSignAttemptIndex:       2,
		StopSignAttemptCount:       4,
		StopSignPhase:              StopSignPhaseStopHold,
	})
	if !updated.StopSignTargetConfigured || updated.StopSignPhase != StopSignPhaseStopHold || !updated.StopSignStopped {
		t.Fatalf("unexpected stop-sign temporal telemetry: %+v", updated)
	}
	if updated.StopSignPose == nil || updated.StopLinePose == nil || updated.StopSignEgoStopPose == nil {
		t.Fatalf("expected all canonical stop-sign poses: %+v", updated)
	}
	if updated.StopSignDwellElapsedMS != 3200 || updated.StopSignDwellTargetMS != 5000 {
		t.Fatalf("unexpected dwell telemetry: %+v", updated)
	}
	sign.X = -999
	updated.StopLinePose.X = -999
	state := store.State().Telemetry
	if state == nil || state.StopSignPose == nil || state.StopSignPose.X != 100 || state.StopLinePose == nil || state.StopLinePose.X != 104 {
		t.Fatalf("stop-sign poses must be independent snapshots: %+v", state)
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
	if !telemetry.ParkingStartConfigured {
		t.Fatalf("expected configured parking start: %+v", telemetry)
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
		{Type: CommandPrepareParkingEvaluation, SafetyEpoch: safetyEpoch, Seed: "evaluation"},
		{Type: CommandStartParkingRun, SafetyEpoch: safetyEpoch, AttemptCount: 2, Seed: "parking-run"},
		validParkingBatchCommandRequest(safetyEpoch),
		validStopSignBatchCommandRequest(safetyEpoch),
	}
}

func validStopSignBatchCommandRequest(safetyEpoch *uint64) CommandRequest {
	return CommandRequest{
		Type:            CommandStartStopSignBatch,
		SafetyEpoch:     safetyEpoch,
		StopSignBatchID: "city-stops",
		PlanFingerprint: "sha256:72f8a2a4127d2c515e46c6bd9f6543a2ecfdf8a16df376cf231428edd459fac5",
		StopSignJobs: []StopSignBatchJob{{
			ID:               "alta:base",
			EntryID:          "alta",
			VariationID:      "base",
			SignPose:         StopSignPose{X: 10, Y: 20, Z: 2, Heading: 0},
			StopLinePose:     StopSignPose{X: 10, Y: 17, Z: 2, Heading: 0},
			EgoStopPose:      StopSignPose{X: 10, Y: 14.5, Z: 2, Heading: 0},
			StartPose:        StopSignPose{X: 10, Y: -25.5, Z: 2, Heading: 0},
			StopDistanceM:    3,
			EgoCenterOffsetM: 2.5,
			StartDistanceM:   40,
			TargetSpeedMPS:   8,
			DwellMS:          5000,
			AttemptCount:     3,
			Weather:          "EXTRASUNNY",
			Time:             StopSignTime{Hour: 12, Minute: 0},
			Vehicle: StopSignVehicle{
				Model: "sultan",
				Color: &StopSignColor{R: 255, G: 128, B: 64},
			},
			Seed: "fresh:alta:base",
		}},
	}
}

func validParkingBatchCommandRequest(safetyEpoch *uint64) CommandRequest {
	return CommandRequest{
		Type:            CommandStartParkingBatch,
		SafetyEpoch:     safetyEpoch,
		ParkingBatchID:  "hold-barrier-batch",
		PlanFingerprint: "sha256:375c329e8c5f35f94222cd3e2307e6e4d6b9fc7b113055acfdbf5a232b202203",
		ParkingJobs: []ParkingBatchJob{{
			ID:               "bay-1:base",
			CollectionAmount: 1,
			Seed:             "hold-barrier-batch:bay-1:base",
		}},
	}
}
