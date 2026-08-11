package dataset

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

type processingHarness struct {
	mu         sync.Mutex
	active     int
	maxActive  int
	calls      map[string]int
	entered    chan string
	release    chan struct{}
	processErr error
}

func newProcessingHarness(block bool) *processingHarness {
	harness := &processingHarness{
		calls:   make(map[string]int),
		entered: make(chan string, 32),
	}
	if block {
		harness.release = make(chan struct{})
	}
	return harness
}

func (h *processingHarness) factory() TripProcessor {
	return &harnessTripProcessor{harness: h}
}

func (h *processingHarness) callCount(tripDir string) int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.calls[tripDir]
}

func (h *processingHarness) peakConcurrency() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.maxActive
}

type harnessTripProcessor struct {
	harness *processingHarness
}

func (p *harnessTripProcessor) Queue(tripDir string) (string, error) {
	statusPath := filepath.Join(tripDir, "processing.json")
	return statusPath, writeStatusFile(statusPath, ProcessingStatus{State: "queued"})
}

func (p *harnessTripProcessor) ProcessTrip(ctx context.Context, tripDir string) error {
	h := p.harness
	h.mu.Lock()
	h.active++
	if h.active > h.maxActive {
		h.maxActive = h.active
	}
	h.calls[tripDir]++
	h.mu.Unlock()

	select {
	case h.entered <- tripDir:
	default:
	}

	var err error
	if h.release != nil {
		select {
		case <-ctx.Done():
			err = ctx.Err()
		case <-h.release:
		}
	}
	if err == nil {
		err = h.processErr
	}

	h.mu.Lock()
	h.active--
	h.mu.Unlock()
	state := "completed"
	errorText := ""
	if err != nil {
		state = "failed"
		errorText = err.Error()
	}
	_ = writeStatusFile(filepath.Join(tripDir, "processing.json"), ProcessingStatus{
		State: state,
		Error: errorText,
	})
	return err
}

func TestProcessingServiceBoundsConcurrencyAndSuppressesDuplicates(t *testing.T) {
	runsRoot := t.TempDir()
	tripDirs := make([]string, 0, 5)
	for index := 0; index < 5; index++ {
		tripDirs = append(tripDirs, createReadyProcessingTrip(t, runsRoot, index, ""))
	}
	harness := newProcessingHarness(true)
	service, err := NewProcessingService(ProcessingServiceConfig{
		Workers:               2,
		ReadinessTimeout:      time.Second,
		ReadinessPollInterval: time.Millisecond,
	}, harness.factory)
	if err != nil {
		t.Fatalf("NewProcessingService: %v", err)
	}
	if err := service.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}
	t.Cleanup(func() { _ = service.Close() })

	for _, tripDir := range tripDirs {
		accepted, enqueueErr := service.Enqueue(tripDir)
		if enqueueErr != nil || !accepted {
			t.Fatalf("enqueue %s: accepted=%t err=%v", tripDir, accepted, enqueueErr)
		}
	}
	accepted, err := service.Enqueue(tripDirs[0])
	if err != nil {
		t.Fatalf("duplicate enqueue: %v", err)
	}
	if accepted {
		t.Fatal("duplicate queued/active trip must be suppressed")
	}

	for entered := 0; entered < 2; entered++ {
		select {
		case <-harness.entered:
		case <-time.After(time.Second):
			t.Fatal("workers did not begin processing")
		}
	}
	snapshot := service.Snapshot()
	if len(snapshot.Active) != 2 || len(snapshot.Queued) != 3 {
		t.Fatalf("unexpected bounded snapshot: %+v", snapshot)
	}
	close(harness.release)
	waitCtx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := service.WaitIdle(waitCtx); err != nil {
		t.Fatalf("WaitIdle: %v", err)
	}
	snapshot = service.Snapshot()
	if len(snapshot.Completed) != len(tripDirs) || len(snapshot.Failed) != 0 {
		t.Fatalf("unexpected terminal snapshot: %+v", snapshot)
	}
	if harness.peakConcurrency() != 2 {
		t.Fatalf("processing exceeded or failed to use worker bound: peak=%d", harness.peakConcurrency())
	}
	for _, tripDir := range tripDirs {
		if count := harness.callCount(tripDir); count != 1 {
			t.Fatalf("trip processed %d times: %s", count, tripDir)
		}
	}
}

func TestProcessingServicePersistsQueuedIntentAndRejectsQueueOverflow(t *testing.T) {
	runsRoot := t.TempDir()
	activeTrip := createReadyProcessingTrip(t, runsRoot, 0, "")
	queuedTrip := createIncompleteProcessingTrip(t, runsRoot, 1)
	overflowTrip := createReadyProcessingTrip(t, runsRoot, 2, "")
	harness := newProcessingHarness(true)
	service, err := NewProcessingService(ProcessingServiceConfig{
		Workers:               1,
		QueueCapacity:         1,
		ReadinessTimeout:      time.Second,
		ReadinessPollInterval: time.Millisecond,
	}, harness.factory)
	if err != nil {
		t.Fatalf("NewProcessingService: %v", err)
	}
	if err := service.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}
	t.Cleanup(func() { _ = service.Close() })

	if accepted, err := service.Enqueue(activeTrip); err != nil || !accepted {
		t.Fatalf("enqueue active trip: accepted=%t err=%v", accepted, err)
	}
	select {
	case <-harness.entered:
	case <-time.After(time.Second):
		t.Fatal("worker did not begin processing")
	}
	if accepted, err := service.Enqueue(queuedTrip); err != nil || !accepted {
		t.Fatalf("enqueue pending trip: accepted=%t err=%v", accepted, err)
	}
	status, err := ReadStatusFile(filepath.Join(queuedTrip, "processing.json"))
	if err != nil {
		t.Fatalf("read durable queued status: %v", err)
	}
	if status.State != "queued" {
		t.Fatalf("enqueue must durably record queued intent before returning: %+v", status)
	}
	accepted, err := service.Enqueue(overflowTrip)
	if accepted || !errors.Is(err, ErrProcessingQueueFull) {
		t.Fatalf("overflow enqueue: accepted=%t err=%v", accepted, err)
	}
	if _, err := os.Stat(filepath.Join(overflowTrip, "processing.json")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("rejected enqueue must not persist queued state: %v", err)
	}
	if snapshot := service.Snapshot(); snapshot.QueueCapacity != 1 || len(snapshot.Queued) != 1 {
		t.Fatalf("unexpected bounded queue snapshot: %+v", snapshot)
	}

	close(harness.release)
	finishReadyProcessingTrip(t, queuedTrip, 1)
	waitCtx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := service.WaitIdle(waitCtx); err != nil {
		t.Fatalf("WaitIdle: %v", err)
	}
}

func TestProcessingServiceRecoversQueuedAndRunningTrips(t *testing.T) {
	runsRoot := t.TempDir()
	queuedTrip := createReadyProcessingTrip(t, runsRoot, 0, "queued")
	runningTrip := createReadyProcessingTrip(t, runsRoot, 1, "running")
	completedTrip := createReadyProcessingTrip(t, runsRoot, 2, "completed")
	staleWorkspace := filepath.Join(queuedTrip, processingWorkspacePrefix+"stale")
	if err := os.MkdirAll(staleWorkspace, 0o755); err != nil {
		t.Fatalf("mkdir stale workspace: %v", err)
	}
	harness := newProcessingHarness(false)
	service, err := NewProcessingService(ProcessingServiceConfig{
		Workers:               2,
		RunsRoot:              runsRoot,
		ReadinessTimeout:      time.Second,
		ReadinessPollInterval: time.Millisecond,
	}, harness.factory)
	if err != nil {
		t.Fatalf("NewProcessingService: %v", err)
	}
	if err := service.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}
	t.Cleanup(func() { _ = service.Close() })
	waitCtx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := service.WaitIdle(waitCtx); err != nil {
		t.Fatalf("WaitIdle: %v", err)
	}

	snapshot := service.Snapshot()
	if snapshot.Recovered != 2 || len(snapshot.Completed) != 2 {
		t.Fatalf("unexpected recovery snapshot: %+v", snapshot)
	}
	if harness.callCount(queuedTrip) != 1 || harness.callCount(runningTrip) != 1 {
		t.Fatalf("recovered trips were not processed exactly once: calls=%v", harness.calls)
	}
	if harness.callCount(completedTrip) != 0 {
		t.Fatal("completed trip must not be recovered")
	}
	if _, err := os.Stat(staleWorkspace); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("startup recovery must remove stale processing workspace: %v", err)
	}
}

func TestProcessingServiceRecoveryPredicateLeavesExcludedTripUntouched(t *testing.T) {
	runsRoot := t.TempDir()
	parkingTrip := createReadyProcessingTrip(t, runsRoot, 0, "queued")
	parkingSceneDir := filepath.Join(runsRoot, "run-a", "parking-forward-bay_default")
	if err := os.Rename(filepath.Dir(parkingTrip), parkingSceneDir); err != nil {
		t.Fatalf("move parking fixture into parking scene: %v", err)
	}
	parkingTrip = filepath.Join(parkingSceneDir, filepath.Base(parkingTrip))

	legacyTrip := createReadyProcessingTrip(t, runsRoot, 1, "queued")
	legacyWorkspace := filepath.Join(legacyTrip, processingWorkspacePrefix+"untouched")
	if err := os.MkdirAll(legacyWorkspace, 0o755); err != nil {
		t.Fatalf("mkdir excluded stale workspace: %v", err)
	}

	harness := newProcessingHarness(false)
	service, err := NewProcessingService(ProcessingServiceConfig{
		Workers:               1,
		RunsRoot:              runsRoot,
		ReadinessTimeout:      time.Second,
		ReadinessPollInterval: time.Millisecond,
		RecoveryTripPredicate: func(tripDir string) bool {
			sceneDir := filepath.Base(filepath.Dir(filepath.Clean(tripDir)))
			return sceneDir == "parking-forward-bay_default"
		},
	}, harness.factory)
	if err != nil {
		t.Fatalf("NewProcessingService: %v", err)
	}
	if err := service.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}
	t.Cleanup(func() { _ = service.Close() })
	waitCtx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := service.WaitIdle(waitCtx); err != nil {
		t.Fatalf("WaitIdle: %v", err)
	}

	if harness.callCount(parkingTrip) != 1 {
		t.Fatalf("included parking trip was not recovered once: %d", harness.callCount(parkingTrip))
	}
	if harness.callCount(legacyTrip) != 0 {
		t.Fatalf("excluded legacy trip was recovered: %d", harness.callCount(legacyTrip))
	}
	status, err := ReadStatusFile(filepath.Join(legacyTrip, "processing.json"))
	if err != nil || status.State != "queued" {
		t.Fatalf("excluded legacy status changed: status=%+v err=%v", status, err)
	}
	if !pathExists(legacyWorkspace) {
		t.Fatal("excluded legacy workspace was cleaned before recovery predicate admission")
	}
}

func TestProcessingServiceKeepsRecoveryQueueWithinCapacity(t *testing.T) {
	runsRoot := t.TempDir()
	tripDirs := make([]string, 0, 4)
	for index := 0; index < 4; index++ {
		tripDirs = append(tripDirs, createReadyProcessingTrip(t, runsRoot, index, "queued"))
	}
	harness := newProcessingHarness(true)
	service, err := NewProcessingService(ProcessingServiceConfig{
		Workers:               2,
		QueueCapacity:         1,
		RunsRoot:              runsRoot,
		ReadinessTimeout:      time.Second,
		ReadinessPollInterval: time.Millisecond,
	}, harness.factory)
	if err != nil {
		t.Fatalf("NewProcessingService: %v", err)
	}
	if err := service.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}
	t.Cleanup(func() { _ = service.Close() })

	for entered := 0; entered < 2; entered++ {
		select {
		case <-harness.entered:
		case <-time.After(time.Second):
			t.Fatal("bounded recovery did not refill an available worker slot")
		}
	}
	if snapshot := service.Snapshot(); snapshot.Recovered != len(tripDirs) || len(snapshot.Queued) > 1 {
		t.Fatalf("recovery exceeded configured queue capacity: %+v", snapshot)
	}

	close(harness.release)
	waitCtx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := service.WaitIdle(waitCtx); err != nil {
		t.Fatalf("WaitIdle: %v", err)
	}
	for _, tripDir := range tripDirs {
		if count := harness.callCount(tripDir); count != 1 {
			t.Fatalf("recovered trip processed %d times: %s", count, tripDir)
		}
	}
}

func TestProcessingServiceRetriesBusyRecoveryWithoutRestart(t *testing.T) {
	runsRoot := t.TempDir()
	tripDir := createReadyProcessingTrip(t, runsRoot, 0, "queued")
	activeWorkspace := filepath.Join(tripDir, processingWorkspacePrefix+"active-owner")
	if err := os.MkdirAll(activeWorkspace, 0o755); err != nil {
		t.Fatalf("mkdir active owner workspace: %v", err)
	}
	lock, err := acquireTripProcessingLock(tripDir)
	if err != nil {
		t.Fatalf("acquire competing trip lock: %v", err)
	}
	if _, err := NewProcessor().Queue(tripDir); !errors.Is(err, ErrTripProcessingLocked) {
		t.Fatalf("second processor did not receive typed busy error: %v", err)
	}
	harness := newProcessingHarness(false)
	service, err := NewProcessingService(ProcessingServiceConfig{
		Workers:               1,
		QueueCapacity:         1,
		RunsRoot:              runsRoot,
		ReadinessTimeout:      time.Second,
		ReadinessPollInterval: 5 * time.Millisecond,
	}, harness.factory)
	if err != nil {
		t.Fatalf("NewProcessingService: %v", err)
	}
	if err := service.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}
	t.Cleanup(func() { _ = service.Close() })
	select {
	case <-harness.entered:
		t.Fatal("service processed a trip while another process held its lock")
	case <-time.After(25 * time.Millisecond):
	}
	if !pathExists(activeWorkspace) {
		t.Fatal("startup recovery removed another process's active workspace")
	}
	if err := lock.Release(); err != nil {
		t.Fatalf("release competing trip lock: %v", err)
	}

	waitCtx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := service.WaitIdle(waitCtx); err != nil {
		t.Fatalf("WaitIdle after lock release: %v", err)
	}
	if harness.callCount(tripDir) != 1 {
		t.Fatalf("busy queued trip was not recovered without restart: calls=%d", harness.callCount(tripDir))
	}
	if snapshot := service.Snapshot(); snapshot.Recovered != 1 {
		t.Fatalf("late recovery was not reflected in snapshot: %+v", snapshot)
	}
}

func TestProcessingServiceSurfacesAndRetriesRecoveryScanErrors(t *testing.T) {
	runsRoot := filepath.Join(t.TempDir(), "runs-file")
	if err := os.WriteFile(runsRoot, []byte("not a directory"), 0o644); err != nil {
		t.Fatalf("write invalid runs root: %v", err)
	}
	service, err := NewProcessingService(ProcessingServiceConfig{
		RunsRoot: runsRoot,
	}, newProcessingHarness(false).factory)
	if err != nil {
		t.Fatalf("NewProcessingService: %v", err)
	}
	service.ctx = context.Background()
	service.recoveryPending = true
	service.refillRecoveryQueue()

	snapshot := service.Snapshot()
	if !snapshot.RecoveryPending || snapshot.RecoveryError == "" {
		t.Fatalf("transient recovery scan failure became invisible or terminal: %+v", snapshot)
	}
}

func TestProcessingServiceCloseLeavesQueuedWorkRecoverable(t *testing.T) {
	runsRoot := t.TempDir()
	activeTrip := createReadyProcessingTrip(t, runsRoot, 0, "")
	queuedTrip := createReadyProcessingTrip(t, runsRoot, 1, "")
	blockingHarness := newProcessingHarness(true)
	service, err := NewProcessingService(ProcessingServiceConfig{
		Workers:               1,
		QueueCapacity:         2,
		RunsRoot:              runsRoot,
		ReadinessTimeout:      time.Second,
		ReadinessPollInterval: time.Millisecond,
	}, blockingHarness.factory)
	if err != nil {
		t.Fatalf("NewProcessingService: %v", err)
	}
	if err := service.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}
	if accepted, err := service.Enqueue(activeTrip); err != nil || !accepted {
		t.Fatalf("enqueue active trip: accepted=%t err=%v", accepted, err)
	}
	if accepted, err := service.Enqueue(queuedTrip); err != nil || !accepted {
		t.Fatalf("enqueue queued trip: accepted=%t err=%v", accepted, err)
	}
	select {
	case <-blockingHarness.entered:
	case <-time.After(time.Second):
		t.Fatal("worker did not begin processing")
	}
	if err := service.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if accepted, err := service.Enqueue(queuedTrip); accepted || !errors.Is(err, ErrProcessingServiceClosed) {
		t.Fatalf("closed service accepted work: accepted=%t err=%v", accepted, err)
	}
	if err := service.Start(context.Background()); !errors.Is(err, ErrProcessingServiceClosed) {
		t.Fatalf("closed service restarted unexpectedly: %v", err)
	}
	queuedStatus, err := ReadStatusFile(filepath.Join(queuedTrip, "processing.json"))
	if err != nil || queuedStatus.State != "queued" {
		t.Fatalf("close must preserve undispatched durable work: status=%+v err=%v", queuedStatus, err)
	}
	activeStatus, err := ReadStatusFile(filepath.Join(activeTrip, "processing.json"))
	if err != nil || activeStatus.State != "queued" {
		t.Fatalf("close must return interrupted active work to the durable queue: status=%+v err=%v", activeStatus, err)
	}

	restartHarness := newProcessingHarness(false)
	restarted, err := NewProcessingService(ProcessingServiceConfig{
		Workers:               1,
		QueueCapacity:         1,
		RunsRoot:              runsRoot,
		ReadinessTimeout:      time.Second,
		ReadinessPollInterval: time.Millisecond,
	}, restartHarness.factory)
	if err != nil {
		t.Fatalf("NewProcessingService after close: %v", err)
	}
	if err := restarted.Start(context.Background()); err != nil {
		t.Fatalf("restart Start: %v", err)
	}
	t.Cleanup(func() { _ = restarted.Close() })
	waitCtx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := restarted.WaitIdle(waitCtx); err != nil {
		t.Fatalf("restart WaitIdle: %v", err)
	}
	if restartHarness.callCount(queuedTrip) != 1 {
		t.Fatalf("queued trip was not recovered after close: calls=%d", restartHarness.callCount(queuedTrip))
	}
	if restartHarness.callCount(activeTrip) != 1 {
		t.Fatalf("cancelled active trip was not returned to the durable queue: calls=%d", restartHarness.callCount(activeTrip))
	}
}

func TestProcessingServiceWaitsForTripReadiness(t *testing.T) {
	runsRoot := t.TempDir()
	tripDir := createIncompleteProcessingTrip(t, runsRoot, 0)
	harness := newProcessingHarness(false)
	service, err := NewProcessingService(ProcessingServiceConfig{
		Workers:               1,
		ReadinessTimeout:      time.Second,
		ReadinessPollInterval: 5 * time.Millisecond,
	}, harness.factory)
	if err != nil {
		t.Fatalf("NewProcessingService: %v", err)
	}
	if err := service.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}
	t.Cleanup(func() { _ = service.Close() })
	if accepted, err := service.Enqueue(tripDir); err != nil || !accepted {
		t.Fatalf("Enqueue: accepted=%t err=%v", accepted, err)
	}
	status, err := ReadStatusFile(filepath.Join(tripDir, "processing.json"))
	if err != nil || status.State != "queued" {
		t.Fatalf("queued intent was not persisted before readiness: status=%+v err=%v", status, err)
	}

	select {
	case <-harness.entered:
		t.Fatal("processor ran before metadata and run manifest were ready")
	case <-time.After(30 * time.Millisecond):
	}
	finishReadyProcessingTrip(t, tripDir, 0)
	waitCtx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := service.WaitIdle(waitCtx); err != nil {
		t.Fatalf("WaitIdle: %v", err)
	}
	if harness.callCount(tripDir) != 1 {
		t.Fatalf("ready trip was not processed once: calls=%d", harness.callCount(tripDir))
	}
}

func createReadyProcessingTrip(t *testing.T, runsRoot string, index int, state string) string {
	t.Helper()
	tripDir := createIncompleteProcessingTrip(t, runsRoot, index)
	finishReadyProcessingTrip(t, tripDir, index)
	if state != "" {
		writeJSONFile(t, filepath.Join(tripDir, "processing.json"), ProcessingStatus{State: state})
	}
	return tripDir
}

func createIncompleteProcessingTrip(t *testing.T, runsRoot string, index int) string {
	t.Helper()
	sceneDir := filepath.Join(runsRoot, "run-a", fmt.Sprintf("scene-%03d_default", index))
	tripDir := filepath.Join(sceneDir, fmt.Sprintf("trip-%03d", index))
	if err := os.MkdirAll(tripDir, 0o755); err != nil {
		t.Fatalf("mkdir processing trip: %v", err)
	}
	if err := os.WriteFile(filepath.Join(tripDir, "video.mkv"), []byte("video"), 0o644); err != nil {
		t.Fatalf("write processing video: %v", err)
	}
	return tripDir
}

func finishReadyProcessingTrip(t *testing.T, tripDir string, index int) {
	t.Helper()
	writeJSONFile(t, filepath.Join(tripDir, "metadata.json"), tripMetadata{
		RunID:     "run-a",
		TripIndex: index,
	})
	writeJSONLinesFile(t, filepath.Join(filepath.Dir(tripDir), "run.jsonl"), []runTripRecord{{
		RunID:     "run-a",
		TripIndex: index,
	}})
}
