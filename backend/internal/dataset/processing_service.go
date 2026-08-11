package dataset

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

const (
	defaultProcessingWorkers          = 1
	defaultProcessingQueueCapacity    = 256
	defaultProcessingReadinessTimeout = 45 * time.Second
	defaultProcessingReadinessPoll    = 500 * time.Millisecond
)

var (
	ErrProcessingServiceNotStarted = errors.New("dataset processing service is not started")
	ErrProcessingServiceStarted    = errors.New("dataset processing service is already started")
	ErrProcessingServiceClosed     = errors.New("dataset processing service is closed")
	ErrProcessingQueueFull         = errors.New("dataset processing queue is full")
)

// TripProcessor is the atomic unit run by ProcessingService workers.
type TripProcessor interface {
	Queue(tripDir string) (string, error)
	ProcessTrip(ctx context.Context, tripDir string) error
}

// ProcessorFactory gives each worker its own immutable processor instance.
type ProcessorFactory func() TripProcessor

// ProcessingServiceConfig controls bounded work and trip-readiness waiting.
type ProcessingServiceConfig struct {
	Workers               int
	QueueCapacity         int
	RunsRoot              string
	ReadinessTimeout      time.Duration
	ReadinessPollInterval time.Duration
	// RecoveryTripPredicate limits startup/background recovery only. Explicit
	// Enqueue calls remain available for any caller-selected trip.
	RecoveryTripPredicate func(tripDir string) bool
}

// ProcessingJobSnapshot is one immutable view of a queued or terminal job.
type ProcessingJobSnapshot struct {
	TripDir     string    `json:"tripDir"`
	State       string    `json:"state"`
	Error       string    `json:"error,omitempty"`
	QueuedAt    time.Time `json:"queuedAt"`
	StartedAt   time.Time `json:"startedAt,omitempty"`
	CompletedAt time.Time `json:"completedAt,omitempty"`
}

// ProcessingSnapshot describes all work known to this service instance.
type ProcessingSnapshot struct {
	WorkerCount     int                     `json:"workerCount"`
	QueueCapacity   int                     `json:"queueCapacity"`
	Recovered       int                     `json:"recovered"`
	RecoveryPending bool                    `json:"recoveryPending"`
	RecoveryError   string                  `json:"recoveryError,omitempty"`
	Queued          []ProcessingJobSnapshot `json:"queued"`
	Active          []ProcessingJobSnapshot `json:"active"`
	Completed       []ProcessingJobSnapshot `json:"completed"`
	Failed          []ProcessingJobSnapshot `json:"failed"`
}

type processingJob struct {
	tripDir     string
	state       string
	errorText   string
	queuedAt    time.Time
	startedAt   time.Time
	completedAt time.Time
}

func (j processingJob) snapshot() ProcessingJobSnapshot {
	return ProcessingJobSnapshot{
		TripDir:     j.tripDir,
		State:       j.state,
		Error:       j.errorText,
		QueuedAt:    j.queuedAt,
		StartedAt:   j.startedAt,
		CompletedAt: j.completedAt,
	}
}

// ProcessingService owns a durable-on-disk, bounded in-memory dispatch queue.
// Queued and running processing.json states are recovered when Start is called.
type ProcessingService struct {
	config         ProcessingServiceConfig
	factory        ProcessorFactory
	queueProcessor TripProcessor

	mu              sync.Mutex
	condition       *sync.Cond
	queue           []processingJob
	inFlight        map[string]struct{}
	active          map[string]processingJob
	completed       map[string]processingJob
	failed          map[string]processingJob
	recoveredKnown  map[string]struct{}
	recovered       int
	recoveryPending bool
	recoveryErr     error
	starting        bool
	started         bool
	closed          bool

	ctx          context.Context
	cancel       context.CancelFunc
	workers      sync.WaitGroup
	changed      chan struct{}
	recoveryWake chan struct{}
}

// NewProcessingService constructs a service with an injectable processor factory.
func NewProcessingService(config ProcessingServiceConfig, factory ProcessorFactory) (*ProcessingService, error) {
	if factory == nil {
		return nil, errors.New("dataset processor factory must not be nil")
	}
	config = normalizeProcessingServiceConfig(config)
	queueProcessor := factory()
	if queueProcessor == nil {
		return nil, errors.New("dataset processor factory returned nil")
	}
	service := &ProcessingService{
		config:         config,
		factory:        factory,
		queueProcessor: queueProcessor,
		inFlight:       make(map[string]struct{}),
		active:         make(map[string]processingJob),
		completed:      make(map[string]processingJob),
		failed:         make(map[string]processingJob),
		recoveredKnown: make(map[string]struct{}),
		changed:        make(chan struct{}, 1),
		recoveryWake:   make(chan struct{}, 1),
	}
	service.condition = sync.NewCond(&service.mu)
	return service, nil
}

// NewDefaultProcessingService composes the service with the production Processor.
func NewDefaultProcessingService(config ProcessingServiceConfig, processorOptions ...Option) (*ProcessingService, error) {
	return NewProcessingService(config, func() TripProcessor {
		return NewProcessor(processorOptions...)
	})
}

// Start recovers interrupted jobs and launches exactly config.Workers workers.
func (s *ProcessingService) Start(parent context.Context) error {
	if parent == nil {
		return errors.New("dataset processing service context must not be nil")
	}
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return ErrProcessingServiceClosed
	}
	if s.starting || s.started {
		s.mu.Unlock()
		return ErrProcessingServiceStarted
	}
	s.starting = true
	s.mu.Unlock()

	recoveredTripDirs, busyRecoveryTrips, err := scanRecoverableProcessingTripDirs(
		s.config.RunsRoot,
		true,
		s.config.RecoveryTripPredicate,
	)
	if err != nil {
		s.mu.Lock()
		s.starting = false
		s.mu.Unlock()
		return err
	}

	s.mu.Lock()
	if s.closed {
		s.starting = false
		s.mu.Unlock()
		return ErrProcessingServiceClosed
	}
	s.ctx, s.cancel = context.WithCancel(parent)
	s.starting = false
	s.started = true
	s.recoveryPending = busyRecoveryTrips
	for _, tripDir := range recoveredTripDirs {
		s.recoveredKnown[tripDir] = struct{}{}
		if len(s.queue) >= s.config.QueueCapacity {
			s.recoveryPending = true
			continue
		}
		s.enqueueLocked(tripDir, time.Now())
	}
	s.recovered = len(s.recoveredKnown)
	s.workers.Add(s.config.Workers)
	if s.recoveryPending {
		s.workers.Add(1)
	}
	s.mu.Unlock()

	for workerID := 1; workerID <= s.config.Workers; workerID++ {
		go s.runWorker(workerID)
	}
	if s.recoveryPending {
		go s.runRecoveryPump()
	}
	go s.wakeWorkersWhenContextEnds()
	s.signalChanged()
	return nil
}

// Enqueue adds a trip once while it is queued or active. The boolean reports acceptance.
func (s *ProcessingService) Enqueue(tripDir string) (bool, error) {
	resolved, err := canonicalTripDir(tripDir)
	if err != nil {
		return false, err
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.started {
		return false, ErrProcessingServiceNotStarted
	}
	if s.closed {
		return false, ErrProcessingServiceClosed
	}
	if s.ctx == nil || s.ctx.Err() != nil {
		return false, ErrProcessingServiceClosed
	}
	if _, exists := s.inFlight[resolved]; exists {
		return false, nil
	}
	if len(s.queue) >= s.config.QueueCapacity {
		return false, ErrProcessingQueueFull
	}
	if _, err := s.queueProcessor.Queue(resolved); err != nil {
		return false, fmt.Errorf("persist queued processing status: %w", err)
	}
	s.enqueueLocked(resolved, time.Now())
	s.condition.Signal()
	s.signalChanged()
	return true, nil
}

// Snapshot returns deterministically ordered queue, active, and terminal state.
func (s *ProcessingService) Snapshot() ProcessingSnapshot {
	s.mu.Lock()
	defer s.mu.Unlock()

	snapshot := ProcessingSnapshot{
		WorkerCount:     s.config.Workers,
		QueueCapacity:   s.config.QueueCapacity,
		Recovered:       s.recovered,
		RecoveryPending: s.recoveryPending,
		Queued:          snapshotJobs(s.queue),
		Active:          snapshotJobMap(s.active),
		Completed:       snapshotJobMap(s.completed),
		Failed:          snapshotJobMap(s.failed),
	}
	if s.recoveryErr != nil {
		snapshot.RecoveryError = s.recoveryErr.Error()
	}
	return snapshot
}

// WaitIdle waits until all accepted work is terminal.
func (s *ProcessingService) WaitIdle(ctx context.Context) error {
	if ctx == nil {
		return errors.New("dataset processing wait context must not be nil")
	}
	for {
		s.mu.Lock()
		recoveryPending := s.recoveryPending
		idle := len(s.queue) == 0 && len(s.active) == 0 && !recoveryPending
		closed := s.closed
		recoveryErr := s.recoveryErr
		s.mu.Unlock()
		if recoveryErr != nil && !recoveryPending {
			return recoveryErr
		}
		if idle {
			return nil
		}
		if closed {
			return ErrProcessingServiceClosed
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-s.changed:
		}
	}
}

// Close stops dispatch, cancels active processors, and waits for workers to exit.
func (s *ProcessingService) Close() error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		s.workers.Wait()
		return nil
	}
	s.closed = true
	if s.cancel != nil {
		s.cancel()
	}
	s.condition.Broadcast()
	s.mu.Unlock()
	s.signalChanged()
	s.workers.Wait()
	return nil
}

func normalizeProcessingServiceConfig(config ProcessingServiceConfig) ProcessingServiceConfig {
	if config.Workers < 1 {
		config.Workers = defaultProcessingWorkers
	}
	if config.QueueCapacity < 1 {
		config.QueueCapacity = defaultProcessingQueueCapacity
	}
	config.RunsRoot = strings.TrimSpace(config.RunsRoot)
	if config.ReadinessTimeout <= 0 {
		config.ReadinessTimeout = defaultProcessingReadinessTimeout
	}
	if config.ReadinessPollInterval <= 0 {
		config.ReadinessPollInterval = defaultProcessingReadinessPoll
	}
	return config
}

func (s *ProcessingService) enqueueLocked(tripDir string, queuedAt time.Time) {
	job := processingJob{
		tripDir:  tripDir,
		state:    "queued",
		queuedAt: queuedAt,
	}
	s.queue = append(s.queue, job)
	s.inFlight[tripDir] = struct{}{}
	delete(s.completed, tripDir)
	delete(s.failed, tripDir)
}

func (s *ProcessingService) runWorker(_ int) {
	defer s.workers.Done()
	processor := s.factory()
	for {
		job, ok := s.takeNextJob()
		if !ok {
			return
		}
		err := s.processJob(processor, job)
		s.finishJob(job, err)
	}
}

func (s *ProcessingService) takeNextJob() (processingJob, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for len(s.queue) == 0 && !s.closed && s.ctx.Err() == nil {
		s.condition.Wait()
	}
	if s.closed || s.ctx.Err() != nil {
		return processingJob{}, false
	}

	job := s.queue[0]
	s.queue = append([]processingJob(nil), s.queue[1:]...)
	job.state = "running"
	job.startedAt = time.Now()
	s.active[job.tripDir] = job
	s.signalRecoveryRefill()
	s.signalChanged()
	return job, true
}

func (s *ProcessingService) runRecoveryPump() {
	defer s.workers.Done()
	ticker := time.NewTicker(s.config.ReadinessPollInterval)
	defer ticker.Stop()
	for {
		select {
		case <-s.ctx.Done():
			return
		case <-s.recoveryWake:
			s.refillRecoveryQueue()
		case <-ticker.C:
			s.mu.Lock()
			pending := s.recoveryPending
			s.mu.Unlock()
			if pending {
				s.refillRecoveryQueue()
			}
		}
	}
}

func (s *ProcessingService) signalRecoveryRefill() {
	if !s.recoveryPending {
		return
	}
	select {
	case s.recoveryWake <- struct{}{}:
	default:
	}
}

func (s *ProcessingService) refillRecoveryQueue() {
	recoverable, busy, err := scanRecoverableProcessingTripDirs(
		s.config.RunsRoot,
		false,
		s.config.RecoveryTripPredicate,
	)

	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed || s.ctx.Err() != nil {
		return
	}
	if err != nil {
		s.recoveryErr = fmt.Errorf("refill recovered processing queue: %w", err)
		s.recoveryPending = true
		s.signalChanged()
		return
	}

	remaining := busy
	added := 0
	for _, tripDir := range recoverable {
		if _, exists := s.inFlight[tripDir]; exists {
			continue
		}
		status, statusErr := ReadStatusFile(filepath.Join(tripDir, "processing.json"))
		if statusErr != nil || (status.State != "queued" && status.State != "running") {
			continue
		}
		if len(s.queue) >= s.config.QueueCapacity {
			remaining = true
			continue
		}
		s.enqueueLocked(tripDir, time.Now())
		if _, known := s.recoveredKnown[tripDir]; !known {
			s.recoveredKnown[tripDir] = struct{}{}
			s.recovered++
		}
		added++
	}
	s.recoveryPending = remaining
	s.recoveryErr = nil
	if added > 0 {
		s.condition.Broadcast()
	}
	s.signalChanged()
}

func (s *ProcessingService) processJob(processor TripProcessor, job processingJob) error {
	if processor == nil {
		err := errors.New("dataset processor factory returned nil")
		return errors.Join(err, writeServiceFailureStatus(job.tripDir, err))
	}
	readinessCtx, cancel := context.WithTimeout(s.ctx, s.config.ReadinessTimeout)
	defer cancel()
	if err := WaitForTripReadiness(readinessCtx, job.tripDir, s.config.ReadinessPollInterval); err != nil {
		if s.ctx.Err() != nil {
			return errors.Join(err, writeServiceQueuedStatus(job.tripDir))
		}
		return errors.Join(err, writeServiceFailureStatus(job.tripDir, err))
	}
	for {
		err := processor.ProcessTrip(s.ctx, job.tripDir)
		if !errors.Is(err, ErrTripProcessingLocked) {
			if err == nil {
				return nil
			}
			if s.ctx.Err() != nil {
				return errors.Join(err, writeServiceQueuedStatus(job.tripDir))
			}
			return errors.Join(err, writeServiceFailureStatus(job.tripDir, err))
		}
		select {
		case <-s.ctx.Done():
			return errors.Join(s.ctx.Err(), writeServiceQueuedStatus(job.tripDir))
		case <-time.After(s.config.ReadinessPollInterval):
		}
	}
}

func (s *ProcessingService) finishJob(job processingJob, processingErr error) {
	completedAt := time.Now()
	job.completedAt = completedAt
	job.errorText = ""
	job.state = terminalProcessingState(job.tripDir, processingErr)
	if processingErr != nil {
		job.errorText = processingErr.Error()
		job.state = "failed"
	}

	s.mu.Lock()
	delete(s.active, job.tripDir)
	delete(s.inFlight, job.tripDir)
	if processingErr != nil {
		s.failed[job.tripDir] = job
		delete(s.completed, job.tripDir)
	} else {
		s.completed[job.tripDir] = job
		delete(s.failed, job.tripDir)
	}
	s.mu.Unlock()
	s.signalChanged()
}

func terminalProcessingState(tripDir string, processingErr error) string {
	if processingErr != nil {
		return "failed"
	}
	status, err := ReadStatusFile(filepath.Join(tripDir, "processing.json"))
	if err == nil && (status.State == "completed" || status.State == "skipped") {
		return status.State
	}
	return "completed"
}

func (s *ProcessingService) wakeWorkersWhenContextEnds() {
	<-s.ctx.Done()
	s.mu.Lock()
	s.closed = true
	s.condition.Broadcast()
	s.mu.Unlock()
	s.signalChanged()
}

func (s *ProcessingService) signalChanged() {
	select {
	case s.changed <- struct{}{}:
	default:
	}
}

func canonicalTripDir(tripDir string) (string, error) {
	resolved, err := resolveTripDir(tripDir)
	if err != nil {
		return "", err
	}
	absolute, err := filepath.Abs(resolved)
	if err != nil {
		return "", fmt.Errorf("resolve absolute trip directory: %w", err)
	}
	return filepath.Clean(absolute), nil
}

func recoverableProcessingTripDirs(runsRoot string) ([]string, error) {
	recoverable, _, err := scanRecoverableProcessingTripDirs(runsRoot, true, nil)
	return recoverable, err
}

func scanRecoverableProcessingTripDirs(
	runsRoot string,
	cleanupStaleWorkspaces bool,
	predicate func(tripDir string) bool,
) ([]string, bool, error) {
	if strings.TrimSpace(runsRoot) == "" {
		return nil, false, nil
	}
	info, err := os.Stat(runsRoot)
	if errors.Is(err, os.ErrNotExist) {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, fmt.Errorf("stat processing runs root: %w", err)
	}
	if !info.IsDir() {
		return nil, false, fmt.Errorf("processing runs root is not a directory: %s", runsRoot)
	}

	tripDirs, err := CollectTripDirs(runsRoot, nil)
	if err != nil {
		return nil, false, err
	}
	recoverable := make([]string, 0)
	busy := false
	for _, tripDir := range tripDirs {
		if predicate != nil && !predicate(tripDir) {
			continue
		}
		shouldRecover, tripBusy, inspectErr := inspectTripForRecovery(tripDir, cleanupStaleWorkspaces)
		if inspectErr != nil {
			return nil, false, inspectErr
		}
		if tripBusy {
			busy = true
		}
		if !shouldRecover {
			continue
		}
		resolved, resolveErr := canonicalTripDir(tripDir)
		if resolveErr != nil {
			return nil, false, resolveErr
		}
		recoverable = append(recoverable, resolved)
	}
	sort.Strings(recoverable)
	return recoverable, busy, nil
}

func inspectTripForRecovery(tripDir string, cleanupStaleWorkspaces bool) (recoverable bool, busy bool, err error) {
	lock, err := acquireTripProcessingLock(tripDir)
	if errors.Is(err, ErrTripProcessingLocked) {
		return false, true, nil
	}
	if err != nil {
		return false, false, err
	}
	defer func() {
		err = errors.Join(err, lock.Release())
	}()
	if cleanupStaleWorkspaces {
		if err := recoverStaleProcessingArtifacts(tripDir); err != nil {
			return false, false, err
		}
	}
	status, err := ReadStatusFile(filepath.Join(tripDir, "processing.json"))
	if err != nil {
		return false, false, nil
	}
	return status.State == "queued" || status.State == "running", false, nil
}

func recoverStaleProcessingArtifacts(tripDir string) error {
	entries, err := os.ReadDir(tripDir)
	if err != nil {
		return fmt.Errorf("read trip directory for stale processing artifacts: %w", err)
	}
	status, statusErr := ReadStatusFile(filepath.Join(tripDir, "processing.json"))
	committed := statusErr == nil && (status.State == "completed" || status.State == "skipped")
	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), ".processing.json.tmp-") && entry.Type().IsRegular() {
			if err := os.Remove(filepath.Join(tripDir, entry.Name())); err != nil {
				return fmt.Errorf("remove stale processing status temp file: %w", err)
			}
			continue
		}
		if !strings.HasPrefix(entry.Name(), processingWorkspacePrefix) {
			continue
		}
		workspacePath := filepath.Join(tripDir, entry.Name())
		info, err := os.Lstat(workspacePath)
		if err != nil {
			return fmt.Errorf("inspect stale processing workspace %s: %w", workspacePath, err)
		}
		if info.Mode()&os.ModeSymlink != 0 || !info.IsDir() {
			continue
		}
		workspace := processingWorkspaceForRoot(workspacePath)
		if committed {
			if err := removeProcessingWorkspace(workspace); err != nil {
				return err
			}
			continue
		}
		if fileExists(workspace.journalPath) {
			if err := rollbackProcessingWorkspace(tripDir, workspace); err != nil {
				return fmt.Errorf("recover stale processing workspace %s: %w", workspacePath, err)
			}
		} else if pathExists(filepath.Join(workspacePath, "previous-frames")) || pathExists(filepath.Join(workspacePath, "previous-dataset.jsonl")) {
			return fmt.Errorf("stale processing workspace has backups but no recovery journal; preserved at %s", workspacePath)
		}
		if err := removeProcessingWorkspace(workspace); err != nil {
			return err
		}
	}
	return nil
}

func writeServiceFailureStatus(tripDir string, processingErr error) error {
	statusPath := filepath.Join(tripDir, "processing.json")
	status, err := ReadStatusFile(statusPath)
	if err != nil {
		status = ProcessingStatus{
			FramesDir:   "frames",
			DatasetFile: "dataset.jsonl",
		}
	}
	status.State = "failed"
	status.Error = processingErr.Error()
	if err := writeStatusFile(statusPath, status); err != nil {
		return fmt.Errorf("persist failed processing status: %w", err)
	}
	return nil
}

func writeServiceQueuedStatus(tripDir string) error {
	statusPath := filepath.Join(tripDir, "processing.json")
	status, err := ReadStatusFile(statusPath)
	if err != nil {
		status = ProcessingStatus{
			FramesDir:   "frames",
			DatasetFile: "dataset.jsonl",
		}
	}
	status.State = "queued"
	status.Error = ""
	if err := writeStatusFile(statusPath, status); err != nil {
		return fmt.Errorf("persist interrupted processing status: %w", err)
	}
	return nil
}

func snapshotJobs(jobs []processingJob) []ProcessingJobSnapshot {
	snapshots := make([]ProcessingJobSnapshot, 0, len(jobs))
	for _, job := range jobs {
		snapshots = append(snapshots, job.snapshot())
	}
	sort.Slice(snapshots, func(i, j int) bool {
		return snapshots[i].TripDir < snapshots[j].TripDir
	})
	return snapshots
}

func snapshotJobMap(jobs map[string]processingJob) []ProcessingJobSnapshot {
	values := make([]processingJob, 0, len(jobs))
	for _, job := range jobs {
		values = append(values, job)
	}
	return snapshotJobs(values)
}
