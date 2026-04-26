package training

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	toml "github.com/pelletier/go-toml/v2"
)

const (
	StatusQueued    = "queued"
	StatusStarting  = "starting"
	StatusRunning   = "running"
	StatusCompleted = "completed"
	StatusFailed    = "failed"
	StatusCanceled  = "canceled"
	StatusStopped   = "stopped"
)

var (
	ErrJobNotFound        = errors.New("training job not found")
	ErrJobNotPending      = errors.New("training job is not pending")
	ErrJobNotActive       = errors.New("training job is not active")
	ErrInvalidJobRequest  = errors.New("invalid training job request")
	ErrTrainingNotRunning = errors.New("training job is not running")
)

type commandFactory func(ctx context.Context, name string, args ...string) *exec.Cmd

type JobSpec struct {
	Name         string             `json:"name,omitempty"`
	Notes        string             `json:"notes,omitempty"`
	LearningRate *float64           `json:"learningRate,omitempty"`
	LossWeights  map[string]float64 `json:"lossWeights,omitempty"`
	Consistency  map[string]float64 `json:"consistency,omitempty"`
}

type Job struct {
	ID              string             `json:"id"`
	Name            string             `json:"name,omitempty"`
	Notes           string             `json:"notes,omitempty"`
	Status          string             `json:"status"`
	LearningRate    *float64           `json:"learningRate,omitempty"`
	LossWeights     map[string]float64 `json:"lossWeights,omitempty"`
	Consistency     map[string]float64 `json:"consistency,omitempty"`
	CreatedAt       string             `json:"createdAt"`
	StartedAt       string             `json:"startedAt,omitempty"`
	FinishedAt      string             `json:"finishedAt,omitempty"`
	ConfigPath      string             `json:"configPath,omitempty"`
	LogPath         string             `json:"logPath,omitempty"`
	JobDir          string             `json:"jobDir,omitempty"`
	RunDir          string             `json:"runDir,omitempty"`
	RunMetricsPath  string             `json:"runMetricsPath,omitempty"`
	ExitCode        *int               `json:"exitCode,omitempty"`
	Error           string             `json:"error,omitempty"`
	Command         []string           `json:"command,omitempty"`
	CancelRequested bool               `json:"cancelRequested,omitempty"`
	StopRequested   bool               `json:"stopRequested,omitempty"`
	LastUpdatedAt   string             `json:"lastUpdatedAt,omitempty"`
}

type State struct {
	ActiveJobID   string `json:"activeJobId,omitempty"`
	QueuedCount   int    `json:"queuedCount"`
	Running       bool   `json:"running"`
	QueuedJobs    []Job  `json:"queuedJobs"`
	ActiveJob     *Job   `json:"activeJob,omitempty"`
	RecentJobs    []Job  `json:"recentJobs"`
	JobsDirectory string `json:"jobsDirectory"`
}

type Manager struct {
	mu             sync.Mutex
	cfg            Config
	baseConfigPath string
	nowFunc        func() time.Time
	newCommand     commandFactory
	recentLimit    int
	triggerCh      chan struct{}
	jobs           map[string]*Job
	queue          []string
	activeJobID    string
	activeCancel   context.CancelFunc
	activeDone     chan struct{}
	activeProcess  *os.Process
	activeStopReq  bool
}

type jobFile struct {
	Job Job `json:"job"`
}

func NewManager(cfg Config, baseConfigPath string) (*Manager, error) {
	if err := os.MkdirAll(cfg.JobsDir, 0o755); err != nil {
		return nil, fmt.Errorf("create jobs dir: %w", err)
	}
	manager := &Manager{
		cfg:            cfg,
		baseConfigPath: strings.TrimSpace(baseConfigPath),
		nowFunc:        time.Now,
		newCommand: func(ctx context.Context, name string, args ...string) *exec.Cmd {
			return exec.CommandContext(ctx, name, args...)
		},
		recentLimit: 50,
		triggerCh:   make(chan struct{}, 1),
		jobs:        make(map[string]*Job),
	}
	if err := manager.loadExistingJobs(); err != nil {
		return nil, err
	}
	go manager.runLoop()
	return manager, nil
}

func (m *Manager) State() State {
	m.mu.Lock()
	defer m.mu.Unlock()

	queuedJobs := make([]Job, 0, len(m.queue))
	for _, id := range m.queue {
		if job, ok := m.jobs[id]; ok {
			queuedJobs = append(queuedJobs, cloneJob(job))
		}
	}

	var active *Job
	if m.activeJobID != "" {
		if job, ok := m.jobs[m.activeJobID]; ok {
			copy := cloneJob(job)
			active = &copy
		}
	}

	recent := make([]Job, 0, len(m.jobs))
	for _, job := range m.jobs {
		if isTerminalStatus(job.Status) {
			recent = append(recent, cloneJob(job))
		}
	}
	sortJobsByCreatedDesc(recent)
	if len(recent) > m.recentLimit {
		recent = recent[:m.recentLimit]
	}

	return State{
		ActiveJobID:   m.activeJobID,
		QueuedCount:   len(m.queue),
		Running:       m.activeJobID != "",
		QueuedJobs:    queuedJobs,
		ActiveJob:     active,
		RecentJobs:    recent,
		JobsDirectory: m.cfg.JobsDir,
	}
}

func (m *Manager) ListJobs() []Job {
	m.mu.Lock()
	defer m.mu.Unlock()
	items := make([]Job, 0, len(m.jobs))
	for _, job := range m.jobs {
		items = append(items, cloneJob(job))
	}
	sortJobsByCreatedDesc(items)
	return items
}

func (m *Manager) GetJob(id string) (Job, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	job, ok := m.jobs[strings.TrimSpace(id)]
	if !ok {
		return Job{}, ErrJobNotFound
	}
	return cloneJob(job), nil
}

func (m *Manager) ReadLog(id string) (string, error) {
	job, err := m.GetJob(id)
	if err != nil {
		return "", err
	}
	if strings.TrimSpace(job.LogPath) == "" {
		return "", nil
	}
	body, err := os.ReadFile(job.LogPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "", nil
		}
		return "", err
	}
	return string(body), nil
}

func (m *Manager) ReadLogTail(id string, tailLines int) (string, error) {
	text, err := m.ReadLog(id)
	if err != nil || tailLines <= 0 {
		return text, err
	}
	lines := strings.Split(text, "\n")
	if len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	if len(lines) <= tailLines {
		return text, nil
	}
	return strings.Join(lines[len(lines)-tailLines:], "\n") + "\n", nil
}

func (m *Manager) Enqueue(specs []JobSpec) ([]Job, error) {
	if len(specs) == 0 {
		return nil, fmt.Errorf("%w: at least one job is required", ErrInvalidJobRequest)
	}

	now := m.nowFunc().UTC()
	created := make([]Job, 0, len(specs))

	m.mu.Lock()
	defer m.mu.Unlock()

	for index, spec := range specs {
		normalized, err := normalizeJobSpec(spec)
		if err != nil {
			return nil, err
		}
		jobID := fmt.Sprintf("train-%d-%03d", now.UnixNano(), index+1)
		jobDir := filepath.Join(m.cfg.JobsDir, jobID)
		job := &Job{
			ID:            jobID,
			Name:          normalized.Name,
			Notes:         normalized.Notes,
			Status:        StatusQueued,
			LearningRate:  normalized.LearningRate,
			LossWeights:   cloneFloatMap(normalized.LossWeights),
			Consistency:   cloneFloatMap(normalized.Consistency),
			CreatedAt:     now.Format(time.RFC3339Nano),
			LastUpdatedAt: now.Format(time.RFC3339Nano),
			JobDir:        jobDir,
			ConfigPath:    filepath.Join(jobDir, "derived_train_config.toml"),
			LogPath:       filepath.Join(jobDir, "train.log"),
		}
		if err := os.MkdirAll(jobDir, 0o755); err != nil {
			return nil, fmt.Errorf("create job dir: %w", err)
		}
		if err := m.persistJobLocked(job); err != nil {
			return nil, err
		}
		m.jobs[job.ID] = job
		m.queue = append(m.queue, job.ID)
		created = append(created, cloneJob(job))
	}
	m.signalLocked()
	return created, nil
}

func (m *Manager) Cancel(id string) (Job, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	job, ok := m.jobs[strings.TrimSpace(id)]
	if !ok {
		return Job{}, ErrJobNotFound
	}
	if job.Status != StatusQueued {
		return Job{}, ErrJobNotPending
	}
	job.Status = StatusCanceled
	job.CancelRequested = true
	job.FinishedAt = m.nowFunc().UTC().Format(time.RFC3339Nano)
	job.LastUpdatedAt = job.FinishedAt
	m.queue = removeQueueID(m.queue, job.ID)
	if err := m.persistJobLocked(job); err != nil {
		return Job{}, err
	}
	return cloneJob(job), nil
}

func (m *Manager) Stop(id string) (Job, error) {
	m.mu.Lock()
	job, ok := m.jobs[strings.TrimSpace(id)]
	if !ok {
		m.mu.Unlock()
		return Job{}, ErrJobNotFound
	}
	if job.ID != m.activeJobID || m.activeCancel == nil {
		m.mu.Unlock()
		return Job{}, ErrJobNotActive
	}
	job.StopRequested = true
	job.LastUpdatedAt = m.nowFunc().UTC().Format(time.RFC3339Nano)
	m.activeStopReq = true
	cancel := m.activeCancel
	done := m.activeDone
	process := m.activeProcess
	if err := m.persistJobLocked(job); err != nil {
		m.mu.Unlock()
		return Job{}, err
	}
	snapshot := cloneJob(job)
	m.mu.Unlock()

	cancel()
	if done != nil {
		select {
		case <-done:
			return snapshot, nil
		case <-time.After(3 * time.Second):
		}
	}
	if process != nil {
		_ = process.Kill()
	}
	return snapshot, nil
}

func (m *Manager) runLoop() {
	for range m.triggerCh {
		for {
			job := m.nextRunnableJob()
			if job == nil {
				break
			}
			m.runJob(job)
		}
	}
}

func (m *Manager) nextRunnableJob() *Job {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.activeJobID != "" {
		return nil
	}
	for len(m.queue) > 0 {
		id := m.queue[0]
		m.queue = m.queue[1:]
		job, ok := m.jobs[id]
		if !ok || job.Status != StatusQueued {
			continue
		}
		m.activeJobID = job.ID
		m.activeStopReq = false
		return cloneJobPtr(job)
	}
	return nil
}

func (m *Manager) runJob(job *Job) {
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})

	m.mu.Lock()
	if stored, ok := m.jobs[job.ID]; ok {
		now := m.nowFunc().UTC().Format(time.RFC3339Nano)
		stored.Status = StatusStarting
		stored.StartedAt = now
		stored.LastUpdatedAt = now
		stored.StopRequested = false
		stored.CancelRequested = false
		m.activeCancel = cancel
		m.activeDone = done
		_ = m.persistJobLocked(stored)
	}
	m.mu.Unlock()

	exitCode, runDir, runMetricsPath, runErr := m.executeJob(ctx, job.ID)

	m.mu.Lock()
	defer m.mu.Unlock()
	defer close(done)
	stored := m.jobs[job.ID]
	if stored != nil {
		now := m.nowFunc().UTC().Format(time.RFC3339Nano)
		stored.RunDir = runDir
		stored.RunMetricsPath = runMetricsPath
		stored.LastUpdatedAt = now
		if exitCode != nil {
			value := *exitCode
			stored.ExitCode = &value
		}
		if runErr == nil {
			stored.Status = StatusCompleted
			stored.FinishedAt = now
			stored.Error = ""
		} else {
			stored.FinishedAt = now
			stored.Error = runErr.Error()
			if m.activeStopReq {
				stored.Status = StatusStopped
			} else {
				stored.Status = StatusFailed
			}
		}
		_ = m.persistJobLocked(stored)
	}
	m.activeJobID = ""
	m.activeCancel = nil
	m.activeDone = nil
	m.activeProcess = nil
	m.activeStopReq = false
	m.signalLocked()
}

func (m *Manager) executeJob(ctx context.Context, jobID string) (*int, string, string, error) {
	m.mu.Lock()
	job, ok := m.jobs[jobID]
	m.mu.Unlock()
	if !ok {
		return nil, "", "", ErrJobNotFound
	}

	if err := m.writeDerivedConfig(job); err != nil {
		return nil, "", "", err
	}

	logFile, err := os.OpenFile(job.LogPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND|os.O_TRUNC, 0o644)
	if err != nil {
		return nil, "", "", fmt.Errorf("open training log: %w", err)
	}
	defer logFile.Close()

	args := []string{m.cfg.TrainScript, "--config", job.ConfigPath}
	cmd := m.newCommand(ctx, m.cfg.PythonBin, args...)
	if cmd.Cancel != nil {
		cmd.Cancel = func() error { return nil }
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, "", "", fmt.Errorf("prepare stdout pipe: %w", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return nil, "", "", fmt.Errorf("prepare stderr pipe: %w", err)
	}

	m.mu.Lock()
	if stored, ok := m.jobs[jobID]; ok {
		stored.Status = StatusRunning
		stored.Command = append([]string{m.cfg.PythonBin}, args...)
		stored.LastUpdatedAt = m.nowFunc().UTC().Format(time.RFC3339Nano)
		m.activeProcess = cmd.Process
		_ = m.persistJobLocked(stored)
	}
	m.mu.Unlock()

	if err := cmd.Start(); err != nil {
		return nil, "", "", fmt.Errorf("start training process: %w", err)
	}

	outputCh := make(chan parsedOutput, 2)
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		streamTrainingOutput("stdout", stdout, logFile, outputCh)
	}()
	go func() {
		defer wg.Done()
		streamTrainingOutput("stderr", stderr, logFile, outputCh)
	}()

	waitErr := cmd.Wait()
	wg.Wait()
	close(outputCh)

	runDir := ""
	runMetricsPath := ""
	for parsed := range outputCh {
		if parsed.runDir != "" {
			runDir = parsed.runDir
		}
		if parsed.runMetrics != "" {
			runMetricsPath = parsed.runMetrics
		}
	}

	exitCode := 0
	if cmd.ProcessState != nil {
		exitCode = cmd.ProcessState.ExitCode()
	}

	if waitErr != nil {
		if ctx.Err() != nil {
			return &exitCode, runDir, runMetricsPath, fmt.Errorf("training process stopped: %w", ctx.Err())
		}
		return &exitCode, runDir, runMetricsPath, fmt.Errorf("training process failed: %w", waitErr)
	}
	if ctx.Err() != nil {
		return &exitCode, runDir, runMetricsPath, fmt.Errorf("training process stopped: %w", ctx.Err())
	}
	return &exitCode, runDir, runMetricsPath, nil
}

func streamTrainingOutput(streamName string, reader io.Reader, logFile *os.File, outputCh chan<- parsedOutput) {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		line := scanner.Text()
		_, _ = fmt.Fprintf(logFile, "[%s] %s\n", streamName, line)
		runDir, runMetrics := parseTrainingOutputLine(line)
		if runDir != "" || runMetrics != "" {
			outputCh <- parsedOutput{runDir: runDir, runMetrics: runMetrics}
		}
	}
}

type parsedOutput struct {
	runDir     string
	runMetrics string
}

func parseTrainingOutputLine(line string) (string, string) {
	runDir := parseKeyValueFromLine(line, "run_dir")
	runMetrics := parseKeyValueFromLine(line, "run_metrics")
	return runDir, runMetrics
}

func parseKeyValueFromLine(line string, key string) string {
	pattern := key + "="
	index := strings.Index(line, pattern)
	if index < 0 {
		return ""
	}
	value := strings.TrimSpace(line[index+len(pattern):])
	if value == "" {
		return ""
	}
	if split := strings.IndexAny(value, " \t\r\n"); split > 0 {
		value = value[:split]
	}
	return strings.Trim(value, "\"'")
}

func (m *Manager) writeDerivedConfig(job *Job) error {
	raw, err := os.ReadFile(m.baseConfigPath)
	if err != nil {
		return fmt.Errorf("read base train config: %w", err)
	}
	var parsed map[string]any
	if err := toml.Unmarshal(raw, &parsed); err != nil {
		return fmt.Errorf("parse base train config: %w", err)
	}
	trainingMap := ensureMap(parsed, "training")
	if job.LearningRate != nil {
		trainingMap["learning_rate"] = *job.LearningRate
	}
	if len(job.LossWeights) > 0 {
		trainingMap["loss_weights"] = mapStringFloatToAny(job.LossWeights)
	}
	if len(job.Consistency) > 0 {
		trainingMap["consistency"] = mapStringFloatToAny(job.Consistency)
	}

	encoded, err := toml.Marshal(parsed)
	if err != nil {
		return fmt.Errorf("marshal derived train config: %w", err)
	}
	if err := os.WriteFile(job.ConfigPath, encoded, 0o644); err != nil {
		return fmt.Errorf("write derived train config: %w", err)
	}
	return nil
}

func ensureMap(root map[string]any, key string) map[string]any {
	if existing, ok := root[key]; ok {
		if typed, ok := existing.(map[string]any); ok {
			return typed
		}
	}
	created := make(map[string]any)
	root[key] = created
	return created
}

func mapStringFloatToAny(input map[string]float64) map[string]any {
	out := make(map[string]any, len(input))
	for key, value := range input {
		out[key] = value
	}
	return out
}

func normalizeJobSpec(spec JobSpec) (JobSpec, error) {
	normalized := JobSpec{
		Name:  strings.TrimSpace(spec.Name),
		Notes: strings.TrimSpace(spec.Notes),
	}
	if spec.LearningRate != nil {
		if !isFiniteFloat(*spec.LearningRate) || *spec.LearningRate <= 0 {
			return JobSpec{}, fmt.Errorf("%w: learningRate must be > 0", ErrInvalidJobRequest)
		}
		value := *spec.LearningRate
		normalized.LearningRate = &value
	}
	if spec.LossWeights != nil {
		normalized.LossWeights = make(map[string]float64, len(spec.LossWeights))
		for key, value := range spec.LossWeights {
			name := strings.TrimSpace(key)
			if name == "" || !isFiniteFloat(value) {
				return JobSpec{}, fmt.Errorf("%w: lossWeights must be a string->finite number map", ErrInvalidJobRequest)
			}
			normalized.LossWeights[name] = value
		}
	}
	if spec.Consistency != nil {
		normalized.Consistency = make(map[string]float64, len(spec.Consistency))
		for key, value := range spec.Consistency {
			name := strings.TrimSpace(key)
			if name == "" || !isFiniteFloat(value) {
				return JobSpec{}, fmt.Errorf("%w: consistency must be a string->finite number map", ErrInvalidJobRequest)
			}
			normalized.Consistency[name] = value
		}
	}
	return normalized, nil
}

func isFiniteFloat(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func removeQueueID(queue []string, id string) []string {
	filtered := queue[:0]
	for _, item := range queue {
		if item != id {
			filtered = append(filtered, item)
		}
	}
	return filtered
}

func cloneJob(job *Job) Job {
	if job == nil {
		return Job{}
	}
	copy := *job
	copy.LossWeights = cloneFloatMap(job.LossWeights)
	copy.Consistency = cloneFloatMap(job.Consistency)
	copy.Command = append([]string(nil), job.Command...)
	if job.LearningRate != nil {
		value := *job.LearningRate
		copy.LearningRate = &value
	}
	if job.ExitCode != nil {
		value := *job.ExitCode
		copy.ExitCode = &value
	}
	return copy
}

func cloneJobPtr(job *Job) *Job {
	copy := cloneJob(job)
	return &copy
}

func cloneFloatMap(input map[string]float64) map[string]float64 {
	if len(input) == 0 {
		return nil
	}
	out := make(map[string]float64, len(input))
	for key, value := range input {
		out[key] = value
	}
	return out
}

func sortJobsByCreatedDesc(items []Job) {
	sort.Slice(items, func(i, j int) bool {
		return items[i].CreatedAt > items[j].CreatedAt
	})
}

func isTerminalStatus(status string) bool {
	switch status {
	case StatusCompleted, StatusFailed, StatusCanceled, StatusStopped:
		return true
	default:
		return false
	}
}

func (m *Manager) signalLocked() {
	select {
	case m.triggerCh <- struct{}{}:
	default:
	}
}

func (m *Manager) persistJobLocked(job *Job) error {
	if job == nil {
		return nil
	}
	if err := os.MkdirAll(job.JobDir, 0o755); err != nil {
		return fmt.Errorf("create job dir: %w", err)
	}
	path := filepath.Join(job.JobDir, "job.json")
	body, err := json.MarshalIndent(jobFile{Job: cloneJob(job)}, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal job file: %w", err)
	}
	if err := os.WriteFile(path, append(body, '\n'), 0o644); err != nil {
		return fmt.Errorf("write job file: %w", err)
	}
	return nil
}

func (m *Manager) loadExistingJobs() error {
	entries, err := os.ReadDir(m.cfg.JobsDir)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return err
	}
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		jobPath := filepath.Join(m.cfg.JobsDir, entry.Name(), "job.json")
		body, err := os.ReadFile(jobPath)
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				continue
			}
			return err
		}
		var payload jobFile
		if err := json.Unmarshal(body, &payload); err != nil {
			return fmt.Errorf("parse job metadata %s: %w", jobPath, err)
		}
		job := payload.Job
		if job.ID == "" {
			continue
		}
		if !isTerminalStatus(job.Status) {
			job.Status = StatusFailed
			job.Error = "backend restarted before job completed"
			job.FinishedAt = m.nowFunc().UTC().Format(time.RFC3339Nano)
			job.LastUpdatedAt = job.FinishedAt
		}
		jobCopy := cloneJob(&job)
		m.jobs[job.ID] = &jobCopy
	}
	return nil
}

func ParseJobSpecs(body []byte) ([]JobSpec, error) {
	trimmed := bytes.TrimSpace(body)
	if len(trimmed) == 0 {
		return nil, fmt.Errorf("%w: request body is empty", ErrInvalidJobRequest)
	}
	if trimmed[0] == '[' {
		var specs []JobSpec
		if err := json.Unmarshal(trimmed, &specs); err != nil {
			return nil, fmt.Errorf("%w: %v", ErrInvalidJobRequest, err)
		}
		return specs, nil
	}
	var wrapped struct {
		Jobs []JobSpec `json:"jobs"`
	}
	if err := json.Unmarshal(trimmed, &wrapped); err == nil && len(wrapped.Jobs) > 0 {
		return wrapped.Jobs, nil
	}
	var spec JobSpec
	if err := json.Unmarshal(trimmed, &spec); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidJobRequest, err)
	}
	return []JobSpec{spec}, nil
}
