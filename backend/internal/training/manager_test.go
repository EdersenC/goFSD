package training

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestManagerEnqueueAndCompleteJob(t *testing.T) {
	baseConfig := writeBaseTrainConfig(t)
	cfg, err := LoadConfig(baseConfig, t.TempDir())
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	manager, err := NewManager(cfg, baseConfig)
	if err != nil {
		t.Fatalf("NewManager: %v", err)
	}
	manager.newCommand = helperCommandFactory("success")

	learningRate := 0.0005
	jobs, err := manager.Enqueue([]JobSpec{{
		Name:         "job-a",
		LearningRate: &learningRate,
		LossWeights: map[string]float64{
			"future_yaw_delta": 2.2,
		},
		Consistency: map[string]float64{
			"future_speed_vs_delta_speed_weight": 0.7,
		},
	}})
	if err != nil {
		t.Fatalf("Enqueue: %v", err)
	}
	job := waitForTerminalJob(t, manager, jobs[0].ID, 5*time.Second)
	if job.Status != StatusCompleted {
		t.Fatalf("expected completed job, got %+v", job)
	}
	if job.RunDir == "" {
		t.Fatalf("expected run dir to be parsed from logs")
	}
	configBody, err := os.ReadFile(job.ConfigPath)
	if err != nil {
		t.Fatalf("read derived config: %v", err)
	}
	configText := string(configBody)
	if !strings.Contains(configText, "learning_rate = 0.0005") {
		t.Fatalf("derived config missing learning rate override: %s", configText)
	}
	if !strings.Contains(configText, "future_yaw_delta = 2.2") {
		t.Fatalf("derived config missing loss weight override: %s", configText)
	}
	if !strings.Contains(configText, "future_speed_vs_delta_speed_weight = 0.7") {
		t.Fatalf("derived config missing consistency override: %s", configText)
	}
	logBody, err := manager.ReadLog(job.ID)
	if err != nil {
		t.Fatalf("ReadLog: %v", err)
	}
	if !strings.Contains(logBody, "run_dir=") {
		t.Fatalf("expected run_dir in log, got %s", logBody)
	}
}

func TestLoadPageConfig(t *testing.T) {
	baseConfig := writeBaseTrainConfig(t)
	cfg, err := LoadConfig(baseConfig, t.TempDir())
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	page, err := LoadPageConfig(baseConfig, cfg)
	if err != nil {
		t.Fatalf("LoadPageConfig: %v", err)
	}
	if page.LearningRate != 0.001 {
		t.Fatalf("unexpected learning rate: %v", page.LearningRate)
	}
	if page.LossWeights["future_yaw_delta"] != 1.0 {
		t.Fatalf("unexpected loss weights: %+v", page.LossWeights)
	}
	if page.Consistency["future_speed_vs_delta_speed_weight"] != 0.5 {
		t.Fatalf("unexpected consistency defaults: %+v", page.Consistency)
	}
	if len(page.AllowedLossWeightKeys) == 0 || len(page.AllowedConsistencyKeys) == 0 {
		t.Fatalf("expected editable keys to be populated: %+v", page)
	}
}

func TestLoadConfigResolvesPythonCommandAndProjectRelativeTrainScript(t *testing.T) {
	baseConfig := writeBaseTrainConfigWithBackendSection(t, `
[backend]
[backend.training]
python_bin = "python"
train_script = "fsd_trainer/src/gta_fsd/train.py"
jobs_dir = "S:\\fsd_fivem_data\\training_jobs"
`)
	cfg, err := LoadConfig(baseConfig, t.TempDir())
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.PythonBin != "python" {
		t.Fatalf("expected python command to stay unresolved as command, got %q", cfg.PythonBin)
	}
	expectedScript := filepath.Join(filepath.Dir(baseConfig), "..", "fsd_trainer", "src", "gta_fsd", "train.py")
	expectedScript = filepath.Clean(expectedScript)
	if cfg.TrainScript != expectedScript {
		t.Fatalf("unexpected train script path: got=%q want=%q", cfg.TrainScript, expectedScript)
	}
}

func TestManagerCancelPendingJob(t *testing.T) {
	baseConfig := writeBaseTrainConfig(t)
	cfg, err := LoadConfig(baseConfig, t.TempDir())
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	manager, err := NewManager(cfg, baseConfig)
	if err != nil {
		t.Fatalf("NewManager: %v", err)
	}

	callCount := 0
	manager.newCommand = func(ctx context.Context, _ string, _ ...string) *exec.Cmd {
		callCount++
		action := "success"
		if callCount == 1 {
			action = "sleep"
		}
		cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=TestTrainingManagerHelperProcess", "--", action)
		cmd.Env = append(os.Environ(), "GO_WANT_TRAINING_HELPER_PROCESS=1")
		return cmd
	}

	jobs, err := manager.Enqueue([]JobSpec{{Name: "first"}, {Name: "second"}})
	if err != nil {
		t.Fatalf("Enqueue: %v", err)
	}
	time.Sleep(150 * time.Millisecond)
	canceled, err := manager.Cancel(jobs[1].ID)
	if err != nil {
		t.Fatalf("Cancel: %v", err)
	}
	if canceled.Status != StatusCanceled {
		t.Fatalf("expected canceled job, got %+v", canceled)
	}

	first := waitForTerminalJob(t, manager, jobs[0].ID, 5*time.Second)
	if first.Status != StatusStopped && first.Status != StatusCompleted {
		t.Fatalf("unexpected first job status: %+v", first)
	}
	second := waitForTerminalJob(t, manager, jobs[1].ID, 2*time.Second)
	if second.Status != StatusCanceled {
		t.Fatalf("expected second job canceled, got %+v", second)
	}
}

func TestManagerStopActiveJob(t *testing.T) {
	baseConfig := writeBaseTrainConfig(t)
	cfg, err := LoadConfig(baseConfig, t.TempDir())
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	manager, err := NewManager(cfg, baseConfig)
	if err != nil {
		t.Fatalf("NewManager: %v", err)
	}
	manager.newCommand = helperCommandFactory("sleep")

	jobs, err := manager.Enqueue([]JobSpec{{Name: "stop-me"}})
	if err != nil {
		t.Fatalf("Enqueue: %v", err)
	}

	waitForStatus(t, manager, jobs[0].ID, StatusRunning, 5*time.Second)
	if _, err := manager.Stop(jobs[0].ID); err != nil {
		t.Fatalf("Stop: %v", err)
	}
	job := waitForTerminalJob(t, manager, jobs[0].ID, 5*time.Second)
	if job.Status != StatusStopped {
		t.Fatalf("expected stopped job, got %+v", job)
	}
}

func TestParseJobSpecsSupportsSingleArrayAndWrapped(t *testing.T) {
	cases := []struct {
		name string
		body string
		want int
	}{
		{name: "single", body: `{"name":"a"}`, want: 1},
		{name: "array", body: `[{"name":"a"},{"name":"b"}]`, want: 2},
		{name: "wrapped", body: `{"jobs":[{"name":"a"},{"name":"b"}]}`, want: 2},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			specs, err := ParseJobSpecs([]byte(tc.body))
			if err != nil {
				t.Fatalf("ParseJobSpecs: %v", err)
			}
			if len(specs) != tc.want {
				t.Fatalf("unexpected spec count: got=%d want=%d", len(specs), tc.want)
			}
		})
	}
}

func TestManagerReadLogTail(t *testing.T) {
	baseConfig := writeBaseTrainConfig(t)
	cfg, err := LoadConfig(baseConfig, t.TempDir())
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	manager, err := NewManager(cfg, baseConfig)
	if err != nil {
		t.Fatalf("NewManager: %v", err)
	}
	manager.newCommand = helperCommandFactory("success")

	jobs, err := manager.Enqueue([]JobSpec{{Name: "tail-test"}})
	if err != nil {
		t.Fatalf("Enqueue: %v", err)
	}
	job := waitForTerminalJob(t, manager, jobs[0].ID, 5*time.Second)
	if job.Status != StatusCompleted {
		t.Fatalf("expected completed job, got %+v", job)
	}

	tail, err := manager.ReadLogTail(job.ID, 2)
	if err != nil {
		t.Fatalf("ReadLogTail: %v", err)
	}
	if !strings.Contains(tail, "run_dir=/tmp/fake-training-run") || !strings.Contains(tail, "run_metrics=/tmp/fake-training-run/run_metrics.json") {
		t.Fatalf("unexpected tail: %q", tail)
	}
	if strings.Contains(tail, "Using device: cpu") {
		t.Fatalf("tail should not include earlier lines: %q", tail)
	}
}

func TestTrainingManagerHelperProcess(t *testing.T) {
	if os.Getenv("GO_WANT_TRAINING_HELPER_PROCESS") != "1" {
		return
	}
	action := os.Args[len(os.Args)-1]
	fmt.Println("Using device: cpu")
	fmt.Println("run_dir=/tmp/fake-training-run")
	fmt.Println("run_metrics=/tmp/fake-training-run/run_metrics.json")
	switch action {
	case "success":
		os.Exit(0)
	case "sleep":
		time.Sleep(1 * time.Second)
		os.Exit(0)
	default:
		fmt.Fprintln(os.Stderr, "forced failure")
		os.Exit(3)
	}
}

func waitForStatus(t *testing.T, manager *Manager, id string, status string, timeout time.Duration) Job {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		job, err := manager.GetJob(id)
		if err == nil && job.Status == status {
			return job
		}
		time.Sleep(25 * time.Millisecond)
	}
	job, _ := manager.GetJob(id)
	t.Fatalf("timed out waiting for status %s, last=%+v", status, job)
	return Job{}
}

func waitForTerminalJob(t *testing.T, manager *Manager, id string, timeout time.Duration) Job {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		job, err := manager.GetJob(id)
		if err == nil && isTerminalStatus(job.Status) {
			return job
		}
		time.Sleep(25 * time.Millisecond)
	}
	job, _ := manager.GetJob(id)
	t.Fatalf("timed out waiting for terminal job, last=%+v", job)
	return Job{}
}

func helperCommandFactory(action string) commandFactory {
	return func(ctx context.Context, _ string, _ ...string) *exec.Cmd {
		cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=TestTrainingManagerHelperProcess", "--", action)
		cmd.Env = append(os.Environ(), "GO_WANT_TRAINING_HELPER_PROCESS=1")
		return cmd
	}
}

func writeBaseTrainConfig(t *testing.T) string {
	t.Helper()
	return writeBaseTrainConfigWithBackendSection(t, `
[backend]
[backend.training]
`)
}

func writeBaseTrainConfigWithBackendSection(t *testing.T, backendSection string) string {
	t.Helper()
	root := t.TempDir()
	projectRoot := filepath.Join(root, "project")
	configDir := filepath.Join(projectRoot, "fsd_trainer")
	if err := os.MkdirAll(configDir, 0o755); err != nil {
		t.Fatalf("mkdir config dir: %v", err)
	}
	trainScriptPath := filepath.Join(projectRoot, "fsd_trainer", "src", "gta_fsd")
	if err := os.MkdirAll(trainScriptPath, 0o755); err != nil {
		t.Fatalf("mkdir train script dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(trainScriptPath, "train.py"), []byte("print('ok')\n"), 0o644); err != nil {
		t.Fatalf("write train script: %v", err)
	}
	path := filepath.Join(configDir, "train_config.toml")
	body := backendSection + `
[dataset]
frame_stride = 2
sample_stride = 2
window_size = 3
image_width = 224
image_height = 224
label_tolerance = "100ms"
delta_speed_clip = 2.0
delta_speed_normalize = true
sync_flash_brightness_threshold = 245.0
sync_flash_frame_limit = 90

[output]
base_dir = "fsd_trainer/training_runs"

[training]
device = "cpu"
epochs = 1
learning_rate = 0.001

[training.loss_weights]
future_yaw_delta = 1.0

[training.consistency]
future_speed_vs_delta_speed_weight = 0.5
`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatalf("write base config: %v", err)
	}
	return path
}
