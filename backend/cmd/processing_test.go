package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"awesomeProject/internal/capture"
	datasetproc "awesomeProject/internal/dataset"
)

type reconcileEnqueueResult struct {
	accepted bool
	err      error
}

type reconcileServiceStub struct {
	results map[string]reconcileEnqueueResult
	calls   []string
}

func (s *reconcileServiceStub) Snapshot() datasetproc.ProcessingSnapshot {
	return datasetproc.ProcessingSnapshot{}
}

func (s *reconcileServiceStub) Enqueue(tripDir string) (bool, error) {
	s.calls = append(s.calls, tripDir)
	result, ok := s.results[tripDir]
	if !ok {
		return false, errors.New("unexpected reconcile enqueue")
	}
	return result.accepted, result.err
}

func TestDatasetProcessorOptionsPreserveExactImageOffsets(t *testing.T) {
	config := capture.DefaultDatasetConfig()
	config.WindowSize = 3
	config.FrameStride = 4
	config.ImageOffsets = []int{-8, -3, 0}

	configured := datasetproc.NewProcessor(datasetProcessorOptions(config)...)
	expected := datasetproc.NewProcessor(
		datasetproc.WithImageSize(config.ImageWidth, config.ImageHeight),
		datasetproc.WithSamplingConfig(config.WindowSize, config.FrameStride, config.SampleStride),
		datasetproc.WithImageOffsets(config.ImageOffsets),
		datasetproc.WithLabelTolerance(config.LabelTolerance),
		datasetproc.WithTelemetryTimelineConfig(config.TelemetryOffsets, config.FutureOffsets, config.TelemetrySampleInterval),
		datasetproc.WithSyncFlashDetection(config.SyncFlashBrightnessThreshold, config.SyncFlashFrameLimit),
	)
	legacyDerived := datasetproc.NewProcessor(
		datasetproc.WithSamplingConfig(config.WindowSize, config.FrameStride, config.SampleStride),
	)

	if configured.ConfigFingerprint() != expected.ConfigFingerprint() {
		t.Fatal("shared processor options did not preserve the configured timeline")
	}
	if configured.ConfigFingerprint() == legacyDerived.ConfigFingerprint() {
		t.Fatal("nonuniform image offsets collapsed to the legacy window/stride timeline")
	}
}

func TestDatasetReportConfigPreservesExactTimelines(t *testing.T) {
	config := capture.DefaultDatasetConfig()
	config.ImageOffsets = []int{-8, -3, 0}
	config.TelemetryOffsets = []int{-5, -2, 0}
	config.FutureOffsets = []int{1, 3, 5}

	reportConfig := buildDatasetReportConfig(config)
	if !reflect.DeepEqual(reportConfig.ImageOffsets, config.ImageOffsets) ||
		!reflect.DeepEqual(reportConfig.TelemetryOffsets, config.TelemetryOffsets) ||
		!reflect.DeepEqual(reportConfig.FutureOffsets, config.FutureOffsets) {
		t.Fatalf("report lost exact timeline configuration: %+v", reportConfig)
	}
	if reportConfig.TelemetrySampleIntervalMs != int(config.TelemetrySampleInterval.Milliseconds()) {
		t.Fatalf("unexpected report telemetry interval: %d", reportConfig.TelemetrySampleIntervalMs)
	}
}

func TestProcessingStateHandlerReturnsServiceSnapshot(t *testing.T) {
	runsRoot := t.TempDir()
	config := capture.DefaultDatasetConfig()
	service, err := newDatasetProcessingService(config, runsRoot)
	if err != nil {
		t.Fatalf("newDatasetProcessingService: %v", err)
	}
	if err := service.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}
	t.Cleanup(func() { _ = service.Close() })

	mux := http.NewServeMux()
	processor := datasetproc.NewProcessor(datasetProcessorOptions(config)...)
	fingerprint := processor.ConfigFingerprint()
	registerProcessingHandlers(mux, service, processor, runsRoot)

	response := httptest.NewRecorder()
	mux.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/processing/state", nil))
	if response.Code != http.StatusOK {
		t.Fatalf("GET status: got=%d body=%s", response.Code, response.Body.String())
	}
	var state processingStateResponse
	if err := json.Unmarshal(response.Body.Bytes(), &state); err != nil {
		t.Fatalf("decode snapshot: %v", err)
	}
	if state.WorkerCount != liveProcessingWorkers || state.QueueCapacity != liveProcessingQueueCapacity {
		t.Fatalf("unexpected service snapshot: %+v", state.ProcessingSnapshot)
	}
	if state.ConfigFingerprint != fingerprint {
		t.Fatalf("unexpected processing fingerprint: got=%q want=%q", state.ConfigFingerprint, fingerprint)
	}

	response = httptest.NewRecorder()
	mux.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/processing/state", nil))
	if response.Code != http.StatusMethodNotAllowed {
		t.Fatalf("POST status: got=%d body=%s", response.Code, response.Body.String())
	}
}

func TestProcessingReadinessUsesAuthoritativePublishedOutputCheck(t *testing.T) {
	runsRoot := t.TempDir()
	config := capture.DefaultDatasetConfig()
	processor := datasetproc.NewProcessor(datasetProcessorOptions(config)...)
	currentTrip := filepath.Join(runsRoot, "run-a", stopSignSceneFolder, "trip-000")
	writeCurrentProcessingFixture(t, currentTrip, config, processor, 1)
	truncatedTrip := filepath.Join(runsRoot, "run-a", stopSignSceneFolder, "trip-001")
	writeCurrentProcessingFixture(t, truncatedTrip, config, processor, 1)
	if err := os.WriteFile(filepath.Join(truncatedTrip, "dataset.jsonl"), nil, 0o644); err != nil {
		t.Fatalf("truncate dataset: %v", err)
	}
	legacyTrip := filepath.Join(runsRoot, "run-old", "inner-city-driving_default", "trip-000")
	writeCurrentProcessingFixture(t, legacyTrip, config, processor, 1)

	summary, err := buildProcessingReadinessSummary(runsRoot, true, processor)
	if err != nil {
		t.Fatalf("buildProcessingReadinessSummary: %v", err)
	}
	if summary.RunCount != 2 || summary.TotalTripCount != 3 || summary.TotalDatasetCount != 3 {
		t.Fatalf("unexpected total counts: %+v", summary)
	}
	if summary.SelectedTripCount != 2 || summary.SelectedDatasetCount != 2 ||
		summary.CurrentTripCount != 1 || summary.UnreadyDatasetCount != 1 ||
		summary.ExcludedTripCount != 1 {
		t.Fatalf("truncated output was not excluded from current count: %+v", summary)
	}
	if summary.TrainingEligibleTripCount != 0 || summary.TrainingSampleCount != 0 ||
		len(summary.TrainingEligibleRunIDs) != 0 || summary.TrainingReady {
		t.Fatalf("trip without successful stop-sign metadata must not be offered to training: %+v", summary)
	}

	mux := http.NewServeMux()
	service, err := newDatasetProcessingService(config, runsRoot)
	if err != nil {
		t.Fatalf("newDatasetProcessingService: %v", err)
	}
	registerProcessingHandlers(mux, service, processor, runsRoot)
	response := httptest.NewRecorder()
	mux.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/processing/readiness", nil))
	if response.Code != http.StatusOK {
		t.Fatalf("GET readiness: got=%d body=%s", response.Code, response.Body.String())
	}
	var fromAPI processingReadinessSummary
	if err := json.Unmarshal(response.Body.Bytes(), &fromAPI); err != nil {
		t.Fatalf("decode readiness: %v", err)
	}
	if !reflect.DeepEqual(fromAPI, summary) {
		t.Fatalf("API readiness drifted from shared summary: got=%+v want=%+v", fromAPI, summary)
	}
}

func TestProcessingReadinessRequiresTwoIndependentEligibleRunsForTraining(t *testing.T) {
	runsRoot := t.TempDir()
	config := capture.DefaultDatasetConfig()
	processor := datasetproc.NewProcessor(datasetProcessorOptions(config)...)
	writeSuccessfulStopSignStageSetFixture(t, runsRoot, "run-a", 1, []int{3, 1, 1}, config, processor)
	writeSuccessfulStopSignStageSetFixture(t, runsRoot, "run-b", 101, []int{5, 1, 1}, config, processor)

	summary, err := buildProcessingReadinessSummary(runsRoot, true, processor)
	if err != nil {
		t.Fatalf("buildProcessingReadinessSummary: %v", err)
	}
	if !summary.TrainingReady || summary.TrainingEligibleTripCount != 2 || summary.TrainingSampleCount != 12 {
		t.Fatalf("two independent nonempty runs should be training-ready: %+v", summary)
	}
	if !reflect.DeepEqual(summary.TrainingClipStageCounts, map[string]int{"approach": 2, "brake_stop": 2, "release": 2}) {
		t.Fatalf("every continuous run must expose all three logical stages: %+v", summary)
	}
	if !reflect.DeepEqual(summary.TrainingEligibleRunIDs, []string{"run-a", "run-b"}) ||
		!reflect.DeepEqual(summary.ReadyRunIDs, []string{"run-a", "run-b"}) {
		t.Fatalf("ready run ids should be stable and sorted: %+v", summary)
	}
	if summary.TrainingLocationCount != 2 || len(summary.SuggestedTrainRunIDs) != 1 || len(summary.SuggestedValRunIDs) != 1 {
		t.Fatalf("training split must isolate physical stop locations: %+v", summary)
	}
}

func TestProcessingReadinessRejectsRunOnlySplitAtOnePhysicalLocation(t *testing.T) {
	runsRoot := t.TempDir()
	config := capture.DefaultDatasetConfig()
	processor := datasetproc.NewProcessor(datasetProcessorOptions(config)...)
	for _, runID := range []string{"run-a", "run-b"} {
		writeSuccessfulStopSignStageSetFixture(t, runsRoot, runID, 1, []int{3, 1, 1}, config, processor)
	}

	summary, err := buildProcessingReadinessSummary(runsRoot, true, processor)
	if err != nil {
		t.Fatalf("buildProcessingReadinessSummary: %v", err)
	}
	if summary.TrainingReady || summary.TrainingLocationCount != 1 ||
		len(summary.SuggestedTrainRunIDs) != 0 || len(summary.SuggestedValRunIDs) != 0 {
		t.Fatalf("two runs at one location must not be presented as a valid generalization split: %+v", summary)
	}
}

func TestProcessingReadinessMatchesTrainerStopSignOutcomeFilter(t *testing.T) {
	runsRoot := t.TempDir()
	config := capture.DefaultDatasetConfig()
	processor := datasetproc.NewProcessor(datasetProcessorOptions(config)...)
	failedTrip := filepath.Join(runsRoot, "run-a", stopSignSceneFolder, "trip-000")
	successfulTrip := filepath.Join(runsRoot, "run-b", stopSignSceneFolder, "trip-000")
	malformedTrip := filepath.Join(runsRoot, "run-c", stopSignSceneFolder, "trip-000")
	writeCurrentProcessingFixture(t, failedTrip, config, processor, 7)
	writeCurrentProcessingFixture(t, successfulTrip, config, processor, 11)
	writeCurrentProcessingFixture(t, malformedTrip, config, processor, 13)
	writeSuccessfulStopSignMetadataFixture(t, failedTrip)
	writeSuccessfulStopSignMetadataFixture(t, successfulTrip)
	writeSuccessfulStopSignMetadataFixture(t, malformedTrip)
	writeCommandJSONFile(t, filepath.Join(failedTrip, "metadata.json"), map[string]any{
		"sceneId": "stop-sign", "sceneVariant": "continuous-v2",
		"stopSignGoal": map[string]any{
			"task": "stop-sign", "contract": "stop-sign-goal.v3", "captureMode": "continuous",
			"logicalClipStages": []string{"approach", "brake_stop", "release"},
			"signPose":          map[string]any{"x": 1, "y": 2, "z": 3, "heading": 4},
			"stopLinePose":      map[string]any{"x": 1, "y": 0, "z": 3, "heading": 4},
			"egoStopPose":       map[string]any{"x": 1, "y": -2.5, "z": 3, "heading": 4},
			"startPose":         map[string]any{"x": 1, "y": -42.5, "z": 3, "heading": 4},
			"exitPose":          map[string]any{"x": 1, "y": 10, "z": 3, "heading": 4},
		},
		"stopSignOutcome": map[string]any{"success": false, "status": "collision"},
	})
	if err := os.WriteFile(filepath.Join(malformedTrip, "metadata.json"), []byte("{"), 0o644); err != nil {
		t.Fatalf("write malformed metadata: %v", err)
	}

	summary, err := buildProcessingReadinessSummary(runsRoot, true, processor)
	if err != nil {
		t.Fatalf("buildProcessingReadinessSummary: %v", err)
	}
	if summary.TrainingEligibleTripCount != 1 || summary.TrainingSampleCount != 11 ||
		!reflect.DeepEqual(summary.TrainingEligibleRunIDs, []string{"run-b"}) || summary.TrainingReady {
		t.Fatalf("training readiness did not match the trainer success filter: %+v", summary)
	}
	if summary.TrainingEligibilityErrors != 1 {
		t.Fatalf("malformed metadata should remain visible as a readiness error: %+v", summary)
	}
}

func TestProcessingReconcileQueuesOnlyStopSignTrips(t *testing.T) {
	runsRoot := t.TempDir()
	config := capture.DefaultDatasetConfig()
	processor := datasetproc.NewProcessor(datasetProcessorOptions(config)...)
	stopSignScene := filepath.Join(runsRoot, "run-a", stopSignSceneFolder)

	currentTrip := filepath.Join(stopSignScene, "trip-000")
	writeCurrentProcessingFixture(t, currentTrip, config, processor, 1)
	incompleteTrip := filepath.Join(stopSignScene, "trip-001")
	writeIncompleteRawProcessingFixture(t, incompleteTrip)
	queuedTrip := filepath.Join(stopSignScene, "trip-002")
	writeCompleteRawProcessingFixture(t, queuedTrip)
	alreadyQueuedTrip := filepath.Join(stopSignScene, "trip-003")
	writeCompleteRawProcessingFixture(t, alreadyQueuedTrip)
	failedTrip := filepath.Join(stopSignScene, "trip-004")
	writeCompleteRawProcessingFixture(t, failedTrip)
	legacyTrip := filepath.Join(runsRoot, "run-a", "inner-city-driving_default", "trip-005")
	writeCompleteRawProcessingFixture(t, legacyTrip)

	service := &reconcileServiceStub{results: map[string]reconcileEnqueueResult{
		queuedTrip:        {accepted: true},
		alreadyQueuedTrip: {accepted: false},
		failedTrip:        {err: errors.New("queue unavailable")},
	}}
	mux := http.NewServeMux()
	registerProcessingHandlers(mux, service, processor, runsRoot)

	response := httptest.NewRecorder()
	mux.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/processing/reconcile", nil))
	if response.Code != http.StatusOK {
		t.Fatalf("POST reconcile: got=%d body=%s", response.Code, response.Body.String())
	}
	var result processingReconcileResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatalf("decode reconcile response: %v", err)
	}
	if result.Selected != 5 || result.Current != 1 || result.Queued != 1 ||
		result.AlreadyQueued != 1 || result.Incomplete != 1 || result.Failed != 1 {
		t.Fatalf("unexpected reconcile counts: %+v", result)
	}
	if len(result.Failures) != 1 ||
		result.Failures[0].Trip != filepath.ToSlash(filepath.Join("run-a", stopSignSceneFolder, "trip-004")) ||
		result.Failures[0].Error != "queue unavailable" {
		t.Fatalf("unexpected reconcile failures: %+v", result.Failures)
	}
	wantCalls := []string{queuedTrip, alreadyQueuedTrip, failedTrip}
	if !reflect.DeepEqual(service.calls, wantCalls) {
		t.Fatalf("unexpected reconcile enqueue calls: got=%v want=%v", service.calls, wantCalls)
	}
}

func TestProcessingReconcileRejectsUnsupportedMethodAndInput(t *testing.T) {
	runsRoot := t.TempDir()
	config := capture.DefaultDatasetConfig()
	processor := datasetproc.NewProcessor(datasetProcessorOptions(config)...)
	service := &reconcileServiceStub{results: make(map[string]reconcileEnqueueResult)}
	mux := http.NewServeMux()
	registerProcessingHandlers(mux, service, processor, runsRoot)

	tests := []struct {
		name   string
		method string
		target string
		body   *bytes.Reader
		status int
	}{
		{name: "method", method: http.MethodGet, target: "/processing/reconcile", body: bytes.NewReader(nil), status: http.StatusMethodNotAllowed},
		{name: "body", method: http.MethodPost, target: "/processing/reconcile", body: bytes.NewReader([]byte(`{"root":"elsewhere"}`)), status: http.StatusBadRequest},
		{name: "query", method: http.MethodPost, target: "/processing/reconcile?root=elsewhere", body: bytes.NewReader(nil), status: http.StatusBadRequest},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			response := httptest.NewRecorder()
			request := httptest.NewRequest(test.method, test.target, test.body)
			mux.ServeHTTP(response, request)
			if response.Code != test.status {
				t.Fatalf("unexpected status: got=%d want=%d body=%s", response.Code, test.status, response.Body.String())
			}
		})
	}
	if len(service.calls) != 0 {
		t.Fatalf("rejected reconcile requests must not enqueue trips: %v", service.calls)
	}
}

func TestRunProcessingStatusWritesMachineReadableSummary(t *testing.T) {
	runsRoot := t.TempDir()
	configPath := writeCLIConfig(t)
	t.Setenv("FSD_CONFIG_PATH", configPath)
	config, err := capture.LoadDatasetConfig(configPath)
	if err != nil {
		t.Fatalf("LoadDatasetConfig: %v", err)
	}
	processor := datasetproc.NewProcessor(datasetProcessorOptions(config)...)
	tripDir := filepath.Join(runsRoot, "run-a", stopSignSceneFolder, "trip-000")
	writeCurrentProcessingFixture(t, tripDir, config, processor, 0)

	var output bytes.Buffer
	if err := runProcessingStatus([]string{"-root", runsRoot, "-stop-sign-only"}, &output); err != nil {
		t.Fatalf("runProcessingStatus: %v", err)
	}
	var summary processingReadinessSummary
	if err := json.Unmarshal(output.Bytes(), &summary); err != nil {
		t.Fatalf("decode processing status output: %v", err)
	}
	if summary.Scope != "stop-sign-continuous-v2" || summary.CurrentTripCount != 1 || summary.SelectedDatasetCount != 1 {
		t.Fatalf("unexpected CLI summary: %+v", summary)
	}
}

func writeCurrentProcessingFixture(
	t *testing.T,
	tripDir string,
	config capture.DatasetConfig,
	processor *datasetproc.Processor,
	sampleCount int,
) {
	t.Helper()
	framesDir := filepath.Join(tripDir, "frames")
	if err := os.MkdirAll(framesDir, 0o755); err != nil {
		t.Fatalf("mkdir current output fixture: %v", err)
	}
	if err := os.WriteFile(filepath.Join(framesDir, "000001.jpg"), []byte("frame"), 0o644); err != nil {
		t.Fatalf("write current output frame: %v", err)
	}
	rows := make([]map[string]any, 0, sampleCount)
	for index := 0; index < sampleCount; index++ {
		rows = append(rows, map[string]any{"sample": index})
	}
	writeJSONLinesFile(t, filepath.Join(tripDir, "dataset.jsonl"), rows)
	writeCommandJSONFile(t, filepath.Join(tripDir, "processing.json"), datasetproc.ProcessingStatus{
		State:                     "completed",
		ConfigFingerprint:         processor.ConfigFingerprint(),
		ImageWidth:                config.ImageWidth,
		ImageHeight:               config.ImageHeight,
		ImageOffsets:              append([]int(nil), config.ImageOffsets...),
		TelemetryOffsets:          append([]int(nil), config.TelemetryOffsets...),
		FutureOffsets:             append([]int(nil), config.FutureOffsets...),
		TelemetrySampleIntervalMs: float64(config.TelemetrySampleInterval) / float64(time.Millisecond),
		FrameCount:                1,
		SampleCount:               sampleCount,
	})
}

func writeSuccessfulStopSignMetadataFixture(t *testing.T, tripDir string) {
	t.Helper()
	writeSuccessfulStopSignMetadataFixtureAt(t, tripDir, 1)
}

func writeSuccessfulStopSignMetadataFixtureAt(t *testing.T, tripDir string, signX float64) {
	t.Helper()
	writeCommandJSONFile(t, filepath.Join(tripDir, "metadata.json"), map[string]any{
		"sceneId": "stop-sign", "sceneVariant": "continuous-v2",
		"stopSignGoal": map[string]any{
			"task": "stop-sign", "contract": "stop-sign-goal.v3", "captureMode": "continuous",
			"logicalClipStages": []string{"approach", "brake_stop", "release"},
			"signPose":          map[string]any{"x": signX, "y": 2, "z": 3, "heading": 4},
			"stopLinePose":      map[string]any{"x": 1, "y": 0, "z": 3, "heading": 4},
			"egoStopPose":       map[string]any{"x": 1, "y": -2.5, "z": 3, "heading": 4},
			"startPose":         map[string]any{"x": 1, "y": -42.5, "z": 3, "heading": 4},
			"exitPose":          map[string]any{"x": 1, "y": 10, "z": 3, "heading": 4},
		},
		"stopSignOutcome": map[string]any{
			"success": true,
			"status":  "succeeded",
			"stageTransitions": map[string]any{
				"approachStartedGameTimeMs":  1000,
				"brakeStopStartedGameTimeMs": 2000,
				"stopConfirmedGameTimeMs":    3000,
				"releaseStartedGameTimeMs":   3000,
				"completedGameTimeMs":        4000,
			},
		},
	})
}

func writeSuccessfulStopSignStageSetFixture(
	t *testing.T,
	runsRoot string,
	runID string,
	signX float64,
	sampleCounts []int,
	config capture.DatasetConfig,
	processor *datasetproc.Processor,
) {
	t.Helper()
	if len(sampleCounts) != 3 {
		t.Fatalf("continuous fixture requires approach, brake-stop, and release sample counts, got %d", len(sampleCounts))
	}
	totalSamples := 0
	for _, count := range sampleCounts {
		totalSamples += count
	}
	tripDir := filepath.Join(runsRoot, runID, stopSignSceneFolder, "trip-000")
	writeCurrentProcessingFixture(t, tripDir, config, processor, totalSamples)
	writeSuccessfulStopSignMetadataFixtureAt(t, tripDir, signX)
}

func writeIncompleteRawProcessingFixture(t *testing.T, tripDir string) {
	t.Helper()
	if err := os.MkdirAll(tripDir, 0o755); err != nil {
		t.Fatalf("mkdir incomplete raw processing fixture: %v", err)
	}
	if err := os.WriteFile(filepath.Join(tripDir, "video.mkv"), []byte("video"), 0o644); err != nil {
		t.Fatalf("write incomplete raw processing video: %v", err)
	}
}

func writeCompleteRawProcessingFixture(t *testing.T, tripDir string) {
	t.Helper()
	writeIncompleteRawProcessingFixture(t, tripDir)
	if err := os.WriteFile(filepath.Join(tripDir, "metadata.json"), []byte("{}"), 0o644); err != nil {
		t.Fatalf("write raw processing metadata: %v", err)
	}
	if err := os.WriteFile(filepath.Join(filepath.Dir(tripDir), "run.jsonl"), []byte("{}\n"), 0o644); err != nil {
		t.Fatalf("write raw processing run manifest: %v", err)
	}
}
