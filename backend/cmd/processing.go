package main

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"awesomeProject/internal/capture"
	datasetproc "awesomeProject/internal/dataset"
)

type processingStateResponse struct {
	datasetproc.ProcessingSnapshot
	ConfigFingerprint string `json:"configFingerprint"`
}

type processingTripEnqueuer interface {
	Enqueue(tripDir string) (bool, error)
}

type processingServiceAPI interface {
	processingTripEnqueuer
	Snapshot() datasetproc.ProcessingSnapshot
}

type processingTripOutputChecker interface {
	TripOutputsCurrent(tripDir string) bool
}

type processingProcessorAPI interface {
	processingTripOutputChecker
	ConfigFingerprint() string
}

type processingReconcileFailure struct {
	Trip  string `json:"trip"`
	Error string `json:"error"`
}

type processingReconcileResponse struct {
	Selected      int                          `json:"selected"`
	Current       int                          `json:"current"`
	Queued        int                          `json:"queued"`
	AlreadyQueued int                          `json:"alreadyQueued"`
	Incomplete    int                          `json:"incomplete"`
	Failed        int                          `json:"failed"`
	Failures      []processingReconcileFailure `json:"failures"`
}

const (
	liveProcessingWorkers       = 2
	liveProcessingQueueCapacity = 128
)

func datasetProcessorOptions(config capture.DatasetConfig) []datasetproc.Option {
	return []datasetproc.Option{
		datasetproc.WithImageSize(config.ImageWidth, config.ImageHeight),
		datasetproc.WithSamplingConfig(config.WindowSize, config.FrameStride, config.SampleStride),
		datasetproc.WithImageOffsets(config.ImageOffsets),
		datasetproc.WithLabelTolerance(config.LabelTolerance),
		datasetproc.WithTelemetryTimelineConfig(config.TelemetryOffsets, config.FutureOffsets, config.TelemetrySampleInterval),
		datasetproc.WithFutureSpeedDeltaTargetConfig(config.FutureSpeedDeltaClip, config.FutureSpeedDeltaNormalize),
		datasetproc.WithSyncFlashDetection(config.SyncFlashBrightnessThreshold, config.SyncFlashFrameLimit),
	}
}

func newDatasetProcessingService(
	config capture.DatasetConfig,
	runsRoot string,
) (*datasetproc.ProcessingService, error) {
	return datasetproc.NewDefaultProcessingService(
		datasetproc.ProcessingServiceConfig{
			Workers:               liveProcessingWorkers,
			QueueCapacity:         liveProcessingQueueCapacity,
			RunsRoot:              runsRoot,
			RecoveryTripPredicate: isStopSignTripDir,
		},
		datasetProcessorOptions(config)...,
	)
}

func datasetProcessingConfigFingerprint(config capture.DatasetConfig) string {
	return datasetproc.NewProcessor(datasetProcessorOptions(config)...).ConfigFingerprint()
}

func runProcessingFingerprint(output io.Writer) error {
	if output == nil {
		return fmt.Errorf("processing fingerprint output must not be nil")
	}
	config, err := loadDatasetConfigForCLI()
	if err != nil {
		return err
	}
	_, err = fmt.Fprintln(output, datasetProcessingConfigFingerprint(config))
	return err
}

func registerProcessingHandlers(
	mux *http.ServeMux,
	service processingServiceAPI,
	processor processingProcessorAPI,
	runsRoot string,
) {
	if mux == nil {
		panic("processing handlers require a non-nil HTTP mux")
	}
	if service == nil {
		panic("processing handlers require a non-nil service")
	}
	if processor == nil {
		panic("processing handlers require a non-nil processor")
	}

	mux.HandleFunc("/processing/state", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		writeJSON(w, http.StatusOK, processingStateResponse{
			ProcessingSnapshot: service.Snapshot(),
			ConfigFingerprint:  processor.ConfigFingerprint(),
		})
	})

	mux.HandleFunc("/processing/readiness", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		summary, err := buildProcessingReadinessSummary(runsRoot, true, processor)
		if err != nil {
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, summary)
	})

	mux.HandleFunc("/processing/reconcile", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		if r.URL.RawQuery != "" || requestHasBody(r) {
			writeError(w, http.StatusBadRequest, "processing reconcile does not accept query parameters or a request body")
			return
		}
		result, err := reconcileStopSignProcessing(runsRoot, processor, service)
		if err != nil {
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, result)
	})
}

func requestHasBody(r *http.Request) bool {
	return r.Body != nil && r.Body != http.NoBody && r.ContentLength != 0
}

func reconcileStopSignProcessing(
	runsRoot string,
	processor processingTripOutputChecker,
	service processingTripEnqueuer,
) (processingReconcileResponse, error) {
	result := processingReconcileResponse{
		Failures: make([]processingReconcileFailure, 0),
	}
	if processor == nil {
		return result, errors.New("processing reconcile requires a processor")
	}
	if service == nil {
		return result, errors.New("processing reconcile requires a service")
	}

	tripDirs, err := collectStopSignProcessingTripDirs(runsRoot)
	if err != nil {
		return result, err
	}
	result.Selected = len(tripDirs)
	for _, tripDir := range tripDirs {
		if processor.TripOutputsCurrent(tripDir) {
			result.Current++
			continue
		}

		complete, readinessErr := processingRawInputsComplete(tripDir)
		if readinessErr != nil {
			result.addFailure(runsRoot, tripDir, readinessErr)
			continue
		}
		if !complete {
			result.Incomplete++
			continue
		}

		accepted, enqueueErr := service.Enqueue(tripDir)
		if enqueueErr != nil {
			result.addFailure(runsRoot, tripDir, enqueueErr)
			continue
		}
		if accepted {
			result.Queued++
			continue
		}
		result.AlreadyQueued++
	}
	return result, nil
}

func collectStopSignProcessingTripDirs(runsRoot string) ([]string, error) {
	if strings.TrimSpace(runsRoot) == "" {
		return nil, errors.New("processing reconcile requires a configured runs root")
	}
	tripDirs, err := datasetproc.CollectTripDirs(runsRoot, nil)
	if errors.Is(err, os.ErrNotExist) {
		return []string{}, nil
	}
	if err != nil {
		return nil, fmt.Errorf("scan stop-sign processing trips: %w", err)
	}
	return filterStopSignTripDirs(tripDirs), nil
}

func processingRawInputsComplete(tripDir string) (bool, error) {
	requiredFiles := []string{
		filepath.Join(tripDir, "video.mkv"),
		filepath.Join(tripDir, "metadata.json"),
		filepath.Join(filepath.Dir(tripDir), "run.jsonl"),
	}
	for _, path := range requiredFiles {
		info, err := os.Stat(path)
		if errors.Is(err, os.ErrNotExist) {
			return false, nil
		}
		if err != nil {
			return false, fmt.Errorf("inspect %s: %w", filepath.Base(path), err)
		}
		if !info.Mode().IsRegular() {
			return false, nil
		}
	}
	return true, nil
}

func (r *processingReconcileResponse) addFailure(runsRoot string, tripDir string, err error) {
	r.Failed++
	r.Failures = append(r.Failures, processingReconcileFailure{
		Trip:  processingTripLabel(runsRoot, tripDir),
		Error: err.Error(),
	})
}

func processingTripLabel(runsRoot string, tripDir string) string {
	relative, err := filepath.Rel(runsRoot, tripDir)
	if err != nil || relative == "." || relative == ".." || strings.HasPrefix(relative, ".."+string(filepath.Separator)) {
		return filepath.Base(tripDir)
	}
	return filepath.ToSlash(relative)
}
