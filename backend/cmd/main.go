package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"awesomeProject/internal/actuator"
	"awesomeProject/internal/capture"
	"awesomeProject/internal/control"
	datasetproc "awesomeProject/internal/dataset"
)

func main() {
	if err := runBackend(os.Args[1:], os.Stdout); err != nil {
		log.Fatal(err)
	}
}

func runBackend(args []string, output io.Writer) error {
	const backendBuildID = "2026-08-09-safety-barrier-v9"
	handled, err := dispatchBackendCommand(args, output)
	if err != nil {
		return err
	}
	if handled {
		return nil
	}
	addr := backendListenAddress(os.Getenv("HOST"), os.Getenv("PORT"))
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		return fmt.Errorf("failed to bind capture API on %s: %w", addr, err)
	}
	defer listener.Close()

	configPath, err := capture.ResolveInferenceConfigPath("")
	if err != nil {
		log.Printf("backend config path not resolved, using defaults: %v", err)
	}
	inferenceConfig, err := capture.LoadInferenceConfig(configPath)
	if err != nil {
		return fmt.Errorf("failed to load backend inference config: %w", err)
	}
	if inferenceConfig.ConfigPath != "" {
		log.Printf("loaded backend inference config from %s", inferenceConfig.ConfigPath)
	}
	log.Printf("backend build=%s capture_stop_output_validation=enabled", backendBuildID)
	actuatorConfig, err := actuator.LoadConfig(configPath)
	if err != nil {
		return fmt.Errorf("failed to load backend actuator config: %w", err)
	}
	datasetConfig, err := capture.LoadDatasetConfig(configPath)
	if err != nil {
		return fmt.Errorf("failed to load dataset frame-window config: %w", err)
	}

	svc := capture.NewService()
	controlStore := control.NewStore()
	actuatorService := actuator.NewService(actuatorConfig, configPath, controlStore)
	inferencer := capture.NewInferencer(inferenceConfig, actuatorConfig, controlStore, actuatorService)
	trainingProxyBaseURL := strings.TrimRight(strings.TrimSpace(inferenceConfig.ModelServerURL), "/")
	trainingProxyClient := &http.Client{Timeout: inferenceConfig.RequestTimeout}
	if err := actuatorService.Start(); err != nil {
		log.Printf("virtual controller actuator unavailable; expert collection remains available: %v", err)
	}
	defer func() {
		if err := actuatorService.Close(); err != nil {
			log.Printf("failed to close virtual controller actuator: %v", err)
		}
	}()
	runsRoot := filepath.Join(defaultBackendDataRoot(), "runs")
	processingService, err := newDatasetProcessingService(datasetConfig, runsRoot)
	if err != nil {
		return fmt.Errorf("failed to configure dataset processing service: %w", err)
	}
	processingContext, cancelProcessing := context.WithCancel(context.Background())
	defer cancelProcessing()
	if err := processingService.Start(processingContext); err != nil {
		return fmt.Errorf("failed to start dataset processing service: %w", err)
	}
	defer func() {
		if err := processingService.Close(); err != nil {
			log.Printf("failed to close dataset processing service: %v", err)
		}
	}()
	processingSnapshot := processingService.Snapshot()
	log.Printf(
		"dataset processing workers=%d queue_capacity=%d recovered=%d",
		processingSnapshot.WorkerCount,
		processingSnapshot.QueueCapacity,
		processingSnapshot.Recovered,
	)
	mux := http.NewServeMux()
	datasetProcessor := datasetproc.NewProcessor(datasetProcessorOptions(datasetConfig)...)
	registerWebHandlers(mux)
	registerProcessingHandlers(mux, processingService, datasetProcessor, runsRoot)
	registerDataInspectorHandlers(mux, runsRoot, datasetProcessor)
	registerStopSignBatchHandlers(mux, controlStore)
	registerControlDispatchHandlers(mux, controlStore)

	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		writeJSON(w, http.StatusOK, backendHealthResponse())
	})

	mux.HandleFunc("/capture/sources", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}

		sources, err := svc.DiscoverSources(r.Context())
		if err != nil {
			switch {
			case errors.Is(err, capture.ErrUnsupportedPlatform):
				writeError(w, http.StatusNotImplemented, "windows-only in v1")
			default:
				writeError(w, http.StatusInternalServerError, err.Error())
			}
			return
		}

		writeJSON(w, http.StatusOK, map[string]any{"sources": sources})
	})

	mux.HandleFunc("/capture/snapshot", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		var req capture.SnapshotRequest
		if r.ContentLength > 0 {
			if err := decodeJSONBody(r, &req); err != nil {
				writeError(w, http.StatusBadRequest, err.Error())
				return
			}
		}
		result, err := svc.Snapshot(r.Context(), req)
		if err != nil {
			switch {
			case errors.Is(err, capture.ErrUnsupportedPlatform):
				writeError(w, http.StatusNotImplemented, "windows-only in v1")
			case errors.Is(err, capture.ErrUnsupportedFFmpeg):
				writeError(w, http.StatusFailedDependency, err.Error())
			case errors.Is(err, capture.ErrAlreadyRunning):
				writeError(w, http.StatusConflict, err.Error())
			case errors.Is(err, capture.ErrInvalidRequest), errors.Is(err, capture.ErrSourceNotFound):
				writeError(w, http.StatusBadRequest, err.Error())
			default:
				writeError(w, http.StatusInternalServerError, err.Error())
			}
			return
		}
		writeJSON(w, http.StatusOK, result)
	})

	mux.HandleFunc("/inference/status", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}

		writeJSON(w, http.StatusOK, inferencer.Status())
	})

	mux.HandleFunc("/inference/models", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		models, err := inferencer.Models(r.Context(), r.URL.Query().Get("modelServerUrl"))
		if err != nil {
			writeError(w, http.StatusBadGateway, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"models": models})
	})

	mux.HandleFunc("/inference/model/load", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		var req capture.InferenceModelLoadRequest
		if r.ContentLength > 0 {
			if err := decodeJSONBody(r, &req); err != nil {
				writeError(w, http.StatusBadRequest, err.Error())
				return
			}
		}
		res, err := inferencer.LoadModel(r.Context(), req)
		if err != nil {
			if errors.Is(err, capture.ErrInferenceAlreadyRunning) {
				writeError(w, http.StatusConflict, err.Error())
				return
			}
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, res)
	})

	mux.HandleFunc("/inference/start", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}

		var req capture.InferenceStartRequest
		if r.ContentLength > 0 {
			if err := decodeJSONBody(r, &req); err != nil {
				writeError(w, http.StatusBadRequest, err.Error())
				return
			}
		}

		res, err := inferencer.Start(r.Context(), req)
		if err != nil {
			switch {
			case errors.Is(err, capture.ErrUnsupportedPlatform):
				writeError(w, http.StatusNotImplemented, "windows-only in v1")
			case errors.Is(err, capture.ErrUnsupportedFFmpeg):
				writeError(w, http.StatusFailedDependency, err.Error())
			case errors.Is(err, capture.ErrInferenceAlreadyRunning):
				writeError(w, http.StatusConflict, err.Error())
			case errors.Is(err, capture.ErrSourceNotFound), errors.Is(err, capture.ErrInferenceStartFailed):
				writeError(w, http.StatusBadRequest, err.Error())
			default:
				writeError(w, http.StatusInternalServerError, err.Error())
			}
			return
		}

		writeJSON(w, http.StatusOK, res)
	})

	mux.HandleFunc("/inference/stop", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}

		res, err := inferencer.Stop(r.Context())
		if err != nil {
			switch {
			case errors.Is(err, capture.ErrInferenceNotRunning):
				writeError(w, http.StatusNotFound, err.Error())
			default:
				writeError(w, http.StatusInternalServerError, err.Error())
			}
			return
		}

		writeJSON(w, http.StatusOK, res)
	})

	mux.HandleFunc("/actuator/state", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}

		state := actuatorService.State()
		if !state.Supported {
			writeJSON(w, http.StatusNotImplemented, state)
			return
		}
		if !state.Ready {
			writeJSON(w, http.StatusServiceUnavailable, state)
			return
		}

		writeJSON(w, http.StatusOK, state)
	})

	mux.HandleFunc("/actuator/command", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}

		var req actuator.CommandRequest
		if err := decodeJSONBody(r, &req); err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}

		state, err := actuatorService.Submit(req)
		if err != nil {
			switch {
			case errors.Is(err, actuator.ErrUnsupportedPlatform):
				writeJSON(w, http.StatusNotImplemented, state)
			case errors.Is(err, actuator.ErrInvalidInputMode):
				writeError(w, http.StatusBadRequest, err.Error())
			case errors.Is(err, actuator.ErrNotReady):
				writeJSON(w, http.StatusServiceUnavailable, state)
			default:
				writeError(w, http.StatusInternalServerError, err.Error())
			}
			return
		}

		writeJSON(w, http.StatusOK, map[string]any{
			"status": "accepted",
			"state":  state,
		})
	})

	mux.HandleFunc("/actuator/tuning", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		writeJSON(w, http.StatusOK, actuatorService.TuningState())
	})

	mux.HandleFunc("/actuator/tuning/apply", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}

		var req actuator.Tuning
		if err := decodeJSONBody(r, &req); err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}

		state, err := actuatorService.ApplyTuning(req)
		if err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, state)
	})

	mux.HandleFunc("/actuator/tuning/save", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}

		state, err := actuatorService.SaveTuning()
		if err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, state)
	})

	mux.HandleFunc("/actuator/tuning/reset", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}

		writeJSON(w, http.StatusOK, actuatorService.ResetTuning())
	})

	mux.HandleFunc("/control/state", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}

		writeJSON(w, http.StatusOK, controlStore.State())
	})

	mux.HandleFunc("/control/command", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}

		var req control.CommandRequest
		if err := decodeJSONBody(r, &req); err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}

		command, err := controlStore.Enqueue(req)
		if err != nil {
			writeControlCommandError(w, err)
			return
		}

		writeJSON(w, http.StatusOK, map[string]any{
			"status":  "queued",
			"command": command,
		})
	})

	mux.HandleFunc("/control/poll", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}

		command := controlStore.Poll(r.URL.Query().Get("lastSeenCommandId"))
		writeJSON(w, http.StatusOK, map[string]any{"command": command})
	})

	mux.HandleFunc("/control/status", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}

		var req control.StatusUpdate
		if err := decodeJSONBody(r, &req); err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}

		writeJSON(w, http.StatusOK, controlStore.UpdateStatus(req))
	})

	mux.HandleFunc("/control/telemetry", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}

		var req control.TelemetryUpdate
		if err := decodeJSONBody(r, &req); err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}

		writeJSON(w, http.StatusOK, map[string]any{
			"status":    "ok",
			"telemetry": controlStore.UpdateTelemetry(req),
		})
	})

	mux.HandleFunc("/control/connect", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}

		reset := controlStore.ResetConsumerSessionWithSafetyEpoch()
		writeJSON(w, http.StatusOK, map[string]any{
			"status":      "reset",
			"sessionId":   reset.SessionID,
			"safetyEpoch": reset.SafetyEpoch,
		})
	})

	mux.HandleFunc("/control/scenes", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}

		var req struct {
			Scenes []control.SceneOption `json:"scenes"`
		}
		if err := decodeJSONBody(r, &req); err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}

		writeJSON(w, http.StatusOK, map[string]any{
			"scenes": controlStore.SetAvailableScenes(req.Scenes),
		})
	})

	mux.HandleFunc("/training/state", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		proxyTrainingRequest(w, r, trainingProxyClient, trainingProxyBaseURL)
	})

	mux.HandleFunc("/training/config", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		proxyTrainingRequest(w, r, trainingProxyClient, trainingProxyBaseURL)
	})

	mux.HandleFunc("/training/jobs", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet && r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		proxyTrainingRequest(w, r, trainingProxyClient, trainingProxyBaseURL)
	})

	mux.HandleFunc("/training/history/clear", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		proxyTrainingRequest(w, r, trainingProxyClient, trainingProxyBaseURL)
	})

	mux.HandleFunc("/training/jobs/", func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet, http.MethodPost:
			proxyTrainingRequest(w, r, trainingProxyClient, trainingProxyBaseURL)
			return
		default:
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
	})

	mux.HandleFunc("/capture/start", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}

		var req capture.StartRequest
		if err := decodeJSONBody(r, &req); err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}

		res, err := svc.Start(r.Context(), req)
		if err != nil {
			switch {
			case errors.Is(err, capture.ErrUnsupportedPlatform):
				writeError(w, http.StatusNotImplemented, "windows-only in v1")
			case errors.Is(err, capture.ErrUnsupportedFFmpeg):
				writeError(w, http.StatusFailedDependency, err.Error())
			case errors.Is(err, capture.ErrInvalidRequest), errors.Is(err, capture.ErrSourceNotFound):
				writeError(w, http.StatusBadRequest, err.Error())
			case errors.Is(err, capture.ErrAlreadyRunning):
				writeError(w, http.StatusConflict, err.Error())
			default:
				writeError(w, http.StatusInternalServerError, err.Error())
			}
			return
		}

		writeJSON(w, http.StatusOK, res)
	})

	mux.HandleFunc("/capture/stop", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}

		var req struct {
			RunID           string `json:"runId"`
			TripIndex       int    `json:"tripIndex"`
			SceneID         string `json:"sceneId"`
			SceneVariant    string `json:"sceneVariant"`
			SceneName       string `json:"sceneName"`
			SkipPostProcess bool   `json:"skipPostProcess"`
			AbortOnly       bool   `json:"abortOnly"`
		}
		if err := decodeJSONBody(r, &req); err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}
		log.Printf("capture stop request run=%s scene=%s variant=%s trip=%d legacy_skip_post_process=%t",
			req.RunID,
			req.SceneID,
			req.SceneVariant,
			req.TripIndex,
			req.SkipPostProcess,
		)

		var res capture.StopResult
		var err error
		if req.AbortOnly {
			log.Printf("capture abort request run=%s scene=%s variant=%s trip=%d", req.RunID, req.SceneID, req.SceneVariant, req.TripIndex)
			res, err = svc.ForceStop(r.Context())
		} else {
			res, err = svc.Stop(r.Context())
		}
		if err != nil {
			switch {
			case errors.Is(err, capture.ErrNotRunning):
				writeError(w, http.StatusNotFound, err.Error())
			default:
				writeError(w, http.StatusInternalServerError, err.Error())
			}
			return
		}

		tripDir := filepath.Dir(res.OutputFile)
		processingPath := filepath.Join(tripDir, "processing.json")
		postProcessStatus := "queued"
		postProcessError := ""
		log.Printf("capture stop success run=%s scene=%s variant=%s trip=%d output=%s bytes=%d",
			req.RunID,
			req.SceneID,
			req.SceneVariant,
			req.TripIndex,
			res.OutputFile,
			res.OutputBytes,
		)
		if req.AbortOnly {
			postProcessStatus = "aborted"
			postProcessError = "capture aborted after hard failure; post-processing skipped"
		} else if accepted, enqueueErr := processingService.Enqueue(tripDir); enqueueErr != nil {
			postProcessStatus = "failed"
			postProcessError = enqueueErr.Error()
			log.Printf("post-processing enqueue failed for %s: %v", tripDir, enqueueErr)
		} else {
			if !accepted {
				log.Printf("post-processing already queued or active for %s", tripDir)
			}
		}

		writeJSON(w, http.StatusOK, map[string]any{
			"status":            res.Status,
			"sessionId":         res.SessionID,
			"outputFile":        res.OutputFile,
			"logFile":           res.LogFile,
			"outputBytes":       res.OutputBytes,
			"postProcessStatus": postProcessStatus,
			"processingFile":    processingPath,
			"postProcessError":  postProcessError,
		})
	})

	log.Printf("capture API listening on %s", listener.Addr())
	server := &http.Server{
		Addr:    addr,
		Handler: mux,
	}

	shutdownDone := make(chan struct{})
	go func() {
		defer close(shutdownDone)
		signals := make(chan os.Signal, 1)
		signal.Notify(signals, os.Interrupt, syscall.SIGTERM)
		defer signal.Stop(signals)

		<-signals
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		if err := server.Shutdown(ctx); err != nil {
			log.Printf("http shutdown failed: %v", err)
		}
	}()

	if err := server.Serve(listener); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return fmt.Errorf("capture API stopped unexpectedly: %w", err)
	}
	<-shutdownDone
	return nil
}

func backendHealthResponse() map[string]string {
	return map[string]string{
		"status":  "ok",
		"service": "stop-sign-lab-backend",
	}
}

func writeJSON(w http.ResponseWriter, status int, data any) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(data)
}

func decodeJSONBody(r *http.Request, dest any) error {
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(dest); err != nil {
		return errors.New("invalid json body")
	}
	return nil
}

func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, map[string]string{"error": message})
}

func writeControlCommandError(w http.ResponseWriter, err error) {
	status := http.StatusInternalServerError
	switch {
	case errors.Is(err, control.ErrSafetyEpochRequired):
		status = http.StatusPreconditionRequired
	case errors.Is(err, control.ErrSafetyEpochMismatch):
		status = http.StatusConflict
	case errors.Is(err, control.ErrInvalidCommand):
		status = http.StatusBadRequest
	}
	writeError(w, status, err.Error())
}

func runProcessRuns(args []string) error {
	fs := flag.NewFlagSet("process-runs", flag.ContinueOnError)
	root := fs.String("root", filepath.Join(defaultBackendDataRoot(), "runs"), "root directory to scan for trip folders")
	workers := fs.Int("workers", 4, "number of parallel workers")
	force := fs.Bool("force", false, "reprocess trips even when their fingerprint and published outputs are complete")
	datasetOnly := fs.Bool("dataset-only", false, "reuse existing frames/ and regenerate only dataset.jsonl")
	stopSignOnly := fs.Bool("stop-sign-only", false, "process only stop-sign temporal-v1 scenes; reports retain every scene in affected runs")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *workers < 1 {
		return fmt.Errorf("workers must be at least 1")
	}

	datasetConfig, err := loadDatasetConfigForCLI()
	if err != nil {
		return fmt.Errorf("load dataset frame-window config: %w", err)
	}

	tripDirs, err := datasetproc.CollectTripDirs(*root, fs.Args())
	if err != nil {
		return err
	}
	reportTripDirs := tripDirs
	if *stopSignOnly {
		tripDirs = filterStopSignTripDirs(tripDirs)
		var collectErr error
		reportTripDirs, collectErr = collectTripDirsForSelectedRuns(tripDirs)
		if collectErr != nil {
			return collectErr
		}
	}
	if len(tripDirs) == 0 {
		if *stopSignOnly {
			fmt.Println("No stop-sign trip folders found.")
		} else {
			fmt.Println("No trip folders found.")
		}
		return nil
	}

	fmt.Printf(
		"Processing %d trip folders from %s with workers=%d stop_sign_only=%t force=%t dataset_only=%t image_size=%dx%d image_offsets=%v telemetry_offsets=%v future_offsets=%v telemetry_interval=%s sample_stride=%d label_tolerance=%s\n",
		len(tripDirs),
		*root,
		*workers,
		*stopSignOnly,
		*force,
		*datasetOnly,
		datasetConfig.ImageWidth,
		datasetConfig.ImageHeight,
		datasetConfig.ImageOffsets,
		datasetConfig.TelemetryOffsets,
		datasetConfig.FutureOffsets,
		datasetConfig.TelemetrySampleInterval,
		datasetConfig.SampleStride,
		datasetConfig.LabelTolerance,
	)

	var completed int
	var skipped int
	var failed int
	var processed int
	var active int

	processingContext, cancelProcessing := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancelProcessing()
	processorOptions := append(
		datasetProcessorOptions(datasetConfig),
		datasetproc.WithForce(*force),
		datasetproc.WithDatasetOnly(*datasetOnly),
	)
	results := datasetproc.ProcessTripDirsWithCallback(
		processingContext,
		tripDirs,
		*workers,
		func(result datasetproc.TripProcessResult) {
			if result.Event == "started" {
				active++
				fmt.Printf(
					"[start] worker=%d active=%d/%d trip=%s\n",
					result.WorkerID,
					active,
					*workers,
					result.TripDir,
				)
				return
			}

			processed++
			if active > 0 {
				active--
			}
			switch result.State {
			case "completed":
				completed++
			case "skipped":
				skipped++
			default:
				if result.Error != nil {
					failed++
				} else {
					completed++
				}
			}

			statusLabel := strings.ToUpper(result.State)
			if result.Error != nil {
				fmt.Printf(
					"[%d/%d] FAIL worker=%d active=%d/%d trip=%s elapsed=%s :: %v\n",
					processed,
					len(tripDirs),
					result.WorkerID,
					active,
					*workers,
					result.TripDir,
					result.Duration.Round(time.Millisecond),
					result.Error,
				)
				return
			}

			fmt.Printf(
				"[%d/%d] %s worker=%d active=%d/%d trip=%s elapsed=%s (completed=%d skipped=%d failed=%d)\n",
				processed,
				len(tripDirs),
				statusLabel,
				result.WorkerID,
				active,
				*workers,
				result.TripDir,
				result.Duration.Round(time.Millisecond),
				completed,
				skipped,
				failed,
			)
		},
		processorOptions...,
	)

	if processed != len(results) {
		completed = 0
		skipped = 0
		failed = 0
		for _, result := range results {
			switch result.State {
			case "completed":
				completed++
			case "skipped":
				skipped++
			default:
				if result.Error != nil {
					failed++
				} else {
					completed++
				}
			}
		}
	}

	fmt.Printf(
		"Summary: completed=%d skipped=%d failed=%d finished=%d discovered=%d\n",
		completed,
		skipped,
		failed,
		len(results),
		len(tripDirs),
	)

	processErr := processRunError(len(tripDirs), len(results), failed, processingContext.Err())
	var reportErr error
	if processingContext.Err() == nil {
		var reports []datasetproc.GeneratedRunDatasetReport
		reports, reportErr = datasetproc.WriteRunDatasetReports(reportTripDirs, buildDatasetReportConfig(datasetConfig))
		printRunDatasetReportSummaries(reports)
	}
	if reportErr != nil {
		reportErr = fmt.Errorf("dataset report generation failed: %w", reportErr)
	}
	return errors.Join(processErr, reportErr)
}

func processRunError(discovered int, finished int, failed int, contextErr error) error {
	if contextErr != nil {
		return fmt.Errorf("processing interrupted after %d of %d trip(s): %w", finished, discovered, contextErr)
	}
	if finished != discovered {
		return fmt.Errorf("processing ended after %d of %d trip(s)", finished, discovered)
	}
	if failed > 0 {
		return fmt.Errorf("processing failed for %d trip(s)", failed)
	}
	return nil
}

func runReportRuns(args []string) error {
	fs := flag.NewFlagSet("report-runs", flag.ContinueOnError)
	root := fs.String("root", filepath.Join(defaultBackendDataRoot(), "runs"), "root directory to scan for trip folders")
	stopSignOnly := fs.Bool("stop-sign-only", false, "report runs containing stop-sign temporal-v1 scenes, retaining every scene in those runs")
	if err := fs.Parse(args); err != nil {
		return err
	}

	datasetConfig, err := loadDatasetConfigForCLI()
	if err != nil {
		return fmt.Errorf("load dataset frame-window config: %w", err)
	}

	tripDirs, err := datasetproc.CollectTripDirs(*root, fs.Args())
	if err != nil {
		return err
	}
	if *stopSignOnly {
		stopSignTripDirs := filterStopSignTripDirs(tripDirs)
		var collectErr error
		tripDirs, collectErr = collectTripDirsForSelectedRuns(stopSignTripDirs)
		if collectErr != nil {
			return collectErr
		}
	}
	if len(tripDirs) == 0 {
		if *stopSignOnly {
			fmt.Println("No stop-sign trip folders found.")
		} else {
			fmt.Println("No trip folders found.")
		}
		return nil
	}

	fmt.Printf(
		"Reporting %d trip folders from %s stop_sign_only=%t image_size=%dx%d image_offsets=%v telemetry_offsets=%v future_offsets=%v telemetry_interval=%s sample_stride=%d label_tolerance=%s\n",
		len(tripDirs),
		*root,
		*stopSignOnly,
		datasetConfig.ImageWidth,
		datasetConfig.ImageHeight,
		datasetConfig.ImageOffsets,
		datasetConfig.TelemetryOffsets,
		datasetConfig.FutureOffsets,
		datasetConfig.TelemetrySampleInterval,
		datasetConfig.SampleStride,
		datasetConfig.LabelTolerance,
	)

	reports, reportErr := datasetproc.WriteRunDatasetReports(tripDirs, buildDatasetReportConfig(datasetConfig))
	printRunDatasetReportSummaries(reports)
	if reportErr != nil {
		return fmt.Errorf("dataset report generation failed: %w", reportErr)
	}
	return nil
}

func loadDatasetConfigForCLI() (capture.DatasetConfig, error) {
	configPath, err := capture.ResolveInferenceConfigPath("")
	if err != nil {
		log.Printf("backend config path not resolved, using default dataset frame window: %v", err)
	} else {
		log.Printf("loaded dataset config from %s", configPath)
	}
	return capture.LoadDatasetConfig(configPath)
}

func buildDatasetReportConfig(datasetConfig capture.DatasetConfig) datasetproc.DatasetReportConfig {
	return datasetproc.DatasetReportConfig{
		ImageWidth:                datasetConfig.ImageWidth,
		ImageHeight:               datasetConfig.ImageHeight,
		WindowSize:                datasetConfig.WindowSize,
		FrameStride:               datasetConfig.FrameStride,
		ImageOffsets:              append([]int(nil), datasetConfig.ImageOffsets...),
		TelemetryOffsets:          append([]int(nil), datasetConfig.TelemetryOffsets...),
		FutureOffsets:             append([]int(nil), datasetConfig.FutureOffsets...),
		TelemetrySampleIntervalMs: int(datasetConfig.TelemetrySampleInterval / time.Millisecond),
		SampleStride:              datasetConfig.SampleStride,
		LabelTolerance:            datasetConfig.LabelTolerance.String(),
	}
}

func printRunDatasetReportSummaries(reports []datasetproc.GeneratedRunDatasetReport) {
	for _, generated := range reports {
		summary := generated.Report.Summary
		fmt.Printf(
			"Report: run=%s trips=%d samples=%d stopped_share=%.3f path=%s\n",
			generated.RunID,
			summary.TripCount,
			summary.SampleCount,
			summary.StoppedSampleShare,
			generated.ReportPath,
		)
	}
}

func proxyTrainingRequest(w http.ResponseWriter, r *http.Request, client *http.Client, baseURL string) {
	baseURL = strings.TrimRight(strings.TrimSpace(baseURL), "/")
	if baseURL == "" {
		writeError(w, http.StatusBadGateway, "python model server url is not configured")
		return
	}

	targetURL := baseURL + r.URL.Path
	if rawQuery := strings.TrimSpace(r.URL.RawQuery); rawQuery != "" {
		targetURL += "?" + rawQuery
	}

	var body []byte
	if r.Body != nil {
		readBody, err := io.ReadAll(r.Body)
		if err != nil {
			writeError(w, http.StatusBadRequest, "failed to read request body")
			return
		}
		body = readBody
	}

	req, err := http.NewRequestWithContext(r.Context(), r.Method, targetURL, strings.NewReader(string(body)))
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if contentType := strings.TrimSpace(r.Header.Get("Content-Type")); contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}

	resp, err := client.Do(req)
	if err != nil {
		writeError(w, http.StatusBadGateway, fmt.Sprintf("python training proxy failed: %v", err))
		return
	}
	defer resp.Body.Close()

	if contentType := strings.TrimSpace(resp.Header.Get("Content-Type")); contentType != "" {
		w.Header().Set("Content-Type", contentType)
	}
	w.WriteHeader(resp.StatusCode)
	_, _ = io.Copy(w, resp.Body)
}

func defaultBackendDataRoot() string {
	if value := capture.NormalizeDataRoot(os.Getenv("FSD_DATA_ROOT")); value != "" {
		return value
	}
	if os.PathSeparator == '\\' {
		return `S:\fsd_fivem_data`
	}
	return "/mnt/s/fsd_fivem_data"
}

func backendListenAddress(host string, port string) string {
	host = strings.TrimSpace(host)
	if host == "" {
		host = "127.0.0.1"
	}
	port = strings.TrimSpace(port)
	if port == "" {
		port = "8080"
	}
	return net.JoinHostPort(host, port)
}
