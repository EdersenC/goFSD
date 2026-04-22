package main

import (
	"context"
	"embed"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
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

//go:embed web/index.html web/app.ts
var webAssets embed.FS

func main() {
	const backendBuildID = "2026-04-21-capture-failfast-v1"
	if len(os.Args) > 1 {
		switch os.Args[1] {
		case "process-runs":
			if err := runProcessRuns(os.Args[2:]); err != nil {
				log.Fatal(err)
			}
			return
		case "report-runs":
			if err := runReportRuns(os.Args[2:]); err != nil {
				log.Fatal(err)
			}
			return
		}
	}

	configPath, err := capture.ResolveInferenceConfigPath("")
	if err != nil {
		log.Printf("backend config path not resolved, using defaults: %v", err)
	}
	inferenceConfig, err := capture.LoadInferenceConfig(configPath)
	if err != nil {
		log.Fatalf("failed to load backend inference config: %v", err)
	}
	if inferenceConfig.ConfigPath != "" {
		log.Printf("loaded backend inference config from %s", inferenceConfig.ConfigPath)
	}
	log.Printf("backend build=%s capture_stop_output_validation=enabled", backendBuildID)
	actuatorConfig, err := actuator.LoadConfig(configPath)
	if err != nil {
		log.Fatalf("failed to load backend actuator config: %v", err)
	}
	datasetConfig, err := capture.LoadDatasetConfig(configPath)
	if err != nil {
		log.Fatalf("failed to load dataset frame-window config: %v", err)
	}

	svc := capture.NewService()
	controlStore := control.NewStore()
	actuatorService := actuator.NewService(actuatorConfig, configPath)
	inferencer := capture.NewInferencer(inferenceConfig, controlStore, actuatorService)
	if err := actuatorService.Start(); err != nil {
		log.Fatalf("failed to start virtual controller actuator: %v", err)
	}
	defer func() {
		if err := actuatorService.Close(); err != nil {
			log.Printf("failed to close virtual controller actuator: %v", err)
		}
	}()
	processor := datasetproc.NewProcessor(
		datasetproc.WithImageSize(datasetConfig.ImageWidth, datasetConfig.ImageHeight),
		datasetproc.WithSamplingConfig(datasetConfig.WindowSize, datasetConfig.FrameStride, datasetConfig.SampleStride),
		datasetproc.WithLabelTolerance(datasetConfig.LabelTolerance),
		datasetproc.WithDeltaSpeedTargetConfig(datasetConfig.DeltaSpeedClip, datasetConfig.DeltaSpeedNormalize),
		datasetproc.WithSyncFlashDetection(datasetConfig.SyncFlashBrightnessThreshold, datasetConfig.SyncFlashFrameLimit),
	)
	mux := http.NewServeMux()

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		if r.Method != http.MethodGet {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}

		writeEmbeddedFile(w, "web/index.html", "text/html; charset=utf-8")
	})

	mux.HandleFunc("/app.ts", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}

		writeEmbeddedFile(w, "web/app.ts", "text/javascript; charset=utf-8")
	})

	registerDataInspectorHandlers(mux, filepath.Join(defaultBackendDataRoot(), "runs"))

	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
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
			switch {
			case errors.Is(err, control.ErrInvalidCommand):
				writeError(w, http.StatusBadRequest, err.Error())
			default:
				writeError(w, http.StatusInternalServerError, err.Error())
			}
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

		writeJSON(w, http.StatusOK, map[string]any{
			"status":    "reset",
			"sessionId": controlStore.ResetConsumerSession(),
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
		} else if _, err := processor.Queue(tripDir); err != nil {
			postProcessStatus = "failed"
			postProcessError = err.Error()
		} else {
			go func(targetTripDir string) {
				readinessCtx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
				defer cancel()
				if err := datasetproc.WaitForTripReadiness(readinessCtx, targetTripDir, 500*time.Millisecond); err != nil {
					log.Printf("post-processing readiness failed for %s: %v", targetTripDir, err)
					return
				}
				if err := processor.ProcessTrip(context.Background(), targetTripDir); err != nil {
					log.Printf("post-processing failed for %s: %v", targetTripDir, err)
				}
			}(tripDir)
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

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	addr := ":" + port
	log.Printf("capture API listening on %s", addr)
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

	if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatal(err)
	}
	<-shutdownDone
}

func writeJSON(w http.ResponseWriter, status int, data any) {
	w.Header().Set("Content-Type", "application/json")
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

func writeEmbeddedFile(w http.ResponseWriter, name string, contentType string) {
	body, err := webAssets.ReadFile(name)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to read asset")
		return
	}

	w.Header().Set("Content-Type", contentType)
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(body)
}

func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, map[string]string{"error": message})
}

func runProcessRuns(args []string) error {
	fs := flag.NewFlagSet("process-runs", flag.ContinueOnError)
	root := fs.String("root", filepath.Join(defaultBackendDataRoot(), "runs"), "root directory to scan for trip folders")
	workers := fs.Int("workers", 4, "number of parallel workers")
	force := fs.Bool("force", false, "reprocess trips even if frames/ or dataset.jsonl already exist")
	datasetOnly := fs.Bool("dataset-only", false, "reuse existing frames/ and regenerate only dataset.jsonl")
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
	if len(tripDirs) == 0 {
		fmt.Println("No trip folders found.")
		return nil
	}

	fmt.Printf(
		"Processing %d trip folders from %s with workers=%d force=%t dataset_only=%t image_size=%dx%d window_size=%d frame_stride=%d sample_stride=%d label_tolerance=%s delta_speed_clip=%.3f delta_speed_normalize=%t\n",
		len(tripDirs),
		*root,
		*workers,
		*force,
		*datasetOnly,
		datasetConfig.ImageWidth,
		datasetConfig.ImageHeight,
		datasetConfig.WindowSize,
		datasetConfig.FrameStride,
		datasetConfig.SampleStride,
		datasetConfig.LabelTolerance,
		datasetConfig.DeltaSpeedClip,
		datasetConfig.DeltaSpeedNormalize,
	)

	var completed int
	var skipped int
	var failed int
	var processed int
	var active int

	results := datasetproc.ProcessTripDirsWithCallback(
		context.Background(),
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
		datasetproc.WithForce(*force),
		datasetproc.WithDatasetOnly(*datasetOnly),
		datasetproc.WithImageSize(datasetConfig.ImageWidth, datasetConfig.ImageHeight),
		datasetproc.WithSamplingConfig(datasetConfig.WindowSize, datasetConfig.FrameStride, datasetConfig.SampleStride),
		datasetproc.WithLabelTolerance(datasetConfig.LabelTolerance),
		datasetproc.WithDeltaSpeedTargetConfig(datasetConfig.DeltaSpeedClip, datasetConfig.DeltaSpeedNormalize),
		datasetproc.WithSyncFlashDetection(datasetConfig.SyncFlashBrightnessThreshold, datasetConfig.SyncFlashFrameLimit),
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

	fmt.Printf("Summary: completed=%d skipped=%d failed=%d total=%d\n", completed, skipped, failed, len(results))

	reports, reportErr := datasetproc.WriteRunDatasetReports(tripDirs, buildDatasetReportConfig(datasetConfig))
	printRunDatasetReportSummaries(reports)

	var processErr error
	if failed > 0 {
		processErr = fmt.Errorf("processing failed for %d trip(s)", failed)
	}
	if reportErr != nil {
		reportErr = fmt.Errorf("dataset report generation failed: %w", reportErr)
	}
	return errors.Join(processErr, reportErr)
}

func runReportRuns(args []string) error {
	fs := flag.NewFlagSet("report-runs", flag.ContinueOnError)
	root := fs.String("root", filepath.Join(defaultBackendDataRoot(), "runs"), "root directory to scan for trip folders")
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
	if len(tripDirs) == 0 {
		fmt.Println("No trip folders found.")
		return nil
	}

	fmt.Printf(
		"Reporting %d trip folders from %s image_size=%dx%d window_size=%d frame_stride=%d sample_stride=%d label_tolerance=%s delta_speed_clip=%.3f delta_speed_normalize=%t\n",
		len(tripDirs),
		*root,
		datasetConfig.ImageWidth,
		datasetConfig.ImageHeight,
		datasetConfig.WindowSize,
		datasetConfig.FrameStride,
		datasetConfig.SampleStride,
		datasetConfig.LabelTolerance,
		datasetConfig.DeltaSpeedClip,
		datasetConfig.DeltaSpeedNormalize,
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
	}
	return capture.LoadDatasetConfig(configPath)
}

func buildDatasetReportConfig(datasetConfig capture.DatasetConfig) datasetproc.DatasetReportConfig {
	return datasetproc.DatasetReportConfig{
		ImageWidth:          datasetConfig.ImageWidth,
		ImageHeight:         datasetConfig.ImageHeight,
		WindowSize:          datasetConfig.WindowSize,
		FrameStride:         datasetConfig.FrameStride,
		SampleStride:        datasetConfig.SampleStride,
		LabelTolerance:      datasetConfig.LabelTolerance.String(),
		DeltaSpeedClip:      datasetConfig.DeltaSpeedClip,
		DeltaSpeedNormalize: datasetConfig.DeltaSpeedNormalize,
	}
}

func printRunDatasetReportSummaries(reports []datasetproc.GeneratedRunDatasetReport) {
	for _, generated := range reports {
		summary := generated.Report.Summary
		fmt.Printf(
			"Report: run=%s trips=%d samples=%d stopped_share=%.3f delta_speed_clip_rate=%.3f path=%s\n",
			generated.RunID,
			summary.TripCount,
			summary.SampleCount,
			summary.StoppedSampleShare,
			summary.DeltaSpeedClip.AnyClipRate,
			generated.ReportPath,
		)
	}
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
