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
	"path/filepath"
	"strings"

	"awesomeProject/internal/capture"
	"awesomeProject/internal/control"
	datasetproc "awesomeProject/internal/dataset"
)

//go:embed web/index.html web/app.ts
var webAssets embed.FS

func main() {
	if len(os.Args) > 1 && os.Args[1] == "process-runs" {
		if err := runProcessRuns(os.Args[2:]); err != nil {
			log.Fatal(err)
		}
		return
	}

	svc := capture.NewService()
	processor := datasetproc.NewProcessor()
	controlStore := control.NewStore()
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

		res, err := svc.Stop(r.Context())
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
		if _, err := processor.Queue(tripDir); err != nil {
			postProcessStatus = "failed"
			postProcessError = err.Error()
		} else {
			go func(targetTripDir string) {
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
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatal(err)
	}
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
	if err := fs.Parse(args); err != nil {
		return err
	}

	tripDirs, err := datasetproc.CollectTripDirs(*root, fs.Args())
	if err != nil {
		return err
	}
	if len(tripDirs) == 0 {
		fmt.Println("No trip folders found.")
		return nil
	}

	results := datasetproc.ProcessTripDirs(
		context.Background(),
		tripDirs,
		*workers,
		datasetproc.WithForce(*force),
	)

	var completed int
	var skipped int
	var failed int
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

		if result.Error != nil {
			fmt.Printf("FAIL %s :: %v\n", result.TripDir, result.Error)
			continue
		}
		fmt.Printf("%s %s\n", strings.ToUpper(result.State), result.TripDir)
	}

	fmt.Printf("Summary: completed=%d skipped=%d failed=%d total=%d\n", completed, skipped, failed, len(results))
	if failed > 0 {
		return fmt.Errorf("processing failed for %d trip(s)", failed)
	}
	return nil
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
