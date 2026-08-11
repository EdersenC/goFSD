package main

import (
	"errors"
	"net/http"
	"strings"

	"awesomeProject/internal/control"
	"awesomeProject/internal/stopsignbatch"
)

type stopSignBatchEnqueuer interface {
	Enqueue(control.CommandRequest) (control.Command, error)
}

func registerStopSignBatchHandlers(mux *http.ServeMux, commands stopSignBatchEnqueuer) {
	mux.HandleFunc("/control/stop-sign-batches", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}

		var request struct {
			stopsignbatch.Plan
			SafetyEpoch *uint64 `json:"safetyEpoch"`
		}
		if err := decodeJSONBody(r, &request); err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}
		jobs, err := stopsignbatch.Expand(request.Plan)
		if err != nil {
			if errors.Is(err, stopsignbatch.ErrInvalidPlan) {
				writeError(w, http.StatusBadRequest, err.Error())
				return
			}
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}
		planFingerprint, err := stopsignbatch.PlanFingerprint(request.Plan)
		if err != nil {
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}

		command, err := commands.Enqueue(control.CommandRequest{
			Type:            control.CommandStartStopSignBatch,
			SafetyEpoch:     request.SafetyEpoch,
			StopSignBatchID: strings.TrimSpace(request.Plan.ID),
			PlanFingerprint: planFingerprint,
			StopSignJobs:    controlStopSignBatchJobs(jobs),
		})
		if err != nil {
			writeControlCommandError(w, err)
			return
		}

		writeJSON(w, http.StatusOK, map[string]any{
			"status":          "queued",
			"planId":          strings.TrimSpace(request.Plan.ID),
			"planFingerprint": planFingerprint,
			"jobCount":        len(jobs),
			"jobs":            jobs,
			"command":         command,
		})
	})
}

func controlStopSignBatchJobs(jobs []stopsignbatch.Job) []control.StopSignBatchJob {
	return append([]control.StopSignBatchJob(nil), jobs...)
}
