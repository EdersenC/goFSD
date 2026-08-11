package main

import (
	"errors"
	"net/http"
	"strings"

	"awesomeProject/internal/control"
	"awesomeProject/internal/parkingbatch"
)

type parkingBatchEnqueuer interface {
	Enqueue(control.CommandRequest) (control.Command, error)
}

func registerParkingBatchHandlers(mux *http.ServeMux, commands parkingBatchEnqueuer) {
	mux.HandleFunc("/control/parking-batches", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}

		var request struct {
			parkingbatch.Plan
			SafetyEpoch *uint64 `json:"safetyEpoch"`
		}
		if err := decodeJSONBody(r, &request); err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}
		plan := request.Plan
		jobs, err := parkingbatch.Expand(plan)
		if err != nil {
			if errors.Is(err, parkingbatch.ErrInvalidPlan) {
				writeError(w, http.StatusBadRequest, err.Error())
				return
			}
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}
		planFingerprint := parkingbatch.PlanFingerprint(plan)

		command, err := commands.Enqueue(control.CommandRequest{
			Type:            control.CommandStartParkingBatch,
			SafetyEpoch:     request.SafetyEpoch,
			ParkingBatchID:  strings.TrimSpace(plan.ID),
			PlanFingerprint: planFingerprint,
			ParkingJobs:     controlParkingBatchJobs(jobs),
		})
		if err != nil {
			writeControlCommandError(w, err)
			return
		}

		writeJSON(w, http.StatusOK, map[string]any{
			"status":          "queued",
			"planId":          strings.TrimSpace(plan.ID),
			"planFingerprint": planFingerprint,
			"jobCount":        len(jobs),
			"jobs":            jobs,
			"command":         command,
		})
	})
}

func controlParkingBatchJobs(jobs []parkingbatch.Job) []control.ParkingBatchJob {
	result := make([]control.ParkingBatchJob, 0, len(jobs))
	for _, job := range jobs {
		result = append(result, control.ParkingBatchJob{
			ID:               job.ID,
			ParkDest:         controlParkingPose(job.ParkDest),
			StartDest:        controlParkingPose(job.StartDest),
			CollectionAmount: job.CollectionAmount,
			Seed:             job.Seed,
		})
	}
	return result
}

func controlParkingPose(pose parkingbatch.Pose) control.ParkingPose {
	return control.ParkingPose{
		X:       pose.X,
		Y:       pose.Y,
		Z:       pose.Z,
		Heading: pose.Heading,
	}
}
