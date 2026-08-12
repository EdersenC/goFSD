package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"awesomeProject/internal/control"
)

func TestStopSignBatchHandlerQueuesOneDeterministicCommand(t *testing.T) {
	store := control.NewStore()
	mux := http.NewServeMux()
	registerStopSignBatchHandlers(mux, store)
	req := httptest.NewRequest(http.MethodPost, "/control/stop-sign-batches", bytes.NewReader(validStopSignBatchPayload(store.State().SafetyEpoch)))
	res := httptest.NewRecorder()

	mux.ServeHTTP(res, req)

	if res.Code != http.StatusOK {
		t.Fatalf("unexpected status=%d body=%s", res.Code, res.Body.String())
	}
	var body struct {
		JobCount        int                        `json:"jobCount"`
		PlanFingerprint string                     `json:"planFingerprint"`
		Jobs            []control.StopSignBatchJob `json:"jobs"`
		Command         control.Command            `json:"command"`
	}
	if err := json.Unmarshal(res.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if body.JobCount != 2 || len(body.Jobs) != 2 || body.Command.Type != control.CommandStartStopSignBatch {
		t.Fatalf("unexpected response: %+v", body)
	}
	if body.Command.StopSignBatchID != "city-stops" || body.PlanFingerprint == "" || body.Command.PlanFingerprint != body.PlanFingerprint {
		t.Fatalf("command lost plan identity: %+v", body.Command)
	}
	if len(body.Command.StopSignJobs) != 2 || body.Command.StopSignJobs[1].ID != "alta:rain-blue" || body.Command.StopSignJobs[1].Seed != "fresh-stop-data:alta:rain-blue" {
		t.Fatalf("unexpected deterministic jobs: %+v", body.Command.StopSignJobs)
	}
	job := body.Command.StopSignJobs[1]
	if !body.Command.StopSignJobs[0].VariationProfile.Baseline || body.Command.StopSignJobs[0].VariationProfile.ChangeCount != 0 {
		t.Fatalf("first job must be the explicit variation baseline: %+v", body.Command.StopSignJobs[0].VariationProfile)
	}
	if job.VariationProfile.Baseline || job.VariationProfile.ChangeCount < 5 || job.VariationProfile.CombinationMagnitudePct <= 0 {
		t.Fatalf("varied job must report a measured multi-axis combination: %+v", job.VariationProfile)
	}
	if job.StopLinePose.X != 104 || job.EgoStopPose.X != 107 || job.StartPose.X != 167 || job.ExitPose.X != 92 || job.StopLinePose.Y != 200 || job.EgoCenterOffsetM != 3 || job.ExitDistanceM != 8 || job.TargetSpeedMPS != 7.5 || job.StopConfirmationMS != 500 || job.AttemptCount != 4 {
		t.Fatalf("unexpected derived/varied job: %+v", job)
	}
	if job.BrakingDecelerationMPS2 != 3.8 || job.ReleaseAccelerationMPS2 != 3.5 {
		t.Fatalf("manual job lost the default behavior profile: %+v", job)
	}
	if job.Weather != "RAIN" || job.Time.Hour != 17 || job.Vehicle.Model != "sultan" || job.Vehicle.Color == nil || job.Vehicle.Color.B != 255 {
		t.Fatalf("unexpected conditions: %+v", job)
	}
	state := store.State()
	if len(state.PendingCommands) != 1 || state.PendingCommands[0].Type != control.CommandStartStopSignBatch {
		t.Fatalf("expected one atomic pending command, got=%+v", state.PendingCommands)
	}
}

func TestStopSignBatchHandlerRejectsMissingAndStaleSafetyEpoch(t *testing.T) {
	store := control.NewStore()
	mux := http.NewServeMux()
	registerStopSignBatchHandlers(mux, store)

	payload := validStopSignBatchPayload(store.State().SafetyEpoch)
	missingPayload := bytes.Replace(payload, []byte(fmt.Sprintf("\"safetyEpoch\":%d,", store.State().SafetyEpoch)), nil, 1)
	missingRequest := httptest.NewRequest(http.MethodPost, "/control/stop-sign-batches", bytes.NewReader(missingPayload))
	missingResponse := httptest.NewRecorder()
	mux.ServeHTTP(missingResponse, missingRequest)
	if missingResponse.Code != http.StatusPreconditionRequired {
		t.Fatalf("missing epoch: status=%d body=%s", missingResponse.Code, missingResponse.Body.String())
	}

	staleEpoch := store.State().SafetyEpoch
	if _, err := store.Enqueue(control.CommandRequest{Type: control.CommandEndAllScenes}); err != nil {
		t.Fatalf("enqueue Hold: %v", err)
	}
	staleRequest := httptest.NewRequest(http.MethodPost, "/control/stop-sign-batches", bytes.NewReader(validStopSignBatchPayload(staleEpoch)))
	staleResponse := httptest.NewRecorder()
	mux.ServeHTTP(staleResponse, staleRequest)
	if staleResponse.Code != http.StatusConflict {
		t.Fatalf("stale epoch after Hold: status=%d body=%s", staleResponse.Code, staleResponse.Body.String())
	}
	if pending := store.State().PendingCommands; len(pending) != 1 || pending[0].Type != control.CommandEndAllScenes {
		t.Fatalf("late batch must not enqueue after Hold: %+v", pending)
	}
}

func TestStopSignBatchHandlerRejectsInvalidPlanWithoutQueueing(t *testing.T) {
	store := control.NewStore()
	mux := http.NewServeMux()
	registerStopSignBatchHandlers(mux, store)
	payload := []byte(fmt.Sprintf(`{
		"safetyEpoch":%d,
		"id":"bad-stops",
		"seed":"seed",
		"entries":[{"id":"sign","signPose":{"x":0,"y":0,"z":0,"heading":0},"targetSpeedMps":99}]
	}`, store.State().SafetyEpoch))
	req := httptest.NewRequest(http.MethodPost, "/control/stop-sign-batches", bytes.NewReader(payload))
	res := httptest.NewRecorder()

	mux.ServeHTTP(res, req)

	if res.Code != http.StatusBadRequest {
		t.Fatalf("unexpected status=%d body=%s", res.Code, res.Body.String())
	}
	if pending := store.State().PendingCommands; len(pending) != 0 {
		t.Fatalf("invalid plan must not queue partial work: %+v", pending)
	}
}

func TestStopSignBatchHandlerReportsUnknownJSONField(t *testing.T) {
	store := control.NewStore()
	mux := http.NewServeMux()
	registerStopSignBatchHandlers(mux, store)
	payload := bytes.Replace(
		validStopSignBatchPayload(store.State().SafetyEpoch),
		[]byte(`"id":"alta",`),
		[]byte(`"id":"alta","dwellMs":5000,`),
		1,
	)
	req := httptest.NewRequest(http.MethodPost, "/control/stop-sign-batches", bytes.NewReader(payload))
	res := httptest.NewRecorder()

	mux.ServeHTTP(res, req)

	if res.Code != http.StatusBadRequest || !bytes.Contains(res.Body.Bytes(), []byte(`unknown field \"dwellMs\"`)) {
		t.Fatalf("expected actionable unknown-field error, status=%d body=%s", res.Code, res.Body.String())
	}
}

func validStopSignBatchPayload(safetyEpoch uint64) []byte {
	return []byte(fmt.Sprintf(`{
		"safetyEpoch":%d,
		"id":"city-stops",
		"seed":"fresh-stop-data",
		"entries":[{
			"id":"alta",
			"signPose":{"x":100,"y":200,"z":8,"heading":90},
			"stopDistanceM":4,
			"startDistanceM":40,
			"targetSpeedMps":8,
			"stopConfirmationMs":250,
			"attemptCount":4,
			"weather":"EXTRASUNNY",
			"time":{"hour":12,"minute":0},
			"vehicle":{"model":"sultan","color":{"r":255,"g":255,"b":255}},
			"variations":[
				{"id":"base"},
				{"id":"rain-blue","egoCenterOffsetM":3,"startDistanceM":60,"targetSpeedMps":7.5,"stopConfirmationMs":500,"weather":"RAIN","time":{"hour":17,"minute":30},"vehicle":{"color":{"r":20,"g":40,"b":255}}}
			]
		}]
	}`, safetyEpoch))
}
