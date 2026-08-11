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

func TestParkingBatchHandlerQueuesOneSequentialBatchCommand(t *testing.T) {
	store := control.NewStore()
	mux := http.NewServeMux()
	registerParkingBatchHandlers(mux, store)
	payload := validParkingBatchPayload(store.State().SafetyEpoch)
	req := httptest.NewRequest(http.MethodPost, "/control/parking-batches", bytes.NewReader(payload))
	res := httptest.NewRecorder()

	mux.ServeHTTP(res, req)

	if res.Code != http.StatusOK {
		t.Fatalf("unexpected status=%d body=%s", res.Code, res.Body.String())
	}
	var body struct {
		JobCount        int             `json:"jobCount"`
		PlanFingerprint string          `json:"planFingerprint"`
		Command         control.Command `json:"command"`
	}
	if err := json.Unmarshal(res.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if body.JobCount != 2 || body.Command.Type != control.CommandStartParkingBatch || body.Command.ParkingBatchID != "lot-a" || len(body.Command.ParkingJobs) != 2 {
		t.Fatalf("unexpected response: %+v", body)
	}
	if body.PlanFingerprint == "" || body.Command.PlanFingerprint != body.PlanFingerprint {
		t.Fatalf("expected one authoritative plan fingerprint in response and command: %+v", body)
	}
	if body.Command.ParkingJobs[1].ID != "bay-1:right-20cm" || body.Command.ParkingJobs[1].Seed != "fresh-data:bay-1:right-20cm" {
		t.Fatalf("unexpected deterministic job: %+v", body.Command.ParkingJobs[1])
	}
	state := store.State()
	if len(state.PendingCommands) != 1 || state.PendingCommands[0].Type != control.CommandStartParkingBatch {
		t.Fatalf("expected one atomic pending command, got=%+v", state.PendingCommands)
	}
	if state.PendingCommands[0].PlanFingerprint != body.PlanFingerprint {
		t.Fatalf("pending command lost plan fingerprint: %+v", state.PendingCommands[0])
	}
}

func TestParkingBatchHandlerRejectsMissingAndStaleSafetyEpoch(t *testing.T) {
	store := control.NewStore()
	mux := http.NewServeMux()
	registerParkingBatchHandlers(mux, store)

	missingRequest := httptest.NewRequest(http.MethodPost, "/control/parking-batches", bytes.NewReader(validParkingBatchPayloadWithoutEpoch()))
	missingResponse := httptest.NewRecorder()
	mux.ServeHTTP(missingResponse, missingRequest)
	if missingResponse.Code != http.StatusPreconditionRequired {
		t.Fatalf("missing epoch: status=%d body=%s", missingResponse.Code, missingResponse.Body.String())
	}

	staleEpoch := store.State().SafetyEpoch
	if _, err := store.Enqueue(control.CommandRequest{Type: control.CommandEndAllScenes}); err != nil {
		t.Fatalf("enqueue Hold: %v", err)
	}
	staleRequest := httptest.NewRequest(http.MethodPost, "/control/parking-batches", bytes.NewReader(validParkingBatchPayload(staleEpoch)))
	staleResponse := httptest.NewRecorder()
	mux.ServeHTTP(staleResponse, staleRequest)
	if staleResponse.Code != http.StatusConflict {
		t.Fatalf("stale epoch after Hold: status=%d body=%s", staleResponse.Code, staleResponse.Body.String())
	}
	if pending := store.State().PendingCommands; len(pending) != 1 || pending[0].Type != control.CommandEndAllScenes {
		t.Fatalf("late batch must not enqueue after Hold: %+v", pending)
	}
}

func TestParkingBatchHandlerRejectsInvalidPlanWithoutQueueing(t *testing.T) {
	store := control.NewStore()
	mux := http.NewServeMux()
	registerParkingBatchHandlers(mux, store)
	payload := []byte(`{
		"id":"lot-a",
		"seed":"fresh-data",
		"entries":[{
			"id":"bay-1",
			"parkDest":{"x":0,"y":0,"z":0,"heading":0},
			"startDest":{"x":1,"y":-4,"z":0,"heading":0},
			"collectionAmount":5
		}]
	}`)
	req := httptest.NewRequest(http.MethodPost, "/control/parking-batches", bytes.NewReader(payload))
	res := httptest.NewRecorder()

	mux.ServeHTTP(res, req)

	if res.Code != http.StatusBadRequest {
		t.Fatalf("unexpected status=%d body=%s", res.Code, res.Body.String())
	}
	if pending := store.State().PendingCommands; len(pending) != 0 {
		t.Fatalf("invalid plan must not queue partial work: %+v", pending)
	}
}

func validParkingBatchPayload(safetyEpoch uint64) []byte {
	return []byte(fmt.Sprintf(`{
		"safetyEpoch":%d,
		"id":"lot-a",
		"seed":"fresh-data",
		"entries":[{
			"id":"bay-1",
			"parkDest":{"x":10,"y":20,"z":3,"heading":0},
			"startDest":{"x":10,"y":9,"z":3,"heading":0},
			"collectionAmount":5,
			"variations":[
				{"id":"base"},
				{"id":"right-20cm","parkDest":{"x":10.2,"y":20,"z":3,"heading":0},"startDest":{"x":10.2,"y":9,"z":3,"heading":0}}
			]
		}]
	}`, safetyEpoch))
}

func validParkingBatchPayloadWithoutEpoch() []byte {
	payload := validParkingBatchPayload(1)
	return bytes.Replace(payload, []byte("\t\t\"safetyEpoch\":1,\n"), nil, 1)
}
