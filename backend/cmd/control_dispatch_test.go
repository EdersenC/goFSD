package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"awesomeProject/internal/control"
)

func TestControlDispatchConfirmationEndpointReturnsCanonicalCommand(t *testing.T) {
	store := control.NewStore()
	epoch := store.State().SafetyEpoch
	command, err := store.Enqueue(control.CommandRequest{
		Type:        control.CommandStartEgo,
		SafetyEpoch: &epoch,
	})
	if err != nil {
		t.Fatalf("enqueue start: %v", err)
	}
	if polled := store.Poll(""); polled == nil || polled.ID != command.ID {
		t.Fatalf("expected command to be polled, got=%+v", polled)
	}

	mux := http.NewServeMux()
	registerControlDispatchHandlers(mux, store)
	response := performControlDispatchConfirmation(t, mux, map[string]any{"commandId": command.ID})
	if response.Code != http.StatusOK {
		t.Fatalf("unexpected status: got=%d body=%s", response.Code, response.Body.String())
	}

	var confirmation control.DispatchConfirmation
	if err := json.Unmarshal(response.Body.Bytes(), &confirmation); err != nil {
		t.Fatalf("decode confirmation: %v", err)
	}
	if !confirmation.Confirmed || confirmation.Command == nil || confirmation.Command.ID != command.ID {
		t.Fatalf("endpoint did not return the canonical confirmed command: %+v", confirmation)
	}
}

func TestControlDispatchConfirmationEndpointReturnsCanceledDecision(t *testing.T) {
	store := control.NewStore()
	epoch := store.State().SafetyEpoch
	command, err := store.Enqueue(control.CommandRequest{
		Type:        control.CommandStartEgo,
		SafetyEpoch: &epoch,
	})
	if err != nil {
		t.Fatalf("enqueue start: %v", err)
	}
	if polled := store.Poll(""); polled == nil || polled.ID != command.ID {
		t.Fatalf("expected command to be polled, got=%+v", polled)
	}
	if _, err := store.Enqueue(control.CommandRequest{Type: control.CommandEndAllScenes}); err != nil {
		t.Fatalf("enqueue Hold: %v", err)
	}

	mux := http.NewServeMux()
	registerControlDispatchHandlers(mux, store)
	response := performControlDispatchConfirmation(t, mux, map[string]any{"commandId": command.ID})
	if response.Code != http.StatusOK {
		t.Fatalf("a clean rejection must remain a successful protocol response: got=%d body=%s", response.Code, response.Body.String())
	}

	var confirmation control.DispatchConfirmation
	if err := json.Unmarshal(response.Body.Bytes(), &confirmation); err != nil {
		t.Fatalf("decode confirmation: %v", err)
	}
	if confirmation.Confirmed || confirmation.Reason != control.DispatchConfirmationCanceled || confirmation.Command != nil {
		t.Fatalf("endpoint must return an explicit canceled decision, got=%+v", confirmation)
	}
}

func TestControlDispatchConfirmationEndpointRejectsInvalidRequests(t *testing.T) {
	mux := http.NewServeMux()
	registerControlDispatchHandlers(mux, control.NewStore())

	tests := []struct {
		name   string
		method string
		body   map[string]any
		want   int
	}{
		{name: "wrong method", method: http.MethodGet, body: map[string]any{}, want: http.StatusMethodNotAllowed},
		{name: "missing command id", method: http.MethodPost, body: map[string]any{}, want: http.StatusBadRequest},
		{name: "unknown field", method: http.MethodPost, body: map[string]any{"commandId": "cmd-1", "extra": true}, want: http.StatusBadRequest},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			payload, err := json.Marshal(test.body)
			if err != nil {
				t.Fatalf("marshal body: %v", err)
			}
			response := httptest.NewRecorder()
			request := httptest.NewRequest(test.method, "/control/dispatch/confirm", bytes.NewReader(payload))
			mux.ServeHTTP(response, request)
			if response.Code != test.want {
				t.Fatalf("unexpected status: got=%d want=%d body=%s", response.Code, test.want, response.Body.String())
			}
		})
	}
}

func performControlDispatchConfirmation(t *testing.T, mux *http.ServeMux, body map[string]any) *httptest.ResponseRecorder {
	t.Helper()
	payload, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal confirmation request: %v", err)
	}
	response := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/control/dispatch/confirm", bytes.NewReader(payload))
	mux.ServeHTTP(response, request)
	return response
}
