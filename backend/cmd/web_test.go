package main

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestWebRoutesServeMUIWorkspace(t *testing.T) {
	tests := []struct {
		path        string
		contentType string
		bodyMarker  string
	}{
		{path: "/", contentType: "text/html; charset=utf-8", bodyMarker: "Stop Sign Lab"},
		{path: "/app.js", contentType: "text/javascript; charset=utf-8", bodyMarker: "Stop Sign Lab"},
		{path: "/guide", contentType: "text/html; charset=utf-8", bodyMarker: "Stop Sign Lab"},
		{path: "/architecture", contentType: "text/html; charset=utf-8", bodyMarker: "Stop Sign Lab"},
	}

	for _, test := range tests {
		t.Run(test.path, func(t *testing.T) {
			mux := http.NewServeMux()
			registerWebHandlers(mux)
			recorder := httptest.NewRecorder()
			mux.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, test.path, nil))

			if recorder.Code != http.StatusOK {
				t.Fatalf("unexpected status: got=%d want=%d", recorder.Code, http.StatusOK)
			}
			if got := recorder.Header().Get("Content-Type"); got != test.contentType {
				t.Fatalf("unexpected content type: got=%q want=%q", got, test.contentType)
			}
			if !strings.Contains(recorder.Body.String(), test.bodyMarker) {
				t.Fatalf("response is missing marker %q", test.bodyMarker)
			}
		})
	}
}

func TestWebRoutesRejectUnsupportedMethod(t *testing.T) {
	mux := http.NewServeMux()
	registerWebHandlers(mux)
	recorder := httptest.NewRecorder()
	mux.ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/architecture", nil))

	if recorder.Code != http.StatusMethodNotAllowed {
		t.Fatalf("unexpected status: got=%d want=%d", recorder.Code, http.StatusMethodNotAllowed)
	}
}

func TestRootWebRouteDoesNotSwallowUnknownPaths(t *testing.T) {
	mux := http.NewServeMux()
	registerWebHandlers(mux)
	recorder := httptest.NewRecorder()
	mux.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/missing", nil))

	if recorder.Code != http.StatusNotFound {
		t.Fatalf("unexpected status: got=%d want=%d", recorder.Code, http.StatusNotFound)
	}
}

func TestMUIWorkspaceExposesStopSignTemporalWorkbench(t *testing.T) {
	body, err := webAssets.ReadFile("web/app.js")
	if err != nil {
		t.Fatalf("read app asset: %v", err)
	}
	text := string(body)
	for _, marker := range []string{
		"Start setup car",
		"Stop Sign Lab",
		"Launch",
		"Approach",
		"Brake",
		"Dwell",
		"Release",
		"Collect",
		"Data",
		"Train",
		"Hold confirmed",
		"/control/stop-sign-batches",
		"/processing/reconcile",
		"/training/jobs",
		"/inference/start",
		"setStopSignTarget",
		"stop-sign-plan.v1",
		`Model \u2192 controller \u2192 game`,
		"scripted after the configured dwell",
	} {
		if !strings.Contains(text, marker) {
			t.Fatalf("MUI stop-sign workspace is missing %q", marker)
		}
	}
}

func TestWebIndexLoadsOnlyTheBundledApplication(t *testing.T) {
	body, err := webAssets.ReadFile("web/index.html")
	if err != nil {
		t.Fatalf("read index asset: %v", err)
	}
	text := string(body)
	if !strings.Contains(text, `<div id="root"></div>`) || !strings.Contains(text, `src="/app.js"`) {
		t.Fatal("web index must load the MUI application root and bundle")
	}
	for _, legacy := range []string{"parking-batch-json", "app.ts", "<style>"} {
		if strings.Contains(text, legacy) {
			t.Fatalf("web index still contains legacy UI marker %q", legacy)
		}
	}
}
