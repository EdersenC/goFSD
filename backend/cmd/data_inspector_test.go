package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestDiscoverInspectorRuns(t *testing.T) {
	root := t.TempDir()
	sceneDir := filepath.Join(root, "run-123", "scene-a_default")
	tripDir := filepath.Join(sceneDir, "trip-000")
	if err := os.MkdirAll(tripDir, 0o755); err != nil {
		t.Fatalf("mkdir trip dir: %v", err)
	}
	writeCommandJSONFile(t, filepath.Join(sceneDir, "run.jsonl"), map[string]any{
		"tripIndex": 0,
	})
	writeJSONLinesFile(t, filepath.Join(tripDir, "dataset.jsonl"), nil)

	runs, err := discoverInspectorRuns(root)
	if err != nil {
		t.Fatalf("discoverInspectorRuns: %v", err)
	}
	if len(runs) != 1 {
		t.Fatalf("expected 1 run, got %d", len(runs))
	}
	if runs[0].RunID != "run-123" {
		t.Fatalf("unexpected run id: %s", runs[0].RunID)
	}
	if len(runs[0].Scenes) != 1 || len(runs[0].Scenes[0].Trips) != 1 {
		t.Fatalf("unexpected run structure: %+v", runs[0])
	}
}

func TestInspectorFieldsAndSeries(t *testing.T) {
	root := t.TempDir()
	sceneDir := filepath.Join(root, "run-123", "scene-a_default")
	tripDir := filepath.Join(sceneDir, "trip-000")
	if err := os.MkdirAll(tripDir, 0o755); err != nil {
		t.Fatalf("mkdir trip dir: %v", err)
	}
	writeJSONLinesFile(t, filepath.Join(sceneDir, "run.jsonl"), []map[string]any{
		{
			"tripIndex": 0,
			"vehicleData": []map[string]any{
				{
					"time":              10,
					"chunkIndex":        0,
					"dataPointIndex":    0,
					"yaw":               11.5,
					"currentSpeed":      2.25,
					"eventWrongWay":     true,
					"coords":            []float64{1, 2, 3},
					"routeForwardDelta": -0.4,
				},
				{
					"time":              11,
					"chunkIndex":        0,
					"dataPointIndex":    1,
					"yaw":               12.5,
					"currentSpeed":      2.75,
					"eventWrongWay":     false,
					"routeForwardDelta": -0.2,
				},
			},
		},
	})
	writeJSONLinesFile(t, filepath.Join(tripDir, "dataset.jsonl"), []map[string]any{
		{
			"anchor_game_time": 1.1,
			"anchor_video_pts": 2.2,
			"label": map[string]any{
				"control": map[string]any{
					"Steering": 0.15,
				},
				"aux": map[string]any{
					"future_yaw_delta": 3.5,
					"move_intent":      true,
				},
			},
			"frame_paths": []string{"a.jpg"},
		},
	})

	selection, err := resolveInspectorTrip(root, "run-123", "scene-a_default", "trip-000")
	if err != nil {
		t.Fatalf("resolveInspectorTrip: %v", err)
	}

	rawFields, err := loadInspectorFields(selection, dataSourceRaw)
	if err != nil {
		t.Fatalf("loadInspectorFields raw: %v", err)
	}
	assertFieldKind(t, rawFields, "yaw", "number")
	assertFieldKind(t, rawFields, "eventWrongWay", "boolean")
	assertFieldMissing(t, rawFields, "coords")

	processedFields, err := loadInspectorFields(selection, dataSourceProcessed)
	if err != nil {
		t.Fatalf("loadInspectorFields processed: %v", err)
	}
	assertFieldKind(t, processedFields, "label.control.Steering", "number")
	assertFieldKind(t, processedFields, "label.aux.future_yaw_delta", "number")
	assertFieldKind(t, processedFields, "label.aux.move_intent", "boolean")
	assertFieldMissing(t, processedFields, "frame_paths")

	rawSeries, err := loadInspectorSeries(selection, dataSourceRaw, []string{"yaw", "currentSpeed"})
	if err != nil {
		t.Fatalf("loadInspectorSeries raw: %v", err)
	}
	if rawSeries.RowCount != 2 {
		t.Fatalf("unexpected raw row count: %d", rawSeries.RowCount)
	}
	if got := rawSeries.Rows[0]["yaw"]; got != 11.5 {
		t.Fatalf("unexpected raw yaw: %#v", got)
	}

	processedSeries, err := loadInspectorSeries(selection, dataSourceProcessed, []string{"label.aux.future_yaw_delta"})
	if err != nil {
		t.Fatalf("loadInspectorSeries processed: %v", err)
	}
	if processedSeries.RowCount != 1 {
		t.Fatalf("unexpected processed row count: %d", processedSeries.RowCount)
	}
	if got := processedSeries.Rows[0]["label.aux.future_yaw_delta"]; got != 3.5 {
		t.Fatalf("unexpected processed yaw delta: %#v", got)
	}
}

func TestDataInspectorHandlers(t *testing.T) {
	root := t.TempDir()
	sceneDir := filepath.Join(root, "run-123", "scene-a_default")
	tripDir := filepath.Join(sceneDir, "trip-000")
	if err := os.MkdirAll(tripDir, 0o755); err != nil {
		t.Fatalf("mkdir trip dir: %v", err)
	}
	writeJSONLinesFile(t, filepath.Join(sceneDir, "run.jsonl"), []map[string]any{
		{
			"tripIndex": 0,
			"vehicleData": []map[string]any{
				{"time": 10, "yaw": 1.25, "chunkIndex": 0, "dataPointIndex": 0},
			},
		},
	})
	writeJSONLinesFile(t, filepath.Join(tripDir, "dataset.jsonl"), []map[string]any{
		{
			"anchor_game_time": 1.0,
			"label": map[string]any{
				"control": map[string]any{},
				"aux": map[string]any{
					"future_yaw_delta": 2.5,
				},
			},
		},
	})

	mux := http.NewServeMux()
	registerDataInspectorHandlers(mux, root)

	req := httptest.NewRequest(http.MethodGet, "/data/runs", nil)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("unexpected status for runs: %d", rec.Code)
	}

	req = httptest.NewRequest(http.MethodGet, "/data/trip/fields?runId=run-123&sceneKey=scene-a_default&tripName=trip-000&source=raw", nil)
	rec = httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("unexpected status for fields: %d body=%s", rec.Code, rec.Body.String())
	}

	req = httptest.NewRequest(http.MethodGet, "/data/trip/series?runId=run-123&sceneKey=scene-a_default&tripName=trip-000&source=processed&field=label.aux.future_yaw_delta", nil)
	rec = httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("unexpected status for series: %d body=%s", rec.Code, rec.Body.String())
	}
}

func assertFieldKind(t *testing.T, fields []inspectorFieldInfo, name string, kind string) {
	t.Helper()
	for _, field := range fields {
		if field.Name == name {
			if field.Kind != kind {
				t.Fatalf("unexpected kind for %s: got=%s want=%s", name, field.Kind, kind)
			}
			return
		}
	}
	t.Fatalf("field %s not found", name)
}

func assertFieldMissing(t *testing.T, fields []inspectorFieldInfo, name string) {
	t.Helper()
	for _, field := range fields {
		if field.Name == name {
			t.Fatalf("field %s should not be present", name)
		}
	}
}

func writeJSONLinesFile(t *testing.T, path string, rows []map[string]any) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir path: %v", err)
	}
	file, err := os.Create(path)
	if err != nil {
		t.Fatalf("create file: %v", err)
	}
	defer func() {
		if err := file.Close(); err != nil {
			t.Fatalf("close file: %v", err)
		}
	}()
	enc := json.NewEncoder(file)
	for _, row := range rows {
		if err := enc.Encode(row); err != nil {
			t.Fatalf("encode row: %v", err)
		}
	}
}
