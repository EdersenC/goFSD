package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

const (
	dataSourceRaw       = "raw"
	dataSourceProcessed = "processed"
)

var errInspectorNotFound = errors.New("inspector item not found")

type inspectorRunSummary struct {
	RunID        string                  `json:"runId"`
	SceneCount   int                     `json:"sceneCount"`
	TripCount    int                     `json:"tripCount"`
	ReportExists bool                    `json:"reportExists"`
	Scenes       []inspectorSceneSummary `json:"scenes"`
}

type inspectorSceneSummary struct {
	SceneKey      string                 `json:"sceneKey"`
	SceneID       string                 `json:"sceneId"`
	SceneVariant  string                 `json:"sceneVariant"`
	RunFileExists bool                   `json:"runFileExists"`
	TripCount     int                    `json:"tripCount"`
	Trips         []inspectorTripSummary `json:"trips"`
}

type inspectorTripSummary struct {
	TripName           string `json:"tripName"`
	TripIndex          int    `json:"tripIndex"`
	RawAvailable       bool   `json:"rawAvailable"`
	ProcessedAvailable bool   `json:"processedAvailable"`
}

type inspectorFieldInfo struct {
	Name string `json:"name"`
	Kind string `json:"kind"`
}

type inspectorSeriesResponse struct {
	RunID          string               `json:"runId"`
	SceneKey       string               `json:"sceneKey"`
	TripName       string               `json:"tripName"`
	Source         string               `json:"source"`
	TimelineFields []string             `json:"timelineFields"`
	Fields         []inspectorFieldInfo `json:"fields"`
	RowCount       int                  `json:"rowCount"`
	Rows           []map[string]any     `json:"rows"`
}

type inspectorTripSelection struct {
	runID        string
	sceneKey     string
	tripName     string
	tripIndex    int
	runDir       string
	sceneDir     string
	sceneRunFile string
	tripDir      string
}

func registerDataInspectorHandlers(mux *http.ServeMux, runsRoot string) {
	mux.HandleFunc("/data/runs", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		runs, err := discoverInspectorRuns(runsRoot)
		if err != nil {
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"runs": runs})
	})

	mux.HandleFunc("/data/trip/fields", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		selection, source, err := parseInspectorSelection(r, runsRoot)
		if err != nil {
			status := http.StatusBadRequest
			if errors.Is(err, errInspectorNotFound) {
				status = http.StatusNotFound
			}
			writeError(w, status, err.Error())
			return
		}
		fields, err := loadInspectorFields(selection, source)
		if err != nil {
			status := http.StatusInternalServerError
			if errors.Is(err, errInspectorNotFound) {
				status = http.StatusNotFound
			}
			writeError(w, status, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"runId":    selection.runID,
			"sceneKey": selection.sceneKey,
			"tripName": selection.tripName,
			"source":   source,
			"fields":   fields,
		})
	})

	mux.HandleFunc("/data/trip/series", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		selection, source, err := parseInspectorSelection(r, runsRoot)
		if err != nil {
			status := http.StatusBadRequest
			if errors.Is(err, errInspectorNotFound) {
				status = http.StatusNotFound
			}
			writeError(w, status, err.Error())
			return
		}
		fields := filterNonEmptyStrings(r.URL.Query()["field"])
		if len(fields) == 0 {
			writeError(w, http.StatusBadRequest, "at least one field is required")
			return
		}
		response, err := loadInspectorSeries(selection, source, fields)
		if err != nil {
			status := http.StatusInternalServerError
			if errors.Is(err, errInspectorNotFound) {
				status = http.StatusNotFound
			}
			writeError(w, status, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, response)
	})
}

func parseInspectorSelection(r *http.Request, runsRoot string) (inspectorTripSelection, string, error) {
	query := r.URL.Query()
	runID := strings.TrimSpace(query.Get("runId"))
	sceneKey := strings.TrimSpace(query.Get("sceneKey"))
	tripName := strings.TrimSpace(query.Get("tripName"))
	source := normalizeInspectorSource(query.Get("source"))
	if runID == "" || sceneKey == "" || tripName == "" {
		return inspectorTripSelection{}, "", errors.New("runId, sceneKey, and tripName are required")
	}
	if source == "" {
		return inspectorTripSelection{}, "", errors.New("source must be raw or processed")
	}
	selection, err := resolveInspectorTrip(runsRoot, runID, sceneKey, tripName)
	if err != nil {
		return inspectorTripSelection{}, "", err
	}
	return selection, source, nil
}

func normalizeInspectorSource(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case dataSourceRaw:
		return dataSourceRaw
	case dataSourceProcessed:
		return dataSourceProcessed
	default:
		return ""
	}
}

func discoverInspectorRuns(runsRoot string) ([]inspectorRunSummary, error) {
	entries, err := os.ReadDir(runsRoot)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return []inspectorRunSummary{}, nil
		}
		return nil, fmt.Errorf("read runs root: %w", err)
	}

	runs := make([]inspectorRunSummary, 0, len(entries))
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		runDir := filepath.Join(runsRoot, entry.Name())
		runSummary, ok, err := discoverInspectorRun(runDir)
		if err != nil {
			return nil, err
		}
		if ok {
			runs = append(runs, runSummary)
		}
	}
	sort.Slice(runs, func(i, j int) bool {
		return runs[i].RunID > runs[j].RunID
	})
	return runs, nil
}

func discoverInspectorRun(runDir string) (inspectorRunSummary, bool, error) {
	sceneEntries, err := os.ReadDir(runDir)
	if err != nil {
		return inspectorRunSummary{}, false, fmt.Errorf("read run dir %s: %w", runDir, err)
	}

	runID := filepath.Base(runDir)
	scenes := make([]inspectorSceneSummary, 0, len(sceneEntries))
	tripCount := 0
	for _, sceneEntry := range sceneEntries {
		if !sceneEntry.IsDir() {
			continue
		}
		sceneDir := filepath.Join(runDir, sceneEntry.Name())
		sceneSummary, ok, err := discoverInspectorScene(sceneDir)
		if err != nil {
			return inspectorRunSummary{}, false, err
		}
		if !ok {
			continue
		}
		tripCount += sceneSummary.TripCount
		scenes = append(scenes, sceneSummary)
	}
	if len(scenes) == 0 {
		return inspectorRunSummary{}, false, nil
	}
	sort.Slice(scenes, func(i, j int) bool {
		return scenes[i].SceneKey < scenes[j].SceneKey
	})
	return inspectorRunSummary{
		RunID:        runID,
		SceneCount:   len(scenes),
		TripCount:    tripCount,
		ReportExists: fileExists(filepath.Join(runDir, "dataset_report.json")),
		Scenes:       scenes,
	}, true, nil
}

func discoverInspectorScene(sceneDir string) (inspectorSceneSummary, bool, error) {
	sceneKey := filepath.Base(sceneDir)
	sceneID, sceneVariant := splitSceneKey(sceneKey)
	entries, err := os.ReadDir(sceneDir)
	if err != nil {
		return inspectorSceneSummary{}, false, fmt.Errorf("read scene dir %s: %w", sceneDir, err)
	}

	trips := make([]inspectorTripSummary, 0, len(entries))
	for _, entry := range entries {
		if !entry.IsDir() || !strings.HasPrefix(entry.Name(), "trip-") {
			continue
		}
		tripName := entry.Name()
		tripDir := filepath.Join(sceneDir, tripName)
		trips = append(trips, inspectorTripSummary{
			TripName:           tripName,
			TripIndex:          parseTripIndex(tripName),
			RawAvailable:       fileExists(filepath.Join(sceneDir, "run.jsonl")),
			ProcessedAvailable: fileExists(filepath.Join(tripDir, "dataset.jsonl")),
		})
	}
	if len(trips) == 0 {
		return inspectorSceneSummary{}, false, nil
	}
	sort.Slice(trips, func(i, j int) bool {
		return trips[i].TripIndex < trips[j].TripIndex
	})
	return inspectorSceneSummary{
		SceneKey:      sceneKey,
		SceneID:       sceneID,
		SceneVariant:  sceneVariant,
		RunFileExists: fileExists(filepath.Join(sceneDir, "run.jsonl")),
		TripCount:     len(trips),
		Trips:         trips,
	}, true, nil
}

func resolveInspectorTrip(runsRoot string, runID string, sceneKey string, tripName string) (inspectorTripSelection, error) {
	if filepath.Base(runID) != runID || filepath.Base(sceneKey) != sceneKey || filepath.Base(tripName) != tripName {
		return inspectorTripSelection{}, errors.New("invalid path selection")
	}
	runDir := filepath.Join(runsRoot, runID)
	sceneDir := filepath.Join(runDir, sceneKey)
	tripDir := filepath.Join(sceneDir, tripName)
	if !dirExists(runDir) || !dirExists(sceneDir) || !dirExists(tripDir) {
		return inspectorTripSelection{}, fmt.Errorf("%w: selected trip does not exist", errInspectorNotFound)
	}
	tripIndex := parseTripIndex(tripName)
	if tripIndex < 0 {
		return inspectorTripSelection{}, errors.New("invalid trip name")
	}
	return inspectorTripSelection{
		runID:        runID,
		sceneKey:     sceneKey,
		tripName:     tripName,
		tripIndex:    tripIndex,
		runDir:       runDir,
		sceneDir:     sceneDir,
		sceneRunFile: filepath.Join(sceneDir, "run.jsonl"),
		tripDir:      tripDir,
	}, nil
}

func loadInspectorFields(selection inspectorTripSelection, source string) ([]inspectorFieldInfo, error) {
	rows, err := loadInspectorRows(selection, source)
	if err != nil {
		return nil, err
	}
	fieldKinds := make(map[string]string)
	for _, row := range rows {
		for key, value := range row {
			kind := scalarKind(value)
			if kind == "" {
				continue
			}
			fieldKinds[key] = mergeScalarKinds(fieldKinds[key], kind)
		}
	}
	fields := make([]inspectorFieldInfo, 0, len(fieldKinds))
	for key, kind := range fieldKinds {
		fields = append(fields, inspectorFieldInfo{Name: key, Kind: kind})
	}
	sort.Slice(fields, func(i, j int) bool {
		return fields[i].Name < fields[j].Name
	})
	return fields, nil
}

func loadInspectorSeries(selection inspectorTripSelection, source string, selectedFields []string) (inspectorSeriesResponse, error) {
	rows, err := loadInspectorRows(selection, source)
	if err != nil {
		return inspectorSeriesResponse{}, err
	}
	fieldsMap := make(map[string]struct{}, len(selectedFields))
	for _, field := range selectedFields {
		fieldsMap[field] = struct{}{}
	}

	timelineFields := inspectorTimelineFields(source)
	responseRows := make([]map[string]any, 0, len(rows))
	fieldKinds := make(map[string]string)
	for _, row := range rows {
		item := make(map[string]any, len(timelineFields)+len(selectedFields))
		for _, timelineField := range timelineFields {
			if value, ok := row[timelineField]; ok {
				item[timelineField] = value
			}
		}
		for field := range fieldsMap {
			if value, ok := row[field]; ok {
				item[field] = value
				if kind := scalarKind(value); kind != "" {
					fieldKinds[field] = mergeScalarKinds(fieldKinds[field], kind)
				}
			}
		}
		responseRows = append(responseRows, item)
	}

	fields := make([]inspectorFieldInfo, 0, len(selectedFields))
	for _, field := range selectedFields {
		fields = append(fields, inspectorFieldInfo{
			Name: field,
			Kind: fieldKinds[field],
		})
	}

	return inspectorSeriesResponse{
		RunID:          selection.runID,
		SceneKey:       selection.sceneKey,
		TripName:       selection.tripName,
		Source:         source,
		TimelineFields: timelineFields,
		Fields:         fields,
		RowCount:       len(responseRows),
		Rows:           responseRows,
	}, nil
}

func loadInspectorRows(selection inspectorTripSelection, source string) ([]map[string]any, error) {
	switch source {
	case dataSourceRaw:
		return loadRawInspectorRows(selection)
	case dataSourceProcessed:
		return loadProcessedInspectorRows(selection)
	default:
		return nil, errors.New("unsupported source")
	}
}

func loadRawInspectorRows(selection inspectorTripSelection) ([]map[string]any, error) {
	if !fileExists(selection.sceneRunFile) {
		return nil, fmt.Errorf("%w: run.jsonl not found", errInspectorNotFound)
	}
	file, err := os.Open(selection.sceneRunFile)
	if err != nil {
		return nil, fmt.Errorf("open run.jsonl: %w", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 0, 1024*1024), 16*1024*1024)
	for scanner.Scan() {
		var tripRecord struct {
			TripIndex   int              `json:"tripIndex"`
			VehicleData []map[string]any `json:"vehicleData"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &tripRecord); err != nil {
			return nil, fmt.Errorf("parse run.jsonl row: %w", err)
		}
		if tripRecord.TripIndex != selection.tripIndex {
			continue
		}
		rows := make([]map[string]any, 0, len(tripRecord.VehicleData))
		for idx, vehicleRow := range tripRecord.VehicleData {
			flat := make(map[string]any)
			flattenScalarFields("", vehicleRow, flat)
			flat["rowIndex"] = idx
			rows = append(rows, flat)
		}
		return rows, nil
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read run.jsonl: %w", err)
	}
	return nil, fmt.Errorf("%w: trip %s not present in run.jsonl", errInspectorNotFound, selection.tripName)
}

func loadProcessedInspectorRows(selection inspectorTripSelection) ([]map[string]any, error) {
	datasetPath := filepath.Join(selection.tripDir, "dataset.jsonl")
	if !fileExists(datasetPath) {
		return nil, fmt.Errorf("%w: dataset.jsonl not found", errInspectorNotFound)
	}
	file, err := os.Open(datasetPath)
	if err != nil {
		return nil, fmt.Errorf("open dataset.jsonl: %w", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 0, 1024*1024), 16*1024*1024)
	rows := make([]map[string]any, 0)
	rowIndex := 0
	for scanner.Scan() {
		var sample map[string]any
		if err := json.Unmarshal(scanner.Bytes(), &sample); err != nil {
			return nil, fmt.Errorf("parse dataset.jsonl row: %w", err)
		}
		flat := make(map[string]any)
		flattenScalarFields("", sample, flat)
		flat["rowIndex"] = rowIndex
		rows = append(rows, flat)
		rowIndex++
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read dataset.jsonl: %w", err)
	}
	return rows, nil
}

func flattenScalarFields(prefix string, value any, out map[string]any) {
	switch typed := value.(type) {
	case map[string]any:
		keys := make([]string, 0, len(typed))
		for key := range typed {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, key := range keys {
			nextPrefix := key
			if prefix != "" {
				nextPrefix = prefix + "." + key
			}
			flattenScalarFields(nextPrefix, typed[key], out)
		}
	case []any:
		return
	case nil:
		return
	case bool, string:
		if prefix != "" {
			out[prefix] = typed
		}
	case float64, float32, int, int64, int32, int16, int8, uint, uint64, uint32, uint16, uint8:
		if prefix != "" {
			out[prefix] = typed
		}
	default:
		return
	}
}

func scalarKind(value any) string {
	switch value.(type) {
	case bool:
		return "boolean"
	case string:
		return "string"
	case float64, float32, int, int64, int32, int16, int8, uint, uint64, uint32, uint16, uint8:
		return "number"
	default:
		return ""
	}
}

func mergeScalarKinds(current string, next string) string {
	if current == "" || current == next {
		return next
	}
	if current == "string" || next == "string" {
		return "string"
	}
	if current == "number" || next == "number" {
		return "number"
	}
	return current
}

func inspectorTimelineFields(source string) []string {
	if source == dataSourceRaw {
		return []string{"rowIndex", "time", "chunkIndex", "dataPointIndex", "timeSinceSyncMs", "timeSinceChunkStartMs"}
	}
	return []string{"rowIndex", "anchor_game_time", "anchor_video_pts"}
}

func splitSceneKey(sceneKey string) (string, string) {
	index := strings.LastIndex(sceneKey, "_")
	if index <= 0 || index >= len(sceneKey)-1 {
		return sceneKey, ""
	}
	return sceneKey[:index], sceneKey[index+1:]
}

func parseTripIndex(tripName string) int {
	if !strings.HasPrefix(tripName, "trip-") {
		return -1
	}
	value, err := strconv.Atoi(strings.TrimPrefix(tripName, "trip-"))
	if err != nil {
		return -1
	}
	return value
}

func filterNonEmptyStrings(values []string) []string {
	filtered := make([]string, 0, len(values))
	seen := make(map[string]struct{}, len(values))
	for _, value := range values {
		trimmed := strings.TrimSpace(value)
		if trimmed == "" {
			continue
		}
		if _, ok := seen[trimmed]; ok {
			continue
		}
		seen[trimmed] = struct{}{}
		filtered = append(filtered, trimmed)
	}
	return filtered
}

func dirExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.IsDir()
}

func fileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}
