package main

import (
	"path/filepath"
	"sort"

	datasetproc "awesomeProject/internal/dataset"
)

const stopSignSceneFolder = "stop-sign_continuous-v2"

func filterStopSignTripDirs(tripDirs []string) []string {
	filtered := make([]string, 0, len(tripDirs))
	for _, tripDir := range tripDirs {
		if isStopSignTripDir(tripDir) {
			filtered = append(filtered, tripDir)
		}
	}
	return filtered
}

func isStopSignTripDir(tripDir string) bool {
	sceneFolder := filepath.Base(filepath.Dir(filepath.Clean(tripDir)))
	return sceneFolder == stopSignSceneFolder
}

func collectTripDirsForSelectedRuns(selectedTripDirs []string) ([]string, error) {
	selectedRuns := make(map[string]struct{}, len(selectedTripDirs))
	for _, tripDir := range selectedTripDirs {
		selectedRuns[tripRunDir(tripDir)] = struct{}{}
	}

	runDirs := make([]string, 0, len(selectedRuns))
	for runDir := range selectedRuns {
		runDirs = append(runDirs, runDir)
	}
	sort.Strings(runDirs)

	tripDirs := make([]string, 0, len(selectedTripDirs))
	for _, runDir := range runDirs {
		runTripDirs, err := datasetproc.CollectTripDirs(runDir, nil)
		if err != nil {
			return nil, err
		}
		tripDirs = append(tripDirs, runTripDirs...)
	}
	return tripDirs, nil
}

func tripRunDir(tripDir string) string {
	return filepath.Dir(filepath.Dir(filepath.Clean(tripDir)))
}
