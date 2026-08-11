package main

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestFilterStopSignTripDirsAcceptsOnlyCurrentTemporalContract(t *testing.T) {
	root := t.TempDir()
	stopSignCurrent := filepath.Join(root, "run-a", stopSignSceneFolder, "trip-000")
	stopSignOld := filepath.Join(root, "run-a", "stop-sign_temporal-v0", "trip-001")
	legacyDriving := filepath.Join(root, "run-old", "inner-city-driving_default", "trip-000")
	legacyParking := filepath.Join(root, "run-old", "parking-forward-bay_straight-stop-v2", "trip-001")

	got := filterStopSignTripDirs([]string{legacyDriving, stopSignOld, legacyParking, stopSignCurrent})
	want := []string{stopSignCurrent}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("unexpected stop-sign scope: got=%v want=%v", got, want)
	}
}

func TestCollectTripDirsForSelectedRunsPreservesEverySceneInAffectedRun(t *testing.T) {
	root := t.TempDir()
	stopSign := filepath.Join(root, "run-mixed", stopSignSceneFolder, "trip-000")
	legacySameRun := filepath.Join(root, "run-mixed", "inner-city-driving_default", "trip-001")
	legacyOtherRun := filepath.Join(root, "run-legacy", "inner-city-driving_default", "trip-000")
	for _, tripDir := range []string{stopSign, legacySameRun, legacyOtherRun} {
		if err := os.MkdirAll(tripDir, 0o755); err != nil {
			t.Fatalf("mkdir trip: %v", err)
		}
	}

	got, err := collectTripDirsForSelectedRuns([]string{stopSign})
	if err != nil {
		t.Fatalf("collectTripDirsForSelectedRuns: %v", err)
	}
	want := []string{legacySameRun, stopSign}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("unexpected report scope: got=%v want=%v", got, want)
	}
}
