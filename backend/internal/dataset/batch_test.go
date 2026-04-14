package dataset

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestCollectTripDirs(t *testing.T) {
	tmp := t.TempDir()
	tripA := filepath.Join(tmp, "run-a", "scene-a", "trip-000")
	tripB := filepath.Join(tmp, "run-a", "scene-a", "trip-001")
	other := filepath.Join(tmp, "run-a", "scene-a", "misc")
	for _, dir := range []string{tripA, tripB, other} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatalf("mkdir %s: %v", dir, err)
		}
	}

	tripDirs, err := CollectTripDirs(tmp, nil)
	if err != nil {
		t.Fatalf("CollectTripDirs: %v", err)
	}
	if len(tripDirs) != 2 {
		t.Fatalf("unexpected trip dir count: got=%d want=2", len(tripDirs))
	}
}

func TestProcessTripDirsSkipsExistingOutputs(t *testing.T) {
	tmp := t.TempDir()
	tripDir := filepath.Join(tmp, "run-a", "scene-a", "trip-000")
	framesDir := filepath.Join(tripDir, "frames")
	if err := os.MkdirAll(framesDir, 0o755); err != nil {
		t.Fatalf("mkdir frames: %v", err)
	}
	if err := os.WriteFile(filepath.Join(framesDir, "000001.jpg"), []byte("x"), 0o644); err != nil {
		t.Fatalf("write frame: %v", err)
	}

	results := ProcessTripDirs(context.Background(), []string{tripDir}, 2)
	if len(results) != 1 {
		t.Fatalf("unexpected result count: got=%d want=1", len(results))
	}
	if results[0].State != "skipped" {
		t.Fatalf("unexpected state: %+v", results[0])
	}
	if results[0].Error != nil {
		t.Fatalf("unexpected error: %v", results[0].Error)
	}
}
