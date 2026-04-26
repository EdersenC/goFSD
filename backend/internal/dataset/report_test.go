package dataset

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestBuildRunDatasetReportAggregatesTripsAndFlags(t *testing.T) {
	tmp := t.TempDir()
	runDir := filepath.Join(tmp, "run-001")

	tripA := filepath.Join(runDir, "scene-a_default", "trip-000")
	writeDatasetReportTripFixture(t, tripA, tripFixture{
		runID:        "run-001",
		sceneID:      "scene-a",
		sceneVariant: "default",
		tripSeed:     "seed-a",
		weatherType:  "CLEAR",
		timeOfDay:    "midday",
		vehicleModel: "sultan",
		vehicleColor: "Black",
		status: ProcessingStatus{
			State:      "completed",
			FrameCount: 5,
		},
		samples: []DatasetSample{
			reportSample(0.1, 20.0, 0.2, -2.0, -1.0, 3.0, 3.5, reportTelemetry{
				currentSpeed:        5.0,
				routeDistance:       12.0,
				leadVehicleDistance: 18.0,
				hasLeadVehicle:      true,
				isStopped:           0,
				extraRaw:            map[string]any{"isInJunction": true, "eventOffroad": false},
			}),
			reportSample(0.1, 15.0, 0.2, 0.0, 0.0, 5.0, 5.0, reportTelemetry{
				currentSpeed:        5.0,
				routeDistance:       6.0,
				leadVehicleDistance: 10.0,
				hasLeadVehicle:      true,
				isStopped:           1,
				extraRaw:            map[string]any{"isInJunction": false, "eventOffroad": true},
			}),
			reportSample(0.1, 5.0, 0.2, 2.0, 1.0, 7.0, 7.5, reportTelemetry{
				currentSpeed:   5.0,
				routeDistance:  2.0,
				hasLeadVehicle: false,
				isStopped:      true,
				extraRaw:       map[string]any{"isInJunction": false, "eventOffroad": false},
			}),
		},
	})

	tripB := filepath.Join(runDir, "scene-a_default", "trip-001")
	writeDatasetReportTripFixture(t, tripB, tripFixture{
		runID:        "run-001",
		sceneID:      "scene-a",
		sceneVariant: "default",
		tripSeed:     "seed-b",
		weatherType:  "CLEAR",
		timeOfDay:    "midday",
		vehicleModel: "sultan",
		vehicleColor: "Black",
		status: ProcessingStatus{
			State:             "failed",
			Error:             "ffmpeg failed",
			Warning:           "no dataset samples generated",
			ZeroSampleReasons: map[string]int{"missing_yaw_rate_target": 4},
		},
	})

	tripC := filepath.Join(runDir, "scene-b_default", "trip-000")
	writeDatasetReportTripFixture(t, tripC, tripFixture{
		runID:        "run-001",
		sceneID:      "scene-b",
		sceneVariant: "default",
		tripSeed:     "seed-c",
		weatherType:  "RAIN",
		timeOfDay:    "night",
		vehicleModel: "sultan",
		vehicleColor: "Red",
		status: ProcessingStatus{
			State:      "skipped",
			FrameCount: 2,
		},
		samples: []DatasetSample{
			reportSample(0.5, 40.0, 0.2, 1.0, 0.5, 11.0, 11.0, reportTelemetry{
				currentSpeed: 10.0,
				isStopped:    0,
			}),
		},
	})

	report, err := BuildRunDatasetReport(runDir, []string{tripA, tripB, tripC}, DatasetReportConfig{DeltaSpeedClip: 2.0})
	if err != nil {
		t.Fatalf("BuildRunDatasetReport: %v", err)
	}

	if report.RunID != "run-001" {
		t.Fatalf("unexpected run id: got=%q want=%q", report.RunID, "run-001")
	}
	if len(report.Trips) != 3 {
		t.Fatalf("unexpected trip count: got=%d want=3", len(report.Trips))
	}
	if len(report.Scenes) != 2 {
		t.Fatalf("unexpected scene count: got=%d want=2", len(report.Scenes))
	}

	summary := report.Summary
	if summary.TripCount != 3 || summary.CompletedTrips != 1 || summary.SkippedTrips != 1 || summary.FailedTrips != 1 {
		t.Fatalf("unexpected trip summary: %+v", summary)
	}
	if summary.MissingDatasetTrips != 1 || summary.ZeroSampleTrips != 1 {
		t.Fatalf("unexpected missing/zero summary: %+v", summary)
	}
	if summary.FrameCount != 7 || summary.SampleCount != 4 {
		t.Fatalf("unexpected frame/sample counts: %+v", summary)
	}
	if summary.StoppedSampleCount != 2 || summary.MovingSampleCount != 2 {
		t.Fatalf("unexpected stopped/moving counts: %+v", summary)
	}
	if math.Abs(summary.StoppedSampleShare-0.5) > 1e-9 {
		t.Fatalf("unexpected stopped sample share: got=%v want=0.5", summary.StoppedSampleShare)
	}
	if summary.DeltaSpeedClip.AnyClipCount != 2 || summary.DeltaSpeedClip.NegativeClipCount != 1 || summary.DeltaSpeedClip.PositiveClipCount != 1 {
		t.Fatalf("unexpected clip summary: %+v", summary.DeltaSpeedClip)
	}
	if math.Abs(summary.DeltaSpeedClip.AnyClipRate-0.5) > 1e-9 {
		t.Fatalf("unexpected clip rate: got=%v want=0.5", summary.DeltaSpeedClip.AnyClipRate)
	}
	if summary.FlatLabelTripCounts["steer"] != 1 {
		t.Fatalf("unexpected flat steer trip count: %+v", summary.FlatLabelTripCounts)
	}
	if summary.BooleanStats["lead_vehicle_present"].TrueCount != 2 || summary.BooleanStats["lead_vehicle_present"].Count != 3 {
		t.Fatalf("unexpected lead vehicle stats: %+v", summary.BooleanStats["lead_vehicle_present"])
	}
	if summary.BooleanStats["event_offroad"].TrueCount != 1 || summary.BooleanStats["in_junction"].TrueCount != 1 {
		t.Fatalf("unexpected event/junction stats: offroad=%+v junction=%+v", summary.BooleanStats["event_offroad"], summary.BooleanStats["in_junction"])
	}
	if routeStats := summary.LabelStats["route_distance"]; math.Abs(routeStats.Mean-6.666666666666667) > 1e-9 {
		t.Fatalf("unexpected route distance stats: %+v", routeStats)
	}
	if summary.Diversity.WeatherTypes.Count != 3 || summary.Diversity.WeatherTypes.UniqueCount != 2 {
		t.Fatalf("unexpected weather diversity: %+v", summary.Diversity.WeatherTypes)
	}
	if !summary.Diversity.VehicleModel.LowDiversity {
		t.Fatalf("expected vehicle model low-diversity warning: %+v", summary.Diversity.VehicleModel)
	}
	if summary.Diversity.TripSeed.UniqueCount != 3 {
		t.Fatalf("unexpected trip seed diversity: %+v", summary.Diversity.TripSeed)
	}

	var foundMissing bool
	var foundFlatSteer bool
	for _, trip := range report.Trips {
		if trip.TripName == "trip-001" {
			foundMissing = trip.MissingDataset && trip.ZeroSamples && trip.ZeroSampleReasons["missing_yaw_rate_target"] == 4
		}
		if trip.TripName == "trip-000" && trip.SceneID == "scene-a" {
			for _, label := range trip.FlatLabels {
				if label == "steer" {
					foundFlatSteer = true
					break
				}
			}
		}
	}
	if !foundMissing {
		t.Fatalf("expected missing dataset trip to be flagged")
	}
	if !foundFlatSteer {
		t.Fatalf("expected flat steer label to be flagged")
	}
}

func TestWriteRunDatasetReportsWritesFile(t *testing.T) {
	tmp := t.TempDir()
	runDir := filepath.Join(tmp, "run-xyz")
	tripDir := filepath.Join(runDir, "scene-a_default", "trip-000")
	writeDatasetReportTripFixture(t, tripDir, tripFixture{
		runID:        "run-xyz",
		sceneID:      "scene-a",
		sceneVariant: "default",
		tripSeed:     "seed-z",
		weatherType:  "CLEAR",
		timeOfDay:    "midday",
		vehicleModel: "sultan",
		vehicleColor: "Black",
		status: ProcessingStatus{
			State:      "completed",
			FrameCount: 1,
		},
		samples: []DatasetSample{
			reportSample(0.2, 25.0, 0.2, 0.0, 0.0, 4.0, 4.0, reportTelemetry{
				currentSpeed: 4.0,
				isStopped:    0,
			}),
		},
	})

	reports, err := WriteRunDatasetReports([]string{tripDir}, DatasetReportConfig{DeltaSpeedClip: 2.0})
	if err != nil {
		t.Fatalf("WriteRunDatasetReports: %v", err)
	}
	if len(reports) != 1 {
		t.Fatalf("unexpected report count: got=%d want=1", len(reports))
	}
	if _, err := os.Stat(filepath.Join(runDir, datasetReportFileName)); err != nil {
		t.Fatalf("dataset report not written: %v", err)
	}
}

type tripFixture struct {
	runID        string
	sceneID      string
	sceneVariant string
	tripSeed     string
	weatherType  string
	timeOfDay    string
	vehicleModel string
	vehicleColor string
	status       ProcessingStatus
	samples      []DatasetSample
}

func writeDatasetReportTripFixture(t *testing.T, tripDir string, fixture tripFixture) {
	t.Helper()

	if err := os.MkdirAll(tripDir, 0o755); err != nil {
		t.Fatalf("mkdir trip dir: %v", err)
	}

	metadata := tripMetadata{
		RunID:        fixture.runID,
		SceneID:      fixture.sceneID,
		SceneVariant: fixture.sceneVariant,
		TripIndex:    0,
		TripSeed:     fixture.tripSeed,
		WeatherType:  fixture.weatherType,
		TimeOfDay:    fixture.timeOfDay,
		VehicleModel: fixture.vehicleModel,
		VehicleColor: fixture.vehicleColor,
	}
	writeJSONFile(t, filepath.Join(tripDir, "metadata.json"), metadata)
	writeJSONFile(t, filepath.Join(tripDir, "processing.json"), fixture.status)

	if len(fixture.samples) > 0 {
		file, err := os.Create(filepath.Join(tripDir, "dataset.jsonl"))
		if err != nil {
			t.Fatalf("create dataset.jsonl: %v", err)
		}
		enc := json.NewEncoder(file)
		for _, sample := range fixture.samples {
			if err := enc.Encode(sample); err != nil {
				_ = file.Close()
				t.Fatalf("encode sample: %v", err)
			}
		}
		if err := file.Close(); err != nil {
			t.Fatalf("close dataset.jsonl: %v", err)
		}
	}
}

func writeJSONFile(t *testing.T, path string, value any) {
	t.Helper()
	body, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("marshal %s: %v", path, err)
	}
	if err := os.WriteFile(path, append(body, '\n'), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

type reportTelemetry struct {
	currentSpeed        any
	routeDistance       any
	leadVehicleDistance any
	hasLeadVehicle      any
	isStopped           any
	extraRaw            map[string]any
}

func reportSample(
	steering float64,
	futureYawDelta float64,
	futureHorizonSeconds float64,
	deltaSpeed float64,
	deltaSpeedTarget float64,
	futureSpeed float64,
	futureSpeedTarget float64,
	telemetry reportTelemetry,
) DatasetSample {
	raw := cloneMap(telemetry.extraRaw)
	return DatasetSample{
		Label: GroupedLabel{
			Control: GroupedLabelControl{
				Steering: steering,
			},
			Aux: GroupedLabelAux{
				DeltaSpeed:           deltaSpeed,
				DeltaSpeedTarget:     deltaSpeedTarget,
				FutureSpeed:          futureSpeed,
				FutureSpeedTarget:    futureSpeedTarget,
				FutureYawDelta:       futureYawDelta,
				FutureHorizonSeconds: futureHorizonSeconds,
			},
		},
		TelemetryHistory: []GroupedTelemetryItem{
			{
				Aux: GroupedTelemetryAux{
					CurrentSpeed:        telemetry.currentSpeed,
					RouteDistance:       telemetry.routeDistance,
					LeadVehicleDistance: telemetry.leadVehicleDistance,
					HasLeadVehicle:      telemetry.hasLeadVehicle,
					IsStopped:           telemetry.isStopped,
				},
				Raw: raw,
			},
		},
	}
}
