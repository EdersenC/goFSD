package dataset

import (
	"encoding/json"
	"math"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestBuildStopSignSamplesPreservesPhasesControlsAndFutureTargets(t *testing.T) {
	phases := []string{
		"accelerate",
		"accelerate",
		"cruise_approach",
		"decelerate",
		"decelerate",
		"stop_hold",
		"stop_hold",
		"stop_hold",
		"release",
		"release",
	}
	desiredSpeeds := []float64{0.5, 1.5, 2.5, 2.4, 2.0, 0, 0, 0, 0.1, 0.2}
	frames := make([]VideoFrame, 0, len(phases))
	labels := make([]timedLabel, 0, len(phases))
	for index, phase := range phases {
		seconds := float64(index) * 0.05
		frames = append(frames, VideoFrame{
			Index:     index,
			PTS:       seconds,
			ImagePath: "frames/frame.jpg",
		})
		labels = append(labels, timedLabel{
			RelativeSeconds: seconds,
			Label: stopSignTelemetryFixture(
				phase,
				float64(index),
				desiredSpeeds[index],
			),
		})
	}

	samples, stats := buildDatasetSamplesWithImageOffsetsAndStats(
		frames,
		labels,
		0,
		[]int{0},
		1,
		25*time.Millisecond,
		[]int{0},
		[]int{1, 2},
		50*time.Millisecond,
	)

	if len(samples) != 8 || stats.GeneratedSampleCount != 8 {
		t.Fatalf("unexpected stop-sign sample count: samples=%d stats=%+v", len(samples), stats)
	}
	phaseCounts := make(map[string]int)
	for _, sample := range samples {
		phaseCounts[sample.Phase]++
	}
	if phaseCounts["stop_hold"] != 3 {
		t.Fatalf("stop-hold frames were thinned instead of preserving the temporal clip: %+v", phaseCounts)
	}

	first := samples[0]
	if first.Task != stopSignTask || first.Phase != "accelerate" {
		t.Fatalf("unexpected sample identity: %+v", first)
	}
	if first.Label.Control.ExpertThrottle != 0.7 || first.Label.Control.ExpertBrake != 0.2 {
		t.Fatalf("expert throttle/brake labels were not retained: %+v", first.Label.Control)
	}
	if first.Label.Control.ExpertGoProbability != 0.8 || first.Label.Control.ExpertStopProbability != 0.2 {
		t.Fatalf("stop/go targets were not retained: %+v", first.Label.Control)
	}
	if !reflect.DeepEqual(first.Label.Aux.FutureSpeedTargetsMps, []float64{1.5, 2.5}) {
		t.Fatalf("unexpected future desired-speed profile: %+v", first.Label.Aux.FutureSpeedTargetsMps)
	}
	if !reflect.DeepEqual(first.Label.Aux.FutureObservedSpeedMps, []float64{1, 2}) {
		t.Fatalf("unexpected future observed-speed diagnostics: %+v", first.Label.Aux.FutureObservedSpeedMps)
	}
	if !reflect.DeepEqual(first.Label.Aux.FutureStopSignPhases, []string{"accelerate", "cruise_approach"}) {
		t.Fatalf("unexpected future phase profile: %+v", first.Label.Aux.FutureStopSignPhases)
	}
	boundary := samples[2]
	if boundary.Phase != "cruise_approach" || boundary.ClipStage != "approach" ||
		!reflect.DeepEqual(boundary.Label.Aux.FutureStopSignPhases, []string{"decelerate", "decelerate"}) {
		t.Fatalf("logical approach window must preserve future context across the braking boundary: %+v", boundary)
	}
	current := first.TelemetryHistory[len(first.TelemetryHistory)-1]
	if current.Control.Acceleration != 0.4 || current.Control.BrakePressureAvg != 0.15 {
		t.Fatalf("raw actuation diagnostics were not retained: %+v", current.Control)
	}
	if current.Raw["eventCollision"] != false {
		t.Fatalf("raw stop-sign telemetry was not retained: %+v", current.Raw)
	}
}

func TestBuildStopSignSamplesUsesCausalAnchorAndFutureFacingTargets(t *testing.T) {
	labels := []timedLabel{
		{RelativeSeconds: 0, Label: stopSignTelemetryFixture("cruise_approach", 5, 5)},
		{RelativeSeconds: 0.05, Label: stopSignTelemetryFixture("decelerate", 5, 5)},
		{RelativeSeconds: 0.10, Label: stopSignTelemetryFixture("decelerate", 4.8, 4.8)},
	}
	frames := []VideoFrame{{Index: 0, PTS: 0.099, ImagePath: "frames/000001.jpg"}}
	samples, stats := buildStopSignSamplesWithImageOffsetsAndStats(
		frames,
		labels,
		0,
		[]int{0},
		1,
		100*time.Millisecond,
		[]int{0},
		[]int{1},
		50*time.Millisecond,
	)
	if len(samples) != 1 || stats.GeneratedSampleCount != 1 {
		t.Fatalf("expected one causal stop-sign sample: samples=%d stats=%+v", len(samples), stats)
	}
	sample := samples[0]
	if sample.Phase != "cruise_approach" || sample.ClipStage != "approach" {
		t.Fatalf("anchor must use the latest label at or before its frame: %+v", sample)
	}
	if sample.Label.Control.ExpertStopProbability != 0.0 {
		t.Fatalf("pre-reduction stop intent must be cleared with the premature phase: %+v", sample.Label.Control)
	}
	if !reflect.DeepEqual(sample.Label.Aux.FutureStopSignPhases, []string{"decelerate"}) {
		t.Fatalf("future targets must remain future-facing: %+v", sample.Label.Aux.FutureStopSignPhases)
	}
}

func TestTrimStopSignRollingCaptureStartsAfterStableCruisePreRoll(t *testing.T) {
	labels := make([]timedLabel, 0, 61)
	for index := 0; index <= 60; index++ {
		seconds := float64(index) * 0.05
		phase := "accelerate"
		desiredSpeed := math.Min(5, seconds*10)
		if seconds >= 0.5 {
			phase = "cruise_approach"
			desiredSpeed = 5
		}
		if seconds >= 2.5 {
			phase = "decelerate"
			desiredSpeed = 4.8
		}
		labels = append(labels, timedLabel{RelativeSeconds: seconds, Label: stopSignTelemetryFixture(phase, desiredSpeed, desiredSpeed)})
	}
	samples := []DatasetSample{
		{AnchorGameTime: 0.5}, {AnchorGameTime: 1.0}, {AnchorGameTime: 1.49},
		{AnchorGameTime: 1.5}, {AnchorGameTime: 2.0}, {AnchorGameTime: 2.5},
	}

	trimmed, warmupCount := trimStopSignRollingCaptureSamples(samples, labels)
	if warmupCount != 3 || len(trimmed) != 3 || trimmed[0].AnchorGameTime != 1.5 {
		t.Fatalf("rolling capture did not retain exactly the stable cruise pre-roll boundary: warmup=%d samples=%+v", warmupCount, trimmed)
	}

	unstable := []timedLabel{
		{RelativeSeconds: 0, Label: stopSignTelemetryFixture("accelerate", 0, 0)},
		{RelativeSeconds: 0.5, Label: stopSignTelemetryFixture("decelerate", 2, 2)},
	}
	trimmed, warmupCount = trimStopSignRollingCaptureSamples(samples, unstable)
	if len(trimmed) != 0 || warmupCount != len(samples) {
		t.Fatalf("an approach without stable cruise must not leak launch-heavy rows: warmup=%d samples=%+v", warmupCount, trimmed)
	}
}

func TestLatestLabelAtOrBeforeRejectsFutureAndStaleLabels(t *testing.T) {
	labels := []timedLabel{
		{RelativeSeconds: 1.00, Label: map[string]any{"phase": "past"}},
		{RelativeSeconds: 1.05, Label: map[string]any{"phase": "future"}},
	}
	tolerance := (100 * time.Millisecond).Seconds()
	label, index, ok := latestLabelAtOrBeforeWithIndex(labels, 1.049, tolerance)
	if !ok || index != 0 || label.Label["phase"] != "past" {
		t.Fatalf("expected the past label, got label=%+v index=%d ok=%t", label, index, ok)
	}
	if _, _, ok := latestLabelAtOrBeforeWithIndex(labels, 1.20, tolerance); ok {
		t.Fatal("stale past labels must be rejected")
	}
}

func TestRunDatasetReportProvidesStopSignPhaseAndLocationCoverage(t *testing.T) {
	tmp := t.TempDir()
	runDir := filepath.Join(tmp, "run-stop-sign")
	goalA := stopSignGoalFixture("base", 100, -20)
	goalB := stopSignGoalFixture("rain", 200, -40)
	tripA := filepath.Join(runDir, "stop-sign_continuous-v2", "trip-000")
	tripB := filepath.Join(runDir, "stop-sign_continuous-v2", "trip-001")

	writeStopSignReportTripFixture(t, tripA, 0, goalA, map[string]any{
		"success": true, "status": "succeeded", "stageTransitions": stopSignTransitionsFixture(),
	}, []string{"accelerate", "decelerate", "stop_hold", "release"})
	writeStopSignReportTripFixture(t, tripB, 1, goalB, map[string]any{
		"success": false,
		"status":  "collision",
	}, []string{"accelerate", "decelerate"})

	report, err := BuildRunDatasetReport(runDir, []string{tripA, tripB}, DatasetReportConfig{})
	if err != nil {
		t.Fatalf("BuildRunDatasetReport: %v", err)
	}
	coverage := report.Summary.StopSignCoverage
	if coverage.ClipCount != 2 || coverage.TrainingEligibleClipCount != 1 || coverage.ExcludedClipCount != 1 {
		t.Fatalf("unexpected clip eligibility coverage: %+v", coverage)
	}
	if coverage.SampleCount != 6 || coverage.TrainingEligibleSampleCount != 4 || coverage.ExcludedSampleCount != 2 {
		t.Fatalf("unexpected sample eligibility coverage: %+v", coverage)
	}
	if coverage.PhaseCounts["accelerate"] != 2 || coverage.PhaseCounts["stop_hold"] != 1 || coverage.PhaseCounts["release"] != 1 {
		t.Fatalf("unexpected phase coverage: %+v", coverage.PhaseCounts)
	}
	if coverage.TrainingEligiblePhaseCounts["accelerate"] != 1 || coverage.TrainingEligiblePhaseCounts["stop_hold"] != 1 {
		t.Fatalf("eligible-only phase coverage is not usable for balancing: %+v", coverage.TrainingEligiblePhaseCounts)
	}
	if len(coverage.LocationCounts) != 2 {
		t.Fatalf("location-level splits were not visible in the report: %+v", coverage.LocationCounts)
	}
	if len(coverage.TrainingEligibleLocationCounts) != 1 || len(coverage.TrainingEligibleLocationPhaseCounts) != 1 {
		t.Fatalf("eligible-only location coverage is not usable for scenario splits: %+v", coverage)
	}
	for location, phaseCounts := range coverage.LocationPhaseCounts {
		if coverage.LocationCounts[location] == 0 || len(phaseCounts) == 0 {
			t.Fatalf("invalid phase-by-location coverage: location=%s phases=%+v", location, phaseCounts)
		}
	}
	if report.Trips[1].TrainingEligible == nil || *report.Trips[1].TrainingEligible {
		t.Fatalf("failed trip should remain reported as excluded: %+v", report.Trips[1])
	}
}

func TestDecorateStopSignSamplesCarriesGoalOutcomeAndTrainingEligibility(t *testing.T) {
	metadata := tripMetadata{
		StopSignGoal:    stopSignGoalFixture("rain-red", 100, -20),
		StopSignOutcome: map[string]any{"success": true, "status": "succeeded", "stageTransitions": stopSignTransitionsFixture()},
	}
	samples := decorateStopSignSamples([]DatasetSample{{Task: stopSignTask, Phase: "stop_hold", ClipStage: "brake_stop"}}, metadata)
	if len(samples) != 1 || samples[0].TrainingEligible == nil || !*samples[0].TrainingEligible {
		t.Fatalf("successful clip should be eligible: %+v", samples)
	}
	if samples[0].ScenarioLocationID == "" || samples[0].ScenarioSplitGroup != samples[0].ScenarioLocationID {
		t.Fatalf("expected a stable location-level split group: %+v", samples[0])
	}
	if samples[0].VariationID != "rain-red" || samples[0].ClipStage != "brake_stop" || samples[0].StopSignGoal["contract"] != "stop-sign-goal.v3" {
		t.Fatalf("goal context was not carried into the dataset row: %+v", samples[0])
	}
	variationProfile, ok := samples[0].StopSignGoal["variationProfile"].(map[string]any)
	if !ok || variationProfile["contract"] != "stop-sign-variation-profile.v1" || variationProfile["combinationMagnitudePct"] != 37.5 {
		t.Fatalf("variation combination profile was not preserved in the dataset row: %+v", samples[0].StopSignGoal)
	}
	if samples[0].StopSignOutcome["status"] != "succeeded" {
		t.Fatalf("outcome context was not carried into the dataset row: %+v", samples[0])
	}

	metadata.StopSignOutcome = map[string]any{"success": false, "status": "collision"}
	excluded := decorateStopSignSamples([]DatasetSample{{Task: stopSignTask, Phase: "decelerate", ClipStage: "brake_stop"}}, metadata)
	if excluded[0].TrainingEligible == nil || *excluded[0].TrainingEligible {
		t.Fatalf("failed clip must remain inspectable but be excluded from expert training: %+v", excluded[0])
	}
	if excluded[0].TrainingExclusionReason != "stop_sign_outcome_not_succeeded" {
		t.Fatalf("unexpected exclusion reason: %+v", excluded[0])
	}
	legacyMetadata := metadata
	legacyMetadata.StopSignGoal = cloneMap(metadata.StopSignGoal)
	legacyMetadata.StopSignGoal["contract"] = "stop-sign-goal.v2"
	legacy := decorateStopSignSamples([]DatasetSample{{Task: stopSignTask, Phase: "decelerate", ClipStage: "brake_stop"}}, legacyMetadata)
	if legacy[0].TrainingEligible == nil || *legacy[0].TrainingEligible || legacy[0].TrainingExclusionReason != "invalid_stop_sign_goal" {
		t.Fatalf("legacy split clips must not enter continuous training: %+v", legacy[0])
	}
	body, err := json.Marshal(excluded[0])
	if err != nil {
		t.Fatalf("marshal excluded sample: %v", err)
	}
	serialized := string(body)
	for _, field := range []string{
		`"training_eligible":false`,
		`"clip_stage":"brake_stop"`,
		`"stopSignGoal"`,
		`"stopSignOutcome"`,
		`"scenario_split_group"`,
	} {
		if !strings.Contains(serialized, field) {
			t.Fatalf("serialized dataset row is missing %s: %s", field, serialized)
		}
	}
}

func stopSignTelemetryFixture(phase string, currentSpeed float64, desiredSpeed float64) map[string]any {
	return map[string]any{
		"time":                              currentSpeed * 50,
		"currentSpeed":                      currentSpeed,
		"acceleration":                      0.4,
		"brakePressureAvg":                  0.15,
		"stopSignPhase":                     phase,
		"stopLineDistanceM":                 12.0 - currentSpeed,
		"stopSignLongitudinalErrorM":        -12.0 + currentSpeed,
		"stopSignLateralErrorM":             0.1,
		"stopSignHeadingErrorDeg":           0.2,
		"expertDesiredWheelSteerNormalized": 0.05,
		"expertDesiredSpeedMps":             desiredSpeed,
		"expertThrottle":                    0.7,
		"expertBrake":                       0.2,
		"expertStopProbability":             0.2,
		"expertGoProbability":               0.8,
		"eventCollision":                    false,
	}
}

func stopSignGoalFixture(variationID string, x float64, y float64) map[string]any {
	return map[string]any{
		"task":              "stop-sign",
		"contract":          "stop-sign-goal.v3",
		"captureMode":       "continuous",
		"logicalClipStages": []string{"approach", "brake_stop", "release"},
		"variationId":       variationID,
		"signPose": map[string]any{
			"x": x, "y": y, "z": 30.0, "heading": 180.0,
		},
		"stopLinePose": map[string]any{
			"x": x, "y": y + 2, "z": 30.0, "heading": 180.0,
		},
		"egoStopPose": map[string]any{
			"x": x, "y": y + 4.5, "z": 30.0, "heading": 180.0,
		},
		"startPose": map[string]any{
			"x": x, "y": y + 24.5, "z": 30.0, "heading": 180.0,
		},
		"exitPose": map[string]any{
			"x": x, "y": y - 8, "z": 30.0, "heading": 180.0,
		},
		"variationProfile": map[string]any{
			"contract":                    "stop-sign-variation-profile.v1",
			"baseline":                    false,
			"configuredMotionVariancePct": 20.0,
			"changedDimensions":           []string{"target_speed", "weather", "time"},
			"changeCount":                 3,
			"combinationMagnitudePct":     37.5,
		},
	}
}

func stopSignTransitionsFixture() map[string]any {
	return map[string]any{
		"approachStartedGameTimeMs":  1000.0,
		"brakeStopStartedGameTimeMs": 2000.0,
		"stopConfirmedGameTimeMs":    3000.0,
		"releaseStartedGameTimeMs":   3000.0,
		"completedGameTimeMs":        4000.0,
	}
}

func writeStopSignReportTripFixture(
	t *testing.T,
	tripDir string,
	tripIndex int,
	goal map[string]any,
	outcome map[string]any,
	phases []string,
) {
	t.Helper()
	samples := make([]DatasetSample, 0, len(phases))
	for _, phase := range phases {
		samples = append(samples, DatasetSample{Task: stopSignTask, Phase: phase, ClipStage: stopSignClipStageForPhase(phase)})
	}
	samples = decorateStopSignSamples(samples, tripMetadata{
		StopSignGoal:    goal,
		StopSignOutcome: outcome,
	})
	writeDatasetReportTripFixture(t, tripDir, tripFixture{
		runID:        "run-stop-sign",
		sceneID:      "stop-sign",
		sceneVariant: "continuous-v2",
		status:       ProcessingStatus{State: "completed", FrameCount: len(phases), SampleCount: len(phases)},
		samples:      samples,
	})
	writeJSONFile(t, filepath.Join(tripDir, "metadata.json"), tripMetadata{
		RunID:           "run-stop-sign",
		SceneID:         "stop-sign",
		SceneVariant:    "continuous-v2",
		TripIndex:       tripIndex,
		StopSignGoal:    cloneMap(goal),
		StopSignOutcome: cloneMap(outcome),
	})
}
