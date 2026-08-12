package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	datasetproc "awesomeProject/internal/dataset"
)

type processingReadinessSummary struct {
	ConfigFingerprint         string         `json:"configFingerprint"`
	Scope                     string         `json:"scope"`
	RunCount                  int            `json:"runCount"`
	TotalTripCount            int            `json:"totalTripCount"`
	TotalDatasetCount         int            `json:"totalDatasetCount"`
	SelectedTripCount         int            `json:"selectedTripCount"`
	SelectedDatasetCount      int            `json:"selectedDatasetCount"`
	CurrentTripCount          int            `json:"currentTripCount"`
	MissingDatasetCount       int            `json:"missingDatasetCount"`
	UnreadyDatasetCount       int            `json:"unreadyDatasetCount"`
	ExcludedTripCount         int            `json:"excludedTripCount"`
	ReadyRunIDs               []string       `json:"readyRunIds"`
	TrainingEligibleTripCount int            `json:"trainingEligibleTripCount"`
	TrainingSampleCount       int            `json:"trainingSampleCount"`
	TrainingEligibleRunIDs    []string       `json:"trainingEligibleRunIds"`
	TrainingLocationCount     int            `json:"trainingLocationCount"`
	TrainingClipStageCounts   map[string]int `json:"trainingClipStageCounts"`
	SuggestedTrainRunIDs      []string       `json:"suggestedTrainRunIds"`
	SuggestedValRunIDs        []string       `json:"suggestedValRunIds"`
	TrainingEligibilityErrors int            `json:"trainingEligibilityErrorCount"`
	TrainingReady             bool           `json:"trainingReady"`
}

type processingRunReadiness struct {
	tripCount         int
	currentTripCount  int
	eligibleTripCount int
	eligibilityError  bool
	locationKeys      map[string]struct{}
	clipStages        map[string]struct{}
}

type trainingStopSignMetadata struct {
	SceneID      string `json:"sceneId"`
	SceneVariant string `json:"sceneVariant"`
	StopSignGoal struct {
		Task              string                `json:"task"`
		Contract          string                `json:"contract"`
		CaptureMode       string                `json:"captureMode"`
		LogicalClipStages []string              `json:"logicalClipStages"`
		CatalogID         string                `json:"catalogId"`
		SignPose          *trainingStopSignPose `json:"signPose"`
		StopLinePose      *trainingStopSignPose `json:"stopLinePose"`
		EgoStopPose       *trainingStopSignPose `json:"egoStopPose"`
		StartPose         *trainingStopSignPose `json:"startPose"`
		ExitPose          *trainingStopSignPose `json:"exitPose"`
	} `json:"stopSignGoal"`
	StopSignOutcome struct {
		Success          bool                          `json:"success"`
		Status           string                        `json:"status"`
		StageTransitions *trainingStageTransitionTimes `json:"stageTransitions"`
	} `json:"stopSignOutcome"`
}

type trainingStageTransitionTimes struct {
	ApproachStartedGameTimeMS  *float64 `json:"approachStartedGameTimeMs"`
	BrakeStopStartedGameTimeMS *float64 `json:"brakeStopStartedGameTimeMs"`
	StopConfirmedGameTimeMS    *float64 `json:"stopConfirmedGameTimeMs"`
	ReleaseStartedGameTimeMS   *float64 `json:"releaseStartedGameTimeMs"`
	CompletedGameTimeMS        *float64 `json:"completedGameTimeMs"`
}

type trainingStopSignPose struct {
	X       *float64 `json:"x"`
	Y       *float64 `json:"y"`
	Z       *float64 `json:"z"`
	Heading *float64 `json:"heading"`
}

func runProcessingStatus(args []string, output io.Writer) error {
	if output == nil {
		return errors.New("processing status output must not be nil")
	}
	flags := flag.NewFlagSet("processing-status", flag.ContinueOnError)
	flags.SetOutput(output)
	root := flags.String("root", filepath.Join(defaultBackendDataRoot(), "runs"), "root directory to inspect for trip folders")
	stopSignOnly := flags.Bool("stop-sign-only", false, "summarize only stop-sign continuous-v2 scene folders")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 {
		return fmt.Errorf("processing-status does not accept trip arguments: %v", flags.Args())
	}

	config, err := loadDatasetConfigForCLI()
	if err != nil {
		return fmt.Errorf("load dataset frame-window config: %w", err)
	}
	processor := datasetproc.NewProcessor(datasetProcessorOptions(config)...)
	summary, err := buildProcessingReadinessSummary(*root, *stopSignOnly, processor)
	if err != nil {
		return err
	}
	encoder := json.NewEncoder(output)
	encoder.SetIndent("", "  ")
	return encoder.Encode(summary)
}

func buildProcessingReadinessSummary(
	runsRoot string,
	stopSignOnly bool,
	processor processingProcessorAPI,
) (processingReadinessSummary, error) {
	if processor == nil {
		return processingReadinessSummary{}, errors.New("processing readiness requires a processor")
	}
	summary := processingReadinessSummary{
		ConfigFingerprint:       processor.ConfigFingerprint(),
		Scope:                   "all",
		ReadyRunIDs:             make([]string, 0),
		TrainingEligibleRunIDs:  make([]string, 0),
		SuggestedTrainRunIDs:    make([]string, 0),
		SuggestedValRunIDs:      make([]string, 0),
		TrainingClipStageCounts: make(map[string]int),
	}
	if stopSignOnly {
		summary.Scope = "stop-sign-continuous-v2"
	}

	runCount, rootExists, err := countProcessingRunDirs(runsRoot)
	if err != nil {
		return processingReadinessSummary{}, err
	}
	summary.RunCount = runCount
	if !rootExists {
		return summary, nil
	}

	tripDirs, err := datasetproc.CollectTripDirs(runsRoot, nil)
	if err != nil {
		return processingReadinessSummary{}, err
	}
	summary.TotalTripCount = len(tripDirs)
	for _, tripDir := range tripDirs {
		if fileExists(filepath.Join(tripDir, "dataset.jsonl")) {
			summary.TotalDatasetCount++
		}
	}

	selectedTripDirs := tripDirs
	if stopSignOnly {
		selectedTripDirs = filterStopSignTripDirs(tripDirs)
	}
	summary.SelectedTripCount = len(selectedTripDirs)
	summary.ExcludedTripCount = summary.TotalTripCount - summary.SelectedTripCount
	runReadiness := make(map[string]*processingRunReadiness)
	for _, tripDir := range selectedTripDirs {
		runID := filepath.Base(tripRunDir(tripDir))
		run := runReadiness[runID]
		if run == nil {
			run = &processingRunReadiness{}
			runReadiness[runID] = run
		}
		run.tripCount++
		if fileExists(filepath.Join(tripDir, "dataset.jsonl")) {
			summary.SelectedDatasetCount++
		}
		if !processor.TripOutputsCurrent(tripDir) {
			continue
		}
		summary.CurrentTripCount++
		run.currentTripCount++
		eligible, metadataValid, locationKey, clipStage := trainingStopSignTripEligible(tripDir)
		if !metadataValid {
			summary.TrainingEligibilityErrors++
			run.eligibilityError = true
			continue
		}
		if !eligible {
			continue
		}
		status, err := datasetproc.ReadStatusFile(filepath.Join(tripDir, "processing.json"))
		if err != nil || status.SampleCount <= 0 {
			continue
		}
		summary.TrainingEligibleTripCount++
		summary.TrainingSampleCount += status.SampleCount
		run.eligibleTripCount++
		if run.locationKeys == nil {
			run.locationKeys = make(map[string]struct{})
		}
		run.locationKeys[locationKey] = struct{}{}
		if run.clipStages == nil {
			run.clipStages = make(map[string]struct{})
		}
		for _, stage := range logicalClipStagesForTrip(clipStage) {
			run.clipStages[stage] = struct{}{}
			summary.TrainingClipStageCounts[stage]++
		}
	}
	for runID, run := range runReadiness {
		if run.tripCount == 0 || run.currentTripCount != run.tripCount {
			continue
		}
		summary.ReadyRunIDs = append(summary.ReadyRunIDs, runID)
		if run.eligibleTripCount > 0 && !run.eligibilityError && hasEveryTrainingClipStage(run.clipStages) {
			summary.TrainingEligibleRunIDs = append(summary.TrainingEligibleRunIDs, runID)
		}
	}
	sort.Strings(summary.ReadyRunIDs)
	sort.Strings(summary.TrainingEligibleRunIDs)
	summary.SuggestedTrainRunIDs, summary.SuggestedValRunIDs, summary.TrainingLocationCount =
		buildDisjointTrainingSplit(runReadiness)
	summary.TrainingReady = len(summary.SuggestedTrainRunIDs) > 0 && len(summary.SuggestedValRunIDs) > 0
	summary.MissingDatasetCount = summary.SelectedTripCount - summary.SelectedDatasetCount
	summary.UnreadyDatasetCount = summary.SelectedDatasetCount - summary.CurrentTripCount
	return summary, nil
}

func trainingStopSignTripEligible(tripDir string) (eligible bool, metadataValid bool, locationKey, clipStage string) {
	body, err := os.ReadFile(filepath.Join(tripDir, "metadata.json"))
	if errors.Is(err, os.ErrNotExist) {
		return false, true, "", ""
	}
	if err != nil {
		return false, false, "", ""
	}
	var metadata trainingStopSignMetadata
	if err := json.Unmarshal(body, &metadata); err != nil {
		return false, false, "", ""
	}
	eligible = strings.EqualFold(strings.TrimSpace(metadata.SceneID), "stop-sign") &&
		strings.EqualFold(strings.TrimSpace(metadata.SceneVariant), "continuous-v2") &&
		strings.EqualFold(strings.TrimSpace(metadata.StopSignGoal.Task), "stop-sign") &&
		strings.EqualFold(strings.TrimSpace(metadata.StopSignGoal.Contract), "stop-sign-goal.v3") &&
		strings.EqualFold(strings.TrimSpace(metadata.StopSignGoal.CaptureMode), "continuous") &&
		validTrainingLogicalClipStages(metadata.StopSignGoal.LogicalClipStages) &&
		finiteTrainingStopSignPose(metadata.StopSignGoal.SignPose) &&
		finiteTrainingStopSignPose(metadata.StopSignGoal.StopLinePose) &&
		finiteTrainingStopSignPose(metadata.StopSignGoal.EgoStopPose) &&
		finiteTrainingStopSignPose(metadata.StopSignGoal.StartPose) &&
		finiteTrainingStopSignPose(metadata.StopSignGoal.ExitPose) &&
		metadata.StopSignOutcome.Success &&
		strings.EqualFold(strings.TrimSpace(metadata.StopSignOutcome.Status), "succeeded") &&
		orderedTrainingStageTransitions(metadata.StopSignOutcome.StageTransitions)
	if !eligible {
		return false, true, "", ""
	}
	clipStage = "continuous"
	if catalogID := strings.ToLower(strings.TrimSpace(metadata.StopSignGoal.CatalogID)); catalogID != "" {
		return true, true, "catalog:" + catalogID, clipStage
	}
	return true, true, trainingStopSignLocationKey(metadata.StopSignGoal.SignPose), clipStage
}

func logicalClipStagesForTrip(value string) []string {
	if value == "continuous" {
		return []string{"approach", "brake_stop", "release"}
	}
	return []string{value}
}

func sliceToStringSet(values []string) map[string]struct{} {
	result := make(map[string]struct{}, len(values))
	for _, value := range values {
		result[strings.TrimSpace(value)] = struct{}{}
	}
	return result
}

func validTrainingLogicalClipStages(values []string) bool {
	return len(values) == 3 && hasEveryTrainingClipStage(sliceToStringSet(values))
}

func orderedTrainingStageTransitions(value *trainingStageTransitionTimes) bool {
	if value == nil {
		return false
	}
	values := []*float64{
		value.ApproachStartedGameTimeMS,
		value.BrakeStopStartedGameTimeMS,
		value.StopConfirmedGameTimeMS,
		value.ReleaseStartedGameTimeMS,
		value.CompletedGameTimeMS,
	}
	previous := -math.MaxFloat64
	for _, current := range values {
		if !finiteFloatPointer(current) || *current < previous {
			return false
		}
		previous = *current
	}
	return true
}

func hasEveryTrainingClipStage(stages map[string]struct{}) bool {
	for _, stage := range []string{"approach", "brake_stop", "release"} {
		if _, ok := stages[stage]; !ok {
			return false
		}
	}
	return true
}

func trainingStopSignLocationKey(pose *trainingStopSignPose) string {
	heading := math.Mod(*pose.Heading, 360)
	if heading < 0 {
		heading += 360
	}
	return fmt.Sprintf("sign:%.2f:%.2f:%.2f:%.1f", *pose.X, *pose.Y, *pose.Z, heading)
}

func buildDisjointTrainingSplit(runs map[string]*processingRunReadiness) (trainRunIDs, valRunIDs []string, locationCount int) {
	eligibleRuns := make(map[string]*processingRunReadiness)
	locations := make(map[string]struct{})
	for runID, run := range runs {
		if run.tripCount == 0 || run.currentTripCount != run.tripCount || run.eligibleTripCount == 0 || run.eligibilityError ||
			!hasEveryTrainingClipStage(run.clipStages) {
			continue
		}
		eligibleRuns[runID] = run
		for locationKey := range run.locationKeys {
			locations[locationKey] = struct{}{}
		}
	}
	locationCount = len(locations)
	locationKeys := make([]string, 0, len(locations))
	for locationKey := range locations {
		locationKeys = append(locationKeys, locationKey)
	}
	sort.Strings(locationKeys)

	for _, validationLocation := range locationKeys {
		candidateTrain := make([]string, 0)
		candidateVal := make([]string, 0)
		for runID, run := range eligibleRuns {
			_, containsValidationLocation := run.locationKeys[validationLocation]
			switch {
			case len(run.locationKeys) == 1 && containsValidationLocation:
				candidateVal = append(candidateVal, runID)
			case !containsValidationLocation:
				candidateTrain = append(candidateTrain, runID)
			}
		}
		if len(candidateTrain) == 0 || len(candidateVal) == 0 {
			continue
		}
		sort.Strings(candidateTrain)
		sort.Strings(candidateVal)
		if len(candidateTrain) > len(trainRunIDs) ||
			(len(candidateTrain) == len(trainRunIDs) && (len(valRunIDs) == 0 || len(candidateVal) < len(valRunIDs))) {
			trainRunIDs = candidateTrain
			valRunIDs = candidateVal
		}
	}
	return trainRunIDs, valRunIDs, locationCount
}

func finiteTrainingStopSignPose(pose *trainingStopSignPose) bool {
	return pose != nil &&
		finiteFloatPointer(pose.X) &&
		finiteFloatPointer(pose.Y) &&
		finiteFloatPointer(pose.Z) &&
		finiteFloatPointer(pose.Heading)
}

func finiteFloatPointer(value *float64) bool {
	return value != nil && !math.IsNaN(*value) && !math.IsInf(*value, 0)
}

func countProcessingRunDirs(runsRoot string) (count int, exists bool, err error) {
	entries, err := os.ReadDir(runsRoot)
	if errors.Is(err, os.ErrNotExist) {
		return 0, false, nil
	}
	if err != nil {
		return 0, false, fmt.Errorf("read processing runs root: %w", err)
	}
	for _, entry := range entries {
		if entry.IsDir() {
			count++
		}
	}
	return count, true, nil
}
