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
	ConfigFingerprint         string   `json:"configFingerprint"`
	Scope                     string   `json:"scope"`
	RunCount                  int      `json:"runCount"`
	TotalTripCount            int      `json:"totalTripCount"`
	TotalDatasetCount         int      `json:"totalDatasetCount"`
	SelectedTripCount         int      `json:"selectedTripCount"`
	SelectedDatasetCount      int      `json:"selectedDatasetCount"`
	CurrentTripCount          int      `json:"currentTripCount"`
	MissingDatasetCount       int      `json:"missingDatasetCount"`
	UnreadyDatasetCount       int      `json:"unreadyDatasetCount"`
	ExcludedTripCount         int      `json:"excludedTripCount"`
	ReadyRunIDs               []string `json:"readyRunIds"`
	TrainingEligibleTripCount int      `json:"trainingEligibleTripCount"`
	TrainingSampleCount       int      `json:"trainingSampleCount"`
	TrainingEligibleRunIDs    []string `json:"trainingEligibleRunIds"`
	TrainingLocationCount     int      `json:"trainingLocationCount"`
	SuggestedTrainRunIDs      []string `json:"suggestedTrainRunIds"`
	SuggestedValRunIDs        []string `json:"suggestedValRunIds"`
	TrainingEligibilityErrors int      `json:"trainingEligibilityErrorCount"`
	TrainingReady             bool     `json:"trainingReady"`
}

type processingRunReadiness struct {
	tripCount         int
	currentTripCount  int
	eligibleTripCount int
	eligibilityError  bool
	locationKeys      map[string]struct{}
}

type trainingStopSignMetadata struct {
	SceneID      string `json:"sceneId"`
	SceneVariant string `json:"sceneVariant"`
	StopSignGoal struct {
		Task         string                `json:"task"`
		Contract     string                `json:"contract"`
		SignPose     *trainingStopSignPose `json:"signPose"`
		StopLinePose *trainingStopSignPose `json:"stopLinePose"`
		EgoStopPose  *trainingStopSignPose `json:"egoStopPose"`
		StartPose    *trainingStopSignPose `json:"startPose"`
	} `json:"stopSignGoal"`
	StopSignOutcome struct {
		Success bool   `json:"success"`
		Status  string `json:"status"`
	} `json:"stopSignOutcome"`
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
	stopSignOnly := flags.Bool("stop-sign-only", false, "summarize only stop-sign temporal-v1 scene folders")
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
		ConfigFingerprint:      processor.ConfigFingerprint(),
		Scope:                  "all",
		ReadyRunIDs:            make([]string, 0),
		TrainingEligibleRunIDs: make([]string, 0),
		SuggestedTrainRunIDs:   make([]string, 0),
		SuggestedValRunIDs:     make([]string, 0),
	}
	if stopSignOnly {
		summary.Scope = "stop-sign-temporal-v1"
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
		eligible, metadataValid, locationKey := trainingStopSignTripEligible(tripDir)
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
	}
	for runID, run := range runReadiness {
		if run.tripCount == 0 || run.currentTripCount != run.tripCount {
			continue
		}
		summary.ReadyRunIDs = append(summary.ReadyRunIDs, runID)
		if run.eligibleTripCount > 0 && !run.eligibilityError {
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

func trainingStopSignTripEligible(tripDir string) (eligible bool, metadataValid bool, locationKey string) {
	body, err := os.ReadFile(filepath.Join(tripDir, "metadata.json"))
	if errors.Is(err, os.ErrNotExist) {
		return false, true, ""
	}
	if err != nil {
		return false, false, ""
	}
	var metadata trainingStopSignMetadata
	if err := json.Unmarshal(body, &metadata); err != nil {
		return false, false, ""
	}
	eligible = strings.EqualFold(strings.TrimSpace(metadata.SceneID), "stop-sign") &&
		strings.EqualFold(strings.TrimSpace(metadata.SceneVariant), "temporal-v1") &&
		strings.EqualFold(strings.TrimSpace(metadata.StopSignGoal.Task), "stop-sign") &&
		strings.EqualFold(strings.TrimSpace(metadata.StopSignGoal.Contract), "stop-sign-goal.v1") &&
		finiteTrainingStopSignPose(metadata.StopSignGoal.SignPose) &&
		finiteTrainingStopSignPose(metadata.StopSignGoal.StopLinePose) &&
		finiteTrainingStopSignPose(metadata.StopSignGoal.EgoStopPose) &&
		finiteTrainingStopSignPose(metadata.StopSignGoal.StartPose) &&
		metadata.StopSignOutcome.Success &&
		strings.EqualFold(strings.TrimSpace(metadata.StopSignOutcome.Status), "succeeded")
	if !eligible {
		return false, true, ""
	}
	return true, true, trainingStopSignLocationKey(metadata.StopSignGoal.SignPose)
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
		if run.tripCount == 0 || run.currentTripCount != run.tripCount || run.eligibleTripCount == 0 || run.eligibilityError {
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
