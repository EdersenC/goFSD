package dataset

import (
	"crypto/sha256"
	"fmt"
	"math"
	"strings"
	"time"
)

const stopSignTask = "stop-sign"

var stopSignPhases = map[string]struct{}{
	"accelerate":      {},
	"cruise_approach": {},
	"decelerate":      {},
	"stop_hold":       {},
	"release":         {},
}

func isStopSignTimeline(labels []timedLabel) bool {
	for _, label := range labels {
		if _, ok := stopSignPhase(label.Label["stopSignPhase"]); ok {
			return true
		}
	}
	return false
}

func buildStopSignSamplesWithImageOffsetsAndStats(
	frames []VideoFrame,
	labels []timedLabel,
	anchorPTS float64,
	imageOffsets []int,
	sampleStride int,
	tolerance time.Duration,
	telemetryOffsets []int,
	futureOffsets []int,
	telemetrySampleInterval time.Duration,
) ([]DatasetSample, sampleBuildStats) {
	var stats sampleBuildStats
	toleranceSeconds := tolerance.Seconds()
	alignmentTolerance := telemetryAlignmentTolerance(tolerance, telemetrySampleInterval)
	samples := make([]DatasetSample, 0)

	for anchorIndex := 0; anchorIndex < len(frames); anchorIndex += sampleStride {
		stats.CandidateWindowCount++
		window, ok := buildFrameWindowAtOffsets(frames, anchorIndex, imageOffsets)
		if !ok {
			stats.IncompleteFrameHistoryCount++
			continue
		}

		anchorFrame := frames[anchorIndex]
		anchorGameTime := anchorFrame.PTS - anchorPTS
		current, labelIndex, ok := nearestLabelWithIndex(labels, anchorGameTime, toleranceSeconds)
		if !ok {
			stats.MissingCurrentLabelCount++
			continue
		}
		history, ok := buildTelemetryHistory(
			labels,
			labelIndex,
			telemetryOffsets,
			telemetrySampleInterval,
			alignmentTolerance,
		)
		if !ok {
			stats.IncompleteTelemetryHistoryCount++
			continue
		}
		future, ok := buildTelemetryFuture(
			labels,
			labelIndex,
			futureOffsets,
			telemetrySampleInterval,
			alignmentTolerance,
		)
		if !ok {
			stats.IncompleteTelemetryFutureCount++
			continue
		}

		phase, ok := stopSignPhase(current.Label["stopSignPhase"])
		if !ok {
			stats.InvalidDerivedLabelCount++
			continue
		}
		trainingLabel, ok := buildStopSignTrainingLabel(current.Label, phase, future, futureOffsets)
		if !ok {
			stats.InvalidDerivedLabelCount++
			continue
		}

		samples = append(samples, DatasetSample{
			AnchorVideoPTS:   anchorFrame.PTS,
			AnchorGameTime:   anchorGameTime,
			FramePaths:       window,
			TelemetryHistory: history,
			TelemetryFuture:  future,
			Label:            trainingLabel,
			Task:             stopSignTask,
			Phase:            phase,
		})
		stats.GeneratedSampleCount++
	}

	return samples, stats
}

func buildStopSignTrainingLabel(
	current map[string]any,
	phase string,
	future []GroupedTelemetryItem,
	futureOffsets []int,
) (GroupedLabel, bool) {
	desiredSteer, ok := boundedNumber(current["expertDesiredWheelSteerNormalized"], -1, 1)
	if !ok {
		return GroupedLabel{}, false
	}
	desiredSpeed, ok := nonNegativeNumber(current["expertDesiredSpeedMps"])
	if !ok {
		return GroupedLabel{}, false
	}
	throttle, ok := boundedNumber(current["expertThrottle"], 0, 1)
	if !ok {
		return GroupedLabel{}, false
	}
	brake, ok := boundedNumber(current["expertBrake"], 0, 1)
	if !ok {
		return GroupedLabel{}, false
	}
	stopProbability, ok := boundedNumber(current["expertStopProbability"], 0, 1)
	if !ok {
		return GroupedLabel{}, false
	}
	goProbability, ok := boundedNumber(current["expertGoProbability"], 0, 1)
	if !ok {
		return GroupedLabel{}, false
	}

	futureSpeedTargets, futureObservedSpeeds, futureStop, futureGo, futurePhases, ok := stopSignFutureTargets(future, futureOffsets)
	if !ok {
		return GroupedLabel{}, false
	}

	return GroupedLabel{
		Control: GroupedLabelControl{
			ExpertDesiredWheelSteerNormalized: desiredSteer,
			ExpertDesiredSpeedMps:             desiredSpeed,
			ExpertThrottle:                    throttle,
			ExpertBrake:                       brake,
			ExpertStopProbability:             stopProbability,
			ExpertGoProbability:               goProbability,
		},
		Aux: GroupedLabelAux{
			StopSignPhase:           phase,
			FutureSpeedTargetsMps:   futureSpeedTargets,
			FutureObservedSpeedMps:  futureObservedSpeeds,
			FutureStopProbabilities: futureStop,
			FutureGoProbabilities:   futureGo,
			FutureStopSignPhases:    futurePhases,
		},
	}, true
}

func stopSignFutureTargets(
	future []GroupedTelemetryItem,
	futureOffsets []int,
) ([]float64, []float64, []float64, []float64, []string, bool) {
	if len(future) == 0 || !validFutureOffsets(futureOffsets) {
		return nil, nil, nil, nil, nil, false
	}
	desiredSpeeds := make([]float64, 0, len(futureOffsets))
	observedSpeeds := make([]float64, 0, len(futureOffsets))
	stopProbabilities := make([]float64, 0, len(futureOffsets))
	goProbabilities := make([]float64, 0, len(futureOffsets))
	phases := make([]string, 0, len(futureOffsets))
	for _, offset := range futureOffsets {
		index := offset - 1
		if index < 0 || index >= len(future) {
			return nil, nil, nil, nil, nil, false
		}
		item := future[index]
		desiredSpeed, ok := nonNegativeNumber(item.Control.ExpertDesiredSpeedMps)
		if !ok {
			return nil, nil, nil, nil, nil, false
		}
		observedSpeed, ok := nonNegativeNumber(item.Aux.CurrentSpeed)
		if !ok {
			return nil, nil, nil, nil, nil, false
		}
		stopProbability, ok := boundedNumber(item.Control.ExpertStopProbability, 0, 1)
		if !ok {
			return nil, nil, nil, nil, nil, false
		}
		goProbability, ok := boundedNumber(item.Control.ExpertGoProbability, 0, 1)
		if !ok {
			return nil, nil, nil, nil, nil, false
		}
		phase, ok := stopSignPhase(item.Aux.StopSignPhase)
		if !ok {
			return nil, nil, nil, nil, nil, false
		}
		desiredSpeeds = append(desiredSpeeds, desiredSpeed)
		observedSpeeds = append(observedSpeeds, observedSpeed)
		stopProbabilities = append(stopProbabilities, stopProbability)
		goProbabilities = append(goProbabilities, goProbability)
		phases = append(phases, phase)
	}
	return desiredSpeeds, observedSpeeds, stopProbabilities, goProbabilities, phases, true
}

func decorateStopSignSamples(samples []DatasetSample, metadata tripMetadata) []DatasetSample {
	eligible, exclusionReason := stopSignTrainingEligibility(metadata.StopSignGoal, metadata.StopSignOutcome)
	locationID := stopSignLocationID(metadata.StopSignGoal)
	variationID, _ := nonEmptyString(metadata.StopSignGoal["variationId"])

	for index := range samples {
		value := eligible
		samples[index].Task = stopSignTask
		samples[index].ScenarioLocationID = locationID
		samples[index].ScenarioSplitGroup = locationID
		samples[index].VariationID = variationID
		samples[index].TrainingEligible = &value
		samples[index].TrainingExclusionReason = exclusionReason
		samples[index].StopSignGoal = cloneMap(metadata.StopSignGoal)
		samples[index].StopSignOutcome = cloneMap(metadata.StopSignOutcome)
	}
	return samples
}

func stopSignTrainingEligibility(goal map[string]any, outcome map[string]any) (bool, string) {
	if len(goal) == 0 {
		return false, "missing_stop_sign_goal"
	}
	task, taskOK := nonEmptyString(goal["task"])
	contract, contractOK := nonEmptyString(goal["contract"])
	if !taskOK || !strings.EqualFold(task, stopSignTask) || !contractOK || contract != "stop-sign-goal.v1" || stopSignLocationID(goal) == "" {
		return false, "invalid_stop_sign_goal"
	}
	if len(outcome) == 0 {
		return false, "missing_stop_sign_outcome"
	}
	success, successOK := outcome["success"].(bool)
	status, statusOK := nonEmptyString(outcome["status"])
	if !successOK || !statusOK {
		return false, "invalid_stop_sign_outcome"
	}
	if !success || !strings.EqualFold(status, "succeeded") {
		return false, "stop_sign_outcome_not_succeeded"
	}
	return true, ""
}

func stopSignLocationID(goal map[string]any) string {
	pose, ok := goal["signPose"].(map[string]any)
	if !ok {
		return ""
	}
	x, xOK := finiteNumber(pose["x"])
	y, yOK := finiteNumber(pose["y"])
	z, zOK := finiteNumber(pose["z"])
	heading, headingOK := finiteNumber(pose["heading"])
	if !xOK || !yOK || !zOK || !headingOK {
		return ""
	}
	canonical := fmt.Sprintf("%.3f,%.3f,%.3f,%.2f", x, y, z, heading)
	digest := sha256.Sum256([]byte(canonical))
	return fmt.Sprintf("stop-sign-location-%x", digest[:8])
}

func stopSignPhase(value any) (string, bool) {
	phase, ok := nonEmptyString(value)
	if !ok {
		return "", false
	}
	_, ok = stopSignPhases[phase]
	return phase, ok
}

func nonEmptyString(value any) (string, bool) {
	text, ok := value.(string)
	if !ok {
		return "", false
	}
	text = strings.TrimSpace(text)
	return text, text != ""
}

func finiteNumber(value any) (float64, bool) {
	number, ok := numberField(value)
	return number, ok && !math.IsNaN(number) && !math.IsInf(number, 0)
}

func nonNegativeNumber(value any) (float64, bool) {
	number, ok := finiteNumber(value)
	return number, ok && number >= 0
}

func boundedNumber(value any, minimum float64, maximum float64) (float64, bool) {
	number, ok := finiteNumber(value)
	return number, ok && number >= minimum && number <= maximum
}
