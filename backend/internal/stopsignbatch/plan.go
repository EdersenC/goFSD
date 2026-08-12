// Package stopsignbatch validates and expands deterministic stop-sign collection plans.
package stopsignbatch

import (
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"strings"
)

const (
	MaximumEntries      = 100
	MaximumExpandedJobs = 5000
	MaximumAttemptCount = 50
	MaximumAutoVariants = 100
	MinimumAutoVariants = 2

	DefaultStopDistanceM           = 3.0
	DefaultEgoCenterOffsetM        = 2.5
	DefaultStartDistanceM          = 40.0
	DefaultExitDistanceM           = 8.0
	DefaultTargetSpeedMPS          = 10.0
	DefaultBrakingDecelerationMPS2 = 3.8
	DefaultReleaseAccelerationMPS2 = 3.5
	DefaultStopConfirmationMS      = 250
	DefaultAttemptCount            = 1
	DefaultWeather                 = "EXTRASUNNY"
	DefaultHour                    = 12
	DefaultMinute                  = 0

	minimumStopDistanceM               = 0.5
	maximumStopDistanceM               = 15.0
	minimumEgoCenterOffsetM            = 0.5
	maximumEgoCenterOffsetM            = 8.0
	minimumStartDistanceM              = 5.0
	maximumStartDistanceM              = 250.0
	minimumExitDistanceM               = 2.0
	maximumExitDistanceM               = 50.0
	minimumTargetSpeedMPS              = 0.5
	minimumAutoTargetSpeedMPS          = 10.0
	maximumTargetSpeedMPS              = 15.0
	minimumStopConfirmationMS          = 100
	maximumStopConfirmationMS          = 1_000
	maximumMotionVariancePct           = 25.0
	minimumCapturedStartM              = 16.0
	minimumCapturedExitM               = 8.0
	launchAccelerationMPS2             = 3.5
	minimumBrakingDecelerationMPS2     = 3.5
	maximumBrakingDecelerationMPS2     = 4.2
	maximumAutoBrakingDecelerationMPS2 = 4.05
	minimumReleaseAccelerationMPS2     = 3.0
	maximumReleaseAccelerationMPS2     = 5.0
	maximumAutoReleaseAccelerationMPS2 = 4.2
	minimumRollingCruiseS              = 1.25
)

var (
	ErrInvalidPlan = errors.New("invalid stop-sign collection plan")
	validWeather   = map[string]struct{}{
		"BLIZZARD": {}, "CLEAR": {}, "CLEARING": {}, "CLOUDS": {}, "EXTRASUNNY": {},
		"FOGGY": {}, "HALLOWEEN": {}, "NEUTRAL": {}, "OVERCAST": {}, "RAIN": {},
		"SMOG": {}, "SNOW": {}, "SNOWLIGHT": {}, "THUNDER": {}, "XMAS": {},
	}
)

// Pose is a GTA world pose. Sign heading is the vehicle travel heading through
// the intersection, not the physical prop's facing direction.
type Pose struct {
	X       float64 `json:"x"`
	Y       float64 `json:"y"`
	Z       float64 `json:"z"`
	Heading float64 `json:"heading"`
}

// WorldPosition identifies the physical catalog prop. It is deliberately
// separate from the lane poses captured by the operator.
type WorldPosition struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
	Z float64 `json:"z"`
}

type TimeOfDay struct {
	Hour   int `json:"hour"`
	Minute int `json:"minute"`
}

type RGBColor struct {
	R int `json:"r"`
	G int `json:"g"`
	B int `json:"b"`
}

// VehicleVariant changes only fields present in the request. An empty vehicle
// keeps the runner's current model and color.
type VehicleVariant struct {
	Model string    `json:"model,omitempty"`
	Color *RGBColor `json:"color,omitempty"`
}

// Variation overrides collection conditions for one sign. Geometry remains
// sign-relative so every job has one unambiguous stop line and approach.
type Variation struct {
	ID                      string          `json:"id"`
	StopDistanceM           *float64        `json:"stopDistanceM,omitempty"`
	EgoCenterOffsetM        *float64        `json:"egoCenterOffsetM,omitempty"`
	StartDistanceM          *float64        `json:"startDistanceM,omitempty"`
	ExitDistanceM           *float64        `json:"exitDistanceM,omitempty"`
	TargetSpeedMPS          *float64        `json:"targetSpeedMps,omitempty"`
	BrakingDecelerationMPS2 *float64        `json:"brakingDecelerationMps2,omitempty"`
	ReleaseAccelerationMPS2 *float64        `json:"releaseAccelerationMps2,omitempty"`
	StopConfirmationMS      *int            `json:"stopConfirmationMs,omitempty"`
	AttemptCount            *int            `json:"attemptCount,omitempty"`
	Weather                 *string         `json:"weather,omitempty"`
	Time                    *TimeOfDay      `json:"time,omitempty"`
	Vehicle                 *VehicleVariant `json:"vehicle,omitempty"`
}

// AutoVariationSpec expands one captured scene into deterministic collection
// jobs. MotionVariancePct bounds route and exit changes. Target speeds cover
// the configured minimum through the global maximum. Each generated start is
// derived from its target speed so faster jobs have enough
// acceleration, stable-cruise, and braking distance.
type AutoVariationSpec struct {
	Count             int     `json:"count"`
	MotionVariancePct float64 `json:"motionVariancePct"`
}

// Entry defines one ordered stop sign plus optional base collection settings.
// Missing settings use the package defaults exported above.
type Entry struct {
	ID                      string             `json:"id"`
	SignPose                Pose               `json:"signPose"`
	CatalogID               string             `json:"catalogId,omitempty"`
	CatalogPosition         *WorldPosition     `json:"catalogPosition,omitempty"`
	StartPose               *Pose              `json:"startPose,omitempty"`
	EgoStopPose             *Pose              `json:"egoStopPose,omitempty"`
	ExitPose                *Pose              `json:"exitPose,omitempty"`
	AutoVariations          *AutoVariationSpec `json:"autoVariations,omitempty"`
	StopDistanceM           *float64           `json:"stopDistanceM,omitempty"`
	EgoCenterOffsetM        *float64           `json:"egoCenterOffsetM,omitempty"`
	StartDistanceM          *float64           `json:"startDistanceM,omitempty"`
	ExitDistanceM           *float64           `json:"exitDistanceM,omitempty"`
	TargetSpeedMPS          *float64           `json:"targetSpeedMps,omitempty"`
	BrakingDecelerationMPS2 *float64           `json:"brakingDecelerationMps2,omitempty"`
	ReleaseAccelerationMPS2 *float64           `json:"releaseAccelerationMps2,omitempty"`
	StopConfirmationMS      *int               `json:"stopConfirmationMs,omitempty"`
	AttemptCount            *int               `json:"attemptCount,omitempty"`
	Weather                 *string            `json:"weather,omitempty"`
	Time                    *TimeOfDay         `json:"time,omitempty"`
	Vehicle                 *VehicleVariant    `json:"vehicle,omitempty"`
	Variations              []Variation        `json:"variations,omitempty"`
}

type Plan struct {
	ID      string  `json:"id"`
	Seed    string  `json:"seed"`
	Entries []Entry `json:"entries"`
}

// Job is a fully explicit collection unit. Catalog scenes preserve the three
// operator-captured poses; compatibility scenes remain sign-relative.
type Job struct {
	ID                      string           `json:"id"`
	EntryID                 string           `json:"entryId"`
	VariationID             string           `json:"variationId"`
	CatalogID               string           `json:"catalogId,omitempty"`
	CatalogPosition         *WorldPosition   `json:"catalogPosition,omitempty"`
	SignPose                Pose             `json:"signPose"`
	StopLinePose            Pose             `json:"stopLinePose"`
	EgoStopPose             Pose             `json:"egoStopPose"`
	StartPose               Pose             `json:"startPose"`
	ExitPose                Pose             `json:"exitPose"`
	StopDistanceM           float64          `json:"stopDistanceM"`
	EgoCenterOffsetM        float64          `json:"egoCenterOffsetM"`
	StartDistanceM          float64          `json:"startDistanceM"`
	ExitDistanceM           float64          `json:"exitDistanceM"`
	TargetSpeedMPS          float64          `json:"targetSpeedMps"`
	BrakingDecelerationMPS2 float64          `json:"brakingDecelerationMps2"`
	ReleaseAccelerationMPS2 float64          `json:"releaseAccelerationMps2"`
	StopConfirmationMS      int              `json:"stopConfirmationMs"`
	AttemptCount            int              `json:"attemptCount"`
	Weather                 string           `json:"weather"`
	Time                    TimeOfDay        `json:"time"`
	Vehicle                 VehicleVariant   `json:"vehicle"`
	Seed                    string           `json:"seed"`
	VariationProfile        VariationProfile `json:"variationProfile"`
}

type settings struct {
	stopDistanceM           float64
	egoCenterOffsetM        float64
	startDistanceM          float64
	exitDistanceM           float64
	targetSpeedMPS          float64
	brakingDecelerationMPS2 float64
	releaseAccelerationMPS2 float64
	stopConfirmationMS      int
	attemptCount            int
	weather                 string
	time                    TimeOfDay
	vehicle                 VehicleVariant
}

// Expand validates a plan and emits jobs in entry order, then variation order.
// It never reads time, randomness, world state, or map natives.
func Expand(plan Plan) ([]Job, error) {
	planID := strings.TrimSpace(plan.ID)
	seed := strings.TrimSpace(plan.Seed)
	if planID == "" {
		return nil, invalid("id is required")
	}
	if seed == "" {
		return nil, invalid("seed is required")
	}
	if len(plan.Entries) == 0 || len(plan.Entries) > MaximumEntries {
		return nil, invalid("entries must contain between 1 and %d items", MaximumEntries)
	}

	entryIDs := make(map[string]struct{}, len(plan.Entries))
	jobIDs := make(map[string]struct{}, len(plan.Entries))
	jobs := make([]Job, 0, len(plan.Entries))
	for entryIndex, entry := range plan.Entries {
		entryID := strings.TrimSpace(entry.ID)
		if entryID == "" {
			return nil, invalid("entries[%d].id is required", entryIndex)
		}
		if _, exists := entryIDs[entryID]; exists {
			return nil, invalid("entry id %q is duplicated", entryID)
		}
		entryIDs[entryID] = struct{}{}
		if capturedEntry(entry) {
			capturedJobs, err := expandCapturedEntry(seed, entryID, entry, entryIndex)
			if err != nil {
				return nil, err
			}
			for _, job := range capturedJobs {
				if _, exists := jobIDs[job.ID]; exists {
					return nil, invalid("expanded job id %q is duplicated", job.ID)
				}
				jobIDs[job.ID] = struct{}{}
				jobs = append(jobs, job)
				if len(jobs) > MaximumExpandedJobs {
					return nil, invalid("plan expands to more than %d jobs", MaximumExpandedJobs)
				}
			}
			continue
		}
		if err := validateSignPose(fmt.Sprintf("entries[%d].signPose", entryIndex), entry.SignPose); err != nil {
			return nil, err
		}

		base := defaultSettings()
		if err := applyEntrySettings(&base, entry, fmt.Sprintf("entries[%d]", entryIndex)); err != nil {
			return nil, err
		}
		entryJobStart := len(jobs)
		variations := entry.Variations
		if len(variations) == 0 {
			variations = []Variation{{ID: "base"}}
		}
		variationIDs := make(map[string]struct{}, len(variations))
		for variationIndex, variation := range variations {
			variationID := strings.TrimSpace(variation.ID)
			if variationID == "" {
				return nil, invalid("entries[%d].variations[%d].id is required", entryIndex, variationIndex)
			}
			if _, exists := variationIDs[variationID]; exists {
				return nil, invalid("entries[%d] variation id %q is duplicated", entryIndex, variationID)
			}
			variationIDs[variationID] = struct{}{}

			resolved := cloneSettings(base)
			label := fmt.Sprintf("entries[%d].variations[%d]", entryIndex, variationIndex)
			if err := applyVariationSettings(&resolved, variation, label); err != nil {
				return nil, err
			}
			stopLinePose := poseBehind(entry.SignPose, resolved.stopDistanceM)
			egoStopPose := poseBehind(stopLinePose, resolved.egoCenterOffsetM)
			startPose := poseBehind(egoStopPose, resolved.startDistanceM)
			exitPose := poseAhead(entry.SignPose, resolved.exitDistanceM)
			if err := validatePose(label+".stopLinePose", stopLinePose); err != nil {
				return nil, err
			}
			if err := validatePose(label+".egoStopPose", egoStopPose); err != nil {
				return nil, err
			}
			if err := validatePose(label+".startPose", startPose); err != nil {
				return nil, err
			}
			if err := validatePose(label+".exitPose", exitPose); err != nil {
				return nil, err
			}

			jobID := entryID + ":" + variationID
			if _, exists := jobIDs[jobID]; exists {
				return nil, invalid("expanded job id %q is duplicated; choose unambiguous entry and variation ids", jobID)
			}
			jobIDs[jobID] = struct{}{}
			jobs = append(jobs, Job{
				ID:                      jobID,
				EntryID:                 entryID,
				VariationID:             variationID,
				SignPose:                entry.SignPose,
				StopLinePose:            stopLinePose,
				EgoStopPose:             egoStopPose,
				StartPose:               startPose,
				ExitPose:                exitPose,
				StopDistanceM:           resolved.stopDistanceM,
				EgoCenterOffsetM:        resolved.egoCenterOffsetM,
				StartDistanceM:          resolved.startDistanceM,
				ExitDistanceM:           resolved.exitDistanceM,
				TargetSpeedMPS:          resolved.targetSpeedMPS,
				BrakingDecelerationMPS2: resolved.brakingDecelerationMPS2,
				ReleaseAccelerationMPS2: resolved.releaseAccelerationMPS2,
				StopConfirmationMS:      resolved.stopConfirmationMS,
				AttemptCount:            resolved.attemptCount,
				Weather:                 resolved.weather,
				Time:                    resolved.time,
				Vehicle:                 resolved.vehicle,
				Seed:                    seed + ":" + jobID,
			})
			if len(jobs) > MaximumExpandedJobs {
				return nil, invalid("plan expands to more than %d jobs", MaximumExpandedJobs)
			}
		}
		applyVariationProfiles(jobs[entryJobStart:], 0)
	}

	return jobs, nil
}

func capturedEntry(entry Entry) bool {
	return entry.CatalogPosition != nil || entry.StartPose != nil || entry.EgoStopPose != nil ||
		entry.ExitPose != nil || entry.AutoVariations != nil || strings.TrimSpace(entry.CatalogID) != ""
}

func expandCapturedEntry(seed, entryID string, entry Entry, entryIndex int) ([]Job, error) {
	label := fmt.Sprintf("entries[%d]", entryIndex)
	catalogID := strings.TrimSpace(entry.CatalogID)
	if catalogID == "" || len(catalogID) > 120 {
		return nil, invalid("%s.catalogId must contain between 1 and 120 characters", label)
	}
	if err := validateWorldPosition(label+".catalogPosition", entry.CatalogPosition); err != nil {
		return nil, err
	}
	if entry.StartPose == nil || entry.EgoStopPose == nil || entry.ExitPose == nil {
		return nil, invalid("%s requires captured startPose, egoStopPose, and exitPose", label)
	}
	if entry.AutoVariations == nil {
		return nil, invalid("%s.autoVariations is required for a captured scene", label)
	}
	if len(entry.Variations) != 0 {
		return nil, invalid("%s cannot combine autoVariations with manual variations", label)
	}
	for poseLabel, pose := range map[string]Pose{
		"startPose": *entry.StartPose, "egoStopPose": *entry.EgoStopPose, "exitPose": *entry.ExitPose,
	} {
		if err := validateSignPose(label+"."+poseLabel, pose); err != nil {
			return nil, err
		}
	}
	if err := validateCapturedBaseRoute(label, *entry.StartPose, *entry.EgoStopPose, *entry.ExitPose); err != nil {
		return nil, err
	}
	spec := *entry.AutoVariations
	if spec.Count < MinimumAutoVariants || spec.Count > MaximumAutoVariants {
		return nil, invalid("%s.autoVariations.count must be between %d and %d", label, MinimumAutoVariants, MaximumAutoVariants)
	}
	if math.IsNaN(spec.MotionVariancePct) || math.IsInf(spec.MotionVariancePct, 0) ||
		spec.MotionVariancePct < 0 || spec.MotionVariancePct > maximumMotionVariancePct {
		return nil, invalid("%s.autoVariations.motionVariancePct must be between 0 and %.0f", label, maximumMotionVariancePct)
	}

	base := defaultSettings()
	if err := applyEntrySettings(&base, entry, label); err != nil {
		return nil, err
	}
	baseStartDistanceM := planarDistance(*entry.StartPose, *entry.EgoStopPose)
	if base.targetSpeedMPS != minimumAutoTargetSpeedMPS {
		return nil, invalid("%s.targetSpeedMps must equal %.1f for the automatic %.1f-%.1fm/s range", label, minimumAutoTargetSpeedMPS, minimumAutoTargetSpeedMPS, maximumTargetSpeedMPS)
	}
	if base.brakingDecelerationMPS2 != DefaultBrakingDecelerationMPS2 || base.releaseAccelerationMPS2 != DefaultReleaseAccelerationMPS2 {
		return nil, invalid(
			"%s automatic behavior variants require baseline brakingDecelerationMps2 %.2f and releaseAccelerationMps2 %.2f",
			label,
			DefaultBrakingDecelerationMPS2,
			DefaultReleaseAccelerationMPS2,
		)
	}
	jobs := make([]Job, 0, spec.Count)
	for variantIndex := 0; variantIndex < spec.Count; variantIndex++ {
		variationID := fmt.Sprintf("auto-%03d", variantIndex+1)
		variantSeed := seed + ":" + entryID + ":" + variationID
		baseline := variantIndex == 0
		resolved := capturedVariantSettings(base, variantSeed, variantIndex, spec.MotionVariancePct)
		startPose, stopPose, exitPose := capturedVariantPoses(
			variantSeed,
			*entry.StartPose,
			*entry.EgoStopPose,
			*entry.ExitPose,
			spec.MotionVariancePct,
			baseline,
		)
		startDistanceM := coupledVariantStartDistanceM(
			baseStartDistanceM,
			base.targetSpeedMPS,
			base.brakingDecelerationMPS2,
			resolved.targetSpeedMPS,
			resolved.brakingDecelerationMPS2,
		)
		startPose = poseAtApproachDistance(startPose, stopPose, startDistanceM)
		startDistanceM = planarDistance(startPose, stopPose)
		exitDistanceM := planarDistance(stopPose, exitPose)
		resolved.startDistanceM = startDistanceM
		resolved.exitDistanceM = exitDistanceM
		if err := validateSettings(resolved, label+"."+variationID); err != nil {
			return nil, err
		}
		if err := validateRollingApproachDistance(label+"."+variationID, startDistanceM, resolved.targetSpeedMPS, resolved.brakingDecelerationMPS2); err != nil {
			return nil, err
		}
		if err := validateCapturedRoute(label+"."+variationID, startPose, stopPose, exitPose); err != nil {
			return nil, err
		}
		stopLinePose := poseAhead(stopPose, resolved.egoCenterOffsetM)
		signPose := poseAhead(stopLinePose, resolved.stopDistanceM)
		catalogPosition := *entry.CatalogPosition
		jobs = append(jobs, Job{
			ID:                      entryID + ":" + variationID,
			EntryID:                 entryID,
			VariationID:             variationID,
			CatalogID:               catalogID,
			CatalogPosition:         &catalogPosition,
			SignPose:                signPose,
			StopLinePose:            stopLinePose,
			EgoStopPose:             stopPose,
			StartPose:               startPose,
			ExitPose:                exitPose,
			StopDistanceM:           resolved.stopDistanceM,
			EgoCenterOffsetM:        resolved.egoCenterOffsetM,
			StartDistanceM:          startDistanceM,
			ExitDistanceM:           exitDistanceM,
			TargetSpeedMPS:          resolved.targetSpeedMPS,
			BrakingDecelerationMPS2: resolved.brakingDecelerationMPS2,
			ReleaseAccelerationMPS2: resolved.releaseAccelerationMPS2,
			StopConfirmationMS:      resolved.stopConfirmationMS,
			AttemptCount:            resolved.attemptCount,
			Weather:                 resolved.weather,
			Time:                    resolved.time,
			Vehicle:                 cloneVehicle(resolved.vehicle),
			Seed:                    variantSeed,
		})
	}
	applyVariationProfiles(jobs, spec.MotionVariancePct)
	return jobs, nil
}

func validateRollingApproachDistance(label string, startDistanceM, targetSpeedMPS, brakingDecelerationMPS2 float64) error {
	requiredDistanceM := requiredRollingStartDistanceM(targetSpeedMPS, brakingDecelerationMPS2)
	if startDistanceM+1e-6 < requiredDistanceM {
		return invalid(
			"%s.startPose must be at least %.1fm before egoStopPose to reach %.1fm/s (%.1fmph) and record stable cruise",
			label,
			requiredDistanceM,
			targetSpeedMPS,
			targetSpeedMPS*2.2369362921,
		)
	}
	return nil
}

func requiredRollingStartDistanceM(targetSpeedMPS, brakingDecelerationMPS2 float64) float64 {
	accelerationDistanceM := targetSpeedMPS * targetSpeedMPS / (2 * launchAccelerationMPS2)
	cruiseDistanceM := targetSpeedMPS * minimumRollingCruiseS
	brakingDistanceM := targetSpeedMPS * targetSpeedMPS / (2 * brakingDecelerationMPS2)
	return accelerationDistanceM + cruiseDistanceM + brakingDistanceM
}

func coupledVariantStartDistanceM(
	baseStartDistanceM, baseTargetSpeedMPS, baseBrakingDecelerationMPS2,
	targetSpeedMPS, targetBrakingDecelerationMPS2 float64,
) float64 {
	capturedCruiseBufferM := math.Max(0, baseStartDistanceM-requiredRollingStartDistanceM(baseTargetSpeedMPS, baseBrakingDecelerationMPS2))
	targetDistanceM := requiredRollingStartDistanceM(targetSpeedMPS, targetBrakingDecelerationMPS2) + capturedCruiseBufferM
	return math.Min(maximumStartDistanceM, math.Max(minimumCapturedStartM, targetDistanceM))
}

func capturedVariantPoses(seed string, start, stop, exit Pose, variancePct float64, baseline bool) (Pose, Pose, Pose) {
	if baseline || variancePct == 0 {
		return start, stop, exit
	}
	strength := variancePct / maximumMotionVariancePct
	motionFraction := variancePct / 100
	stopForward, stopRight := headingBasis(stop.Heading)
	stop = translatePose(
		stop,
		stopForward,
		stopRight,
		centeredVariant(seed, "stop-longitudinal")*0.25*strength,
		centeredVariant(seed, "stop-lateral")*0.15*strength,
	)
	stop.Heading = normalizedHeading(stop.Heading + centeredVariant(seed, "stop-heading")*2*strength)
	start = scalePoseFromAnchor(start, stop, 1+centeredVariant(seed, "start-distance")*motionFraction)
	exit = scalePoseFromAnchor(exit, stop, 1+centeredVariant(seed, "exit-distance")*motionFraction)
	startForward, startRight := headingBasis(start.Heading)
	exitForward, exitRight := headingBasis(exit.Heading)
	start = translatePose(start, startForward, startRight, 0, centeredVariant(seed, "start-lateral")*0.5*strength)
	exit = translatePose(exit, exitForward, exitRight, 0, centeredVariant(seed, "exit-lateral")*0.5*strength)
	start.Heading = normalizedHeading(start.Heading + centeredVariant(seed, "start-heading")*3*strength)
	exit.Heading = normalizedHeading(exit.Heading + centeredVariant(seed, "exit-heading")*3*strength)
	return start, stop, exit
}

func capturedVariantSettings(base settings, seed string, jobIndex int, motionVariancePct float64) settings {
	resolved := cloneSettings(base)
	if jobIndex == 0 {
		return resolved
	}
	strength := motionVariancePct / maximumMotionVariancePct
	resolved.targetSpeedMPS = maximumTargetSpeedMPS
	if jobIndex > 1 {
		resolved.targetSpeedMPS = base.targetSpeedMPS + variantUnit(seed, "target-speed")*(maximumTargetSpeedMPS-base.targetSpeedMPS)
	}
	resolved.brakingDecelerationMPS2 = base.brakingDecelerationMPS2 +
		variantUnit(seed, "braking-deceleration")*(maximumAutoBrakingDecelerationMPS2-base.brakingDecelerationMPS2)*strength
	resolved.releaseAccelerationMPS2 = base.releaseAccelerationMPS2 +
		variantUnit(seed, "release-acceleration")*(maximumAutoReleaseAccelerationMPS2-base.releaseAccelerationMPS2)*strength
	weatherPool := []string{"CLEAR", "EXTRASUNNY", "CLOUDS", "OVERCAST", "RAIN", "FOGGY", "SMOG", "THUNDER"}
	resolved.weather = weatherPool[variantIndex(seed, "weather", len(weatherPool))]
	resolved.time = TimeOfDay{
		Hour:   variantIndex(seed, "hour", 24),
		Minute: variantIndex(seed, "minute", 4) * 15,
	}
	colorPool := []RGBColor{
		{R: 26, G: 86, B: 219}, {R: 230, G: 230, B: 230}, {R: 35, G: 35, B: 40},
		{R: 180, G: 35, B: 35}, {R: 35, G: 145, B: 80}, {R: 210, G: 150, B: 30},
	}
	color := colorPool[variantIndex(seed, "vehicle-color", len(colorPool))]
	resolved.vehicle.Color = &color
	return resolved
}

func validateCapturedRoute(label string, start, stop, exit Pose) error {
	startRelative := poseRelativeTo(start, stop)
	exitRelative := poseRelativeTo(exit, stop)
	if startRelative.longitudinal > -minimumStartDistanceM {
		return invalid("%s.startPose must be at least %.1fm before egoStopPose", label, minimumStartDistanceM)
	}
	if exitRelative.longitudinal < minimumExitDistanceM {
		return invalid("%s.exitPose must be at least %.1fm beyond egoStopPose", label, minimumExitDistanceM)
	}
	if planarDistance(start, stop) > maximumStartDistanceM {
		return invalid("%s start-to-stop distance exceeds %.1fm", label, maximumStartDistanceM)
	}
	if planarDistance(stop, exit) > maximumExitDistanceM {
		return invalid("%s stop-to-end distance exceeds %.1fm", label, maximumExitDistanceM)
	}
	if math.Abs(startRelative.lateral) > 8 || math.Abs(exitRelative.lateral) > 8 {
		return invalid("%s captured points must stay within 8m of one approach lane", label)
	}
	return nil
}

func validateCapturedBaseRoute(label string, start, stop, exit Pose) error {
	if err := validateCapturedRoute(label, start, stop, exit); err != nil {
		return err
	}
	startRelative := poseRelativeTo(start, stop)
	exitRelative := poseRelativeTo(exit, stop)
	if startRelative.longitudinal > -minimumCapturedStartM {
		return invalid("%s.startPose must be at least %.1fm before egoStopPose so the approach clip has temporal context", label, minimumCapturedStartM)
	}
	if exitRelative.longitudinal < minimumCapturedExitM {
		return invalid("%s.exitPose must be at least %.1fm beyond egoStopPose so the release clip has temporal context", label, minimumCapturedExitM)
	}
	return nil
}

type relativePose struct {
	longitudinal float64
	lateral      float64
}

func poseRelativeTo(pose, origin Pose) relativePose {
	forward, right := headingBasis(origin.Heading)
	deltaX, deltaY := pose.X-origin.X, pose.Y-origin.Y
	return relativePose{
		longitudinal: deltaX*forward[0] + deltaY*forward[1],
		lateral:      deltaX*right[0] + deltaY*right[1],
	}
}

func scalePoseFromAnchor(pose, anchor Pose, scale float64) Pose {
	pose.X = anchor.X + (pose.X-anchor.X)*scale
	pose.Y = anchor.Y + (pose.Y-anchor.Y)*scale
	pose.Z = anchor.Z + (pose.Z-anchor.Z)*scale
	return pose
}

func poseAtApproachDistance(pose, anchor Pose, distanceM float64) Pose {
	currentDistanceM := planarDistance(pose, anchor)
	if currentDistanceM <= 0 {
		panic("stop-sign variant invariant violated: captured start and stop poses must be distinct")
	}
	relative := poseRelativeTo(pose, anchor)
	if distanceM <= math.Abs(relative.lateral) {
		panic("stop-sign variant invariant violated: generated start distance must exceed its lane offset")
	}
	forward, right := headingBasis(anchor.Heading)
	resolved := translatePose(anchor, forward, right, -math.Sqrt(distanceM*distanceM-relative.lateral*relative.lateral), relative.lateral)
	resolved.Z = anchor.Z + (pose.Z-anchor.Z)*(distanceM/currentDistanceM)
	resolved.Heading = pose.Heading
	return resolved
}

func translatePose(pose Pose, forward, right [2]float64, longitudinal, lateral float64) Pose {
	pose.X += forward[0]*longitudinal + right[0]*lateral
	pose.Y += forward[1]*longitudinal + right[1]*lateral
	return pose
}

func headingBasis(heading float64) ([2]float64, [2]float64) {
	radians := heading * math.Pi / 180
	return [2]float64{-math.Sin(radians), math.Cos(radians)}, [2]float64{math.Cos(radians), math.Sin(radians)}
}

func planarDistance(first, second Pose) float64 {
	return math.Hypot(first.X-second.X, first.Y-second.Y)
}

func normalizedHeading(value float64) float64 {
	return math.Mod(math.Mod(value, 360)+360, 360)
}

func centeredVariant(seed, field string) float64 {
	return variantUnit(seed, field)*2 - 1
}

func variantIndex(seed, field string, count int) int {
	if count < 1 {
		panic("stop-sign variant invariant violated: variant choice count must be positive")
	}
	index := int(math.Floor(variantUnit(seed, field) * float64(count)))
	if index >= count {
		return count - 1
	}
	return index
}

func variantUnit(seed, field string) float64 {
	digest := sha256.Sum256([]byte(seed + ":" + field))
	return float64(binary.BigEndian.Uint64(digest[:8])) / float64(^uint64(0))
}

// ValidateExpandedJob protects command consumers from manually constructed
// jobs that bypass Plan expansion or carry contradictory derived geometry.
func ValidateExpandedJob(job Job) error {
	jobID := strings.TrimSpace(job.ID)
	entryID := strings.TrimSpace(job.EntryID)
	variationID := strings.TrimSpace(job.VariationID)
	if jobID == "" || entryID == "" || variationID == "" || strings.TrimSpace(job.Seed) == "" {
		return invalid("expanded job requires id, entryId, variationId, and seed")
	}
	if jobID != entryID+":"+variationID {
		return invalid("expanded job id %q must equal entryId:variationId", jobID)
	}
	for label, pose := range map[string]Pose{
		"signPose": job.SignPose, "stopLinePose": job.StopLinePose,
		"egoStopPose": job.EgoStopPose, "startPose": job.StartPose, "exitPose": job.ExitPose,
	} {
		if err := validatePose("expanded job "+label, pose); err != nil {
			return err
		}
	}
	if err := validateSignPose("expanded job signPose", job.SignPose); err != nil {
		return err
	}
	resolved := settings{
		stopDistanceM:           job.StopDistanceM,
		egoCenterOffsetM:        job.EgoCenterOffsetM,
		startDistanceM:          job.StartDistanceM,
		exitDistanceM:           job.ExitDistanceM,
		targetSpeedMPS:          job.TargetSpeedMPS,
		brakingDecelerationMPS2: job.BrakingDecelerationMPS2,
		releaseAccelerationMPS2: job.ReleaseAccelerationMPS2,
		stopConfirmationMS:      job.StopConfirmationMS,
		attemptCount:            job.AttemptCount,
		weather:                 job.Weather,
		time:                    job.Time,
		vehicle:                 cloneVehicle(job.Vehicle),
	}
	if err := validateSettings(resolved, "expanded job"); err != nil {
		return err
	}
	if err := ValidateVariationProfile(job.VariationProfile); err != nil {
		return err
	}
	if job.Weather != strings.ToUpper(strings.TrimSpace(job.Weather)) {
		return invalid("expanded job weather must be canonical uppercase")
	}
	if job.Vehicle.Model != strings.ToLower(strings.TrimSpace(job.Vehicle.Model)) {
		return invalid("expanded job vehicle.model must be canonical lowercase")
	}
	if strings.TrimSpace(job.CatalogID) != "" || job.CatalogPosition != nil {
		if catalogID := strings.TrimSpace(job.CatalogID); catalogID == "" || len(catalogID) > 120 {
			return invalid("expanded job catalogId must contain between 1 and 120 characters")
		}
		if err := validateWorldPosition("expanded job catalogPosition", job.CatalogPosition); err != nil {
			return err
		}
		if err := validateCapturedRoute("expanded job", job.StartPose, job.EgoStopPose, job.ExitPose); err != nil {
			return err
		}
		if err := validateRollingApproachDistance("expanded job", job.StartDistanceM, job.TargetSpeedMPS, job.BrakingDecelerationMPS2); err != nil {
			return err
		}
		expectedStopLine := poseAhead(job.EgoStopPose, job.EgoCenterOffsetM)
		expectedSign := poseAhead(expectedStopLine, job.StopDistanceM)
		if !posesNear(job.StopLinePose, expectedStopLine, 1e-6) {
			return invalid("expanded job stopLinePose contradicts captured ego stop pose")
		}
		if !posesNear(job.SignPose, expectedSign, 1e-6) {
			return invalid("expanded job signPose contradicts captured stop geometry")
		}
		if math.Abs(job.StartDistanceM-planarDistance(job.StartPose, job.EgoStopPose)) > 1e-6 {
			return invalid("expanded job startDistanceM contradicts captured start pose")
		}
		if math.Abs(job.ExitDistanceM-planarDistance(job.EgoStopPose, job.ExitPose)) > 1e-6 {
			return invalid("expanded job exitDistanceM contradicts captured end pose")
		}
		return nil
	}
	expectedStopLine := poseBehind(job.SignPose, job.StopDistanceM)
	expectedEgoStop := poseBehind(expectedStopLine, job.EgoCenterOffsetM)
	expectedStart := poseBehind(expectedEgoStop, job.StartDistanceM)
	expectedExit := poseAhead(job.SignPose, job.ExitDistanceM)
	for label, pair := range map[string][2]Pose{
		"stopLinePose": {job.StopLinePose, expectedStopLine},
		"egoStopPose":  {job.EgoStopPose, expectedEgoStop},
		"startPose":    {job.StartPose, expectedStart},
		"exitPose":     {job.ExitPose, expectedExit},
	} {
		if !posesNear(pair[0], pair[1], 1e-6) {
			return invalid("expanded job %s contradicts sign-relative distances", label)
		}
	}
	return nil
}

func defaultSettings() settings {
	return settings{
		stopDistanceM:           DefaultStopDistanceM,
		egoCenterOffsetM:        DefaultEgoCenterOffsetM,
		startDistanceM:          DefaultStartDistanceM,
		exitDistanceM:           DefaultExitDistanceM,
		targetSpeedMPS:          DefaultTargetSpeedMPS,
		brakingDecelerationMPS2: DefaultBrakingDecelerationMPS2,
		releaseAccelerationMPS2: DefaultReleaseAccelerationMPS2,
		stopConfirmationMS:      DefaultStopConfirmationMS,
		attemptCount:            DefaultAttemptCount,
		weather:                 DefaultWeather,
		time:                    TimeOfDay{Hour: DefaultHour, Minute: DefaultMinute},
	}
}

func cloneSettings(source settings) settings {
	clone := source
	clone.vehicle = cloneVehicle(source.vehicle)
	return clone
}

func applyEntrySettings(destination *settings, entry Entry, label string) error {
	return applySettings(destination, entry.StopDistanceM, entry.EgoCenterOffsetM, entry.StartDistanceM, entry.ExitDistanceM, entry.TargetSpeedMPS,
		entry.BrakingDecelerationMPS2, entry.ReleaseAccelerationMPS2,
		entry.StopConfirmationMS, entry.AttemptCount, entry.Weather, entry.Time, entry.Vehicle, label)
}

func applyVariationSettings(destination *settings, variation Variation, label string) error {
	return applySettings(destination, variation.StopDistanceM, variation.EgoCenterOffsetM, variation.StartDistanceM, variation.ExitDistanceM, variation.TargetSpeedMPS,
		variation.BrakingDecelerationMPS2, variation.ReleaseAccelerationMPS2,
		variation.StopConfirmationMS, variation.AttemptCount, variation.Weather, variation.Time, variation.Vehicle, label)
}

func applySettings(
	destination *settings,
	stopDistanceM, egoCenterOffsetM, startDistanceM, exitDistanceM, targetSpeedMPS *float64,
	brakingDecelerationMPS2, releaseAccelerationMPS2 *float64,
	stopConfirmationMS, attemptCount *int,
	weather *string,
	timeOfDay *TimeOfDay,
	vehicle *VehicleVariant,
	label string,
) error {
	if stopDistanceM != nil {
		destination.stopDistanceM = *stopDistanceM
	}
	if egoCenterOffsetM != nil {
		destination.egoCenterOffsetM = *egoCenterOffsetM
	}
	if startDistanceM != nil {
		destination.startDistanceM = *startDistanceM
	}
	if exitDistanceM != nil {
		destination.exitDistanceM = *exitDistanceM
	}
	if targetSpeedMPS != nil {
		destination.targetSpeedMPS = *targetSpeedMPS
	}
	if brakingDecelerationMPS2 != nil {
		destination.brakingDecelerationMPS2 = *brakingDecelerationMPS2
	}
	if releaseAccelerationMPS2 != nil {
		destination.releaseAccelerationMPS2 = *releaseAccelerationMPS2
	}
	if stopConfirmationMS != nil {
		destination.stopConfirmationMS = *stopConfirmationMS
	}
	if attemptCount != nil {
		destination.attemptCount = *attemptCount
	}
	if weather != nil {
		destination.weather = strings.ToUpper(strings.TrimSpace(*weather))
	}
	if timeOfDay != nil {
		destination.time = *timeOfDay
	}
	if vehicle != nil {
		destination.vehicle = mergeVehicle(destination.vehicle, *vehicle)
	}
	return validateSettings(*destination, label)
}

func validateSettings(value settings, label string) error {
	if err := validateRange(label+".stopDistanceM", value.stopDistanceM, minimumStopDistanceM, maximumStopDistanceM); err != nil {
		return err
	}
	if err := validateRange(label+".egoCenterOffsetM", value.egoCenterOffsetM, minimumEgoCenterOffsetM, maximumEgoCenterOffsetM); err != nil {
		return err
	}
	if err := validateRange(label+".startDistanceM", value.startDistanceM, minimumStartDistanceM, maximumStartDistanceM); err != nil {
		return err
	}
	if err := validateRange(label+".exitDistanceM", value.exitDistanceM, minimumExitDistanceM, maximumExitDistanceM); err != nil {
		return err
	}
	if err := validateRange(label+".targetSpeedMps", value.targetSpeedMPS, minimumTargetSpeedMPS, maximumTargetSpeedMPS); err != nil {
		return err
	}
	if err := validateRange(label+".brakingDecelerationMps2", value.brakingDecelerationMPS2, minimumBrakingDecelerationMPS2, maximumBrakingDecelerationMPS2); err != nil {
		return err
	}
	if err := validateRange(label+".releaseAccelerationMps2", value.releaseAccelerationMPS2, minimumReleaseAccelerationMPS2, maximumReleaseAccelerationMPS2); err != nil {
		return err
	}
	if value.stopConfirmationMS < minimumStopConfirmationMS || value.stopConfirmationMS > maximumStopConfirmationMS {
		return invalid("%s.stopConfirmationMs must be between %d and %d", label, minimumStopConfirmationMS, maximumStopConfirmationMS)
	}
	if value.attemptCount < 1 || value.attemptCount > MaximumAttemptCount {
		return invalid("%s.attemptCount must be between 1 and %d", label, MaximumAttemptCount)
	}
	if _, ok := validWeather[value.weather]; !ok {
		return invalid("%s.weather %q is unsupported", label, value.weather)
	}
	if value.time.Hour < 0 || value.time.Hour > 23 || value.time.Minute < 0 || value.time.Minute > 59 {
		return invalid("%s.time must contain hour 0-23 and minute 0-59", label)
	}
	if err := validateVehicle(value.vehicle, label+".vehicle"); err != nil {
		return err
	}
	return nil
}

func validateVehicle(vehicle VehicleVariant, label string) error {
	if len(vehicle.Model) > 64 {
		return invalid("%s.model must contain at most 64 characters", label)
	}
	for _, character := range vehicle.Model {
		if (character < 'a' || character > 'z') && (character < '0' || character > '9') && character != '_' && character != '-' {
			return invalid("%s.model must use only letters, numbers, underscore, or hyphen", label)
		}
	}
	if vehicle.Color == nil {
		return nil
	}
	for channel, value := range map[string]int{"r": vehicle.Color.R, "g": vehicle.Color.G, "b": vehicle.Color.B} {
		if value < 0 || value > 255 {
			return invalid("%s.color.%s must be between 0 and 255", label, channel)
		}
	}
	return nil
}

func validateRange(label string, value, minimum, maximum float64) error {
	if math.IsNaN(value) || math.IsInf(value, 0) || value < minimum || value > maximum {
		return invalid("%s must be between %.1f and %.1f", label, minimum, maximum)
	}
	return nil
}

func validatePose(label string, pose Pose) error {
	for _, value := range []float64{pose.X, pose.Y, pose.Z, pose.Heading} {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return invalid("%s must contain only finite values", label)
		}
	}
	if pose.Heading < 0 || pose.Heading >= 360 {
		return invalid("%s.heading must be in [0, 360)", label)
	}
	return nil
}

func validateWorldPosition(label string, position *WorldPosition) error {
	if position == nil {
		return invalid("%s is required", label)
	}
	for _, value := range []float64{position.X, position.Y, position.Z} {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return invalid("%s must contain only finite values", label)
		}
	}
	if position.X < -10_000 || position.X > 10_000 || position.Y < -10_000 || position.Y > 10_000 ||
		position.Z < -1_000 || position.Z > 3_000 {
		return invalid("%s is outside the supported GTA world bounds", label)
	}
	return nil
}

func validateSignPose(label string, pose Pose) error {
	if err := validatePose(label, pose); err != nil {
		return err
	}
	if pose.X == 0 && pose.Y == 0 && pose.Z == 0 {
		return invalid("%s is still the uncalibrated origin placeholder", label)
	}
	return nil
}

func poseBehind(origin Pose, distanceM float64) Pose {
	headingRadians := origin.Heading * math.Pi / 180
	return Pose{
		X:       origin.X + math.Sin(headingRadians)*distanceM,
		Y:       origin.Y - math.Cos(headingRadians)*distanceM,
		Z:       origin.Z,
		Heading: origin.Heading,
	}
}

func poseAhead(origin Pose, distanceM float64) Pose {
	return poseBehind(origin, -distanceM)
}

func posesNear(first, second Pose, tolerance float64) bool {
	return math.Abs(first.X-second.X) <= tolerance &&
		math.Abs(first.Y-second.Y) <= tolerance &&
		math.Abs(first.Z-second.Z) <= tolerance &&
		math.Abs(first.Heading-second.Heading) <= tolerance
}

func cloneVehicle(source VehicleVariant) VehicleVariant {
	clone := source
	if source.Color != nil {
		color := *source.Color
		clone.Color = &color
	}
	return clone
}

func mergeVehicle(base VehicleVariant, override VehicleVariant) VehicleVariant {
	merged := cloneVehicle(base)
	if model := strings.ToLower(strings.TrimSpace(override.Model)); model != "" {
		merged.Model = model
	}
	if override.Color != nil {
		color := *override.Color
		merged.Color = &color
	}
	return merged
}

func invalid(format string, args ...any) error {
	return fmt.Errorf("%w: %s", ErrInvalidPlan, fmt.Sprintf(format, args...))
}
