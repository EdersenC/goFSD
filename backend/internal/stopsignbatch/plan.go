// Package stopsignbatch validates and expands deterministic stop-sign collection plans.
package stopsignbatch

import (
	"errors"
	"fmt"
	"math"
	"strings"
)

const (
	MaximumEntries      = 100
	MaximumExpandedJobs = 100
	MaximumAttemptCount = 50

	DefaultStopDistanceM    = 3.0
	DefaultEgoCenterOffsetM = 2.5
	DefaultStartDistanceM   = 40.0
	DefaultTargetSpeedMPS   = 8.0
	DefaultDwellMS          = 5000
	DefaultAttemptCount     = 1
	DefaultWeather          = "EXTRASUNNY"
	DefaultHour             = 12
	DefaultMinute           = 0

	minimumStopDistanceM    = 0.5
	maximumStopDistanceM    = 15.0
	minimumEgoCenterOffsetM = 0.5
	maximumEgoCenterOffsetM = 8.0
	minimumStartDistanceM   = 5.0
	maximumStartDistanceM   = 250.0
	minimumTargetSpeedMPS   = 0.5
	maximumTargetSpeedMPS   = 8.0
	minimumDwellMS          = 500
	maximumDwellMS          = 30_000
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
	ID               string          `json:"id"`
	StopDistanceM    *float64        `json:"stopDistanceM,omitempty"`
	EgoCenterOffsetM *float64        `json:"egoCenterOffsetM,omitempty"`
	StartDistanceM   *float64        `json:"startDistanceM,omitempty"`
	TargetSpeedMPS   *float64        `json:"targetSpeedMps,omitempty"`
	DwellMS          *int            `json:"dwellMs,omitempty"`
	AttemptCount     *int            `json:"attemptCount,omitempty"`
	Weather          *string         `json:"weather,omitempty"`
	Time             *TimeOfDay      `json:"time,omitempty"`
	Vehicle          *VehicleVariant `json:"vehicle,omitempty"`
}

// Entry defines one ordered stop sign plus optional base collection settings.
// Missing settings use the package defaults exported above.
type Entry struct {
	ID               string          `json:"id"`
	SignPose         Pose            `json:"signPose"`
	StopDistanceM    *float64        `json:"stopDistanceM,omitempty"`
	EgoCenterOffsetM *float64        `json:"egoCenterOffsetM,omitempty"`
	StartDistanceM   *float64        `json:"startDistanceM,omitempty"`
	TargetSpeedMPS   *float64        `json:"targetSpeedMps,omitempty"`
	DwellMS          *int            `json:"dwellMs,omitempty"`
	AttemptCount     *int            `json:"attemptCount,omitempty"`
	Weather          *string         `json:"weather,omitempty"`
	Time             *TimeOfDay      `json:"time,omitempty"`
	Vehicle          *VehicleVariant `json:"vehicle,omitempty"`
	Variations       []Variation     `json:"variations,omitempty"`
}

type Plan struct {
	ID      string  `json:"id"`
	Seed    string  `json:"seed"`
	Entries []Entry `json:"entries"`
}

// Job is a fully explicit collection unit. StopLinePose, EgoStopPose, and
// StartPose are derived from SignPose and cannot diverge from it.
type Job struct {
	ID               string         `json:"id"`
	EntryID          string         `json:"entryId"`
	VariationID      string         `json:"variationId"`
	SignPose         Pose           `json:"signPose"`
	StopLinePose     Pose           `json:"stopLinePose"`
	EgoStopPose      Pose           `json:"egoStopPose"`
	StartPose        Pose           `json:"startPose"`
	StopDistanceM    float64        `json:"stopDistanceM"`
	EgoCenterOffsetM float64        `json:"egoCenterOffsetM"`
	StartDistanceM   float64        `json:"startDistanceM"`
	TargetSpeedMPS   float64        `json:"targetSpeedMps"`
	DwellMS          int            `json:"dwellMs"`
	AttemptCount     int            `json:"attemptCount"`
	Weather          string         `json:"weather"`
	Time             TimeOfDay      `json:"time"`
	Vehicle          VehicleVariant `json:"vehicle"`
	Seed             string         `json:"seed"`
}

type settings struct {
	stopDistanceM    float64
	egoCenterOffsetM float64
	startDistanceM   float64
	targetSpeedMPS   float64
	dwellMS          int
	attemptCount     int
	weather          string
	time             TimeOfDay
	vehicle          VehicleVariant
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
		if err := validatePose(fmt.Sprintf("entries[%d].signPose", entryIndex), entry.SignPose); err != nil {
			return nil, err
		}

		base := defaultSettings()
		if err := applyEntrySettings(&base, entry, fmt.Sprintf("entries[%d]", entryIndex)); err != nil {
			return nil, err
		}
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
			if err := validatePose(label+".stopLinePose", stopLinePose); err != nil {
				return nil, err
			}
			if err := validatePose(label+".egoStopPose", egoStopPose); err != nil {
				return nil, err
			}
			if err := validatePose(label+".startPose", startPose); err != nil {
				return nil, err
			}

			jobID := entryID + ":" + variationID
			if _, exists := jobIDs[jobID]; exists {
				return nil, invalid("expanded job id %q is duplicated; choose unambiguous entry and variation ids", jobID)
			}
			jobIDs[jobID] = struct{}{}
			jobs = append(jobs, Job{
				ID:               jobID,
				EntryID:          entryID,
				VariationID:      variationID,
				SignPose:         entry.SignPose,
				StopLinePose:     stopLinePose,
				EgoStopPose:      egoStopPose,
				StartPose:        startPose,
				StopDistanceM:    resolved.stopDistanceM,
				EgoCenterOffsetM: resolved.egoCenterOffsetM,
				StartDistanceM:   resolved.startDistanceM,
				TargetSpeedMPS:   resolved.targetSpeedMPS,
				DwellMS:          resolved.dwellMS,
				AttemptCount:     resolved.attemptCount,
				Weather:          resolved.weather,
				Time:             resolved.time,
				Vehicle:          resolved.vehicle,
				Seed:             seed + ":" + jobID,
			})
			if len(jobs) > MaximumExpandedJobs {
				return nil, invalid("plan expands to more than %d jobs", MaximumExpandedJobs)
			}
		}
	}

	return jobs, nil
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
		"egoStopPose": job.EgoStopPose, "startPose": job.StartPose,
	} {
		if err := validatePose("expanded job "+label, pose); err != nil {
			return err
		}
	}
	resolved := settings{
		stopDistanceM:    job.StopDistanceM,
		egoCenterOffsetM: job.EgoCenterOffsetM,
		startDistanceM:   job.StartDistanceM,
		targetSpeedMPS:   job.TargetSpeedMPS,
		dwellMS:          job.DwellMS,
		attemptCount:     job.AttemptCount,
		weather:          job.Weather,
		time:             job.Time,
		vehicle:          cloneVehicle(job.Vehicle),
	}
	if err := validateSettings(resolved, "expanded job"); err != nil {
		return err
	}
	if job.Weather != strings.ToUpper(strings.TrimSpace(job.Weather)) {
		return invalid("expanded job weather must be canonical uppercase")
	}
	if job.Vehicle.Model != strings.ToLower(strings.TrimSpace(job.Vehicle.Model)) {
		return invalid("expanded job vehicle.model must be canonical lowercase")
	}
	expectedStopLine := poseBehind(job.SignPose, job.StopDistanceM)
	expectedEgoStop := poseBehind(expectedStopLine, job.EgoCenterOffsetM)
	expectedStart := poseBehind(expectedEgoStop, job.StartDistanceM)
	for label, pair := range map[string][2]Pose{
		"stopLinePose": {job.StopLinePose, expectedStopLine},
		"egoStopPose":  {job.EgoStopPose, expectedEgoStop},
		"startPose":    {job.StartPose, expectedStart},
	} {
		if !posesNear(pair[0], pair[1], 1e-6) {
			return invalid("expanded job %s contradicts sign-relative distances", label)
		}
	}
	return nil
}

func defaultSettings() settings {
	return settings{
		stopDistanceM:    DefaultStopDistanceM,
		egoCenterOffsetM: DefaultEgoCenterOffsetM,
		startDistanceM:   DefaultStartDistanceM,
		targetSpeedMPS:   DefaultTargetSpeedMPS,
		dwellMS:          DefaultDwellMS,
		attemptCount:     DefaultAttemptCount,
		weather:          DefaultWeather,
		time:             TimeOfDay{Hour: DefaultHour, Minute: DefaultMinute},
	}
}

func cloneSettings(source settings) settings {
	clone := source
	clone.vehicle = cloneVehicle(source.vehicle)
	return clone
}

func applyEntrySettings(destination *settings, entry Entry, label string) error {
	return applySettings(destination, entry.StopDistanceM, entry.EgoCenterOffsetM, entry.StartDistanceM, entry.TargetSpeedMPS,
		entry.DwellMS, entry.AttemptCount, entry.Weather, entry.Time, entry.Vehicle, label)
}

func applyVariationSettings(destination *settings, variation Variation, label string) error {
	return applySettings(destination, variation.StopDistanceM, variation.EgoCenterOffsetM, variation.StartDistanceM, variation.TargetSpeedMPS,
		variation.DwellMS, variation.AttemptCount, variation.Weather, variation.Time, variation.Vehicle, label)
}

func applySettings(
	destination *settings,
	stopDistanceM, egoCenterOffsetM, startDistanceM, targetSpeedMPS *float64,
	dwellMS, attemptCount *int,
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
	if targetSpeedMPS != nil {
		destination.targetSpeedMPS = *targetSpeedMPS
	}
	if dwellMS != nil {
		destination.dwellMS = *dwellMS
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
	if err := validateRange(label+".targetSpeedMps", value.targetSpeedMPS, minimumTargetSpeedMPS, maximumTargetSpeedMPS); err != nil {
		return err
	}
	if value.dwellMS < minimumDwellMS || value.dwellMS > maximumDwellMS {
		return invalid("%s.dwellMs must be between %d and %d", label, minimumDwellMS, maximumDwellMS)
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

func poseBehind(origin Pose, distanceM float64) Pose {
	headingRadians := origin.Heading * math.Pi / 180
	return Pose{
		X:       origin.X + math.Sin(headingRadians)*distanceM,
		Y:       origin.Y - math.Cos(headingRadians)*distanceM,
		Z:       origin.Z,
		Heading: origin.Heading,
	}
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
