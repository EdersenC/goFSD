package parkingbatch

import (
	"errors"
	"fmt"
	"math"
	"strings"
)

const (
	MaximumEntries          = 100
	MaximumExpandedJobs     = 100
	MaximumCollectionAmount = 50

	minimumStartDistanceM = 9.5
	maximumStartDistanceM = 15.0
	maximumLateralErrorM  = 0.35
	maximumHeadingError   = 3.0
	maximumVerticalErrorM = 1.0
)

var ErrInvalidPlan = errors.New("invalid parking collection plan")

// Pose is an exact GTA world pose. Heading uses GTA degrees in [0, 360).
type Pose struct {
	X       float64 `json:"x"`
	Y       float64 `json:"y"`
	Z       float64 `json:"z"`
	Heading float64 `json:"heading"`
}

// Variation overrides either base pose. An empty variation list expands to one
// implicit "base" variation.
type Variation struct {
	ID             string          `json:"id"`
	ParkDest       *Pose           `json:"parkDest,omitempty"`
	StartDest      *Pose           `json:"startDest,omitempty"`
	StartVariation *StartVariation `json:"startVariation,omitempty"`
}

// StartVariation applies a small target-relative change to the saved start.
// Positive distance moves farther behind the target; positive lateral moves
// to the target's right; heading is added to the saved GTA heading.
type StartVariation struct {
	DistanceDeltaM  float64 `json:"distanceDeltaM"`
	LateralDeltaM   float64 `json:"lateralDeltaM"`
	HeadingDeltaDeg float64 `json:"headingDeltaDeg"`
}

type Entry struct {
	ID               string      `json:"id"`
	ParkDest         Pose        `json:"parkDest"`
	StartDest        Pose        `json:"startDest"`
	Variations       []Variation `json:"variations,omitempty"`
	CollectionAmount int         `json:"collectionAmount"`
}

type Plan struct {
	ID      string  `json:"id"`
	Seed    string  `json:"seed"`
	Entries []Entry `json:"entries"`
}

type Job struct {
	ID               string `json:"id"`
	EntryID          string `json:"entryId"`
	VariationID      string `json:"variationId"`
	ParkDest         Pose   `json:"parkDest"`
	StartDest        Pose   `json:"startDest"`
	CollectionAmount int    `json:"collectionAmount"`
	Seed             string `json:"seed"`
}

// Expand validates a user plan and deterministically emits jobs in entry order,
// then variation order. It never reads time, randomness, or runtime state.
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
		if entry.CollectionAmount < 1 || entry.CollectionAmount > MaximumCollectionAmount {
			return nil, invalid("entries[%d].collectionAmount must be between 1 and %d", entryIndex, MaximumCollectionAmount)
		}
		if err := validatePose(fmt.Sprintf("entries[%d].parkDest", entryIndex), entry.ParkDest); err != nil {
			return nil, err
		}
		if err := validatePose(fmt.Sprintf("entries[%d].startDest", entryIndex), entry.StartDest); err != nil {
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

			parkDest := entry.ParkDest
			if variation.ParkDest != nil {
				parkDest = *variation.ParkDest
			}
			startDest := entry.StartDest
			if variation.StartDest != nil {
				startDest = *variation.StartDest
			}
			if variation.StartVariation != nil {
				if variation.ParkDest != nil || variation.StartDest != nil {
					return nil, invalid(
						"entries[%d] variation %q cannot mix startVariation with full pose overrides",
						entryIndex,
						variationID,
					)
				}
				var err error
				startDest, err = applyStartVariation(entry.ParkDest, entry.StartDest, *variation.StartVariation)
				if err != nil {
					return nil, invalid("entries[%d] variation %q: %v", entryIndex, variationID, err)
				}
			}
			if err := validatePose(fmt.Sprintf("entries[%d].variations[%d].parkDest", entryIndex, variationIndex), parkDest); err != nil {
				return nil, err
			}
			if err := validatePose(fmt.Sprintf("entries[%d].variations[%d].startDest", entryIndex, variationIndex), startDest); err != nil {
				return nil, err
			}
			if err := validateStraightApproach(parkDest, startDest); err != nil {
				return nil, invalid("entries[%d] variation %q: %v", entryIndex, variationID, err)
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
				ParkDest:         parkDest,
				StartDest:        startDest,
				CollectionAmount: entry.CollectionAmount,
				Seed:             seed + ":" + jobID,
			})
			if len(jobs) > MaximumExpandedJobs {
				return nil, invalid("plan expands to more than %d jobs", MaximumExpandedJobs)
			}
		}
	}

	return jobs, nil
}

func applyStartVariation(parkDest Pose, startDest Pose, variation StartVariation) (Pose, error) {
	values := []float64{variation.DistanceDeltaM, variation.LateralDeltaM, variation.HeadingDeltaDeg}
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return Pose{}, errors.New("startVariation must contain only finite values")
		}
	}
	if math.Abs(variation.HeadingDeltaDeg) > maximumHeadingError {
		return Pose{}, fmt.Errorf(
			"startVariation heading delta must be within %.1f degrees; got %.3f",
			maximumHeadingError,
			variation.HeadingDeltaDeg,
		)
	}

	headingRadians := parkDest.Heading * math.Pi / 180
	forwardX := -math.Sin(headingRadians)
	forwardY := math.Cos(headingRadians)
	rightX := math.Cos(headingRadians)
	rightY := math.Sin(headingRadians)
	deltaX := startDest.X - parkDest.X
	deltaY := startDest.Y - parkDest.Y
	longitudinal := deltaX*forwardX + deltaY*forwardY - variation.DistanceDeltaM
	lateral := deltaX*rightX + deltaY*rightY + variation.LateralDeltaM

	result := startDest
	result.X = parkDest.X + (forwardX * longitudinal) + (rightX * lateral)
	result.Y = parkDest.Y + (forwardY * longitudinal) + (rightY * lateral)
	result.Heading = normalizeHeading(startDest.Heading + variation.HeadingDeltaDeg)
	return result, nil
}

func validatePose(label string, pose Pose) error {
	values := []float64{pose.X, pose.Y, pose.Z, pose.Heading}
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return invalid("%s must contain only finite values", label)
		}
	}
	if pose.Heading < 0 || pose.Heading >= 360 {
		return invalid("%s.heading must be in [0, 360)", label)
	}
	return nil
}

func validateStraightApproach(parkDest Pose, startDest Pose) error {
	headingRadians := parkDest.Heading * math.Pi / 180
	forwardX := -math.Sin(headingRadians)
	forwardY := math.Cos(headingRadians)
	rightX := math.Cos(headingRadians)
	rightY := math.Sin(headingRadians)
	deltaX := startDest.X - parkDest.X
	deltaY := startDest.Y - parkDest.Y
	longitudinal := deltaX*forwardX + deltaY*forwardY
	lateral := deltaX*rightX + deltaY*rightY
	headingError := math.Abs(shortestHeadingDelta(startDest.Heading, parkDest.Heading))

	if longitudinal < -maximumStartDistanceM || longitudinal > -minimumStartDistanceM {
		return fmt.Errorf("startDest must be %.1f-%.1f m behind parkDest; got %.3f m", minimumStartDistanceM, maximumStartDistanceM, -longitudinal)
	}
	if math.Abs(lateral) > maximumLateralErrorM {
		return fmt.Errorf("startDest lateral error must be within %.2f m; got %.3f m", maximumLateralErrorM, lateral)
	}
	if headingError > maximumHeadingError {
		return fmt.Errorf("startDest heading error must be within %.1f degrees; got %.3f", maximumHeadingError, headingError)
	}
	if math.Abs(startDest.Z-parkDest.Z) > maximumVerticalErrorM {
		return fmt.Errorf("startDest vertical error must be within %.1f m; got %.3f m", maximumVerticalErrorM, math.Abs(startDest.Z-parkDest.Z))
	}
	return nil
}

func shortestHeadingDelta(source, target float64) float64 {
	delta := math.Mod(source-target+540, 360) - 180
	return delta
}

func normalizeHeading(heading float64) float64 {
	return math.Mod(math.Mod(heading, 360)+360, 360)
}

func invalid(format string, args ...any) error {
	return fmt.Errorf("%w: %s", ErrInvalidPlan, fmt.Sprintf(format, args...))
}
