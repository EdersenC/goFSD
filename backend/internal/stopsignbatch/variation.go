package stopsignbatch

import (
	"math"
	"strings"
)

const VariationProfileContract = "stop-sign-variation-profile.v2"

var variationDimensions = []string{
	"target_speed",
	"braking_deceleration",
	"release_acceleration",
	"start_distance",
	"exit_distance",
	"stop_pose",
	"stop_heading",
	"start_lane_offset",
	"start_heading",
	"exit_lane_offset",
	"exit_heading",
	"weather",
	"time",
	"vehicle_model",
	"vehicle_color",
}

// VariationProfile measures one resolved job against the first job for its
// scene. CombinationMagnitudePct is the RMS of the normalized dimensions
// above; exact physical and categorical deltas remain available for audits.
type VariationProfile struct {
	Contract                     string   `json:"contract"`
	Baseline                     bool     `json:"baseline"`
	ConfiguredMotionVariancePct  float64  `json:"configuredMotionVariancePct"`
	ChangedDimensions            []string `json:"changedDimensions"`
	ChangeCount                  int      `json:"changeCount"`
	CombinationMagnitudePct      float64  `json:"combinationMagnitudePct"`
	TargetSpeedDeltaMPS          float64  `json:"targetSpeedDeltaMps"`
	TargetSpeedDeltaPct          float64  `json:"targetSpeedDeltaPct"`
	BrakingDecelerationDeltaMPS2 float64  `json:"brakingDecelerationDeltaMps2"`
	ReleaseAccelerationDeltaMPS2 float64  `json:"releaseAccelerationDeltaMps2"`
	StartDistanceDeltaM          float64  `json:"startDistanceDeltaM"`
	ExitDistanceDeltaM           float64  `json:"exitDistanceDeltaM"`
	StopOffsetM                  float64  `json:"stopOffsetM"`
	StopHeadingDeltaDeg          float64  `json:"stopHeadingDeltaDeg"`
	StartLaneOffsetDeltaM        float64  `json:"startLaneOffsetDeltaM"`
	StartHeadingDeltaDeg         float64  `json:"startHeadingDeltaDeg"`
	ExitLaneOffsetDeltaM         float64  `json:"exitLaneOffsetDeltaM"`
	ExitHeadingDeltaDeg          float64  `json:"exitHeadingDeltaDeg"`
	TimeDeltaMinutes             int      `json:"timeDeltaMinutes"`
	WeatherChanged               bool     `json:"weatherChanged"`
	VehicleModelChanged          bool     `json:"vehicleModelChanged"`
	VehicleColorDeltaPct         float64  `json:"vehicleColorDeltaPct"`
}

func applyVariationProfiles(jobs []Job, motionVariancePct float64) {
	if len(jobs) == 0 {
		panic("stop-sign variation invariant violated: a scene must contain at least one expanded job")
	}
	baseline := jobs[0]
	for index := range jobs {
		jobs[index].VariationProfile = buildVariationProfile(jobs[index], baseline, motionVariancePct, index == 0)
	}
}

func buildVariationProfile(job, baseline Job, motionVariancePct float64, baselineJob bool) VariationProfile {
	startLaneOffset := poseRelativeTo(job.StartPose, job.EgoStopPose).lateral
	baselineStartLaneOffset := poseRelativeTo(baseline.StartPose, baseline.EgoStopPose).lateral
	exitLaneOffset := poseRelativeTo(job.ExitPose, job.EgoStopPose).lateral
	baselineExitLaneOffset := poseRelativeTo(baseline.ExitPose, baseline.EgoStopPose).lateral

	profile := VariationProfile{
		Contract:                     VariationProfileContract,
		Baseline:                     baselineJob,
		ConfiguredMotionVariancePct:  roundedVariation(motionVariancePct),
		TargetSpeedDeltaMPS:          roundedVariation(job.TargetSpeedMPS - baseline.TargetSpeedMPS),
		TargetSpeedDeltaPct:          roundedVariation(percentDelta(job.TargetSpeedMPS, baseline.TargetSpeedMPS)),
		BrakingDecelerationDeltaMPS2: roundedVariation(job.BrakingDecelerationMPS2 - baseline.BrakingDecelerationMPS2),
		ReleaseAccelerationDeltaMPS2: roundedVariation(job.ReleaseAccelerationMPS2 - baseline.ReleaseAccelerationMPS2),
		StartDistanceDeltaM:          roundedVariation(job.StartDistanceM - baseline.StartDistanceM),
		ExitDistanceDeltaM:           roundedVariation(job.ExitDistanceM - baseline.ExitDistanceM),
		StopOffsetM:                  roundedVariation(planarDistance(job.EgoStopPose, baseline.EgoStopPose)),
		StopHeadingDeltaDeg:          roundedVariation(headingDelta(job.EgoStopPose.Heading, baseline.EgoStopPose.Heading)),
		StartLaneOffsetDeltaM:        roundedVariation(startLaneOffset - baselineStartLaneOffset),
		StartHeadingDeltaDeg:         roundedVariation(headingDelta(job.StartPose.Heading, baseline.StartPose.Heading)),
		ExitLaneOffsetDeltaM:         roundedVariation(exitLaneOffset - baselineExitLaneOffset),
		ExitHeadingDeltaDeg:          roundedVariation(headingDelta(job.ExitPose.Heading, baseline.ExitPose.Heading)),
		TimeDeltaMinutes:             circularMinuteDelta(job.Time, baseline.Time),
		WeatherChanged:               job.Weather != baseline.Weather,
		VehicleModelChanged:          job.Vehicle.Model != baseline.Vehicle.Model,
		VehicleColorDeltaPct:         roundedVariation(colorDeltaPct(job.Vehicle.Color, baseline.Vehicle.Color)),
	}
	profile.ChangedDimensions = changedVariationDimensions(profile)
	profile.ChangeCount = len(profile.ChangedDimensions)
	profile.CombinationMagnitudePct = roundedVariation(combinationMagnitudePct(profile, baseline))
	return profile
}

func combinationMagnitudePct(profile VariationProfile, baseline Job) float64 {
	exitScale := math.Max(.001, baseline.ExitDistanceM*maximumMotionVariancePct/100)
	stopScale := math.Hypot(.25, .15)
	startLaneScale, exitLaneScale := .5, .5
	stopHeadingScale, startHeadingScale, exitHeadingScale := 2.0, 3.0, 3.0
	speedScale := maximumTargetSpeedMPS - minimumTargetSpeedMPS
	startDistanceScale := maximumStartDistanceM - minimumStartDistanceM
	if strings.TrimSpace(baseline.CatalogID) != "" {
		speedScale = maximumTargetSpeedMPS - minimumAutoTargetSpeedMPS
		startDistanceScale = requiredRollingStartDistanceM(maximumTargetSpeedMPS, DefaultBrakingDecelerationMPS2) -
			requiredRollingStartDistanceM(minimumAutoTargetSpeedMPS, DefaultBrakingDecelerationMPS2)
	}
	factors := []float64{
		normalizedVariation(profile.TargetSpeedDeltaMPS, speedScale),
		normalizedVariation(profile.BrakingDecelerationDeltaMPS2, maximumAutoBrakingDecelerationMPS2-DefaultBrakingDecelerationMPS2),
		normalizedVariation(profile.ReleaseAccelerationDeltaMPS2, maximumAutoReleaseAccelerationMPS2-DefaultReleaseAccelerationMPS2),
		normalizedVariation(profile.StartDistanceDeltaM, startDistanceScale),
		normalizedVariation(profile.ExitDistanceDeltaM, exitScale),
		normalizedVariation(profile.StopOffsetM, stopScale),
		normalizedVariation(profile.StopHeadingDeltaDeg, stopHeadingScale),
		normalizedVariation(profile.StartLaneOffsetDeltaM, startLaneScale),
		normalizedVariation(profile.StartHeadingDeltaDeg, startHeadingScale),
		normalizedVariation(profile.ExitLaneOffsetDeltaM, exitLaneScale),
		normalizedVariation(profile.ExitHeadingDeltaDeg, exitHeadingScale),
		boolVariation(profile.WeatherChanged),
		normalizedVariation(float64(profile.TimeDeltaMinutes), 720),
		boolVariation(profile.VehicleModelChanged),
		normalizedVariation(profile.VehicleColorDeltaPct, 100),
	}
	sumSquares := 0.0
	for _, factor := range factors {
		sumSquares += factor * factor
	}
	return math.Sqrt(sumSquares/float64(len(factors))) * 100
}

func changedVariationDimensions(profile VariationProfile) []string {
	changed := make([]string, 0, len(variationDimensions))
	values := []bool{
		nonZeroVariation(profile.TargetSpeedDeltaMPS),
		nonZeroVariation(profile.BrakingDecelerationDeltaMPS2),
		nonZeroVariation(profile.ReleaseAccelerationDeltaMPS2),
		nonZeroVariation(profile.StartDistanceDeltaM),
		nonZeroVariation(profile.ExitDistanceDeltaM),
		nonZeroVariation(profile.StopOffsetM),
		nonZeroVariation(profile.StopHeadingDeltaDeg),
		nonZeroVariation(profile.StartLaneOffsetDeltaM),
		nonZeroVariation(profile.StartHeadingDeltaDeg),
		nonZeroVariation(profile.ExitLaneOffsetDeltaM),
		nonZeroVariation(profile.ExitHeadingDeltaDeg),
		profile.WeatherChanged,
		profile.TimeDeltaMinutes > 0,
		profile.VehicleModelChanged,
		nonZeroVariation(profile.VehicleColorDeltaPct),
	}
	for index, dimension := range variationDimensions {
		if values[index] {
			changed = append(changed, dimension)
		}
	}
	return changed
}

func ValidateVariationProfile(profile VariationProfile) error {
	if profile.Contract != VariationProfileContract {
		return invalid("variationProfile.contract must equal %q", VariationProfileContract)
	}
	if profile.ConfiguredMotionVariancePct < 0 || profile.ConfiguredMotionVariancePct > maximumMotionVariancePct ||
		math.IsNaN(profile.ConfiguredMotionVariancePct) || math.IsInf(profile.ConfiguredMotionVariancePct, 0) {
		return invalid("variationProfile.configuredMotionVariancePct must be between 0 and %.0f", maximumMotionVariancePct)
	}
	if profile.CombinationMagnitudePct < 0 || profile.CombinationMagnitudePct > 100 ||
		math.IsNaN(profile.CombinationMagnitudePct) || math.IsInf(profile.CombinationMagnitudePct, 0) {
		return invalid("variationProfile.combinationMagnitudePct must be between 0 and 100")
	}
	if profile.ChangeCount != len(profile.ChangedDimensions) {
		return invalid("variationProfile.changeCount must match changedDimensions")
	}
	if err := validateChangedDimensions(profile); err != nil {
		return err
	}
	if profile.Baseline && (profile.ChangeCount != 0 || nonZeroVariation(profile.CombinationMagnitudePct)) {
		return invalid("baseline variationProfile must have zero changes and zero magnitude")
	}
	return nil
}

func validateChangedDimensions(profile VariationProfile) error {
	expected := changedVariationDimensions(profile)
	if len(expected) != len(profile.ChangedDimensions) {
		return invalid("variationProfile.changedDimensions contradicts its exact deltas")
	}
	for index := range expected {
		if expected[index] != profile.ChangedDimensions[index] {
			return invalid("variationProfile.changedDimensions must use canonical dimension order")
		}
	}
	for _, value := range []float64{
		profile.TargetSpeedDeltaMPS, profile.TargetSpeedDeltaPct,
		profile.BrakingDecelerationDeltaMPS2, profile.ReleaseAccelerationDeltaMPS2,
		profile.StartDistanceDeltaM,
		profile.ExitDistanceDeltaM, profile.StopOffsetM, profile.StopHeadingDeltaDeg,
		profile.StartLaneOffsetDeltaM, profile.StartHeadingDeltaDeg, profile.ExitLaneOffsetDeltaM,
		profile.ExitHeadingDeltaDeg, profile.VehicleColorDeltaPct,
	} {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return invalid("variationProfile exact deltas must be finite")
		}
	}
	if profile.StopOffsetM < 0 || profile.StopHeadingDeltaDeg < 0 || profile.StopHeadingDeltaDeg > 180 ||
		profile.StartHeadingDeltaDeg < 0 || profile.StartHeadingDeltaDeg > 180 ||
		profile.ExitHeadingDeltaDeg < 0 || profile.ExitHeadingDeltaDeg > 180 ||
		profile.TimeDeltaMinutes < 0 || profile.TimeDeltaMinutes > 720 ||
		profile.VehicleColorDeltaPct < 0 || profile.VehicleColorDeltaPct > 100 {
		return invalid("variationProfile exact deltas are outside supported bounds")
	}
	return nil
}

// CloneVariationProfile copies the profile and its changed-dimension slice.
func CloneVariationProfile(source VariationProfile) VariationProfile {
	clone := source
	clone.ChangedDimensions = make([]string, len(source.ChangedDimensions))
	copy(clone.ChangedDimensions, source.ChangedDimensions)
	return clone
}

func percentDelta(value, baseline float64) float64 {
	if baseline == 0 {
		return 0
	}
	return (value - baseline) / baseline * 100
}

func circularMinuteDelta(value, baseline TimeOfDay) int {
	delta := int(math.Abs(float64(value.Hour*60 + value.Minute - baseline.Hour*60 - baseline.Minute)))
	return min(delta, 24*60-delta)
}

func colorDeltaPct(value, baseline *RGBColor) float64 {
	if value == nil && baseline == nil {
		return 0
	}
	if value == nil || baseline == nil {
		return 100
	}
	distance := math.Sqrt(
		math.Pow(float64(value.R-baseline.R), 2) +
			math.Pow(float64(value.G-baseline.G), 2) +
			math.Pow(float64(value.B-baseline.B), 2),
	)
	return distance / math.Sqrt(3*255*255) * 100
}

func headingDelta(value, baseline float64) float64 {
	delta := math.Abs(normalizedHeading(value) - normalizedHeading(baseline))
	return math.Min(delta, 360-delta)
}

func normalizedVariation(value, scale float64) float64 {
	if scale <= 0 {
		return 0
	}
	return math.Min(1, math.Abs(value)/scale)
}

func boolVariation(value bool) float64 {
	if value {
		return 1
	}
	return 0
}

func nonZeroVariation(value float64) bool {
	return math.Abs(value) > 1e-6
}

func roundedVariation(value float64) float64 {
	if math.Abs(value) < 5e-7 {
		return 0
	}
	return math.Round(value*1_000_000) / 1_000_000
}
