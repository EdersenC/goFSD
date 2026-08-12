package stopsignbatch

import (
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"
)

func TestExpandDerivesOrderedStopAndStartPoses(t *testing.T) {
	plan := Plan{
		ID:   "  signs-a  ",
		Seed: "  fresh-1  ",
		Entries: []Entry{
			{
				ID:             "northbound",
				SignPose:       Pose{X: 100, Y: 200, Z: 8, Heading: 0},
				StopDistanceM:  float64Ptr(4),
				StartDistanceM: float64Ptr(30),
				AttemptCount:   intPtr(3),
				Variations: []Variation{
					{ID: "clear"},
					{ID: "rain", Weather: stringPtr(" rain "), TargetSpeedMPS: float64Ptr(7.5), ExitDistanceM: float64Ptr(20)},
				},
			},
			{
				ID:             "eastbound",
				SignPose:       Pose{X: 50, Y: -10, Z: 2, Heading: 90},
				StopDistanceM:  float64Ptr(2),
				StartDistanceM: float64Ptr(20),
			},
		},
	}

	first, err := Expand(plan)
	if err != nil {
		t.Fatalf("Expand: %v", err)
	}
	second, err := Expand(plan)
	if err != nil {
		t.Fatalf("Expand second time: %v", err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("expansion is not deterministic:\nfirst=%+v\nsecond=%+v", first, second)
	}
	if got, want := []string{first[0].ID, first[1].ID, first[2].ID}, []string{"northbound:clear", "northbound:rain", "eastbound:base"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("unexpected job order: got=%v want=%v", got, want)
	}
	assertPoseNear(t, first[0].StopLinePose, Pose{X: 100, Y: 196, Z: 8, Heading: 0})
	assertPoseNear(t, first[0].EgoStopPose, Pose{X: 100, Y: 193.5, Z: 8, Heading: 0})
	assertPoseNear(t, first[0].StartPose, Pose{X: 100, Y: 163.5, Z: 8, Heading: 0})
	assertPoseNear(t, first[0].ExitPose, Pose{X: 100, Y: 208, Z: 8, Heading: 0})
	assertPoseNear(t, first[1].ExitPose, Pose{X: 100, Y: 220, Z: 8, Heading: 0})
	assertPoseNear(t, first[2].StopLinePose, Pose{X: 52, Y: -10, Z: 2, Heading: 90})
	assertPoseNear(t, first[2].EgoStopPose, Pose{X: 54.5, Y: -10, Z: 2, Heading: 90})
	assertPoseNear(t, first[2].StartPose, Pose{X: 74.5, Y: -10, Z: 2, Heading: 90})
	assertPoseNear(t, first[2].ExitPose, Pose{X: 42, Y: -10, Z: 2, Heading: 90})
	if first[1].Weather != "RAIN" || first[1].TargetSpeedMPS != 7.5 || first[1].ExitDistanceM != 20 || first[1].AttemptCount != 3 {
		t.Fatalf("variation did not inherit and override settings: %+v", first[1])
	}
	if first[1].Seed != "fresh-1:northbound:rain" {
		t.Fatalf("unexpected deterministic seed: %q", first[1].Seed)
	}
}

func TestExpandUsesDocumentedDefaults(t *testing.T) {
	jobs, err := Expand(Plan{
		ID:   "defaults",
		Seed: "seed",
		Entries: []Entry{{
			ID:       "sign",
			SignPose: Pose{X: 10, Y: 20, Z: 3, Heading: 180},
		}},
	})
	if err != nil {
		t.Fatalf("Expand: %v", err)
	}
	job := jobs[0]
	if job.StopDistanceM != DefaultStopDistanceM || job.EgoCenterOffsetM != DefaultEgoCenterOffsetM || job.StartDistanceM != DefaultStartDistanceM || job.ExitDistanceM != DefaultExitDistanceM ||
		job.TargetSpeedMPS != DefaultTargetSpeedMPS || job.StopConfirmationMS != DefaultStopConfirmationMS ||
		job.AttemptCount != DefaultAttemptCount || job.Weather != DefaultWeather ||
		job.Time != (TimeOfDay{Hour: DefaultHour, Minute: DefaultMinute}) {
		t.Fatalf("unexpected defaults: %+v", job)
	}
}

func TestExpandCapturedSceneBuildsDeterministicBoundedVariants(t *testing.T) {
	start := Pose{X: 0, Y: -40, Z: 30, Heading: 0}
	stop := Pose{X: 0, Y: 0, Z: 30, Heading: 0}
	exit := Pose{X: 0, Y: 12, Z: 30, Heading: 0}
	plan := Plan{
		ID: "captured-scenes", Seed: "repeatable-seed",
		Entries: []Entry{{
			ID:                 "gta-v-sign-0023",
			CatalogID:          "gta-v-sign-0023",
			CatalogPosition:    &WorldPosition{X: -1830.7697, Y: 3206.3914, Z: 31.846758},
			StartPose:          &start,
			EgoStopPose:        &stop,
			ExitPose:           &exit,
			AutoVariations:     &AutoVariationSpec{Count: 50, MotionVariancePct: 20},
			TargetSpeedMPS:     float64Ptr(5),
			StopConfirmationMS: intPtr(250),
			AttemptCount:       intPtr(1),
		}},
	}

	first, err := Expand(plan)
	if err != nil {
		t.Fatalf("Expand captured scene: %v", err)
	}
	second, err := Expand(plan)
	if err != nil {
		t.Fatalf("Expand captured scene again: %v", err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatal("captured scene expansion changed for the same seed")
	}
	if len(first) != 50 {
		t.Fatalf("expected 50 generated jobs, got=%d", len(first))
	}
	baseline := first[0]
	if baseline.VariationID != "auto-001" || baseline.CatalogID != "gta-v-sign-0023" ||
		baseline.CatalogPosition == nil || baseline.StartPose != start || baseline.EgoStopPose != stop || baseline.ExitPose != exit {
		t.Fatalf("baseline did not preserve captured scene: %+v", baseline)
	}
	for index, job := range first {
		if err := ValidateExpandedJob(job); err != nil {
			t.Fatalf("generated job %d is invalid: %v", index, err)
		}
		if job.StartDistanceM < 31.5 || job.StartDistanceM > 48.5 {
			t.Fatalf("generated start distance escaped 20%% bound: %+v", job)
		}
		if job.ExitDistanceM < 9.4 || job.ExitDistanceM > 14.6 {
			t.Fatalf("generated end distance escaped 20%% bound: %+v", job)
		}
		if job.TargetSpeedMPS < 4 || job.TargetSpeedMPS > 6 {
			t.Fatalf("generated speed escaped 20%% bound: %+v", job)
		}
		if planarDistance(job.EgoStopPose, stop) > .35 {
			t.Fatalf("stop target jitter is too large: %+v", job.EgoStopPose)
		}
	}
	if first[1].Weather == baseline.Weather && first[1].Time == baseline.Time && reflect.DeepEqual(first[1].Vehicle.Color, baseline.Vehicle.Color) {
		t.Fatalf("generated conditions did not vary: %+v", first[1])
	}
}

func TestExpandCapturedSceneRejectsIncompleteOrBackwardsRoutes(t *testing.T) {
	start := Pose{X: 0, Y: -40, Z: 30, Heading: 0}
	stop := Pose{X: 0, Y: 0, Z: 30, Heading: 0}
	exit := Pose{X: 0, Y: -12, Z: 30, Heading: 0}
	plan := Plan{ID: "captured", Seed: "seed", Entries: []Entry{{
		ID: "sign", CatalogID: "sign", CatalogPosition: &WorldPosition{X: 1, Y: 2, Z: 3},
		StartPose: &start, EgoStopPose: &stop, ExitPose: &exit,
		AutoVariations: &AutoVariationSpec{Count: 50, MotionVariancePct: 20},
	}}}
	if _, err := Expand(plan); !errors.Is(err, ErrInvalidPlan) || !strings.Contains(err.Error(), "exitPose") {
		t.Fatalf("expected backwards end rejection, got=%v", err)
	}
	plan.Entries[0].ExitPose = nil
	if _, err := Expand(plan); !errors.Is(err, ErrInvalidPlan) || !strings.Contains(err.Error(), "requires captured") {
		t.Fatalf("expected incomplete anchor rejection, got=%v", err)
	}
}

func TestExpandCapturedSceneRequiresRoomForStableRollingCaptureAtHighSpeed(t *testing.T) {
	start := Pose{X: 0, Y: -100, Z: 30, Heading: 0}
	stop := Pose{X: 0, Y: 0, Z: 30, Heading: 0}
	exit := Pose{X: 0, Y: 12, Z: 30, Heading: 0}
	targetSpeed := 15.0
	plan := Plan{ID: "high-speed", Seed: "seed", Entries: []Entry{{
		ID: "sign", CatalogID: "gta-v-sign-0001", CatalogPosition: &WorldPosition{X: 1, Y: 2, Z: 3},
		StartPose: &start, EgoStopPose: &stop, ExitPose: &exit,
		AutoVariations: &AutoVariationSpec{Count: 1}, TargetSpeedMPS: &targetSpeed,
	}}}
	if _, err := Expand(plan); err == nil || !strings.Contains(err.Error(), "record stable cruise") {
		t.Fatalf("short high-speed approach must be rejected with actionable guidance: %v", err)
	}

	start.Y = -140
	jobs, err := Expand(plan)
	if err != nil {
		t.Fatalf("long high-speed approach should be accepted: %v", err)
	}
	if len(jobs) != 1 || jobs[0].TargetSpeedMPS != 15 {
		t.Fatalf("high-speed job did not preserve the selected target: %+v", jobs)
	}
}

func TestExpandMergesVehicleVariationFields(t *testing.T) {
	jobs, err := Expand(Plan{
		ID:   "vehicles",
		Seed: "seed",
		Entries: []Entry{{
			ID:       "sign",
			SignPose: Pose{X: 10, Y: 20, Z: 3, Heading: 0},
			Vehicle: &VehicleVariant{
				Model: " SULTAN ",
				Color: &RGBColor{R: 10, G: 20, B: 30},
			},
			Variations: []Variation{
				{ID: "blue", Vehicle: &VehicleVariant{Color: &RGBColor{R: 0, G: 40, B: 255}}},
				{ID: "blista", Vehicle: &VehicleVariant{Model: "Blista"}},
			},
		}},
	})
	if err != nil {
		t.Fatalf("Expand: %v", err)
	}
	if jobs[0].Vehicle.Model != "sultan" || jobs[0].Vehicle.Color == nil || jobs[0].Vehicle.Color.B != 255 {
		t.Fatalf("color variation lost base model: %+v", jobs[0].Vehicle)
	}
	if jobs[1].Vehicle.Model != "blista" || jobs[1].Vehicle.Color == nil || jobs[1].Vehicle.Color.R != 10 {
		t.Fatalf("model variation lost base color: %+v", jobs[1].Vehicle)
	}
}

func TestExpandRejectsInvalidPlans(t *testing.T) {
	valid := Plan{
		ID:   "signs",
		Seed: "seed",
		Entries: []Entry{{
			ID:       "sign-a",
			SignPose: Pose{X: 1, Y: 2, Z: 3, Heading: 45},
		}},
	}

	tests := []struct {
		name string
		edit func(*Plan)
		want string
	}{
		{name: "missing id", edit: func(plan *Plan) { plan.ID = "" }, want: "id is required"},
		{name: "missing seed", edit: func(plan *Plan) { plan.Seed = "" }, want: "seed is required"},
		{name: "duplicate entry", edit: func(plan *Plan) { plan.Entries = append(plan.Entries, plan.Entries[0]) }, want: "duplicated"},
		{name: "non finite pose", edit: func(plan *Plan) { plan.Entries[0].SignPose.X = math.NaN() }, want: "finite"},
		{name: "heading", edit: func(plan *Plan) { plan.Entries[0].SignPose.Heading = 360 }, want: "heading"},
		{name: "uncalibrated origin", edit: func(plan *Plan) { plan.Entries[0].SignPose = Pose{} }, want: "uncalibrated origin placeholder"},
		{name: "stop distance", edit: func(plan *Plan) { plan.Entries[0].StopDistanceM = float64Ptr(0) }, want: "stopDistanceM"},
		{name: "ego center offset", edit: func(plan *Plan) { plan.Entries[0].EgoCenterOffsetM = float64Ptr(0) }, want: "egoCenterOffsetM"},
		{name: "start distance", edit: func(plan *Plan) { plan.Entries[0].StartDistanceM = float64Ptr(251) }, want: "startDistanceM"},
		{name: "exit distance", edit: func(plan *Plan) { plan.Entries[0].ExitDistanceM = float64Ptr(1) }, want: "exitDistanceM"},
		{name: "target speed", edit: func(plan *Plan) { plan.Entries[0].TargetSpeedMPS = float64Ptr(41) }, want: "targetSpeedMps"},
		{name: "stop confirmation", edit: func(plan *Plan) { plan.Entries[0].StopConfirmationMS = intPtr(50) }, want: "stopConfirmationMs"},
		{name: "attempts", edit: func(plan *Plan) { plan.Entries[0].AttemptCount = intPtr(51) }, want: "attemptCount"},
		{name: "weather", edit: func(plan *Plan) { plan.Entries[0].Weather = stringPtr("tornado") }, want: "weather"},
		{name: "time", edit: func(plan *Plan) { plan.Entries[0].Time = &TimeOfDay{Hour: 24} }, want: "time"},
		{name: "color", edit: func(plan *Plan) { plan.Entries[0].Vehicle = &VehicleVariant{Color: &RGBColor{R: -1}} }, want: "color.r"},
		{name: "model", edit: func(plan *Plan) { plan.Entries[0].Vehicle = &VehicleVariant{Model: "not valid!"} }, want: "model"},
		{name: "duplicate variation", edit: func(plan *Plan) { plan.Entries[0].Variations = []Variation{{ID: "same"}, {ID: "same"}} }, want: "variation id"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			plan := valid
			plan.Entries = append([]Entry(nil), valid.Entries...)
			test.edit(&plan)
			_, err := Expand(plan)
			if !errors.Is(err, ErrInvalidPlan) || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("expected invalid plan containing %q, got=%v", test.want, err)
			}
		})
	}
}

func TestValidateExpandedJobRejectsContradictoryGeometry(t *testing.T) {
	jobs, err := Expand(Plan{
		ID: "validation", Seed: "seed",
		Entries: []Entry{{ID: "sign", SignPose: Pose{X: 10, Y: 20, Z: 2, Heading: 45}}},
	})
	if err != nil {
		t.Fatalf("Expand: %v", err)
	}
	if err := ValidateExpandedJob(jobs[0]); err != nil {
		t.Fatalf("expanded job should validate: %v", err)
	}
	tampered := jobs[0]
	tampered.EgoStopPose.X += 0.25
	if err := ValidateExpandedJob(tampered); !errors.Is(err, ErrInvalidPlan) || !strings.Contains(err.Error(), "egoStopPose") {
		t.Fatalf("expected contradictory ego stop rejection, got=%v", err)
	}
	tampered = jobs[0]
	tampered.ExitPose.Y += 0.25
	if err := ValidateExpandedJob(tampered); !errors.Is(err, ErrInvalidPlan) || !strings.Contains(err.Error(), "exitPose") {
		t.Fatalf("expected contradictory exit rejection, got=%v", err)
	}
}

func assertPoseNear(t *testing.T, got, want Pose) {
	t.Helper()
	const tolerance = 1e-9
	if math.Abs(got.X-want.X) > tolerance || math.Abs(got.Y-want.Y) > tolerance ||
		math.Abs(got.Z-want.Z) > tolerance || math.Abs(got.Heading-want.Heading) > tolerance {
		t.Fatalf("pose mismatch: got=%+v want=%+v", got, want)
	}
}

func float64Ptr(value float64) *float64 { return &value }
func intPtr(value int) *int             { return &value }
func stringPtr(value string) *string    { return &value }
