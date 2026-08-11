package parkingbatch

import (
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"
)

func TestExpandUsesStableEntryAndVariationOrder(t *testing.T) {
	plan := Plan{
		ID:   "  morning-lot  ",
		Seed: "  return-1  ",
		Entries: []Entry{
			{
				ID:               "bay-a",
				ParkDest:         Pose{X: 10, Y: 20, Z: 3, Heading: 0},
				StartDest:        Pose{X: 10, Y: 9, Z: 3, Heading: 0},
				CollectionAmount: 8,
				Variations: []Variation{
					{ID: "base"},
					{
						ID:        "right-20cm",
						ParkDest:  &Pose{X: 10.2, Y: 20, Z: 3, Heading: 0},
						StartDest: &Pose{X: 10.2, Y: 9, Z: 3, Heading: 0},
					},
				},
			},
			{
				ID:               "bay-b",
				ParkDest:         Pose{X: 50, Y: 50, Z: 3, Heading: 90},
				StartDest:        Pose{X: 61, Y: 50, Z: 3, Heading: 90},
				CollectionAmount: 3,
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
	if len(first) != 3 {
		t.Fatalf("unexpected expanded job count: got=%d want=3", len(first))
	}
	wantIDs := []string{"bay-a:base", "bay-a:right-20cm", "bay-b:base"}
	gotIDs := []string{first[0].ID, first[1].ID, first[2].ID}
	if !reflect.DeepEqual(gotIDs, wantIDs) {
		t.Fatalf("unexpected job order: got=%v want=%v", gotIDs, wantIDs)
	}
	if first[1].Seed != "return-1:bay-a:right-20cm" || first[1].CollectionAmount != 8 {
		t.Fatalf("unexpected expanded variation: %+v", first[1])
	}
	if first[2].VariationID != "base" {
		t.Fatalf("empty variation list must produce an implicit base job: %+v", first[2])
	}
}

func TestExpandRejectsUnsafeOrAmbiguousPlans(t *testing.T) {
	valid := Plan{
		ID:   "lot",
		Seed: "seed",
		Entries: []Entry{{
			ID:               "a",
			ParkDest:         Pose{X: 0, Y: 0, Z: 0, Heading: 0},
			StartDest:        Pose{X: 0, Y: -11, Z: 0, Heading: 0},
			CollectionAmount: 2,
		}},
	}

	tests := []struct {
		name string
		edit func(*Plan)
		want string
	}{
		{name: "missing seed", edit: func(plan *Plan) { plan.Seed = "" }, want: "seed is required"},
		{name: "duplicate entry", edit: func(plan *Plan) { plan.Entries = append(plan.Entries, plan.Entries[0]) }, want: "duplicated"},
		{name: "too close", edit: func(plan *Plan) { plan.Entries[0].StartDest.Y = -5 }, want: "9.5-15.0 m behind"},
		{name: "lateral", edit: func(plan *Plan) { plan.Entries[0].StartDest.X = 0.5 }, want: "lateral error"},
		{name: "heading", edit: func(plan *Plan) { plan.Entries[0].StartDest.Heading = 4 }, want: "heading error"},
		{name: "count", edit: func(plan *Plan) { plan.Entries[0].CollectionAmount = 0 }, want: "collectionAmount"},
		{name: "duplicate variation", edit: func(plan *Plan) {
			plan.Entries[0].Variations = []Variation{{ID: "same"}, {ID: "same"}}
		}, want: "variation id"},
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

func TestExpandAppliesTargetRelativeStartVariations(t *testing.T) {
	plan := Plan{
		ID:   "lot",
		Seed: "morning",
		Entries: []Entry{{
			ID:               "east-facing",
			ParkDest:         Pose{X: 100, Y: 200, Z: 8, Heading: 90},
			StartDest:        Pose{X: 111, Y: 200, Z: 8, Heading: 90},
			CollectionAmount: 4,
			Variations: []Variation{{
				ID: "farther-left-angle",
				StartVariation: &StartVariation{
					DistanceDeltaM:  0.5,
					LateralDeltaM:   -0.2,
					HeadingDeltaDeg: 2,
				},
			}},
		}},
	}

	jobs, err := Expand(plan)
	if err != nil {
		t.Fatalf("Expand: %v", err)
	}
	if len(jobs) != 1 {
		t.Fatalf("unexpected job count: %d", len(jobs))
	}
	got := jobs[0].StartDest
	if math.Abs(got.X-111.5) > 1e-9 || math.Abs(got.Y-199.8) > 1e-9 {
		t.Fatalf("unexpected target-relative start: %+v", got)
	}
	if got.Heading != 92 {
		t.Fatalf("unexpected varied heading: %+v", got)
	}
}

func TestExpandRejectsMixedRelativeAndAbsoluteVariation(t *testing.T) {
	plan := Plan{
		ID:   "lot",
		Seed: "morning",
		Entries: []Entry{{
			ID:               "bay",
			ParkDest:         Pose{X: 0, Y: 0, Z: 0, Heading: 0},
			StartDest:        Pose{X: 0, Y: -11, Z: 0, Heading: 0},
			CollectionAmount: 2,
			Variations: []Variation{{
				ID:             "ambiguous",
				StartDest:      &Pose{X: 0, Y: -11, Z: 0, Heading: 0},
				StartVariation: &StartVariation{DistanceDeltaM: 0.5},
			}},
		}},
	}

	_, err := Expand(plan)
	if !errors.Is(err, ErrInvalidPlan) || !strings.Contains(err.Error(), "cannot mix") {
		t.Fatalf("expected ambiguous variation rejection, got=%v", err)
	}
}
