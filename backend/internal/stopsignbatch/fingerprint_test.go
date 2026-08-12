package stopsignbatch

import "testing"

func TestPlanFingerprintIsDeterministicAndSemantic(t *testing.T) {
	plan := Plan{
		ID:   "  city-stops ",
		Seed: " seed-1 ",
		Entries: []Entry{{
			ID:       "sign-a",
			SignPose: Pose{X: 1, Y: 2, Z: 3, Heading: 90},
			Weather:  stringPtr(" rain "),
			Vehicle:  &VehicleVariant{Model: " SULTAN "},
		}},
	}
	first, err := PlanFingerprint(plan)
	if err != nil {
		t.Fatalf("PlanFingerprint: %v", err)
	}
	second, err := PlanFingerprint(plan)
	if err != nil {
		t.Fatalf("PlanFingerprint second: %v", err)
	}
	if first != second || len(first) != len("sha256:")+64 {
		t.Fatalf("fingerprint is not stable sha256: first=%q second=%q", first, second)
	}

	equivalent := plan
	equivalent.ID = "city-stops"
	equivalent.Seed = "seed-1"
	equivalent.Entries = append([]Entry(nil), plan.Entries...)
	equivalent.Entries[0].Weather = stringPtr("RAIN")
	equivalent.Entries[0].Vehicle = &VehicleVariant{Model: "sultan"}
	equivalentFingerprint, err := PlanFingerprint(equivalent)
	if err != nil {
		t.Fatalf("PlanFingerprint equivalent: %v", err)
	}
	if equivalentFingerprint != first {
		t.Fatalf("semantic normalization changed fingerprint: first=%q equivalent=%q", first, equivalentFingerprint)
	}

	changed := equivalent
	changed.Entries = append([]Entry(nil), equivalent.Entries...)
	changed.Entries[0].StartDistanceM = float64Ptr(60)
	changedFingerprint, err := PlanFingerprint(changed)
	if err != nil {
		t.Fatalf("PlanFingerprint changed: %v", err)
	}
	if changedFingerprint == first {
		t.Fatal("meaningful plan change did not change fingerprint")
	}

	changedConfirmation := equivalent
	changedConfirmation.Entries = append([]Entry(nil), equivalent.Entries...)
	changedConfirmation.Entries[0].StopConfirmationMS = intPtr(500)
	confirmationFingerprint, err := PlanFingerprint(changedConfirmation)
	if err != nil {
		t.Fatalf("PlanFingerprint changed confirmation: %v", err)
	}
	if confirmationFingerprint == first {
		t.Fatal("stop-confirmation contract change did not change fingerprint")
	}
}
