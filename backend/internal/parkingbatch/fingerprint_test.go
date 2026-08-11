package parkingbatch

import (
	"math"
	"testing"
)

func TestPlanFingerprintTracksExactPlanDefinition(t *testing.T) {
	plan := Plan{
		ID:   "test-plan",
		Seed: "test-seed",
		Entries: []Entry{{
			ID:               "bay-01",
			ParkDest:         Pose{X: 0, Y: 0, Z: 0, Heading: 0},
			StartDest:        Pose{X: 0, Y: -11, Z: 0, Heading: 0},
			Variations:       []Variation{{ID: "base"}},
			CollectionAmount: 3,
		}},
	}
	const expected = "sha256:375c329e8c5f35f94222cd3e2307e6e4d6b9fc7b113055acfdbf5a232b202203"
	if got := PlanFingerprint(plan); got != expected {
		t.Fatalf("unexpected cross-runtime fixture fingerprint: got=%q want=%q", got, expected)
	}

	changedSeed := plan
	changedSeed.Seed = "changed-seed"
	if PlanFingerprint(changedSeed) == PlanFingerprint(plan) {
		t.Fatal("seed edits must change the fingerprint")
	}

	changedPose := plan
	changedPose.Entries = append([]Entry(nil), plan.Entries...)
	changedPose.Entries[0].StartDest.Y = -12
	if PlanFingerprint(changedPose) == PlanFingerprint(plan) {
		t.Fatal("pose edits must change the fingerprint")
	}

	signedZero := plan
	signedZero.Entries = append([]Entry(nil), plan.Entries...)
	signedZero.Entries[0].ParkDest.X = math.Copysign(0, -1)
	if PlanFingerprint(signedZero) != PlanFingerprint(plan) {
		t.Fatal("signed zero must canonicalize across JSON persistence")
	}
}
