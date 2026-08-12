package stopsignbatch

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"strings"
)

// PlanFingerprint returns a stable identity for the normalized expanded plan.
// Semantically equivalent whitespace and weather/model casing hash equally.
func PlanFingerprint(plan Plan) (string, error) {
	jobs, err := Expand(plan)
	if err != nil {
		return "", err
	}
	canonical := struct {
		Version string `json:"version"`
		ID      string `json:"id"`
		Seed    string `json:"seed"`
		Jobs    []Job  `json:"jobs"`
	}{
		Version: "stop-sign-plan-fingerprint-v4",
		ID:      strings.TrimSpace(plan.ID),
		Seed:    strings.TrimSpace(plan.Seed),
		Jobs:    jobs,
	}
	encoded, err := json.Marshal(canonical)
	if err != nil {
		return "", fmt.Errorf("encode stop-sign plan fingerprint: %w", err)
	}
	sum := sha256.Sum256(encoded)
	return fmt.Sprintf("sha256:%x", sum), nil
}
