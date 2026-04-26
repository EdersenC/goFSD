package capture

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
)

type telemetryNormalizer struct {
	stats map[string]normalizationStat
}

type normalizationStat struct {
	Mean float64 `json:"mean"`
	Std  float64 `json:"std"`
}

func loadTelemetryNormalizer(path string) (*telemetryNormalizer, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var parsed map[string]normalizationStat
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, fmt.Errorf("decode telemetry normalization stats: %w", err)
	}
	for key, item := range parsed {
		if !isFinite(item.Mean) || !isFinite(item.Std) || item.Std <= 0 {
			return nil, fmt.Errorf("invalid telemetry normalization stats for %s", key)
		}
	}
	return &telemetryNormalizer{stats: parsed}, nil
}

func (n *telemetryNormalizer) Normalize(name string, value float64) float64 {
	if n == nil {
		return value
	}
	stat, ok := n.stats[name]
	if !ok || stat.Std == 0 {
		return value
	}
	return (value - stat.Mean) / stat.Std
}

func isFinite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}
