package actuator

import "errors"

var ErrUnsupportedPlatform = errors.New("virtual controller actuator is only supported on Windows")

type controlState struct {
	Steer     float64 `json:"steer"`
	Throttle  float64 `json:"throttle"`
	Brake     float64 `json:"brake"`
	Handbrake bool    `json:"handbrake"`
}

type controller interface {
	Apply(controlState) error
	Close()
}
