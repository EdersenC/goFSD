package parkingcontrol

import (
	"fmt"
	"math"
)

// SteeringCalibrationPoint maps normalized physical wheel steer to normalized
// controller steer. Points are interpolated linearly.
type SteeringCalibrationPoint struct {
	WheelSteer float64 `json:"wheel_steer" toml:"wheel_steer"`
	Command    float64 `json:"command" toml:"command"`
}

type steeringProfile struct {
	points []SteeringCalibrationPoint
}

func newSteeringProfile(points []SteeringCalibrationPoint) (steeringProfile, error) {
	if len(points) < 2 {
		return steeringProfile{}, configError("steering profile must contain at least two points")
	}
	cloned := append([]SteeringCalibrationPoint(nil), points...)
	for index, point := range cloned {
		if !finite(point.WheelSteer) || !finite(point.Command) {
			return steeringProfile{}, configError("steering profile point %d contains a non-finite value", index)
		}
		if point.WheelSteer < -1 || point.WheelSteer > 1 || point.Command < -1 || point.Command > 1 {
			return steeringProfile{}, configError("steering profile point %d must stay within normalized range [-1, 1]", index)
		}
		if index == 0 {
			continue
		}
		previous := cloned[index-1]
		if point.WheelSteer <= previous.WheelSteer {
			return steeringProfile{}, configError("steering profile wheel values must be strictly increasing at point %d", index)
		}
		if point.Command < previous.Command {
			return steeringProfile{}, configError("steering profile commands must be monotonic at point %d", index)
		}
	}
	if cloned[0].WheelSteer != -1 || cloned[len(cloned)-1].WheelSteer != 1 {
		return steeringProfile{}, configError("steering profile must cover normalized wheel range from -1 to 1")
	}
	return steeringProfile{points: cloned}, nil
}

func (p steeringProfile) commandFor(wheelSteer float64) float64 {
	if wheelSteer <= p.points[0].WheelSteer {
		return p.points[0].Command
	}
	last := p.points[len(p.points)-1]
	if wheelSteer >= last.WheelSteer {
		return last.Command
	}
	for index := 1; index < len(p.points); index++ {
		right := p.points[index]
		if wheelSteer > right.WheelSteer {
			continue
		}
		left := p.points[index-1]
		span := right.WheelSteer - left.WheelSteer
		fraction := (wheelSteer - left.WheelSteer) / span
		return left.Command + fraction*(right.Command-left.Command)
	}
	panic(fmt.Sprintf("steering profile interpolation missed covered value %.6f", wheelSteer))
}

func finite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func clamp(value, minimum, maximum float64) float64 {
	return math.Min(math.Max(value, minimum), maximum)
}
