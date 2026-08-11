package parkingcontrol

import (
	"errors"
	"fmt"
	"math"
)

const (
	// StopSignMotionPlanContractV1 keeps learned desired motion separate from
	// deterministic, feedback-controlled game actuation.
	StopSignMotionPlanContractV1  = "stop_sign_motion_plan_v1"
	StopSignDirectionForward      = "forward"
	StopSignMotionPlanMaxSpeedMPS = 8.0

	// Transitional aliases keep the internal controller migration source-compatible.
	ParkingSetpointContractV1  = StopSignMotionPlanContractV1
	ParkingDirectionForward    = StopSignDirectionForward
	ParkingSetpointMaxSpeedMPS = StopSignMotionPlanMaxSpeedMPS
)

var ErrInvalidPlan = errors.New("invalid parking setpoint plan")
var ErrInvalidSampleAge = errors.New("invalid parking setpoint sample age")

// Setpoint describes the desired physical state at one point in the plan
// horizon. DtMs is relative to Plan.SampledAtS.
type Setpoint struct {
	DtMs                        int     `json:"dt_ms"`
	DesiredWheelSteerNormalized float64 `json:"desired_wheel_steer_normalized"`
	DesiredSpeedMPS             float64 `json:"desired_speed_mps"`
	StopProbability             float64 `json:"stop_probability"`
}

// Plan is the versioned model-to-controller wire contract. SampledAtS is the
// observation time used by the model; ReceivedAtS is assigned by the backend.
type Plan struct {
	Contract    string     `json:"contract"`
	SampledAtS  float64    `json:"sampled_at_s"`
	ReceivedAtS float64    `json:"received_at_s"`
	Points      []Setpoint `json:"points"`
	Direction   string     `json:"direction"`
}

// PlanSampler owns a validated copy of a plan's setpoints.
type PlanSampler struct {
	points []Setpoint
}

// ValidatePlan rejects plans that do not match the exact forward-only v1
// contract or violate its physical bounds.
func ValidatePlan(plan Plan) error {
	if plan.Contract != ParkingSetpointContractV1 {
		return planError("contract %q must equal %q", plan.Contract, ParkingSetpointContractV1)
	}
	if plan.Direction != ParkingDirectionForward {
		return planError("direction %q must equal %q", plan.Direction, ParkingDirectionForward)
	}
	if !finite(plan.SampledAtS) || plan.SampledAtS <= 0 {
		return planError("sampled_at_s %.6f must be finite and > 0", plan.SampledAtS)
	}
	if !finite(plan.ReceivedAtS) || plan.ReceivedAtS <= 0 {
		return planError("received_at_s %.6f must be finite and > 0", plan.ReceivedAtS)
	}
	if plan.ReceivedAtS < plan.SampledAtS {
		return planError("received_at_s %.6f must be >= sampled_at_s %.6f", plan.ReceivedAtS, plan.SampledAtS)
	}
	if len(plan.Points) == 0 {
		return planError("points must not be empty")
	}

	previousDtMs := 0
	for index, point := range plan.Points {
		if point.DtMs <= previousDtMs {
			if index == 0 {
				return planError("point %d dt_ms=%d must be > 0", index, point.DtMs)
			}
			return planError("point %d dt_ms=%d must be strictly greater than previous dt_ms=%d", index, point.DtMs, previousDtMs)
		}
		if !finite(point.DesiredWheelSteerNormalized) || point.DesiredWheelSteerNormalized < -1 || point.DesiredWheelSteerNormalized > 1 {
			return planError("point %d desired_wheel_steer_normalized %.6f is outside [-1, 1]", index, point.DesiredWheelSteerNormalized)
		}
		if !finite(point.DesiredSpeedMPS) || point.DesiredSpeedMPS < 0 || point.DesiredSpeedMPS > ParkingSetpointMaxSpeedMPS {
			return planError("point %d desired_speed_mps %.6f is outside [0, %.2f]", index, point.DesiredSpeedMPS, ParkingSetpointMaxSpeedMPS)
		}
		if !finite(point.StopProbability) || point.StopProbability < 0 || point.StopProbability > 1 {
			return planError("point %d stop_probability %.6f is outside [0, 1]", index, point.StopProbability)
		}
		previousDtMs = point.DtMs
	}
	return nil
}

// NewPlanSampler validates plan and retains an immutable copy of its points.
func NewPlanSampler(plan Plan) (*PlanSampler, error) {
	if err := ValidatePlan(plan); err != nil {
		return nil, err
	}
	return &PlanSampler{points: append([]Setpoint(nil), plan.Points...)}, nil
}

// SampleByAge linearly interpolates the plan at ageMs. Ages outside the
// horizon clamp to the nearest endpoint.
func (s *PlanSampler) SampleByAge(ageMs float64) (Setpoint, error) {
	if s == nil || len(s.points) == 0 {
		return Setpoint{}, sampleAgeError("sampler is not initialized")
	}
	if !finite(ageMs) || ageMs < 0 {
		return Setpoint{}, sampleAgeError("age_ms %.6f must be finite and >= 0", ageMs)
	}
	if ageMs <= float64(s.points[0].DtMs) {
		return s.points[0], nil
	}

	last := s.points[len(s.points)-1]
	if ageMs >= float64(last.DtMs) {
		return last, nil
	}
	for index := 1; index < len(s.points); index++ {
		right := s.points[index]
		if ageMs > float64(right.DtMs) {
			continue
		}
		left := s.points[index-1]
		fraction := (ageMs - float64(left.DtMs)) / float64(right.DtMs-left.DtMs)
		return interpolateSetpoint(left, right, ageMs, fraction), nil
	}

	panic("validated parking setpoint sampler failed to cover an in-range age")
}

func interpolateSetpoint(left, right Setpoint, ageMs, fraction float64) Setpoint {
	return Setpoint{
		DtMs:                        int(math.Round(ageMs)),
		DesiredWheelSteerNormalized: interpolate(left.DesiredWheelSteerNormalized, right.DesiredWheelSteerNormalized, fraction),
		DesiredSpeedMPS:             interpolate(left.DesiredSpeedMPS, right.DesiredSpeedMPS, fraction),
		StopProbability:             interpolate(left.StopProbability, right.StopProbability, fraction),
	}
}

func interpolate(left, right, fraction float64) float64 {
	return left + fraction*(right-left)
}

func planError(format string, args ...any) error {
	return fmt.Errorf("%w: %s", ErrInvalidPlan, fmt.Sprintf(format, args...))
}

func sampleAgeError(format string, args ...any) error {
	return fmt.Errorf("%w: %s", ErrInvalidSampleAge, fmt.Sprintf(format, args...))
}
