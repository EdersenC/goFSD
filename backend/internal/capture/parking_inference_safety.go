package capture

import (
	"errors"
	"fmt"
	"math"
	"strings"
	"time"

	"awesomeProject/internal/control"
)

var ErrParkingInferencePrecondition = errors.New("parking inference precondition failed")
var ErrParkingInferenceComplete = errors.New("parking inference completed successfully")
var ErrParkingInferenceDeadlineExceeded = errors.New("parking evaluation deadline exceeded")

// These bounds mirror the complete forward-bay start distribution produced by
// fivem/src/parking/curriculum.ts, including its maximum deterministic jitter.
const (
	parkingStartLongitudinalMinM = -17.0
	parkingStartLongitudinalMaxM = -9.5
	parkingStartLateralLimitM    = 2.75
	parkingStartHeadingLimitDeg  = 17.0
	parkingTargetPositionDriftM  = 0.25
	parkingTargetHeadingDriftDeg = 1.0
	parkingMaximumTiltDeg        = 5.0
	parkingReverseSpeedLimitMPS  = -0.1
	parkingSourceClockLeadLimit  = 50 * time.Millisecond
)

func (i *Inferencer) validateParkingInferenceStart() error {
	_, err := i.parkingInferenceStartTarget()
	return err
}

func (i *Inferencer) validateInferenceActuatorReady() error {
	if i.actuator == nil {
		return fmt.Errorf("%w: actuator service is not configured", ErrInferenceActuatorUnavailable)
	}
	provider, ok := i.actuator.(actuatorStateProvider)
	if !ok {
		return fmt.Errorf("%w: actuator does not expose readiness state", ErrInferenceActuatorUnavailable)
	}
	if _, ok := i.actuator.(actuatorParkingPlanSubmitter); !ok {
		return fmt.Errorf("%w: actuator does not support parking setpoint plans", ErrInferenceActuatorUnavailable)
	}
	if _, ok := i.actuator.(actuatorParkingSafetyStopper); !ok {
		return fmt.Errorf("%w: actuator does not support parking safety stops", ErrInferenceActuatorUnavailable)
	}
	state := provider.State()
	if !state.Supported {
		return fmt.Errorf("%w: virtual controller is unsupported on %s", ErrInferenceActuatorUnavailable, state.Platform)
	}
	if !state.Ready {
		detail := strings.TrimSpace(state.LastError)
		if detail == "" {
			detail = "virtual controller is not ready"
		}
		return fmt.Errorf("%w: %s", ErrInferenceActuatorUnavailable, detail)
	}
	if detail := strings.TrimSpace(state.LastApplyError); detail != "" {
		return fmt.Errorf("%w: virtual controller has an active apply fault: %s", ErrInferenceActuatorUnavailable, detail)
	}
	if !state.ParkingController.Ready {
		return fmt.Errorf(
			"%w: parking controller requires a verified vehicle calibration profile",
			ErrInferenceActuatorUnavailable,
		)
	}
	if state.ParkingController.Contract != i.config.ControlContract {
		return fmt.Errorf(
			"%w: actuator control contract %q does not match inference contract %q",
			ErrInferenceActuatorUnavailable,
			state.ParkingController.Contract,
			i.config.ControlContract,
		)
	}
	if err := validateOrderedInts("actuator control horizon timing", state.ParkingController.ExpectedHorizonDtMs, i.config.ControlHorizonDtMs); err != nil {
		return fmt.Errorf("%w: %v", ErrInferenceActuatorUnavailable, err)
	}
	if i.telemetry == nil {
		return fmt.Errorf("%w: current vehicle identity is unavailable", ErrInferenceActuatorUnavailable)
	}
	ego, _ := i.telemetry.LatestEgoTelemetrySnapshot()
	if ego == nil || ego.VehicleModelHash == 0 {
		return fmt.Errorf("%w: current vehicle model hash is unavailable", ErrInferenceActuatorUnavailable)
	}
	if ego.VehicleModelHash != state.ParkingController.Calibration.VehicleModelHash {
		return fmt.Errorf(
			"%w: current vehicle model hash %d does not match calibrated hash %d",
			ErrInferenceActuatorUnavailable,
			ego.VehicleModelHash,
			state.ParkingController.Calibration.VehicleModelHash,
		)
	}
	return nil
}

func (i *Inferencer) parkingInferenceStartTarget() (parkingInferenceTarget, error) {
	telemetry, err := i.currentParkingInferenceTelemetry()
	if err != nil {
		return parkingInferenceTarget{}, err
	}
	if err := validateParkingStartEnvelope(*telemetry); err != nil {
		return parkingInferenceTarget{}, err
	}
	return parkingTargetFromTelemetry(*telemetry)
}

func (i *Inferencer) validateActiveParkingInference() error {
	i.mu.Lock()
	completed := i.parkingCompleted
	safetyTripped := i.parkingSafetyTripped
	expectedTarget := cloneParkingInferenceTarget(i.parkingTarget)
	startEnvelopePending := i.parkingStartEnvelopePending
	i.mu.Unlock()
	if completed {
		return ErrParkingInferenceComplete
	}
	if safetyTripped {
		return parkingInferenceError("parking safety interlock is latched; stop and restart inference after restoring the target and start pose")
	}

	telemetry, err := i.currentParkingInferenceTelemetry()
	if err != nil {
		return err
	}
	if expectedTarget == nil {
		return parkingInferenceError("parking inference session target is unavailable")
	}
	currentTarget, err := parkingTargetFromTelemetry(*telemetry)
	if err != nil {
		return err
	}
	if !sameParkingInferenceTarget(*expectedTarget, currentTarget) {
		return parkingInferenceError("parking target changed while inference was active; stop and restart from a valid curriculum pose")
	}
	if startEnvelopePending {
		if err := validateParkingStartEnvelope(*telemetry); err != nil {
			return err
		}
		i.mu.Lock()
		if !i.parkingSafetyTripped {
			i.parkingStartEnvelopePending = false
		}
		i.mu.Unlock()
	}
	return nil
}

func (i *Inferencer) currentParkingInferenceTelemetry() (*control.RuntimeTelemetry, error) {
	if i.telemetry == nil {
		return nil, parkingInferenceError("telemetry store is not configured")
	}

	telemetry, telemetryAt := i.telemetry.LatestTelemetrySnapshot()
	if telemetry == nil {
		return nil, parkingInferenceError("FiveM telemetry is unavailable")
	}
	now := i.nowFunc().UTC()
	if telemetryAt.IsZero() || telemetryAge(now, telemetryAt) > i.telemetryStaleAfter {
		return nil, parkingInferenceError("FiveM telemetry is stale")
	}
	if err := validateParkingSourceTimestamp("FiveM telemetry", now, float64(telemetry.TimestampMs)/1000.0, i.telemetryStaleAfter); err != nil {
		return nil, parkingInferenceError("%v", err)
	}
	if !telemetry.ParkingTargetConfigured {
		return nil, parkingInferenceError("parking target is not configured; calibrate a bay before starting inference")
	}

	ego, egoAt := i.telemetry.LatestEgoTelemetrySnapshot()
	if ego == nil {
		return nil, parkingInferenceError("active ego telemetry is unavailable")
	}
	if egoAt.IsZero() || telemetryAge(now, egoAt) > i.telemetryStaleAfter {
		return nil, parkingInferenceError("active ego telemetry is stale")
	}
	if err := validateParkingSourceTimestamp("active ego telemetry", now, ego.TimestampS, i.telemetryStaleAfter); err != nil {
		return nil, parkingInferenceError("%v", err)
	}
	if !ego.Valid {
		reason := ego.InvalidReason
		if reason == "" {
			reason = "unknown validation failure"
		}
		return nil, parkingInferenceError("active ego telemetry is invalid: %s", reason)
	}
	if err := validateParkingOperatingState(*telemetry); err != nil {
		return nil, err
	}
	return telemetry, nil
}

func validateParkingSourceTimestamp(label string, now time.Time, timestampS float64, maxAge time.Duration) error {
	if !finiteParkingValue(timestampS) || timestampS <= 0 {
		return fmt.Errorf("%s source timestamp is unavailable", label)
	}
	if maxAge <= 0 {
		return fmt.Errorf("%s source timestamp age limit is invalid", label)
	}
	sourceAt := time.Unix(0, int64(timestampS*float64(time.Second))).UTC()
	age := now.Sub(sourceAt)
	if age < -parkingSourceClockLeadLimit {
		return fmt.Errorf("%s source timestamp is %s in the future", label, -age)
	}
	if age > maxAge {
		return fmt.Errorf("%s source telemetry is stale: age=%s", label, age)
	}
	return nil
}

func validateParkingOperatingState(telemetry control.RuntimeTelemetry) error {
	phase := strings.ToLower(strings.TrimSpace(telemetry.ParkingPhase))
	if phase == "succeeded" && telemetry.ParkingParked {
		return ErrParkingInferenceComplete
	}
	evaluationPhase := phase == "ready" || (phase == "settling" && telemetry.ParkingAttemptCount == 0)
	if !evaluationPhase {
		if phase == "" {
			phase = "missing"
		}
		return parkingInferenceError("parking phase must be ready or evaluation settling for inference; current phase is %s", phase)
	}
	if telemetry.OnGround == nil {
		return parkingInferenceError("vehicle ground-contact telemetry is unavailable")
	}
	if !*telemetry.OnGround {
		return parkingInferenceError("vehicle left the ground")
	}
	if collision := strings.TrimSpace(telemetry.CollisionState); collision != "" {
		return parkingInferenceError("vehicle collision detected: %s", collision)
	}
	if telemetry.PitchDeg == nil || telemetry.RollDeg == nil {
		return parkingInferenceError("vehicle pitch/roll telemetry is unavailable")
	}
	if !finiteParkingValue(*telemetry.PitchDeg) || !finiteParkingValue(*telemetry.RollDeg) {
		return parkingInferenceError("vehicle pitch/roll telemetry is invalid")
	}
	if math.Abs(*telemetry.PitchDeg) > parkingMaximumTiltDeg || math.Abs(*telemetry.RollDeg) > parkingMaximumTiltDeg {
		return parkingInferenceError("vehicle exceeded %.1f degree parking tilt limit", parkingMaximumTiltDeg)
	}

	forwardSpeed, err := parkingForwardSpeedMPS(telemetry)
	if err != nil {
		return err
	}
	if forwardSpeed < parkingReverseSpeedLimitMPS {
		return parkingInferenceError("reverse motion detected at %.3fm/s during forward-bay inference", forwardSpeed)
	}
	return nil
}

func parkingForwardSpeedMPS(telemetry control.RuntimeTelemetry) (float64, error) {
	if telemetry.VelocityX == nil || telemetry.VelocityY == nil {
		return 0, parkingInferenceError("vehicle velocity telemetry is unavailable")
	}
	if !finiteParkingValue(*telemetry.VelocityX) || !finiteParkingValue(*telemetry.VelocityY) || !finiteParkingValue(telemetry.CurrentYaw) {
		return 0, parkingInferenceError("vehicle velocity/heading telemetry is invalid")
	}
	headingRad := telemetry.CurrentYaw * math.Pi / 180
	forwardX := -math.Sin(headingRad)
	forwardY := math.Cos(headingRad)
	return (*telemetry.VelocityX * forwardX) + (*telemetry.VelocityY * forwardY), nil
}

func validateParkingStartEnvelope(telemetry control.RuntimeTelemetry) error {
	longitudinal := telemetry.ParkingLongitudinalError
	if !finiteParkingValue(longitudinal) || longitudinal < parkingStartLongitudinalMinM || longitudinal > parkingStartLongitudinalMaxM {
		return parkingInferenceError(
			"parking start longitudinal offset %.3fm is outside the forward curriculum range [%.1f, %.1f]m",
			longitudinal,
			parkingStartLongitudinalMinM,
			parkingStartLongitudinalMaxM,
		)
	}

	lateral := telemetry.ParkingLateralError
	if !finiteParkingValue(lateral) || math.Abs(lateral) > parkingStartLateralLimitM {
		return parkingInferenceError(
			"parking start lateral offset %.3fm exceeds the forward curriculum limit +/-%.2fm",
			lateral,
			parkingStartLateralLimitM,
		)
	}

	heading := telemetry.ParkingHeadingError
	if !finiteParkingValue(heading) || math.Abs(heading) > parkingStartHeadingLimitDeg {
		return parkingInferenceError(
			"parking start heading error %.3fdeg exceeds the forward curriculum limit +/-%.1fdeg",
			heading,
			parkingStartHeadingLimitDeg,
		)
	}
	return nil
}

type parkingInferenceTarget struct {
	x          float64
	y          float64
	headingDeg float64
}

func parkingTargetFromTelemetry(telemetry control.RuntimeTelemetry) (parkingInferenceTarget, error) {
	if telemetry.PositionX == nil || telemetry.PositionY == nil {
		return parkingInferenceTarget{}, parkingInferenceError("ego position is required to bind the inference session to its calibrated parking target")
	}
	if !finiteParkingValue(*telemetry.PositionX) || !finiteParkingValue(*telemetry.PositionY) || !finiteParkingValue(telemetry.CurrentYaw) {
		return parkingInferenceTarget{}, parkingInferenceError("ego pose contains a non-finite value")
	}

	headingDeg := normalizeParkingHeading(telemetry.CurrentYaw + telemetry.ParkingHeadingError)
	headingRad := headingDeg * math.Pi / 180
	forwardX := -math.Sin(headingRad)
	forwardY := math.Cos(headingRad)
	rightX := math.Cos(headingRad)
	rightY := math.Sin(headingRad)
	target := parkingInferenceTarget{
		x:          *telemetry.PositionX - (forwardX * telemetry.ParkingLongitudinalError) - (rightX * telemetry.ParkingLateralError),
		y:          *telemetry.PositionY - (forwardY * telemetry.ParkingLongitudinalError) - (rightY * telemetry.ParkingLateralError),
		headingDeg: headingDeg,
	}
	if !finiteParkingValue(target.x) || !finiteParkingValue(target.y) {
		return parkingInferenceTarget{}, parkingInferenceError("calibrated parking target pose is invalid")
	}
	return target, nil
}

func sameParkingInferenceTarget(expected parkingInferenceTarget, actual parkingInferenceTarget) bool {
	positionDrift := math.Hypot(actual.x-expected.x, actual.y-expected.y)
	headingDrift := math.Abs(parkingHeadingDelta(actual.headingDeg, expected.headingDeg))
	return positionDrift <= parkingTargetPositionDriftM && headingDrift <= parkingTargetHeadingDriftDeg
}

func cloneParkingInferenceTarget(target *parkingInferenceTarget) *parkingInferenceTarget {
	if target == nil {
		return nil
	}
	copyTarget := *target
	return &copyTarget
}

func normalizeParkingHeading(heading float64) float64 {
	normalized := math.Mod(heading, 360)
	if normalized < 0 {
		normalized += 360
	}
	return normalized
}

func parkingHeadingDelta(target float64, source float64) float64 {
	delta := normalizeParkingHeading(target) - normalizeParkingHeading(source)
	if delta > 180 {
		delta -= 360
	}
	if delta < -180 {
		delta += 360
	}
	return delta
}

func parkingInferenceError(format string, args ...any) error {
	return fmt.Errorf("%w: %s", ErrParkingInferencePrecondition, fmt.Sprintf(format, args...))
}

func telemetryAge(now time.Time, updatedAt time.Time) time.Duration {
	age := now.Sub(updatedAt)
	if age < 0 {
		return 0
	}
	return age
}

func finiteParkingValue(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}
