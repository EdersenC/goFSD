package capture

import (
	"errors"
	"fmt"
	"math"
	"strings"
	"time"

	"awesomeProject/internal/control"
)

var ErrStopSignInferencePrecondition = errors.New("stop-sign inference precondition failed")
var ErrStopSignInferenceComplete = errors.New("stop-sign inference completed successfully")
var ErrStopSignInferenceDeadlineExceeded = errors.New("stop-sign evaluation deadline exceeded")

// These bounds mirror the exact user-calibrated Phase 1 straight corridor.
const (
	stopSignStartLongitudinalMinM = -250.0
	stopSignStartLongitudinalMaxM = -5.0
	stopSignStartLateralLimitM    = 1.5
	stopSignStartHeadingLimitDeg  = 10.0
	stopSignTargetPositionDriftM  = 0.25
	stopSignTargetHeadingDriftDeg = 1.0
	stopSignMaximumTiltDeg        = 5.0
	stopSignReverseSpeedLimitMPS  = -0.1
	stopSignSourceClockLeadLimit  = 50 * time.Millisecond
)

func (i *Inferencer) validateStopSignInferenceStart() error {
	_, err := i.stopSignInferenceStartTarget()
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
	if _, ok := i.actuator.(actuatorStopSignPlanSubmitter); !ok {
		return fmt.Errorf("%w: actuator does not support stopSign setpoint plans", ErrInferenceActuatorUnavailable)
	}
	if _, ok := i.actuator.(actuatorStopSignSafetyStopper); !ok {
		return fmt.Errorf("%w: actuator does not support stopSign safety stops", ErrInferenceActuatorUnavailable)
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
	if !state.StopSignController.Ready {
		return fmt.Errorf(
			"%w: stop-sign controller requires a verified vehicle calibration profile",
			ErrInferenceActuatorUnavailable,
		)
	}
	if state.StopSignController.Contract != i.config.ControlContract {
		return fmt.Errorf(
			"%w: actuator control contract %q does not match inference contract %q",
			ErrInferenceActuatorUnavailable,
			state.StopSignController.Contract,
			i.config.ControlContract,
		)
	}
	if err := validateOrderedInts("actuator control horizon timing", state.StopSignController.ExpectedHorizonDtMs, i.config.ControlHorizonDtMs); err != nil {
		return fmt.Errorf("%w: %v", ErrInferenceActuatorUnavailable, err)
	}
	if i.telemetry == nil {
		return fmt.Errorf("%w: current vehicle identity is unavailable", ErrInferenceActuatorUnavailable)
	}
	ego, _ := i.telemetry.LatestEgoTelemetrySnapshot()
	if ego == nil || ego.VehicleModelHash == 0 {
		return fmt.Errorf("%w: current vehicle model hash is unavailable", ErrInferenceActuatorUnavailable)
	}
	if ego.VehicleModelHash != state.StopSignController.Calibration.VehicleModelHash {
		return fmt.Errorf(
			"%w: current vehicle model hash %d does not match calibrated hash %d",
			ErrInferenceActuatorUnavailable,
			ego.VehicleModelHash,
			state.StopSignController.Calibration.VehicleModelHash,
		)
	}
	return nil
}

func (i *Inferencer) stopSignInferenceStartTarget() (stopSignInferenceTarget, error) {
	telemetry, err := i.currentStopSignInferenceTelemetry()
	if err != nil {
		return stopSignInferenceTarget{}, err
	}
	if err := validateStopSignStartEnvelope(*telemetry); err != nil {
		return stopSignInferenceTarget{}, err
	}
	return stopSignTargetFromTelemetry(*telemetry)
}

func (i *Inferencer) validateActiveStopSignInference() error {
	i.mu.Lock()
	completed := i.stopSignCompleted
	safetyTripped := i.stopSignSafetyTripped
	expectedTarget := cloneStopSignInferenceTarget(i.stopSignTarget)
	startEnvelopePending := i.stopSignStartEnvelopePending
	i.mu.Unlock()
	if completed {
		return ErrStopSignInferenceComplete
	}
	if safetyTripped {
		return stopSignInferenceError("stop-sign safety interlock is latched; stop and restart inference after restoring the target and start pose")
	}

	telemetry, err := i.currentStopSignInferenceTelemetry()
	if err != nil {
		return err
	}
	if expectedTarget == nil {
		return stopSignInferenceError("stop-sign inference session target is unavailable")
	}
	currentTarget, err := stopSignTargetFromTelemetry(*telemetry)
	if err != nil {
		return err
	}
	if !sameStopSignInferenceTarget(*expectedTarget, currentTarget) {
		return stopSignInferenceError("stop-sign target changed while inference was active; stop and restart from a valid approach pose")
	}
	if startEnvelopePending {
		if err := validateStopSignStartEnvelope(*telemetry); err != nil {
			return err
		}
		i.mu.Lock()
		if !i.stopSignSafetyTripped {
			i.stopSignStartEnvelopePending = false
		}
		i.mu.Unlock()
	}
	return nil
}

func (i *Inferencer) currentStopSignInferenceTelemetry() (*control.RuntimeTelemetry, error) {
	if i.telemetry == nil {
		return nil, stopSignInferenceError("telemetry store is not configured")
	}

	telemetry, telemetryAt := i.telemetry.LatestTelemetrySnapshot()
	if telemetry == nil {
		return nil, stopSignInferenceError("FiveM telemetry is unavailable")
	}
	now := i.nowFunc().UTC()
	if telemetryAt.IsZero() || telemetryAge(now, telemetryAt) > i.telemetryStaleAfter {
		return nil, stopSignInferenceError("FiveM telemetry is stale")
	}
	if err := validateStopSignSourceTimestamp("FiveM telemetry", now, float64(telemetry.TimestampMs)/1000.0, i.telemetryStaleAfter); err != nil {
		return nil, stopSignInferenceError("%v", err)
	}
	if !telemetry.StopSignTargetConfigured || telemetry.StopSignEgoStopPose == nil {
		return nil, stopSignInferenceError("stop-sign target is not configured; mark a sign before starting inference")
	}

	ego, egoAt := i.telemetry.LatestEgoTelemetrySnapshot()
	if ego == nil {
		return nil, stopSignInferenceError("active ego telemetry is unavailable")
	}
	if egoAt.IsZero() || telemetryAge(now, egoAt) > i.telemetryStaleAfter {
		return nil, stopSignInferenceError("active ego telemetry is stale")
	}
	if err := validateStopSignSourceTimestamp("active ego telemetry", now, ego.TimestampS, i.telemetryStaleAfter); err != nil {
		return nil, stopSignInferenceError("%v", err)
	}
	if !ego.Valid {
		reason := ego.InvalidReason
		if reason == "" {
			reason = "unknown validation failure"
		}
		return nil, stopSignInferenceError("active ego telemetry is invalid: %s", reason)
	}
	if err := validateStopSignOperatingState(*telemetry); err != nil {
		return nil, err
	}
	return telemetry, nil
}

func validateStopSignSourceTimestamp(label string, now time.Time, timestampS float64, maxAge time.Duration) error {
	if !finiteStopSignValue(timestampS) || timestampS <= 0 {
		return fmt.Errorf("%s source timestamp is unavailable", label)
	}
	if maxAge <= 0 {
		return fmt.Errorf("%s source timestamp age limit is invalid", label)
	}
	sourceAt := time.Unix(0, int64(timestampS*float64(time.Second))).UTC()
	age := now.Sub(sourceAt)
	if age < -stopSignSourceClockLeadLimit {
		return fmt.Errorf("%s source timestamp is %s in the future", label, -age)
	}
	if age > maxAge {
		return fmt.Errorf("%s source telemetry is stale: age=%s", label, age)
	}
	return nil
}

func validateStopSignOperatingState(telemetry control.RuntimeTelemetry) error {
	phase := strings.ToLower(strings.TrimSpace(telemetry.StopSignPhase))
	if phase == "complete" {
		return ErrStopSignInferenceComplete
	}
	allowedPhase := phase == "idle" || phase == "accelerate" || phase == "cruise_approach" ||
		phase == "decelerate" || phase == "stop_hold" || phase == "release"
	if !allowedPhase {
		if phase == "" {
			phase = "missing"
		}
		return stopSignInferenceError("stop-sign phase is not safe for inference; current phase is %s", phase)
	}
	if telemetry.OnGround == nil {
		return stopSignInferenceError("vehicle ground-contact telemetry is unavailable")
	}
	if !*telemetry.OnGround {
		return stopSignInferenceError("vehicle left the ground")
	}
	if collision := strings.TrimSpace(telemetry.CollisionState); collision != "" {
		return stopSignInferenceError("vehicle collision detected: %s", collision)
	}
	if telemetry.PitchDeg == nil || telemetry.RollDeg == nil {
		return stopSignInferenceError("vehicle pitch/roll telemetry is unavailable")
	}
	if !finiteStopSignValue(*telemetry.PitchDeg) || !finiteStopSignValue(*telemetry.RollDeg) {
		return stopSignInferenceError("vehicle pitch/roll telemetry is invalid")
	}
	if math.Abs(*telemetry.PitchDeg) > stopSignMaximumTiltDeg || math.Abs(*telemetry.RollDeg) > stopSignMaximumTiltDeg {
		return stopSignInferenceError("vehicle exceeded %.1f degree stop-sign tilt limit", stopSignMaximumTiltDeg)
	}

	forwardSpeed, err := stopSignForwardSpeedMPS(telemetry)
	if err != nil {
		return err
	}
	if forwardSpeed < stopSignReverseSpeedLimitMPS {
		return stopSignInferenceError("reverse motion detected at %.3fm/s during stop-sign inference", forwardSpeed)
	}
	return nil
}

func stopSignForwardSpeedMPS(telemetry control.RuntimeTelemetry) (float64, error) {
	if telemetry.VelocityX == nil || telemetry.VelocityY == nil {
		return 0, stopSignInferenceError("vehicle velocity telemetry is unavailable")
	}
	if !finiteStopSignValue(*telemetry.VelocityX) || !finiteStopSignValue(*telemetry.VelocityY) || !finiteStopSignValue(telemetry.CurrentYaw) {
		return 0, stopSignInferenceError("vehicle velocity/heading telemetry is invalid")
	}
	headingRad := telemetry.CurrentYaw * math.Pi / 180
	forwardX := -math.Sin(headingRad)
	forwardY := math.Cos(headingRad)
	return (*telemetry.VelocityX * forwardX) + (*telemetry.VelocityY * forwardY), nil
}

func validateStopSignStartEnvelope(telemetry control.RuntimeTelemetry) error {
	longitudinal := telemetry.StopSignLongitudinalErrorM
	if !finiteStopSignValue(longitudinal) || longitudinal < stopSignStartLongitudinalMinM || longitudinal > stopSignStartLongitudinalMaxM {
		return stopSignInferenceError(
			"stop-sign start longitudinal offset %.3fm is outside the approach range [%.1f, %.1f]m",
			longitudinal,
			stopSignStartLongitudinalMinM,
			stopSignStartLongitudinalMaxM,
		)
	}

	lateral := telemetry.StopSignLateralErrorM
	if !finiteStopSignValue(lateral) || math.Abs(lateral) > stopSignStartLateralLimitM {
		return stopSignInferenceError(
			"stop-sign start lateral offset %.3fm exceeds the approach limit +/-%.2fm",
			lateral,
			stopSignStartLateralLimitM,
		)
	}

	heading := telemetry.StopSignHeadingErrorDeg
	if !finiteStopSignValue(heading) || math.Abs(heading) > stopSignStartHeadingLimitDeg {
		return stopSignInferenceError(
			"stop-sign start heading error %.3fdeg exceeds the approach limit +/-%.1fdeg",
			heading,
			stopSignStartHeadingLimitDeg,
		)
	}
	return nil
}

type stopSignInferenceTarget struct {
	x          float64
	y          float64
	headingDeg float64
}

func stopSignTargetFromTelemetry(telemetry control.RuntimeTelemetry) (stopSignInferenceTarget, error) {
	pose := telemetry.StopSignEgoStopPose
	if pose == nil {
		return stopSignInferenceTarget{}, stopSignInferenceError("ego stop pose is required to bind the inference session")
	}
	target := stopSignInferenceTarget{
		x:          pose.X,
		y:          pose.Y,
		headingDeg: normalizeStopSignHeading(pose.Heading),
	}
	if !finiteStopSignValue(target.x) || !finiteStopSignValue(target.y) {
		return stopSignInferenceTarget{}, stopSignInferenceError("calibrated stop-sign target pose is invalid")
	}
	return target, nil
}

func sameStopSignInferenceTarget(expected stopSignInferenceTarget, actual stopSignInferenceTarget) bool {
	positionDrift := math.Hypot(actual.x-expected.x, actual.y-expected.y)
	headingDrift := math.Abs(stopSignHeadingDelta(actual.headingDeg, expected.headingDeg))
	return positionDrift <= stopSignTargetPositionDriftM && headingDrift <= stopSignTargetHeadingDriftDeg
}

func cloneStopSignInferenceTarget(target *stopSignInferenceTarget) *stopSignInferenceTarget {
	if target == nil {
		return nil
	}
	copyTarget := *target
	return &copyTarget
}

func normalizeStopSignHeading(heading float64) float64 {
	normalized := math.Mod(heading, 360)
	if normalized < 0 {
		normalized += 360
	}
	return normalized
}

func stopSignHeadingDelta(target float64, source float64) float64 {
	delta := normalizeStopSignHeading(target) - normalizeStopSignHeading(source)
	if delta > 180 {
		delta -= 360
	}
	if delta < -180 {
		delta += 360
	}
	return delta
}

func stopSignInferenceError(format string, args ...any) error {
	return fmt.Errorf("%w: %s", ErrStopSignInferencePrecondition, fmt.Sprintf(format, args...))
}

func telemetryAge(now time.Time, updatedAt time.Time) time.Duration {
	age := now.Sub(updatedAt)
	if age < 0 {
		return 0
	}
	return age
}

func finiteStopSignValue(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}
