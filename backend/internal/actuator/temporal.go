package actuator

import (
	"fmt"
	"log"
	"math"
	"sort"
	"strings"
	"time"

	"awesomeProject/internal/control"
)

const (
	PlanStateMissing = "missing"
	PlanStateFresh   = "fresh"
	PlanStateUsable  = "usable"
	PlanStateStale   = "stale"
	PlanStateExpired = "expired"
)

var DefaultPredictionHorizonDtMs = []int{100, 200, 350, 500, 750, 1000, 1500}

type FuturePoint struct {
	DtMs            int      `json:"dt_ms"`
	X               *float64 `json:"x,omitempty"`
	Y               *float64 `json:"y,omitempty"`
	DesiredSpeedMPS *float64 `json:"desired_speed_mps,omitempty"`
	HeadingRad      *float64 `json:"heading_rad,omitempty"`
	Steer           *float64 `json:"steer,omitempty"`
	Throttle        *float64 `json:"throttle,omitempty"`
	Brake           *float64 `json:"brake,omitempty"`
}

type PredictionHorizon struct {
	InputTimestampS    float64       `json:"input_timestamp_s"`
	ReceivedTimestampS float64       `json:"received_timestamp_s"`
	Points             []FuturePoint `json:"points"`
	Confidence         float64       `json:"confidence"`
	Source             string        `json:"source,omitempty"`
}

type TemporalPlanBuffer struct {
	ActivePlan         *PredictionHorizon
	PreviousPlan       *PredictionHorizon
	LastControls       ControlCommand
	LastValidPlanTimeS float64
}

type TemporalDebug struct {
	Enabled                bool         `json:"enabled"`
	NowS                   float64      `json:"now_s,omitempty"`
	PlanState              string       `json:"plan_state"`
	PlanAgeMs              float64      `json:"plan_age_ms"`
	EgoStateTimestampS     float64      `json:"ego_state_timestamp_s,omitempty"`
	TelemetryAgeMs         float64      `json:"telemetry_age_ms"`
	TelemetryValid         bool         `json:"telemetry_valid"`
	TelemetryInvalidReason string       `json:"telemetry_invalid_reason,omitempty"`
	TargetDtMs             float64      `json:"target_dt_ms"`
	LookaheadMs            float64      `json:"lookahead_ms"`
	EstimatedLatencyMs     float64      `json:"estimated_actuation_latency_ms"`
	InputTimestampS        float64      `json:"input_timestamp_s,omitempty"`
	ReceivedTimestampS     float64      `json:"received_timestamp_s,omitempty"`
	PlanConfidence         float64      `json:"plan_confidence,omitempty"`
	Source                 string       `json:"source,omitempty"`
	CurrentSpeedMPS        float64      `json:"current_speed_mps"`
	SelectedPoint          *FuturePoint `json:"selected_point,omitempty"`
	HorizonClamped         bool         `json:"horizon_clamped"`
	DesiredSpeedMPS        *float64     `json:"desired_speed_mps,omitempty"`
	LateralMode            string       `json:"lateral_mode,omitempty"`
	LongitudinalMode       string       `json:"longitudinal_mode,omitempty"`
	ControlMode            string       `json:"control_mode,omitempty"`
	RequestedSteer         *float64     `json:"requested_steer,omitempty"`
	RequestedThrottle      *float64     `json:"requested_throttle,omitempty"`
	RequestedBrake         *float64     `json:"requested_brake,omitempty"`
	RawSteer               float64      `json:"raw_steer"`
	SmoothedSteer          float64      `json:"smoothed_steer"`
	AppliedSteer           float64      `json:"applied_steer"`
	RawThrottle            float64      `json:"raw_throttle"`
	RawBrake               float64      `json:"raw_brake"`
	SmoothedThrottle       float64      `json:"smoothed_throttle"`
	SmoothedBrake          float64      `json:"smoothed_brake"`
	AppliedThrottle        float64      `json:"applied_throttle"`
	AppliedBrake           float64      `json:"applied_brake"`
	FallbackApplied        bool         `json:"fallback_applied"`
	ValidationError        string       `json:"validation_error,omitempty"`
}

type LateralController interface {
	Steer(target FuturePoint, currentSpeedMPS float64) (float64, string)
}

type LongitudinalController interface {
	ThrottleBrake(target FuturePoint, currentSpeedMPS float64) (float64, float64, string)
}

type PurePursuitLateralController struct {
	MaxHeadingRad float64
}

type SpeedTrackingLongitudinalController struct {
	Gain     float64
	Deadband float64
}

type ControlSmoother struct {
	MaxSteerDeltaPerTick    float64
	MaxThrottleDeltaPerTick float64
	MaxBrakeDeltaPerTick    float64
}

func LegacyPredictionAdapter(command ControlCommand, inputTimestampS float64, receivedTimestampS float64, desiredSpeedMPS *float64, source string) PredictionHorizon {
	bins := HorizonDtBins(len(DefaultPredictionHorizonDtMs))
	points := make([]FuturePoint, 0, len(bins))
	for _, dt := range bins {
		point := FuturePoint{
			DtMs:     dt,
			Steer:    floatPtr(command.Steering),
			Throttle: floatPtr(command.Throttle),
			Brake:    floatPtr(command.BrakePressureAvg),
		}
		if desiredSpeedMPS != nil {
			point.DesiredSpeedMPS = floatPtr(*desiredSpeedMPS)
		}
		points = append(points, point)
	}
	return PredictionHorizon{
		InputTimestampS:    inputTimestampS,
		ReceivedTimestampS: receivedTimestampS,
		Points:             points,
		Confidence:         1.0,
		Source:             strings.TrimSpace(source),
	}
}

func HorizonDtBins(count int) []int {
	if count <= 0 {
		return nil
	}
	out := make([]int, count)
	copy(out, DefaultPredictionHorizonDtMs)
	if count <= len(DefaultPredictionHorizonDtMs) {
		return out
	}
	step := DefaultPredictionHorizonDtMs[len(DefaultPredictionHorizonDtMs)-1] - DefaultPredictionHorizonDtMs[len(DefaultPredictionHorizonDtMs)-2]
	if step <= 0 {
		step = 500
	}
	for index := len(DefaultPredictionHorizonDtMs); index < count; index++ {
		out[index] = out[index-1] + step
	}
	return out
}

func (b *TemporalPlanBuffer) Accept(plan PredictionHorizon, now time.Time) error {
	normalized, err := NormalizePredictionHorizon(plan, timeToSeconds(now))
	if err != nil {
		return err
	}
	if b.ActivePlan != nil {
		previous := clonePredictionHorizon(*b.ActivePlan)
		b.PreviousPlan = &previous
	}
	active := clonePredictionHorizon(normalized)
	b.ActivePlan = &active
	b.LastValidPlanTimeS = timeToSeconds(now)
	return nil
}

func NormalizePredictionHorizon(plan PredictionHorizon, defaultReceivedTimestampS float64) (PredictionHorizon, error) {
	if !isFinite(plan.InputTimestampS) || plan.InputTimestampS <= 0 {
		return PredictionHorizon{}, fmt.Errorf("prediction horizon input_timestamp_s must be finite and > 0")
	}
	if plan.ReceivedTimestampS <= 0 {
		plan.ReceivedTimestampS = defaultReceivedTimestampS
	}
	if !isFinite(plan.ReceivedTimestampS) || plan.ReceivedTimestampS <= 0 {
		return PredictionHorizon{}, fmt.Errorf("prediction horizon received_timestamp_s must be finite and > 0")
	}
	if plan.Confidence <= 0 {
		plan.Confidence = 1.0
	}
	if !isFinite(plan.Confidence) {
		return PredictionHorizon{}, fmt.Errorf("prediction horizon confidence must be finite")
	}
	plan.Confidence = clamp(plan.Confidence, 0, 1)
	plan.Source = strings.TrimSpace(plan.Source)
	if len(plan.Points) == 0 {
		return PredictionHorizon{}, fmt.Errorf("prediction horizon points must not be empty")
	}
	points := make([]FuturePoint, 0, len(plan.Points))
	for _, point := range plan.Points {
		if point.DtMs < 0 {
			return PredictionHorizon{}, fmt.Errorf("prediction horizon dt_ms must be >= 0")
		}
		if !futurePointHasSignal(point) {
			return PredictionHorizon{}, fmt.Errorf("prediction horizon point at dt_ms=%d has no target/control fields", point.DtMs)
		}
		if err := validateFuturePoint(point); err != nil {
			return PredictionHorizon{}, err
		}
		points = append(points, cloneFuturePoint(point))
	}
	sort.SliceStable(points, func(left, right int) bool {
		return points[left].DtMs < points[right].DtMs
	})
	for index := 1; index < len(points); index++ {
		if points[index].DtMs == points[index-1].DtMs {
			return PredictionHorizon{}, fmt.Errorf("prediction horizon duplicate dt_ms=%d", points[index].DtMs)
		}
	}
	plan.Points = points
	return plan, nil
}

func (b *TemporalPlanBuffer) Resolve(now time.Time, currentSpeedMPS float64, cfg Config) (ControlCommand, TemporalDebug) {
	nowS := timeToSeconds(now)
	return b.ActuatorTick(nowS, control.ActuatorEgoState{
		TimestampS:          nowS,
		SpeedMPS:            math.Max(currentSpeedMPS, 0),
		LastAppliedSteer:    b.LastControls.Steering,
		LastAppliedThrottle: b.LastControls.Throttle,
		LastAppliedBrake:    b.LastControls.BrakePressureAvg,
		Valid:               true,
	}, cfg)
}

func (b *TemporalPlanBuffer) ActuatorTick(nowS float64, egoState control.ActuatorEgoState, cfg Config) (ControlCommand, TemporalDebug) {
	currentSpeedMPS := math.Max(egoState.SpeedMPS, 0)
	trace := TemporalDebug{
		Enabled:                true,
		PlanState:              PlanStateMissing,
		NowS:                   nowS,
		CurrentSpeedMPS:        currentSpeedMPS,
		EgoStateTimestampS:     egoState.TimestampS,
		TelemetryValid:         egoState.Valid,
		TelemetryInvalidReason: egoState.InvalidReason,
		EstimatedLatencyMs:     durationMs(cfg.TemporalEstimatedActuationLatency),
	}
	if egoState.TimestampS > 0 && isFinite(egoState.TimestampS) && isFinite(nowS) {
		trace.TelemetryAgeMs = (nowS - egoState.TimestampS) * 1000
	}
	smoother := NewControlSmoother(cfg)

	if !egoState.Valid {
		raw := invalidTelemetryFallbackControls(b.LastControls, currentSpeedMPS)
		smoothed := smoother.Smooth(raw, raw)
		b.LastControls = smoothed
		trace.FallbackApplied = true
		trace.ControlMode = "fallback"
		trace.ValidationError = egoState.InvalidReason
		if b.ActivePlan != nil {
			plan := b.ActivePlan
			planAgeMs := ComputePlanAgeMsFromSeconds(*plan, nowS)
			trace.PlanAgeMs = planAgeMs
			trace.PlanState = ClassifyPlanAge(planAgeMs, cfg)
			trace.InputTimestampS = plan.InputTimestampS
			trace.ReceivedTimestampS = plan.ReceivedTimestampS
			trace.PlanConfidence = plan.Confidence
			trace.Source = plan.Source
		}
		fillTemporalCommandDebug(&trace, raw, smoothed)
		return smoothed, trace
	}

	if b.ActivePlan == nil {
		smoothed := smoother.Smooth(ControlCommand{}, b.LastControls)
		b.LastControls = smoothed
		trace.ControlMode = "fallback"
		fillTemporalCommandDebug(&trace, ControlCommand{}, smoothed)
		return smoothed, trace
	}

	plan := b.ActivePlan
	planAgeMs := ComputePlanAgeMsFromSeconds(*plan, nowS)
	lookaheadMs := SpeedLookaheadMs(trace.CurrentSpeedMPS)
	targetDtMs := ComputeTargetDtMs(planAgeMs, trace.EstimatedLatencyMs, lookaheadMs)
	planState := ClassifyPlanAge(planAgeMs, cfg)
	trace.PlanState = planState
	trace.PlanAgeMs = planAgeMs
	trace.LookaheadMs = lookaheadMs
	trace.TargetDtMs = targetDtMs
	trace.InputTimestampS = plan.InputTimestampS
	trace.ReceivedTimestampS = plan.ReceivedTimestampS
	trace.PlanConfidence = plan.Confidence
	trace.Source = plan.Source

	raw := ControlCommand{}
	if planState == PlanStateExpired {
		raw = expiredFallbackControls(b.LastControls, trace.CurrentSpeedMPS)
		trace.FallbackApplied = true
		trace.ControlMode = "fallback"
	} else {
		target, clamped := SamplePredictionHorizon(*plan, targetDtMs)
		trace.HorizonClamped = clamped
		targetCopy := cloneFuturePoint(target)
		trace.SelectedPoint = &targetCopy
		trace.DesiredSpeedMPS = cloneFloatPtr(target.DesiredSpeedMPS)
		trace.RequestedSteer = cloneFloatPtr(target.Steer)
		trace.RequestedThrottle = cloneFloatPtr(target.Throttle)
		trace.RequestedBrake = cloneFloatPtr(target.Brake)

		lateral := PurePursuitLateralController{MaxHeadingRad: math.Pi / 4}
		longitudinal := SpeedTrackingLongitudinalController{Gain: 0.35, Deadband: 0.05}
		raw.Steering, trace.LateralMode = lateral.Steer(target, trace.CurrentSpeedMPS)
		raw.Throttle, raw.BrakePressureAvg, trace.LongitudinalMode = longitudinal.ThrottleBrake(target, trace.CurrentSpeedMPS)
		trace.ControlMode = controlModeForTarget(target, plan.Source)
	}

	smoothingBase := b.LastControls
	if trace.FallbackApplied {
		smoothingBase = raw
	}
	smoothed := smoother.Smooth(raw, smoothingBase)
	b.LastControls = smoothed
	fillTemporalCommandDebug(&trace, raw, smoothed)
	return smoothed, trace
}

func ComputePlanAgeMs(plan PredictionHorizon, now time.Time) float64 {
	return ComputePlanAgeMsFromSeconds(plan, timeToSeconds(now))
}

func ComputePlanAgeMsFromSeconds(plan PredictionHorizon, nowS float64) float64 {
	return (nowS - plan.InputTimestampS) * 1000
}

func SpeedLookaheadMs(speedMPS float64) float64 {
	return clamp(250+(math.Max(speedMPS, 0)*25), 250, 850)
}

func ComputeTargetDtMs(planAgeMs float64, estimatedLatencyMs float64, lookaheadMs float64) float64 {
	return planAgeMs + estimatedLatencyMs + lookaheadMs
}

func ClassifyPlanAge(planAgeMs float64, cfg Config) string {
	if planAgeMs < 0 {
		planAgeMs = 0
	}
	if planAgeMs <= durationMs(cfg.TemporalFreshPlanAge) {
		return PlanStateFresh
	}
	if planAgeMs <= durationMs(cfg.TemporalUsablePlanAge) {
		return PlanStateUsable
	}
	if planAgeMs <= durationMs(cfg.TemporalStalePlanAge) {
		return PlanStateStale
	}
	return PlanStateExpired
}

func SamplePredictionHorizon(plan PredictionHorizon, targetDtMs float64) (FuturePoint, bool) {
	if len(plan.Points) == 0 {
		return FuturePoint{}, false
	}
	if targetDtMs <= float64(plan.Points[0].DtMs) {
		return cloneFuturePoint(plan.Points[0]), targetDtMs < float64(plan.Points[0].DtMs)
	}
	last := plan.Points[len(plan.Points)-1]
	if targetDtMs >= float64(last.DtMs) {
		return cloneFuturePoint(last), targetDtMs > float64(last.DtMs)
	}
	for index := 1; index < len(plan.Points); index++ {
		right := plan.Points[index]
		left := plan.Points[index-1]
		if targetDtMs <= float64(right.DtMs) {
			span := float64(right.DtMs - left.DtMs)
			if span <= 0 {
				return cloneFuturePoint(right), false
			}
			ratio := (targetDtMs - float64(left.DtMs)) / span
			return InterpolateFuturePoint(left, right, targetDtMs, ratio), false
		}
	}
	return cloneFuturePoint(last), true
}

func InterpolateFuturePoint(left FuturePoint, right FuturePoint, targetDtMs float64, ratio float64) FuturePoint {
	ratio = clamp(ratio, 0, 1)
	return FuturePoint{
		DtMs:            int(math.Round(targetDtMs)),
		X:               interpolateOptional(left.X, right.X, ratio),
		Y:               interpolateOptional(left.Y, right.Y, ratio),
		DesiredSpeedMPS: interpolateOptional(left.DesiredSpeedMPS, right.DesiredSpeedMPS, ratio),
		HeadingRad:      interpolateOptional(left.HeadingRad, right.HeadingRad, ratio),
		Steer:           interpolateOptional(left.Steer, right.Steer, ratio),
		Throttle:        interpolateOptional(left.Throttle, right.Throttle, ratio),
		Brake:           interpolateOptional(left.Brake, right.Brake, ratio),
	}
}

func (c PurePursuitLateralController) Steer(target FuturePoint, currentSpeedMPS float64) (float64, string) {
	maxHeading := c.MaxHeadingRad
	if maxHeading <= 0 {
		maxHeading = math.Pi / 4
	}
	if target.X != nil && target.Y != nil {
		forward := *target.Y
		if math.Abs(forward) < 0.1 {
			if forward < 0 {
				forward = -0.1
			} else {
				forward = 0.1
			}
		}
		heading := math.Atan2(*target.X, forward)
		return clamp(heading/maxHeading, -1, 1), "pure_pursuit"
	}
	if target.HeadingRad != nil {
		return clamp(*target.HeadingRad/maxHeading, -1, 1), "heading"
	}
	if target.Steer != nil {
		return clamp(*target.Steer, -1, 1), "direct_steer"
	}
	return 0, "neutral"
}

func (c SpeedTrackingLongitudinalController) ThrottleBrake(target FuturePoint, currentSpeedMPS float64) (float64, float64, string) {
	gain := c.Gain
	if gain <= 0 {
		gain = 0.35
	}
	deadband := c.Deadband
	if deadband < 0 {
		deadband = 0
	}
	if target.DesiredSpeedMPS != nil {
		speedError := *target.DesiredSpeedMPS - math.Max(currentSpeedMPS, 0)
		if math.Abs(speedError) <= deadband {
			return 0, 0, "desired_speed_deadband"
		}
		if speedError > 0 {
			return clamp(speedError*gain, 0, 1), 0, "desired_speed_throttle"
		}
		return 0, clamp(-speedError*gain, 0, 1), "desired_speed_brake"
	}
	throttle := 0.0
	brake := 0.0
	if target.Throttle != nil {
		throttle = *target.Throttle
	}
	if target.Brake != nil {
		brake = *target.Brake
	}
	return clamp(throttle, 0, 1), clamp(brake, 0, 1), "direct_control"
}

func NewControlSmoother(cfg Config) ControlSmoother {
	return ControlSmoother{
		MaxSteerDeltaPerTick:    cfg.TemporalSteeringMaxDelta,
		MaxThrottleDeltaPerTick: cfg.TemporalThrottleMaxDelta,
		MaxBrakeDeltaPerTick:    cfg.TemporalBrakeMaxDelta,
	}
}

func (s ControlSmoother) Smooth(raw ControlCommand, last ControlCommand) ControlCommand {
	target := ControlCommand{
		Steering:         clamp(raw.Steering, -1, 1),
		Throttle:         clamp(raw.Throttle, 0, 1),
		BrakePressureAvg: clamp(raw.BrakePressureAvg, 0, 1),
	}
	target = resolveTemporalThrottleBrakeExclusivity(target)
	smoothed := ControlCommand{
		Steering:         RateLimit(clamp(last.Steering, -1, 1), target.Steering, s.MaxSteerDeltaPerTick),
		Throttle:         RateLimit(clamp(last.Throttle, 0, 1), target.Throttle, s.MaxThrottleDeltaPerTick),
		BrakePressureAvg: RateLimit(clamp(last.BrakePressureAvg, 0, 1), target.BrakePressureAvg, s.MaxBrakeDeltaPerTick),
	}
	return resolveTemporalThrottleBrakeExclusivity(smoothed)
}

func LogTemporalDebug(trace TemporalDebug) {
	log.Printf("[actuator.temporal] now_s=%.3f ego_ts=%.3f telemetry_age_ms=%.1f telemetry_valid=%t telemetry_invalid_reason=%q input_ts=%.3f received_ts=%.3f state=%s plan_age_ms=%.1f target_dt_ms=%.1f lookahead_ms=%.1f latency_ms=%.1f speed_mps=%.3f selected=%s desired_speed_mps=%s requested steer=%s throttle=%s brake=%s raw steer=%.3f throttle=%.3f brake=%.3f smoothed steer=%.3f throttle=%.3f brake=%.3f applied steer=%.3f throttle=%.3f brake=%.3f confidence=%.3f source=%s fallback=%t mode=%s",
		trace.NowS,
		trace.EgoStateTimestampS,
		trace.TelemetryAgeMs,
		trace.TelemetryValid,
		trace.TelemetryInvalidReason,
		trace.InputTimestampS,
		trace.ReceivedTimestampS,
		trace.PlanState,
		trace.PlanAgeMs,
		trace.TargetDtMs,
		trace.LookaheadMs,
		trace.EstimatedLatencyMs,
		trace.CurrentSpeedMPS,
		formatFuturePoint(trace.SelectedPoint),
		formatOptionalFloat(trace.DesiredSpeedMPS),
		formatOptionalFloat(trace.RequestedSteer),
		formatOptionalFloat(trace.RequestedThrottle),
		formatOptionalFloat(trace.RequestedBrake),
		trace.RawSteer,
		trace.RawThrottle,
		trace.RawBrake,
		trace.SmoothedSteer,
		trace.SmoothedThrottle,
		trace.SmoothedBrake,
		trace.AppliedSteer,
		trace.AppliedThrottle,
		trace.AppliedBrake,
		trace.PlanConfidence,
		trace.Source,
		trace.FallbackApplied,
		trace.ControlMode,
	)
}

func expiredFallbackControls(last ControlCommand, currentSpeedMPS float64) ControlCommand {
	brake := last.BrakePressureAvg * 0.5
	if currentSpeedMPS >= 8.0 && brake < 0.15 {
		brake = 0.15
	}
	return ControlCommand{
		Steering:         clamp(last.Steering*0.5, -0.25, 0.25),
		Throttle:         clamp(math.Min(last.Throttle*0.3, 0.20), 0, 1),
		BrakePressureAvg: clamp(brake, 0, 1),
	}
}

func invalidTelemetryFallbackControls(last ControlCommand, currentSpeedMPS float64) ControlCommand {
	return expiredFallbackControls(last, currentSpeedMPS)
}

func controlModeForTarget(target FuturePoint, source string) string {
	if target.X != nil && target.Y != nil {
		return "waypoint_tracking"
	}
	if strings.Contains(strings.ToLower(source), "legacy") {
		return "legacy"
	}
	if target.Steer != nil || target.Throttle != nil || target.Brake != nil {
		return "control_hint"
	}
	return "fallback"
}

func resolveTemporalThrottleBrakeExclusivity(input ControlCommand) ControlCommand {
	output := input
	if output.BrakePressureAvg <= 0.05 || output.Throttle <= 0.05 {
		return output
	}
	output.Throttle = 0
	return output
}

func fillTemporalCommandDebug(trace *TemporalDebug, raw ControlCommand, smoothed ControlCommand) {
	trace.RawSteer = raw.Steering
	trace.RawThrottle = raw.Throttle
	trace.RawBrake = raw.BrakePressureAvg
	trace.SmoothedSteer = smoothed.Steering
	trace.SmoothedThrottle = smoothed.Throttle
	trace.SmoothedBrake = smoothed.BrakePressureAvg
	trace.AppliedSteer = smoothed.Steering
	trace.AppliedThrottle = smoothed.Throttle
	trace.AppliedBrake = smoothed.BrakePressureAvg
}

func formatFuturePoint(point *FuturePoint) string {
	if point == nil {
		return "none"
	}
	return fmt.Sprintf("dt_ms=%d x=%s y=%s desired_speed=%s heading=%s steer=%s throttle=%s brake=%s",
		point.DtMs,
		formatOptionalFloat(point.X),
		formatOptionalFloat(point.Y),
		formatOptionalFloat(point.DesiredSpeedMPS),
		formatOptionalFloat(point.HeadingRad),
		formatOptionalFloat(point.Steer),
		formatOptionalFloat(point.Throttle),
		formatOptionalFloat(point.Brake),
	)
}

func formatOptionalFloat(value *float64) string {
	if value == nil {
		return "nil"
	}
	return fmt.Sprintf("%.3f", *value)
}

func futurePointHasSignal(point FuturePoint) bool {
	return point.X != nil ||
		point.Y != nil ||
		point.DesiredSpeedMPS != nil ||
		point.HeadingRad != nil ||
		point.Steer != nil ||
		point.Throttle != nil ||
		point.Brake != nil
}

func validateFuturePoint(point FuturePoint) error {
	for name, value := range map[string]*float64{
		"x":                 point.X,
		"y":                 point.Y,
		"desired_speed_mps": point.DesiredSpeedMPS,
		"heading_rad":       point.HeadingRad,
		"steer":             point.Steer,
		"throttle":          point.Throttle,
		"brake":             point.Brake,
	} {
		if value != nil && !isFinite(*value) {
			return fmt.Errorf("prediction horizon %s at dt_ms=%d must be finite", name, point.DtMs)
		}
	}
	return nil
}

func interpolateOptional(left *float64, right *float64, ratio float64) *float64 {
	if left == nil || right == nil {
		return nil
	}
	value := *left + ((*right - *left) * ratio)
	return &value
}

func clonePredictionHorizon(plan PredictionHorizon) PredictionHorizon {
	out := plan
	out.Points = make([]FuturePoint, 0, len(plan.Points))
	for _, point := range plan.Points {
		out.Points = append(out.Points, cloneFuturePoint(point))
	}
	return out
}

func cloneFuturePoint(point FuturePoint) FuturePoint {
	return FuturePoint{
		DtMs:            point.DtMs,
		X:               cloneFloatPtr(point.X),
		Y:               cloneFloatPtr(point.Y),
		DesiredSpeedMPS: cloneFloatPtr(point.DesiredSpeedMPS),
		HeadingRad:      cloneFloatPtr(point.HeadingRad),
		Steer:           cloneFloatPtr(point.Steer),
		Throttle:        cloneFloatPtr(point.Throttle),
		Brake:           cloneFloatPtr(point.Brake),
	}
}

func cloneFloatPtr(value *float64) *float64 {
	if value == nil {
		return nil
	}
	out := *value
	return &out
}

func floatPtr(value float64) *float64 {
	return &value
}

func timeToSeconds(value time.Time) float64 {
	return float64(value.UTC().UnixNano()) / float64(time.Second)
}

func durationMs(value time.Duration) float64 {
	return float64(value) / float64(time.Millisecond)
}

func isFinite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}
