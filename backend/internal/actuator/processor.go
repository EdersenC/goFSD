package actuator

import "time"

type ControlCommand struct {
	Steering         float64 `json:"steering"`
	Throttle         float64 `json:"throttle"`
	BrakePressureAvg float64 `json:"brakePressureAvg"`
}

type CommandRange struct {
	MinSteering float64 `json:"minSteering"`
	MaxSteering float64 `json:"maxSteering"`
	MinThrottle float64 `json:"minThrottle"`
	MaxThrottle float64 `json:"maxThrottle"`
	MinBrake    float64 `json:"minBrakePressureAvg"`
	MaxBrake    float64 `json:"maxBrakePressureAvg"`
}

type ProcessorStats struct {
	ClampedCommands     int `json:"clampedCommands"`
	DeadzoneCommands    int `json:"deadzoneCommands"`
	RateLimitedCommands int `json:"rateLimitedCommands"`
	SmoothedCommands    int `json:"smoothedCommands"`
	FallbackCommands    int `json:"fallbackCommands"`
	HistoryWindow       int `json:"historyWindow"`
}

type ProcessorState struct {
	PrevSteering  float64        `json:"prevSteering"`
	PrevThrottle  float64        `json:"prevThrottle"`
	PrevBrake     float64        `json:"prevBrakePressureAvg"`
	LastCommandAt time.Time      `json:"-"`
	RecentRange   CommandRange   `json:"recentRange"`
	Stats         ProcessorStats `json:"stats"`
	history       []ControlCommand
}

type ProcessingDebug struct {
	Raw              ControlCommand `json:"raw"`
	Clamped          ControlCommand `json:"clamped"`
	AfterDeadzone    ControlCommand `json:"afterDeadzone"`
	AfterConflict    ControlCommand `json:"afterConflictResolution"`
	AfterRateLimit   ControlCommand `json:"afterRateLimit"`
	Final            ControlCommand `json:"final"`
	ClampApplied     bool           `json:"clampApplied"`
	DeadzoneApplied  bool           `json:"deadzoneApplied"`
	ConflictApplied  bool           `json:"conflictApplied"`
	RateLimitApplied bool           `json:"rateLimitApplied"`
	SmoothingApplied bool           `json:"smoothingApplied"`
	FallbackApplied  bool           `json:"fallbackApplied"`
}

func Clamp(x, lo, hi float64) float64 {
	if x < lo {
		return lo
	}
	if x > hi {
		return hi
	}
	return x
}

func ApplyDeadzone(x float64, deadzone float64) float64 {
	if deadzone <= 0 {
		return x
	}
	if x > -deadzone && x < deadzone {
		return 0
	}
	return x
}

func RateLimit(prev, target, maxDelta float64) float64 {
	if maxDelta <= 0 {
		return target
	}
	if target > prev+maxDelta {
		return prev + maxDelta
	}
	if target < prev-maxDelta {
		return prev - maxDelta
	}
	return target
}

func Lerp(prev, target, alpha float64) float64 {
	if alpha <= 0 {
		return prev
	}
	if alpha >= 1 {
		return target
	}
	return prev + ((target - prev) * alpha)
}

func ProcessActuatorCommand(raw ControlCommand, state *ProcessorState, cfg Config) (ControlCommand, ProcessingDebug) {
	debug := ProcessingDebug{Raw: raw}
	if state == nil {
		state = &ProcessorState{}
	}

	clamped := ControlCommand{
		Steering:         Clamp(raw.Steering, -1.0, 1.0),
		Throttle:         Clamp(raw.Throttle, 0.0, 1.0),
		BrakePressureAvg: Clamp(raw.BrakePressureAvg, 0.0, 1.0),
	}
	debug.Clamped = clamped
	debug.ClampApplied = clamped != raw

	deadzoned := clamped
	if cfg.Profile != "debug_raw" {
		deadzoned = ControlCommand{
			Steering:         ApplyDeadzone(clamped.Steering, cfg.SteeringDeadzone),
			Throttle:         ApplyDeadzone(clamped.Throttle, cfg.ThrottleDeadzone),
			BrakePressureAvg: ApplyDeadzone(clamped.BrakePressureAvg, cfg.BrakeDeadzone),
		}
	}
	debug.AfterDeadzone = deadzoned
	debug.DeadzoneApplied = deadzoned != clamped

	debug.AfterConflict = deadzoned
	debug.ConflictApplied = false

	rateLimited := deadzoned
	if cfg.Profile != "debug_raw" {
		rateLimited = ControlCommand{
			Steering:         RateLimit(state.PrevSteering, deadzoned.Steering, cfg.SteeringMaxDelta),
			Throttle:         RateLimit(state.PrevThrottle, deadzoned.Throttle, cfg.ThrottleMaxDelta),
			BrakePressureAvg: RateLimit(state.PrevBrake, deadzoned.BrakePressureAvg, cfg.BrakeMaxDelta),
		}
	}
	debug.AfterRateLimit = rateLimited
	debug.RateLimitApplied = rateLimited != deadzoned

	smoothed := rateLimited
	if cfg.Profile != "debug_raw" {
		smoothed = ControlCommand{
			Steering:         Lerp(state.PrevSteering, rateLimited.Steering, cfg.SteeringAlpha),
			Throttle:         Lerp(state.PrevThrottle, rateLimited.Throttle, cfg.ThrottleAlpha),
			BrakePressureAvg: Lerp(state.PrevBrake, rateLimited.BrakePressureAvg, cfg.BrakeAlpha),
		}
	}
	debug.SmoothingApplied = smoothed != rateLimited

	final := ControlCommand{
		Steering:         Clamp(smoothed.Steering, -1.0, 1.0),
		Throttle:         Clamp(smoothed.Throttle, 0.0, 1.0),
		BrakePressureAvg: Clamp(smoothed.BrakePressureAvg, 0.0, 1.0),
	}
	debug.Final = final
	recordProcessorResult(state, final, debug, cfg)
	return final, debug
}

func ApplyFallbackDecay(state *ProcessorState, cfg Config) (ControlCommand, ProcessingDebug) {
	if state == nil {
		state = &ProcessorState{}
	}
	raw := ControlCommand{
		Steering:         state.PrevSteering,
		Throttle:         state.PrevThrottle,
		BrakePressureAvg: state.PrevBrake,
	}
	if cfg.FallbackDecayEnabled {
		raw.Steering *= cfg.FallbackSteeringDecay
		raw.Throttle *= cfg.FallbackThrottleDecay
		raw.BrakePressureAvg *= cfg.FallbackBrakeDecay
	}
	final, debug := ProcessActuatorCommand(raw, state, cfg)
	debug.FallbackApplied = true
	state.Stats.FallbackCommands++
	return final, debug
}

func recordProcessorResult(state *ProcessorState, final ControlCommand, debug ProcessingDebug, cfg Config) {
	if debug.ClampApplied {
		state.Stats.ClampedCommands++
	}
	if debug.DeadzoneApplied {
		state.Stats.DeadzoneCommands++
	}
	if debug.RateLimitApplied {
		state.Stats.RateLimitedCommands++
	}
	if debug.SmoothingApplied {
		state.Stats.SmoothedCommands++
	}
	state.Stats.HistoryWindow = cfg.RecentCommandWindow
	state.PrevSteering = final.Steering
	state.PrevThrottle = final.Throttle
	state.PrevBrake = final.BrakePressureAvg
	state.LastCommandAt = time.Now().UTC()
	state.history = append(state.history, final)
	if len(state.history) > cfg.RecentCommandWindow {
		state.history = append([]ControlCommand(nil), state.history[len(state.history)-cfg.RecentCommandWindow:]...)
	}
	state.RecentRange = computeRecentRange(state.history)
}

func computeRecentRange(history []ControlCommand) CommandRange {
	if len(history) == 0 {
		return CommandRange{}
	}
	out := CommandRange{
		MinSteering: history[0].Steering,
		MaxSteering: history[0].Steering,
		MinThrottle: history[0].Throttle,
		MaxThrottle: history[0].Throttle,
		MinBrake:    history[0].BrakePressureAvg,
		MaxBrake:    history[0].BrakePressureAvg,
	}
	for _, item := range history[1:] {
		if item.Steering < out.MinSteering {
			out.MinSteering = item.Steering
		}
		if item.Steering > out.MaxSteering {
			out.MaxSteering = item.Steering
		}
		if item.Throttle < out.MinThrottle {
			out.MinThrottle = item.Throttle
		}
		if item.Throttle > out.MaxThrottle {
			out.MaxThrottle = item.Throttle
		}
		if item.BrakePressureAvg < out.MinBrake {
			out.MinBrake = item.BrakePressureAvg
		}
		if item.BrakePressureAvg > out.MaxBrake {
			out.MaxBrake = item.BrakePressureAvg
		}
	}
	return out
}
