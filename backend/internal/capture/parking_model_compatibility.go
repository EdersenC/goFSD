package capture

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"awesomeProject/internal/parkingcontrol"
)

var ErrParkingModelIncompatible = errors.New("model is not compatible with stop-sign inference")

const requiredParkingPlannerFormatVersion = 1

// Phase 1 intentionally provides no scalar state vector. The model must locate
// the orange bay marker from RGB; measured speed arrives through temporal telemetry.
var requiredParkingStateInputs = []string{}

var requiredParkingControlHeads = []string{
	"future_speed_mps",
	"stop_intent",
}

type parkingControlContract struct {
	Name              string               `json:"name"`
	Version           int                  `json:"version"`
	Direction         string               `json:"direction"`
	Targets           []string             `json:"targets"`
	OutputActivations map[string]string    `json:"output_activations"`
	OutputRanges      map[string][]float64 `json:"output_ranges"`
}

type parkingModelStatus struct {
	Loaded                    bool                             `json:"loaded"`
	Checkpoint                string                           `json:"checkpoint"`
	Device                    string                           `json:"device"`
	PlannerFormat             string                           `json:"planner_format"`
	PlannerFormatVersion      int                              `json:"planner_format_version"`
	ControlContract           parkingControlContract           `json:"control_contract"`
	ControlHorizonDtMs        []int                            `json:"control_horizon_dt_ms"`
	TelemetrySampleIntervalMs int                              `json:"telemetry_sample_interval_ms"`
	Direction                 string                           `json:"direction"`
	ImageSize                 parkingModelImageSize            `json:"image_size"`
	FrameWindow               parkingModelFrameWindow          `json:"frame_window"`
	ImageOffsets              []int                            `json:"image_offsets"`
	TelemetryOffsets          []int                            `json:"telemetry_offsets"`
	FutureOffsets             []int                            `json:"future_offsets"`
	TelemetryFeatures         []string                         `json:"telemetry_feature_names"`
	ControlTargetNames        []string                         `json:"control_target_names"`
	AuxTargetNames            []string                         `json:"aux_target_names"`
	StateInputs               map[string]parkingModelInputSpec `json:"state_inputs"`
}

type parkingModelImageSize struct {
	Width  int `json:"width"`
	Height int `json:"height"`
}

type parkingModelFrameWindow struct {
	Size          int `json:"size"`
	FrameStride   int `json:"frame_stride"`
	InputChannels int `json:"input_channels"`
}

type parkingModelInputSpec struct {
	Enabled bool `json:"enabled"`
}

func (i *Inferencer) validateLoadedParkingModel(ctx context.Context, modelServerURL string) (parkingModelStatus, error) {
	status, err := i.fetchParkingModelStatus(ctx, modelServerURL)
	if err != nil {
		return parkingModelStatus{}, err
	}
	if err := validateParkingModelStatus(status, i.config); err != nil {
		return parkingModelStatus{}, err
	}
	return status, nil
}

func (i *Inferencer) fetchParkingModelStatus(ctx context.Context, modelServerURL string) (parkingModelStatus, error) {
	requestCtx, cancel := context.WithTimeout(ctx, i.requestTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(requestCtx, http.MethodGet, modelServerURL+"/model", nil)
	if err != nil {
		return parkingModelStatus{}, err
	}
	resp, err := i.httpClient.Do(req)
	if err != nil {
		return parkingModelStatus{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		payload, _ := io.ReadAll(resp.Body)
		return parkingModelStatus{}, fmt.Errorf("model status failed: status=%d body=%s", resp.StatusCode, strings.TrimSpace(string(payload)))
	}

	var status parkingModelStatus
	if err := json.NewDecoder(resp.Body).Decode(&status); err != nil {
		return parkingModelStatus{}, err
	}
	return status, nil
}

func (i *Inferencer) validateParkingPredictionModel(prediction pythonPredictResponse) error {
	i.mu.Lock()
	expectedCheckpoint := strings.TrimSpace(i.parkingCheckpoint)
	i.mu.Unlock()
	checkpoint := strings.TrimSpace(prediction.Checkpoint)
	if expectedCheckpoint == "" || checkpoint != expectedCheckpoint {
		return parkingModelCompatibilityError(
			"prediction checkpoint %q does not match session checkpoint %q",
			checkpoint,
			expectedCheckpoint,
		)
	}
	if strings.TrimSpace(prediction.PlannerFormat) != strings.TrimSpace(i.config.PlannerFormat) {
		return parkingModelCompatibilityError("prediction planner format changed while inference was active")
	}
	if prediction.PlannerFormatVersion != requiredParkingPlannerFormatVersion {
		return parkingModelCompatibilityError("prediction planner format version must be %d, got=%d", requiredParkingPlannerFormatVersion, prediction.PlannerFormatVersion)
	}
	if err := validateParkingControlContract(prediction.ControlContract); err != nil {
		return err
	}
	if strings.TrimSpace(prediction.Direction) != parkingcontrol.ParkingDirectionForward {
		return parkingModelCompatibilityError("prediction direction must be %q", parkingcontrol.ParkingDirectionForward)
	}
	if prediction.TelemetrySampleIntervalMs != int(i.config.TelemetrySampleInterval.Milliseconds()) {
		return parkingModelCompatibilityError("prediction telemetry sample interval must be %dms, got=%dms", i.config.TelemetrySampleInterval.Milliseconds(), prediction.TelemetrySampleIntervalMs)
	}
	if err := validateOrderedInts("prediction control horizon timing", prediction.ControlHorizonDtMs, i.config.ControlHorizonDtMs); err != nil {
		return err
	}
	if err := validateOrderedStrings("prediction control heads", prediction.ControlTargetNames, i.config.ControlOutputNames); err != nil {
		return err
	}
	if err := validateOrderedStrings("prediction auxiliary heads", prediction.AuxTargetNames, i.config.AuxOutputNames); err != nil {
		return err
	}
	if err := validateOrderedInts("prediction image offsets", prediction.ImageOffsets, i.config.ImageOffsets); err != nil {
		return err
	}
	if err := validateOrderedInts("prediction telemetry offsets", prediction.TelemetryOffsets, i.config.TelemetryOffsets); err != nil {
		return err
	}
	if err := validateOrderedInts("prediction future offsets", prediction.FutureOffsets, i.config.FutureOffsets); err != nil {
		return err
	}
	if err := validateOrderedStrings("prediction telemetry features", prediction.TelemetryFeatureNames, i.config.TelemetryFeatureNames); err != nil {
		return err
	}
	if err := validateEnabledParkingStateInputs(prediction.StateInputs); err != nil {
		return err
	}
	return nil
}

func validateParkingModelStatus(status parkingModelStatus, expected InferenceConfig) error {
	if !status.Loaded || strings.TrimSpace(status.Checkpoint) == "" {
		return parkingModelCompatibilityError("load a parking checkpoint before starting inference")
	}
	if plannerFormat := strings.TrimSpace(status.PlannerFormat); plannerFormat != strings.TrimSpace(expected.PlannerFormat) {
		return parkingModelCompatibilityError(
			"planner format %q does not match required format %q",
			plannerFormat,
			strings.TrimSpace(expected.PlannerFormat),
		)
	}
	if status.PlannerFormatVersion != requiredParkingPlannerFormatVersion {
		return parkingModelCompatibilityError("checkpoint planner format version must be %d, got=%d", requiredParkingPlannerFormatVersion, status.PlannerFormatVersion)
	}
	if err := validateParkingControlContract(status.ControlContract); err != nil {
		return err
	}
	if strings.TrimSpace(status.Direction) != parkingcontrol.ParkingDirectionForward {
		return parkingModelCompatibilityError("checkpoint direction must be %q", parkingcontrol.ParkingDirectionForward)
	}
	if status.TelemetrySampleIntervalMs != int(expected.TelemetrySampleInterval.Milliseconds()) {
		return parkingModelCompatibilityError("checkpoint telemetry sample interval must be %dms, got=%dms", expected.TelemetrySampleInterval.Milliseconds(), status.TelemetrySampleIntervalMs)
	}
	if err := validateOrderedInts("checkpoint control horizon timing", status.ControlHorizonDtMs, expected.ControlHorizonDtMs); err != nil {
		return err
	}
	if status.ImageSize.Width != expected.FrameWidth || status.ImageSize.Height != expected.FrameHeight {
		return parkingModelCompatibilityError(
			"checkpoint image size %dx%d does not match required size %dx%d",
			status.ImageSize.Width,
			status.ImageSize.Height,
			expected.FrameWidth,
			expected.FrameHeight,
		)
	}
	if status.FrameWindow.Size != expected.WindowSize ||
		status.FrameWindow.FrameStride != expected.FrameStride ||
		status.FrameWindow.InputChannels != expected.WindowSize*3 {
		return parkingModelCompatibilityError(
			"checkpoint frame window size=%d stride=%d channels=%d does not match required size=%d stride=%d channels=%d",
			status.FrameWindow.Size,
			status.FrameWindow.FrameStride,
			status.FrameWindow.InputChannels,
			expected.WindowSize,
			expected.FrameStride,
			expected.WindowSize*3,
		)
	}
	if err := validateOrderedInts("checkpoint image offsets", status.ImageOffsets, expected.ImageOffsets); err != nil {
		return err
	}
	if err := validateOrderedInts("checkpoint telemetry offsets", status.TelemetryOffsets, expected.TelemetryOffsets); err != nil {
		return err
	}
	if err := validateOrderedInts("checkpoint future offsets", status.FutureOffsets, expected.FutureOffsets); err != nil {
		return err
	}
	if err := validateOrderedStrings("checkpoint telemetry features", status.TelemetryFeatures, expected.TelemetryFeatureNames); err != nil {
		return err
	}
	if err := validateOrderedStrings("checkpoint control heads", status.ControlTargetNames, expected.ControlOutputNames); err != nil {
		return err
	}
	if err := validateOrderedStrings("checkpoint auxiliary heads", status.AuxTargetNames, expected.AuxOutputNames); err != nil {
		return err
	}
	if err := validateEnabledParkingStateInputs(status.StateInputs); err != nil {
		return err
	}
	return nil
}

func validateEnabledParkingStateInputs(inputs map[string]parkingModelInputSpec) error {
	expected := make(map[string]struct{}, len(requiredParkingStateInputs))
	for _, name := range requiredParkingStateInputs {
		expected[name] = struct{}{}
		spec, ok := inputs[name]
		if !ok || !spec.Enabled {
			return parkingModelCompatibilityError("checkpoint must enable state input %q", name)
		}
	}
	for name, spec := range inputs {
		if !spec.Enabled {
			continue
		}
		if _, ok := expected[name]; !ok {
			return parkingModelCompatibilityError("checkpoint enables unexpected state input %q", name)
		}
	}
	return nil
}

func validateParkingControlContract(contract parkingControlContract) error {
	if strings.TrimSpace(contract.Name) != parkingcontrol.ParkingSetpointContractV1 {
		return parkingModelCompatibilityError(
			"control contract %q is incompatible; retrain with %q",
			strings.TrimSpace(contract.Name),
			parkingcontrol.ParkingSetpointContractV1,
		)
	}
	if contract.Version != 1 {
		return parkingModelCompatibilityError("control contract version must be 1, got=%d", contract.Version)
	}
	if strings.TrimSpace(contract.Direction) != parkingcontrol.ParkingDirectionForward {
		return parkingModelCompatibilityError("control contract direction must be %q", parkingcontrol.ParkingDirectionForward)
	}
	if err := validateOrderedStrings("control contract targets", contract.Targets, requiredParkingControlHeads); err != nil {
		return err
	}
	expectedActivations := map[string]string{
		"future_speed_mps": "sigmoid",
		"stop_intent":      "sigmoid",
	}
	for name, expected := range expectedActivations {
		if strings.ToLower(strings.TrimSpace(contract.OutputActivations[name])) != expected {
			return parkingModelCompatibilityError("control contract activation for %s must be %s", name, expected)
		}
	}
	if len(contract.OutputActivations) != len(expectedActivations) {
		return parkingModelCompatibilityError("control contract output activations differ: got=%v", contract.OutputActivations)
	}
	expectedRanges := map[string][2]float64{
		"future_speed_mps": {0, parkingcontrol.StopSignMotionPlanMaxSpeedMPS},
		"stop_intent":      {0, 1},
	}
	for name, expected := range expectedRanges {
		actual := contract.OutputRanges[name]
		if len(actual) != 2 || actual[0] != expected[0] || actual[1] != expected[1] {
			return parkingModelCompatibilityError("control contract range for %s must be [%g, %g]", name, expected[0], expected[1])
		}
	}
	if len(contract.OutputRanges) != len(expectedRanges) {
		return parkingModelCompatibilityError("control contract output ranges differ: got=%v", contract.OutputRanges)
	}
	return nil
}

func cloneParkingControlContract(source parkingControlContract) parkingControlContract {
	out := source
	out.Targets = append([]string(nil), source.Targets...)
	out.OutputActivations = make(map[string]string, len(source.OutputActivations))
	for key, value := range source.OutputActivations {
		out.OutputActivations[key] = value
	}
	out.OutputRanges = make(map[string][]float64, len(source.OutputRanges))
	for key, value := range source.OutputRanges {
		out.OutputRanges[key] = append([]float64(nil), value...)
	}
	return out
}

func validateOrderedStrings(label string, actual, expected []string) error {
	if len(actual) != len(expected) {
		return parkingModelCompatibilityError("%s differ: got=%v want=%v", label, actual, expected)
	}
	for index := range expected {
		if strings.TrimSpace(actual[index]) != strings.TrimSpace(expected[index]) {
			return parkingModelCompatibilityError("%s differ: got=%v want=%v", label, actual, expected)
		}
	}
	return nil
}

func validateOrderedInts(label string, actual, expected []int) error {
	if len(actual) != len(expected) {
		return parkingModelCompatibilityError("%s differ: got=%v want=%v", label, actual, expected)
	}
	for index := range expected {
		if actual[index] != expected[index] {
			return parkingModelCompatibilityError("%s differ: got=%v want=%v", label, actual, expected)
		}
	}
	return nil
}

func parkingModelCompatibilityError(format string, args ...any) error {
	return fmt.Errorf("%w: %s", ErrParkingModelIncompatible, fmt.Sprintf(format, args...))
}
