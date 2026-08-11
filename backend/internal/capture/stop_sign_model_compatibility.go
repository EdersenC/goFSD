package capture

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"awesomeProject/internal/stopsigncontrol"
)

var ErrStopSignModelIncompatible = errors.New("model is not compatible with stop-sign inference")

const requiredStopSignPlannerFormatVersion = 1

// Phase 1 intentionally provides no scalar state vector. The model must locate
// the orange bay marker from RGB; measured speed arrives through temporal telemetry.
var requiredStopSignStateInputs = []string{}

var requiredStopSignControlHeads = []string{
	"future_speed_mps",
	"stop_intent",
}

type stopSignControlContract struct {
	Name              string               `json:"name"`
	Version           int                  `json:"version"`
	Direction         string               `json:"direction"`
	Targets           []string             `json:"targets"`
	OutputActivations map[string]string    `json:"output_activations"`
	OutputRanges      map[string][]float64 `json:"output_ranges"`
}

type stopSignModelStatus struct {
	Loaded                    bool                              `json:"loaded"`
	Checkpoint                string                            `json:"checkpoint"`
	Device                    string                            `json:"device"`
	PlannerFormat             string                            `json:"planner_format"`
	PlannerFormatVersion      int                               `json:"planner_format_version"`
	ControlContract           stopSignControlContract           `json:"control_contract"`
	ControlHorizonDtMs        []int                             `json:"control_horizon_dt_ms"`
	TelemetrySampleIntervalMs int                               `json:"telemetry_sample_interval_ms"`
	Direction                 string                            `json:"direction"`
	ImageSize                 stopSignModelImageSize            `json:"image_size"`
	FrameWindow               stopSignModelFrameWindow          `json:"frame_window"`
	ImageOffsets              []int                             `json:"image_offsets"`
	TelemetryOffsets          []int                             `json:"telemetry_offsets"`
	FutureOffsets             []int                             `json:"future_offsets"`
	TelemetryFeatures         []string                          `json:"telemetry_feature_names"`
	ControlTargetNames        []string                          `json:"control_target_names"`
	AuxTargetNames            []string                          `json:"aux_target_names"`
	StateInputs               map[string]stopSignModelInputSpec `json:"state_inputs"`
}

type stopSignModelImageSize struct {
	Width  int `json:"width"`
	Height int `json:"height"`
}

type stopSignModelFrameWindow struct {
	Size          int `json:"size"`
	FrameStride   int `json:"frame_stride"`
	InputChannels int `json:"input_channels"`
}

type stopSignModelInputSpec struct {
	Enabled bool `json:"enabled"`
}

func (i *Inferencer) validateLoadedStopSignModel(ctx context.Context, modelServerURL string) (stopSignModelStatus, error) {
	status, err := i.fetchStopSignModelStatus(ctx, modelServerURL)
	if err != nil {
		return stopSignModelStatus{}, err
	}
	if err := validateStopSignModelStatus(status, i.config); err != nil {
		return stopSignModelStatus{}, err
	}
	return status, nil
}

func (i *Inferencer) fetchStopSignModelStatus(ctx context.Context, modelServerURL string) (stopSignModelStatus, error) {
	requestCtx, cancel := context.WithTimeout(ctx, i.requestTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(requestCtx, http.MethodGet, modelServerURL+"/model", nil)
	if err != nil {
		return stopSignModelStatus{}, err
	}
	resp, err := i.httpClient.Do(req)
	if err != nil {
		return stopSignModelStatus{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		payload, _ := io.ReadAll(resp.Body)
		return stopSignModelStatus{}, fmt.Errorf("model status failed: status=%d body=%s", resp.StatusCode, strings.TrimSpace(string(payload)))
	}

	var status stopSignModelStatus
	if err := json.NewDecoder(resp.Body).Decode(&status); err != nil {
		return stopSignModelStatus{}, err
	}
	return status, nil
}

func (i *Inferencer) validateStopSignPredictionModel(prediction pythonPredictResponse) error {
	i.mu.Lock()
	expectedCheckpoint := strings.TrimSpace(i.stopSignCheckpoint)
	i.mu.Unlock()
	checkpoint := strings.TrimSpace(prediction.Checkpoint)
	if expectedCheckpoint == "" || checkpoint != expectedCheckpoint {
		return stopSignModelCompatibilityError(
			"prediction checkpoint %q does not match session checkpoint %q",
			checkpoint,
			expectedCheckpoint,
		)
	}
	if strings.TrimSpace(prediction.PlannerFormat) != strings.TrimSpace(i.config.PlannerFormat) {
		return stopSignModelCompatibilityError("prediction planner format changed while inference was active")
	}
	if prediction.PlannerFormatVersion != requiredStopSignPlannerFormatVersion {
		return stopSignModelCompatibilityError("prediction planner format version must be %d, got=%d", requiredStopSignPlannerFormatVersion, prediction.PlannerFormatVersion)
	}
	if err := validateStopSignControlContract(prediction.ControlContract); err != nil {
		return err
	}
	if strings.TrimSpace(prediction.Direction) != stopsigncontrol.StopSignDirectionForward {
		return stopSignModelCompatibilityError("prediction direction must be %q", stopsigncontrol.StopSignDirectionForward)
	}
	if prediction.TelemetrySampleIntervalMs != int(i.config.TelemetrySampleInterval.Milliseconds()) {
		return stopSignModelCompatibilityError("prediction telemetry sample interval must be %dms, got=%dms", i.config.TelemetrySampleInterval.Milliseconds(), prediction.TelemetrySampleIntervalMs)
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
	if err := validateEnabledStopSignStateInputs(prediction.StateInputs); err != nil {
		return err
	}
	return nil
}

func validateStopSignModelStatus(status stopSignModelStatus, expected InferenceConfig) error {
	if !status.Loaded || strings.TrimSpace(status.Checkpoint) == "" {
		return stopSignModelCompatibilityError("load a stopSign checkpoint before starting inference")
	}
	if plannerFormat := strings.TrimSpace(status.PlannerFormat); plannerFormat != strings.TrimSpace(expected.PlannerFormat) {
		return stopSignModelCompatibilityError(
			"planner format %q does not match required format %q",
			plannerFormat,
			strings.TrimSpace(expected.PlannerFormat),
		)
	}
	if status.PlannerFormatVersion != requiredStopSignPlannerFormatVersion {
		return stopSignModelCompatibilityError("checkpoint planner format version must be %d, got=%d", requiredStopSignPlannerFormatVersion, status.PlannerFormatVersion)
	}
	if err := validateStopSignControlContract(status.ControlContract); err != nil {
		return err
	}
	if strings.TrimSpace(status.Direction) != stopsigncontrol.StopSignDirectionForward {
		return stopSignModelCompatibilityError("checkpoint direction must be %q", stopsigncontrol.StopSignDirectionForward)
	}
	if status.TelemetrySampleIntervalMs != int(expected.TelemetrySampleInterval.Milliseconds()) {
		return stopSignModelCompatibilityError("checkpoint telemetry sample interval must be %dms, got=%dms", expected.TelemetrySampleInterval.Milliseconds(), status.TelemetrySampleIntervalMs)
	}
	if err := validateOrderedInts("checkpoint control horizon timing", status.ControlHorizonDtMs, expected.ControlHorizonDtMs); err != nil {
		return err
	}
	if status.ImageSize.Width != expected.FrameWidth || status.ImageSize.Height != expected.FrameHeight {
		return stopSignModelCompatibilityError(
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
		return stopSignModelCompatibilityError(
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
	if err := validateEnabledStopSignStateInputs(status.StateInputs); err != nil {
		return err
	}
	return nil
}

func validateEnabledStopSignStateInputs(inputs map[string]stopSignModelInputSpec) error {
	expected := make(map[string]struct{}, len(requiredStopSignStateInputs))
	for _, name := range requiredStopSignStateInputs {
		expected[name] = struct{}{}
		spec, ok := inputs[name]
		if !ok || !spec.Enabled {
			return stopSignModelCompatibilityError("checkpoint must enable state input %q", name)
		}
	}
	for name, spec := range inputs {
		if !spec.Enabled {
			continue
		}
		if _, ok := expected[name]; !ok {
			return stopSignModelCompatibilityError("checkpoint enables unexpected state input %q", name)
		}
	}
	return nil
}

func validateStopSignControlContract(contract stopSignControlContract) error {
	if strings.TrimSpace(contract.Name) != stopsigncontrol.StopSignMotionPlanContractV1 {
		return stopSignModelCompatibilityError(
			"control contract %q is incompatible; retrain with %q",
			strings.TrimSpace(contract.Name),
			stopsigncontrol.StopSignMotionPlanContractV1,
		)
	}
	if contract.Version != 1 {
		return stopSignModelCompatibilityError("control contract version must be 1, got=%d", contract.Version)
	}
	if strings.TrimSpace(contract.Direction) != stopsigncontrol.StopSignDirectionForward {
		return stopSignModelCompatibilityError("control contract direction must be %q", stopsigncontrol.StopSignDirectionForward)
	}
	if err := validateOrderedStrings("control contract targets", contract.Targets, requiredStopSignControlHeads); err != nil {
		return err
	}
	expectedActivations := map[string]string{
		"future_speed_mps": "sigmoid",
		"stop_intent":      "sigmoid",
	}
	for name, expected := range expectedActivations {
		if strings.ToLower(strings.TrimSpace(contract.OutputActivations[name])) != expected {
			return stopSignModelCompatibilityError("control contract activation for %s must be %s", name, expected)
		}
	}
	if len(contract.OutputActivations) != len(expectedActivations) {
		return stopSignModelCompatibilityError("control contract output activations differ: got=%v", contract.OutputActivations)
	}
	expectedRanges := map[string][2]float64{
		"future_speed_mps": {0, stopsigncontrol.StopSignMotionPlanMaxSpeedMPS},
		"stop_intent":      {0, 1},
	}
	for name, expected := range expectedRanges {
		actual := contract.OutputRanges[name]
		if len(actual) != 2 || actual[0] != expected[0] || actual[1] != expected[1] {
			return stopSignModelCompatibilityError("control contract range for %s must be [%g, %g]", name, expected[0], expected[1])
		}
	}
	if len(contract.OutputRanges) != len(expectedRanges) {
		return stopSignModelCompatibilityError("control contract output ranges differ: got=%v", contract.OutputRanges)
	}
	return nil
}

func cloneStopSignControlContract(source stopSignControlContract) stopSignControlContract {
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
		return stopSignModelCompatibilityError("%s differ: got=%v want=%v", label, actual, expected)
	}
	for index := range expected {
		if strings.TrimSpace(actual[index]) != strings.TrimSpace(expected[index]) {
			return stopSignModelCompatibilityError("%s differ: got=%v want=%v", label, actual, expected)
		}
	}
	return nil
}

func validateOrderedInts(label string, actual, expected []int) error {
	if len(actual) != len(expected) {
		return stopSignModelCompatibilityError("%s differ: got=%v want=%v", label, actual, expected)
	}
	for index := range expected {
		if actual[index] != expected[index] {
			return stopSignModelCompatibilityError("%s differ: got=%v want=%v", label, actual, expected)
		}
	}
	return nil
}

func stopSignModelCompatibilityError(format string, args ...any) error {
	return fmt.Errorf("%w: %s", ErrStopSignModelIncompatible, fmt.Sprintf(format, args...))
}
