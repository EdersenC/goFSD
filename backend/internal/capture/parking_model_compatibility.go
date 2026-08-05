package capture

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
)

var ErrParkingModelIncompatible = errors.New("model is not compatible with parking inference")

var requiredParkingStateInputs = []string{
	"current_speed",
	"parking_target_configured",
	"parking_longitudinal_error",
	"parking_lateral_error",
	"parking_heading_error",
}

var requiredParkingControlHeads = []string{
	"steering",
	"acceleration",
	"brakePressureAvg",
}

type parkingModelStatus struct {
	Loaded             bool                             `json:"loaded"`
	Checkpoint         string                           `json:"checkpoint"`
	PlannerFormat      string                           `json:"planner_format"`
	ImageSize          parkingModelImageSize            `json:"image_size"`
	FrameWindow        parkingModelFrameWindow          `json:"frame_window"`
	ImageOffsets       []int                            `json:"image_offsets"`
	TelemetryOffsets   []int                            `json:"telemetry_offsets"`
	FutureOffsets      []int                            `json:"future_offsets"`
	TelemetryFeatures  []string                         `json:"telemetry_feature_names"`
	ControlTargetNames []string                         `json:"control_target_names"`
	AuxTargetNames     []string                         `json:"aux_target_names"`
	StateInputs        map[string]parkingModelInputSpec `json:"state_inputs"`
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
	if err := validateParkingModelStatus(status, i.config); err != nil {
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
