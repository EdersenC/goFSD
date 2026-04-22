package dataset

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const (
	datasetReportFileName = "dataset_report.json"
	flatRangeEpsilon      = 1e-6
	clipBoundaryEpsilon   = 1e-6
)

var trackedLabelFields = map[string]string{
	"steer":               "Steering",
	"future_yaw_delta":    "future_yaw_delta",
	"future_horizon_s":    "future_horizon_seconds",
	"delta_speed":         "delta_speed",
	"delta_speed_target":  "delta_speed_target",
	"future_speed":        "future_speed",
	"future_speed_target": "future_speed_target",
	"current_speed":       "currentSpeed",
	"velocity_forward":    "velocityForward",
	"velocity_lateral":    "velocityLateral",
	"velocity_vertical":   "velocityVertical",
	"yaw_rate":            "yaw_rate",
	"gear":                "gear",
	"rpm":                 "rpm",
	"engine_health":       "engineHealth",
	"body_health":         "bodyHealth",
	"route_distance":      "routeDistance",
	"route_heading_err":   "routeHeadingError",
	"route_forward":       "routeForwardDelta",
	"route_lateral":       "routeLateralDelta",
	"road_node_distance":  "roadNodeDistance",
	"road_node_heading":   "roadNodeHeading",
	"road_node_density":   "roadNodeDensity",
	"road_lanes_fwd":      "roadLaneCountForward",
	"road_lanes_back":     "roadLaneCountBackward",
	"road_edge_span":      "roadEdgeSpan",
	"nearby_vehicles":     "nearbyVehicleCount30m",
	"nearby_peds":         "nearbyPedCount20m",
	"lead_distance":       "leadVehicleDistance",
	"lead_rel_speed":      "leadVehicleRelativeSpeed",
	"lead_ttc":            "leadVehicleTTC",
	"lead_heading_delta":  "leadVehicleHeadingDelta",
	"time_since_sync_ms":  "timeSinceSyncMs",
	"time_since_chunk_ms": "timeSinceChunkStartMs",
}

var trackedBooleanFields = map[string]string{
	"move_intent":              "move_intent",
	"route_gps_valid":          "routeGpsValid",
	"on_road":                  "isOnRoad",
	"offroad_node":             "isOffroadNode",
	"in_junction":              "isInJunction",
	"traffic_light_node":       "hasTrafficLightNode",
	"highway":                  "isHighway",
	"lead_vehicle_present":     "hasLeadVehicle",
	"stopped_at_traffic_light": "isStoppedAtTrafficLights",
	"event_collision":          "eventCollision",
	"event_offroad":            "eventOffroad",
	"event_wrong_way":          "eventWrongWay",
	"event_reversing":          "eventReversing",
	"event_handbrake":          "eventHandbrake",
}

type DatasetReportConfig struct {
	ImageWidth          int     `json:"image_width"`
	ImageHeight         int     `json:"image_height"`
	WindowSize          int     `json:"window_size"`
	FrameStride         int     `json:"frame_stride"`
	SampleStride        int     `json:"sample_stride"`
	LabelTolerance      string  `json:"label_tolerance"`
	DeltaSpeedClip      float64 `json:"delta_speed_clip"`
	DeltaSpeedNormalize bool    `json:"delta_speed_normalize"`
}

type NumericSummary struct {
	Count int     `json:"count"`
	Min   float64 `json:"min"`
	Max   float64 `json:"max"`
	Mean  float64 `json:"mean"`
	Std   float64 `json:"std"`
	Flat  bool    `json:"flat"`
}

type BooleanSummary struct {
	Count      int     `json:"count"`
	TrueCount  int     `json:"true_count"`
	FalseCount int     `json:"false_count"`
	TrueRate   float64 `json:"true_rate"`
	FalseRate  float64 `json:"false_rate"`
}

type DeltaSpeedClipSummary struct {
	ClipValue         float64 `json:"clip_value"`
	NegativeClipCount int     `json:"negative_clip_count"`
	PositiveClipCount int     `json:"positive_clip_count"`
	AnyClipCount      int     `json:"any_clip_count"`
	NegativeClipRate  float64 `json:"negative_clip_rate"`
	PositiveClipRate  float64 `json:"positive_clip_rate"`
	AnyClipRate       float64 `json:"any_clip_rate"`
}

type CategoricalSummary struct {
	Count         int            `json:"count"`
	UniqueCount   int            `json:"unique_count"`
	Counts        map[string]int `json:"counts"`
	DominantValue string         `json:"dominant_value,omitempty"`
	DominantShare float64        `json:"dominant_share"`
	LowDiversity  bool           `json:"low_diversity"`
}

type DatasetDiversitySummary struct {
	WeatherTypes CategoricalSummary `json:"weather_types"`
	TimeOfDay    CategoricalSummary `json:"time_of_day"`
	VehicleModel CategoricalSummary `json:"vehicle_model"`
	VehicleColor CategoricalSummary `json:"vehicle_color"`
	TripSeed     CategoricalSummary `json:"trip_seed"`
	Warnings     []string           `json:"warnings,omitempty"`
}

type DatasetReportSummary struct {
	TripCount           int                       `json:"trip_count"`
	CompletedTrips      int                       `json:"completed_trips"`
	SkippedTrips        int                       `json:"skipped_trips"`
	FailedTrips         int                       `json:"failed_trips"`
	MissingDatasetTrips int                       `json:"missing_dataset_trips"`
	ZeroSampleTrips     int                       `json:"zero_sample_trips"`
	FrameCount          int                       `json:"frame_count"`
	SampleCount         int                       `json:"sample_count"`
	StoppedSampleCount  int                       `json:"stopped_sample_count"`
	MovingSampleCount   int                       `json:"moving_sample_count"`
	StoppedSampleShare  float64                   `json:"stopped_sample_share"`
	TripStates          map[string]int            `json:"trip_states"`
	FlatLabelTripCounts map[string]int            `json:"flat_label_trip_counts"`
	LabelStats          map[string]NumericSummary `json:"label_stats"`
	BooleanStats        map[string]BooleanSummary `json:"boolean_stats"`
	DeltaSpeedClip      DeltaSpeedClipSummary     `json:"delta_speed_clip"`
	Diversity           DatasetDiversitySummary   `json:"diversity"`
}

type TripDatasetReport struct {
	RunID              string                    `json:"run_id"`
	SceneID            string                    `json:"scene_id"`
	SceneVariant       string                    `json:"scene_variant"`
	SceneKey           string                    `json:"scene_key"`
	TripName           string                    `json:"trip_name"`
	TripDir            string                    `json:"trip_dir"`
	ProcessingState    string                    `json:"processing_state"`
	ProcessingError    string                    `json:"processing_error,omitempty"`
	ProcessingWarning  string                    `json:"processing_warning,omitempty"`
	FrameCount         int                       `json:"frame_count"`
	SampleCount        int                       `json:"sample_count"`
	MissingDataset     bool                      `json:"missing_dataset"`
	ZeroSamples        bool                      `json:"zero_samples"`
	ZeroSampleReasons  map[string]int            `json:"zero_sample_reasons,omitempty"`
	TripSeed           string                    `json:"trip_seed,omitempty"`
	WeatherType        string                    `json:"weather_type,omitempty"`
	TimeOfDay          string                    `json:"time_of_day,omitempty"`
	VehicleModel       string                    `json:"vehicle_model,omitempty"`
	VehicleColor       string                    `json:"vehicle_color,omitempty"`
	StoppedSampleCount int                       `json:"stopped_sample_count"`
	MovingSampleCount  int                       `json:"moving_sample_count"`
	StoppedSampleShare float64                   `json:"stopped_sample_share"`
	FlatLabels         []string                  `json:"flat_labels"`
	LabelStats         map[string]NumericSummary `json:"label_stats"`
	BooleanStats       map[string]BooleanSummary `json:"boolean_stats"`
	DeltaSpeedClip     DeltaSpeedClipSummary     `json:"delta_speed_clip"`
	Warnings           []string                  `json:"warnings,omitempty"`
}

type SceneDatasetReport struct {
	SceneID      string               `json:"scene_id"`
	SceneVariant string               `json:"scene_variant"`
	SceneKey     string               `json:"scene_key"`
	Summary      DatasetReportSummary `json:"summary"`
}

type RunDatasetReport struct {
	GeneratedAt time.Time            `json:"generated_at"`
	RunID       string               `json:"run_id"`
	RunDir      string               `json:"run_dir"`
	ReportFile  string               `json:"report_file"`
	Config      DatasetReportConfig  `json:"dataset_config"`
	Summary     DatasetReportSummary `json:"summary"`
	Scenes      []SceneDatasetReport `json:"scenes"`
	Trips       []TripDatasetReport  `json:"trips"`
}

type GeneratedRunDatasetReport struct {
	RunID      string
	RunDir     string
	ReportPath string
	Report     RunDatasetReport
}

type numericAccumulator struct {
	count      int
	sum        float64
	sumSquares float64
	min        float64
	max        float64
}

type booleanAccumulator struct {
	count     int
	trueCount int
}

type categoricalAccumulator struct {
	counts map[string]int
	total  int
}

func (n *numericAccumulator) add(value float64) {
	if n.count == 0 {
		n.min = value
		n.max = value
	} else {
		if value < n.min {
			n.min = value
		}
		if value > n.max {
			n.max = value
		}
	}
	n.count++
	n.sum += value
	n.sumSquares += value * value
}

func (n *numericAccumulator) summary() NumericSummary {
	if n.count == 0 {
		return NumericSummary{}
	}
	mean := n.sum / float64(n.count)
	variance := (n.sumSquares / float64(n.count)) - (mean * mean)
	if variance < 0 {
		variance = 0
	}
	return NumericSummary{
		Count: n.count,
		Min:   n.min,
		Max:   n.max,
		Mean:  mean,
		Std:   math.Sqrt(variance),
		Flat:  n.count > 1 && math.Abs(n.max-n.min) <= flatRangeEpsilon,
	}
}

func newCategoricalAccumulator() *categoricalAccumulator {
	return &categoricalAccumulator{
		counts: make(map[string]int),
	}
}

func (b *booleanAccumulator) add(value bool) {
	b.count++
	if value {
		b.trueCount++
	}
}

func (b *booleanAccumulator) summary() BooleanSummary {
	if b.count == 0 {
		return BooleanSummary{}
	}
	falseCount := b.count - b.trueCount
	return BooleanSummary{
		Count:      b.count,
		TrueCount:  b.trueCount,
		FalseCount: falseCount,
		TrueRate:   float64(b.trueCount) / float64(b.count),
		FalseRate:  float64(falseCount) / float64(b.count),
	}
}

func (c *categoricalAccumulator) add(value string) {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return
	}
	c.counts[trimmed]++
	c.total++
}

func (c *categoricalAccumulator) summary() CategoricalSummary {
	if c.total == 0 {
		return CategoricalSummary{Counts: map[string]int{}}
	}
	counts := make(map[string]int, len(c.counts))
	dominantValue := ""
	dominantCount := 0
	for key, value := range c.counts {
		counts[key] = value
		if value > dominantCount {
			dominantValue = key
			dominantCount = value
		}
	}
	dominantShare := float64(dominantCount) / float64(c.total)
	uniqueCount := len(counts)
	return CategoricalSummary{
		Count:         c.total,
		UniqueCount:   uniqueCount,
		Counts:        counts,
		DominantValue: dominantValue,
		DominantShare: dominantShare,
		LowDiversity:  c.total >= 3 && uniqueCount <= 2 && dominantShare >= 0.8,
	}
}

type summaryAccumulator struct {
	tripCount           int
	completedTrips      int
	skippedTrips        int
	failedTrips         int
	missingDatasetTrips int
	zeroSampleTrips     int
	frameCount          int
	sampleCount         int
	stoppedSampleCount  int
	movingSampleCount   int
	tripStates          map[string]int
	flatLabelTripCounts map[string]int
	labelStats          map[string]*numericAccumulator
	booleanStats        map[string]*booleanAccumulator
	weatherTypes        *categoricalAccumulator
	timeOfDay           *categoricalAccumulator
	vehicleModels       *categoricalAccumulator
	vehicleColors       *categoricalAccumulator
	tripSeeds           *categoricalAccumulator
	clipValue           float64
	negativeClipCount   int
	positiveClipCount   int
	anyClipCount        int
}

func newSummaryAccumulator(clipValue float64) *summaryAccumulator {
	return &summaryAccumulator{
		tripStates:          make(map[string]int),
		flatLabelTripCounts: make(map[string]int),
		labelStats:          make(map[string]*numericAccumulator),
		booleanStats:        make(map[string]*booleanAccumulator),
		weatherTypes:        newCategoricalAccumulator(),
		timeOfDay:           newCategoricalAccumulator(),
		vehicleModels:       newCategoricalAccumulator(),
		vehicleColors:       newCategoricalAccumulator(),
		tripSeeds:           newCategoricalAccumulator(),
		clipValue:           clipValue,
	}
}

func (a *summaryAccumulator) addTrip(report TripDatasetReport) {
	a.tripCount++
	a.frameCount += report.FrameCount
	switch report.ProcessingState {
	case "completed":
		a.completedTrips++
	case "skipped":
		a.skippedTrips++
	case "failed":
		a.failedTrips++
	}
	a.tripStates[report.ProcessingState]++
	if report.MissingDataset {
		a.missingDatasetTrips++
	}
	if report.ZeroSamples {
		a.zeroSampleTrips++
	}
	for _, label := range report.FlatLabels {
		a.flatLabelTripCounts[label]++
	}
	a.weatherTypes.add(report.WeatherType)
	a.timeOfDay.add(report.TimeOfDay)
	a.vehicleModels.add(report.VehicleModel)
	a.vehicleColors.add(report.VehicleColor)
	a.tripSeeds.add(report.TripSeed)
}

func (a *summaryAccumulator) addSample(label map[string]any) {
	for key, rawKey := range trackedLabelFields {
		value, ok := numberField(label[rawKey])
		if !ok {
			continue
		}
		acc := a.labelStats[key]
		if acc == nil {
			acc = &numericAccumulator{}
			a.labelStats[key] = acc
		}
		acc.add(value)
		if key == "delta_speed" {
			if math.Abs(value+a.clipValue) <= clipBoundaryEpsilon {
				a.negativeClipCount++
			}
			if math.Abs(value-a.clipValue) <= clipBoundaryEpsilon {
				a.positiveClipCount++
			}
			if math.Abs(math.Abs(value)-a.clipValue) <= clipBoundaryEpsilon {
				a.anyClipCount++
			}
		}
	}
	for key, rawKey := range trackedBooleanFields {
		value, ok := booleanField(label[rawKey])
		if !ok {
			continue
		}
		acc := a.booleanStats[key]
		if acc == nil {
			acc = &booleanAccumulator{}
			a.booleanStats[key] = acc
		}
		acc.add(value)
	}
	if isStoppedLabelValue(label["isStopped"]) {
		a.stoppedSampleCount++
	} else {
		a.movingSampleCount++
	}
	a.sampleCount++
}

func (a *summaryAccumulator) summary() DatasetReportSummary {
	labelStats := make(map[string]NumericSummary, len(a.labelStats))
	for key, acc := range a.labelStats {
		labelStats[key] = acc.summary()
	}
	booleanStats := make(map[string]BooleanSummary, len(a.booleanStats))
	for key, acc := range a.booleanStats {
		booleanStats[key] = acc.summary()
	}

	clipSummary := DeltaSpeedClipSummary{ClipValue: a.clipValue}
	if deltaStats, ok := labelStats["delta_speed"]; ok && deltaStats.Count > 0 {
		denom := float64(deltaStats.Count)
		clipSummary.NegativeClipCount = a.negativeClipCount
		clipSummary.PositiveClipCount = a.positiveClipCount
		clipSummary.AnyClipCount = a.anyClipCount
		clipSummary.NegativeClipRate = float64(a.negativeClipCount) / denom
		clipSummary.PositiveClipRate = float64(a.positiveClipCount) / denom
		clipSummary.AnyClipRate = float64(a.anyClipCount) / denom
	}

	stoppedShare := 0.0
	if a.sampleCount > 0 {
		stoppedShare = float64(a.stoppedSampleCount) / float64(a.sampleCount)
	}

	tripStates := make(map[string]int, len(a.tripStates))
	for key, value := range a.tripStates {
		tripStates[key] = value
	}

	flatTripCounts := make(map[string]int, len(a.flatLabelTripCounts))
	for key, value := range a.flatLabelTripCounts {
		flatTripCounts[key] = value
	}
	diversity := DatasetDiversitySummary{
		WeatherTypes: a.weatherTypes.summary(),
		TimeOfDay:    a.timeOfDay.summary(),
		VehicleModel: a.vehicleModels.summary(),
		VehicleColor: a.vehicleColors.summary(),
		TripSeed:     a.tripSeeds.summary(),
	}
	warnings := make([]string, 0, 4)
	if diversity.WeatherTypes.LowDiversity {
		warnings = append(warnings, "weather_types")
	}
	if diversity.TimeOfDay.LowDiversity {
		warnings = append(warnings, "time_of_day")
	}
	if diversity.VehicleModel.LowDiversity {
		warnings = append(warnings, "vehicle_model")
	}
	if diversity.VehicleColor.LowDiversity {
		warnings = append(warnings, "vehicle_color")
	}
	diversity.Warnings = warnings

	return DatasetReportSummary{
		TripCount:           a.tripCount,
		CompletedTrips:      a.completedTrips,
		SkippedTrips:        a.skippedTrips,
		FailedTrips:         a.failedTrips,
		MissingDatasetTrips: a.missingDatasetTrips,
		ZeroSampleTrips:     a.zeroSampleTrips,
		FrameCount:          a.frameCount,
		SampleCount:         a.sampleCount,
		StoppedSampleCount:  a.stoppedSampleCount,
		MovingSampleCount:   a.movingSampleCount,
		StoppedSampleShare:  stoppedShare,
		TripStates:          tripStates,
		FlatLabelTripCounts: flatTripCounts,
		LabelStats:          labelStats,
		BooleanStats:        booleanStats,
		DeltaSpeedClip:      clipSummary,
		Diversity:           diversity,
	}
}

type tripContext struct {
	tripDir      string
	runDir       string
	runID        string
	sceneID      string
	sceneVariant string
	sceneKey     string
	tripName     string
}

func WriteRunDatasetReports(tripDirs []string, config DatasetReportConfig) ([]GeneratedRunDatasetReport, error) {
	grouped, err := groupTripDirsByRun(tripDirs)
	if err != nil {
		return nil, err
	}

	runDirs := make([]string, 0, len(grouped))
	for runDir := range grouped {
		runDirs = append(runDirs, runDir)
	}
	sort.Strings(runDirs)

	reports := make([]GeneratedRunDatasetReport, 0, len(runDirs))
	var reportErrors []error
	for _, runDir := range runDirs {
		report, err := BuildRunDatasetReport(runDir, grouped[runDir], config)
		if err != nil {
			reportErrors = append(reportErrors, err)
			continue
		}

		reportPath := filepath.Join(runDir, datasetReportFileName)
		report.ReportFile = reportPath
		body, err := json.MarshalIndent(report, "", "  ")
		if err != nil {
			reportErrors = append(reportErrors, fmt.Errorf("marshal dataset report for %s: %w", runDir, err))
			continue
		}
		body = append(body, '\n')
		if err := os.WriteFile(reportPath, body, 0o644); err != nil {
			reportErrors = append(reportErrors, fmt.Errorf("write dataset report for %s: %w", runDir, err))
			continue
		}

		reports = append(reports, GeneratedRunDatasetReport{
			RunID:      report.RunID,
			RunDir:     runDir,
			ReportPath: reportPath,
			Report:     report,
		})
	}

	return reports, errors.Join(reportErrors...)
}

func BuildRunDatasetReport(runDir string, tripDirs []string, config DatasetReportConfig) (RunDatasetReport, error) {
	if strings.TrimSpace(runDir) == "" {
		return RunDatasetReport{}, fmt.Errorf("run directory must not be empty")
	}
	if len(tripDirs) == 0 {
		return RunDatasetReport{}, fmt.Errorf("no trip directories provided for run %s", runDir)
	}

	contexts := make([]tripContext, 0, len(tripDirs))
	for _, tripDir := range tripDirs {
		ctx, err := buildTripContext(tripDir)
		if err != nil {
			return RunDatasetReport{}, err
		}
		contexts = append(contexts, ctx)
	}
	sort.Slice(contexts, func(i, j int) bool {
		if contexts[i].sceneKey != contexts[j].sceneKey {
			return contexts[i].sceneKey < contexts[j].sceneKey
		}
		return contexts[i].tripName < contexts[j].tripName
	})

	runSummary := newSummaryAccumulator(config.DeltaSpeedClip)
	sceneAccumulators := make(map[string]*summaryAccumulator)
	sceneDescriptors := make(map[string]SceneDatasetReport)
	tripReports := make([]TripDatasetReport, 0, len(contexts))

	runID := filepath.Base(filepath.Clean(runDir))
	for _, ctx := range contexts {
		report, rawLabels, err := buildTripDatasetReport(ctx, config.DeltaSpeedClip)
		if err != nil {
			return RunDatasetReport{}, err
		}
		tripReports = append(tripReports, report)
		runSummary.addTrip(report)
		for _, label := range rawLabels {
			runSummary.addSample(label)
		}

		sceneAcc := sceneAccumulators[ctx.sceneKey]
		if sceneAcc == nil {
			sceneAcc = newSummaryAccumulator(config.DeltaSpeedClip)
			sceneAccumulators[ctx.sceneKey] = sceneAcc
			sceneDescriptors[ctx.sceneKey] = SceneDatasetReport{
				SceneID:      ctx.sceneID,
				SceneVariant: ctx.sceneVariant,
				SceneKey:     ctx.sceneKey,
			}
		}
		sceneAcc.addTrip(report)
		for _, label := range rawLabels {
			sceneAcc.addSample(label)
		}

		if report.RunID != "" {
			runID = report.RunID
		}
	}

	sceneKeys := make([]string, 0, len(sceneAccumulators))
	for key := range sceneAccumulators {
		sceneKeys = append(sceneKeys, key)
	}
	sort.Strings(sceneKeys)

	scenes := make([]SceneDatasetReport, 0, len(sceneKeys))
	for _, key := range sceneKeys {
		scene := sceneDescriptors[key]
		scene.Summary = sceneAccumulators[key].summary()
		scenes = append(scenes, scene)
	}

	return RunDatasetReport{
		GeneratedAt: time.Now().UTC(),
		RunID:       runID,
		RunDir:      runDir,
		Config:      config,
		Summary:     runSummary.summary(),
		Scenes:      scenes,
		Trips:       tripReports,
	}, nil
}

func groupTripDirsByRun(tripDirs []string) (map[string][]string, error) {
	grouped := make(map[string][]string)
	for _, tripDir := range tripDirs {
		ctx, err := buildTripContext(tripDir)
		if err != nil {
			return nil, err
		}
		grouped[ctx.runDir] = append(grouped[ctx.runDir], ctx.tripDir)
	}
	for runDir := range grouped {
		sort.Strings(grouped[runDir])
	}
	return grouped, nil
}

func buildTripContext(tripDir string) (tripContext, error) {
	resolvedTripDir, err := resolveTripDir(tripDir)
	if err != nil {
		return tripContext{}, err
	}

	sceneDir := filepath.Dir(resolvedTripDir)
	runDir := filepath.Dir(sceneDir)
	if sceneDir == "." || runDir == "." || sceneDir == resolvedTripDir || runDir == sceneDir {
		return tripContext{}, fmt.Errorf("trip directory does not have expected run/scene layout: %s", resolvedTripDir)
	}

	ctx := tripContext{
		tripDir:  resolvedTripDir,
		runDir:   runDir,
		runID:    filepath.Base(runDir),
		sceneKey: filepath.Base(sceneDir),
		tripName: filepath.Base(resolvedTripDir),
	}
	sceneID, sceneVariant := parseSceneKey(ctx.sceneKey)
	ctx.sceneID = sceneID
	ctx.sceneVariant = sceneVariant

	metadata, err := loadTripMetadata(filepath.Join(resolvedTripDir, "metadata.json"))
	if err == nil {
		if strings.TrimSpace(metadata.RunID) != "" {
			ctx.runID = strings.TrimSpace(metadata.RunID)
		}
		if strings.TrimSpace(metadata.SceneID) != "" {
			ctx.sceneID = strings.TrimSpace(metadata.SceneID)
		}
		if strings.TrimSpace(metadata.SceneVariant) != "" {
			ctx.sceneVariant = strings.TrimSpace(metadata.SceneVariant)
		}
		if ctx.sceneID != "" && ctx.sceneVariant != "" {
			ctx.sceneKey = fmt.Sprintf("%s_%s", ctx.sceneID, ctx.sceneVariant)
		}
	}

	return ctx, nil
}

func parseSceneKey(sceneKey string) (string, string) {
	idx := strings.LastIndex(sceneKey, "_")
	if idx <= 0 || idx >= len(sceneKey)-1 {
		return sceneKey, "default"
	}
	return sceneKey[:idx], sceneKey[idx+1:]
}

func buildTripDatasetReport(ctx tripContext, deltaSpeedClip float64) (TripDatasetReport, []map[string]any, error) {
	statusPath := filepath.Join(ctx.tripDir, "processing.json")
	status, err := ReadStatusFile(statusPath)
	if err != nil {
		status = ProcessingStatus{
			State: "missing",
			Error: err.Error(),
		}
	}

	samples, missingDataset, err := readDatasetSamples(filepath.Join(ctx.tripDir, "dataset.jsonl"))
	if err != nil {
		return TripDatasetReport{}, nil, fmt.Errorf("read dataset for %s: %w", ctx.tripDir, err)
	}

	frameCount := status.FrameCount
	if frameCount <= 0 {
		frameCount = countFrameImages(filepath.Join(ctx.tripDir, "frames"))
	}
	var metadata tripMetadata
	if loaded, metadataErr := loadTripMetadata(filepath.Join(ctx.tripDir, "metadata.json")); metadataErr == nil {
		metadata = loaded
	}

	acc := newSummaryAccumulator(deltaSpeedClip)
	rawLabels := make([]map[string]any, 0, len(samples))
	for _, sample := range samples {
		rawLabels = append(rawLabels, sample.Label)
		acc.addSample(sample.Label)
	}
	summary := acc.summary()

	flatLabels := make([]string, 0)
	for key, stats := range summary.LabelStats {
		if stats.Flat {
			flatLabels = append(flatLabels, key)
		}
	}
	sort.Strings(flatLabels)

	processingState := strings.TrimSpace(status.State)
	if processingState == "" {
		processingState = "missing"
	}

	warnings := make([]string, 0, 4+len(flatLabels))
	switch processingState {
	case "failed":
		warnings = append(warnings, "processing_failed")
	case "skipped":
		warnings = append(warnings, "processing_skipped")
	case "missing":
		warnings = append(warnings, "processing_status_missing")
	}
	if missingDataset {
		warnings = append(warnings, "missing_dataset")
	}
	if len(samples) == 0 {
		warnings = append(warnings, "zero_samples")
	}
	if len(status.ZeroSampleReasons) > 0 {
		reasonKeys := make([]string, 0, len(status.ZeroSampleReasons))
		for key := range status.ZeroSampleReasons {
			reasonKeys = append(reasonKeys, key)
		}
		sort.Strings(reasonKeys)
		for _, key := range reasonKeys {
			warnings = append(warnings, fmt.Sprintf("zero_sample_reason:%s", key))
		}
	}
	for _, label := range flatLabels {
		warnings = append(warnings, fmt.Sprintf("flat_label:%s", label))
	}

	return TripDatasetReport{
		RunID:              ctx.runID,
		SceneID:            ctx.sceneID,
		SceneVariant:       ctx.sceneVariant,
		SceneKey:           ctx.sceneKey,
		TripName:           ctx.tripName,
		TripDir:            ctx.tripDir,
		ProcessingState:    processingState,
		ProcessingError:    strings.TrimSpace(status.Error),
		ProcessingWarning:  strings.TrimSpace(status.Warning),
		FrameCount:         frameCount,
		SampleCount:        len(samples),
		MissingDataset:     missingDataset,
		ZeroSamples:        len(samples) == 0,
		ZeroSampleReasons:  status.ZeroSampleReasons,
		TripSeed:           strings.TrimSpace(metadata.TripSeed),
		WeatherType:        strings.TrimSpace(metadata.WeatherType),
		TimeOfDay:          strings.TrimSpace(metadata.TimeOfDay),
		VehicleModel:       strings.TrimSpace(metadata.VehicleModel),
		VehicleColor:       strings.TrimSpace(metadata.VehicleColor),
		StoppedSampleCount: summary.StoppedSampleCount,
		MovingSampleCount:  summary.MovingSampleCount,
		StoppedSampleShare: summary.StoppedSampleShare,
		FlatLabels:         flatLabels,
		LabelStats:         summary.LabelStats,
		BooleanStats:       summary.BooleanStats,
		DeltaSpeedClip:     summary.DeltaSpeedClip,
		Warnings:           warnings,
	}, rawLabels, nil
}

func readDatasetSamples(path string) ([]DatasetSample, bool, error) {
	file, err := os.Open(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, true, nil
		}
		return nil, false, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 64*1024), 8*1024*1024)
	samples := make([]DatasetSample, 0)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var sample DatasetSample
		if err := json.Unmarshal([]byte(line), &sample); err != nil {
			return nil, false, fmt.Errorf("parse dataset line: %w", err)
		}
		samples = append(samples, sample)
	}
	if err := scanner.Err(); err != nil {
		return nil, false, err
	}
	return samples, false, nil
}

func countFrameImages(framesDir string) int {
	entries, err := os.ReadDir(framesDir)
	if err != nil {
		return 0
	}
	count := 0
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := strings.ToLower(entry.Name())
		if strings.HasSuffix(name, ".jpg") || strings.HasSuffix(name, ".jpeg") || strings.HasSuffix(name, ".png") {
			count++
		}
	}
	return count
}
