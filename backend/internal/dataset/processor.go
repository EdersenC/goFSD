package dataset

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	_ "image/jpeg"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const (
	defaultFFmpegBin                = "ffmpeg"
	defaultFFprobeBin               = "ffprobe"
	defaultImageWidth               = 224
	defaultImageHeight              = 224
	defaultWindowSize               = 3
	defaultFrameStride              = 2
	defaultSampleStride             = 2
	defaultFutureTelemetryCount     = 6
	defaultTelemetrySampleInterval  = 50 * time.Millisecond
	defaultLabelTolerance           = 100 * time.Millisecond
	defaultFlashBrightnessThreshold = 245.0
	defaultFlashFrameLimit          = 90
	defaultStoppedSampleBurst       = 3
	defaultStoppedSampleSpacing     = 2.0
	futureTargetSmoothingRadius     = 2
	processingConfigVersion         = 5
	processingWorkspacePrefix       = ".processing-work-"
	processingJournalVersion        = 1
	processingJournalFile           = "promotion.json"
)

var (
	ErrInvalidTripDir     = errors.New("invalid trip dir")
	ErrMissingTripFiles   = errors.New("missing trip files")
	ErrTripRecordNotFound = errors.New("trip record not found in run manifest")
	ErrSyncFlashNotFound  = errors.New("sync flash not found")
)

type commandFactory func(ctx context.Context, name string, args ...string) *exec.Cmd

type VideoFrame struct {
	Index     int     `json:"index"`
	PTS       float64 `json:"pts"`
	ImagePath string  `json:"imagePath,omitempty"`
}

type ProcessingStatus struct {
	State                     string         `json:"state"`
	ConfigFingerprint         string         `json:"configFingerprint,omitempty"`
	StartedAt                 string         `json:"startedAt,omitempty"`
	CompletedAt               string         `json:"completedAt,omitempty"`
	Error                     string         `json:"error,omitempty"`
	Warning                   string         `json:"warning,omitempty"`
	FramesDir                 string         `json:"framesDir,omitempty"`
	DatasetFile               string         `json:"datasetFile,omitempty"`
	ImageWidth                int            `json:"imageWidth,omitempty"`
	ImageHeight               int            `json:"imageHeight,omitempty"`
	ImageOffsets              []int          `json:"imageOffsets,omitempty"`
	TelemetryOffsets          []int          `json:"telemetryOffsets,omitempty"`
	FutureOffsets             []int          `json:"futureOffsets,omitempty"`
	TelemetrySampleIntervalMs float64        `json:"telemetrySampleIntervalMs,omitempty"`
	FrameCount                int            `json:"frameCount"`
	SampleCount               int            `json:"sampleCount"`
	ZeroSampleReasons         map[string]int `json:"zeroSampleReasons,omitempty"`
}

type DatasetSample struct {
	AnchorVideoPTS          float64                `json:"anchor_video_pts"`
	AnchorGameTime          float64                `json:"anchor_game_time"`
	FramePaths              []string               `json:"frame_paths"`
	TelemetryHistory        []GroupedTelemetryItem `json:"telemetry_history,omitempty"`
	TelemetryFuture         []GroupedTelemetryItem `json:"telemetry_future,omitempty"`
	Label                   GroupedLabel           `json:"label"`
	Task                    string                 `json:"task,omitempty"`
	Phase                   string                 `json:"phase,omitempty"`
	ScenarioLocationID      string                 `json:"scenario_location_id,omitempty"`
	ScenarioSplitGroup      string                 `json:"scenario_split_group,omitempty"`
	VariationID             string                 `json:"variation_id,omitempty"`
	TrainingEligible        *bool                  `json:"training_eligible,omitempty"`
	TrainingExclusionReason string                 `json:"training_exclusion_reason,omitempty"`
	StopSignGoal            map[string]any         `json:"stopSignGoal,omitempty"`
	StopSignOutcome         map[string]any         `json:"stopSignOutcome,omitempty"`
}

type GroupedLabel struct {
	Control GroupedLabelControl `json:"control"`
	Aux     GroupedLabelAux     `json:"aux"`
}

type GroupedLabelControl struct {
	Steering                          any `json:"Steering,omitempty"`
	ExpertDesiredWheelSteerNormalized any `json:"expertDesiredWheelSteerNormalized,omitempty"`
	ExpertDesiredSpeedMps             any `json:"expertDesiredSpeedMps,omitempty"`
	ExpertThrottle                    any `json:"expertThrottle,omitempty"`
	ExpertBrake                       any `json:"expertBrake,omitempty"`
	ExpertStopProbability             any `json:"expertStopProbability,omitempty"`
	ExpertGoProbability               any `json:"expertGoProbability,omitempty"`
}

type GroupedLabelAux struct {
	FutureSpeed             any       `json:"future_speed,omitempty"`
	FutureSpeedTarget       any       `json:"future_speed_target,omitempty"`
	FutureHorizonSeconds    any       `json:"future_horizon_seconds,omitempty"`
	RouteForwardDelta       any       `json:"routeForwardDelta,omitempty"`
	StopSignPhase           any       `json:"stopSignPhase,omitempty"`
	FutureSpeedTargetsMps   []float64 `json:"future_speed_targets_mps,omitempty"`
	FutureObservedSpeedMps  []float64 `json:"future_observed_speed_mps,omitempty"`
	FutureStopProbabilities []float64 `json:"future_stop_probabilities,omitempty"`
	FutureGoProbabilities   []float64 `json:"future_go_probabilities,omitempty"`
	FutureStopSignPhases    []string  `json:"future_stop_sign_phases,omitempty"`
}

type GroupedTelemetryItem struct {
	Control GroupedTelemetryControl `json:"control"`
	Aux     GroupedTelemetryAux     `json:"aux"`
	Raw     map[string]any          `json:"raw,omitempty"`
}

type GroupedTelemetryControl struct {
	Steering                          any `json:"Steering,omitempty"`
	Acceleration                      any `json:"acceleration,omitempty"`
	BrakePressureAvg                  any `json:"brakePressureAvg,omitempty"`
	ExpertDesiredWheelSteerNormalized any `json:"expertDesiredWheelSteerNormalized,omitempty"`
	ExpertDesiredSpeedMps             any `json:"expertDesiredSpeedMps,omitempty"`
	ExpertThrottle                    any `json:"expertThrottle,omitempty"`
	ExpertBrake                       any `json:"expertBrake,omitempty"`
	ExpertStopProbability             any `json:"expertStopProbability,omitempty"`
	ExpertGoProbability               any `json:"expertGoProbability,omitempty"`
}

type GroupedTelemetryAux struct {
	CurrentSpeed                  any `json:"currentSpeed,omitempty"`
	Yaw                           any `json:"yaw,omitempty"`
	YawRate                       any `json:"yawRate,omitempty"`
	RouteDirectionCode            any `json:"routeDirectionCode,omitempty"`
	RouteDirectionDistanceM       any `json:"routeDirectionDistanceM,omitempty"`
	RouteDirectionUnknown         any `json:"routeDirectionUnknown,omitempty"`
	RouteDirectionKeepStraight    any `json:"routeDirectionKeepStraight,omitempty"`
	RouteDirectionTurnLeft        any `json:"routeDirectionTurnLeft,omitempty"`
	RouteDirectionTurnRight       any `json:"routeDirectionTurnRight,omitempty"`
	RouteDirectionRerouteWrongWay any `json:"routeDirectionRerouteWrongWay,omitempty"`
	RouteForwardDelta             any `json:"routeForwardDelta,omitempty"`
	RouteHeadingError             any `json:"routeHeadingError,omitempty"`
	RouteDistance                 any `json:"routeDistance,omitempty"`
	LeadVehicleDistance           any `json:"leadVehicleDistance,omitempty"`
	HasLeadVehicle                any `json:"hasLeadVehicle,omitempty"`
	GPS                           any `json:"gps,omitempty"`
	IsStopped                     any `json:"isStopped,omitempty"`
	RouteGPSValid                 any `json:"routeGpsValid,omitempty"`
	IsStoppedAtTraffic            any `json:"isStoppedAtTrafficLights,omitempty"`
	LeadVehicleRelSpeed           any `json:"leadVehicleRelativeSpeed,omitempty"`
	LeadVehicleHeadingDiff        any `json:"leadVehicleHeadingDelta,omitempty"`
	LeadVehicleTTC                any `json:"leadVehicleTTC,omitempty"`
	StopSignPhase                 any `json:"stopSignPhase,omitempty"`
	StopLineDistanceM             any `json:"stopLineDistanceM,omitempty"`
	StopSignLongitudinalErrorM    any `json:"stopSignLongitudinalErrorM,omitempty"`
	StopSignLateralErrorM         any `json:"stopSignLateralErrorM,omitempty"`
	StopSignHeadingErrorDeg       any `json:"stopSignHeadingErrorDeg,omitempty"`
}

type sampleBuildStats struct {
	CandidateWindowCount            int
	GeneratedSampleCount            int
	IncompleteFrameHistoryCount     int
	IncompleteTelemetryHistoryCount int
	IncompleteTelemetryFutureCount  int
	MissingCurrentLabelCount        int
	MissingFutureLabelCount         int
	MissingRouteForwardDeltaCount   int
	MissingFutureSpeedTargetCount   int
	InvalidFutureHorizonCount       int
	InvalidDerivedLabelCount        int
}

func (s sampleBuildStats) zeroSampleReasons() map[string]int {
	reasons := map[string]int{
		"candidate_windows": s.CandidateWindowCount,
	}
	if s.IncompleteFrameHistoryCount > 0 {
		reasons["incomplete_frame_history"] = s.IncompleteFrameHistoryCount
	}
	if s.IncompleteTelemetryHistoryCount > 0 {
		reasons["incomplete_telemetry_history"] = s.IncompleteTelemetryHistoryCount
	}
	if s.IncompleteTelemetryFutureCount > 0 {
		reasons["incomplete_telemetry_future"] = s.IncompleteTelemetryFutureCount
	}
	if s.MissingCurrentLabelCount > 0 {
		reasons["missing_current_label"] = s.MissingCurrentLabelCount
	}
	if s.MissingFutureLabelCount > 0 {
		reasons["missing_future_label"] = s.MissingFutureLabelCount
	}
	if s.MissingRouteForwardDeltaCount > 0 {
		reasons["missing_route_forward_delta"] = s.MissingRouteForwardDeltaCount
	}
	if s.MissingFutureSpeedTargetCount > 0 {
		reasons["missing_future_speed_target"] = s.MissingFutureSpeedTargetCount
	}
	if s.InvalidFutureHorizonCount > 0 {
		reasons["invalid_future_horizon"] = s.InvalidFutureHorizonCount
	}
	if s.InvalidDerivedLabelCount > 0 {
		reasons["invalid_derived_label"] = s.InvalidDerivedLabelCount
	}
	return reasons
}

type Processor struct {
	ffmpegBin                string
	ffprobeBin               string
	newCommand               commandFactory
	imageWidth               int
	imageHeight              int
	windowSize               int
	frameStride              int
	imageOffsets             []int
	sampleStride             int
	labelTolerance           time.Duration
	telemetryOffsets         []int
	futureOffsets            []int
	telemetrySampleInterval  time.Duration
	flashBrightnessThreshold float64
	flashFrameLimit          int
	force                    bool
	datasetOnly              bool
}

type Option func(*Processor)

type ffprobeFrame struct {
	PTSTime string `json:"pts_time"`
}

type ffprobeOutput struct {
	Frames []ffprobeFrame `json:"frames"`
}

type tripMetadata struct {
	RunID        string  `json:"runId"`
	SceneID      string  `json:"sceneId"`
	SceneVariant string  `json:"sceneVariant"`
	TripIndex    int     `json:"tripIndex"`
	SyncTime     float64 `json:"syncTime"`
	TripSeed     string  `json:"tripSeed"`
	WeatherType  string  `json:"weatherType"`
	TimeOfDay    string  `json:"timeOfDay"`
	Time         struct {
		Hour   int `json:"hour"`
		Minute int `json:"minute"`
		Second int `json:"second"`
	} `json:"time"`
	VehicleModel    string         `json:"vehicleModel"`
	VehicleColor    string         `json:"vehicleColor"`
	StopSignGoal    map[string]any `json:"stopSignGoal"`
	StopSignOutcome map[string]any `json:"stopSignOutcome"`
}

type runTripRecord struct {
	RunID        string           `json:"runId"`
	SceneID      string           `json:"sceneId"`
	SceneVariant string           `json:"sceneVariant"`
	TripIndex    int              `json:"tripIndex"`
	VehicleData  []map[string]any `json:"vehicleData"`
}

type timedLabel struct {
	RelativeSeconds float64
	Label           map[string]any
}

func NewProcessor(opts ...Option) *Processor {
	p := &Processor{
		ffmpegBin:  envOrDefault("FFMPEG_BIN", defaultFFmpegBin),
		ffprobeBin: envOrDefault("FFPROBE_BIN", defaultFFprobeBin),
		newCommand: func(ctx context.Context, name string, args ...string) *exec.Cmd {
			return exec.CommandContext(ctx, name, args...)
		},
		imageWidth:               defaultImageWidth,
		imageHeight:              defaultImageHeight,
		windowSize:               defaultWindowSize,
		frameStride:              defaultFrameStride,
		sampleStride:             defaultSampleStride,
		labelTolerance:           defaultLabelTolerance,
		telemetryOffsets:         defaultTelemetryHistoryOffsets(),
		futureOffsets:            defaultFutureOffsets(),
		telemetrySampleInterval:  defaultTelemetrySampleInterval,
		flashBrightnessThreshold: defaultFlashBrightnessThreshold,
		flashFrameLimit:          defaultFlashFrameLimit,
	}

	for _, opt := range opts {
		opt(p)
	}

	if p.windowSize < 1 || p.windowSize%2 == 0 {
		p.windowSize = defaultWindowSize
	}
	if p.frameStride < 1 {
		p.frameStride = defaultFrameStride
	}
	if !validImageOffsets(p.imageOffsets) {
		p.imageOffsets = deriveImageOffsets(p.windowSize, p.frameStride)
	}
	if p.imageWidth < 1 {
		p.imageWidth = defaultImageWidth
	}
	if p.imageHeight < 1 {
		p.imageHeight = defaultImageHeight
	}
	if p.sampleStride < 1 {
		p.sampleStride = defaultSampleStride
	}
	if p.labelTolerance <= 0 {
		p.labelTolerance = defaultLabelTolerance
	}
	if p.flashFrameLimit < 1 {
		p.flashFrameLimit = defaultFlashFrameLimit
	}

	return p
}

func WithCommandFactory(factory func(ctx context.Context, name string, args ...string) *exec.Cmd) Option {
	return func(p *Processor) {
		if factory != nil {
			p.newCommand = factory
		}
	}
}

func WithImageSize(width int, height int) Option {
	return func(p *Processor) {
		p.imageWidth = width
		p.imageHeight = height
	}
}

func WithSamplingConfig(size int, frameStride int, sampleStride int) Option {
	return func(p *Processor) {
		p.windowSize = size
		p.frameStride = frameStride
		p.sampleStride = sampleStride
	}
}

// WithImageOffsets sets the exact frame indices selected relative to each anchor frame.
func WithImageOffsets(offsets []int) Option {
	return func(p *Processor) {
		p.imageOffsets = append([]int(nil), offsets...)
	}
}

func WithLabelTolerance(tolerance time.Duration) Option {
	return func(p *Processor) {
		p.labelTolerance = tolerance
	}
}

// WithTelemetryTimelineConfig sets the model history, horizon, and base telemetry cadence.
func WithTelemetryTimelineConfig(telemetryOffsets []int, futureOffsets []int, sampleInterval time.Duration) Option {
	return func(p *Processor) {
		p.telemetryOffsets = append([]int(nil), telemetryOffsets...)
		p.futureOffsets = append([]int(nil), futureOffsets...)
		p.telemetrySampleInterval = sampleInterval
	}
}

func WithSyncFlashDetection(brightnessThreshold float64, frameLimit int) Option {
	return func(p *Processor) {
		p.flashBrightnessThreshold = brightnessThreshold
		p.flashFrameLimit = frameLimit
	}
}

func WithForce(force bool) Option {
	return func(p *Processor) {
		p.force = force
	}
}

func WithDatasetOnly(datasetOnly bool) Option {
	return func(p *Processor) {
		p.datasetOnly = datasetOnly
	}
}

type processingFingerprintConfig struct {
	Version                  int           `json:"version"`
	ImageWidth               int           `json:"image_width"`
	ImageHeight              int           `json:"image_height"`
	ImageOffsets             []int         `json:"image_offsets"`
	SampleStride             int           `json:"sample_stride"`
	LabelTolerance           time.Duration `json:"label_tolerance_ns"`
	TelemetryOffsets         []int         `json:"telemetry_offsets"`
	FutureOffsets            []int         `json:"future_offsets"`
	TelemetrySampleInterval  time.Duration `json:"telemetry_sample_interval_ns"`
	FlashBrightnessThreshold float64       `json:"flash_brightness_threshold"`
	FlashFrameLimit          int           `json:"flash_frame_limit"`
	StoppedSampleBurst       int           `json:"stopped_sample_burst"`
	StoppedSampleSpacing     float64       `json:"stopped_sample_spacing"`
	FutureSmoothingRadius    int           `json:"future_smoothing_radius"`
}

type processingWorkspace struct {
	root        string
	framesDir   string
	datasetPath string
	journalPath string
}

type promotionJournal struct {
	Version       int                         `json:"version"`
	Phase         string                      `json:"phase"`
	IncludeFrames bool                        `json:"includeFrames"`
	Operations    []promotionJournalOperation `json:"operations"`
}

type promotionJournalOperation struct {
	Name           string `json:"name"`
	HadPrior       bool   `json:"hadPrior"`
	BackupStarted  bool   `json:"backupStarted"`
	BackupDone     bool   `json:"backupDone"`
	PublishStarted bool   `json:"publishStarted"`
	PublishDone    bool   `json:"publishDone"`
	RestoreStarted bool   `json:"restoreStarted"`
	RestoreDone    bool   `json:"restoreDone"`
	RemoveStarted  bool   `json:"removeStarted"`
	RemoveDone     bool   `json:"removeDone"`
}

// ConfigFingerprint identifies every processor setting that affects published output.
func (p *Processor) ConfigFingerprint() string {
	payload := processingFingerprintConfig{
		Version:                  processingConfigVersion,
		ImageWidth:               p.imageWidth,
		ImageHeight:              p.imageHeight,
		ImageOffsets:             append([]int(nil), p.imageOffsets...),
		SampleStride:             p.sampleStride,
		LabelTolerance:           p.labelTolerance,
		TelemetryOffsets:         append([]int(nil), p.telemetryOffsets...),
		FutureOffsets:            append([]int(nil), p.futureOffsets...),
		TelemetrySampleInterval:  p.telemetrySampleInterval,
		FlashBrightnessThreshold: p.flashBrightnessThreshold,
		FlashFrameLimit:          p.flashFrameLimit,
		StoppedSampleBurst:       defaultStoppedSampleBurst,
		StoppedSampleSpacing:     defaultStoppedSampleSpacing,
		FutureSmoothingRadius:    futureTargetSmoothingRadius,
	}
	body, err := json.Marshal(payload)
	if err != nil {
		panic(fmt.Sprintf("dataset processing fingerprint config must be JSON serializable: %v", err))
	}
	return fmt.Sprintf("sha256:%x", sha256.Sum256(body))
}

func (p *Processor) Queue(tripDir string) (statusPath string, err error) {
	statusPath, err = resolveStatusPath(tripDir)
	if err != nil {
		return "", err
	}
	tripPath, err := resolveTripDir(tripDir)
	if err != nil {
		return "", err
	}
	lock, err := acquireTripProcessingLock(tripPath)
	if err != nil {
		return "", err
	}
	defer func() {
		err = errors.Join(err, lock.Release())
	}()
	if err := recoverStaleProcessingArtifacts(tripPath); err != nil {
		return "", err
	}
	if !p.force && p.shouldSkipTrip(tripPath) {
		status, readErr := ReadStatusFile(statusPath)
		if readErr != nil {
			return "", readErr
		}
		status.State = "skipped"
		status.CompletedAt = time.Now().Format(time.RFC3339)
		status.Error = ""
		return statusPath, writeStatusFile(statusPath, status)
	}
	status := p.newProcessingStatus("queued")
	return statusPath, writeStatusFile(statusPath, status)
}

func (p *Processor) ProcessTrip(ctx context.Context, tripDir string) (err error) {
	tripPath, err := resolveTripDir(tripDir)
	if err != nil {
		return err
	}
	lock, err := acquireTripProcessingLock(tripPath)
	if err != nil {
		return err
	}
	defer func() {
		err = errors.Join(err, lock.Release())
	}()
	if err := recoverStaleProcessingArtifacts(tripPath); err != nil {
		return err
	}
	if !p.force && p.shouldSkipTrip(tripPath) {
		return p.writeSkippedStatus(tripPath)
	}

	statusPath := filepath.Join(tripPath, "processing.json")
	status := p.newProcessingStatus("running")
	status.StartedAt = time.Now().Format(time.RFC3339)
	if err := writeStatusFile(statusPath, status); err != nil {
		return err
	}

	workspace, err := newProcessingWorkspace(tripPath)
	if err != nil {
		return finishProcessingError(ctx, statusPath, status, err)
	}

	if err := p.buildStagedOutputs(ctx, tripPath, workspace, &status); err != nil {
		cleanupErr := removeProcessingWorkspace(workspace)
		return finishProcessingError(ctx, statusPath, status, errors.Join(err, cleanupErr))
	}
	if err := promoteProcessingWorkspace(tripPath, workspace, !p.datasetOnly); err != nil {
		rollbackErr := rollbackAndCleanupProcessingWorkspace(tripPath, workspace)
		return finishProcessingError(ctx, statusPath, status, errors.Join(err, rollbackErr))
	}

	status.State = "completed"
	status.CompletedAt = time.Now().Format(time.RFC3339)
	status.Error = ""
	if err := writeStatusFile(statusPath, status); err != nil {
		rollbackErr := rollbackAndCleanupProcessingWorkspace(tripPath, workspace)
		return finishProcessingError(ctx, statusPath, status, errors.Join(err, rollbackErr))
	}
	_ = removeProcessingWorkspace(workspace)
	return nil
}

func (p *Processor) buildStagedOutputs(
	ctx context.Context,
	tripPath string,
	workspace processingWorkspace,
	status *ProcessingStatus,
) error {
	if status == nil {
		panic("dataset processing status must not be nil")
	}

	videoPath := filepath.Join(tripPath, "video.mkv")
	metadataPath := filepath.Join(tripPath, "metadata.json")
	runFilePath := filepath.Join(filepath.Dir(tripPath), "run.jsonl")

	if !fileExists(videoPath) || !fileExists(metadataPath) || !fileExists(runFilePath) {
		return fmt.Errorf("%w: expected video.mkv, metadata.json, and run.jsonl at %s", ErrMissingTripFiles, runFilePath)
	}

	metadata, err := loadTripMetadata(metadataPath)
	if err != nil {
		return err
	}

	frames, err := p.ProbeVideoFrames(ctx, videoPath)
	if err != nil {
		return err
	}
	if len(frames) == 0 {
		return errors.New("ffprobe returned no video frames")
	}

	frameRoot := workspace.root
	if p.datasetOnly {
		frameRoot = tripPath
		frames = AttachImagePaths(frames, "frames")
	} else {
		if err := extractFrames(ctx, p.newCommand, p.ffmpegBin, videoPath, workspace.framesDir, p.imageWidth, p.imageHeight); err != nil {
			return err
		}

		frames = AttachImagePaths(frames, "frames")
	}
	if err := validateFrameSet(frameRoot, frames, p.imageWidth, p.imageHeight); err != nil {
		return err
	}

	anchorPTS, err := detectSyncFlashPTS(frameRoot, frames, p.flashFrameLimit, p.flashBrightnessThreshold)
	if err != nil {
		return err
	}

	record, err := loadRunTripRecord(runFilePath, metadata.RunID, metadata.TripIndex)
	if err != nil {
		return err
	}

	labels := buildTimedLabels(record.VehicleData, metadata.SyncTime)
	stopSignTimeline := isStopSignTimeline(labels)
	samples, sampleStats := buildDatasetSamplesWithImageOffsetsAndStats(
		frames,
		labels,
		anchorPTS,
		p.imageOffsets,
		p.sampleStride,
		p.labelTolerance,
		p.telemetryOffsets,
		p.futureOffsets,
		p.telemetrySampleInterval,
	)
	if stopSignTimeline {
		samples = decorateStopSignSamples(samples, metadata)
	} else {
		samples = thinStoppedSamples(samples, defaultStoppedSampleBurst, defaultStoppedSampleSpacing)
	}
	retainedFrameCount := len(frames)
	if !p.datasetOnly && !stopSignTimeline {
		retainedFrameCount, err = pruneUnreferencedJPEGFrames(workspace.framesDir, samples)
		if err != nil {
			return err
		}
	}
	if err := writeDatasetFile(workspace.datasetPath, samples); err != nil {
		return err
	}

	status.FrameCount = retainedFrameCount
	status.SampleCount = len(samples)
	if len(samples) == 0 {
		status.Warning = "no dataset samples generated"
		status.ZeroSampleReasons = sampleStats.zeroSampleReasons()
	}
	return nil
}

func (p *Processor) newProcessingStatus(state string) ProcessingStatus {
	return ProcessingStatus{
		State:                     state,
		ConfigFingerprint:         p.ConfigFingerprint(),
		FramesDir:                 "frames",
		DatasetFile:               "dataset.jsonl",
		ImageWidth:                p.imageWidth,
		ImageHeight:               p.imageHeight,
		ImageOffsets:              append([]int(nil), p.imageOffsets...),
		TelemetryOffsets:          append([]int(nil), p.telemetryOffsets...),
		FutureOffsets:             append([]int(nil), p.futureOffsets...),
		TelemetrySampleIntervalMs: float64(p.telemetrySampleInterval) / float64(time.Millisecond),
	}
}

func (p *Processor) writeSkippedStatus(tripPath string) error {
	statusPath := filepath.Join(tripPath, "processing.json")
	status, err := ReadStatusFile(statusPath)
	if err != nil {
		return err
	}
	status.State = "skipped"
	status.CompletedAt = time.Now().Format(time.RFC3339)
	status.Error = ""
	return writeStatusFile(statusPath, status)
}

func failProcessing(statusPath string, status ProcessingStatus, processingErr error) error {
	status.State = "failed"
	status.CompletedAt = ""
	status.Error = processingErr.Error()
	if statusErr := writeStatusFile(statusPath, status); statusErr != nil {
		return errors.Join(processingErr, fmt.Errorf("write failed processing status: %w", statusErr))
	}
	return processingErr
}

func finishProcessingError(
	ctx context.Context,
	statusPath string,
	status ProcessingStatus,
	processingErr error,
) error {
	if ctx != nil && ctx.Err() != nil {
		status.State = "queued"
		status.CompletedAt = ""
		status.Error = ""
		if statusErr := writeStatusFile(statusPath, status); statusErr != nil {
			return errors.Join(processingErr, fmt.Errorf("write interrupted processing status: %w", statusErr))
		}
		return processingErr
	}
	return failProcessing(statusPath, status, processingErr)
}

func rollbackAndCleanupProcessingWorkspace(tripPath string, workspace processingWorkspace) error {
	if !fileExists(workspace.journalPath) {
		return removeProcessingWorkspace(workspace)
	}
	if err := rollbackProcessingWorkspace(tripPath, workspace); err != nil {
		return fmt.Errorf("rollback processing promotion; workspace preserved at %s: %w", workspace.root, err)
	}
	return removeProcessingWorkspace(workspace)
}

func removeProcessingWorkspace(workspace processingWorkspace) error {
	if err := os.RemoveAll(workspace.root); err != nil {
		return fmt.Errorf("remove processing workspace %s: %w", workspace.root, err)
	}
	return nil
}

func newProcessingWorkspace(tripPath string) (processingWorkspace, error) {
	root, err := os.MkdirTemp(tripPath, processingWorkspacePrefix)
	if err != nil {
		return processingWorkspace{}, fmt.Errorf("create processing workspace: %w", err)
	}
	return processingWorkspaceForRoot(root), nil
}

func processingWorkspaceForRoot(root string) processingWorkspace {
	return processingWorkspace{
		root:        root,
		framesDir:   filepath.Join(root, "frames"),
		datasetPath: filepath.Join(root, "dataset.jsonl"),
		journalPath: filepath.Join(root, processingJournalFile),
	}
}

func (p *Processor) ProbeVideoFrames(ctx context.Context, videoPath string) ([]VideoFrame, error) {
	cmd := p.newCommand(
		ctx,
		p.ffprobeBin,
		"-select_streams", "v:0",
		"-show_frames",
		"-show_entries", "frame=pts_time",
		"-of", "json",
		videoPath,
	)

	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("ffprobe failed: %w", err)
	}

	var parsed ffprobeOutput
	if err := json.Unmarshal(output, &parsed); err != nil {
		return nil, fmt.Errorf("failed to parse ffprobe json: %w", err)
	}

	frames := make([]VideoFrame, 0, len(parsed.Frames))
	for i, frame := range parsed.Frames {
		if strings.TrimSpace(frame.PTSTime) == "" {
			return nil, fmt.Errorf("missing pts_time at frame %d", i)
		}
		var pts float64
		if err := json.Unmarshal([]byte(frame.PTSTime), &pts); err != nil {
			parsedValue, parseErr := parseFloat(frame.PTSTime)
			if parseErr != nil {
				return nil, fmt.Errorf("invalid pts_time at frame %d: %w", i, parseErr)
			}
			pts = parsedValue
		}
		frames = append(frames, VideoFrame{Index: i, PTS: pts})
	}

	return frames, nil
}

func AttachImagePaths(frames []VideoFrame, dir string) []VideoFrame {
	out := make([]VideoFrame, len(frames))
	copy(out, frames)
	cleanDir := filepath.ToSlash(strings.Trim(dir, "/\\"))
	for i := range out {
		out[i].ImagePath = fmt.Sprintf("%s/%06d.jpg", cleanDir, out[i].Index+1)
	}
	return out
}

func extractFrames(
	ctx context.Context,
	factory commandFactory,
	ffmpegBin string,
	videoPath string,
	framesDir string,
	imageWidth int,
	imageHeight int,
) error {
	if err := os.RemoveAll(framesDir); err != nil {
		return fmt.Errorf("failed to clear frames dir: %w", err)
	}
	if err := os.MkdirAll(framesDir, 0o755); err != nil {
		return fmt.Errorf("failed to create frames dir: %w", err)
	}

	outputPattern := filepath.Join(framesDir, "%06d.jpg")
	cmd := factory(ctx, ffmpegBin,
		"-y",
		"-i", videoPath,
		"-vsync", "0",
		"-vf", fmt.Sprintf("scale=%d:%d", imageWidth, imageHeight),
		"-q:v", "2",
		outputPattern,
	)
	if output, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("ffmpeg frame extraction failed: %w: %s", err, strings.TrimSpace(string(output)))
	}
	return nil
}

func detectSyncFlashPTS(tripDir string, frames []VideoFrame, limit int, threshold float64) (float64, error) {
	searchLimit := min(limit, len(frames))
	for i := 0; i < searchLimit; i++ {
		framePath := filepath.Join(tripDir, filepath.FromSlash(frames[i].ImagePath))
		brightness, err := measureAverageBrightness(framePath)
		if err != nil {
			continue
		}
		if brightness >= threshold {
			return frames[i].PTS, nil
		}
	}
	return 0, ErrSyncFlashNotFound
}

func measureAverageBrightness(imagePath string) (float64, error) {
	file, err := os.Open(imagePath)
	if err != nil {
		return 0, err
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		return 0, err
	}

	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()
	if width == 0 || height == 0 {
		return 0, errors.New("empty image")
	}

	xStep := max(1, width/64)
	yStep := max(1, height/64)
	var total float64
	var samples int
	for y := bounds.Min.Y; y < bounds.Max.Y; y += yStep {
		for x := bounds.Min.X; x < bounds.Max.X; x += xStep {
			r, g, b, _ := img.At(x, y).RGBA()
			brightness := (float64(r>>8) + float64(g>>8) + float64(b>>8)) / 3
			total += brightness
			samples++
		}
	}
	if samples == 0 {
		return 0, errors.New("no sampled pixels")
	}
	return total / float64(samples), nil
}

func loadTripMetadata(path string) (tripMetadata, error) {
	var metadata tripMetadata
	body, err := os.ReadFile(path)
	if err != nil {
		return metadata, fmt.Errorf("failed to read trip metadata: %w", err)
	}
	if err := json.Unmarshal(body, &metadata); err != nil {
		return metadata, fmt.Errorf("failed to parse trip metadata: %w", err)
	}
	if metadata.RunID == "" {
		return metadata, fmt.Errorf("failed to parse trip metadata: missing runId")
	}
	return metadata, nil
}

func loadRunTripRecord(path string, runID string, tripIndex int) (runTripRecord, error) {
	file, err := os.Open(path)
	if err != nil {
		return runTripRecord{}, fmt.Errorf("failed to open run.jsonl: %w", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 1024*1024), 128*1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var record runTripRecord
		if err := json.Unmarshal([]byte(line), &record); err != nil {
			return runTripRecord{}, fmt.Errorf("failed to parse run.jsonl row: %w", err)
		}
		if record.RunID == runID && record.TripIndex == tripIndex {
			return record, nil
		}
	}
	if err := scanner.Err(); err != nil {
		return runTripRecord{}, fmt.Errorf("failed to read run.jsonl: %w", err)
	}
	return runTripRecord{}, fmt.Errorf("%w: runId=%s tripIndex=%d", ErrTripRecordNotFound, runID, tripIndex)
}

func WaitForTripReadiness(ctx context.Context, tripDir string, pollInterval time.Duration) error {
	if pollInterval <= 0 {
		pollInterval = 500 * time.Millisecond
	}

	tripPath, err := resolveTripDir(tripDir)
	if err != nil {
		return err
	}

	var lastErr error
	for {
		if err := checkTripReadiness(tripPath); err == nil {
			return nil
		} else {
			lastErr = err
		}

		timer := time.NewTimer(pollInterval)
		select {
		case <-ctx.Done():
			timer.Stop()
			if lastErr != nil {
				return fmt.Errorf("trip not ready before timeout: %w", lastErr)
			}
			return ctx.Err()
		case <-timer.C:
		}
	}
}

func checkTripReadiness(tripPath string) error {
	videoPath := filepath.Join(tripPath, "video.mkv")
	metadataPath := filepath.Join(tripPath, "metadata.json")
	runFilePath := filepath.Join(filepath.Dir(tripPath), "run.jsonl")

	if !fileExists(videoPath) || !fileExists(metadataPath) || !fileExists(runFilePath) {
		return fmt.Errorf("%w: expected video.mkv, metadata.json, and run.jsonl at %s", ErrMissingTripFiles, runFilePath)
	}

	metadata, err := loadTripMetadata(metadataPath)
	if err != nil {
		return err
	}

	if _, err := loadRunTripRecord(runFilePath, metadata.RunID, metadata.TripIndex); err != nil {
		return err
	}
	return nil
}

func buildTimedLabels(vehicleData []map[string]any, syncTime float64) []timedLabel {
	labels := make([]timedLabel, 0, len(vehicleData))
	for _, label := range vehicleData {
		rawTime, ok := numberField(label["time"])
		if !ok || !isFiniteFloat64(rawTime) {
			continue
		}
		labels = append(labels, timedLabel{
			RelativeSeconds: (rawTime - syncTime) / 1000.0,
			Label:           label,
		})
	}
	sort.Slice(labels, func(i, j int) bool {
		return labels[i].RelativeSeconds < labels[j].RelativeSeconds
	})
	return labels
}

func buildDatasetSamples(
	frames []VideoFrame,
	labels []timedLabel,
	anchorPTS float64,
	windowSize int,
	frameStride int,
	sampleStride int,
	tolerance time.Duration,
	telemetryOffsets []int,
	futureOffsets []int,
	telemetrySampleInterval time.Duration,
) []DatasetSample {
	samples, _ := buildDatasetSamplesWithStats(
		frames,
		labels,
		anchorPTS,
		windowSize,
		frameStride,
		sampleStride,
		tolerance,
		telemetryOffsets,
		futureOffsets,
		telemetrySampleInterval,
	)
	return samples
}

func buildDatasetSamplesWithStats(
	frames []VideoFrame,
	labels []timedLabel,
	anchorPTS float64,
	windowSize int,
	frameStride int,
	sampleStride int,
	tolerance time.Duration,
	telemetryOffsets []int,
	futureOffsets []int,
	telemetrySampleInterval time.Duration,
) ([]DatasetSample, sampleBuildStats) {
	return buildDatasetSamplesWithImageOffsetsAndStats(
		frames,
		labels,
		anchorPTS,
		deriveImageOffsets(windowSize, frameStride),
		sampleStride,
		tolerance,
		telemetryOffsets,
		futureOffsets,
		telemetrySampleInterval,
	)
}

func buildDatasetSamplesWithImageOffsetsAndStats(
	frames []VideoFrame,
	labels []timedLabel,
	anchorPTS float64,
	imageOffsets []int,
	sampleStride int,
	tolerance time.Duration,
	telemetryOffsets []int,
	futureOffsets []int,
	telemetrySampleInterval time.Duration,
) ([]DatasetSample, sampleBuildStats) {
	var stats sampleBuildStats
	if len(frames) == 0 ||
		len(labels) == 0 ||
		!validImageOffsets(imageOffsets) ||
		sampleStride < 1 ||
		!validTelemetryOffsets(telemetryOffsets) ||
		!validFutureOffsets(futureOffsets) ||
		telemetrySampleInterval <= 0 {
		return nil, stats
	}
	if isStopSignTimeline(labels) {
		return buildStopSignSamplesWithImageOffsetsAndStats(
			frames,
			labels,
			anchorPTS,
			imageOffsets,
			sampleStride,
			tolerance,
			telemetryOffsets,
			futureOffsets,
			telemetrySampleInterval,
		)
	}
	toleranceSeconds := tolerance.Seconds()
	telemetryAlignmentTolerance := telemetryAlignmentTolerance(tolerance, telemetrySampleInterval)
	samples := make([]DatasetSample, 0)
	for anchorIndex := 0; anchorIndex < len(frames); anchorIndex += sampleStride {
		stats.CandidateWindowCount++

		window, ok := buildFrameWindowAtOffsets(frames, anchorIndex, imageOffsets)
		if !ok {
			stats.IncompleteFrameHistoryCount++
			continue
		}
		anchorFrame := frames[anchorIndex]
		relativeSeconds := anchorFrame.PTS - anchorPTS
		label, labelIndex, ok := nearestLabelWithIndex(labels, relativeSeconds, toleranceSeconds)
		if !ok {
			stats.MissingCurrentLabelCount++
			continue
		}
		telemetryHistory, ok := buildTelemetryHistory(
			labels,
			labelIndex,
			telemetryOffsets,
			telemetrySampleInterval,
			telemetryAlignmentTolerance,
		)
		if !ok {
			stats.IncompleteTelemetryHistoryCount++
			continue
		}
		telemetryFuture, ok := buildTelemetryFuture(
			labels,
			labelIndex,
			futureOffsets,
			telemetrySampleInterval,
			telemetryAlignmentTolerance,
		)
		if !ok {
			stats.IncompleteTelemetryFutureCount++
			continue
		}

		futureCenter := anchorIndex + sampleStride
		if futureCenter >= len(frames) {
			stats.IncompleteTelemetryFutureCount++
			continue
		}
		futureRelativeSeconds := frames[futureCenter].PTS - anchorPTS
		futureLabel, futureLabelIndex, ok := nearestLabelWithIndex(labels, futureRelativeSeconds, toleranceSeconds)
		if !ok {
			stats.MissingFutureLabelCount++
			continue
		}
		futureSpeedTarget, ok := smoothedFutureSpeed(labels, futureLabelIndex, futureTargetSmoothingRadius)
		if !ok {
			stats.MissingFutureSpeedTargetCount++
			continue
		}
		routeForwardDeltaTarget, ok := smoothedRouteForwardDelta(labels, labelIndex, futureTargetSmoothingRadius)
		if !ok {
			stats.MissingRouteForwardDeltaCount++
			continue
		}
		futureHorizonSeconds := futureLabel.RelativeSeconds - label.RelativeSeconds
		if !isFiniteFloat64(futureHorizonSeconds) || futureHorizonSeconds <= 0 {
			stats.InvalidFutureHorizonCount++
			continue
		}
		derivedLabel, ok := buildTrainingLabel(
			label.Label,
			futureLabel.Label,
			futureSpeedTarget,
			routeForwardDeltaTarget,
			futureHorizonSeconds,
		)
		if !ok {
			stats.InvalidDerivedLabelCount++
			continue
		}
		samples = append(samples, DatasetSample{
			AnchorVideoPTS:   anchorFrame.PTS,
			AnchorGameTime:   label.RelativeSeconds,
			FramePaths:       window,
			TelemetryHistory: telemetryHistory,
			TelemetryFuture:  telemetryFuture,
			Label:            derivedLabel,
		})
		stats.GeneratedSampleCount++
	}

	return samples, stats
}

func buildPastOnlyFrameWindow(frames []VideoFrame, anchorIndex int, windowSize int, frameStride int) ([]string, bool) {
	return buildFrameWindowAtOffsets(frames, anchorIndex, deriveImageOffsets(windowSize, frameStride))
}

func buildFrameWindowAtOffsets(frames []VideoFrame, anchorPosition int, imageOffsets []int) ([]string, bool) {
	if len(frames) == 0 || anchorPosition < 0 || anchorPosition >= len(frames) || !validImageOffsets(imageOffsets) {
		return nil, false
	}
	window := make([]string, 0, len(imageOffsets))
	anchorFrameIndex := frames[anchorPosition].Index
	for _, offset := range imageOffsets {
		targetFrameIndex := anchorFrameIndex + offset
		position := sort.Search(len(frames), func(index int) bool {
			return frames[index].Index >= targetFrameIndex
		})
		if position >= len(frames) || frames[position].Index != targetFrameIndex {
			return nil, false
		}
		window = append(window, frames[position].ImagePath)
	}
	return window, true
}

func buildTelemetryHistory(
	labels []timedLabel,
	anchorIndex int,
	telemetryOffsets []int,
	sampleInterval time.Duration,
	alignmentTolerance time.Duration,
) ([]GroupedTelemetryItem, bool) {
	if anchorIndex < 0 || anchorIndex >= len(labels) || !validTelemetryOffsets(telemetryOffsets) || sampleInterval <= 0 || alignmentTolerance < 0 {
		return nil, false
	}

	// Keep the serialized window dense because trainer offsets address it relative to the final row.
	firstOffset := telemetryOffsets[0]
	window := make([]GroupedTelemetryItem, 0, -firstOffset+1)
	previousIndex := -1
	anchorTime := labels[anchorIndex].RelativeSeconds
	if !isFiniteFloat64(anchorTime) {
		return nil, false
	}
	for offset := firstOffset; offset <= 0; offset++ {
		label := labels[anchorIndex]
		labelIndex := anchorIndex
		ok := true
		if offset < 0 {
			targetTime := anchorTime + (float64(offset) * sampleInterval.Seconds())
			label, labelIndex, ok = nearestLabelWithIndexRange(
				labels,
				targetTime,
				alignmentTolerance.Seconds(),
				previousIndex+1,
				anchorIndex-1,
			)
		}
		if !ok || labelIndex <= previousIndex || labelIndex > anchorIndex || !isFiniteFloat64(label.RelativeSeconds) {
			return nil, false
		}
		window = append(window, groupTelemetryItem(label.Label))
		previousIndex = labelIndex
	}
	return window, true
}

func buildTelemetryFuture(
	labels []timedLabel,
	anchorIndex int,
	futureOffsets []int,
	sampleInterval time.Duration,
	alignmentTolerance time.Duration,
) ([]GroupedTelemetryItem, bool) {
	if anchorIndex < 0 || anchorIndex >= len(labels) || !validFutureOffsets(futureOffsets) || sampleInterval <= 0 || alignmentTolerance < 0 {
		return nil, false
	}

	// Keep the serialized window dense because trainer offsets address it by offset-1.
	horizonLength := futureOffsets[len(futureOffsets)-1]
	window := make([]GroupedTelemetryItem, 0, horizonLength)
	previousIndex := anchorIndex
	anchorTime := labels[anchorIndex].RelativeSeconds
	if !isFiniteFloat64(anchorTime) {
		return nil, false
	}
	for offset := 1; offset <= horizonLength; offset++ {
		targetTime := anchorTime + (float64(offset) * sampleInterval.Seconds())
		label, labelIndex, ok := nearestLabelWithIndexRange(
			labels,
			targetTime,
			alignmentTolerance.Seconds(),
			previousIndex+1,
			len(labels)-1,
		)
		if !ok || labelIndex <= previousIndex || !isFiniteFloat64(label.RelativeSeconds) {
			return nil, false
		}
		window = append(window, groupTelemetryItem(label.Label))
		previousIndex = labelIndex
	}
	return window, true
}

func telemetryAlignmentTolerance(labelTolerance time.Duration, sampleInterval time.Duration) time.Duration {
	if labelTolerance <= 0 || sampleInterval <= 0 {
		return 0
	}
	maxJitter := sampleInterval / 2
	if labelTolerance < maxJitter {
		return labelTolerance
	}
	return maxJitter
}

func defaultTelemetryHistoryOffsets() []int {
	return []int{-4, -3, -2, -1, 0}
}

func deriveImageOffsets(windowSize int, frameStride int) []int {
	if windowSize < 1 || frameStride < 1 {
		return nil
	}
	offsets := make([]int, 0, windowSize)
	firstOffset := -((windowSize - 1) * frameStride)
	for index := 0; index < windowSize; index++ {
		offsets = append(offsets, firstOffset+(index*frameStride))
	}
	return offsets
}

func validImageOffsets(offsets []int) bool {
	if len(offsets) == 0 || offsets[len(offsets)-1] != 0 {
		return false
	}
	for index, offset := range offsets {
		if offset > 0 || (index > 0 && offset <= offsets[index-1]) {
			return false
		}
	}
	return true
}

func defaultFutureOffsets() []int {
	offsets := make([]int, defaultFutureTelemetryCount)
	for index := range offsets {
		offsets[index] = index + 1
	}
	return offsets
}

func validFutureOffsets(offsets []int) bool {
	if len(offsets) == 0 {
		return false
	}
	for index, offset := range offsets {
		if offset < 1 || (index > 0 && offset <= offsets[index-1]) {
			return false
		}
	}
	return true
}

func validTelemetryOffsets(offsets []int) bool {
	if len(offsets) == 0 || offsets[len(offsets)-1] != 0 {
		return false
	}
	for index, offset := range offsets {
		if offset > 0 || (index > 0 && offset <= offsets[index-1]) {
			return false
		}
	}
	return true
}

func buildTrainingLabel(
	current map[string]any,
	future map[string]any,
	futureSpeedTarget float64,
	routeForwardDeltaTarget float64,
	futureHorizonSeconds float64,
) (GroupedLabel, bool) {
	futureSpeed, ok := numberField(future["currentSpeed"])
	if !ok {
		return GroupedLabel{}, false
	}
	if !isFiniteFloat64(futureSpeedTarget) || !isFiniteFloat64(routeForwardDeltaTarget) ||
		!isFiniteFloat64(futureHorizonSeconds) || futureHorizonSeconds <= 0 {
		return GroupedLabel{}, false
	}

	derived := GroupedLabel{}
	if steering, ok := current["Steering"]; ok {
		derived.Control.Steering = cloneValue(steering)
	}
	derived.Aux.FutureSpeed = futureSpeed
	derived.Aux.FutureSpeedTarget = futureSpeedTarget
	derived.Aux.FutureHorizonSeconds = futureHorizonSeconds
	derived.Aux.RouteForwardDelta = routeForwardDeltaTarget
	return derived, true
}

func groupTelemetryItem(source map[string]any) GroupedTelemetryItem {
	item := GroupedTelemetryItem{}
	raw := make(map[string]any)

	for key, value := range source {
		cloned := cloneValue(value)
		switch key {
		case "Steering":
			item.Control.Steering = cloned
		case "acceleration":
			item.Control.Acceleration = cloned
		case "brakePressureAvg":
			item.Control.BrakePressureAvg = cloned
		case "expertDesiredWheelSteerNormalized":
			item.Control.ExpertDesiredWheelSteerNormalized = cloned
		case "expertDesiredSpeedMps":
			item.Control.ExpertDesiredSpeedMps = cloned
		case "expertThrottle":
			item.Control.ExpertThrottle = cloned
		case "expertBrake":
			item.Control.ExpertBrake = cloned
		case "expertStopProbability":
			item.Control.ExpertStopProbability = cloned
		case "expertGoProbability":
			item.Control.ExpertGoProbability = cloned
		case "currentSpeed":
			item.Aux.CurrentSpeed = cloned
		case "yaw":
			item.Aux.Yaw = cloned
		case "yawRate":
			item.Aux.YawRate = cloned
		case "routeDirectionCode":
			item.Aux.RouteDirectionCode = cloned
		case "routeDirectionDistanceM":
			item.Aux.RouteDirectionDistanceM = cloned
		case "routeDirectionUnknown":
			item.Aux.RouteDirectionUnknown = cloned
		case "routeDirectionKeepStraight":
			item.Aux.RouteDirectionKeepStraight = cloned
		case "routeDirectionTurnLeft":
			item.Aux.RouteDirectionTurnLeft = cloned
		case "routeDirectionTurnRight":
			item.Aux.RouteDirectionTurnRight = cloned
		case "routeDirectionRerouteWrongWay":
			item.Aux.RouteDirectionRerouteWrongWay = cloned
		case "routeForwardDelta":
			item.Aux.RouteForwardDelta = cloned
		case "routeHeadingError":
			item.Aux.RouteHeadingError = cloned
		case "routeDistance":
			item.Aux.RouteDistance = cloned
		case "leadVehicleDistance":
			item.Aux.LeadVehicleDistance = cloned
		case "hasLeadVehicle":
			item.Aux.HasLeadVehicle = cloned
		case "gps":
			item.Aux.GPS = cloned
		case "isStopped":
			item.Aux.IsStopped = cloned
		case "routeGpsValid":
			item.Aux.RouteGPSValid = cloned
		case "isStoppedAtTrafficLights":
			item.Aux.IsStoppedAtTraffic = cloned
		case "leadVehicleRelativeSpeed":
			item.Aux.LeadVehicleRelSpeed = cloned
		case "leadVehicleHeadingDelta":
			item.Aux.LeadVehicleHeadingDiff = cloned
		case "leadVehicleTTC":
			item.Aux.LeadVehicleTTC = cloned
		case "stopSignPhase":
			item.Aux.StopSignPhase = cloned
		case "stopLineDistanceM":
			item.Aux.StopLineDistanceM = cloned
		case "stopSignLongitudinalErrorM":
			item.Aux.StopSignLongitudinalErrorM = cloned
		case "stopSignLateralErrorM":
			item.Aux.StopSignLateralErrorM = cloned
		case "stopSignHeadingErrorDeg":
			item.Aux.StopSignHeadingErrorDeg = cloned
		default:
			raw[key] = cloned
		}
	}

	if len(raw) > 0 {
		item.Raw = raw
	}
	return item
}

func flattenGroupedLabel(label GroupedLabel) map[string]any {
	flat := make(map[string]any)
	if label.Control.Steering != nil {
		flat["Steering"] = cloneValue(label.Control.Steering)
	}
	appendIfPresent(flat, "expertDesiredWheelSteerNormalized", label.Control.ExpertDesiredWheelSteerNormalized)
	appendIfPresent(flat, "expertDesiredSpeedMps", label.Control.ExpertDesiredSpeedMps)
	appendIfPresent(flat, "expertThrottle", label.Control.ExpertThrottle)
	appendIfPresent(flat, "expertBrake", label.Control.ExpertBrake)
	appendIfPresent(flat, "expertStopProbability", label.Control.ExpertStopProbability)
	appendIfPresent(flat, "expertGoProbability", label.Control.ExpertGoProbability)
	appendIfPresent(flat, "future_speed", label.Aux.FutureSpeed)
	appendIfPresent(flat, "future_speed_target", label.Aux.FutureSpeedTarget)
	appendIfPresent(flat, "future_horizon_seconds", label.Aux.FutureHorizonSeconds)
	appendIfPresent(flat, "routeForwardDelta", label.Aux.RouteForwardDelta)
	appendIfPresent(flat, "stopSignPhase", label.Aux.StopSignPhase)
	return flat
}

func flattenGroupedTelemetry(item GroupedTelemetryItem) map[string]any {
	flat := make(map[string]any)
	appendIfPresent(flat, "Steering", item.Control.Steering)
	appendIfPresent(flat, "acceleration", item.Control.Acceleration)
	appendIfPresent(flat, "brakePressureAvg", item.Control.BrakePressureAvg)
	appendIfPresent(flat, "expertDesiredWheelSteerNormalized", item.Control.ExpertDesiredWheelSteerNormalized)
	appendIfPresent(flat, "expertDesiredSpeedMps", item.Control.ExpertDesiredSpeedMps)
	appendIfPresent(flat, "expertThrottle", item.Control.ExpertThrottle)
	appendIfPresent(flat, "expertBrake", item.Control.ExpertBrake)
	appendIfPresent(flat, "expertStopProbability", item.Control.ExpertStopProbability)
	appendIfPresent(flat, "expertGoProbability", item.Control.ExpertGoProbability)
	appendIfPresent(flat, "currentSpeed", item.Aux.CurrentSpeed)
	appendIfPresent(flat, "yaw", item.Aux.Yaw)
	appendIfPresent(flat, "yawRate", item.Aux.YawRate)
	appendIfPresent(flat, "routeDirectionCode", item.Aux.RouteDirectionCode)
	appendIfPresent(flat, "routeDirectionDistanceM", item.Aux.RouteDirectionDistanceM)
	appendIfPresent(flat, "routeDirectionUnknown", item.Aux.RouteDirectionUnknown)
	appendIfPresent(flat, "routeDirectionKeepStraight", item.Aux.RouteDirectionKeepStraight)
	appendIfPresent(flat, "routeDirectionTurnLeft", item.Aux.RouteDirectionTurnLeft)
	appendIfPresent(flat, "routeDirectionTurnRight", item.Aux.RouteDirectionTurnRight)
	appendIfPresent(flat, "routeDirectionRerouteWrongWay", item.Aux.RouteDirectionRerouteWrongWay)
	appendIfPresent(flat, "routeForwardDelta", item.Aux.RouteForwardDelta)
	appendIfPresent(flat, "routeHeadingError", item.Aux.RouteHeadingError)
	appendIfPresent(flat, "routeDistance", item.Aux.RouteDistance)
	appendIfPresent(flat, "leadVehicleDistance", item.Aux.LeadVehicleDistance)
	appendIfPresent(flat, "hasLeadVehicle", item.Aux.HasLeadVehicle)
	appendIfPresent(flat, "gps", item.Aux.GPS)
	appendIfPresent(flat, "isStopped", item.Aux.IsStopped)
	appendIfPresent(flat, "routeGpsValid", item.Aux.RouteGPSValid)
	appendIfPresent(flat, "isStoppedAtTrafficLights", item.Aux.IsStoppedAtTraffic)
	appendIfPresent(flat, "leadVehicleRelativeSpeed", item.Aux.LeadVehicleRelSpeed)
	appendIfPresent(flat, "leadVehicleHeadingDelta", item.Aux.LeadVehicleHeadingDiff)
	appendIfPresent(flat, "leadVehicleTTC", item.Aux.LeadVehicleTTC)
	appendIfPresent(flat, "stopSignPhase", item.Aux.StopSignPhase)
	appendIfPresent(flat, "stopLineDistanceM", item.Aux.StopLineDistanceM)
	appendIfPresent(flat, "stopSignLongitudinalErrorM", item.Aux.StopSignLongitudinalErrorM)
	appendIfPresent(flat, "stopSignLateralErrorM", item.Aux.StopSignLateralErrorM)
	appendIfPresent(flat, "stopSignHeadingErrorDeg", item.Aux.StopSignHeadingErrorDeg)
	for key, value := range item.Raw {
		flat[key] = cloneValue(value)
	}
	return flat
}

func appendIfPresent(target map[string]any, key string, value any) {
	if value == nil {
		return
	}
	target[key] = cloneValue(value)
}

func smoothedFutureSpeed(labels []timedLabel, centerIndex int, radius int) (float64, bool) {
	if centerIndex < 0 || centerIndex >= len(labels) {
		return 0, false
	}
	startIndex := centerIndex - radius
	if startIndex < 0 {
		startIndex = 0
	}
	endIndex := centerIndex + radius
	if endIndex >= len(labels) {
		endIndex = len(labels) - 1
	}

	var sum float64
	count := 0
	for index := startIndex; index <= endIndex; index++ {
		speed, ok := numberField(labels[index].Label["currentSpeed"])
		if !ok {
			continue
		}
		sum += speed
		count++
	}
	if count == 0 {
		return 0, false
	}
	return sum / float64(count), true
}

func smoothedRouteForwardDelta(labels []timedLabel, centerIndex int, radius int) (float64, bool) {
	if centerIndex < 0 || centerIndex >= len(labels) {
		return 0, false
	}
	startIndex := centerIndex - radius
	if startIndex < 0 {
		startIndex = 0
	}
	endIndex := centerIndex + radius
	if endIndex >= len(labels) {
		endIndex = len(labels) - 1
	}

	var sum float64
	count := 0
	for index := startIndex; index <= endIndex; index++ {
		routeForwardDelta, ok := resolvedRouteForwardDelta(labels[index].Label)
		if !ok || !isFiniteFloat64(routeForwardDelta) {
			continue
		}
		sum += routeForwardDelta
		count++
	}
	if count == 0 {
		return 0, false
	}
	return sum / float64(count), true
}

func resolvedRouteForwardDelta(label map[string]any) (float64, bool) {
	if routeForwardDelta, ok := numberField(label["routeForwardDelta"]); ok && isFiniteFloat64(routeForwardDelta) {
		return routeForwardDelta, true
	}

	coords, ok := vector3Field(label["coords"])
	if !ok {
		return 0, false
	}
	gps, ok := vector3Field(label["gps"])
	if !ok {
		return 0, false
	}
	yaw, ok := numberField(label["yaw"])
	if !ok || !isFiniteFloat64(yaw) {
		return 0, false
	}

	forwardX, forwardY := headingForwardVector(yaw)
	deltaX := gps[0] - coords[0]
	deltaY := gps[1] - coords[1]
	deltaZ := gps[2] - coords[2]
	return (deltaX * forwardX) + (deltaY * forwardY) + (deltaZ * 0.0), true
}

func headingForwardVector(heading float64) (float64, float64) {
	radians := degreesToRadians(heading)
	return math.Sin(radians), math.Cos(radians)
}

func isFiniteFloat64(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func degreesToRadians(value float64) float64 {
	return value * math.Pi / 180.0
}

func clampInt(value int, minimum int, maximum int) int {
	if value < minimum {
		return minimum
	}
	if value > maximum {
		return maximum
	}
	return value
}

func cloneMap(source map[string]any) map[string]any {
	if source == nil {
		return nil
	}

	cloned := make(map[string]any, len(source))
	for key, value := range source {
		cloned[key] = cloneValue(value)
	}
	return cloned
}

func cloneValue(value any) any {
	switch typed := value.(type) {
	case map[string]any:
		return cloneMap(typed)
	case []any:
		cloned := make([]any, len(typed))
		for index, item := range typed {
			cloned[index] = cloneValue(item)
		}
		return cloned
	case []float64:
		cloned := make([]float64, len(typed))
		copy(cloned, typed)
		return cloned
	case []string:
		cloned := make([]string, len(typed))
		copy(cloned, typed)
		return cloned
	case []int:
		cloned := make([]int, len(typed))
		copy(cloned, typed)
		return cloned
	default:
		return value
	}
}

func thinStoppedSamples(samples []DatasetSample, initialBurst int, spacingSeconds float64) []DatasetSample {
	if len(samples) == 0 || initialBurst < 1 || spacingSeconds <= 0 {
		return samples
	}

	filtered := make([]DatasetSample, 0, len(samples))
	inStoppedRun := false
	stoppedKept := 0
	lastStoppedKeepTime := 0.0

	for _, sample := range samples {
		if isStoppedLabelValue(sampleCurrentTelemetryValue(sample, "isStopped")) {
			if !inStoppedRun {
				inStoppedRun = true
				stoppedKept = 1
				lastStoppedKeepTime = sample.AnchorGameTime
				filtered = append(filtered, sample)
				continue
			}

			if stoppedKept < initialBurst || sample.AnchorGameTime-lastStoppedKeepTime >= spacingSeconds {
				stoppedKept++
				lastStoppedKeepTime = sample.AnchorGameTime
				filtered = append(filtered, sample)
			}
			continue
		}

		inStoppedRun = false
		stoppedKept = 0
		lastStoppedKeepTime = 0.0
		filtered = append(filtered, sample)
	}

	return filtered
}

func sampleCurrentTelemetry(sample DatasetSample) *GroupedTelemetryItem {
	if len(sample.TelemetryHistory) > 0 {
		return &sample.TelemetryHistory[len(sample.TelemetryHistory)-1]
	}
	return nil
}

func sampleCurrentTelemetryValue(sample DatasetSample, key string) any {
	current := sampleCurrentTelemetry(sample)
	if current != nil {
		if value, ok := flattenGroupedTelemetry(*current)[key]; ok {
			return value
		}
	}
	if value, ok := flattenGroupedLabel(sample.Label)[key]; ok {
		return value
	}
	return nil
}

func isStoppedLabelValue(value any) bool {
	switch typed := value.(type) {
	case bool:
		return typed
	case float64:
		return typed != 0
	case float32:
		return typed != 0
	case int:
		return typed != 0
	case int8:
		return typed != 0
	case int16:
		return typed != 0
	case int32:
		return typed != 0
	case int64:
		return typed != 0
	case uint:
		return typed != 0
	case uint8:
		return typed != 0
	case uint16:
		return typed != 0
	case uint32:
		return typed != 0
	case uint64:
		return typed != 0
	default:
		return false
	}
}

func booleanField(value any) (bool, bool) {
	switch typed := value.(type) {
	case bool:
		return typed, true
	case float64:
		return typed != 0, true
	case float32:
		return typed != 0, true
	case int:
		return typed != 0, true
	case int64:
		return typed != 0, true
	case json.Number:
		parsed, err := typed.Float64()
		if err != nil {
			return false, false
		}
		return parsed != 0, true
	default:
		return false, false
	}
}

func nearestLabel(labels []timedLabel, target float64, toleranceSeconds float64) (timedLabel, bool) {
	label, _, ok := nearestLabelWithIndex(labels, target, toleranceSeconds)
	return label, ok
}

func nearestLabelWithIndex(labels []timedLabel, target float64, toleranceSeconds float64) (timedLabel, int, bool) {
	return nearestLabelWithIndexRange(labels, target, toleranceSeconds, 0, len(labels)-1)
}

func nearestLabelWithIndexRange(
	labels []timedLabel,
	target float64,
	toleranceSeconds float64,
	firstIndex int,
	lastIndex int,
) (timedLabel, int, bool) {
	if len(labels) == 0 ||
		firstIndex < 0 ||
		lastIndex >= len(labels) ||
		firstIndex > lastIndex ||
		!isFiniteFloat64(target) ||
		!isFiniteFloat64(toleranceSeconds) ||
		toleranceSeconds < 0 {
		return timedLabel{}, -1, false
	}
	rangeLength := lastIndex - firstIndex + 1
	idx := firstIndex + sort.Search(rangeLength, func(offset int) bool {
		return labels[firstIndex+offset].RelativeSeconds >= target
	})

	type candidateLabel struct {
		label timedLabel
		index int
	}
	candidates := make([]candidateLabel, 0, 2)
	if idx <= lastIndex && isFiniteFloat64(labels[idx].RelativeSeconds) {
		candidates = append(candidates, candidateLabel{label: labels[idx], index: idx})
	}
	if idx > firstIndex && isFiniteFloat64(labels[idx-1].RelativeSeconds) {
		candidates = append(candidates, candidateLabel{label: labels[idx-1], index: idx - 1})
	}
	if len(candidates) == 0 {
		return timedLabel{}, -1, false
	}

	best := candidates[0]
	bestDelta := math.Abs(best.label.RelativeSeconds - target)
	for _, candidate := range candidates[1:] {
		delta := math.Abs(candidate.label.RelativeSeconds - target)
		if delta < bestDelta {
			best = candidate
			bestDelta = delta
		}
	}

	if bestDelta > toleranceSeconds {
		return timedLabel{}, -1, false
	}
	return best.label, best.index, true
}

func pruneUnreferencedJPEGFrames(framesDir string, samples []DatasetSample) (int, error) {
	referenced := make(map[string]struct{})
	for _, sample := range samples {
		for _, framePath := range sample.FramePaths {
			name, ok := datasetFrameName(framePath)
			if !ok {
				return 0, fmt.Errorf("invalid dataset frame path %q", framePath)
			}
			referenced[name] = struct{}{}
		}
	}

	entries, err := os.ReadDir(framesDir)
	if err != nil {
		return 0, fmt.Errorf("read staged frames directory: %w", err)
	}
	found := make(map[string]struct{}, len(referenced))
	for _, entry := range entries {
		name := entry.Name()
		if filepath.Ext(name) != ".jpg" {
			continue
		}
		info, err := entry.Info()
		if err != nil {
			return 0, fmt.Errorf("inspect staged frame %s: %w", name, err)
		}
		if !info.Mode().IsRegular() {
			return 0, fmt.Errorf("staged frame is not a regular file: %s", name)
		}
		if _, keep := referenced[name]; !keep {
			if err := os.Remove(filepath.Join(framesDir, name)); err != nil {
				return 0, fmt.Errorf("remove unreferenced staged frame %s: %w", name, err)
			}
			continue
		}
		if info.Size() < 1 {
			return 0, fmt.Errorf("referenced staged frame is empty: %s", name)
		}
		found[name] = struct{}{}
	}
	for name := range referenced {
		if _, ok := found[name]; !ok {
			return 0, fmt.Errorf("referenced staged frame is missing: %s", name)
		}
	}
	return len(referenced), nil
}

func datasetFrameName(framePath string) (string, bool) {
	if strings.Contains(framePath, "\\") || strings.HasPrefix(framePath, "/") {
		return "", false
	}
	parts := strings.Split(framePath, "/")
	if len(parts) != 2 || parts[0] != "frames" || !validNumberedJPEGName(parts[1]) {
		return "", false
	}
	return parts[1], true
}

func validNumberedJPEGName(name string) bool {
	if len(name) != len("000001.jpg") || !strings.HasSuffix(name, ".jpg") {
		return false
	}
	for _, character := range name[:6] {
		if character < '0' || character > '9' {
			return false
		}
	}
	return name[:6] != "000000"
}

func writeDatasetFile(path string, samples []DatasetSample) error {
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create dataset.jsonl: %w", err)
	}

	encoder := json.NewEncoder(file)
	for _, sample := range samples {
		if err := encoder.Encode(sample); err != nil {
			return errors.Join(fmt.Errorf("failed to write dataset.jsonl: %w", err), file.Close())
		}
	}
	if err := file.Sync(); err != nil {
		return errors.Join(fmt.Errorf("sync dataset.jsonl: %w", err), file.Close())
	}
	if err := file.Close(); err != nil {
		return fmt.Errorf("close dataset.jsonl: %w", err)
	}
	return nil
}

func writeStatusFile(path string, status ProcessingStatus) error {
	return writeJSONAtomically(path, status)
}

func writeJSONAtomically(path string, value any) error {
	body, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	body = append(body, '\n')

	dir := filepath.Dir(path)
	temp, err := os.CreateTemp(dir, "."+filepath.Base(path)+".tmp-")
	if err != nil {
		return fmt.Errorf("create atomic JSON temp file: %w", err)
	}
	tempPath := temp.Name()
	closed := false
	defer func() {
		if !closed {
			_ = temp.Close()
		}
		_ = os.Remove(tempPath)
	}()
	if err := temp.Chmod(0o644); err != nil {
		return fmt.Errorf("set atomic JSON temp permissions: %w", err)
	}
	if _, err := temp.Write(body); err != nil {
		return fmt.Errorf("write atomic JSON temp file: %w", err)
	}
	if err := temp.Sync(); err != nil {
		return fmt.Errorf("sync atomic JSON temp file: %w", err)
	}
	if err := temp.Close(); err != nil {
		closed = true
		return fmt.Errorf("close atomic JSON temp file: %w", err)
	}
	closed = true
	if err := replaceFileAtomically(tempPath, path); err != nil {
		return fmt.Errorf("replace JSON file atomically: %w", err)
	}
	return nil
}

func ReadStatusFile(path string) (ProcessingStatus, error) {
	var status ProcessingStatus
	body, err := readFileConsistently(path)
	if err != nil {
		return status, err
	}
	if err := json.Unmarshal(body, &status); err != nil {
		return status, err
	}
	return status, nil
}

func resolveStatusPath(tripDir string) (string, error) {
	tripPath, err := resolveTripDir(tripDir)
	if err != nil {
		return "", err
	}
	return filepath.Join(tripPath, "processing.json"), nil
}

func resolveTripDir(tripDir string) (string, error) {
	cleaned := filepath.Clean(strings.TrimSpace(tripDir))
	if cleaned == "" || cleaned == "." {
		return "", ErrInvalidTripDir
	}
	info, err := os.Stat(cleaned)
	if err != nil {
		return "", fmt.Errorf("%w: %v", ErrInvalidTripDir, err)
	}
	if !info.IsDir() {
		return "", ErrInvalidTripDir
	}
	return cleaned, nil
}

func validateFrameSet(root string, frames []VideoFrame, expectedWidth int, expectedHeight int) error {
	if len(frames) == 0 {
		return errors.New("no probed video frames")
	}
	if expectedWidth < 1 || expectedHeight < 1 {
		return errors.New("expected frame dimensions must be positive")
	}
	if !numberedFramesComplete(filepath.Join(root, "frames"), len(frames)) {
		return fmt.Errorf("frame set is incomplete: expected %d contiguous images", len(frames))
	}
	for position, frame := range frames {
		if frame.Index != position {
			return fmt.Errorf("frame timeline is non-contiguous at position %d: source index %d", position, frame.Index)
		}
		path := filepath.Join(root, filepath.FromSlash(frame.ImagePath))
		file, err := os.Open(path)
		if err != nil {
			return fmt.Errorf("open frame image %s: %w", path, err)
		}
		config, _, decodeErr := image.DecodeConfig(file)
		closeErr := file.Close()
		if decodeErr != nil {
			return errors.Join(fmt.Errorf("decode frame image %s: %w", path, decodeErr), closeErr)
		}
		if closeErr != nil {
			return fmt.Errorf("close frame image %s: %w", path, closeErr)
		}
		if config.Width != expectedWidth || config.Height != expectedHeight {
			return fmt.Errorf(
				"frame dimensions do not match processing config at %s: got %dx%d want %dx%d",
				path,
				config.Width,
				config.Height,
				expectedWidth,
				expectedHeight,
			)
		}
	}
	return nil
}

func (p *Processor) shouldSkipTrip(tripDir string) bool {
	return p.TripOutputsCurrent(tripDir)
}

// TripOutputsCurrent reports whether a trip has complete published outputs for
// this processor's exact configuration. It is intentionally read-only so CLI,
// HTTP, and UI readiness checks can share the same rule as processing skips.
func (p *Processor) TripOutputsCurrent(tripDir string) bool {
	status, err := ReadStatusFile(filepath.Join(tripDir, "processing.json"))
	if err != nil {
		return false
	}
	if status.State != "completed" && status.State != "skipped" {
		return false
	}
	if status.ConfigFingerprint != p.ConfigFingerprint() {
		return false
	}
	if !p.statusContractMatches(status) {
		return false
	}
	return processingOutputsComplete(tripDir, status)
}

func (p *Processor) statusContractMatches(status ProcessingStatus) bool {
	return status.ImageWidth == p.imageWidth &&
		status.ImageHeight == p.imageHeight &&
		p.statusTimelineMatches(status)
}

func (p *Processor) statusTimelineMatches(status ProcessingStatus) bool {
	return equalIntSlices(status.ImageOffsets, p.imageOffsets) &&
		equalIntSlices(status.TelemetryOffsets, p.telemetryOffsets) &&
		equalIntSlices(status.FutureOffsets, p.futureOffsets) &&
		status.TelemetrySampleIntervalMs == float64(p.telemetrySampleInterval)/float64(time.Millisecond)
}

func equalIntSlices(left []int, right []int) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}

func processingOutputsComplete(tripDir string, status ProcessingStatus) bool {
	if status.FrameCount < 0 || status.SampleCount < 0 {
		return false
	}
	if !processingCountsExplicit(filepath.Join(tripDir, "processing.json")) {
		return false
	}
	datasetPath := filepath.Join(tripDir, "dataset.jsonl")
	lineCount, referencedFrames, framePathsExplicit, err := readDatasetFrameReferences(datasetPath)
	if err != nil || lineCount != status.SampleCount {
		return false
	}
	if lineCount == 0 || !framePathsExplicit {
		return numberedFramesComplete(filepath.Join(tripDir, "frames"), status.FrameCount)
	}
	return referencedFramesComplete(filepath.Join(tripDir, "frames"), referencedFrames, status.FrameCount)
}

func processingCountsExplicit(statusPath string) bool {
	body, err := readFileConsistently(statusPath)
	if err != nil {
		return false
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		return false
	}
	for _, key := range []string{"frameCount", "sampleCount"} {
		var value int
		raw, exists := fields[key]
		if !exists || json.Unmarshal(raw, &value) != nil {
			return false
		}
	}
	return true
}

func numberedFramesComplete(framesDir string, expectedCount int) bool {
	if expectedCount < 0 {
		return false
	}
	entries, err := os.ReadDir(framesDir)
	if err != nil {
		return false
	}
	regularFileCount := 0
	for _, entry := range entries {
		if entry.Type().IsRegular() {
			regularFileCount++
		}
	}
	if regularFileCount != expectedCount {
		return false
	}
	if expectedCount == 0 {
		return true
	}
	for index := 1; index <= expectedCount; index++ {
		info, err := os.Stat(filepath.Join(framesDir, fmt.Sprintf("%06d.jpg", index)))
		if err != nil || !info.Mode().IsRegular() || info.Size() < 1 {
			return false
		}
	}
	return true
}

func referencedFramesComplete(framesDir string, referenced map[string]struct{}, expectedCount int) bool {
	if expectedCount < 0 || len(referenced) > expectedCount || !numberedFramesComplete(framesDir, expectedCount) {
		return false
	}
	for name := range referenced {
		info, err := os.Stat(filepath.Join(framesDir, name))
		if err != nil || !info.Mode().IsRegular() || info.Size() < 1 {
			return false
		}
	}
	return true
}

func readDatasetFrameReferences(path string) (int, map[string]struct{}, bool, error) {
	file, err := os.Open(path)
	if err != nil {
		return 0, nil, false, err
	}
	defer file.Close()

	lineCount := 0
	rowsWithFramePaths := 0
	referenced := make(map[string]struct{})
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 64*1024), 128*1024*1024)
	for scanner.Scan() {
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		var row struct {
			FramePaths []string `json:"frame_paths"`
		}
		if err := json.Unmarshal(line, &row); err != nil {
			return 0, nil, false, err
		}
		if len(row.FramePaths) > 0 {
			rowsWithFramePaths++
			for _, framePath := range row.FramePaths {
				name, ok := datasetFrameName(framePath)
				if !ok {
					return 0, nil, false, fmt.Errorf("invalid dataset frame path %q", framePath)
				}
				referenced[name] = struct{}{}
			}
		}
		lineCount++
	}
	if err := scanner.Err(); err != nil {
		return 0, nil, false, err
	}
	if rowsWithFramePaths != 0 && rowsWithFramePaths != lineCount {
		return 0, nil, false, errors.New("dataset mixes rows with and without frame paths")
	}
	return lineCount, referenced, lineCount == 0 || rowsWithFramePaths == lineCount, nil
}

func countValidJSONLines(path string) (int, error) {
	file, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer file.Close()

	count := 0
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 64*1024), 128*1024*1024)
	for scanner.Scan() {
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		if !json.Valid(line) {
			return 0, fmt.Errorf("invalid JSON at dataset line %d", count+1)
		}
		count++
	}
	if err := scanner.Err(); err != nil {
		return 0, err
	}
	return count, nil
}

func promoteProcessingWorkspace(tripPath string, workspace processingWorkspace, includeFrames bool) error {
	return promoteProcessingWorkspaceWithRename(tripPath, workspace, includeFrames, os.Rename)
}

func promoteProcessingWorkspaceWithRename(
	tripPath string,
	workspace processingWorkspace,
	includeFrames bool,
	rename func(string, string) error,
) error {
	journal := newPromotionJournal(includeFrames)
	for _, operation := range journal.Operations {
		paths := resolvePromotionPaths(tripPath, workspace, operation.Name)
		if !pathExists(paths.source) {
			return fmt.Errorf("staged processing output is missing: %s", paths.source)
		}
	}
	journal.Phase = "promoting"
	if err := writePromotionJournal(workspace, journal); err != nil {
		return err
	}

	for index := range journal.Operations {
		operation := &journal.Operations[index]
		paths := resolvePromotionPaths(tripPath, workspace, operation.Name)
		if !pathExists(paths.target) {
			continue
		}
		operation.HadPrior = true
		operation.BackupStarted = true
		if err := writePromotionJournal(workspace, journal); err != nil {
			return err
		}
		if err := rename(paths.target, paths.backup); err != nil {
			return fmt.Errorf("backup processing output %s: %w", paths.target, err)
		}
		operation.BackupDone = true
		if err := writePromotionJournal(workspace, journal); err != nil {
			return err
		}
	}

	for index := range journal.Operations {
		operation := &journal.Operations[index]
		paths := resolvePromotionPaths(tripPath, workspace, operation.Name)
		operation.PublishStarted = true
		if err := writePromotionJournal(workspace, journal); err != nil {
			return err
		}
		if err := rename(paths.source, paths.target); err != nil {
			return fmt.Errorf("publish processing output %s: %w", paths.target, err)
		}
		operation.PublishDone = true
		if err := writePromotionJournal(workspace, journal); err != nil {
			return err
		}
	}
	journal.Phase = "published"
	return writePromotionJournal(workspace, journal)
}

func rollbackProcessingWorkspace(tripPath string, workspace processingWorkspace) error {
	return rollbackProcessingWorkspaceWithFS(tripPath, workspace, os.Rename, os.RemoveAll)
}

func rollbackProcessingWorkspaceWithFS(
	tripPath string,
	workspace processingWorkspace,
	rename func(string, string) error,
	removeAll func(string) error,
) error {
	journal, err := readPromotionJournal(workspace)
	if err != nil {
		return err
	}
	journal.Phase = "rolling_back"
	if err := writePromotionJournal(workspace, journal); err != nil {
		return err
	}

	for index := len(journal.Operations) - 1; index >= 0; index-- {
		operation := &journal.Operations[index]
		paths := resolvePromotionPaths(tripPath, workspace, operation.Name)
		if operation.RestoreDone || operation.RemoveDone {
			continue
		}
		if operation.RestoreStarted && !pathExists(paths.backup) && pathExists(paths.target) {
			operation.RestoreDone = true
			if err := writePromotionJournal(workspace, journal); err != nil {
				return err
			}
			continue
		}
		if pathExists(paths.backup) {
			operation.HadPrior = true
			operation.RestoreStarted = true
			if err := writePromotionJournal(workspace, journal); err != nil {
				return err
			}
			if pathExists(paths.target) {
				if err := removeAll(paths.target); err != nil {
					return fmt.Errorf("remove partially published output %s: %w", paths.target, err)
				}
			}
			if err := rename(paths.backup, paths.target); err != nil {
				return fmt.Errorf("restore previous processing output %s: %w", paths.target, err)
			}
			operation.RestoreDone = true
			if err := writePromotionJournal(workspace, journal); err != nil {
				return err
			}
			continue
		}
		if operation.HadPrior && operation.BackupDone && !operation.RestoreStarted {
			return fmt.Errorf("processing backup disappeared before restore: %s", paths.backup)
		}
		if !operation.HadPrior && operation.PublishStarted && !pathExists(paths.source) && pathExists(paths.target) {
			operation.RemoveStarted = true
			if err := writePromotionJournal(workspace, journal); err != nil {
				return err
			}
			if err := removeAll(paths.target); err != nil {
				return fmt.Errorf("remove newly published output %s: %w", paths.target, err)
			}
			operation.RemoveDone = true
			if err := writePromotionJournal(workspace, journal); err != nil {
				return err
			}
		}
	}
	journal.Phase = "rolled_back"
	return writePromotionJournal(workspace, journal)
}

type promotionPaths struct {
	source string
	target string
	backup string
}

func newPromotionJournal(includeFrames bool) promotionJournal {
	operations := make([]promotionJournalOperation, 0, 2)
	if includeFrames {
		operations = append(operations, promotionJournalOperation{Name: "frames"})
	}
	operations = append(operations, promotionJournalOperation{Name: "dataset"})
	return promotionJournal{
		Version:       processingJournalVersion,
		Phase:         "staged",
		IncludeFrames: includeFrames,
		Operations:    operations,
	}
}

func resolvePromotionPaths(tripPath string, workspace processingWorkspace, name string) promotionPaths {
	switch name {
	case "frames":
		return promotionPaths{
			source: workspace.framesDir,
			target: filepath.Join(tripPath, "frames"),
			backup: filepath.Join(workspace.root, "previous-frames"),
		}
	case "dataset":
		return promotionPaths{
			source: workspace.datasetPath,
			target: filepath.Join(tripPath, "dataset.jsonl"),
			backup: filepath.Join(workspace.root, "previous-dataset.jsonl"),
		}
	default:
		panic(fmt.Sprintf("unknown processing promotion operation %q", name))
	}
}

func writePromotionJournal(workspace processingWorkspace, journal promotionJournal) error {
	if journal.Version != processingJournalVersion {
		return fmt.Errorf("unsupported processing journal version %d", journal.Version)
	}
	if err := writeJSONAtomically(workspace.journalPath, journal); err != nil {
		return fmt.Errorf("write processing promotion journal: %w", err)
	}
	return nil
}

func readPromotionJournal(workspace processingWorkspace) (promotionJournal, error) {
	var journal promotionJournal
	body, err := os.ReadFile(workspace.journalPath)
	if err != nil {
		return journal, fmt.Errorf("read processing promotion journal: %w", err)
	}
	if err := json.Unmarshal(body, &journal); err != nil {
		return journal, fmt.Errorf("parse processing promotion journal: %w", err)
	}
	if journal.Version != processingJournalVersion || len(journal.Operations) == 0 {
		return journal, fmt.Errorf("invalid processing promotion journal version=%d operations=%d", journal.Version, len(journal.Operations))
	}
	for _, operation := range journal.Operations {
		if operation.Name != "frames" && operation.Name != "dataset" {
			return journal, fmt.Errorf("invalid processing promotion operation %q", operation.Name)
		}
	}
	return journal, nil
}

func pathExists(path string) bool {
	_, err := os.Lstat(path)
	return err == nil
}

func fileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

func numberField(value any) (float64, bool) {
	switch typed := value.(type) {
	case float64:
		return typed, true
	case float32:
		return float64(typed), true
	case int:
		return float64(typed), true
	case int64:
		return float64(typed), true
	case json.Number:
		parsed, err := typed.Float64()
		return parsed, err == nil
	default:
		return 0, false
	}
}

func vector3Field(value any) ([3]float64, bool) {
	switch typed := value.(type) {
	case []any:
		if len(typed) < 3 {
			return [3]float64{}, false
		}
		x, ok := numberField(typed[0])
		if !ok || !isFiniteFloat64(x) {
			return [3]float64{}, false
		}
		y, ok := numberField(typed[1])
		if !ok || !isFiniteFloat64(y) {
			return [3]float64{}, false
		}
		z, ok := numberField(typed[2])
		if !ok || !isFiniteFloat64(z) {
			return [3]float64{}, false
		}
		return [3]float64{x, y, z}, true
	case []float64:
		if len(typed) < 3 {
			return [3]float64{}, false
		}
		return [3]float64{typed[0], typed[1], typed[2]}, true
	default:
		return [3]float64{}, false
	}
}

func parseFloat(value string) (float64, error) {
	var parsed float64
	if err := json.Unmarshal([]byte(value), &parsed); err == nil {
		return parsed, nil
	}
	return 0, fmt.Errorf("invalid float %q", value)
}

func envOrDefault(key string, fallback string) string {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback
	}
	return value
}

func min(a int, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a int, b int) int {
	if a > b {
		return a
	}
	return b
}
