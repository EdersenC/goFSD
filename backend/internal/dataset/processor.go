package dataset

import (
	"bufio"
	"context"
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
	defaultLabelTolerance           = 100 * time.Millisecond
	defaultDeltaSpeedClip           = 2.0
	defaultDeltaSpeedNormalize      = true
	defaultFlashBrightnessThreshold = 245.0
	defaultFlashFrameLimit          = 90
	defaultStoppedSampleBurst       = 3
	defaultStoppedSampleSpacing     = 2.0
	futureTargetSmoothingRadius     = 2
	moveIntentFutureWindowAhead     = 2
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
	State             string         `json:"state"`
	StartedAt         string         `json:"startedAt,omitempty"`
	CompletedAt       string         `json:"completedAt,omitempty"`
	Error             string         `json:"error,omitempty"`
	Warning           string         `json:"warning,omitempty"`
	FramesDir         string         `json:"framesDir,omitempty"`
	DatasetFile       string         `json:"datasetFile,omitempty"`
	ImageWidth        int            `json:"imageWidth,omitempty"`
	ImageHeight       int            `json:"imageHeight,omitempty"`
	FrameCount        int            `json:"frameCount,omitempty"`
	SampleCount       int            `json:"sampleCount"`
	ZeroSampleReasons map[string]int `json:"zeroSampleReasons,omitempty"`
}

type DatasetSample struct {
	AnchorVideoPTS   float64                `json:"anchor_video_pts"`
	AnchorGameTime   float64                `json:"anchor_game_time"`
	FramePaths       []string               `json:"frame_paths"`
	TelemetryHistory []GroupedTelemetryItem `json:"telemetry_history,omitempty"`
	TelemetryFuture  []GroupedTelemetryItem `json:"telemetry_future,omitempty"`
	Label            GroupedLabel           `json:"label"`
}

type GroupedLabel struct {
	Control GroupedLabelControl `json:"control"`
	Aux     GroupedLabelAux     `json:"aux"`
}

type GroupedLabelControl struct {
	Steering any `json:"Steering,omitempty"`
}

type GroupedLabelAux struct {
	DeltaSpeed           any `json:"delta_speed,omitempty"`
	DeltaSpeedTarget     any `json:"delta_speed_target,omitempty"`
	FutureSpeed          any `json:"future_speed,omitempty"`
	FutureSpeedTarget    any `json:"future_speed_target,omitempty"`
	FutureYawDelta       any `json:"future_yaw_delta,omitempty"`
	FutureHorizonSeconds any `json:"future_horizon_seconds,omitempty"`
	YawRate              any `json:"yaw_rate,omitempty"`
	RouteForwardDelta    any `json:"routeForwardDelta,omitempty"`
	MoveIntent           any `json:"move_intent,omitempty"`
}

type GroupedTelemetryItem struct {
	Control GroupedTelemetryControl `json:"control"`
	Aux     GroupedTelemetryAux     `json:"aux"`
	Raw     map[string]any          `json:"raw,omitempty"`
}

type GroupedTelemetryControl struct {
	Steering         any `json:"Steering,omitempty"`
	Acceleration     any `json:"acceleration,omitempty"`
	BrakePressureAvg any `json:"brakePressureAvg,omitempty"`
}

type GroupedTelemetryAux struct {
	CurrentSpeed           any `json:"currentSpeed,omitempty"`
	Yaw                    any `json:"yaw,omitempty"`
	YawRate                any `json:"yawRate,omitempty"`
	RouteForwardDelta      any `json:"routeForwardDelta,omitempty"`
	RouteHeadingError      any `json:"routeHeadingError,omitempty"`
	RouteDistance          any `json:"routeDistance,omitempty"`
	LeadVehicleDistance    any `json:"leadVehicleDistance,omitempty"`
	HasLeadVehicle         any `json:"hasLeadVehicle,omitempty"`
	GPS                    any `json:"gps,omitempty"`
	IsStopped              any `json:"isStopped,omitempty"`
	RouteGPSValid          any `json:"routeGpsValid,omitempty"`
	IsStoppedAtTraffic     any `json:"isStoppedAtTrafficLights,omitempty"`
	LeadVehicleRelSpeed    any `json:"leadVehicleRelativeSpeed,omitempty"`
	LeadVehicleHeadingDiff any `json:"leadVehicleHeadingDelta,omitempty"`
	LeadVehicleTTC         any `json:"leadVehicleTTC,omitempty"`
}

type sampleBuildStats struct {
	CandidateWindowCount          int
	GeneratedSampleCount          int
	MissingCurrentLabelCount      int
	MissingFutureLabelCount       int
	MissingRouteForwardDeltaCount int
	MissingFutureSpeedTargetCount int
	MissingFutureYawTargetCount   int
	MissingYawRateTargetCount     int
	InvalidFutureHorizonCount     int
	InvalidDerivedLabelCount      int
}

func (s sampleBuildStats) zeroSampleReasons() map[string]int {
	reasons := map[string]int{
		"candidate_windows": s.CandidateWindowCount,
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
	if s.MissingFutureYawTargetCount > 0 {
		reasons["missing_future_yaw_target"] = s.MissingFutureYawTargetCount
	}
	if s.MissingYawRateTargetCount > 0 {
		reasons["missing_yaw_rate_target"] = s.MissingYawRateTargetCount
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
	sampleStride             int
	labelTolerance           time.Duration
	deltaSpeedClip           float64
	deltaSpeedNormalize      bool
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
	VehicleModel string `json:"vehicleModel"`
	VehicleColor string `json:"vehicleColor"`
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
		deltaSpeedClip:           defaultDeltaSpeedClip,
		deltaSpeedNormalize:      defaultDeltaSpeedNormalize,
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
	if p.deltaSpeedClip <= 0 {
		p.deltaSpeedClip = defaultDeltaSpeedClip
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

func WithLabelTolerance(tolerance time.Duration) Option {
	return func(p *Processor) {
		p.labelTolerance = tolerance
	}
}

func WithDeltaSpeedTargetConfig(clip float64, normalize bool) Option {
	return func(p *Processor) {
		p.deltaSpeedClip = clip
		p.deltaSpeedNormalize = normalize
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

func (p *Processor) Queue(tripDir string) (string, error) {
	statusPath, err := resolveStatusPath(tripDir)
	if err != nil {
		return "", err
	}
	tripPath, err := resolveTripDir(tripDir)
	if err != nil {
		return "", err
	}
	if !p.force && p.shouldSkipTrip(tripPath) {
		status := ProcessingStatus{
			State:       "skipped",
			CompletedAt: time.Now().Format(time.RFC3339),
			FramesDir:   "frames",
			DatasetFile: "dataset.jsonl",
			ImageWidth:  p.imageWidth,
			ImageHeight: p.imageHeight,
		}
		return statusPath, writeStatusFile(statusPath, status)
	}
	status := ProcessingStatus{
		State:       "queued",
		FramesDir:   "frames",
		DatasetFile: "dataset.jsonl",
		ImageWidth:  p.imageWidth,
		ImageHeight: p.imageHeight,
	}
	return statusPath, writeStatusFile(statusPath, status)
}

func (p *Processor) ProcessTrip(ctx context.Context, tripDir string) error {
	tripPath, err := resolveTripDir(tripDir)
	if err != nil {
		return err
	}
	if !p.force && p.shouldSkipTrip(tripPath) {
		return writeStatusFile(filepath.Join(tripPath, "processing.json"), ProcessingStatus{
			State:       "skipped",
			CompletedAt: time.Now().Format(time.RFC3339),
			FramesDir:   "frames",
			DatasetFile: "dataset.jsonl",
			ImageWidth:  p.imageWidth,
			ImageHeight: p.imageHeight,
		})
	}

	statusPath := filepath.Join(tripPath, "processing.json")
	if err := writeStatusFile(statusPath, ProcessingStatus{
		State:       "running",
		StartedAt:   time.Now().Format(time.RFC3339),
		FramesDir:   "frames",
		DatasetFile: "dataset.jsonl",
		ImageWidth:  p.imageWidth,
		ImageHeight: p.imageHeight,
	}); err != nil {
		return err
	}

	status := ProcessingStatus{
		State:       "completed",
		StartedAt:   time.Now().Format(time.RFC3339),
		FramesDir:   "frames",
		DatasetFile: "dataset.jsonl",
		ImageWidth:  p.imageWidth,
		ImageHeight: p.imageHeight,
	}

	defer func() {
		if status.State == "completed" {
			status.CompletedAt = time.Now().Format(time.RFC3339)
		}
		_ = writeStatusFile(statusPath, status)
	}()

	videoPath := filepath.Join(tripPath, "video.mkv")
	metadataPath := filepath.Join(tripPath, "metadata.json")
	runFilePath := filepath.Join(filepath.Dir(tripPath), "run.jsonl")
	framesDir := filepath.Join(tripPath, "frames")
	datasetPath := filepath.Join(tripPath, "dataset.jsonl")

	if !fileExists(videoPath) || !fileExists(metadataPath) || !fileExists(runFilePath) {
		status.State = "failed"
		status.Error = fmt.Sprintf("%v: expected video.mkv, metadata.json, and run.jsonl at %s", ErrMissingTripFiles, runFilePath)
		return fmt.Errorf("%w: expected video.mkv, metadata.json, and run.jsonl at %s", ErrMissingTripFiles, runFilePath)
	}

	metadata, err := loadTripMetadata(metadataPath)
	if err != nil {
		status.State = "failed"
		status.Error = err.Error()
		return err
	}

	frames, err := p.ProbeVideoFrames(ctx, videoPath)
	if err != nil {
		status.State = "failed"
		status.Error = err.Error()
		return err
	}
	if len(frames) == 0 {
		status.State = "failed"
		status.Error = "ffprobe returned no video frames"
		return errors.New("ffprobe returned no video frames")
	}

	if p.datasetOnly {
		frames = AttachImagePaths(frames, "frames")
		frames = filterFramesWithExistingImages(tripPath, frames)
	} else {
		if err := extractFrames(ctx, p.newCommand, p.ffmpegBin, videoPath, framesDir, p.imageWidth, p.imageHeight); err != nil {
			status.State = "failed"
			status.Error = err.Error()
			return err
		}

		frames = AttachImagePaths(frames, "frames")
		frames = filterFramesWithExistingImages(tripPath, frames)
	}
	if len(frames) == 0 {
		status.State = "failed"
		status.Error = "no extracted frames found on disk"
		return errors.New("no extracted frames found on disk")
	}

	anchorPTS, err := detectSyncFlashPTS(tripPath, frames, p.flashFrameLimit, p.flashBrightnessThreshold)
	if err != nil {
		status.State = "failed"
		status.Error = err.Error()
		return err
	}

	record, err := loadRunTripRecord(runFilePath, metadata.RunID, metadata.TripIndex)
	if err != nil {
		status.State = "failed"
		status.Error = err.Error()
		return err
	}

	labels := buildTimedLabels(record.VehicleData, metadata.SyncTime)
	samples, sampleStats := buildDatasetSamplesWithStats(
		frames,
		labels,
		anchorPTS,
		p.windowSize,
		p.frameStride,
		p.sampleStride,
		p.labelTolerance,
		p.deltaSpeedClip,
		p.deltaSpeedNormalize,
	)
	samples = thinStoppedSamples(samples, defaultStoppedSampleBurst, defaultStoppedSampleSpacing)
	if err := writeDatasetFile(datasetPath, samples); err != nil {
		status.State = "failed"
		status.Error = err.Error()
		return err
	}

	status.FrameCount = len(frames)
	status.SampleCount = len(samples)
	if len(samples) == 0 {
		status.Warning = "no dataset samples generated"
		status.ZeroSampleReasons = sampleStats.zeroSampleReasons()
	}
	return nil
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
			continue
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
		out[i].ImagePath = fmt.Sprintf("%s/%06d.jpg", cleanDir, i+1)
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
		if !ok {
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
	deltaSpeedClip float64,
	deltaSpeedNormalize bool,
) []DatasetSample {
	samples, _ := buildDatasetSamplesWithStats(
		frames,
		labels,
		anchorPTS,
		windowSize,
		frameStride,
		sampleStride,
		tolerance,
		deltaSpeedClip,
		deltaSpeedNormalize,
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
	deltaSpeedClip float64,
	deltaSpeedNormalize bool,
) ([]DatasetSample, sampleBuildStats) {
	var stats sampleBuildStats
	if len(frames) == 0 || len(labels) == 0 || windowSize < 1 || windowSize%2 == 0 || frameStride < 1 || sampleStride < 1 {
		return nil, stats
	}

	historyTelemetryCount := ((windowSize - 1) * frameStride) + 1
	toleranceSeconds := tolerance.Seconds()
	samples := make([]DatasetSample, 0)
	for anchorIndex := 0; anchorIndex < len(frames); anchorIndex += sampleStride {
		stats.CandidateWindowCount++

		window := buildPastOnlyFrameWindow(frames, anchorIndex, windowSize, frameStride)
		anchorFrame := frames[anchorIndex]
		relativeSeconds := anchorFrame.PTS - anchorPTS
		label, labelIndex, ok := nearestLabelWithIndex(labels, relativeSeconds, toleranceSeconds)
		if !ok {
			stats.MissingCurrentLabelCount++
			continue
		}

		futureCenter := min(anchorIndex+sampleStride, len(frames)-1)
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
		moveIntentSpeedTarget, ok := forwardFutureSpeed(labels, futureLabelIndex, moveIntentFutureWindowAhead)
		if !ok {
			stats.MissingFutureSpeedTargetCount++
			continue
		}
		futureYawTarget, ok := smoothedFutureYaw(labels, futureLabelIndex, futureTargetSmoothingRadius)
		if !ok {
			stats.MissingFutureYawTargetCount++
			continue
		}
		yawRateTarget, ok := smoothedYawRate(labels, futureLabelIndex, futureTargetSmoothingRadius)
		if !ok {
			stats.MissingYawRateTargetCount++
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
			moveIntentSpeedTarget,
			futureYawTarget,
			yawRateTarget,
			routeForwardDeltaTarget,
			futureHorizonSeconds,
			deltaSpeedClip,
			deltaSpeedNormalize,
		)
		if !ok {
			stats.InvalidDerivedLabelCount++
			continue
		}
		samples = append(samples, DatasetSample{
			AnchorVideoPTS:   anchorFrame.PTS,
			AnchorGameTime:   label.RelativeSeconds,
			FramePaths:       window,
			TelemetryHistory: buildTelemetryHistory(labels, labelIndex, historyTelemetryCount),
			TelemetryFuture:  buildTelemetryFuture(labels, labelIndex, defaultFutureTelemetryCount),
			Label:            derivedLabel,
		})
		stats.GeneratedSampleCount++
	}

	return samples, stats
}

func buildPastOnlyFrameWindow(frames []VideoFrame, anchorIndex int, windowSize int, frameStride int) []string {
	window := make([]string, 0, windowSize)
	for slot := 0; slot < windowSize; slot++ {
		idx := anchorIndex - ((windowSize - 1 - slot) * frameStride)
		window = append(window, frames[clampInt(idx, 0, len(frames)-1)].ImagePath)
	}
	return window
}

func buildTelemetryHistory(labels []timedLabel, anchorIndex int, count int) []GroupedTelemetryItem {
	if count <= 0 {
		return nil
	}
	startIndex := anchorIndex - (count - 1)
	return buildPaddedTelemetryWindow(labels, startIndex, count)
}

func buildTelemetryFuture(labels []timedLabel, anchorIndex int, count int) []GroupedTelemetryItem {
	if count <= 0 {
		return nil
	}
	startIndex := anchorIndex + 1
	return buildPaddedTelemetryWindow(labels, startIndex, count)
}

func buildPaddedTelemetryWindow(labels []timedLabel, startIndex int, count int) []GroupedTelemetryItem {
	if len(labels) == 0 || count <= 0 {
		return nil
	}

	window := make([]GroupedTelemetryItem, 0, count)
	for offset := 0; offset < count; offset++ {
		index := clampInt(startIndex+offset, 0, len(labels)-1)
		window = append(window, groupTelemetryItem(labels[index].Label))
	}
	return window
}

func buildTrainingLabel(
	current map[string]any,
	future map[string]any,
	futureSpeedTarget float64,
	moveIntentSpeedTarget float64,
	futureYawTarget float64,
	yawRateTarget float64,
	routeForwardDeltaTarget float64,
	futureHorizonSeconds float64,
	deltaSpeedClip float64,
	deltaSpeedNormalize bool,
) (GroupedLabel, bool) {
	currentSpeed, ok := numberField(current["currentSpeed"])
	if !ok {
		return GroupedLabel{}, false
	}
	futureSpeed, ok := numberField(future["currentSpeed"])
	if !ok {
		return GroupedLabel{}, false
	}
	currentYaw, ok := numberField(current["yaw"])
	if !ok || !isFiniteFloat64(currentYaw) || !isFiniteFloat64(futureYawTarget) {
		return GroupedLabel{}, false
	}
	if !isFiniteFloat64(yawRateTarget) || !isFiniteFloat64(routeForwardDeltaTarget) ||
		!isFiniteFloat64(futureHorizonSeconds) || futureHorizonSeconds <= 0 {
		return GroupedLabel{}, false
	}

	derived := GroupedLabel{}
	if steering, ok := current["Steering"]; ok {
		derived.Control.Steering = cloneValue(steering)
	}
	deltaSpeed := futureSpeedTarget - currentSpeed
	clippedDeltaSpeed := clampFloat64(deltaSpeed, -deltaSpeedClip, deltaSpeedClip)
	deltaSpeedTarget := clippedDeltaSpeed
	if deltaSpeedNormalize {
		deltaSpeedTarget = clippedDeltaSpeed / deltaSpeedClip
	}
	derived.Aux.DeltaSpeed = clippedDeltaSpeed
	derived.Aux.DeltaSpeedTarget = deltaSpeedTarget
	derived.Aux.FutureSpeed = futureSpeed
	derived.Aux.FutureSpeedTarget = futureSpeedTarget
	derived.Aux.FutureYawDelta = wrapHeadingDeltaDegrees(futureYawTarget - currentYaw)
	derived.Aux.FutureHorizonSeconds = futureHorizonSeconds
	derived.Aux.YawRate = yawRateTarget
	derived.Aux.RouteForwardDelta = routeForwardDeltaTarget
	derived.Aux.MoveIntent = buildMoveIntentLabel(current, moveIntentSpeedTarget, currentSpeed)
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
		case "currentSpeed":
			item.Aux.CurrentSpeed = cloned
		case "yaw":
			item.Aux.Yaw = cloned
		case "yawRate":
			item.Aux.YawRate = cloned
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
	appendIfPresent(flat, "delta_speed", label.Aux.DeltaSpeed)
	appendIfPresent(flat, "delta_speed_target", label.Aux.DeltaSpeedTarget)
	appendIfPresent(flat, "future_speed", label.Aux.FutureSpeed)
	appendIfPresent(flat, "future_speed_target", label.Aux.FutureSpeedTarget)
	appendIfPresent(flat, "future_yaw_delta", label.Aux.FutureYawDelta)
	appendIfPresent(flat, "future_horizon_seconds", label.Aux.FutureHorizonSeconds)
	appendIfPresent(flat, "yaw_rate", label.Aux.YawRate)
	appendIfPresent(flat, "routeForwardDelta", label.Aux.RouteForwardDelta)
	appendIfPresent(flat, "move_intent", label.Aux.MoveIntent)
	return flat
}

func flattenGroupedTelemetry(item GroupedTelemetryItem) map[string]any {
	flat := make(map[string]any)
	appendIfPresent(flat, "Steering", item.Control.Steering)
	appendIfPresent(flat, "acceleration", item.Control.Acceleration)
	appendIfPresent(flat, "brakePressureAvg", item.Control.BrakePressureAvg)
	appendIfPresent(flat, "currentSpeed", item.Aux.CurrentSpeed)
	appendIfPresent(flat, "yaw", item.Aux.Yaw)
	appendIfPresent(flat, "yawRate", item.Aux.YawRate)
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

func buildMoveIntentLabel(current map[string]any, moveIntentSpeedTarget float64, currentSpeed float64) bool {
	routeGPSValid, hasRouteGPS := booleanField(current["routeGpsValid"])
	stoppedAtTrafficLight, _ := booleanField(current["isStoppedAtTrafficLights"])
	eventOffroad, _ := booleanField(current["eventOffroad"])
	eventWrongWay, _ := booleanField(current["eventWrongWay"])
	hasLeadVehicle, _ := booleanField(current["hasLeadVehicle"])
	leadVehicleDistance, hasLeadVehicleDistance := numberField(current["leadVehicleDistance"])
	leadVehicleRelativeSpeed, hasLeadVehicleRelativeSpeed := numberField(current["leadVehicleRelativeSpeed"])

	wantsProgress := currentSpeed > 1.0 ||
		(moveIntentSpeedTarget >= currentSpeed+0.5 &&
			moveIntentSpeedTarget >= 1.25)

	if stoppedAtTrafficLight || eventOffroad || eventWrongWay {
		return false
	}

	if hasLeadVehicle && hasLeadVehicleDistance && leadVehicleDistance <= 12.0 {
		relativeSpeed := 0.0
		if hasLeadVehicleRelativeSpeed {
			relativeSpeed = leadVehicleRelativeSpeed
		}
		if relativeSpeed <= 0.5 {
			return false
		}
	}

	if !hasRouteGPS || !routeGPSValid {
		return wantsProgress
	}

	return wantsProgress
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

func forwardFutureSpeed(labels []timedLabel, startIndex int, lookahead int) (float64, bool) {
	if startIndex < 0 || startIndex >= len(labels) {
		return 0, false
	}
	endIndex := startIndex + lookahead
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

func smoothedFutureYaw(labels []timedLabel, centerIndex int, radius int) (float64, bool) {
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

	var sumSin float64
	var sumCos float64
	count := 0
	for index := startIndex; index <= endIndex; index++ {
		yaw, ok := numberField(labels[index].Label["yaw"])
		if !ok {
			continue
		}
		radians := degreesToRadians(normalizeYawDegrees(yaw))
		sumSin += math.Sin(radians)
		sumCos += math.Cos(radians)
		count++
	}
	if count == 0 {
		return 0, false
	}
	return normalizeYawDegrees(radiansToDegrees(math.Atan2(sumSin, sumCos))), true
}

func smoothedYawRate(labels []timedLabel, centerIndex int, radius int) (float64, bool) {
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
		yawRate, ok := resolvedYawRate(labels, index)
		if !ok {
			continue
		}
		sum += yawRate
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

func resolvedYawRate(labels []timedLabel, index int) (float64, bool) {
	if index < 0 || index >= len(labels) {
		return 0, false
	}
	if yawRate, ok := numberField(labels[index].Label["yawRate"]); ok && isFiniteFloat64(yawRate) {
		return yawRate, true
	}

	leftIndex := index - 1
	rightIndex := index + 1
	switch {
	case leftIndex >= 0 && rightIndex < len(labels):
		return derivedYawRateBetween(labels[leftIndex], labels[rightIndex])
	case leftIndex >= 0:
		return derivedYawRateBetween(labels[leftIndex], labels[index])
	case rightIndex < len(labels):
		return derivedYawRateBetween(labels[index], labels[rightIndex])
	default:
		return 0, false
	}
}

func derivedYawRateBetween(left timedLabel, right timedLabel) (float64, bool) {
	leftYaw, ok := numberField(left.Label["yaw"])
	if !ok || !isFiniteFloat64(leftYaw) {
		return 0, false
	}
	rightYaw, ok := numberField(right.Label["yaw"])
	if !ok || !isFiniteFloat64(rightYaw) {
		return 0, false
	}
	deltaSeconds := right.RelativeSeconds - left.RelativeSeconds
	if !isFiniteFloat64(deltaSeconds) || deltaSeconds <= 0 {
		return 0, false
	}
	deltaDegrees := wrapHeadingDeltaDegrees(rightYaw - leftYaw)
	return degreesToRadians(deltaDegrees) / deltaSeconds, true
}

func clampFloat64(value float64, minimum float64, maximum float64) float64 {
	if value < minimum {
		return minimum
	}
	if value > maximum {
		return maximum
	}
	return value
}

func normalizeYawDegrees(value float64) float64 {
	normalized := math.Mod(value, 360.0)
	if normalized < 0 {
		normalized += 360.0
	}
	return normalized
}

func wrapHeadingDeltaDegrees(value float64) float64 {
	delta := math.Mod(value+180.0, 360.0)
	if delta < 0 {
		delta += 360.0
	}
	return delta - 180.0
}

func isFiniteFloat64(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func degreesToRadians(value float64) float64 {
	return value * math.Pi / 180.0
}

func radiansToDegrees(value float64) float64 {
	return value * 180.0 / math.Pi
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
	if len(labels) == 0 {
		return timedLabel{}, -1, false
	}
	idx := sort.Search(len(labels), func(i int) bool {
		return labels[i].RelativeSeconds >= target
	})

	type candidateLabel struct {
		label timedLabel
		index int
	}
	candidates := make([]candidateLabel, 0, 2)
	if idx < len(labels) {
		candidates = append(candidates, candidateLabel{label: labels[idx], index: idx})
	}
	if idx > 0 {
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

func writeDatasetFile(path string, samples []DatasetSample) error {
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create dataset.jsonl: %w", err)
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	for _, sample := range samples {
		if err := encoder.Encode(sample); err != nil {
			return fmt.Errorf("failed to write dataset.jsonl: %w", err)
		}
	}
	return nil
}

func writeStatusFile(path string, status ProcessingStatus) error {
	body, err := json.MarshalIndent(status, "", "  ")
	if err != nil {
		return err
	}
	body = append(body, '\n')
	return os.WriteFile(path, body, 0o644)
}

func ReadStatusFile(path string) (ProcessingStatus, error) {
	var status ProcessingStatus
	body, err := os.ReadFile(path)
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

func filterFramesWithExistingImages(tripDir string, frames []VideoFrame) []VideoFrame {
	filtered := make([]VideoFrame, 0, len(frames))
	for _, frame := range frames {
		if frame.ImagePath == "" {
			continue
		}
		fullPath := filepath.Join(tripDir, filepath.FromSlash(frame.ImagePath))
		if fileExists(fullPath) {
			filtered = append(filtered, frame)
		}
	}
	return filtered
}

func shouldSkipProcessing(tripDir string) bool {
	datasetPath := filepath.Join(tripDir, "dataset.jsonl")
	if fileExists(datasetPath) {
		return true
	}

	framesDir := filepath.Join(tripDir, "frames")
	entries, err := os.ReadDir(framesDir)
	if err != nil {
		return false
	}
	for _, entry := range entries {
		if !entry.IsDir() {
			return true
		}
	}
	return false
}

func shouldSkipDatasetOnlyProcessing(tripDir string) bool {
	return fileExists(filepath.Join(tripDir, "dataset.jsonl"))
}

func (p *Processor) shouldSkipTrip(tripDir string) bool {
	if p.datasetOnly {
		return shouldSkipDatasetOnlyProcessing(tripDir)
	}
	return shouldSkipProcessing(tripDir)
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
