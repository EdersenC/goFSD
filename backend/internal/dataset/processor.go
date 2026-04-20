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
	defaultLabelTolerance           = 100 * time.Millisecond
	defaultDeltaSpeedClip           = 2.0
	defaultDeltaSpeedNormalize      = true
	defaultFlashBrightnessThreshold = 245.0
	defaultFlashFrameLimit          = 90
	defaultStoppedSampleBurst       = 3
	defaultStoppedSampleSpacing     = 2.0
	futureTargetSmoothingRadius     = 2
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
	State       string `json:"state"`
	StartedAt   string `json:"startedAt,omitempty"`
	CompletedAt string `json:"completedAt,omitempty"`
	Error       string `json:"error,omitempty"`
	FramesDir   string `json:"framesDir,omitempty"`
	DatasetFile string `json:"datasetFile,omitempty"`
	ImageWidth  int    `json:"imageWidth,omitempty"`
	ImageHeight int    `json:"imageHeight,omitempty"`
	FrameCount  int    `json:"frameCount,omitempty"`
	SampleCount int    `json:"sampleCount,omitempty"`
}

type DatasetSample struct {
	AnchorVideoPTS float64        `json:"anchor_video_pts"`
	AnchorGameTime float64        `json:"anchor_game_time"`
	FramePaths     []string       `json:"frame_paths"`
	Label          map[string]any `json:"label"`
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
	samples := buildDatasetSamples(
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
	if len(frames) == 0 || len(labels) == 0 || windowSize < 1 || windowSize%2 == 0 || frameStride < 1 || sampleStride < 1 {
		return nil
	}

	halfWindow := windowSize / 2
	startIndex := halfWindow * frameStride
	endIndex := len(frames) - 1 - (halfWindow * frameStride)
	if startIndex > endIndex {
		return nil
	}

	toleranceSeconds := tolerance.Seconds()
	samples := make([]DatasetSample, 0)
	for center := startIndex; center <= endIndex; center += sampleStride {
		futureCenter := center + sampleStride
		if futureCenter > endIndex {
			break
		}

		window := make([]string, 0, windowSize)
		for offset := -halfWindow; offset <= halfWindow; offset++ {
			idx := center + offset*frameStride
			window = append(window, frames[idx].ImagePath)
		}

		centerFrame := frames[center]
		relativeSeconds := centerFrame.PTS - anchorPTS
		label, _, ok := nearestLabelWithIndex(labels, relativeSeconds, toleranceSeconds)
		if !ok {
			continue
		}
		futureRelativeSeconds := frames[futureCenter].PTS - anchorPTS
		futureLabel, futureLabelIndex, ok := nearestLabelWithIndex(labels, futureRelativeSeconds, toleranceSeconds)
		if !ok {
			continue
		}
		futureSpeedTarget, ok := smoothedFutureSpeed(labels, futureLabelIndex, futureTargetSmoothingRadius)
		if !ok {
			continue
		}
		derivedLabel, ok := buildTrainingLabel(
			label.Label,
			futureLabel.Label,
			futureSpeedTarget,
			deltaSpeedClip,
			deltaSpeedNormalize,
		)
		if !ok {
			continue
		}

		samples = append(samples, DatasetSample{
			AnchorVideoPTS: centerFrame.PTS,
			AnchorGameTime: label.RelativeSeconds,
			FramePaths:     window,
			Label:          derivedLabel,
		})
	}

	return samples
}

func buildTrainingLabel(
	current map[string]any,
	future map[string]any,
	futureSpeedTarget float64,
	deltaSpeedClip float64,
	deltaSpeedNormalize bool,
) (map[string]any, bool) {
	currentSpeed, ok := numberField(current["currentSpeed"])
	if !ok {
		return nil, false
	}
	futureSpeed, ok := numberField(future["currentSpeed"])
	if !ok {
		return nil, false
	}
	futureSteer, ok := numberField(future["Steering"])
	if !ok {
		return nil, false
	}

	derived := make(map[string]any, len(current))
	for key, value := range current {
		if key == "acceleration" {
			continue
		}
		derived[key] = value
	}
	deltaSpeed := futureSpeedTarget - currentSpeed
	clippedDeltaSpeed := clampFloat64(deltaSpeed, -deltaSpeedClip, deltaSpeedClip)
	deltaSpeedTarget := clippedDeltaSpeed
	if deltaSpeedNormalize {
		deltaSpeedTarget = clippedDeltaSpeed / deltaSpeedClip
	}
	derived["delta_speed"] = clippedDeltaSpeed
	derived["delta_speed_target"] = deltaSpeedTarget
	derived["future_speed"] = futureSpeed
	derived["future_speed_target"] = futureSpeedTarget
	derived["future_steer"] = futureSteer
	return derived, true
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

func clampFloat64(value float64, minimum float64, maximum float64) float64 {
	if value < minimum {
		return minimum
	}
	if value > maximum {
		return maximum
	}
	return value
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
		if isStoppedLabelValue(sample.Label["isStopped"]) {
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
