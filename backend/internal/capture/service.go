package capture

import (
	"context"
	"errors"
	"fmt"
	"hash/fnv"
	"io"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	defaultFFmpegBin      = "ffmpeg"
	defaultOutputDir      = "data/captures"
	defaultStopTimeout    = 10 * time.Second
	defaultForcedKillWait = 3 * time.Second
)

var (
	ErrUnsupportedPlatform = errors.New("unsupported platform")
	ErrInvalidRequest      = errors.New("invalid capture request")
	ErrSourceNotFound      = errors.New("capture source not found")
	ErrAlreadyRunning      = errors.New("capture already running")
	ErrNotRunning          = errors.New("capture is not running")
	ErrStartFailed         = errors.New("failed to start capture")
	ErrStopFailed          = errors.New("failed to stop capture")
)

type Source struct {
	ID           string `json:"id"`
	Name         string `json:"name"`
	InputFormat  string `json:"inputFormat"`
	Input        string `json:"input"`
	CaptureType  string `json:"captureType,omitempty"`
	WindowHandle string `json:"windowHandle,omitempty"`
	OffsetX      int    `json:"offsetX,omitempty"`
	OffsetY      int    `json:"offsetY,omitempty"`
	Width        int    `json:"width,omitempty"`
	Height       int    `json:"height,omitempty"`
	IsFallback   bool   `json:"isFallback"`
}

type StartRequest struct {
	SourceID   string `json:"sourceId"`
	OutputFile string `json:"outputFile,omitempty"`
}

type StartResult struct {
	Status     string `json:"status"`
	SessionID  string `json:"sessionId"`
	PID        int    `json:"pid"`
	OutputFile string `json:"outputFile"`
	LogFile    string `json:"logFile"`
}

type StopResult struct {
	Status     string `json:"status"`
	SessionID  string `json:"sessionId"`
	OutputFile string `json:"outputFile"`
	LogFile    string `json:"logFile"`
}

type CommandFactory func(name string, args ...string) *exec.Cmd
type SourceDiscovery func(ctx context.Context) ([]Source, error)

type Option func(*Service)

type Service struct {
	mu sync.Mutex

	ffmpegBin     string
	outputDir     string
	outputRootDir string
	stopTimeout   time.Duration
	forcedKillIn  time.Duration
	nowFunc       func() time.Time

	newCommand CommandFactory
	discover   SourceDiscovery

	active *session
}

type session struct {
	id         string
	outputFile string
	logFile    string
	cmd        *exec.Cmd
	stdin      io.WriteCloser
	done       chan error
}

type monitorInfo struct {
	DeviceName string
	X          int
	Y          int
	Width      int
	Height     int
	Primary    bool
}

type windowInfo struct {
	Handle string
	Title  string
}

func NewService(opts ...Option) *Service {
	outputDir := envOrDefault("CAPTURE_OUTPUT_DIR", defaultOutputDir)
	outputRootDir := envOrDefault("CAPTURE_OUTPUT_ROOT", defaultOutputRootDir(outputDir))

	s := &Service{
		ffmpegBin:     envOrDefault("FFMPEG_BIN", defaultFFmpegBin),
		outputDir:     outputDir,
		outputRootDir: outputRootDir,
		stopTimeout:   parseDurationEnv("CAPTURE_STOP_TIMEOUT", defaultStopTimeout),
		forcedKillIn:  defaultForcedKillWait,
		nowFunc:       time.Now,
		newCommand:    exec.Command,
		discover:      discoverSources,
	}

	for _, opt := range opts {
		opt(s)
	}

	return s
}

func WithCommandFactory(factory CommandFactory) Option {
	return func(s *Service) {
		if factory != nil {
			s.newCommand = factory
		}
	}
}

func WithSourceDiscovery(discovery SourceDiscovery) Option {
	return func(s *Service) {
		if discovery != nil {
			s.discover = discovery
		}
	}
}

func WithOutputDir(dir string) Option {
	return func(s *Service) {
		if strings.TrimSpace(dir) != "" {
			s.outputDir = dir
			s.outputRootDir = defaultOutputRootDir(dir)
		}
	}
}

func WithOutputRootDir(dir string) Option {
	return func(s *Service) {
		if strings.TrimSpace(dir) != "" {
			s.outputRootDir = dir
		}
	}
}

func WithNowFunc(now func() time.Time) Option {
	return func(s *Service) {
		if now != nil {
			s.nowFunc = now
		}
	}
}

func WithStopTimeout(timeout time.Duration) Option {
	return func(s *Service) {
		if timeout > 0 {
			s.stopTimeout = timeout
		}
	}
}

func DiscoverSources(ctx context.Context) ([]Source, error) {
	return discoverSources(ctx)
}

func (s *Service) DiscoverSources(ctx context.Context) ([]Source, error) {
	return s.discover(ctx)
}

func (s *Service) Start(ctx context.Context, req StartRequest) (StartResult, error) {
	sourceID := strings.TrimSpace(req.SourceID)

	sources, err := s.discover(ctx)
	if err != nil {
		return StartResult{}, err
	}

	selected, err := selectSource(sources, sourceID)
	if err != nil {
		return StartResult{}, err
	}

	s.mu.Lock()
	if s.active != nil {
		s.mu.Unlock()
		return StartResult{}, ErrAlreadyRunning
	}

	sessionID := fmt.Sprintf("cap-%d", s.nowFunc().UTC().UnixNano())
	outputFile, err := s.resolveOutputFile(sessionID, req.OutputFile)
	if err != nil {
		s.mu.Unlock()
		return StartResult{}, err
	}
	logFile := strings.TrimSuffix(outputFile, filepath.Ext(outputFile)) + ".log"

	if err := os.MkdirAll(filepath.Dir(outputFile), 0o755); err != nil {
		s.mu.Unlock()
		return StartResult{}, fmt.Errorf("%w: %v", ErrStartFailed, err)
	}

	logHandle, err := os.OpenFile(logFile, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		s.mu.Unlock()
		return StartResult{}, fmt.Errorf("%w: %v", ErrStartFailed, err)
	}

	args := buildFFmpegArgs(selected, outputFile)
	cmd := s.newCommand(s.ffmpegBin, args...)
	cmd.Stdout = logHandle
	cmd.Stderr = logHandle

	stdin, err := cmd.StdinPipe()
	if err != nil {
		s.mu.Unlock()
		_ = logHandle.Close()
		return StartResult{}, fmt.Errorf("%w: %v", ErrStartFailed, err)
	}

	if err := cmd.Start(); err != nil {
		s.mu.Unlock()
		_ = stdin.Close()
		_ = logHandle.Close()
		return StartResult{}, fmt.Errorf("%w: %v", ErrStartFailed, err)
	}

	active := &session{
		id:         sessionID,
		outputFile: outputFile,
		logFile:    logFile,
		cmd:        cmd,
		stdin:      stdin,
		done:       make(chan error, 1),
	}
	s.active = active
	s.mu.Unlock()

	go s.waitForSession(active, logHandle)

	return StartResult{
		Status:     "started",
		SessionID:  sessionID,
		PID:        cmd.Process.Pid,
		OutputFile: outputFile,
		LogFile:    logFile,
	}, nil
}

func (s *Service) Stop(ctx context.Context) (StopResult, error) {
	s.mu.Lock()
	active := s.active
	s.mu.Unlock()

	if active == nil {
		return StopResult{}, ErrNotRunning
	}

	if active.stdin != nil {
		_, _ = io.WriteString(active.stdin, "q\n")
		_ = active.stdin.Close()
	}

	timeout := s.stopTimeout
	if deadline, ok := ctx.Deadline(); ok {
		remaining := time.Until(deadline)
		if remaining > 0 && remaining < timeout {
			timeout = remaining
		}
	}

	select {
	case <-ctx.Done():
		return StopResult{}, fmt.Errorf("%w: %v", ErrStopFailed, ctx.Err())
	case <-time.After(timeout):
		if err := active.cmd.Process.Kill(); err != nil {
			return StopResult{}, fmt.Errorf("%w: %v", ErrStopFailed, err)
		}
		select {
		case err := <-active.done:
			if err != nil && !isExpectedExitErr(err) {
				return StopResult{}, fmt.Errorf("%w: %v", ErrStopFailed, err)
			}
		case <-time.After(s.forcedKillIn):
			return StopResult{}, fmt.Errorf("%w: process did not exit after kill", ErrStopFailed)
		}
	case err := <-active.done:
		if err != nil && !isExpectedExitErr(err) {
			return StopResult{}, fmt.Errorf("%w: %v", ErrStopFailed, err)
		}
	}

	return StopResult{
		Status:     "stopped",
		SessionID:  active.id,
		OutputFile: active.outputFile,
		LogFile:    active.logFile,
	}, nil
}

func (s *Service) waitForSession(active *session, logHandle *os.File) {
	err := active.cmd.Wait()
	_ = logHandle.Close()

	s.mu.Lock()
	if s.active != nil && s.active.id == active.id {
		s.active = nil
	}
	s.mu.Unlock()

	active.done <- err
}

func (s *Service) resolveOutputFile(sessionID, requested string) (string, error) {
	requested = strings.TrimSpace(requested)
	if requested == "" {
		name := sanitizeFileName(sessionID + ".mp4")
		if name == "" {
			return "", ErrInvalidRequest
		}
		return filepath.Join(s.outputDir, name), nil
	}

	cleaned := path.Clean(strings.ReplaceAll(requested, "\\", "/"))
	if cleaned == "." || cleaned == ".." || strings.HasPrefix(cleaned, "../") || strings.HasPrefix(cleaned, "/") || strings.Contains(cleaned, ":") {
		return "", ErrInvalidRequest
	}

	parts := strings.Split(cleaned, "/")
	safeParts := make([]string, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" || part == "." || part == ".." {
			return "", ErrInvalidRequest
		}
		safePart := sanitizeFileName(part)
		if safePart == "" {
			return "", ErrInvalidRequest
		}
		safeParts = append(safeParts, safePart)
	}

	last := safeParts[len(safeParts)-1]
	if ext := strings.ToLower(filepath.Ext(last)); ext == "" {
		last += ".mp4"
	}
	safeParts[len(safeParts)-1] = last

	root := filepath.Clean(s.outputRootDir)
	resolved := filepath.Clean(filepath.Join(root, filepath.Join(safeParts...)))
	if !isWithinRoot(root, resolved) {
		return "", ErrInvalidRequest
	}

	return resolved, nil
}

func discoverSources(ctx context.Context) ([]Source, error) {
	if runtime.GOOS != "windows" {
		return nil, ErrUnsupportedPlatform
	}

	windows, err := queryWindowsWindows(ctx)
	if err != nil {
		return nil, err
	}

	monitors, err := queryWindowsMonitors(ctx)
	if err != nil {
		monitors = nil
	}

	return buildSources(windows, monitors), nil
}

func buildSources(windows []windowInfo, monitors []monitorInfo) []Source {
	sources := make([]Source, 0, len(windows)+len(monitors)+1)
	usedIDs := map[string]struct{}{}

	sort.Slice(monitors, func(i, j int) bool {
		if monitors[i].Primary != monitors[j].Primary {
			return monitors[i].Primary
		}
		if monitors[i].X != monitors[j].X {
			return monitors[i].X < monitors[j].X
		}
		return monitors[i].Y < monitors[j].Y
	})

	for i, mon := range monitors {
		id := fmt.Sprintf("monitor-%d", i+1)
		usedIDs[id] = struct{}{}

		name := fmt.Sprintf("Monitor %d (%dx%d @ %d,%d)", i+1, mon.Width, mon.Height, mon.X, mon.Y)
		if mon.Primary {
			name += " [primary]"
		}

		sources = append(sources, Source{
			ID:          id,
			Name:        name,
			InputFormat: "gdigrab",
			Input:       "desktop",
			CaptureType: "monitor",
			OffsetX:     mon.X,
			OffsetY:     mon.Y,
			Width:       mon.Width,
			Height:      mon.Height,
		})
	}

	seenHandle := map[string]struct{}{}
	windowSources := make([]Source, 0, len(windows))

	for _, w := range windows {
		title := strings.TrimSpace(w.Title)
		handle := strings.TrimSpace(w.Handle)
		if title == "" || handle == "" || !isFiveMTitle(title) {
			continue
		}

		dedupeKey := strings.ToLower(handle)
		if _, ok := seenHandle[dedupeKey]; ok {
			continue
		}
		seenHandle[dedupeKey] = struct{}{}

		id := sourceIDFromName(handle + ":" + title)
		if _, exists := usedIDs[id]; exists {
			for i := 2; ; i++ {
				candidate := id + "-" + strconv.Itoa(i)
				if _, taken := usedIDs[candidate]; !taken {
					id = candidate
					break
				}
			}
		}
		usedIDs[id] = struct{}{}

		windowSources = append(windowSources, Source{
			ID:           id,
			Name:         title,
			InputFormat:  "gdigrab",
			Input:        "hwnd=" + handle,
			CaptureType:  "window",
			WindowHandle: handle,
		})
	}

	sort.Slice(windowSources, func(i, j int) bool {
		return strings.ToLower(windowSources[i].Name) < strings.ToLower(windowSources[j].Name)
	})
	sources = append(sources, windowSources...)

	sources = append(sources, Source{
		ID:          "desktop",
		Name:        "Desktop (all monitors)",
		InputFormat: "gdigrab",
		Input:       "desktop",
		CaptureType: "desktop",
		IsFallback:  true,
	})

	return sources
}

func parseWindowTitles(raw string) []string {
	lines := strings.Split(raw, "\n")
	titles := make([]string, 0, len(lines))
	for _, line := range lines {
		line = strings.TrimSpace(strings.TrimSuffix(line, "\r"))
		if line == "" {
			continue
		}
		titles = append(titles, line)
	}
	return titles
}

func queryWindowsWindows(ctx context.Context) ([]windowInfo, error) {
	script := `$ErrorActionPreference = "Stop"
Add-Type @"
using System;
using System.Runtime.InteropServices;
using System.Text;

public class Win32 {
  public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);
  [DllImport("user32.dll")] public static extern bool EnumWindows(EnumWindowsProc enumProc, IntPtr lParam);
  [DllImport("user32.dll")] public static extern int GetWindowText(IntPtr hWnd, StringBuilder text, int count);
  [DllImport("user32.dll")] public static extern int GetWindowTextLength(IntPtr hWnd);
  [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr hWnd);
}
"@

$windows = New-Object System.Collections.Generic.List[string]
[Win32]::EnumWindows({
  param($hWnd, $lParam)
  if (-not [Win32]::IsWindowVisible($hWnd)) { return $true }
  $len = [Win32]::GetWindowTextLength($hWnd)
  if ($len -le 0) { return $true }
  $sb = New-Object System.Text.StringBuilder($len + 1)
  [void][Win32]::GetWindowText($hWnd, $sb, $sb.Capacity)
  $title = $sb.ToString().Trim()
  if (-not [string]::IsNullOrWhiteSpace($title)) {
    $handle = ('0x{0:X}' -f [Int64]$hWnd)
    $windows.Add("$handle|$title")
  }
  return $true
}, [IntPtr]::Zero) | Out-Null

$windows | Sort-Object -Unique`

	output, err := runPowerShell(ctx, script)
	if err != nil {
		return nil, err
	}
	return parseWindows(output), nil
}

func parseWindows(raw string) []windowInfo {
	lines := strings.Split(raw, "\n")
	windows := make([]windowInfo, 0, len(lines))
	for _, line := range lines {
		line = strings.TrimSpace(strings.TrimSuffix(line, "\r"))
		if line == "" {
			continue
		}

		parts := strings.SplitN(line, "|", 2)
		if len(parts) != 2 {
			continue
		}
		handle := strings.TrimSpace(parts[0])
		title := strings.TrimSpace(parts[1])
		if handle == "" || title == "" {
			continue
		}

		windows = append(windows, windowInfo{
			Handle: handle,
			Title:  title,
		})
	}
	return windows
}

func queryWindowsMonitors(ctx context.Context) ([]monitorInfo, error) {
	script := `$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Windows.Forms
[System.Windows.Forms.Screen]::AllScreens | ForEach-Object {
  $b = $_.Bounds
  "$($_.DeviceName)|$($b.X)|$($b.Y)|$($b.Width)|$($b.Height)|$($_.Primary)"
}`

	output, err := runPowerShell(ctx, script)
	if err != nil {
		return nil, err
	}
	return parseMonitorInfo(output), nil
}

func parseMonitorInfo(raw string) []monitorInfo {
	lines := strings.Split(raw, "\n")
	monitors := make([]monitorInfo, 0, len(lines))

	for _, line := range lines {
		line = strings.TrimSpace(strings.TrimSuffix(line, "\r"))
		if line == "" {
			continue
		}

		parts := strings.Split(line, "|")
		if len(parts) != 6 {
			continue
		}

		x, errX := strconv.Atoi(strings.TrimSpace(parts[1]))
		y, errY := strconv.Atoi(strings.TrimSpace(parts[2]))
		w, errW := strconv.Atoi(strings.TrimSpace(parts[3]))
		h, errH := strconv.Atoi(strings.TrimSpace(parts[4]))
		if errX != nil || errY != nil || errW != nil || errH != nil || w <= 0 || h <= 0 {
			continue
		}

		monitors = append(monitors, monitorInfo{
			DeviceName: strings.TrimSpace(parts[0]),
			X:          x,
			Y:          y,
			Width:      w,
			Height:     h,
			Primary:    strings.EqualFold(strings.TrimSpace(parts[5]), "true"),
		})
	}

	return monitors
}

func buildFFmpegArgs(selected Source, outputFile string) []string {
	args := []string{
		"-y",
		"-f", selected.InputFormat,
		"-framerate", "30",
	}

	if selected.InputFormat == "gdigrab" {
		args = append(args, "-draw_mouse", "1")
		if strings.EqualFold(selected.Input, "desktop") && selected.Width > 0 && selected.Height > 0 {
			args = append(args,
				"-offset_x", strconv.Itoa(selected.OffsetX),
				"-offset_y", strconv.Itoa(selected.OffsetY),
				"-video_size", fmt.Sprintf("%dx%d", selected.Width, selected.Height),
			)
		}
	}

	args = append(args,
		"-i", selected.Input,
		"-vf", "scale=1280:720:flags=lanczos,fps=30",
		"-c:v", "libx264",
		"-pix_fmt", "yuv420p",
		"-profile:v", "main",
		"-level:v", "4.0",
		"-preset", "veryfast",
		"-crf", "23",
		"-movflags", "+faststart",
		outputFile,
	)

	return args
}

func selectSource(sources []Source, sourceID string) (Source, error) {
	if len(sources) == 0 {
		return Source{}, ErrSourceNotFound
	}

	if sourceID != "" {
		for i := range sources {
			if sources[i].ID == sourceID {
				return sources[i], nil
			}
		}
		return Source{}, ErrSourceNotFound
	}

	for i := range sources {
		if sources[i].CaptureType == "window" && isPreferredGameWindow(sources[i].Name) {
			return sources[i], nil
		}
	}

	for i := range sources {
		if sources[i].CaptureType == "window" {
			return sources[i], nil
		}
	}

	for i := range sources {
		if sources[i].ID == "monitor-1" {
			return sources[i], nil
		}
	}

	for i := range sources {
		if sources[i].CaptureType == "monitor" {
			return sources[i], nil
		}
	}

	for i := range sources {
		if sources[i].ID == "desktop" {
			return sources[i], nil
		}
	}

	return sources[0], nil
}

func isPreferredGameWindow(name string) bool {
	n := strings.ToLower(strings.TrimSpace(name))
	if n == "" {
		return false
	}
	if !strings.Contains(n, "fivem") && !strings.Contains(n, "citizenfx") {
		return false
	}
	// Exclude likely server/admin windows when auto-selecting.
	if strings.Contains(n, "fxserver") || strings.Contains(n, "txadmin") {
		return false
	}
	return true
}

func runPowerShell(ctx context.Context, script string) (string, error) {
	var lastErr error
	for _, binary := range []string{"powershell", "pwsh"} {
		cmd := exec.CommandContext(ctx, binary, "-NoProfile", "-NonInteractive", "-Command", script)
		out, err := cmd.CombinedOutput()
		if err == nil {
			return string(out), nil
		}
		lastErr = fmt.Errorf("%s: %w (%s)", binary, err, strings.TrimSpace(string(out)))
		if !isNotFound(err) {
			break
		}
	}
	return "", lastErr
}

func isNotFound(err error) bool {
	var execErr *exec.Error
	return errors.As(err, &execErr) && errors.Is(execErr, exec.ErrNotFound)
}

func isFiveMTitle(title string) bool {
	t := strings.ToLower(title)
	return strings.Contains(t, "fivem") || strings.Contains(t, "citizenfx")
}

func sourceIDFromName(name string) string {
	base := sanitizeToken(strings.ToLower(name))
	if base == "" {
		base = "window"
	}

	h := fnv.New32a()
	_, _ = h.Write([]byte(name))
	hash := strconv.FormatUint(uint64(h.Sum32()), 16)

	return base + "-" + hash[:8]
}

func sanitizeToken(s string) string {
	var b strings.Builder
	lastDash := false
	for _, r := range s {
		ok := (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9')
		if ok {
			b.WriteRune(r)
			lastDash = false
			continue
		}
		if !lastDash {
			b.WriteByte('-')
			lastDash = true
		}
	}
	return strings.Trim(b.String(), "-")
}

func sanitizeFileName(name string) string {
	var b strings.Builder
	for _, r := range name {
		valid := (r >= 'a' && r <= 'z') ||
			(r >= 'A' && r <= 'Z') ||
			(r >= '0' && r <= '9') ||
			r == '.' || r == '_' || r == '-'
		if valid {
			b.WriteRune(r)
		} else {
			b.WriteByte('_')
		}
	}
	return strings.TrimSpace(strings.Trim(b.String(), "._"))
}

func parseDurationEnv(key string, fallback time.Duration) time.Duration {
	raw := strings.TrimSpace(os.Getenv(key))
	if raw == "" {
		return fallback
	}
	d, err := time.ParseDuration(raw)
	if err != nil || d <= 0 {
		return fallback
	}
	return d
}

func envOrDefault(key, fallback string) string {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return fallback
	}
	return v
}

func defaultOutputRootDir(outputDir string) string {
	clean := filepath.Clean(outputDir)
	if strings.EqualFold(filepath.Base(clean), "captures") {
		return filepath.Clean(filepath.Join(clean, ".."))
	}
	return clean
}

func isWithinRoot(root, resolved string) bool {
	rel, err := filepath.Rel(root, resolved)
	if err != nil {
		return false
	}
	if rel == ".." {
		return false
	}
	return !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}

func isExpectedExitErr(err error) bool {
	var exitErr *exec.ExitError
	return errors.As(err, &exitErr)
}
