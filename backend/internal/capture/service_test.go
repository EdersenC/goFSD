package capture

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestParseWindowTitles(t *testing.T) {
	raw := "\r\nFiveM Main\r\n\r\n  CitizenFX - GTA  \nOther App\n"
	got := parseWindowTitles(raw)
	want := []string{"FiveM Main", "CitizenFX - GTA", "Other App"}

	if len(got) != len(want) {
		t.Fatalf("unexpected titles len: got=%d want=%d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("unexpected title at %d: got=%q want=%q", i, got[i], want[i])
		}
	}
}

func TestBuildSourcesFiltersAndIncludesDesktop(t *testing.T) {
	windows := []windowInfo{
		{Handle: "0x100", Title: ""},
		{Handle: "0x101", Title: "Notepad"},
		{Handle: "0x102", Title: "FiveM - Client"},
		{Handle: "0x103", Title: "fiveM - client"},
		{Handle: "0x104", Title: "CitizenFX - Route"},
	}

	got := buildSources(windows, nil)
	if len(got) != 4 {
		t.Fatalf("unexpected source count: got=%d want=4", len(got))
	}
	last := got[len(got)-1]
	if last.ID != "desktop" || !last.IsFallback {
		t.Fatalf("unexpected fallback source: %+v", last)
	}

	names := []string{got[0].Name, got[1].Name, got[2].Name}
	if !contains(names, "FiveM - Client") {
		t.Fatalf("missing FiveM source: %+v", got)
	}
	if !contains(names, "CitizenFX - Route") {
		t.Fatalf("missing CitizenFX source: %+v", got)
	}
}

func TestBuildSourcesIncludesMonitorTargets(t *testing.T) {
	monitors := []monitorInfo{
		{X: 1920, Y: 0, Width: 1920, Height: 1080, Primary: false},
		{X: 0, Y: 0, Width: 1920, Height: 1080, Primary: true},
	}

	got := buildSources(nil, monitors)
	if len(got) != 3 {
		t.Fatalf("unexpected source count: got=%d want=3", len(got))
	}

	if got[0].ID != "monitor-1" || got[0].CaptureType != "monitor" {
		t.Fatalf("unexpected first monitor source: %+v", got[0])
	}
	if got[0].OffsetX != 0 || got[0].OffsetY != 0 || got[0].Width != 1920 || got[0].Height != 1080 {
		t.Fatalf("unexpected primary monitor bounds: %+v", got[0])
	}

	if got[1].ID != "monitor-2" || got[1].OffsetX != 1920 {
		t.Fatalf("unexpected second monitor source: %+v", got[1])
	}
}

func TestSourceIDFromNameStable(t *testing.T) {
	a := sourceIDFromName("FiveM Main Window")
	b := sourceIDFromName("FiveM Main Window")
	c := sourceIDFromName("Other Window")

	if a != b {
		t.Fatalf("source id must be stable: a=%q b=%q", a, b)
	}
	if a == c {
		t.Fatalf("source id should vary between different names: %q == %q", a, c)
	}
}

func TestStartStopAndDuplicateStart(t *testing.T) {
	ctx := context.Background()
	tmp := t.TempDir()

	svc := NewService(
		WithOutputDir(tmp),
		WithSourceDiscovery(func(context.Context) ([]Source, error) {
			return []Source{
				{ID: "desktop", Name: "Desktop", InputFormat: "gdigrab", Input: "desktop", IsFallback: true},
			}, nil
		}),
		WithCommandFactory(testCommandFactory(t)),
		WithStopTimeout(2*time.Second),
	)

	started, err := svc.Start(ctx, StartRequest{SourceID: "desktop"})
	if err != nil {
		t.Fatalf("start failed: %v", err)
	}
	if started.Status != "started" {
		t.Fatalf("unexpected start status: %q", started.Status)
	}
	if started.PID <= 0 {
		t.Fatalf("unexpected pid: %d", started.PID)
	}
	if filepath.Dir(started.OutputFile) != tmp {
		t.Fatalf("unexpected output dir: %q", started.OutputFile)
	}

	if _, err := svc.Start(ctx, StartRequest{SourceID: "desktop"}); !errorsIs(err, ErrAlreadyRunning) {
		t.Fatalf("expected ErrAlreadyRunning, got: %v", err)
	}

	stopped, err := svc.Stop(ctx)
	if err != nil {
		t.Fatalf("stop failed: %v", err)
	}
	if stopped.Status != "stopped" {
		t.Fatalf("unexpected stop status: %q", stopped.Status)
	}

	if _, err := svc.Stop(ctx); !errorsIs(err, ErrNotRunning) {
		t.Fatalf("expected ErrNotRunning, got: %v", err)
	}
}

func TestStartFailsForUnknownSource(t *testing.T) {
	svc := NewService(
		WithSourceDiscovery(func(context.Context) ([]Source, error) {
			return []Source{
				{ID: "desktop", Name: "Desktop", InputFormat: "gdigrab", Input: "desktop", IsFallback: true},
			}, nil
		}),
	)

	_, err := svc.Start(context.Background(), StartRequest{SourceID: "missing"})
	if !errorsIs(err, ErrSourceNotFound) {
		t.Fatalf("expected ErrSourceNotFound, got: %v", err)
	}
}

func TestBuildFFmpegArgsForMonitorAddsRegion(t *testing.T) {
	src := Source{
		ID:          "monitor-1",
		Name:        "Monitor 1",
		InputFormat: "gdigrab",
		Input:       "desktop",
		CaptureType: "monitor",
		OffsetX:     1920,
		OffsetY:     0,
		Width:       1920,
		Height:      1080,
	}

	args := buildFFmpegArgs(src, "out.mp4")
	if !contains(args, "-offset_x") || !contains(args, "1920") {
		t.Fatalf("expected monitor x offset args, got: %v", args)
	}
	if !contains(args, "-video_size") || !contains(args, "1920x1080") {
		t.Fatalf("expected monitor video_size args, got: %v", args)
	}
	if !contains(args, "scale=1280:720:flags=lanczos,fps=30") {
		t.Fatalf("expected 720p scale filter, got: %v", args)
	}
}

func TestResolveOutputFileAllowsRunRelativePath(t *testing.T) {
	root := t.TempDir()
	capturesDir := filepath.Join(root, "captures")
	svc := NewService(WithOutputDir(capturesDir))

	out, err := svc.resolveOutputFile("session-1", "runs/inner-city-driving/default/run-abc123.mp4")
	if err != nil {
		t.Fatalf("expected nested output path to be accepted, got: %v", err)
	}

	want := filepath.Join(root, "runs", "inner-city-driving", "default", "run-abc123.mp4")
	if out != want {
		t.Fatalf("unexpected output file: got=%q want=%q", out, want)
	}
}

func TestResolveOutputFileRejectsTraversal(t *testing.T) {
	root := t.TempDir()
	capturesDir := filepath.Join(root, "captures")
	svc := NewService(WithOutputDir(capturesDir))

	_, err := svc.resolveOutputFile("session-1", "../outside.mp4")
	if !errorsIs(err, ErrInvalidRequest) {
		t.Fatalf("expected ErrInvalidRequest for traversal, got: %v", err)
	}

	_, err = svc.resolveOutputFile("session-1", "runs/../../outside.mp4")
	if !errorsIs(err, ErrInvalidRequest) {
		t.Fatalf("expected ErrInvalidRequest for nested traversal, got: %v", err)
	}
}

func TestHelperProcessFFmpeg(t *testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") != "1" {
		return
	}

	buf := make([]byte, 1024)
	for {
		n, err := os.Stdin.Read(buf)
		if n > 0 && strings.Contains(string(buf[:n]), "q") {
			os.Exit(0)
		}
		if err != nil {
			os.Exit(0)
		}
	}
}

func testCommandFactory(t *testing.T) CommandFactory {
	t.Helper()
	return func(_ string, _ ...string) *exec.Cmd {
		cmd := exec.Command(os.Args[0], "-test.run=TestHelperProcessFFmpeg", "--")
		cmd.Env = append(os.Environ(), "GO_WANT_HELPER_PROCESS=1")
		return cmd
	}
}

func contains(values []string, target string) bool {
	for _, v := range values {
		if v == target {
			return true
		}
	}
	return false
}

func errorsIs(err, target error) bool {
	return errors.Is(err, target)
}

func TestDiscoverSourcesUnsupportedOutsideWindows(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("windows behavior covered by integration/manual checks")
	}
	_, err := DiscoverSources(context.Background())
	if !errorsIs(err, ErrUnsupportedPlatform) {
		t.Fatalf("expected ErrUnsupportedPlatform, got: %v", err)
	}
}

func TestStartWithoutSourceIDAutoSelectsPreferredWindow(t *testing.T) {
	ctx := context.Background()
	tmp := t.TempDir()
	var capturedArgs []string

	svc := NewService(
		WithOutputDir(tmp),
		WithSourceDiscovery(func(context.Context) ([]Source, error) {
			return []Source{
				{ID: "monitor-1", Name: "Monitor 1", InputFormat: "gdigrab", Input: "desktop", CaptureType: "monitor"},
				{ID: "fxserver", Name: "FiveMr by Cfx.re - FXServer", InputFormat: "gdigrab", Input: "hwnd=0x11", CaptureType: "window"},
				{ID: "game", Name: "FiveM GTA Window", InputFormat: "gdigrab", Input: "hwnd=0x22", CaptureType: "window"},
			}, nil
		}),
		WithCommandFactory(func(_ string, args ...string) *exec.Cmd {
			capturedArgs = append([]string{}, args...)
			cmd := exec.Command(os.Args[0], "-test.run=TestHelperProcessFFmpeg", "--")
			cmd.Env = append(os.Environ(), "GO_WANT_HELPER_PROCESS=1")
			return cmd
		}),
	)

	_, err := svc.Start(ctx, StartRequest{})
	if err != nil {
		t.Fatalf("start failed: %v", err)
	}
	t.Cleanup(func() {
		_, _ = svc.Stop(context.Background())
	})

	if !contains(capturedArgs, "hwnd=0x22") {
		t.Fatalf("expected auto-selected gameplay window input, args=%v", capturedArgs)
	}
}

func TestSelectSourceAutoFallbackOrder(t *testing.T) {
	sources := []Source{
		{ID: "monitor-2", CaptureType: "monitor"},
		{ID: "desktop", CaptureType: "desktop"},
	}
	s, err := selectSource(sources, "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if s.ID != "monitor-2" {
		t.Fatalf("expected monitor fallback, got %s", s.ID)
	}

	_, err = selectSource(sources, "missing")
	if !errors.Is(err, ErrSourceNotFound) {
		t.Fatalf("expected ErrSourceNotFound, got %v", err)
	}

	s2, err := selectSource(sources, "desktop")
	if err != nil {
		t.Fatalf("unexpected error for explicit id: %v", err)
	}
	if s2.ID != "desktop" {
		t.Fatalf("expected explicit desktop selection, got %s", s2.ID)
	}
}

func TestIsPreferredGameWindow(t *testing.T) {
	cases := map[string]bool{
		"FiveM GTA Window":              true,
		"CitizenFX - FiveM":             true,
		"FiveMr by Cfx.re - FXServer":   false,
		"txAdmin - Cfx.re":              false,
		"Random App":                    false,
		"FiveM txAdmin dashboard":       false,
		"CitizenFX fxserver controller": false,
	}

	for in, want := range cases {
		t.Run(fmt.Sprintf("%s", in), func(t *testing.T) {
			got := isPreferredGameWindow(in)
			if got != want {
				t.Fatalf("unexpected preferred result for %q: got=%v want=%v", in, got, want)
			}
		})
	}
}
