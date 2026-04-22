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
		{Handle: "0x102", Title: "FiveM - Client", X: 10, Y: 20, Width: 1600, Height: 900},
		{Handle: "0x103", Title: "fiveM - client", X: 10, Y: 20, Width: 1600, Height: 900},
		{Handle: "0x104", Title: "CitizenFX - Route", X: 100, Y: 50, Width: 1280, Height: 720},
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
	for _, source := range got[:3] {
		if source.CaptureType == "window" && source.Input != "desktop" {
			t.Fatalf("expected FiveM windows to use desktop-region capture, got: %+v", source)
		}
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
		WithCapabilityProbe(func(context.Context, string, string) (bool, error) {
			return true, nil
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
		WithCapabilityProbe(func(context.Context, string, string) (bool, error) {
			return true, nil
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
		InputFormat: "ddagrab",
		Input:       "desktop",
		CaptureType: "monitor",
		OffsetX:     1920,
		OffsetY:     0,
		Width:       1920,
		Height:      1080,
		OutputIndex: 1,
	}

	args := buildFFmpegArgs(monitorCaptureSpec(src), "out.mkv")
	if !contains(args, "-f") || !contains(args, "lavfi") {
		t.Fatalf("expected lavfi input, got: %v", args)
	}
	if !contains(args, "ddagrab=output_idx=1:framerate=30:video_size=1920x1080:offset_x=0:offset_y=0") {
		t.Fatalf("expected ddagrab input string, got: %v", args)
	}
	if !contains(args, "hwdownload,format=bgra,scale=1280:720:flags=lanczos,fps=30") {
		t.Fatalf("expected hwdownload scale filter, got: %v", args)
	}
}

func TestParseWindowsIncludesBounds(t *testing.T) {
	raw := "0x100|FiveM - Client|100|200|1600|900\n0x101|CitizenFX - Route|-1920|0|1920|1080\n"

	got := parseWindows(raw)
	if len(got) != 2 {
		t.Fatalf("unexpected window count: got=%d want=2", len(got))
	}
	if got[0].X != 100 || got[0].Y != 200 || got[0].Width != 1600 || got[0].Height != 900 {
		t.Fatalf("unexpected first window bounds: %+v", got[0])
	}
	if got[1].X != -1920 || got[1].Width != 1920 {
		t.Fatalf("unexpected second window bounds: %+v", got[1])
	}
}

func TestResolveOutputFileAllowsRunRelativePath(t *testing.T) {
	root := t.TempDir()
	capturesDir := filepath.Join(root, "captures")
	svc := NewService(WithOutputDir(capturesDir))

	out, err := svc.resolveOutputFile("session-1", "runs/inner-city-driving/default/run-abc123.mkv")
	if err != nil {
		t.Fatalf("expected nested output path to be accepted, got: %v", err)
	}

	want := filepath.Join(root, "runs", "inner-city-driving", "default", "run-abc123.mkv")
	if out != want {
		t.Fatalf("unexpected output file: got=%q want=%q", out, want)
	}
}

func TestResolveOutputFileRejectsTraversal(t *testing.T) {
	root := t.TempDir()
	capturesDir := filepath.Join(root, "captures")
	svc := NewService(WithOutputDir(capturesDir))

	_, err := svc.resolveOutputFile("session-1", "../outside.mkv")
	if !errorsIs(err, ErrInvalidRequest) {
		t.Fatalf("expected ErrInvalidRequest for traversal, got: %v", err)
	}

	_, err = svc.resolveOutputFile("session-1", "runs/../../outside.mkv")
	if !errorsIs(err, ErrInvalidRequest) {
		t.Fatalf("expected ErrInvalidRequest for nested traversal, got: %v", err)
	}
}

func TestNormalizeDataRootFixesWindowsDriveRelativePaths(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{
			name:  "already absolute windows path",
			input: `S:\fsd_fivem_data`,
			want:  `S:\fsd_fivem_data`,
		},
		{
			name:  "drive relative path becomes absolute",
			input: `S:fsd_fivem_data`,
			want:  `S:\fsd_fivem_data`,
		},
		{
			name:  "quoted path is unwrapped",
			input: `"S:fsd_fivem_data"`,
			want:  `S:\fsd_fivem_data`,
		},
		{
			name:  "bare drive gets slash",
			input: `S:`,
			want:  `S:\`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := NormalizeDataRoot(tc.input)
			if got != tc.want {
				t.Fatalf("unexpected normalized path: got=%q want=%q", got, tc.want)
			}
		})
	}
}

func TestHelperProcessFFmpeg(t *testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") != "1" {
		return
	}

	args := os.Args
	if len(args) > 0 {
		outputFile := args[len(args)-1]
		if strings.HasSuffix(strings.ToLower(outputFile), ".mkv") {
			_ = os.MkdirAll(filepath.Dir(outputFile), 0o755)
			_ = os.WriteFile(outputFile, []byte("fake-video-data"), 0o644)
		}
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
	return func(_ string, args ...string) *exec.Cmd {
		cmd := exec.Command(os.Args[0], append([]string{"-test.run=TestHelperProcessFFmpeg", "--"}, args...)...)
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
				{ID: "monitor-1", Name: "Monitor 1", InputFormat: "ddagrab", Input: "desktop", CaptureType: "monitor", Width: 1920, Height: 1080, OutputIndex: 0},
				{ID: "monitor-2", Name: "Monitor 2", InputFormat: "ddagrab", Input: "desktop", CaptureType: "monitor", OffsetX: 1920, Width: 1920, Height: 1080, OutputIndex: 1},
				{ID: "fxserver", Name: "FiveMr by Cfx.re - FXServer", InputFormat: "ddagrab", Input: "desktop", CaptureType: "window", OffsetX: 100, OffsetY: 100, Width: 900, Height: 700},
				{ID: "game", Name: "FiveM GTA Window", InputFormat: "ddagrab", Input: "desktop", CaptureType: "window", OffsetX: 2100, OffsetY: 50, Width: 1280, Height: 720},
			}, nil
		}),
		WithCapabilityProbe(func(context.Context, string, string) (bool, error) {
			return true, nil
		}),
		WithCommandFactory(func(_ string, args ...string) *exec.Cmd {
			capturedArgs = append([]string{}, args...)
			cmd := exec.Command(os.Args[0], append([]string{"-test.run=TestHelperProcessFFmpeg", "--"}, args...)...)
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

	if !contains(capturedArgs, "ddagrab=output_idx=1:framerate=30:video_size=1280x720:offset_x=180:offset_y=50") {
		t.Fatalf("expected cropped monitor capture for gameplay window, args=%v", capturedArgs)
	}
}

func TestResolveCaptureSpecFallsBackToMonitorTwo(t *testing.T) {
	svc := NewService(WithCapabilityProbe(func(context.Context, string, string) (bool, error) {
		return true, nil
	}))

	sources := []Source{
		{ID: "monitor-1", CaptureType: "monitor", Width: 1920, Height: 1080, OutputIndex: 0},
		{ID: "monitor-2", CaptureType: "monitor", OffsetX: 1920, Width: 1920, Height: 1080, OutputIndex: 1},
	}

	spec, err := svc.resolveCaptureSpec(context.Background(), sources, "", "", true)
	if err != nil {
		t.Fatalf("resolveCaptureSpec: %v", err)
	}
	if spec.selectedMonitorID != "monitor-2" {
		t.Fatalf("expected monitor-2 fallback, got %+v", spec)
	}
}

func TestResolveCaptureSpecRejectsMissingDDAGrab(t *testing.T) {
	svc := NewService(WithCapabilityProbe(func(context.Context, string, string) (bool, error) {
		return false, nil
	}))

	_, err := svc.resolveCaptureSpec(context.Background(), []Source{
		{ID: "monitor-2", CaptureType: "monitor", Width: 1920, Height: 1080, OutputIndex: 1},
	}, "", "", true)
	if !errors.Is(err, ErrUnsupportedFFmpeg) {
		t.Fatalf("expected ErrUnsupportedFFmpeg, got %v", err)
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
