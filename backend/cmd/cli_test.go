package main

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"
)

func TestDispatchBackendCommandDefaultsToServer(t *testing.T) {
	for _, args := range [][]string{nil, {"serve"}} {
		handled, err := dispatchBackendCommand(args, &bytes.Buffer{})
		if err != nil {
			t.Fatalf("dispatchBackendCommand(%v): %v", args, err)
		}
		if handled {
			t.Fatalf("expected server command to continue startup: %v", args)
		}
	}
}

func TestDispatchBackendCommandRejectsUnknownCommand(t *testing.T) {
	handled, err := dispatchBackendCommand([]string{"proces-runs"}, &bytes.Buffer{})
	if !handled {
		t.Fatal("unknown command must stop server startup")
	}
	if err == nil || !strings.Contains(err.Error(), "unknown backend command") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestDispatchBackendCommandPrintsHelp(t *testing.T) {
	var output bytes.Buffer
	handled, err := dispatchBackendCommand([]string{"--help"}, &output)
	if err != nil {
		t.Fatalf("dispatchBackendCommand: %v", err)
	}
	if !handled {
		t.Fatal("help must stop server startup")
	}
	for _, marker := range []string{"process-runs", "report-runs", "processing-fingerprint", "processing-status"} {
		if !strings.Contains(output.String(), marker) {
			t.Fatalf("help output is missing %q: %s", marker, output.String())
		}
	}
}

func TestDispatchBackendCommandRejectsProcessingFingerprintArguments(t *testing.T) {
	handled, err := dispatchBackendCommand([]string{"processing-fingerprint", "unexpected"}, &bytes.Buffer{})
	if !handled || err == nil || !strings.Contains(err.Error(), "does not accept arguments") {
		t.Fatalf("unexpected processing-fingerprint result: handled=%t err=%v", handled, err)
	}
}

func TestDispatchBackendCommandTreatsSubcommandHelpAsSuccess(t *testing.T) {
	for _, command := range []string{"process-runs", "report-runs"} {
		handled, err := dispatchBackendCommand([]string{command, "-h"}, &bytes.Buffer{})
		if err != nil {
			t.Fatalf("%s help returned an error: %v", command, err)
		}
		if !handled {
			t.Fatalf("%s help must stop server startup", command)
		}
	}
}

func TestRunProcessRunsRejectsInvalidWorkerCount(t *testing.T) {
	if err := runProcessRuns([]string{"-workers", "0"}); err == nil || !strings.Contains(err.Error(), "workers must be at least 1") {
		t.Fatalf("unexpected invalid-worker result: %v", err)
	}
}

func TestProcessRunErrorReportsInterruptionAndIncompleteDispatch(t *testing.T) {
	interrupted := processRunError(12, 5, 0, context.Canceled)
	if !errors.Is(interrupted, context.Canceled) || !strings.Contains(interrupted.Error(), "5 of 12") {
		t.Fatalf("unexpected interruption error: %v", interrupted)
	}

	incomplete := processRunError(12, 11, 0, nil)
	if incomplete == nil || !strings.Contains(incomplete.Error(), "11 of 12") {
		t.Fatalf("unexpected incomplete-dispatch error: %v", incomplete)
	}

	if err := processRunError(12, 12, 0, nil); err != nil {
		t.Fatalf("complete successful run returned an error: %v", err)
	}
}
