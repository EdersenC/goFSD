package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"strings"
)

func dispatchBackendCommand(args []string, output io.Writer) (bool, error) {
	if len(args) == 0 {
		return false, nil
	}

	command := strings.TrimSpace(args[0])
	commandArgs := args[1:]
	switch command {
	case "serve":
		if len(commandArgs) > 0 {
			return true, fmt.Errorf("serve does not accept arguments: %s", strings.Join(commandArgs, " "))
		}
		return false, nil
	case "process-runs":
		return true, normalizeCommandHelp(runProcessRuns(commandArgs))
	case "report-runs":
		return true, normalizeCommandHelp(runReportRuns(commandArgs))
	case "processing-fingerprint":
		if len(commandArgs) > 0 {
			return true, fmt.Errorf("processing-fingerprint does not accept arguments: %s", strings.Join(commandArgs, " "))
		}
		return true, runProcessingFingerprint(output)
	case "processing-status":
		return true, normalizeCommandHelp(runProcessingStatus(commandArgs, output))
	case "help", "-h", "--help":
		writeBackendUsage(output)
		return true, nil
	default:
		return true, fmt.Errorf("unknown backend command %q; run with --help for available commands", command)
	}
}

func normalizeCommandHelp(err error) error {
	if errors.Is(err, flag.ErrHelp) {
		return nil
	}
	return err
}

func writeBackendUsage(output io.Writer) {
	if output == nil {
		return
	}
	fmt.Fprintln(output, "Stop Sign Lab backend")
	fmt.Fprintln(output)
	fmt.Fprintln(output, "Usage:")
	fmt.Fprintln(output, "  backend [serve]")
	fmt.Fprintln(output, "  backend process-runs [flags] [trip-or-run ...]")
	fmt.Fprintln(output, "  backend report-runs [flags] [trip-or-run ...]")
	fmt.Fprintln(output, "  backend processing-fingerprint")
	fmt.Fprintln(output, "  backend processing-status [-root path] [-stop-sign-only]")
}
