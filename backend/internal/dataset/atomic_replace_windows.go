//go:build windows

package dataset

import (
	"errors"
	"fmt"
	"time"

	"golang.org/x/sys/windows"
)

const atomicReplaceTimeout = 5 * time.Second

func replaceFileAtomically(source string, target string) error {
	sourcePath, err := windows.UTF16PtrFromString(source)
	if err != nil {
		return fmt.Errorf("encode atomic replacement source: %w", err)
	}
	targetPath, err := windows.UTF16PtrFromString(target)
	if err != nil {
		return fmt.Errorf("encode atomic replacement target: %w", err)
	}

	deadline := time.Now().Add(atomicReplaceTimeout)
	for {
		err = windows.MoveFileEx(
			sourcePath,
			targetPath,
			windows.MOVEFILE_REPLACE_EXISTING|windows.MOVEFILE_WRITE_THROUGH,
		)
		if err == nil {
			return nil
		}
		if !errors.Is(err, windows.ERROR_SHARING_VIOLATION) && !errors.Is(err, windows.ERROR_ACCESS_DENIED) {
			return err
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("timed out waiting for readers to release target: %w", err)
		}
		time.Sleep(time.Millisecond)
	}
}
