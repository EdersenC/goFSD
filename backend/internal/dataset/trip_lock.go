package dataset

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

const tripProcessingLockFile = ".processing.lock"

var ErrTripProcessingLocked = errors.New("trip is already being processed")

type tripProcessingLock struct {
	file     *os.File
	platform platformFileLock
}

func acquireTripProcessingLock(tripDir string) (*tripProcessingLock, error) {
	path := filepath.Join(tripDir, tripProcessingLockFile)
	file, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return nil, fmt.Errorf("open trip processing lock: %w", err)
	}
	platform, err := tryLockPlatformFile(file)
	if err != nil {
		_ = file.Close()
		if errors.Is(err, ErrTripProcessingLocked) {
			return nil, fmt.Errorf("%w: %s", ErrTripProcessingLocked, tripDir)
		}
		return nil, fmt.Errorf("lock trip processing file: %w", err)
	}
	return &tripProcessingLock{file: file, platform: platform}, nil
}

func (l *tripProcessingLock) Release() error {
	if l == nil || l.file == nil {
		return nil
	}
	unlockErr := unlockPlatformFile(l.file, l.platform)
	closeErr := l.file.Close()
	l.file = nil
	if closeErr != nil {
		return errors.Join(unlockErr, closeErr)
	}
	return nil
}
