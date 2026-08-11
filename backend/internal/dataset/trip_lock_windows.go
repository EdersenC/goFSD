//go:build windows

package dataset

import (
	"errors"
	"os"

	"golang.org/x/sys/windows"
)

type platformFileLock struct {
	overlapped windows.Overlapped
}

func tryLockPlatformFile(file *os.File) (platformFileLock, error) {
	lock := platformFileLock{}
	err := windows.LockFileEx(
		windows.Handle(file.Fd()),
		windows.LOCKFILE_EXCLUSIVE_LOCK|windows.LOCKFILE_FAIL_IMMEDIATELY,
		0,
		1,
		0,
		&lock.overlapped,
	)
	if errors.Is(err, windows.ERROR_LOCK_VIOLATION) {
		return platformFileLock{}, ErrTripProcessingLocked
	}
	return lock, err
}

func unlockPlatformFile(file *os.File, lock platformFileLock) error {
	return windows.UnlockFileEx(windows.Handle(file.Fd()), 0, 1, 0, &lock.overlapped)
}
