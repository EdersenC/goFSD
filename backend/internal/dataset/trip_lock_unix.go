//go:build !windows

package dataset

import (
	"errors"
	"os"

	"golang.org/x/sys/unix"
)

type platformFileLock struct{}

func tryLockPlatformFile(file *os.File) (platformFileLock, error) {
	err := unix.Flock(int(file.Fd()), unix.LOCK_EX|unix.LOCK_NB)
	if errors.Is(err, unix.EWOULDBLOCK) || errors.Is(err, unix.EAGAIN) {
		return platformFileLock{}, ErrTripProcessingLocked
	}
	return platformFileLock{}, err
}

func unlockPlatformFile(file *os.File, _ platformFileLock) error {
	return unix.Flock(int(file.Fd()), unix.LOCK_UN)
}
