//go:build !windows

package dataset

import "os"

func replaceFileAtomically(source string, target string) error {
	return os.Rename(source, target)
}
