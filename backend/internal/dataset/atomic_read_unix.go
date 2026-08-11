//go:build !windows

package dataset

import "os"

func readFileConsistently(path string) ([]byte, error) {
	return os.ReadFile(path)
}
