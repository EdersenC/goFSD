package dataset

import (
	"context"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

type TripProcessResult struct {
	TripDir     string
	State       string
	Error       error
	Event       string
	WorkerID    int
	StartedAt   time.Time
	CompletedAt time.Time
	Duration    time.Duration
}

type TripProcessCallback func(TripProcessResult)

func CollectTripDirs(root string, inputs []string) ([]string, error) {
	searchRoots := inputs
	if len(searchRoots) == 0 {
		searchRoots = []string{root}
	}

	seen := make(map[string]struct{})
	tripDirs := make([]string, 0)
	for _, input := range searchRoots {
		cleaned := filepath.Clean(strings.TrimSpace(input))
		if cleaned == "" {
			continue
		}

		info, err := os.Stat(cleaned)
		if err != nil {
			return nil, fmt.Errorf("stat %s: %w", cleaned, err)
		}
		if !info.IsDir() {
			continue
		}

		if strings.HasPrefix(filepath.Base(cleaned), "trip-") {
			if _, ok := seen[cleaned]; !ok {
				seen[cleaned] = struct{}{}
				tripDirs = append(tripDirs, cleaned)
			}
			continue
		}

		err = filepath.WalkDir(cleaned, func(path string, d fs.DirEntry, walkErr error) error {
			if walkErr != nil {
				return walkErr
			}
			if !d.IsDir() {
				return nil
			}
			if strings.HasPrefix(d.Name(), "trip-") {
				if _, ok := seen[path]; !ok {
					seen[path] = struct{}{}
					tripDirs = append(tripDirs, path)
				}
				return filepath.SkipDir
			}
			return nil
		})
		if err != nil {
			return nil, fmt.Errorf("walk %s: %w", cleaned, err)
		}
	}

	sort.Strings(tripDirs)
	return tripDirs, nil
}

func ProcessTripDirs(ctx context.Context, tripDirs []string, workers int, opts ...Option) []TripProcessResult {
	return ProcessTripDirsWithCallback(ctx, tripDirs, workers, nil, opts...)
}

func ProcessTripDirsWithCallback(
	ctx context.Context,
	tripDirs []string,
	workers int,
	callback TripProcessCallback,
	opts ...Option,
) []TripProcessResult {
	if workers < 1 {
		workers = 1
	}
	results := make([]TripProcessResult, 0, len(tripDirs))
	if len(tripDirs) == 0 {
		return results
	}

	jobs := make(chan string)
	resultCh := make(chan TripProcessResult, len(tripDirs)*2)
	var wg sync.WaitGroup

	for i := 0; i < workers; i++ {
		workerID := i + 1
		processor := NewProcessor(opts...)
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for tripDir := range jobs {
				startedAt := time.Now()
				resultCh <- TripProcessResult{
					Event:     "started",
					TripDir:   tripDir,
					WorkerID:  workerID,
					StartedAt: startedAt,
				}

				select {
				case <-ctx.Done():
					resultCh <- TripProcessResult{
						Event:       "finished",
						TripDir:     tripDir,
						State:       "failed",
						Error:       ctx.Err(),
						WorkerID:    workerID,
						StartedAt:   startedAt,
						CompletedAt: time.Now(),
						Duration:    time.Since(startedAt),
					}
					continue
				default:
				}

				statusPath, err := processor.Queue(tripDir)
				if err != nil {
					resultCh <- TripProcessResult{
						Event:       "finished",
						TripDir:     tripDir,
						State:       "failed",
						Error:       err,
						WorkerID:    workerID,
						StartedAt:   startedAt,
						CompletedAt: time.Now(),
						Duration:    time.Since(startedAt),
					}
					continue
				}
				if err := processor.ProcessTrip(ctx, tripDir); err != nil {
					status, readErr := ReadStatusFile(statusPath)
					completedAt := time.Now()
					if readErr == nil {
						resultCh <- TripProcessResult{
							Event:       "finished",
							TripDir:     tripDir,
							State:       status.State,
							Error:       err,
							WorkerID:    workerID,
							StartedAt:   startedAt,
							CompletedAt: completedAt,
							Duration:    completedAt.Sub(startedAt),
						}
					} else {
						resultCh <- TripProcessResult{
							Event:       "finished",
							TripDir:     tripDir,
							State:       "failed",
							Error:       err,
							WorkerID:    workerID,
							StartedAt:   startedAt,
							CompletedAt: completedAt,
							Duration:    completedAt.Sub(startedAt),
						}
					}
					continue
				}
				status, err := ReadStatusFile(statusPath)
				completedAt := time.Now()
				if err != nil {
					resultCh <- TripProcessResult{
						Event:       "finished",
						TripDir:     tripDir,
						State:       "completed",
						Error:       nil,
						WorkerID:    workerID,
						StartedAt:   startedAt,
						CompletedAt: completedAt,
						Duration:    completedAt.Sub(startedAt),
					}
					continue
				}
				resultCh <- TripProcessResult{
					Event:       "finished",
					TripDir:     tripDir,
					State:       status.State,
					Error:       nil,
					WorkerID:    workerID,
					StartedAt:   startedAt,
					CompletedAt: completedAt,
					Duration:    completedAt.Sub(startedAt),
				}
			}
		}(workerID)
	}

	go func() {
		defer close(jobs)
		for _, tripDir := range tripDirs {
			select {
			case <-ctx.Done():
				return
			case jobs <- tripDir:
			}
		}
	}()

	go func() {
		wg.Wait()
		close(resultCh)
	}()

	for result := range resultCh {
		if callback != nil {
			callback(result)
		}
		if result.Event == "" || result.Event == "finished" {
			results = append(results, result)
		}
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].TripDir < results[j].TripDir
	})
	return results
}
