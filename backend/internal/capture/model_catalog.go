package capture

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"
)

var epochCheckpointPattern = regexp.MustCompile(`^epoch-(\d+)\.pt$`)

type InferenceModelOption struct {
	Label     string `json:"label"`
	Path      string `json:"path"`
	RunID     string `json:"runId,omitempty"`
	Epoch     int    `json:"epoch,omitempty"`
	IsBest    bool   `json:"isBest"`
	UpdatedAt string `json:"updatedAt,omitempty"`
}

type runMetricsSummary struct {
	BestEpoch int `json:"best_epoch"`
}

func DiscoverInferenceModels(configPath string) ([]InferenceModelOption, error) {
	searchRoot, err := resolveTrainingSearchRoot(configPath)
	if err != nil {
		return nil, err
	}

	runBestEpoch := make(map[string]int)
	_ = filepath.WalkDir(searchRoot, func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || d.Name() != "run_metrics.json" {
			return nil
		}
		raw, readErr := os.ReadFile(path)
		if readErr != nil {
			return nil
		}
		var metrics runMetricsSummary
		if jsonErr := json.Unmarshal(raw, &metrics); jsonErr != nil {
			return nil
		}
		runBestEpoch[filepath.Dir(path)] = metrics.BestEpoch
		return nil
	})

	models := make([]InferenceModelOption, 0)
	if err := filepath.WalkDir(searchRoot, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if d.IsDir() {
			return nil
		}
		matches := epochCheckpointPattern.FindStringSubmatch(d.Name())
		if len(matches) != 2 {
			return nil
		}
		epoch := 0
		fmt.Sscanf(matches[1], "%d", &epoch)
		info, err := d.Info()
		if err != nil {
			return nil
		}
		runDir := filepath.Base(filepath.Dir(path))
		bestEpoch := runBestEpoch[filepath.Dir(path)]
		label := fmt.Sprintf("%s - epoch %03d", runDir, epoch)
		if epoch == bestEpoch && bestEpoch > 0 {
			label += " (best)"
		}
		models = append(models, InferenceModelOption{
			Label:     label,
			Path:      path,
			RunID:     runDir,
			Epoch:     epoch,
			IsBest:    epoch == bestEpoch && bestEpoch > 0,
			UpdatedAt: info.ModTime().UTC().Format(time.RFC3339Nano),
		})
		return nil
	}); err != nil {
		return nil, err
	}

	sort.Slice(models, func(i, j int) bool {
		if models[i].RunID != models[j].RunID {
			return models[i].RunID > models[j].RunID
		}
		if models[i].IsBest != models[j].IsBest {
			return models[i].IsBest
		}
		if models[i].Epoch != models[j].Epoch {
			return models[i].Epoch > models[j].Epoch
		}
		return models[i].Path < models[j].Path
	})
	return models, nil
}

func resolveTrainingSearchRoot(configPath string) (string, error) {
	if strings.TrimSpace(configPath) != "" {
		configDir := filepath.Dir(configPath)
		if _, err := os.Stat(configDir); err == nil {
			return configDir, nil
		}
	}
	cwd, err := os.Getwd()
	if err != nil {
		return "", err
	}
	candidate := filepath.Join(cwd, "..", "fsd_trainer")
	if _, err := os.Stat(candidate); err == nil {
		return filepath.Clean(candidate), nil
	}
	candidate = filepath.Join(cwd, "fsd_trainer")
	if _, err := os.Stat(candidate); err == nil {
		return filepath.Clean(candidate), nil
	}
	return "", fmt.Errorf("training runs root not found")
}
