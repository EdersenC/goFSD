package training

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	toml "github.com/pelletier/go-toml/v2"
)

type Config struct {
	ConfigPath  string
	PythonBin   string
	TrainScript string
	JobsDir     string
}

type PageConfig struct {
	ConfigPath             string             `json:"configPath"`
	PythonBin              string             `json:"pythonBin"`
	TrainScript            string             `json:"trainScript"`
	JobsDir                string             `json:"jobsDir"`
	LearningRate           float64            `json:"learningRate"`
	LossWeights            map[string]float64 `json:"lossWeights"`
	Consistency            map[string]float64 `json:"consistency"`
	AllowedLossWeightKeys  []string           `json:"allowedLossWeightKeys"`
	AllowedConsistencyKeys []string           `json:"allowedConsistencyKeys"`
}

type configFile struct {
	Backend backendSection `toml:"backend"`
}

type backendSection struct {
	Training trainingSection `toml:"training"`
}

type trainingSection struct {
	PythonBin   string `toml:"python_bin"`
	TrainScript string `toml:"train_script"`
	JobsDir     string `toml:"jobs_dir"`
}

type trainingDefaultsFile struct {
	Training trainingDefaultsSection `toml:"training"`
}

type trainingDefaultsSection struct {
	LearningRate float64            `toml:"learning_rate"`
	LossWeights  map[string]float64 `toml:"loss_weights"`
	Consistency  map[string]float64 `toml:"consistency"`
}

var (
	allowedLossWeightKeys = []string{
		"future_yaw_delta",
		"future_speed",
		"move_intent",
		"delta_speed",
		"yaw_rate",
	}
	allowedConsistencyKeys = []string{
		"yaw_delta_vs_yaw_rate_weight",
		"yaw_rate_scale_to_degrees",
		"future_speed_vs_delta_speed_weight",
	}
)

func LoadConfig(path string, defaultDataRoot string) (Config, error) {
	cfg := Config{
		ConfigPath: strings.TrimSpace(path),
	}

	section, exists, err := loadTrainingSection(path)
	if err != nil {
		return Config{}, err
	}

	if cfg.ConfigPath == "" && exists {
		cfg.ConfigPath = strings.TrimSpace(path)
	}

	projectRoot := resolveProjectRoot(path)
	configDir := ""
	if strings.TrimSpace(path) != "" {
		configDir = filepath.Dir(path)
	}

	cfg.PythonBin = resolvePythonBin(projectRoot, configDir, strings.TrimSpace(section.PythonBin))
	cfg.TrainScript = resolveTrainScript(projectRoot, configDir, strings.TrimSpace(section.TrainScript))
	cfg.JobsDir = resolveJobsDir(defaultDataRoot, projectRoot, configDir, strings.TrimSpace(section.JobsDir))

	if strings.TrimSpace(cfg.PythonBin) == "" {
		return Config{}, fmt.Errorf("backend.training.python_bin could not be resolved")
	}
	if strings.TrimSpace(cfg.TrainScript) == "" {
		return Config{}, fmt.Errorf("backend.training.train_script could not be resolved")
	}
	return cfg, nil
}

func LoadPageConfig(baseConfigPath string, runtimeConfig Config) (PageConfig, error) {
	page := PageConfig{
		ConfigPath:             strings.TrimSpace(baseConfigPath),
		PythonBin:              runtimeConfig.PythonBin,
		TrainScript:            runtimeConfig.TrainScript,
		JobsDir:                runtimeConfig.JobsDir,
		LossWeights:            make(map[string]float64),
		Consistency:            make(map[string]float64),
		AllowedLossWeightKeys:  append([]string(nil), allowedLossWeightKeys...),
		AllowedConsistencyKeys: append([]string(nil), allowedConsistencyKeys...),
	}
	if strings.TrimSpace(baseConfigPath) == "" {
		return page, nil
	}

	raw, err := os.ReadFile(baseConfigPath)
	if err != nil {
		return PageConfig{}, err
	}
	var parsed trainingDefaultsFile
	if err := toml.Unmarshal(raw, &parsed); err != nil {
		return PageConfig{}, err
	}

	page.LearningRate = parsed.Training.LearningRate
	for _, key := range allowedLossWeightKeys {
		page.LossWeights[key] = parsed.Training.LossWeights[key]
	}
	for _, key := range allowedConsistencyKeys {
		page.Consistency[key] = parsed.Training.Consistency[key]
	}
	return page, nil
}

func loadTrainingSection(path string) (trainingSection, bool, error) {
	var parsed configFile
	if strings.TrimSpace(path) == "" {
		return trainingSection{}, false, nil
	}

	raw, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return trainingSection{}, false, nil
		}
		return trainingSection{}, false, err
	}
	if err := toml.Unmarshal(raw, &parsed); err != nil {
		return trainingSection{}, false, err
	}
	return parsed.Backend.Training, true, nil
}

func resolveProjectRoot(configPath string) string {
	if strings.TrimSpace(configPath) == "" {
		if cwd, err := os.Getwd(); err == nil {
			return cwd
		}
		return ""
	}
	configDir := filepath.Dir(configPath)
	return filepath.Clean(filepath.Join(configDir, ".."))
}

func resolvePythonBin(projectRoot string, configDir string, explicit string) string {
	explicit = normalizeConfiguredPath(explicit)
	if explicit != "" {
		return resolveConfiguredExecutable(projectRoot, configDir, explicit)
	}

	candidates := []string{}
	for _, root := range []string{projectRoot, configDir} {
		if strings.TrimSpace(root) == "" {
			continue
		}
		if runtime.GOOS == "windows" {
			candidates = append(candidates,
				filepath.Join(root, ".venv", "Scripts", "python.exe"),
				filepath.Join(root, "fsd_trainer", ".venv", "Scripts", "python.exe"),
			)
		} else {
			candidates = append(candidates,
				filepath.Join(root, ".venv", "bin", "python"),
				filepath.Join(root, "fsd_trainer", ".venv", "bin", "python"),
			)
		}
	}
	for _, candidate := range candidates {
		if fileExists(candidate) {
			return candidate
		}
	}
	if runtime.GOOS == "windows" {
		return "python"
	}
	return "python3"
}

func resolveTrainScript(projectRoot string, configDir string, explicit string) string {
	explicit = normalizeConfiguredPath(explicit)
	if explicit != "" {
		return resolveConfiguredPath(projectRoot, configDir, explicit)
	}
	if strings.TrimSpace(projectRoot) == "" {
		return filepath.Clean(filepath.Join("fsd_trainer", "src", "gta_fsd", "train.py"))
	}
	return filepath.Join(projectRoot, "fsd_trainer", "src", "gta_fsd", "train.py")
}

func resolveJobsDir(defaultDataRoot string, projectRoot string, configDir string, explicit string) string {
	explicit = normalizeConfiguredPath(explicit)
	if explicit != "" {
		if filepath.IsAbs(explicit) {
			return explicit
		}
		if configDir != "" {
			return filepath.Clean(filepath.Join(configDir, explicit))
		}
		return explicit
	}
	if strings.TrimSpace(defaultDataRoot) != "" {
		return filepath.Join(defaultDataRoot, "training_jobs")
	}
	if strings.TrimSpace(projectRoot) != "" {
		return filepath.Join(projectRoot, "backend", "training_jobs")
	}
	return filepath.Join("backend", "training_jobs")
}

func fileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

func normalizeConfiguredPath(value string) string {
	value = strings.TrimSpace(strings.Trim(value, `"'`))
	if value == "" {
		return ""
	}
	if runtime.GOOS != "windows" && len(value) >= 3 && value[1] == ':' && (value[2] == '\\' || value[2] == '/') {
		drive := strings.ToLower(string(value[0]))
		rest := strings.ReplaceAll(value[2:], `\`, `/`)
		return filepath.Clean(filepath.Join("/mnt", drive, rest))
	}
	return filepath.Clean(value)
}

func resolveConfiguredExecutable(projectRoot string, configDir string, explicit string) string {
	if explicit == "" {
		return ""
	}
	if filepath.IsAbs(explicit) {
		return explicit
	}
	if !strings.ContainsAny(explicit, `/\`) {
		return explicit
	}
	return resolveConfiguredPath(projectRoot, configDir, explicit)
}

func resolveConfiguredPath(projectRoot string, configDir string, explicit string) string {
	if explicit == "" {
		return ""
	}
	if filepath.IsAbs(explicit) {
		return explicit
	}

	candidates := []string{}
	if strings.TrimSpace(projectRoot) != "" {
		candidates = append(candidates, filepath.Clean(filepath.Join(projectRoot, explicit)))
	}
	if strings.TrimSpace(configDir) != "" {
		candidates = append(candidates, filepath.Clean(filepath.Join(configDir, explicit)))
	}
	for _, candidate := range candidates {
		if fileExists(candidate) || dirExists(candidate) {
			return candidate
		}
	}
	if len(candidates) > 0 {
		return candidates[0]
	}
	return explicit
}

func dirExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.IsDir()
}
