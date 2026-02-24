package api

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

type DataSet struct {
	Id    string `json:"id"`
	Scene string `json:"scene"`
}

type DataSets = map[string]*DataSet

type ScenePayload struct {
	Ego SceneEgo `json:"ego"`
}

type SceneEgo struct {
	Waypoints []SceneWaypoint `json:"waypoints"`
}

type SceneWaypoint struct {
	Destination []float64 `json:"destination"`
}

func (d *DataSet) Validate() error {
	if d == nil {
		return errors.New("dataset is required")
	}
	if strings.TrimSpace(d.Id) == "" {
		return errors.New("id is required")
	}
	if strings.TrimSpace(d.Scene) == "" {
		return errors.New("scene is required")
	}
	if err := ValidateScene(d.Scene); err != nil {
		return err
	}
	return nil
}

func ValidateScene(scene string) error {
	if strings.TrimSpace(scene) == "" {
		return errors.New("scene is required")
	}
	if !json.Valid([]byte(scene)) {
		return errors.New("scene must be a valid JSON string")
	}

	var payload ScenePayload
	if err := json.Unmarshal([]byte(scene), &payload); err != nil {
		return errors.New("scene must be valid JSON object")
	}

	if payload.Ego.Waypoints == nil {
		return errors.New("scene.ego.waypoints is required")
	}
	for i, waypoint := range payload.Ego.Waypoints {
		if len(waypoint.Destination) != 3 {
			return fmt.Errorf("scene.ego.waypoints[%d].destination must have exactly 3 numbers", i)
		}
	}

	return nil
}

func (d *DataSet) Clone() *DataSet {
	if d == nil {
		return nil
	}
	return &DataSet{
		Id:    d.Id,
		Scene: d.Scene,
	}
}
