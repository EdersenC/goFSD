package api

import (
	"encoding/json"
	"errors"
	"strings"
)

type DataSet struct {
	Id    string `json:"id"`
	Scene string `json:"scene"`
}

type DataSets = map[string]*DataSet

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
