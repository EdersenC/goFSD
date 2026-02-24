package api

import (
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
)

const defaultSyncInterval = 5 * time.Second

type Handler struct {
	mu           sync.RWMutex
	data         DataSets
	storagePath  string
	dirty        bool
	revision     uint64
	syncInterval time.Duration
}

type upsertDataSetRequest struct {
	Scene string `json:"scene"`
}

type persistedDataSets struct {
	Items []*DataSet `json:"items"`
}

func NewHandler(storagePath string) (*Handler, error) {
	if strings.TrimSpace(storagePath) == "" {
		return nil, errors.New("storage path is required")
	}

	h := &Handler{
		data:         make(DataSets),
		storagePath:  storagePath,
		syncInterval: defaultSyncInterval,
	}
	if err := h.loadFromDisk(); err != nil {
		return nil, err
	}

	go h.syncLoop()
	return h, nil
}

func (h *Handler) RegisterRoutes(r *gin.Engine) {
	r.GET("/health", h.health)
	r.GET("/datasets", h.listDataSets)
	r.GET("/datasets/:id", h.getDataSet)
	r.GET("/datasets/:id/scene", h.getDataSetScenes)
	r.POST("/datasets", h.createDataSet)
	r.PUT("/datasets/:id", h.replaceDataSet)
	r.PATCH("/datasets/:id", h.updateDataSetScene)
	r.DELETE("/datasets/:id", h.deleteDataSet)
	r.POST("/datasets/sync", h.forceSync)
}

func (h *Handler) health(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"status": "ok"})
}

func (h *Handler) listDataSets(c *gin.Context) {
	h.mu.RLock()
	defer h.mu.RUnlock()

	items := make([]*DataSet, 0, len(h.data))
	ids := make([]string, 0, len(h.data))
	for id := range h.data {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		items = append(items, h.data[id].Clone())
	}

	c.JSON(http.StatusOK, gin.H{"items": items})
}

func (h *Handler) getDataSet(c *gin.Context) {
	id := c.Param("id")

	h.mu.RLock()
	ds, ok := h.data[id]
	h.mu.RUnlock()
	if !ok {
		c.JSON(http.StatusNotFound, errorResponse("dataset not found"))
		return
	}

	c.JSON(http.StatusOK, ds.Clone())
}

func (h *Handler) getDataSetScenes(c *gin.Context) {
	id := c.Param("id")

	h.mu.RLock()
	ds, ok := h.data[id]
	h.mu.RUnlock()
	if !ok {
		c.JSON(http.StatusNotFound, errorResponse("dataset not found"))
		return
	}

	c.JSON(http.StatusOK, gin.H{"scene": ds.Scene})
}

func (h *Handler) createDataSet(c *gin.Context) {
	var req DataSet
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, errorResponse("invalid request body"))
		return
	}
	if err := req.Validate(); err != nil {
		c.JSON(http.StatusBadRequest, errorResponse(err.Error()))
		return
	}

	h.mu.Lock()
	defer h.mu.Unlock()
	if _, exists := h.data[req.Id]; exists {
		c.JSON(http.StatusConflict, errorResponse("dataset already exists"))
		return
	}

	h.data[req.Id] = req.Clone()
	h.markDirtyLocked()
	c.JSON(http.StatusCreated, req)
}

func (h *Handler) replaceDataSet(c *gin.Context) {
	id := c.Param("id")
	var req upsertDataSetRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, errorResponse("invalid request body"))
		return
	}

	ds := &DataSet{Id: id, Scene: req.Scene}
	if err := ds.Validate(); err != nil {
		c.JSON(http.StatusBadRequest, errorResponse(err.Error()))
		return
	}

	h.mu.Lock()
	h.data[id] = ds.Clone()
	h.markDirtyLocked()
	h.mu.Unlock()

	c.JSON(http.StatusOK, ds)
}

func (h *Handler) updateDataSetScene(c *gin.Context) {
	id := c.Param("id")
	var req upsertDataSetRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, errorResponse("invalid request body"))
		return
	}
	if err := ValidateScene(req.Scene); err != nil {
		c.JSON(http.StatusBadRequest, errorResponse(err.Error()))
		return
	}

	h.mu.Lock()
	ds, ok := h.data[id]
	if ok {
		ds.Scene = req.Scene
		h.markDirtyLocked()
	}
	h.mu.Unlock()

	if !ok {
		c.JSON(http.StatusNotFound, errorResponse("dataset not found"))
		return
	}

	c.JSON(http.StatusOK, ds.Clone())
}

func (h *Handler) deleteDataSet(c *gin.Context) {
	id := c.Param("id")

	h.mu.Lock()
	_, ok := h.data[id]
	if ok {
		delete(h.data, id)
		h.markDirtyLocked()
	}
	h.mu.Unlock()

	if !ok {
		c.JSON(http.StatusNotFound, errorResponse("dataset not found"))
		return
	}

	c.Status(http.StatusNoContent)
}

func (h *Handler) forceSync(c *gin.Context) {
	if err := h.flushToDisk(); err != nil {
		c.JSON(http.StatusInternalServerError, errorResponse("sync failed"))
		return
	}
	c.JSON(http.StatusOK, gin.H{"status": "synced"})
}

func (h *Handler) syncLoop() {
	ticker := time.NewTicker(h.syncInterval)
	defer ticker.Stop()

	for range ticker.C {
		_ = h.flushToDisk()
	}
}

func (h *Handler) flushToDisk() error {
	h.mu.Lock()
	if !h.dirty {
		h.mu.Unlock()
		return nil
	}
	snapshotRevision := h.revision

	items := make([]*DataSet, 0, len(h.data))
	ids := make([]string, 0, len(h.data))
	for id := range h.data {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		items = append(items, h.data[id].Clone())
	}
	h.mu.Unlock()

	payload := persistedDataSets{Items: items}
	b, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')

	dir := filepath.Dir(h.storagePath)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(h.storagePath, b, 0o644); err != nil {
		return err
	}

	h.mu.Lock()
	if h.revision == snapshotRevision {
		h.dirty = false
	}
	h.mu.Unlock()
	return nil
}

func (h *Handler) loadFromDisk() error {
	b, err := os.ReadFile(h.storagePath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return err
	}
	if len(strings.TrimSpace(string(b))) == 0 {
		return nil
	}

	var persisted persistedDataSets
	if err := json.Unmarshal(b, &persisted); err != nil {
		return err
	}

	loaded := make(DataSets, len(persisted.Items))
	for _, ds := range persisted.Items {
		if err := ds.Validate(); err != nil {
			return err
		}
		if _, exists := loaded[ds.Id]; exists {
			return errors.New("duplicate dataset id in storage: " + ds.Id)
		}
		loaded[ds.Id] = ds.Clone()
	}

	h.mu.Lock()
	h.data = loaded
	h.dirty = false
	h.revision = 0
	h.mu.Unlock()
	return nil
}

func (h *Handler) markDirtyLocked() {
	h.dirty = true
	h.revision++
}

func errorResponse(message string) gin.H {
	return gin.H{"error": message}
}
