package api

import (
	"net/http"
	"sync"

	"github.com/gin-gonic/gin"
)

type Handler struct {
	mu   sync.RWMutex
	data DataSets
}

type upsertDataSetRequest struct {
	Scene string `json:"scene"`
}

func NewHandler() *Handler {
	return &Handler{
		data: make(DataSets),
	}
}

func (h *Handler) RegisterRoutes(r *gin.Engine) {
	r.GET("/health", h.health)
	r.GET("/datasets", h.listDataSets)
	r.GET("/datasets/:id", h.getDataSet)
	r.POST("/datasets", h.createDataSet)
	r.PUT("/datasets/:id", h.replaceDataSet)
	r.PATCH("/datasets/:id", h.updateDataSetScene)
	r.DELETE("/datasets/:id", h.deleteDataSet)
}

func (h *Handler) health(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"status": "ok"})
}

func (h *Handler) listDataSets(c *gin.Context) {
	h.mu.RLock()
	defer h.mu.RUnlock()

	items := make([]*DataSet, 0, len(h.data))
	for _, ds := range h.data {
		items = append(items, ds.Clone())
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
	if err := validateScene(req.Scene); err != nil {
		c.JSON(http.StatusBadRequest, errorResponse(err.Error()))
		return
	}

	h.mu.Lock()
	ds, ok := h.data[id]
	if ok {
		ds.Scene = req.Scene
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
	}
	h.mu.Unlock()

	if !ok {
		c.JSON(http.StatusNotFound, errorResponse("dataset not found"))
		return
	}

	c.Status(http.StatusNoContent)
}

func errorResponse(message string) gin.H {
	return gin.H{"error": message}
}

func validateScene(scene string) error {
	return ValidateScene(scene)
}
