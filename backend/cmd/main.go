package main

import (
	"awesomeProject/api"
	"log"
	"os"

	"github.com/gin-gonic/gin"
)

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	datasetsFile := os.Getenv("DATASETS_FILE")
	if datasetsFile == "" {
		datasetsFile = "data/datasets.json"
	}

	r := gin.Default()
	h, err := api.NewHandler(datasetsFile)
	if err != nil {
		log.Fatalf("failed to initialize dataset storage: %v", err)
	}
	h.RegisterRoutes(r)

	if err := r.Run(":" + port); err != nil {
		log.Fatalf("server failed: %v", err)
	}
}
