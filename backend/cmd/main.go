package main

import (
	"awesomeProject/backend/api"
	"log"
	"os"

	"github.com/gin-gonic/gin"
)

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	r := gin.Default()
	h := api.NewHandler()
	h.RegisterRoutes(r)

	if err := r.Run(":" + port); err != nil {
		log.Fatalf("server failed: %v", err)
	}
}
