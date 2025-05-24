package main

import (
	"log"
	"tokenflow/pkg/config"
	"tokenflow/pkg/handlers"
	"tokenflow/pkg/routes"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

func main() {
	// Initialize configuration
	config.Init()

	// Set Gin mode
	if config.AppConfig.Environment == "production" {
		gin.SetMode(gin.ReleaseMode)
	}

	// Initialize router
	router := gin.Default()

	// Configure CORS
	router.Use(cors.New(cors.Config{
		AllowOrigins: []string{
			"http://localhost:3000",
			"http://192.168.4.206:3000",
			"http://192.168.5.209:3000", // Add your local network IP
		},
		AllowMethods:     []string{"GET", "POST", "OPTIONS"},
		AllowHeaders:     []string{"Authorization", "Content-Type", "Origin", "Accept"},
		AllowCredentials: true,
	}))

	// Initialize handlers
	modelHandler := handlers.NewModelHandler()
	compressionHandler := handlers.NewCompressionHandler()
	classificationHandler := handlers.NewClassificationHandler()

	// Set up routes
	routes.SetupRoutes(router, modelHandler, classificationHandler, compressionHandler)

	// Start server
	port := config.AppConfig.Port
	log.Printf("Server starting on port :%s", port)
	if err := router.Run(":" + port); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}
