package main

import (
	"log"
	"tokenflow/pkg/config"
	"tokenflow/pkg/handlers"
	"tokenflow/pkg/routes"
	"tokenflow/pkg/services"

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
		AllowMethods:     []string{"GET", "POST", "OPTIONS", "PUT", "DELETE"},
		AllowHeaders:     []string{"Authorization", "Content-Type", "Origin", "Accept"},
		AllowCredentials: true,
	}))

	// Initialize services
	log.Println("Initializing services...")
	modelService := services.NewModelService()
	classificationService := services.NewClassificationService()
	// compressionService is instantiated directly by its handler, which is fine for now
	log.Println("Services initialized.")

	// Initialize handlers, injecting services
	log.Println("Initializing handlers...")
	modelHandler := handlers.NewModelHandler(modelService, classificationService)
	compressionHandler := handlers.NewCompressionHandler()
	classificationHandler := handlers.NewClassificationHandler(classificationService)
	log.Println("Handlers initialized.")

	// Set up routes
	log.Println("Setting up routes...")
	routes.SetupRoutes(router, modelHandler, classificationHandler, compressionHandler)
	log.Println("Routes set up.")

	// Start server
	port := config.AppConfig.Port
	log.Printf("Server starting on port :%s", port)
	if err := router.Run(":" + port); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}
