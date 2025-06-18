package routes

import (
	"tokenflow/pkg/handlers"

	"github.com/gin-gonic/gin"
)

func SetupRoutes(
	router *gin.Engine,
	modelHandler *handlers.ModelHandler,
	classificationHandler *handlers.ClassificationHandler,
	compressionHandler *handlers.CompressionHandler,
) {
	// API routes
	api := router.Group("/api")
	{
		// Model selection and generation
		api.POST("/models/select", modelHandler.StreamModelSelection)
		api.POST("/generate", modelHandler.StreamDirectGeneration)
		api.GET("/model-rankings", modelHandler.GetModelRankings)

		// Classification
		api.POST("/classify", classificationHandler.ClassifyPrompt)

		// Compression
		api.POST("/compress", compressionHandler.CompressText)
	}
}
