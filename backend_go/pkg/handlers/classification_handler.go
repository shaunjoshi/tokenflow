package handlers

import (
	"fmt"
	"log"
	"net/http"
	"tokenflow/pkg/models"
	"tokenflow/pkg/services"

	"github.com/gin-gonic/gin"
)

type ClassificationHandler struct {
	classificationService *services.ClassificationService
}

func NewClassificationHandler(classificationService *services.ClassificationService) *ClassificationHandler {
	return &ClassificationHandler{
		classificationService: classificationService,
	}
}

func (h *ClassificationHandler) ClassifyPrompt(c *gin.Context) {
	var req models.ClassificationRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		log.Printf("ClassificationHandler: Error binding JSON for /classify: %v", err)
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request payload: " + err.Error()})
		return
	}

	if len(req.PossibleCategories) == 0 {
		log.Printf("ClassificationHandler: No categories provided, using default categories.")
		req.PossibleCategories = models.DefaultModelCategories
	}

	log.Printf("ClassificationHandler: Received classification request. Prompt (first 50): %s... Categories: %v", req.Prompt[:min(len(req.Prompt), 50)], req.PossibleCategories)

	classification, err := h.classificationService.ClassifyPrompt(req.Prompt, req.PossibleCategories, req.MultiLabel)
	if err != nil {
		log.Printf("ClassificationHandler: Error from ClassificationService: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to classify prompt: %v", err)})
		return
	}

	log.Printf("ClassificationHandler: Classification successful. Top category: %s", classification.TopCategory)
	c.JSON(http.StatusOK, classification)
}

// Helper function to prevent panic with string slicing (can be moved to a utils package)
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
