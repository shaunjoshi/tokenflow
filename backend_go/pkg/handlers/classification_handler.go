package handlers

import (
	"net/http"
	"tokenflow/pkg/models"
	"tokenflow/pkg/services"

	"github.com/gin-gonic/gin"
)

type ClassificationHandler struct {
	modelService *services.ModelService
}

func NewClassificationHandler() *ClassificationHandler {
	return &ClassificationHandler{
		modelService: services.NewModelService(),
	}
}

func (h *ClassificationHandler) ClassifyPrompt(c *gin.Context) {
	var req models.ClassificationRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Use default categories if none provided
	if len(req.PossibleCategories) == 0 {
		req.PossibleCategories = models.DefaultModelCategories
	}

	// Classify the prompt
	classification, err := h.modelService.ClassifyPrompt(req.Prompt, req.PossibleCategories)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, classification)
}
