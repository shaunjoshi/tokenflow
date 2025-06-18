package handlers

import (
	"log"
	"net/http"
	"tokenflow/pkg/models"
	"tokenflow/pkg/services"

	"github.com/gin-gonic/gin"
)

type CompressionHandler struct {
	compressionService *services.CompressionService
}

func NewCompressionHandler(compressionService *services.CompressionService) *CompressionHandler {
	return &CompressionHandler{
		compressionService: compressionService,
	}
}

func (h *CompressionHandler) CompressText(c *gin.Context) {
	var req models.CompressionRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		log.Printf("ERROR: CompressionHandler: Invalid JSON: %v", err)
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Compress the text
	result, err := h.compressionService.CompressText(req.Text, req.Ratio)
	if err != nil {
		log.Printf("ERROR: CompressionHandler: Service error: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, result)
}
