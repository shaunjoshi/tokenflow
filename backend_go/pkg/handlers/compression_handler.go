package handlers

import (
	"net/http"
	"tokenflow/pkg/models"
	"tokenflow/pkg/services"

	"github.com/gin-gonic/gin"
)

type CompressionHandler struct {
	compressionService *services.CompressionService
}

func NewCompressionHandler() *CompressionHandler {
	return &CompressionHandler{
		compressionService: services.NewCompressionService(),
	}
}

func (h *CompressionHandler) CompressText(c *gin.Context) {
	var req models.CompressionRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Compress the text
	result, err := h.compressionService.CompressText(req.Text, req.TargetToken)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, result)
}
