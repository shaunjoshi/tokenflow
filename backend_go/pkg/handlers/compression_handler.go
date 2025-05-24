package handlers

import (
	"fmt"
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
		fmt.Printf("Error binding JSON: %v\n", err)
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	fmt.Printf("Received compression request: text=%s, ratio=%.2f\n", req.Text, req.Ratio)

	// Compress the text
	result, err := h.compressionService.CompressText(req.Text, req.Ratio)
	if err != nil {
		fmt.Printf("Error compressing text: %v\n", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	fmt.Printf("Compression successful: original_tokens=%d, compressed_tokens=%d\n", result.OriginalTokens, result.CompressedTokens)
	c.JSON(http.StatusOK, result)
}
