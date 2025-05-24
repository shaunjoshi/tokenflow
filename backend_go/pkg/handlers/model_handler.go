package handlers

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"

	"tokenflow/pkg/models"
	"tokenflow/pkg/services"

	"github.com/gin-gonic/gin"
)

type ModelHandler struct {
	modelService *services.ModelService
}

func NewModelHandler() *ModelHandler {
	return &ModelHandler{
		modelService: services.NewModelService(),
	}
}

func (h *ModelHandler) StreamModelSelection(c *gin.Context) {
	var req models.ModelSelectionRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Set up SSE headers
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("Transfer-Encoding", "chunked")

	// Create a channel for streaming
	streamChan := make(chan interface{})
	errorChan := make(chan error)

	// Start the model selection process in a goroutine
	go func() {
		defer close(streamChan)
		defer close(errorChan)

		// Classify the prompt
		classification, err := h.modelService.ClassifyPrompt(req.Prompt, req.PossibleCategories)
		if err != nil {
			errorChan <- err
			return
		}

		// Send metadata
		metadata := map[string]interface{}{
			"prompt_category":  classification.TopCategory,
			"confidence_score": classification.ConfidenceScore,
			"selected_model":   classification.RecommendedModel,
			"all_categories":   classification.AllCategories,
		}
		streamChan <- map[string]interface{}{
			"event": "metadata",
			"data":  metadata,
		}

		// Stream the completion
		err = h.modelService.StreamCompletion(
			req.Prompt,
			classification.RecommendedModel,
			req.Temperature,
			req.TopP,
			req.MaxTokens,
			streamChan,
		)
		if err != nil {
			errorChan <- err
			return
		}

		// Send end stream event
		streamChan <- map[string]interface{}{
			"event": "end_stream",
			"data":  "Stream finished",
		}
	}()

	// Stream the results to the client
	c.Stream(func(w io.Writer) bool {
		select {
		case data, ok := <-streamChan:
			if !ok {
				return false
			}
			eventData, _ := json.Marshal(data)
			// Send the event with the correct SSE format
			fmt.Fprintf(w, "data: %s\n\n", eventData)
			w.(http.Flusher).Flush()
			return true
		case err := <-errorChan:
			errorData := map[string]interface{}{
				"event": "error",
				"data": map[string]string{
					"error":  "An error occurred during processing",
					"detail": err.Error(),
				},
			}
			eventData, _ := json.Marshal(errorData)
			fmt.Fprintf(w, "data: %s\n\n", eventData)
			w.(http.Flusher).Flush()
			return false
		}
	})
}

func (h *ModelHandler) StreamDirectGeneration(c *gin.Context) {
	// Set SSE headers
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("Access-Control-Allow-Origin", "*")

	// Create channels for streaming
	done := make(chan bool)
	streamChan := make(chan interface{})
	errorChan := make(chan error)

	// Get request body
	var req models.GenerateRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		log.Printf("Failed to bind JSON: %v", err)
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// First, classify the prompt
	log.Printf("Classifying prompt: %s", req.Prompt)
	classification, err := h.modelService.ClassifyPrompt(req.Prompt, models.DefaultModelCategories)
	if err != nil {
		log.Printf("Failed to classify prompt: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to classify prompt: %v", err)})
		return
	}

	log.Printf("Classification successful - top category: %s, confidence: %f",
		classification.TopCategory,
		classification.ConfidenceScore)

	// Start generation in a goroutine
	go func() {
		defer close(streamChan)
		defer close(errorChan)
		defer close(done)

		// Send metadata event using the classifier output
		metadata := map[string]interface{}{
			"prompt_category":  classification.TopCategory,
			"confidence_score": classification.ConfidenceScore,
			"selected_model":   req.Model,
			"all_categories":   classification.AllCategories,
		}
		streamChan <- map[string]interface{}{
			"event": "metadata",
			"data":  metadata,
		}

		// Stream the completion
		err := h.modelService.StreamCompletion(
			req.Prompt,
			req.Model,
			req.Temperature,
			req.TopP,
			req.MaxTokens,
			streamChan,
		)
		if err != nil {
			log.Printf("Error in StreamCompletion: %v", err)
			errorChan <- err
			return
		}

		// Send end stream event
		streamChan <- map[string]interface{}{
			"event": "end_stream",
			"data":  "Stream finished",
		}
		done <- true
	}()

	// Stream results to client
	c.Stream(func(w io.Writer) bool {
		select {
		case err := <-errorChan:
			log.Printf("Error received from errorChan: %v", err)
			errorData := map[string]interface{}{
				"event": "error",
				"data": map[string]string{
					"error":  "An error occurred during generation",
					"detail": err.Error(),
				},
			}
			eventData, _ := json.Marshal(errorData)
			fmt.Fprintf(w, "data: %s\n\n", eventData)
			w.(http.Flusher).Flush()
			return false
		case data, ok := <-streamChan:
			if !ok {
				return false
			}
			log.Printf("Sending message to client: %v", data)
			eventData, _ := json.Marshal(data)
			fmt.Fprintf(w, "data: %s\n\n", eventData)
			w.(http.Flusher).Flush()
			return true
		case <-done:
			log.Printf("Generation completed")
			return false
		case <-c.Request.Context().Done():
			log.Printf("Client disconnected")
			return false
		}
	})
}

func (h *ModelHandler) GetModelRankings(c *gin.Context) {
	c.JSON(http.StatusOK, models.ModelRankings)
}
