package handlers

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"tokenflow/pkg/models"
	"tokenflow/pkg/services"

	"github.com/gin-gonic/gin"
)

type ModelHandler struct {
	modelService          *services.ModelService
	classificationService *services.ClassificationService
}

func NewModelHandler(modelService *services.ModelService, classService *services.ClassificationService) *ModelHandler {
	return &ModelHandler{
		modelService:          modelService,
		classificationService: classService,
	}
}

// This sets up the SSE headers which tells the browser that the backend will be streaming events to the browser
func (h *ModelHandler) setupSSEHeaders(c *gin.Context) {
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive") // Keep the connection open for our goroutine
}

// This function streams the response from the background goroutine to the client using Server-Sent Events (SSE)
func (h *ModelHandler) streamResponse(c *gin.Context, streamChan <-chan interface{}, errorChan <-chan error) {
	c.Stream(func(w io.Writer) bool {

		// This is the core of the streaming - we are listening to the channels and sending the data to the browser
		select {

		// When we get data from our background goroutine, stream it immediately
		case data, ok := <-streamChan:
			if !ok {
				return false
			}

			// Marshal the data into a JSON string
			eventData, _ := json.Marshal(data)
			fmt.Fprintf(w, "data: %s\n\n", eventData)

			// Send the data immediately to the browser
			if flusher, ok := w.(http.Flusher); ok {
				flusher.Flush()
			}
			return true

		// Errors are first class data and also send to the browser
		case err := <-errorChan:
			if err != nil {

				errorData := map[string]interface{}{
					"event": "error",
					"data":  map[string]string{"error": "Stream error", "detail": err.Error()},
				}
				eventData, _ := json.Marshal(errorData)
				fmt.Fprintf(w, "data: %s\n\n", eventData)
				if flusher, ok := w.(http.Flusher); ok {
					flusher.Flush()
				}
			}
			return false // Clean shutdown

		// graceful cleanup when the users closes the browser tab
		case <-c.Request.Context().Done():
			return false
		}
	})
}

// This function returns the categories for the prompt
func (h *ModelHandler) getCategories(provided []string) []string {
	if len(provided) == 0 {
		return models.DefaultModelCategories
	}
	return provided
}

// This function creates the metadata for the prompt
func (h *ModelHandler) createMetadata(classification *models.ClassificationResponse, selectedModel string) map[string]interface{} {
	metadata := map[string]interface{}{
		"selected_model": selectedModel,
	}

	if classification != nil {
		metadata["prompt_category"] = classification.TopCategory
		metadata["confidence_score"] = classification.ConfidenceScore
		metadata["all_categories"] = classification.AllCategories
	} else {
		metadata["prompt_category"] = "unknown"
		metadata["confidence_score"] = 0.0
		metadata["all_categories"] = map[string]float64{}
	}

	return metadata
}

// This function streams the model selection
func (h *ModelHandler) StreamModelSelection(c *gin.Context) {
	var req models.ModelSelectionRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Add safety checks
	if h.classificationService == nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Classification service not initialized"})
		return
	}
	if h.modelService == nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Model service not initialized"})
		return
	}

	// Sets up our streaming headers
	h.setupSSEHeaders(c)

	// Create channels for communication between background goroutine and main HTTP handler goroutine
	streamChan := make(chan interface{})
	errorChan := make(chan error)

	// Background goroutine handles SSE connection, classification, and async streaming
	go func() {
		// Ensures our channels get closed no matter what happens
		defer close(streamChan)
		defer close(errorChan)

		// Get the categories for the prompt
		categories := h.getCategories(req.PossibleCategories)
		classification, err := h.classificationService.ClassifyPrompt(req.Prompt, categories, false)
		if err != nil {
			// Send the error through our channel
			errorChan <- fmt.Errorf("classification failed: %v", err)
			return
		}

		// Add safety check for classification result
		if classification == nil {
			errorChan <- fmt.Errorf("classification returned nil result")
			return
		}

		// Select the model for the category
		selectedModel := h.modelService.SelectModelForCategory(classification.TopCategory)

		// Send metadata first so the user knows which model we picked
		metadata := h.createMetadata(classification, selectedModel)
		streamChan <- map[string]interface{}{"event": "metadata", "data": metadata}

		// Stream LLM response in background to prevent browser timeouts
		// React frontend needs real-time updates via SSE
		err = h.modelService.StreamCompletion(req.Prompt, selectedModel, req.Temperature, req.TopP, req.MaxTokens, streamChan)
		if err != nil {
			errorChan <- err
			return
		}

		// Send completion signal so the frontend knows we're done
		streamChan <- map[string]interface{}{"event": "end_stream", "data": "Stream finished"}
	}()

	// The streamResponse method handles reading from these channels and sending the data to the client via SSE
	h.streamResponse(c, streamChan, errorChan)
}

// This function streams the direct generation of the prompt
func (h *ModelHandler) StreamDirectGeneration(c *gin.Context) {
	var req models.GenerateRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	h.setupSSEHeaders(c)
	streamChan := make(chan interface{})
	errorChan := make(chan error)

	go func() {
		defer close(streamChan)
		defer close(errorChan)

		// Classify the prompt
		classification, err := h.classificationService.ClassifyPrompt(req.Prompt, models.DefaultModelCategories, false)
		if err != nil {
			classification = nil
		}

		// Create the metadata for the prompt
		metadata := h.createMetadata(classification, req.Model)
		streamChan <- map[string]interface{}{"event": "metadata", "data": metadata}

		// Now stream the actual LLM response from the model service call
		err = h.modelService.StreamCompletion(req.Prompt, req.Model, req.Temperature, req.TopP, req.MaxTokens, streamChan)
		if err != nil {
			errorChan <- err
			return
		}

		streamChan <- map[string]interface{}{"event": "end_stream", "data": "Stream finished"}
	}()

	h.streamResponse(c, streamChan, errorChan)
}

func (h *ModelHandler) GetModelRankings(c *gin.Context) {
	c.JSON(http.StatusOK, models.ModelRankings)
}
