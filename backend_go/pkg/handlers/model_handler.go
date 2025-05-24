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

// SSE Protocol: Browser opens persistent connection, server pushes data as it becomes available
func (h *ModelHandler) setupSSEHeaders(c *gin.Context) {
	c.Header("Content-Type", "text/event-stream") // Tells browser: "expect streaming text/event-stream format"
	c.Header("Cache-Control", "no-cache")         // Prevents proxies/browsers from caching stream data
	c.Header("Connection", "keep-alive")          // HTTP/1.1: keep connection open for multiple messages
}

// Browser receives: "data: {json}\n\ndata: {json}\n\n..." in real-time
func (h *ModelHandler) streamResponse(c *gin.Context, streamChan <-chan interface{}, errorChan <-chan error) {
	c.Stream(func(w io.Writer) bool {
		// Go's select statement enables concurrent event handling - key to SSE efficiency
		select {

		// Case 1: New data arrives from LLM (e.g., token "Hello", "world", "!")
		case data, ok := <-streamChan:
			if !ok {
				return false // Channel closed = end of stream
			}

			// SSE Message Format: Each message must be prefixed with "data: "
			// Browser JS receives: addEventListener('message', (event) => console.log(event.data))
			eventData, _ := json.Marshal(data)
			fmt.Fprintf(w, "data: %s\n\n", eventData) // "\n\n" signals end of message

			// Critical: Flush immediately sends bytes to browser (no buffering)
			// Without flush, browser waits for full response = no real-time streaming
			if flusher, ok := w.(http.Flusher); ok {
				flusher.Flush() // Forces TCP packet to browser immediately
			}
			return true // Keep streaming connection alive

		// Case 2: Error occurred (network, API, parsing, etc.)
		case err := <-errorChan:
			if err != nil {
				// Send error as SSE event - browser can handle gracefully
				errorData := map[string]interface{}{
					"event": "error", // Browser can filter: addEventListener('error', handler)
					"data":  map[string]string{"error": "Stream error", "detail": err.Error()},
				}
				eventData, _ := json.Marshal(errorData)
				fmt.Fprintf(w, "data: %s\n\n", eventData)
				if flusher, ok := w.(http.Flusher); ok {
					flusher.Flush()
				}
			}
			return false // Close SSE connection on error

		// Case 3: Browser closed tab/navigated away
		case <-c.Request.Context().Done():
			// Context cancellation prevents goroutine leaks when client disconnects
			return false // Clean shutdown
		}
	})
}

// getCategories returns categories to use for classification
func (h *ModelHandler) getCategories(provided []string) []string {
	if len(provided) == 0 {
		return models.DefaultModelCategories
	}
	return provided
}

// createMetadata creates metadata for streaming
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

func (h *ModelHandler) StreamModelSelection(c *gin.Context) {
	var req models.ModelSelectionRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Step 1: Setup SSE connection - browser now expects streaming data
	h.setupSSEHeaders(c)

	// Step 2: Create Go channels for inter-goroutine communication
	// This enables: Main goroutine (SSE) + Background goroutine (LLM processing)
	streamChan := make(chan interface{}) // Data pipeline: LLM → Browser
	errorChan := make(chan error)        // Error pipeline: Any error → Browser

	// Step 3: Background processing in separate goroutine
	// Why goroutine? Prevents blocking HTTP response while waiting for LLM
	go func() {
		defer close(streamChan) // Signal: no more data coming
		defer close(errorChan)  // Signal: no more errors coming

		// Classify and select model
		categories := h.getCategories(req.PossibleCategories)
		classification, err := h.classificationService.ClassifyPrompt(req.Prompt, categories, false)
		if err != nil {
			errorChan <- fmt.Errorf("classification failed: %v", err)
			return
		}

		selectedModel := h.modelService.SelectModelForCategory(classification.TopCategory)

		// SSE Event 1: Send metadata immediately (user sees which model was selected)
		// Browser receives: data: {"event": "metadata", "data": {"selected_model": "llama-3.3-70b"}}
		metadata := h.createMetadata(classification, selectedModel)
		streamChan <- map[string]interface{}{"event": "metadata", "data": metadata}

		// SSE Event 2-N: Stream LLM tokens as they arrive from Groq API
		// Browser receives: data: {"event": "text_chunk", "data": "Hello"}
		//                  data: {"event": "text_chunk", "data": " world"}
		//                  data: {"event": "text_chunk", "data": "!"}
		err = h.modelService.StreamCompletion(req.Prompt, selectedModel, req.Temperature, req.TopP, req.MaxTokens, streamChan)
		if err != nil {
			errorChan <- err
			return
		}

		// SSE Final Event: Signal completion
		// Browser receives: data: {"event": "end_stream", "data": "Stream finished"}
		streamChan <- map[string]interface{}{"event": "end_stream", "data": "Stream finished"}
	}()

	// Step 4: Handle SSE streaming using Go's concurrent select pattern
	// This simultaneously listens for: data, errors, client disconnection
	h.streamResponse(c, streamChan, errorChan)
}

func (h *ModelHandler) StreamDirectGeneration(c *gin.Context) {
	var req models.GenerateRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// SSE Demo: Same pattern as above but with user-specified model
	h.setupSSEHeaders(c)
	streamChan := make(chan interface{})
	errorChan := make(chan error)

	// Background goroutine handles LLM communication while SSE streams to browser
	go func() {
		defer close(streamChan)
		defer close(errorChan)

		// Optional classification for metadata
		classification, err := h.classificationService.ClassifyPrompt(req.Prompt, models.DefaultModelCategories, false)
		if err != nil {
			classification = nil // Handle gracefully - don't block generation
		}

		// SSE Flow: metadata → token stream → end signal
		metadata := h.createMetadata(classification, req.Model)
		streamChan <- map[string]interface{}{"event": "metadata", "data": metadata}

		// Real-time token streaming: each token sent immediately as generated
		err = h.modelService.StreamCompletion(req.Prompt, req.Model, req.Temperature, req.TopP, req.MaxTokens, streamChan)
		if err != nil {
			errorChan <- err
			return
		}

		streamChan <- map[string]interface{}{"event": "end_stream", "data": "Stream finished"}
	}()

	// Demonstrate SSE handling with concurrent channel operations
	h.streamResponse(c, streamChan, errorChan)
}

func (h *ModelHandler) GetModelRankings(c *gin.Context) {
	c.JSON(http.StatusOK, models.ModelRankings)
}
