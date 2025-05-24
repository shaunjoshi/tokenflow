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
	modelService          *services.ModelService
	classificationService *services.ClassificationService
}

func NewModelHandler(modelService *services.ModelService, classService *services.ClassificationService) *ModelHandler {
	return &ModelHandler{
		modelService:          modelService,
		classificationService: classService,
	}
}

func (h *ModelHandler) StreamModelSelection(c *gin.Context) {
	var req models.ModelSelectionRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		log.Printf("ModelHandler: Error binding JSON for /models/select: %v", err)
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request payload: " + err.Error()})
		return
	}

	log.Printf("ModelHandler: /models/select request. Prompt (first 50): %s...", req.Prompt[:min(len(req.Prompt), 50)])

	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("Transfer-Encoding", "chunked")

	streamChan := make(chan interface{})
	errorChan := make(chan error)

	go func() {
		defer close(streamChan)
		defer close(errorChan)

		// Classify the prompt using the injected ClassificationService
		categoriesToUse := req.PossibleCategories
		if len(categoriesToUse) == 0 {
			categoriesToUse = models.DefaultModelCategories
		}
		log.Printf("ModelHandler: Classifying prompt for model selection...")
		// Assuming multiLabel is false for this flow, or it could be part of ModelSelectionRequest
		classification, err := h.classificationService.ClassifyPrompt(req.Prompt, categoriesToUse, false)
		if err != nil {
			log.Printf("ModelHandler: Error classifying prompt for model selection: %v", err)
			errorChan <- fmt.Errorf("failed to classify prompt for model selection: %v", err)
			return
		}
		log.Printf("ModelHandler: Classification for model selection successful. Top category: %s", classification.TopCategory)

		// Select model based on category (this method is in ModelService)
		selectedModelID := h.modelService.SelectModelForCategory(classification.TopCategory)
		log.Printf("ModelHandler: Selected model based on category '%s': %s", classification.TopCategory, selectedModelID)

		metadata := map[string]interface{}{
			"prompt_category":  classification.TopCategory,
			"confidence_score": classification.ConfidenceScore,
			"selected_model":   selectedModelID, // Use dynamically selected model
			"all_categories":   classification.AllCategories,
		}
		streamChan <- map[string]interface{}{"event": "metadata", "data": metadata}

		log.Printf("ModelHandler: Streaming completion from selected model: %s", selectedModelID)
		err = h.modelService.StreamCompletion(
			req.Prompt,
			selectedModelID, // Use the model selected based on classification
			req.Temperature,
			req.TopP,
			req.MaxTokens,
			streamChan,
		)
		if err != nil {
			log.Printf("ModelHandler: Error during StreamCompletion for model selection: %v", err)
			errorChan <- fmt.Errorf("streaming completion failed: %v", err)
			return
		}
		streamChan <- map[string]interface{}{"event": "end_stream", "data": "Stream finished"}
	}()

	c.Stream(func(w io.Writer) bool {
		select {
		case data, ok := <-streamChan:
			if !ok {
				return false
			}
			eventData, _ := json.Marshal(data)
			fmt.Fprintf(w, "data: %s\n\n", eventData)
			// if flusher, ok := w.(http.Flusher); ok { flusher.Flush() } // Ensure type assertion is safe
			if flusher, okAssert := w.(http.Flusher); okAssert {
				flusher.Flush()
			}
			return true
		case err := <-errorChan:
			log.Printf("ModelHandler: Error in stream (model selection): %v", err)
			errorData := map[string]interface{}{
				"event": "error",
				"data":  map[string]string{"error": "Error during model selection stream", "detail": err.Error()},
			}
			eventData, _ := json.Marshal(errorData)
			fmt.Fprintf(w, "data: %s\n\n", eventData)
			// if flusher, ok := w.(http.Flusher); ok { flusher.Flush() }
			if flusher, okAssert := w.(http.Flusher); okAssert {
				flusher.Flush()
			}
			return false
		case <-c.Request.Context().Done():
			log.Printf("ModelHandler: Client disconnected during model selection stream.")
			return false
		}
	})
}

func (h *ModelHandler) StreamDirectGeneration(c *gin.Context) {
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("Access-Control-Allow-Origin", "*") // Consider making this more restrictive

	done := make(chan bool)
	streamChan := make(chan interface{})
	errorChan := make(chan error)

	var req models.GenerateRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		log.Printf("ModelHandler: Failed to bind JSON for /generate: %v", err)
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request payload: " + err.Error()})
		return
	}

	log.Printf("ModelHandler: /generate request. Prompt (first 50): %s... Model: %s", req.Prompt[:min(len(req.Prompt), 50)], req.Model)

	go func() {
		defer close(streamChan)
		defer close(errorChan)
		defer close(done)

		// Classify the prompt first to send as metadata
		log.Printf("ModelHandler: Classifying prompt for metadata (direct generation)...")
		// Assuming multiLabel is false for this metadata classification, or it could be part of GenerateRequest
		classification, err := h.classificationService.ClassifyPrompt(req.Prompt, models.DefaultModelCategories, false)
		if err != nil {
			log.Printf("ModelHandler: Failed to classify prompt for metadata: %v. Proceeding without classification metadata.", err)
			// Don't block generation if classification for metadata fails; just log it and proceed.
			// Send a simplified metadata or error event for metadata part.
			metadata := map[string]interface{}{
				"prompt_category":  "unknown",
				"confidence_score": 0.0,
				"selected_model":   req.Model,
				"all_categories":   map[string]float64{},
				"metadata_error":   "classification failed: " + err.Error(),
			}
			streamChan <- map[string]interface{}{"event": "metadata", "data": metadata}
		} else {
			log.Printf("ModelHandler: Classification for metadata successful. Top category: %s", classification.TopCategory)
			metadata := map[string]interface{}{
				"prompt_category":  classification.TopCategory,
				"confidence_score": classification.ConfidenceScore,
				"selected_model":   req.Model, // Model is directly requested by user
				"all_categories":   classification.AllCategories,
			}
			streamChan <- map[string]interface{}{"event": "metadata", "data": metadata}
		}

		log.Printf("ModelHandler: Streaming completion from requested model: %s", req.Model)
		err = h.modelService.StreamCompletion(
			req.Prompt,
			req.Model,
			req.Temperature,
			req.TopP,
			req.MaxTokens,
			streamChan,
		)
		if err != nil {
			log.Printf("ModelHandler: Error in StreamCompletion for direct generation: %v", err)
			errorChan <- err
			return
		}
		streamChan <- map[string]interface{}{"event": "end_stream", "data": "Stream finished"}
		done <- true
	}()

	c.Stream(func(w io.Writer) bool {
		select {
		case err := <-errorChan:
			log.Printf("ModelHandler: Error in stream (direct generation): %v", err)
			detail := "An error occurred during generation"
			if err != nil { // Ensure err is not nil before calling Error()
				detail = err.Error()
			}
			errorData := map[string]interface{}{
				"event": "error",
				"data":  map[string]string{"error": "Generation stream error", "detail": detail},
			}
			eventData, _ := json.Marshal(errorData)
			fmt.Fprintf(w, "data: %s\n\n", eventData)
			// if flusher, ok := w.(http.Flusher); ok { flusher.Flush() }
			if flusher, okAssert := w.(http.Flusher); okAssert {
				flusher.Flush()
			}
			return false
		case data, ok := <-streamChan:
			if !ok {
				return false
			}
			// log.Printf("ModelHandler: Sending message to client (direct generation): %v", data) // Can be too verbose
			eventData, _ := json.Marshal(data)
			fmt.Fprintf(w, "data: %s\n\n", eventData)
			// if flusher, ok := w.(http.Flusher); ok { flusher.Flush() }
			if flusher, okAssert := w.(http.Flusher); okAssert {
				flusher.Flush()
			}
			return true
		case <-done:
			log.Printf("ModelHandler: Generation completed (direct generation).")
			return false
		case <-c.Request.Context().Done():
			log.Printf("ModelHandler: Client disconnected during direct generation stream.")
			return false
		}
	})
}

func (h *ModelHandler) GetModelRankings(c *gin.Context) {
	log.Printf("ModelHandler: Serving /model-rankings request.")
	c.JSON(http.StatusOK, models.ModelRankings)
}

// Helper function min removed as it's duplicative or should be in a utils package
// func min(a, b int) int {
//     if a < b {
//         return a
//     }
//     return b
// }
