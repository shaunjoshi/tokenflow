package services

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"tokenflow/pkg/models"
)

type CompressionService struct {
	// No client needed as we're using HTTP directly
}

func NewCompressionService() *CompressionService {
	return &CompressionService{}
}

func (s *CompressionService) CompressText(text string, targetToken int) (*models.CompressionResponse, error) {
	// Create a proper JSON request body
	requestBody := struct {
		Text        string `json:"text"`
		TargetToken int    `json:"target_token"`
	}{
		Text:        text,
		TargetToken: targetToken,
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %v", err)
	}

	fmt.Printf("Sending request to Python service: %s\n", string(jsonData))

	resp, err := http.Post(
		"http://localhost:8001/compress",
		"application/json",
		bytes.NewBuffer(jsonData),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to call compression service: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("compression service error (status %d): %s", resp.StatusCode, string(body))
	}

	var result models.CompressionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode compression response: %v", err)
	}

	return &result, nil
}
