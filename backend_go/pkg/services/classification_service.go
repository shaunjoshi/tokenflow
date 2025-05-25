package services

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"tokenflow/pkg/config"
	"tokenflow/pkg/models"
)

type ClassificationService struct{}

func NewClassificationService() *ClassificationService {
	return &ClassificationService{}
}

// ClassifyPrompt sends a prompt to the Python classification service
func (s *ClassificationService) ClassifyPrompt(prompt string, categories []string, multiLabel bool) (*models.ClassificationResponse, error) {
	requestBody := map[string]interface{}{
		"prompt":              prompt,
		"possible_categories": categories,
		"multi_label":         multiLabel,
	}

	jsonBody, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request body: %v", err)
	}

	resp, err := http.Post(
		config.AppConfig.PythonServiceURL+"/classify",
		"application/json",
		bytes.NewBuffer(jsonBody),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to call classification service: %v", err)
	}
	defer resp.Body.Close()

	responseBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %v", err)
	}

	if resp.StatusCode != http.StatusOK {
		// Try to parse detailed error from Python service
		var pyError struct {
			Detail string `json:"detail"`
		}
		if json.Unmarshal(responseBody, &pyError) == nil && pyError.Detail != "" {
			return nil, fmt.Errorf("classification service error (%d): %s", resp.StatusCode, pyError.Detail)
		}
		return nil, fmt.Errorf("classification service error (%d): %s", resp.StatusCode, string(responseBody))
	}

	var result models.ClassificationResponse
	if err := json.Unmarshal(responseBody, &result); err != nil {
		return nil, fmt.Errorf("failed to decode classification response: %v", err)
	}

	return &result, nil
}
