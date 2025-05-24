package services

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"tokenflow/pkg/config"
	"tokenflow/pkg/models"

	"github.com/sashabaranov/go-openai"
)

type ModelService struct {
	openRouterClient *openai.Client
	groqClient       *openai.Client
}

func NewModelService() *ModelService {
	openRouterClient := openai.NewClient(config.AppConfig.OpenRouterAPIKey)

	// Initialize Groq client with the correct configuration
	groqConfig := openai.DefaultConfig(config.AppConfig.GroqAPIKey)
	groqConfig.BaseURL = "https://api.groq.com/openai/v1"
	log.Printf("Initializing Groq client with base URL: %s", groqConfig.BaseURL)
	groqClient := openai.NewClientWithConfig(groqConfig)

	return &ModelService{
		openRouterClient: openRouterClient,
		groqClient:       groqClient,
	}
}

func (s *ModelService) ClassifyPrompt(prompt string, categories []string) (*models.ClassificationResponse, error) {
	// Create request body
	requestBody := map[string]interface{}{
		"prompt":              prompt,
		"possible_categories": categories,
		"multi_label":         false,
	}

	jsonBody, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request body: %v", err)
	}

	// Call Python BART service
	resp, err := http.Post(
		"http://localhost:8001/classify",
		"application/json",
		bytes.NewBuffer(jsonBody),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to call classification service: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("classification service error: %s", string(body))
	}

	var result models.ClassificationResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode classification response: %v", err)
	}

	return &result, nil
}

func (s *ModelService) StreamCompletion(
	prompt string,
	model string,
	temperature float64,
	topP float64,
	maxTokens int,
	streamChan chan<- interface{},
) error {
	// Use Groq client for streaming
	req := openai.ChatCompletionRequest{
		Model:       model,
		Messages:    []openai.ChatCompletionMessage{{Role: "user", Content: prompt}},
		Temperature: float32(temperature),
		TopP:        float32(topP),
		MaxTokens:   maxTokens,
		Stream:      true,
	}

	log.Printf("Sending request to Groq API with model: %s", model)
	stream, err := s.groqClient.CreateChatCompletionStream(context.Background(), req)
	if err != nil {
		log.Printf("Error creating chat completion stream: %v", err)
		return fmt.Errorf("failed to create chat completion stream: %v", err)
	}
	defer stream.Close()

	for {
		response, err := stream.Recv()
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			log.Printf("Error receiving stream: %v", err)
			return fmt.Errorf("error receiving stream: %v", err)
		}

		if len(response.Choices) > 0 && response.Choices[0].Delta.Content != "" {
			streamChan <- map[string]interface{}{
				"event": "text_chunk",
				"data":  response.Choices[0].Delta.Content,
			}
		}
	}

	return nil
}

func (s *ModelService) SelectModelForCategory(category string) string {
	if ranking, ok := models.ModelRankings[category]; ok {
		for _, model := range ranking.Models {
			if model.Primary {
				return model.ID
			}
		}
		if len(ranking.Models) > 0 {
			return ranking.Models[0].ID
		}
	}
	return "llama-3.1-8b-instant" // Default fallback
}
