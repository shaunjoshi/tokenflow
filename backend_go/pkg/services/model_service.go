package services

import (
	"context"
	"fmt"
	"log"
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

	// Initialize Groq client
	groqConfig := openai.DefaultConfig(config.AppConfig.GroqAPIKey)
	groqConfig.BaseURL = config.AppConfig.GroqBaseURL
	groqClient := openai.NewClientWithConfig(groqConfig)

	log.Printf("ModelService: Initialized with Groq base URL: %s", groqConfig.BaseURL)

	return &ModelService{
		openRouterClient: openRouterClient,
		groqClient:       groqClient,
	}
}

func (s *ModelService) StreamCompletion(
	prompt string,
	model string,
	temperature float64,
	topP float64,
	maxTokens int,
	streamChan chan<- interface{},
) error {
	req := openai.ChatCompletionRequest{
		Model:       model,
		Messages:    []openai.ChatCompletionMessage{{Role: "user", Content: prompt}},
		Temperature: float32(temperature),
		TopP:        float32(topP),
		MaxTokens:   maxTokens,
		Stream:      true,
	}

	stream, err := s.groqClient.CreateChatCompletionStream(context.Background(), req)
	if err != nil {
		return fmt.Errorf("failed to create chat completion stream: %v", err)
	}
	defer stream.Close()

	for {
		response, err := stream.Recv()
		if err != nil {
			if err == context.Canceled || err.Error() == "EOF" {
				break
			}
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
