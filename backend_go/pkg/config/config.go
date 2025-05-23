package config

import (
	"log"
	"os"

	"github.com/joho/godotenv"
)

type Config struct {
	Port                   string
	Environment            string
	SupabaseURL            string
	SupabaseServiceRoleKey string
	OpenRouterAPIKey       string
	OpenRouterBaseURL      string
	GroqAPIKey             string
	GroqBaseURL            string
}

var AppConfig Config

func Init() {
	err := godotenv.Load()
	if err != nil {
		log.Printf("Warning: .env file not found: %v", err)
	}

	AppConfig = Config{
		Port:                   getEnv("PORT", "8000"),
		Environment:            getEnv("ENVIRONMENT", "development"),
		SupabaseURL:            getEnv("SUPABASE_URL", ""),
		SupabaseServiceRoleKey: getEnv("SUPABASE_SERVICE_ROLE_KEY", ""),
		OpenRouterAPIKey:       getEnv("OPENROUTER_API_KEY", ""),
		OpenRouterBaseURL:      getEnv("OPENROUTER_API_BASE_URL", "https://openrouter.ai/api/v1"),
		GroqAPIKey:             getEnv("GROQ_API_KEY", ""),
		GroqBaseURL:            getEnv("GROQ_API_BASE_URL", "https://api.groq.com/openai/v1"),
	}

	// Log configuration (excluding sensitive values)
	log.Printf("Configuration loaded:")
	log.Printf("Port: %s", AppConfig.Port)
	log.Printf("Environment: %s", AppConfig.Environment)
	log.Printf("OpenRouter Base URL: %s", AppConfig.OpenRouterBaseURL)
	log.Printf("Groq Base URL: %s", AppConfig.GroqBaseURL)

	// Verify API keys are set
	if AppConfig.GroqAPIKey == "" {
		log.Fatal("GROQ_API_KEY is not set")
	}
	if AppConfig.OpenRouterAPIKey == "" {
		log.Fatal("OPENROUTER_API_KEY is not set")
	}
}

func getEnv(key, defaultValue string) string {
	value := os.Getenv(key)
	if value == "" {
		return defaultValue
	}
	return value
}
