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
	PythonServiceURL       string
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
		PythonServiceURL:       getEnv("PYTHON_SERVICE_URL", "http://localhost:8001"),
	}

	// Only log essential startup info
	log.Printf("TokenFlow backend starting on port %s (%s environment)", AppConfig.Port, AppConfig.Environment)

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
