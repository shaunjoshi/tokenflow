package main

import (
	"os"
	"testing"
	"tokenflow/pkg/config"
	"tokenflow/pkg/handlers"
	"tokenflow/pkg/services"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestServiceInitialization(t *testing.T) {
	os.Setenv("GROQ_API_KEY", "test-key")
	os.Setenv("GROQ_API_BASE_URL", "https://api.groq.com/openai")
	os.Setenv("OPENROUTER_API_KEY", "test-openrouter-key")
	os.Setenv("ENVIRONMENT", "test")
	os.Setenv("PORT", "8000")
	defer func() {
		os.Unsetenv("GROQ_API_KEY")
		os.Unsetenv("GROQ_API_BASE_URL")
		os.Unsetenv("OPENROUTER_API_KEY")
		os.Unsetenv("ENVIRONMENT")
		os.Unsetenv("PORT")
	}()

	config.Init()

	modelService := services.NewModelService()
	classificationService := services.NewClassificationService()
	compressionService := services.NewCompressionService()

	assert.NotNil(t, modelService, "ModelService should be initialized")
	assert.NotNil(t, classificationService, "ClassificationService should be initialized")
	assert.NotNil(t, compressionService, "CompressionService should be initialized")

	modelHandler := handlers.NewModelHandler(modelService, classificationService)
	compressionHandler := handlers.NewCompressionHandler(compressionService)
	classificationHandler := handlers.NewClassificationHandler(classificationService)

	assert.NotNil(t, modelHandler, "ModelHandler should be initialized")
	assert.NotNil(t, compressionHandler, "CompressionHandler should be initialized")
	assert.NotNil(t, classificationHandler, "ClassificationHandler should be initialized")
}

func TestGinSetup(t *testing.T) {
	gin.SetMode(gin.TestMode)

	router := gin.New()
	require.NotNil(t, router, "Gin router should be initialized")

	router.Use(gin.Logger())
	router.Use(gin.Recovery())

	assert.Equal(t, gin.TestMode, gin.Mode(), "Gin should be in test mode")
}

func TestConfigurationLoading(t *testing.T) {
	os.Setenv("GROQ_API_KEY", "test-groq-key")
	os.Setenv("GROQ_API_BASE_URL", "https://api.groq.com/openai")
	os.Setenv("OPENROUTER_API_KEY", "test-openrouter-key")
	os.Setenv("ENVIRONMENT", "test")
	os.Setenv("PORT", "9999")
	defer func() {
		os.Unsetenv("GROQ_API_KEY")
		os.Unsetenv("GROQ_API_BASE_URL")
		os.Unsetenv("OPENROUTER_API_KEY")
		os.Unsetenv("ENVIRONMENT")
		os.Unsetenv("PORT")
	}()

	config.Init()

	assert.Equal(t, "test", config.AppConfig.Environment, "Environment should be set to 'test'")
	assert.Equal(t, "9999", config.AppConfig.Port, "Port should be set to '9999'")
	assert.Equal(t, "test-groq-key", config.AppConfig.GroqAPIKey, "Groq API key should be loaded")
	assert.Equal(t, "https://api.groq.com/openai", config.AppConfig.GroqBaseURL, "Groq base URL should be loaded")
	assert.Equal(t, "test-openrouter-key", config.AppConfig.OpenRouterAPIKey, "OpenRouter API key should be loaded")
}
