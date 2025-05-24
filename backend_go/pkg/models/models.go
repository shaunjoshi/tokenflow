package models

// ModelSelectionRequest represents the request for model selection
type ModelSelectionRequest struct {
	Prompt             string   `json:"prompt" binding:"required"`
	PossibleCategories []string `json:"possible_categories"`
	Temperature        float64  `json:"temperature"`
	TopP               float64  `json:"top_p"`
	MaxTokens          int      `json:"max_tokens"`
}

// ModelSelectionResponse represents the response for model selection
type ModelSelectionResponse struct {
	SelectedModel   string             `json:"selected_model"`
	PromptCategory  string             `json:"prompt_category"`
	ConfidenceScore float64            `json:"confidence_score"`
	AllCategories   map[string]float64 `json:"all_categories"`
	Completion      string             `json:"completion"`
}

// ClassificationRequest represents the request for prompt classification
type ClassificationRequest struct {
	Prompt             string   `json:"prompt" binding:"required"`
	PossibleCategories []string `json:"possible_categories"`
	MultiLabel         bool     `json:"multi_label"`
}

// ClassificationResponse represents the response for prompt classification
type ClassificationResponse struct {
	TopCategory      string             `json:"top_category"`
	ConfidenceScore  float64            `json:"confidence_score"`
	AllCategories    map[string]float64 `json:"all_categories"`
	RecommendedModel string             `json:"recommended_model"`
}

// CompressionRequest represents the request for text compression
type CompressionRequest struct {
	Text  string  `json:"text" binding:"required"`
	Ratio float64 `json:"ratio" binding:"required,gt=0,lte=1"`
}

// CompressionResponse represents the response for text compression
type CompressionResponse struct {
	OriginalText     string  `json:"original_text"`
	CompressedText   string  `json:"compressed_text"`
	OriginalTokens   int     `json:"original_tokens"`
	CompressedTokens int     `json:"compressed_tokens"`
	CompressionRatio float64 `json:"compression_ratio"`
}

// GenerateRequest represents the request for direct generation
type GenerateRequest struct {
	Prompt      string  `json:"prompt" binding:"required"`
	Model       string  `json:"model" binding:"required"`
	Temperature float64 `json:"temperature"`
	TopP        float64 `json:"top_p"`
	MaxTokens   int     `json:"max_tokens"`
}

// ModelRankings defines the available models and their categories
var ModelRankings = map[string]struct {
	Title       string `json:"title"`
	Description string `json:"description"`
	Models      []struct {
		ID      string `json:"id"`
		Name    string `json:"name"`
		Primary bool   `json:"primary"`
	} `json:"models"`
}{
	"reasoning": {
		Title:       "Reasoning",
		Description: "Logic, problem-solving, and analysis",
		Models: []struct {
			ID      string `json:"id"`
			Name    string `json:"name"`
			Primary bool   `json:"primary"`
		}{
			{ID: "llama-3.3-70b-versatile", Name: "Llama 3.3 70B", Primary: true},
			{ID: "gemma2-9b-it", Name: "Gemma 2 9B", Primary: false},
			{ID: "llama-3.1-8b-instant", Name: "Llama 3.1 8B", Primary: false},
		},
	},
	"function-calling": {
		Title:       "Function Calling",
		Description: "API interactions and tool usage",
		Models: []struct {
			ID      string `json:"id"`
			Name    string `json:"name"`
			Primary bool   `json:"primary"`
		}{
			{ID: "llama-3.3-70b-versatile", Name: "Llama 3.3 70B", Primary: true},
			{ID: "llama-3.1-8b-instant", Name: "Llama 3.1 8B", Primary: false},
		},
	},
	"text-to-text": {
		Title:       "Text to Text",
		Description: "General text generation and transformation",
		Models: []struct {
			ID      string `json:"id"`
			Name    string `json:"name"`
			Primary bool   `json:"primary"`
		}{
			{ID: "llama-3.3-70b-versatile", Name: "Llama 3.3 70B", Primary: true},
			{ID: "gemma2-9b-it", Name: "Gemma 2 9B", Primary: false},
			{ID: "llama-3.1-8b-instant", Name: "Llama 3.1 8B", Primary: false},
		},
	},
	"multilingual": {
		Title:       "Multilingual",
		Description: "Cross-language generation and translation",
		Models: []struct {
			ID      string `json:"id"`
			Name    string `json:"name"`
			Primary bool   `json:"primary"`
		}{
			{ID: "llama-3.3-70b-versatile", Name: "Llama 3.3 70B", Primary: true},
			{ID: "llama-3.1-8b-instant", Name: "Llama 3.1 8B", Primary: false},
		},
	},
	"nsfw": {
		Title:       "NSFW Detection",
		Description: "Content moderation and safety evaluation",
		Models: []struct {
			ID      string `json:"id"`
			Name    string `json:"name"`
			Primary bool   `json:"primary"`
		}{
			{ID: "meta-llama/Llama-Guard-4-12B", Name: "Llama Guard 4 12B", Primary: true},
			{ID: "llama-3.3-70b-versatile", Name: "Llama 3.3 70B", Primary: false},
		},
	},
}

var DefaultModelCategories = []string{
	"reasoning",
	"function-calling",
	"text-to-text",
	"multilingual",
	"nsfw",
}
