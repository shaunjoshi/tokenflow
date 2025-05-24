package main

import (
	"fmt"
	"log"
	"tokenflow/pkg/services"
)

func main() {
	// Initialize services
	modelService := services.NewModelService()
	compressionService := services.NewCompressionService()

	// Test classification
	fmt.Println("\nTesting Classification:")
	prompt := "Write a function to calculate the fibonacci sequence"
	categories := []string{"reasoning", "function-calling", "text-to-text"}

	classification, err := modelService.ClassifyPrompt(prompt, categories)
	if err != nil {
		log.Fatalf("Classification error: %v", err)
	}

	fmt.Printf("Top Category: %s\n", classification.TopCategory)
	fmt.Printf("Confidence Score: %.2f\n", classification.ConfidenceScore)
	fmt.Printf("All Categories: %v\n", classification.AllCategories)
	fmt.Printf("Recommended Model: %s\n", classification.RecommendedModel)

	// Test compression
	fmt.Println("\nTesting Compression:")
	text := "This is a test text that needs to be compressed. It contains multiple sentences and should be reduced to a smaller size while maintaining the main meaning."
	targetTokens := 10

	compression, err := compressionService.CompressText(text, targetTokens)
	if err != nil {
		log.Fatalf("Compression error: %v", err)
	}

	fmt.Printf("Original Text: %s\n", compression.OriginalText)
	fmt.Printf("Compressed Text: %s\n", compression.CompressedText)
	fmt.Printf("Original Tokens: %d\n", compression.OriginalTokens)
	fmt.Printf("Compressed Tokens: %d\n", compression.CompressedTokens)
	fmt.Printf("Compression Ratio: %.2f\n", compression.CompressionRatio)
}
