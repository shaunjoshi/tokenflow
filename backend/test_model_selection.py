#!/usr/bin/env python
"""
Test script for the model selection and classification API.
This script demonstrates using the API endpoints for classifying prompts
and selecting appropriate models from OpenRouter.
"""
import asyncio
import httpx
import json
from pprint import pprint

API_BASE_URL = "http://localhost:8000"  # Change this to your deployed API URL if needed

# Example prompts for different categories
TEST_PROMPTS = {
    "creative": "Write a short sci-fi story about a robot who discovers emotions",
    "factual": "What were the major causes of World War II?",
    "coding": "Create a Python function that sorts a list of dictionaries by a specific key",
    "math": "Solve the differential equation dy/dx = 2x + 3y",
    "reasoning": "If all A are B, and some B are C, what can we conclude about A and C?"
}


async def test_classification_endpoint():
    """Test the /api/classify endpoint with different prompts."""
    print("\n=== Testing Classification Endpoint ===\n")
    
    async with httpx.AsyncClient() as client:
        for category, prompt in TEST_PROMPTS.items():
            print(f"Testing prompt expected to be '{category}':")
            print(f"Prompt: {prompt}\n")
            
            # Make the API request
            response = await client.post(
                f"{API_BASE_URL}/api/classify",
                json={
                    "prompt": prompt,
                    "possible_categories": list(TEST_PROMPTS.keys())
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"Classification Result:")
                pprint(result)
                print("\n" + "-" * 70 + "\n")
            else:
                print(f"Error: {response.status_code} - {response.text}")
                print("\n" + "-" * 70 + "\n")


async def test_model_selection_endpoint():
    """Test the /api/models/select endpoint with a sample prompt."""
    print("\n=== Testing Model Selection Endpoint ===\n")
    
    # Choose one prompt for full model selection test
    prompt = TEST_PROMPTS["creative"]
    print(f"Testing full model selection with prompt: {prompt}\n")
    
    async with httpx.AsyncClient() as client:
        # Make the API request
        response = await client.post(
            f"{API_BASE_URL}/api/models/select",
            json={
                "prompt": prompt,
                "possible_categories": list(TEST_PROMPTS.keys()),
                "temperature": 0.7
            },
            timeout=60.0  # Longer timeout for completion generation
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Model Selected: {result['selected_model']}")
            print(f"Category: {result['prompt_category']} (confidence: {result['confidence_score']})")
            print("\nAll Category Scores:")
            pprint(result['all_categories'])
            print("\nCompletion Result:")
            print(result['completion'])
        else:
            print(f"Error: {response.status_code} - {response.text}")


async def main():
    """Run all tests."""
    print("Starting API tests...")
    
    # Test classification only
    await test_classification_endpoint()
    
    # Test full model selection with completion
    await test_model_selection_endpoint()


if __name__ == "__main__":
    asyncio.run(main()) 