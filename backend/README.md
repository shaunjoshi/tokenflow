# Intelligent Model Selection API

This API selects the most appropriate model from OpenRouter.ai based on prompt classification performed by Facebook's BART model.

## Features

- **Prompt Classification**: Uses Facebook BART for zero-shot classification of user prompts
- **Smart Model Selection**: Automatically selects the appropriate model from OpenRouter based on the prompt category
- **Completion Generation**: Processes the prompt with the selected model and returns the result
- **Classification-Only Mode**: Option to classify a prompt without generating a completion

## API Endpoints

### 1. Classify Prompt

```
POST /api/classify
```

Classifies a prompt using Facebook BART and returns the classification results along with the recommended model.

**Request Body:**
```json
{
  "prompt": "Write a story about a robot",
  "possible_categories": ["creative", "factual", "coding", "math", "reasoning"],
  "multi_label": false
}
```

**Response:**
```json
{
  "top_category": "creative",
  "confidence_score": 0.94,
  "all_categories": {
    "creative": 0.94,
    "factual": 0.03,
    "coding": 0.02,
    "math": 0.01,
    "reasoning": 0.0
  },
  "recommended_model": "anthropic/claude-3-opus:beta"
}
```

### 2. Select Model and Generate Completion

```
POST /api/models/select
```

Classifies a prompt, selects the appropriate model, and generates a completion.

**Request Body:**
```json
{
  "prompt": "Write a story about a robot",
  "possible_categories": ["creative", "factual", "coding", "math", "reasoning"],
  "temperature": 0.7,
  "top_p": 0.9
}
```

**Response:**
```json
{
  "selected_model": "anthropic/claude-3-opus:beta",
  "prompt_category": "creative",
  "confidence_score": 0.94,
  "all_categories": {
    "creative": 0.94,
    "factual": 0.03,
    "coding": 0.02,
    "math": 0.01,
    "reasoning": 0.0
  },
  "completion": "In the gleaming metropolis of Neo-Tokyo..."
}
```

## Model Selection Logic

The API selects models based on the following category mapping:

- **creative**: anthropic/claude-3-opus:beta (Best for creative tasks)
- **factual**: google/gemini-1.0-pro (Good for fact-based tasks)
- **coding**: openai/gpt-4-turbo (Strong for coding)
- **math**: anthropic/claude-3-opus:beta (Good for mathematical reasoning)
- **reasoning**: anthropic/claude-3-sonnet:beta (Good for general reasoning)
- **default fallback**: mistralai/mistral-7b

## Setup

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Set up environment variables:
```
OPENROUTER_API_KEY=your_api_key
OPENROUTER_API_BASE_URL=https://openrouter.ai/api/v1
```

3. Run the server:
```
uvicorn main:app --reload
```

4. Test the API:
```
python test_model_selection.py
```

## Requirements

- Python 3.9+
- See requirements.txt for all package dependencies 