# TokenFlow Python Backend

This is the Python backend service for TokenFlow, providing ML-powered classification and text compression capabilities.

## Features

- Text classification using BART model
- Text compression using LLMLingua
- FastAPI-based REST API
- CORS support for frontend integration

## Requirements

- Python >=3.8,<3.13
- uv (Python package installer)

## Setup

1. Create a virtual environment:
```bash
uv venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
uv pip install -e .
```

3. Run the server:
```bash
python main.py
```

The server will start on http://localhost:8001

## API Endpoints

### Classification
- POST `/classify`
  - Input: `{"prompt": string, "possible_categories": string[], "multi_label": boolean}`
  - Output: `{"top_category": string, "confidence_score": float, "all_categories": object, "recommended_model": string}`

### Compression
- POST `/compress`
  - Input: `{"text": string, "target_token": number}`
  - Output: `{"original_text": string, "compressed_text": string, "original_tokens": number, "compressed_tokens": number, "compression_ratio": number}`

## Development

This project uses:
- FastAPI for the web framework
- BART for text classification
- LLMLingua for text compression
- Ruff for linting and formatting
