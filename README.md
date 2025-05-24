# TokenFlow

A hybrid Go + Python backend system for intelligent prompt compression and LLM generation, designed to demonstrate modern microservice architecture patterns and efficient ML model serving.

## Architecture Overview

TokenFlow uses a **multilanguage microservice architecture** that leverages the strengths of both Go and Python:

- **Go Backend**: High-performance HTTP server acting as the main API gateway and orchestrator
- **Python Backend**: Specialized ML model server for prompt compression and classification
- **External LLM APIs**: Integration with Groq for powerful text generation

## System Architecture

![TokenFlow Architecture](TokenFlow.png)

## Key Design Patterns

### Service Separation
- **ClassificationService**: Dedicated to prompt classification using Python's Hugging Face Transformers
- **CompressionService**: Handles text compression via Python's LLMLingua library  
- **ModelService**: Manages direct LLM generation and model selection logic

### Constructor Pattern (Idiomatic Go)
Each service follows Go's standard constructor pattern:
```go
type ModelService struct { /* fields */ }           // Struct definition
func NewModelService() *ModelService { /* init */ }  // Constructor function
```

### Dependency Injection
Services are initialized once and injected into handlers:
```go
modelService := services.NewModelService()
classificationService := services.NewClassificationService()
modelHandler := handlers.NewModelHandler(modelService, classificationService)
```

## Technology Stack

### Go Backend
- **Framework**: Gin (high-performance HTTP router)
- **HTTP Client**: Standard `net/http` for Python service calls
- **Streaming**: Server-Sent Events (SSE) for real-time LLM output
- **Configuration**: Environment variables with `godotenv`

### Python Backend  
- **Framework**: FastAPI (async Python web framework)
- **Package Management**: uv (fast Python package installer and resolver)
- **Environment Management**: Hermit (reproducible development environments)
- **ML Libraries**: 
  - LLMLingua (prompt compression)
  - Hugging Face Transformers (classification)
- **Models**: 
  - BART-MNLI for zero-shot classification
  - BERT multilingual for compression

### External APIs
- **Groq**: High-performance LLM inference
- **OpenRouter**: Alternative LLM provider (configured but not actively used)

## Data Flow Examples

### Compression Flow
1. Frontend → `POST /api/compress` → Go CompressionHandler
2. Go CompressionService → `POST http://localhost:8001/compress` → Python FastAPI
3. Python LLMLingua processing → JSON response → Go
4. Go → Frontend (compressed text + metrics)

### Classification Flow  
1. Frontend → `POST /api/classify` → Go ClassificationHandler
2. Go ClassificationService → `POST http://localhost:8001/classify` → Python FastAPI
3. Python Transformers pipeline → JSON response → Go
4. Go → Frontend (categories + confidence scores)

### Generation Flow
1. Frontend → `POST /api/generate` → Go ModelHandler
2. Go ModelService → Groq API (streaming)
3. Groq → Go (token-by-token via SSE) → Frontend

## Project Structure

```
tokenflow/
├── frontend/                 # React/Next.js UI
├── backend_go/              # Go API Gateway
│   ├── cmd/server/          # Application entry point
│   ├── pkg/
│   │   ├── handlers/        # HTTP request handlers
│   │   ├── services/        # Business logic services
│   │   ├── models/          # Data structures
│   │   └── config/          # Configuration management
│   └── go.mod
├── backend_python/          # Python ML Server
│   └── src/tokenflow_python/
│       └── main.py          # FastAPI application
└── README.md               # This file
```

## Getting Started

### Prerequisites
- **Go 1.24+**
- **Python 3.9+** (managed via Hermit)
- **uv** (Python package manager - installed via Hermit)
- **Hermit** (for reproducible development environment)
- Environment variables configured in `.env`

### Environment Setup
1. **Install Hermit** (if not already installed):
   ```bash
   curl -fsSL https://github.com/cashapp/hermit/releases/latest/download/install.sh | bash
   ```

2. **Activate Hermit environment** (from project root):
   ```bash
   . bin/activate-hermit
   ```
   This automatically installs the correct Python version and uv.

> **Why Hermit + uv?**
> - **Hermit** ensures reproducible development environments across machines
> - **uv** provides fast Python package installation and dependency resolution
> - Together they eliminate "works on my machine" issues and speed up development

### Run the System
1. **Python Backend** (Terminal 1):
   ```bash
   cd backend_python
   uv run --active python -m src.tokenflow_python.main
   # Runs on http://localhost:8001
   ```

2. **Go Backend** (Terminal 2):
   ```bash
   cd backend_go
   go run cmd/server/main.go
   # Runs on http://localhost:8000
   ```

3. **Frontend** (Terminal 3):
   ```bash
   cd frontend
   npm run dev
   # Runs on http://localhost:3000
   ```

## API Documentation

### Core Endpoints
- `POST /api/compress` - Compress text using LLMLingua (ratio-based)
- `POST /api/classify` - Classify prompts into categories
- `POST /api/generate` - Stream LLM generation from specified model
- `POST /api/models/select` - Auto-select and stream from best model
- `GET /api/model-rankings` - Available model information

### Request/Response Examples

**Compression Request:**
```json
{
  "text": "Long text to be compressed...",
  "ratio": 0.5
}
```

**Classification Request:**
```json
{
  "prompt": "Write a function to calculate fibonacci",
  "possible_categories": ["reasoning", "function-calling", "text-to-text"],
  "multi_label": false
}
```

## Architecture Benefits

1. **Performance**: Go handles concurrent HTTP requests efficiently
2. **Specialization**: Python excels at ML model serving 
3. **Modularity**: Services can be developed, deployed, and scaled independently
4. **Type Safety**: Go's strong typing catches errors at compile time
5. **Maintainability**: Clear separation of concerns between services
6. **Scalability**: Each service can be horizontally scaled based on demand

## Future Improvements

- [ ] Service discovery for dynamic Python service URLs
- [ ] Health checks and circuit breakers
- [ ] Metrics and observability (Prometheus/Grafana)
- [ ] Testing and CI/CD