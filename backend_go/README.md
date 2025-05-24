# TokenFlow Go Backend

This is the Go implementation of the TokenFlow backend service. It acts as the main API gateway and orchestrator, delegating specialized ML tasks to the Python backend while handling HTTP routing, streaming, and business logic.

## Features

- **HTTP API Gateway**: Gin-based high-performance web server
- **Service Architecture**: Modular services with dependency injection
- **Streaming Support**: Server-Sent Events (SSE) for real-time LLM responses
- **External Integrations**: Groq and OpenRouter API clients
- **ML Task Delegation**: Calls Python backend for classification and compression

## Architecture

### Service Layer
- **ModelService**: Manages LLM generation and model selection logic
- **ClassificationService**: Delegates prompt classification to Python backend
- **CompressionService**: Delegates text compression to Python backend

### Handler Layer
- **ModelHandler**: Handles direct generation and model selection endpoints
- **ClassificationHandler**: Manages classification API requests
- **CompressionHandler**: Manages compression API requests

### Models Layer
- Defines request/response structures for type safety
- Shared data models across handlers and services

## Prerequisites

- Go 1.24 or later
- **Python backend running on port 8001** (using Hermit + uv)
- **Hermit environment activated** (`. bin/activate-hermit` from project root)
- Environment variables in `.env` file:
  - `GROQ_API_KEY`
  - `GROQ_API_BASE_URL` 
  - `OPENROUTER_API_KEY`
  - `OPENROUTER_API_BASE_URL`
  - `PORT` (optional, defaults to 8000)

## Project Structure

```
backend_go/
├── cmd/
│   └── server/         # Main application entry point
├── pkg/
│   ├── config/         # Configuration management
│   ├── handlers/       # HTTP request handlers
│   │   ├── classification_handler.go
│   │   ├── compression_handler.go
│   │   └── model_handler.go
│   ├── services/       # Business logic services
│   │   ├── classification_service.go
│   │   ├── compression_service.go
│   │   └── model_service.go
│   ├── models/         # Data models and types
│   └── routes/         # Route definitions
├── .env               # Environment variables
├── go.mod             # Go module file
└── README.md          # This file
```

## Getting Started

1. **Install dependencies:**
   ```bash
   go mod download
   ```

2. **Start the Python backend first:**
   ```bash
   # In another terminal, from project root
   . bin/activate-hermit              # Activate Hermit environment
   cd backend_python
   uv run --active python -m src.tokenflow_python.main
   ```

3. **Run the Go server:**
   ```bash
   go run cmd/server/main.go
   ```

The server will start on port 8000 by default.

## API Endpoints

### Core Endpoints
- `POST /api/compress` - Compress text using Python LLMLingua service
- `POST /api/classify` - Classify prompts using Python Transformers service  
- `POST /api/generate` - Stream directly from a specified LLM model
- `POST /api/models/select` - Auto-select model based on classification and stream
- `GET /api/model-rankings` - Get available model information

### Request Examples

**Compression:**
```bash
curl -X POST http://localhost:8000/api/compress \
  -H "Content-Type: application/json" \
  -d '{"text": "Your long text here", "ratio": 0.5}'
```

**Classification:**
```bash
curl -X POST http://localhost:8000/api/classify \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a function", "possible_categories": ["reasoning", "function-calling"]}'
```

**Generation (SSE Stream):**
```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum computing", "model": "llama-3.3-70b-versatile"}'
```

## Development

### Code Organization
- **Handlers**: Receive HTTP requests, validate input, call services, return responses
- **Services**: Contain business logic, make external API calls, handle data processing
- **Models**: Define data structures for requests, responses, and domain objects

### Dependency Injection Pattern
Services are initialized once and injected into handlers:
```go
// Initialize services
modelService := services.NewModelService()
classificationService := services.NewClassificationService()

// Inject into handlers  
modelHandler := handlers.NewModelHandler(modelService, classificationService)
classificationHandler := handlers.NewClassificationHandler(classificationService)
```

### Error Handling
- Services return Go errors with descriptive messages
- Handlers convert service errors to appropriate HTTP status codes
- Python service errors are parsed and forwarded to clients

## Testing

```bash
# Run all tests
go test ./...

# Run tests with verbose output
go test -v ./...

# Run tests for specific package
go test ./pkg/services
```

## Configuration

Environment variables (`.env` file):
```env
PORT=8000
GROQ_API_KEY=your_groq_key_here
GROQ_API_BASE_URL=https://api.groq.com/openai/v1
OPENROUTER_API_KEY=your_openrouter_key_here
OPENROUTER_API_BASE_URL=https://openrouter.ai/api/v1
ENVIRONMENT=development
```

## Performance Considerations

- **Concurrent Requests**: Go's goroutines handle multiple requests efficiently
- **Streaming**: SSE provides real-time user experience for LLM generation
- **Connection Pooling**: HTTP clients reuse connections to external APIs
- **Resource Management**: Proper cleanup of streams and connections 