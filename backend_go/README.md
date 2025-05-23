# TokenFlow Go Backend

This is the Go implementation of the TokenFlow backend service. It provides the same functionality as the Python backend but is written in Go for better performance and type safety.

## Features

- Model selection and classification
- Text compression
- Streaming responses using Server-Sent Events (SSE)
- Integration with Groq and OpenRouter APIs
- Supabase integration

## Prerequisites

- Go 1.24 or later
- Make sure you have the following environment variables set in your `.env` file:
  - `SUPABASE_URL`
  - `SUPABASE_SERVICE_ROLE_KEY`
  - `OPENROUTER_API_KEY`
  - `OPENROUTER_API_BASE_URL`
  - `GROQ_API_KEY`
  - `GROQ_API_BASE_URL`

## Project Structure

```
backend_go/
├── cmd/
│   └── server/         # Main application entry point
├── pkg/
│   ├── config/         # Configuration management
│   ├── handlers/       # HTTP request handlers
│   ├── models/         # Data models and types
│   ├── services/       # Business logic
│   └── utils/          # Utility functions
├── .env               # Environment variables
├── go.mod             # Go module file
└── README.md          # This file
```

## Getting Started

1. Install dependencies:
   ```bash
   go mod download
   ```

2. Run the server:
   ```bash
   go run cmd/server/main.go
   ```

The server will start on port 8000 by default.

## API Endpoints

- `POST /api/models/select` - Select and stream from the most appropriate model
- `POST /api/generate` - Stream directly from a specified model
- `POST /api/classify` - Classify a prompt
- `POST /api/compress` - Compress text
- `GET /api/model-rankings` - Get model rankings data

## Development

To run the server in development mode with hot reloading:

```bash
go run cmd/server/main.go
```

## Testing

To run tests:

```bash
go test ./...
```

## TODO

- [ ] Implement BART classification
- [ ] Implement LLMLingua compression
- [ ] Add proper error handling and logging
- [ ] Add tests
- [ ] Add documentation
- [ ] Add CI/CD 