# backend/main.py
import contextlib
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
import asyncio

import httpx
import json
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import AsyncOpenAI
from pydantic import Field, BaseModel as PydanticBaseModel
from supabase import create_async_client, AsyncClient

# Define model categories as a constant to be used throughout the application
DEFAULT_MODEL_CATEGORIES = ["reasoning", "function-calling", "text-to-text", "multilingual", "nsfw"]

# Define model rankings data for frontend consumption
MODEL_RANKINGS = {
    "reasoning": {
        "title": "Reasoning",
        "description": "Logic, problem-solving, and analysis",
        "models": [
            {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B", "primary": True},
            {"id": "gemma2-9b-it", "name": "Gemma 2 9B", "primary": False},
            {"id": "llama-3.1-8b-instant", "name": "Llama 3.1 8B", "primary": False},
        ]
    },
    "function-calling": {
        "title": "Function Calling",
        "description": "API interactions and tool usage",
        "models": [
            {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B", "primary": True},
            {"id": "llama-3.1-8b-instant", "name": "Llama 3.1 8B", "primary": False},
        ]
    },
    "text-to-text": {
        "title": "Text to Text",
        "description": "General text generation and transformation",
        "models": [
            {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B", "primary": True},
            {"id": "gemma2-9b-it", "name": "Gemma 2 9B", "primary": False},
            {"id": "llama-3.1-8b-instant", "name": "Llama 3.1 8B", "primary": False},
        ]
    },
    "multilingual": {
        "title": "Multilingual",
        "description": "Cross-language generation and translation",
        "models": [
            {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B", "primary": True},
            {"id": "llama-3.1-8b-instant", "name": "Llama 3.1 8B", "primary": False},
        ]
    },
    "nsfw": {
        "title": "NSFW Detection",
        "description": "Content moderation and safety evaluation",
        "models": [
            {"id": "llama-guard-3-8b", "name": "Llama Guard 3 8B", "primary": True},
            {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B", "primary": False},
        ]
    }
}

try:
    import torch
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers or PyTorch not available. Model classification features will be disabled.")

from openai import AsyncOpenAI as AsyncOpenRouter
from openai import AsyncOpenAI as AsyncGroqAI

import backend.shared as shared
from backend.shared import settings, get_current_user

try:
    import llmlingua
    LLMLINGUA_AVAILABLE = True
except ImportError:
    LLMLINGUA_AVAILABLE = False
    logging.warning("llmlingua not available. Compression features will be disabled.")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

app_state: Dict[str, Any] = {}

lingua_compressor: Optional[llmlingua.PromptCompressor] = None
if LLMLINGUA_AVAILABLE:
    try:
        lingua_compressor = llmlingua.PromptCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            use_llmlingua2=True,
            device_map="cpu"
        )
        log.info("LLMLingua compressor initialized successfully using CPU.")
    except Exception as e:
        log.error(f"Failed to initialize LLMLingua compressor: {e}")
        lingua_compressor = None
else:
    log.warning("LLMLingua dependency not found, compressor not initialized.")

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup and shutdown events."""
    log.info("Application startup: Initializing clients and analyzers...")

    initialized_supabase_client: AsyncClient | None = None
    try:
        if settings.SUPABASE_URL and settings.SUPABASE_SERVICE_ROLE_KEY:
            service_key = settings.SUPABASE_SERVICE_ROLE_KEY
            if hasattr(service_key, 'get_secret_value'):
                service_key = service_key.get_secret_value()
                
            initialized_supabase_client = await create_async_client(
                settings.SUPABASE_URL,
                service_key
            )
            if isinstance(initialized_supabase_client, AsyncClient):
                log.info("Supabase AsyncClient initialized successfully via lifespan.")
            else:
                log.error(
                    f"Lifespan Error: create_client returned {type(initialized_supabase_client)} instead of AsyncClient!")
                initialized_supabase_client = None
        else:
            log.error("Lifespan Error: Supabase URL/Service Key missing in settings.")
            initialized_supabase_client = None
    except Exception as e:
        log.critical(f"CRITICAL: Failed to initialize Supabase client during startup: {e}")
        initialized_supabase_client = None
    finally:
        app_state["supabase_client"] = initialized_supabase_client
    
    initialized_openrouter_client: AsyncOpenRouter | None = None
    try:
        if settings.OPENROUTER_API_BASE_URL and settings.OPENROUTER_API_KEY:
            api_key = settings.OPENROUTER_API_KEY
            if hasattr(api_key, 'get_secret_value'):
                api_key = api_key.get_secret_value()
                
            initialized_openrouter_client = AsyncOpenRouter(
                api_key=api_key,
                base_url=settings.OPENROUTER_API_BASE_URL,
                timeout=30.0,
            )
            log.info("OpenRouter client initialized successfully via lifespan.")
        else:
            log.error("Lifespan Error: OpenRouter URL/Key missing in settings.")
            initialized_openrouter_client = None
    except Exception as e:
        log.critical(f"CRITICAL: Failed to initialize OpenRouter client during startup: {e}")
        initialized_openrouter_client = None
    finally:
        app_state["openrouter_client"] = initialized_openrouter_client

    initialized_groq_client: AsyncGroqAI | None = None
    try:
        if settings.GROQ_API_BASE_URL and settings.GROQ_API_KEY:
            api_key = settings.GROQ_API_KEY
            if hasattr(api_key, 'get_secret_value'):
                api_key = api_key.get_secret_value()
                
            initialized_groq_client = AsyncGroqAI(
                api_key=api_key,
                base_url=settings.GROQ_API_BASE_URL,
                timeout=30.0,
            )
            log.info("Groq client initialized successfully via lifespan.")
        else:
            log.error("Lifespan Error: Groq URL/Key missing in settings.")
            initialized_groq_client = None
    except Exception as e:
        log.critical(f"CRITICAL: Failed to initialize Groq client during startup: {e}")
        initialized_groq_client = None
    finally:
        app_state["groq_client"] = initialized_groq_client

    initialized_bart_classifier = None
    try:
        if not TRANSFORMERS_AVAILABLE:
            log.warning("Skipping BART classifier initialization as dependencies are not available.")
            initialized_bart_classifier = None
        else:
            model_name = "facebook/bart-large-mnli"
            if torch.cuda.is_available():
                device = 0
                log.info(f"CUDA available, initializing BART classifier on GPU device {device}")
            else:
                device = -1
                log.warning("CUDA not available, initializing BART classifier on CPU (slower performance)")
            
            initialized_bart_classifier = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=device
            )
            log.info("BART Classifier initialized successfully via lifespan.")
    except Exception as e:
        log.critical(f"CRITICAL: Failed to initialize BART Classifier: {e}")
    finally:
        app_state["bart_classifier"] = initialized_bart_classifier

    text_splitter = None
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=250,
            chunk_overlap=30,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
        )
        log.info("Text Splitter initialized.")
        app_state["text_splitter"] = text_splitter
    except Exception as e:
        log.critical(f"CRITICAL: Failed to initialize Text Splitter: {e}")
        app_state["text_splitter"] = None

    log.info("Client/Analyzer initialization process complete.")
    
    app_state["lingua_compressor"] = lingua_compressor
    
    yield
    
    log.info("Application shutdown: Cleaning up resources...")
    supabase_client_to_close = app_state.get("supabase_client")
    if supabase_client_to_close and hasattr(supabase_client_to_close, 'aclose'):
        try:
            log.info("Checked Supabase client for cleanup (v2 usually self-manages).")
        except Exception as e:
            log.error(f"Error during Supabase client hypothetical cleanup: {e}")

app = FastAPI(
    title="Intelligent Model Selection API",
    description="API for prompt classification and intelligent model selection using BART and OpenRouter",
    version="0.2.0",
    lifespan=lifespan
)

origins = [
    "http://localhost:3000",
    "http://192.168.4.206:3000",
    # Add production frontend URL here
    # "https://your-deployed-app.com",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)
log.info(f"CORS middleware configured for origins: {origins}")

class ModelSelectionRequest(PydanticBaseModel):
    prompt: str = Field(..., min_length=5, max_length=2000, description="The prompt text to classify and process")
    possible_categories: list[str] = Field(
        default=DEFAULT_MODEL_CATEGORIES, 
        description="Categories to classify the prompt against"
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature for model generation")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p for model generation")
    max_tokens: int = Field(default=250, gt=0, description="Maximum tokens to generate")

class ModelSelectionResponse(PydanticBaseModel):
    selected_model: str
    prompt_category: str
    confidence_score: float
    all_categories: Dict[str, float]
    completion: str

class ClassificationRequest(PydanticBaseModel):
    prompt: str = Field(..., min_length=5, max_length=2000, description="The prompt text to classify")
    possible_categories: list[str] = Field(
        default=DEFAULT_MODEL_CATEGORIES, 
        description="Categories to classify the prompt against"
    )
    multi_label: bool = Field(default=False, description="Whether to allow multiple category labels")

class ClassificationResponse(PydanticBaseModel):
    top_category: str
    confidence_score: float
    all_categories: Dict[str, float]
    recommended_model: str

class CompressionRequest(PydanticBaseModel):
    text: str = Field(..., min_length=10, description="The text to compress")
    target_token: int = Field(default=100, gt=0, description="Target number of tokens after compression")

class CompressionResponse(PydanticBaseModel):
    original_text: str
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float

class GenerateRequest(PydanticBaseModel):
    prompt: str = Field(..., min_length=1, description="The prompt to send to the LLM")
    model: str = Field(..., description="The specific OpenRouter model ID to use (e.g., 'anthropic/claude-3-sonnet:beta')")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature for model generation")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p for model generation")
    max_tokens: int = Field(default=250, gt=0, description="Maximum tokens to generate")

def get_supabase_client() -> AsyncClient:
    """Dependency injector for the initialized Supabase client."""
    client = app_state.get("supabase_client")
    if client is None:
        log.error("Supabase client requested but is not available (initialization failed?).")
        raise HTTPException(status_code=503, detail="Database service temporarily unavailable.")
    return client

def get_bart_classifier():
    """Dependency injector for the initialized BART classifier."""
    classifier = app_state.get("bart_classifier")
    if classifier is None:
        log.error("BART classifier requested but is not available (initialization failed?).")
        raise HTTPException(status_code=503, detail="Text classification service temporarily unavailable.")
    return classifier

def get_openrouter_client() -> AsyncOpenRouter:
    """Dependency injector for the initialized OpenRouter client."""
    client = app_state.get("openrouter_client")
    if client is None:
        log.error("OpenRouter client requested but is not available (initialization failed?).")
        raise HTTPException(status_code=503, detail="OpenRouter service temporarily unavailable.")
    return client

def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """Dependency injector for the initialized text splitter."""
    splitter = app_state.get("text_splitter")
    if splitter is None:
        log.error("Text splitter requested but is not available (initialization failed?).")
        raise HTTPException(status_code=503, detail="Text processing service temporarily unavailable.")
    return splitter

def get_lingua_compressor() -> llmlingua.PromptCompressor:
    """Dependency injector for the initialized LLMLingua compressor."""
    compressor = app_state.get("lingua_compressor")
    if compressor is None:
        log.error("LLMLingua compressor requested but is not available (initialization failed or dependency missing?).")
        raise HTTPException(status_code=503, detail="Compression service temporarily unavailable.")
    return compressor

def get_groq_client() -> AsyncGroqAI:
    """Dependency injector for the initialized Groq client."""
    client = app_state.get("groq_client")
    if client is None:
        log.error("Groq client requested but is not available (initialization failed?).")
        raise HTTPException(status_code=503, detail="Groq service temporarily unavailable.")
    return client

async def select_model_for_category(category: str) -> str:
    """
    Select the most appropriate Groq model based on the prompt category.
    Returns the model ID to be used with Groq API.
    """
    # Check if category exists in MODEL_RANKINGS
    if category.lower() in MODEL_RANKINGS:
        # Get the primary model for this category
        for model in MODEL_RANKINGS[category.lower()]["models"]:
            if model["primary"]:
                return model["id"]
        
        # If no primary model found, use the first one
        if MODEL_RANKINGS[category.lower()]["models"]:
            return MODEL_RANKINGS[category.lower()]["models"][0]["id"]
    
    # Default fallback if category not found or no models defined
    return "llama-3.1-8b-instant"

@app.post("/api/models/select")
async def stream_model_selection(
    request_data: ModelSelectionRequest,
    request: Request,
    bart_classifier=Depends(get_bart_classifier),
    groq_client=Depends(get_groq_client),
):
    """
    Classifies a prompt using Facebook BART and selects the appropriate model from OpenRouter.ai.
    Then processes the prompt with the selected model and returns the result.
    """
    prompt = request_data.prompt
    categories = request_data.possible_categories
    temperature = request_data.temperature
    top_p = request_data.top_p
    max_tokens = request_data.max_tokens
    
    log.info(f"Initiating streaming request for prompt length: {len(prompt)}")
    log.info(f"Categories: {categories}, Temp: {temperature}, Top-P: {top_p}, Max Tokens: {max_tokens}")

    async def event_generator() -> AsyncGenerator[Dict[str, Any], None]:
        try:
            log.info("Classifying prompt with BART...")
            classification_result = bart_classifier(
                prompt, categories, multi_label=False
            )
            top_category = classification_result["labels"][0]
            top_score = classification_result["scores"][0]
            all_categories = {
                label: score 
                for label, score in zip(classification_result["labels"], classification_result["scores"])
            }
            log.info(f"Classified as '{top_category}' ({top_score:.2f})")

            selected_model = await select_model_for_category(top_category)
            log.info(f"Selected model: {selected_model}")

            metadata = {
                "prompt_category": top_category,
                "confidence_score": top_score,
                "selected_model": selected_model,
                "all_categories": all_categories,
            }
            log.info("Yielding metadata event")
            yield {
                "event": "metadata",
                "data": json.dumps(metadata) 
            }
            log.info("Sent metadata event")

            log.info(f"Streaming from Groq model: {selected_model}")
            stream = await groq_client.chat.completions.create(
                model=selected_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stream=True,
            )
            log.info("Got stream object from Groq. Starting iteration...")
            
            chunk_count = 0
            async for chunk in stream:
                chunk_count += 1
                log.debug(f"Received chunk {chunk_count} from Groq stream")
                if await request.is_disconnected():
                    log.warning("Client disconnected, stopping stream.")
                    break
                content = chunk.choices[0].delta.content
                
                log.info(f"[BACKEND STREAM] Raw chunk content: {repr(content)}")

                if content:
                    log.info(f"Yielding text chunk {chunk_count}: {content[:50]}...")
                    yield {
                        "event": "text_chunk",
                        "data": content
                    }
            log.info(f"Finished iterating through Groq stream after {chunk_count} chunks.")
            
            log.info("Yielding end_stream event")
            yield {"event": "end_stream", "data": "Stream finished"}
            log.info("Finished streaming and sent end event.")

        except Exception as e:
            log.error(f"Error during stream generation: {e}", exc_info=True)
            error_data = {"error": "An error occurred during processing.", "detail": str(e)}
            log.info("Yielding error event")
            yield {
                "event": "error",
                "data": json.dumps(error_data)
            }

    return EventSourceResponse(event_generator())

@app.post("/api/compress", response_model=CompressionResponse)
async def compress_text(
    request: CompressionRequest,
    compressor: llmlingua.PromptCompressor = Depends(get_lingua_compressor)
):
    """Compresses the input text using LLMLingua."""
    log.info(f"Received compression request. Target tokens: {request.target_token}")
    try:
        result = await asyncio.to_thread(
            compressor.compress_prompt,
            request.text,
            target_token=request.target_token,
        )
        
        original_text = request.text
        compressed_text = result.get("compressed_prompt", "")
        original_tokens = result.get("origin_tokens", 0)
        compressed_tokens = result.get("compressed_tokens", 0)
        compression_ratio = original_tokens / compressed_tokens if compressed_tokens > 0 else 0
        
        log.info(f"Prompt classified as '{top_category}' with confidence {top_score:.2f}")
        
        # Convert all classification results to dictionary
        all_categories = {
            label: score 
            for label, score in zip(classification_result["labels"], classification_result["scores"])
        }
        
        # Select the appropriate model based on the classification
        selected_model = await select_model_for_category(top_category)
        log.info(f"Selected model: {selected_model} for category: {top_category}")
        
        # Process the prompt with the selected model
        log.info(f"Sending prompt to OpenRouter with model: {selected_model}")
        completion_response = await openrouter_client.chat.completions.create(
            model=selected_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=request.temperature,
            top_p=request.top_p,
        )
        
        # Extract the completion text
        completion_text = completion_response.choices[0].message.content
        log.info(f"Received completion from OpenRouter, length: {len(completion_text)} characters")
        
        # Return the results
        return ModelSelectionResponse(
            selected_model=selected_model,
            prompt_category=top_category,
            confidence_score=top_score,
            all_categories=all_categories,
            completion=completion_text
        )
        
    except Exception as e:
        log.error(f"Error in model selection endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process request: {str(e)}"
        )

@app.post("/api/generate")
async def stream_direct_generation(
    request_data: GenerateRequest,
    request: Request,
    groq_client=Depends(get_groq_client),
):
    """Streams a completion directly from a specified Groq model using SSE."""
    prompt = request_data.prompt
    model = request_data.model
    temperature = request_data.temperature
    top_p = request_data.top_p
    max_tokens = request_data.max_tokens
    
    log.info(f"Initiating direct generation stream. Model: {model}, Prompt length: {len(prompt)}")
    log.info(f"Params: Temp: {temperature}, Top-P: {top_p}, Max Tokens: {max_tokens}")

    async def event_generator() -> AsyncGenerator[Dict[str, Any], None]:
        try:
            log.info(f"Streaming from Groq model: {model}")
            stream = await groq_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stream=True,
            )
            log.info("Got stream object from Groq. Starting iteration...")
            
            chunk_count = 0
            async for chunk in stream:
                chunk_count += 1
                log.debug(f"Received chunk {chunk_count} from Groq stream")
                if await request.is_disconnected():
                    log.warning("Client disconnected, stopping stream.")
                    break
                
                content = chunk.choices[0].delta.content
                log.info(f"[BACKEND GEN STREAM] Raw chunk content: {repr(content)}")
                
                if content:
                    log.info(f"Yielding text chunk {chunk_count}: {content[:50]}...")
                    yield {
                        "event": "text_chunk",
                        "data": content
                    }
            log.info(f"Finished iterating through Groq stream after {chunk_count} chunks.")
            
            log.info("Yielding end_stream event")
            yield {"event": "end_stream", "data": "Stream finished"}
            log.info("Finished streaming and sent end event.")

        except Exception as e:
            log.error(f"Error during direct generation stream: {e}", exc_info=True)
            error_data = {"error": "An error occurred during generation.", "detail": str(e)}
            log.info("Yielding error event")
            yield {
                "event": "error",
                "data": json.dumps(error_data)
            }

    return EventSourceResponse(event_generator())

@app.post("/api/classify", response_model=ClassificationResponse)
async def classify_prompt(
    request: ClassificationRequest,
    bart_classifier = Depends(get_bart_classifier)
):
    """
    Classifies a prompt using Facebook BART without generating a completion.
    Returns the classification results and recommended model.
    """
    prompt = request.prompt
    categories = request.possible_categories
    
    log.info(f"Processing classification request with prompt length: {len(prompt)}")
    log.info(f"Categories to classify against: {categories}")
    
    try:
        log.info("Classifying prompt with BART...")
        classification_result = bart_classifier(
            prompt, 
            categories,
            multi_label=request.multi_label
        )
        
        top_category = classification_result["labels"][0]
        top_score = classification_result["scores"][0]
        
        log.info(f"Prompt classified as '{top_category}' with confidence {top_score:.2f}")
        
        all_categories = {
            label: score 
            for label, score in zip(classification_result["labels"], classification_result["scores"])
        }
        
        recommended_model = await select_model_for_category(top_category)
        
        return ClassificationResponse(
            top_category=top_category,
            confidence_score=top_score,
            all_categories=all_categories,
            recommended_model=recommended_model
        )
        
    except Exception as e:
        log.error(f"Error in classification endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process classification request: {str(e)}"
        )

@app.get("/api/model-rankings")
async def get_model_rankings():
    """
    Returns the model rankings data for all categories.
    This data is used by the frontend to display model recommendations.
    """
    try:
        return MODEL_RANKINGS
    except Exception as e:
        log.error(f"Error retrieving model rankings: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve model rankings data"
        )

