# backend/main.py
import contextlib
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, AsyncGenerator  # Added AsyncGenerator
import asyncio # Added asyncio for sleep

import httpx
import json # Added json
from fastapi import FastAPI, HTTPException, Depends, Request # Added Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse # Added for SSE
# --- Text Splitter ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import AsyncOpenAI
from pydantic import Field, BaseModel as PydanticBaseModel
# --- Import Supabase create_client and AsyncClient ---
from supabase import create_async_client, AsyncClient

# --- Import for BART classification ---
try:
    import torch
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers or PyTorch not available. Model classification features will be disabled.")

# --- OpenRouter Client ---
from openai import AsyncOpenAI as AsyncOpenRouter

# --- Import from shared module ---
# Use absolute imports instead of relative imports
import backend.shared as shared
from backend.shared import settings, get_current_user

# Uncomment and ensure dependencies are installed if /upload endpoint is used
# import fitz # PyMuPDF
# from fastapi import UploadFile, File, Form
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.vectorstores import Chroma

# --- Import for LLMLingua --- NEW
try:
    import llmlingua
    LLMLINGUA_AVAILABLE = True
except ImportError:
    LLMLINGUA_AVAILABLE = False
    logging.warning("llmlingua not available. Compression features will be disabled.")

# --- Logging Setup ---
# Configure logging (consider moving to shared.py or a dedicated logging setup)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)  # Logger for this module

# --- App State for Clients & Analyzers ---
# Stores clients initialized during startup
app_state: Dict[str, Any] = {}

# --- LLMLingua Compressor Instance --- NEW
# Initialize compressor once globally (or manage via lifespan if preferred)
# Note: LLMLingua might download models on first run.
lingua_compressor: Optional[llmlingua.PromptCompressor] = None
if LLMLINGUA_AVAILABLE:
    try:
        # Consider initializing in lifespan for cleaner management
        # For simplicity here, initialize globally. Choose model as needed.
        lingua_compressor = llmlingua.PromptCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank", # Or choose another model
            use_llmlingua2=True, # Use LLMLingua2
            # --- Explicitly set device to CPU --- 
            device_map="cpu" 
            # --- End Explicit CPU setting ---
        )
        log.info("LLMLingua compressor initialized successfully using CPU.")
    except Exception as e:
        log.error(f"Failed to initialize LLMLingua compressor: {e}")
        lingua_compressor = None
else:
    log.warning("LLMLingua dependency not found, compressor not initialized.")


# Add these near your other Pydantic models in main.py


# --- FastAPI Lifespan Event Handler ---
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup and shutdown events."""
    # --- Startup Logic ---
    log.info("Application startup: Initializing clients and analyzers...")

    # Initialize Supabase Async Client
    initialized_supabase_client: AsyncClient | None = None
    try:
        if settings.SUPABASE_URL and settings.SUPABASE_SERVICE_ROLE_KEY:
            # Make sure to handle SecretStr appropriately
            service_key = settings.SUPABASE_SERVICE_ROLE_KEY
            if hasattr(service_key, 'get_secret_value'):
                service_key = service_key.get_secret_value()
                
            # Use await here because we are in an async context
            initialized_supabase_client = await create_async_client(
                settings.SUPABASE_URL,
                service_key
            )
            # Check type robustly
            if isinstance(initialized_supabase_client, AsyncClient):
                log.info("Supabase AsyncClient initialized successfully via lifespan.")
                # Optional: Perform a quick test connection if desired
                # try:
                #     await initialized_supabase_client.table('analyses').select('id', head=True).limit(1).execute()
                #     log.info("Supabase connection test successful.")
                # except Exception as db_conn_err:
                #     log.error(f"Supabase connection test failed: {db_conn_err}")
                #     initialized_supabase_client = None # Treat as failure if test fails
            else:
                log.error(
                    f"Lifespan Error: create_client returned {type(initialized_supabase_client)} instead of AsyncClient!")
                initialized_supabase_client = None  # Ensure it's None if wrong type
        else:
            log.error("Lifespan Error: Supabase URL/Service Key missing in settings.")
            initialized_supabase_client = None
    except Exception as e:
        log.critical(f"CRITICAL: Failed to initialize Supabase client during startup: {e}")
        initialized_supabase_client = None
    finally:
        # Store the result (client or None) in app state
        app_state["supabase_client"] = initialized_supabase_client
    
    # Initialize OpenRouter Client
    initialized_openrouter_client: AsyncOpenRouter | None = None
    try:
        if settings.OPENROUTER_API_BASE_URL and settings.OPENROUTER_API_KEY:
            # Make sure to handle SecretStr appropriately
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
        # Store the result (client or None) in app state
        app_state["openrouter_client"] = initialized_openrouter_client

    # Initialize BART Classifier
    initialized_bart_classifier = None
    try:
        if not TRANSFORMERS_AVAILABLE:
            log.warning("Skipping BART classifier initialization as dependencies are not available.")
            initialized_bart_classifier = None
        else:
            # Load BART model for text classification
            model_name = "facebook/bart-large-mnli"
            # Only initialize if CUDA is available, otherwise log warning
            if torch.cuda.is_available():
                device = 0  # Use first GPU
                log.info(f"CUDA available, initializing BART classifier on GPU device {device}")
            else:
                device = -1  # CPU
                log.warning("CUDA not available, initializing BART classifier on CPU (slower performance)")
            
            # Initialize the classification pipeline
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

    # Store Text Splitter (initialized outside lifespan)
    text_splitter = None
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=250,  # Adjust as needed
            chunk_overlap=30,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],  # Prioritize sentences/paragraphs
        )
        log.info("Text Splitter initialized.")
        app_state["text_splitter"] = text_splitter
    except Exception as e:
        log.critical(f"CRITICAL: Failed to initialize Text Splitter: {e}")
        app_state["text_splitter"] = None

    log.info("Client/Analyzer initialization process complete.")
    
    # --- ADDED: Store compressor in app_state (if initialized globally) ---
    app_state["lingua_compressor"] = lingua_compressor
    # --- END ADDED ---
    
    yield  # Application runs after yield
    
    # --- Shutdown Logic ---
    log.info("Application shutdown: Cleaning up resources...")
    # Example: Close httpx clients if needed (openai client handles its own)
    supabase_client_to_close = app_state.get("supabase_client")
    if supabase_client_to_close and hasattr(supabase_client_to_close, 'aclose'):
        try:
            # Supabase client v2 doesn't have an explicit aclose, it relies on httpx client closure.
            # If you were managing a raw httpx client, you'd close it here.
            # For Supabase v2, usually no specific action needed on shutdown.
            log.info("Checked Supabase client for cleanup (v2 usually self-manages).")
        except Exception as e:
            log.error(f"Error during Supabase client hypothetical cleanup: {e}")


# --- FastAPI App Initialization with Lifespan ---
app = FastAPI(
    title="Intelligent Model Selection API",
    description="API for prompt classification and intelligent model selection using BART and OpenRouter",
    version="0.2.0",
    lifespan=lifespan
)

# --- CORS Middleware ---
origins = [
    "http://localhost:3000",
    # Add production frontend URL here
    # "https://your-deployed-app.com",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Limit methods
    allow_headers=["Authorization", "Content-Type"],  # Limit headers
)
log.info(f"CORS middleware configured for origins: {origins}")


# --- Pydantic Models for Model Selection ---
class ModelSelectionRequest(PydanticBaseModel):
    prompt: str = Field(..., min_length=5, max_length=2000, description="The prompt text to classify and process")
    possible_categories: list[str] = Field(
        default=["creative", "factual", "coding", "math", "reasoning"], 
        description="Categories to classify the prompt against"
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature for model generation")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p for model generation")


class ModelSelectionResponse(PydanticBaseModel):
    selected_model: str
    prompt_category: str
    confidence_score: float
    all_categories: Dict[str, float]
    completion: str


# --- Pydantic Models for Classification ---
class ClassificationRequest(PydanticBaseModel):
    prompt: str = Field(..., min_length=5, max_length=2000, description="The prompt text to classify")
    possible_categories: list[str] = Field(
        default=["creative", "factual", "coding", "math", "reasoning"], 
        description="Categories to classify the prompt against"
    )
    multi_label: bool = Field(default=False, description="Whether to allow multiple category labels")


class ClassificationResponse(PydanticBaseModel):
    top_category: str
    confidence_score: float
    all_categories: Dict[str, float]
    recommended_model: str


# --- Pydantic Models for Compression --- NEW
class CompressionRequest(PydanticBaseModel):
    text: str = Field(..., min_length=10, description="The text to compress")
    target_token: int = Field(default=100, gt=0, description="Target number of tokens after compression")
    # Add other llmlingua parameters if needed (e.g., rank_method, context_budget)

class CompressionResponse(PydanticBaseModel):
    original_text: str
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    # Add other llmlingua output fields if desired


# --- Pydantic Models for Direct Generation --- NEW
class GenerateRequest(PydanticBaseModel):
    prompt: str = Field(..., min_length=1, description="The prompt to send to the LLM")
    model: str = Field(..., description="The specific OpenRouter model ID to use (e.g., 'anthropic/claude-3-sonnet:beta')")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature for model generation")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p for model generation")
    max_tokens: int = Field(default=250, gt=0, description="Maximum tokens to generate")


# --- Dependency Injectors for Clients/Tools ---
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


# --- Dependency Injector for LLMLingua --- NEW
def get_lingua_compressor() -> llmlingua.PromptCompressor:
    """Dependency injector for the initialized LLMLingua compressor."""
    compressor = app_state.get("lingua_compressor")
    if compressor is None:
        log.error("LLMLingua compressor requested but is not available (initialization failed or dependency missing?).")
        raise HTTPException(status_code=503, detail="Compression service temporarily unavailable.")
    return compressor


# --- Define Model Selection Logic ---
async def select_model_for_category(category: str) -> str:
    """
    Select the most appropriate OpenRouter model based on the prompt category.
    Returns the model ID to be used with OpenRouter.
    """
    model_mapping = {
        "creative": "anthropic/claude-3-sonnet:beta",#"anthropic/claude-3-opus:beta",  # Best for creative tasks
        "factual": "anthropic/claude-3-sonnet:beta",  # Good for fact-based tasks
        "coding":   "anthropic/claude-3-sonnet:beta", # "openai/gpt-4-turbo",  # Strong for coding
        "math": "anthropic/claude-3-opus:beta",  # Good for mathematical reasoning
        "reasoning":  "openai/gpt-4-turbo", #"anthropic/claude-3-sonnet:beta",  # Good for general reasoning
        # Default fallback
        "default": "mistralai/mistral-7b-instruct"
    }
    
    return model_mapping.get(category.lower(), model_mapping["default"])


# --- Model Selection Streaming Endpoint --- 
@app.post("/api/models/select") # Keep POST, but response will be SSE
async def stream_model_selection(
    request_data: ModelSelectionRequest, # Receive data in body
    request: Request, # Needed for EventSourceResponse
    bart_classifier=Depends(get_bart_classifier),
    openrouter_client=Depends(get_openrouter_client),
):
    """
    Classifies prompt, selects model, then streams the completion using SSE.
    Sends metadata first, then text chunks.
    """
    prompt = request_data.prompt
    categories = request_data.possible_categories
    temperature = request_data.temperature
    top_p = request_data.top_p
    
    log.info(f"Initiating streaming request for prompt length: {len(prompt)}")
    log.info(f"Categories: {categories}, Temp: {temperature}, Top-P: {top_p}")

    async def event_generator() -> AsyncGenerator[Dict[str, Any], None]:
        # Send an immediate ping event to test connection
        # try:
        #     log.info("Sending initial ping event")
        #     yield {"event": "ping", "data": datetime.now().isoformat()}
        #     log.info("Initial ping event sent")
        # except Exception as ping_err:
        #     log.error(f"Error sending initial ping: {ping_err}")
        
        try:
            # 1. Classify Prompt
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

            # 2. Select Model
            selected_model = await select_model_for_category(top_category)
            log.info(f"Selected model: {selected_model}")

            # 3. Send Metadata Event
            metadata = {
                "prompt_category": top_category,
                "confidence_score": top_score,
                "selected_model": selected_model,
                "all_categories": all_categories,
            }
            log.info("Yielding metadata event") # Log before yield
            yield {
                "event": "metadata",
                "data": json.dumps(metadata) 
            }
            log.info("Sent metadata event")

            # 4. Stream Completion from OpenRouter
            log.info(f"Streaming from OpenRouter model: {selected_model}")
            stream = await openrouter_client.chat.completions.create(
                model=selected_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                max_tokens=250,
                stream=True, # Enable streaming
            )
            log.info("Got stream object from OpenRouter. Starting iteration...") # Log after create
            
            # 5. Yield Text Chunks
            chunk_count = 0
            async for chunk in stream:
                chunk_count += 1
                log.debug(f"Received chunk {chunk_count} from OpenRouter stream") # More verbose debug log
                if await request.is_disconnected():
                    log.warning("Client disconnected, stopping stream.")
                    break
                content = chunk.choices[0].delta.content
                
                # --- ADD LOGGING HERE ---
                # Log raw content *before* potential trimming
                log.info(f"[BACKEND STREAM] Raw chunk content: {repr(content)}") 
                # --- END LOGGING ---

                # --- REMOVED TRIM WHITESPACE --- 
                # if content:
                #     content = content.strip()
                #     log.info(f"[BACKEND STREAM] Trimmed chunk content: {repr(content)}") # Log the content AFTER trimming
                # --- END REMOVED TRIM ---

                if content: # Check content again (it might be None)
                    log.info(f"Yielding text chunk {chunk_count}: {content[:50]}...")  # Log before yield
                    yield {
                        "event": "text_chunk",
                        "data": content
                    }
            log.info(f"Finished iterating through OpenRouter stream after {chunk_count} chunks.")
            
            # 6. Send End Event
            log.info("Yielding end_stream event") # Log before yield
            yield {"event": "end_stream", "data": "Stream finished"}
            log.info("Finished streaming and sent end event.")

        except Exception as e:
            log.error(f"Error during stream generation: {e}", exc_info=True)
            error_data = {"error": "An error occurred during processing.", "detail": str(e)}
            log.info("Yielding error event") # Log before yield
            yield {
                "event": "error",
                "data": json.dumps(error_data)
            }

    return EventSourceResponse(event_generator())


# --- Compression Endpoint --- NEW
@app.post("/api/compress", response_model=CompressionResponse)
async def compress_text(
    request: CompressionRequest,
    compressor: llmlingua.PromptCompressor = Depends(get_lingua_compressor)
):
    """Compresses the input text using LLMLingua."""
    log.info(f"Received compression request. Target tokens: {request.target_token}")
    try:
        # Perform compression
        result = await asyncio.to_thread(
            compressor.compress_prompt,
            request.text,
            target_token=request.target_token,
            # Add other parameters here if needed
            # e.g., rate_limit=0.5, context_budget="+100", rank_method="longllmlingua"
        )
        
        # Extract results (structure might vary slightly based on llmlingua version)
        original_text = request.text
        compressed_text = result.get("compressed_prompt", "")
        original_tokens = result.get("origin_tokens", 0)
        compressed_tokens = result.get("compressed_tokens", 0)
        compression_ratio = original_tokens / compressed_tokens if compressed_tokens > 0 else 0
        
        log.info(f"Compression successful. Original tokens: {original_tokens}, Compressed: {compressed_tokens}")
        
        return CompressionResponse(
            original_text=original_text,
            compressed_text=compressed_text,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compression_ratio
        )

    except Exception as e:
        log.error(f"Error during text compression: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compress text: {str(e)}"
        )


# --- Direct Generation Streaming Endpoint --- NEW
@app.post("/api/generate") # Response will be SSE
async def stream_direct_generation(
    request_data: GenerateRequest, # Receive data in body
    request: Request, # Needed for EventSourceResponse
    openrouter_client=Depends(get_openrouter_client),
):
    """Streams a completion directly from a specified OpenRouter model using SSE."""
    prompt = request_data.prompt
    model = request_data.model
    temperature = request_data.temperature
    top_p = request_data.top_p
    max_tokens = request_data.max_tokens
    
    log.info(f"Initiating direct generation stream. Model: {model}, Prompt length: {len(prompt)}")
    log.info(f"Params: Temp: {temperature}, Top-P: {top_p}, Max Tokens: {max_tokens}")

    async def event_generator() -> AsyncGenerator[Dict[str, Any], None]:
        try:
            # 1. Stream Completion from specified OpenRouter model
            log.info(f"Streaming from OpenRouter model: {model}")
            stream = await openrouter_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                max_tokens=100, # change this
                stream=True, # Enable streaming
            )
            log.info("Got stream object from OpenRouter. Starting iteration...")
            
            # 2. Yield Text Chunks
            chunk_count = 0
            async for chunk in stream:
                chunk_count += 1
                log.debug(f"Received chunk {chunk_count} from OpenRouter stream")
                if await request.is_disconnected():
                    log.warning("Client disconnected, stopping stream.")
                    break
                
                content = chunk.choices[0].delta.content
                log.info(f"[BACKEND GEN STREAM] Raw chunk content: {repr(content)}")
                
                if content: # Send non-empty chunks
                    log.info(f"Yielding text chunk {chunk_count}: {content[:50]}...")
                    yield {
                        "event": "text_chunk",
                        "data": content
                    }
            log.info(f"Finished iterating through OpenRouter stream after {chunk_count} chunks.")
            
            # 3. Send End Event
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


# --- Classification-Only Endpoint ---
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
        # Use BART to classify prompt
        log.info("Classifying prompt with BART...")
        classification_result = bart_classifier(
            prompt, 
            categories,
            multi_label=request.multi_label
        )
        
        # Extract classification results
        top_category = classification_result["labels"][0]
        top_score = classification_result["scores"][0]
        
        log.info(f"Prompt classified as '{top_category}' with confidence {top_score:.2f}")
        
        # Convert all classification results to dictionary
        all_categories = {
            label: score 
            for label, score in zip(classification_result["labels"], classification_result["scores"])
        }
        
        # Get recommended model without actually using it
        recommended_model = await select_model_for_category(top_category)
        
        # Return the results
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

