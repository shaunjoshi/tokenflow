# backend/main.py
import contextlib
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Union  # Import List for type hinting

import httpx
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
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


# --- Define Model Selection Logic ---
async def select_model_for_category(category: str) -> str:
    """
    Select the most appropriate OpenRouter model based on the prompt category.
    Returns the model ID to be used with OpenRouter.
    """
    model_mapping = {
        "creative": "anthropic/claude-3-opus:beta",  # Best for creative tasks
        "factual": "microsoft/phi-4-reasoning-plus:free",  # Good for fact-based tasks
        "coding": "openai/gpt-4-turbo",  # Strong for coding
        "math": "anthropic/claude-3-opus:beta",  # Good for mathematical reasoning
        "reasoning": "anthropic/claude-3-sonnet:beta",  # Good for general reasoning
        # Default fallback
        "default": "mistralai/mistral-7b"
    }
    
    return model_mapping.get(category.lower(), model_mapping["default"])


# --- Model Selection Endpoint ---
@app.post("/api/models/select", response_model=ModelSelectionResponse)
async def select_model(
    request: ModelSelectionRequest,
    bart_classifier = Depends(get_bart_classifier),
    openrouter_client = Depends(get_openrouter_client)
):
    """
    Classifies a prompt using Facebook BART and selects the appropriate model from OpenRouter.ai.
    Then processes the prompt with the selected model and returns the result.
    """
    prompt = request.prompt
    categories = request.possible_categories
    
    log.info(f"Processing model selection request with prompt length: {len(prompt)}")
    log.info(f"Categories to classify against: {categories}")
    
    try:
        # Use BART to classify prompt
        log.info("Classifying prompt with BART...")
        classification_result = bart_classifier(
            prompt, 
            categories,
            multi_label=False  # We want a single category
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

