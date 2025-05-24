import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TRANSFORMERS_FORCE_CPU"] = "1"
import asyncio
import atexit
import logging
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Dict, List, Optional

import llmlingua
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from llmlingua import PromptCompressor
from pydantic import BaseModel, Field, ValidationError
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# Load environment variables
load_dotenv()

app = FastAPI()

# Configure CORS
origins = [
    "http://localhost:3000",
    "http://192.168.4.206:3000",
    "http://192.168.5.209:3000",  # Add your local network IP
    "http://localhost:8001",
    "http://192.168.5.209:8001",  # Add Python backend IP
    # Add production frontend URL here
    # "https://your-deployed-app.com",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Initialize global variables
model = None
tokenizer = None
compressor = None
thread_pool = None
bart_classifier = None

# Maximum text length (in characters) to prevent memory issues
MAX_TEXT_LENGTH = 2000

def get_model():
    """Lazy load the BART model, tokenizer, and zero-shot classification pipeline."""
    global model, tokenizer, bart_classifier
    model_name = "facebook/bart-large-mnli"
    device = "cpu"  # Enforce CPU

    if model is None or tokenizer is None:
        logger.info(f"Loading BART model and tokenizer from {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        logger.info("BART model and tokenizer loaded successfully")

    if bart_classifier is None:
        logger.info(f"Initializing zero-shot classification pipeline with {model_name} on {device}...")
        # Ensure model and tokenizer are loaded before initializing pipeline
        if model is None or tokenizer is None: # Should not happen if logic above is correct
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)

        bart_classifier = pipeline(
            "zero-shot-classification",
            model=model, # Use pre-loaded model
            tokenizer=tokenizer, # Use pre-loaded tokenizer
            device=device
        )
        logger.info("Zero-shot classification pipeline initialized successfully")
    return model, tokenizer, bart_classifier

def get_compressor():
    """Lazy load the LLMLingua compressor."""
    global compressor
    if compressor is None:
        try:
            model_to_use = "bert-base-multilingual-cased"
            logger.info(f"Initializing LLMLingua compressor with model: {model_to_use} and use_llmlingua2=True...")
            compressor = PromptCompressor(
                model_name=model_to_use,
                device_map="cpu",
                use_llmlingua2=True # Enable LLMLingua v2 features
            )
            logger.info(f"LLMLingua compressor ({model_to_use}, use_llmlingua2=True) initialized successfully using CPU.")
        except Exception as e:
            logger.error(f"Failed to initialize LLMLingua compressor with {model_to_use} and use_llmlingua2=True: {e}", exc_info=True)
            compressor = None # Ensure compressor is None on failure
    return compressor

def cleanup_resources():
    """Cleanup function to be called on shutdown."""
    global model, tokenizer, compressor, thread_pool
    print("\nCleaning up resources...")

    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Shutdown thread pool
    if thread_pool:
        thread_pool.shutdown(wait=False)

    # Clear model references
    model = None
    tokenizer = None
    compressor = None
    thread_pool = None

    print("Cleanup complete.")

def initialize_models():
    """Initialize models and resources."""
    global model, tokenizer, compressor, thread_pool

    logger.info("Starting model initialization...")

    # Initialize BART model for classification
    try:
        logger.info("Loading BART model and tokenizer...")
        model_name = "facebook/bart-large-mnli"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info("Tokenizer loaded successfully")
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        logger.info("BART model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to initialize BART model: {e}", exc_info=True)
        raise

    # Initialize LLMLingua for compression
    try:
        logger.info("Initializing LLMLingua compressor...")
        compressor = PromptCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            device_map="cpu"
        )
        logger.info("LLMLingua compressor initialized successfully using CPU.")
    except Exception as e:
        logger.error(f"Failed to initialize LLMLingua compressor: {e}", exc_info=True)
        compressor = None

    # Create a thread pool for running compression
    thread_pool = ThreadPoolExecutor(max_workers=1)
    logger.info("Thread pool initialized")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClassificationRequest(BaseModel):
    prompt: str
    possible_categories: List[str]
    multi_label: bool = False

class ClassificationResponse(BaseModel):
    top_category: str
    confidence_score: float
    all_categories: Dict[str, float]
    recommended_model: str

class CompressionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH, description="The text to compress")
    ratio: float = Field(..., gt=0, le=1.0, description="Target compression ratio (e.g., 0.5 for 50% of original tokens)")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "This is a long piece of text that needs to be compressed significantly to save tokens.",
                "ratio": 0.3 # Aim for 30% of original tokens
            }
        }

class CompressionResponse(BaseModel):
    original_text: str
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float

@app.post("/classify", response_model=ClassificationResponse)
async def classify_prompt(request: ClassificationRequest):
    try:
        logger.info("=== Starting classification request ===")
        logger.info(f"Received request with prompt: {request.prompt[:100]}...")
        logger.info(f"Possible categories: {request.possible_categories}")

        # Get the zero-shot classification pipeline
        logger.info("Getting zero-shot classification pipeline...")
        _, _, classifier = get_model() # Model and tokenizer are also returned but not directly used here
        if classifier is None:
            logger.error("Failed to get zero-shot-classification pipeline.")
            raise HTTPException(status_code=503, detail="Classification service not available.")
        logger.info("Zero-shot classification pipeline obtained successfully")

        # Perform classification
        logger.info("Performing zero-shot classification...")
        classification_result = classifier(
            request.prompt,
            request.possible_categories,
            multi_label=request.multi_label
        )
        logger.info(f"Classification result: {classification_result}")

        # Extract results
        top_category = classification_result["labels"][0]
        top_score = float(classification_result["scores"][0]) # Ensure float
        all_categories = {
            label: float(score) # Ensure float
            for label, score in zip(classification_result["labels"], classification_result["scores"])
        }
        logger.info(f"Top category: {top_category} with score {top_score}")
        logger.info(f"All category scores: {all_categories}")

        # Map category to recommended model (using your existing mapping)
        model_mapping = {
            "reasoning": "llama-3.1-8b-instant",
            "function-calling": "llama-3.1-8b-instant",
            "text-to-text": "llama-3.1-8b-instant"
        }
        recommended_model = model_mapping.get(top_category, "llama-3.1-8b-instant")
        logger.info(f"Recommended model: {recommended_model}")

        response = ClassificationResponse(
            top_category=top_category,
            confidence_score=top_score,
            all_categories=all_categories,
            recommended_model=recommended_model
        )
        logger.info("=== Classification request completed successfully ===")
        return response
    except Exception as e:
        logger.error(f"Error in classification endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compress", response_model=CompressionResponse)
async def compress_text(request: CompressionRequest):
    """Compresses the input text using LLMLingua."""
    try:
        logger.info("=== Starting compression request ===")
        logger.info(f"Received compression request. Target ratio: {request.ratio*100:.0f}%, Text length: {len(request.text)} chars")
        logger.info(f"Using LLMLingua version: {llmlingua.__version__ if 'llmlingua' in globals() and hasattr(llmlingua, '__version__') else 'unknown'}")

        if not (0 < request.ratio <= 1.0):
            raise HTTPException(status_code=400, detail="Ratio must be between 0 (exclusive) and 1.0 (inclusive).")

        if len(request.text) > MAX_TEXT_LENGTH:
            raise HTTPException(status_code=400, detail=f"Text too long (max {MAX_TEXT_LENGTH} chars).")

        logger.info("Getting compressor...")
        compressor = get_compressor() # Initialized with use_llmlingua2=True
        if compressor is None:
            logger.error("Compressor is None after get_compressor() call")
            raise HTTPException(status_code=503, detail="Compression service unavailable.")
        logger.info("Compressor obtained successfully")

        logger.info("Preparing to call LLMLingua's compress_prompt with dynamic rate:")
        logger.info(f"  - Text (first 50 chars): '{request.text[:50]}...'")
        logger.info(f"  - Requested ratio: {request.ratio}")

        # Parameters for LLMLingua - using request.ratio for the 'rate' parameter
        # The 'rate' in llmlingua's compress_prompt is actually the target ratio of compressed size / original size.
        compression_params = {
            "rate": request.ratio,
        }

        logger.info(f"Keyword arguments for LLMLingua compress_prompt: {compression_params}")

        try:
            logger.info("Starting compression in thread pool...")
            result = await asyncio.to_thread(
                compressor.compress_prompt,
                context=[request.text],
                **compression_params
            )
            logger.info(f"LLMLingua raw compression result dictionary: {result}")
        except Exception as e:
            logger.error(f"Error during compression: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error during compression: {str(e)}")

        original_text = request.text
        compressed_prompt_value = result.get("compressed_prompt", "")
        if isinstance(compressed_prompt_value, list):
            compressed_text = compressed_prompt_value[0] if compressed_prompt_value else ""
        elif isinstance(compressed_prompt_value, str):
            compressed_text = compressed_prompt_value
        else:
            logger.warning(f"LLMLingua returned unexpected type for compressed_prompt: {type(compressed_prompt_value)}, value: {compressed_prompt_value}")
            compressed_text = str(compressed_prompt_value)

        original_tokens = result.get("origin_tokens", 0)
        compressed_tokens = result.get("compressed_tokens", 0)

        # Calculate actual_ratio based on tokens
        actual_ratio = 0.0
        if original_tokens > 0 and compressed_tokens > 0:
             actual_ratio = compressed_tokens / original_tokens # compressed/original for user understanding of achieved ratio
        elif original_tokens == 0 and compressed_tokens == 0:
            actual_ratio = 1.0 # No tokens in, no tokens out
        elif original_tokens > 0 and compressed_tokens == 0:
            actual_ratio = 0.0 # Full compression to 0 tokens

        # The ratio returned by LLMLingua in its dict is origin/compressed. We are using compressed/origin.
        logger.info(f"Compression successful. Original: {original_tokens}, Compressed: {compressed_tokens}, Achieved Ratio (compressed/original): {actual_ratio:.2f}")
        logger.info(f"Original text (first 50): {original_text[:50]}...")
        logger.info(f"Compressed text (first 50): {compressed_text[:50]}...")
        logger.info("=== Compression request completed ===")

        # Note: The CompressionResponse model still uses 'compression_ratio' which was original_tokens / compressed_tokens.
        # We should clarify if the UI expects this or the achieved_ratio (compressed/original).
        # For now, let's stick to the existing CompressionResponse structure.
        # The llmlingua dict has 'ratio' (orig/comp) and 'rate' (comp/orig in %).
        # Our response has 'compression_ratio'. Let's align this to be achieved_ratio (compressed/original)

        # Let's redefine compression_ratio in the response to be actual_ratio (compressed/original)
        # If a different definition (like llmlingua's 1/rate) is needed, this would change.
        response_compression_ratio = actual_ratio

        return CompressionResponse(
            original_text=original_text,
            compressed_text=compressed_text,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=response_compression_ratio
        )

    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error in compression endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process compression request: {str(e)}")

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "service": "tokenflow-python"}

def start():
    """Start the FastAPI application."""
    import uvicorn

    try:
        logger.info("Starting TokenFlow backend...")

        # Create thread pool
        global thread_pool
        thread_pool = ThreadPoolExecutor(max_workers=1)
        logger.info("Thread pool initialized")

        # Register cleanup handler
        atexit.register(cleanup_resources)

        logger.info("Starting FastAPI server...")
        # Start the server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8001,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Failed to start TokenFlow backend: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    start()
