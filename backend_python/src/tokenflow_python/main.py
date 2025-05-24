import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TRANSFORMERS_FORCE_CPU"] = "1"
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Optional
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from llmlingua import PromptCompressor
from dotenv import load_dotenv
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import signal
import sys
import atexit

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

# Maximum text length (in characters) to prevent memory issues
MAX_TEXT_LENGTH = 2000

def get_model():
    """Lazy load the BART model."""
    global model, tokenizer
    if model is None or tokenizer is None:
        logger.info("Loading BART model and tokenizer...")
        model_name = "facebook/bart-large-mnli"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        logger.info("BART model loaded successfully")
    return model, tokenizer

def get_compressor():
    """Lazy load the LLMLingua compressor."""
    global compressor
    if compressor is None:
        try:
            logger.info("Initializing LLMLingua compressor...")
            logger.info("Available models:")
            logger.info("- microsoft/llmlingua-2-xlm-roberta-large-meetingbank")
            logger.info("Initializing with device_map='cpu'")
            
            compressor = PromptCompressor(
                model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                device_map="cpu"
            )
            logger.info("LLMLingua compressor initialized successfully using CPU.")
        except Exception as e:
            logger.error(f"Failed to initialize LLMLingua compressor: {e}", exc_info=True)
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error args: {e.args}")
            compressor = None
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
    target_token: int = Field(..., gt=0, description="Target number of tokens after compression")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "This is a test prompt that we want to compress.",
                "target_token": 20
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
        # Get model and tokenizer
        model, tokenizer = get_model()
        
        # Prepare the input for BART
        inputs = tokenizer(
            request.prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        # Get model predictions
        with torch.no_grad():
            outputs = model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1)

        # Map scores to categories
        category_scores = {}
        for i, category in enumerate(request.possible_categories):
            category_scores[category] = float(scores[0][i])

        # Find top category
        top_category = max(category_scores.items(), key=lambda x: x[1])

        # Map category to recommended model
        model_mapping = {
            "reasoning": "llama-3.1-8b-instant",
            "function-calling": "llama-3.1-8b-instant",
            "text-to-text": "llama-3.1-8b-instant"
        }
        recommended_model = model_mapping.get(top_category[0], "llama-3.1-8b-instant")

        return ClassificationResponse(
            top_category=top_category[0],
            confidence_score=top_category[1],
            all_categories=category_scores,
            recommended_model=recommended_model
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compress", response_model=CompressionResponse)
async def compress_text(request: CompressionRequest):
    """Compresses the input text using LLMLingua."""
    try:
        logger.info("=== Starting compression request ===")
        logger.info(f"Received compression request. Target tokens: {request.target_token}")
        logger.info(f"Request text length: {len(request.text)} characters")
        
        if len(request.text) > MAX_TEXT_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Text is too long. Maximum length is {MAX_TEXT_LENGTH} characters."
            )
        
        logger.info("Getting compressor...")
        compressor = get_compressor()
        if compressor is None:
            logger.error("Compressor is None after get_compressor() call")
            raise HTTPException(
                status_code=503,
                detail="Compression service is not available. LLMLingua compressor failed to initialize."
            )
        logger.info("Compressor obtained successfully")
        
        # Get model and tokenizer for token counting
        logger.info("Getting model and tokenizer...")
        _, tokenizer = get_model()
        logger.info("Model and tokenizer obtained successfully")
        
        # Log the parameters we're about to use
        logger.info("Attempting compression with parameters:")
        logger.info(f"- Text length: {len(request.text)} characters")
        logger.info(f"- Target tokens: {request.target_token}")
        
        try:
            logger.info("Starting compression in thread pool...")
            # Try with minimal parameters first
            result = await asyncio.to_thread(
                compressor.compress_prompt,
                [request.text],
                target_token=request.target_token
            )
            logger.info(f"Compression result: {result}")
        except Exception as e:
            logger.error(f"Error during compression: {str(e)}", exc_info=True)
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error args: {e.args}")
            raise HTTPException(
                status_code=500,
                detail=f"Error during compression: {str(e)}"
            )
        
        original_text = request.text
        compressed_text = result.get("compressed_prompt", "")
        original_tokens = result.get("origin_tokens", 0)
        compressed_tokens = result.get("compressed_tokens", 0)
        compression_ratio = original_tokens / compressed_tokens if compressed_tokens > 0 else 0
        
        logger.info(f"Compression successful. Original tokens: {original_tokens}, Compressed tokens: {compressed_tokens}")
        logger.info(f"Compressed text: {compressed_text}")
        logger.info("=== Compression request completed ===")

        return CompressionResponse(
            original_text=original_text,
            compressed_text=compressed_text,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compression_ratio
        )
        
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=422,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error in compression endpoint: {e}", exc_info=True)
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error args: {e.args}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process compression request: {str(e)}"
        )

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