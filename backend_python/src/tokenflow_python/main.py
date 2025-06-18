import asyncio
import atexit
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TRANSFORMERS_FORCE_CPU"] = "1"

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from llmlingua import PromptCompressor
from pydantic import BaseModel, Field, ValidationError
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_TEXT_LENGTH = 2000
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://192.168.4.206:3000",
    "http://192.168.5.209:3000",
    "http://localhost:8001",
    "http://192.168.5.209:8001",
]

# Global variables
model = None
tokenizer = None
compressor = None
bart_classifier = None
thread_pool = None

# FastAPI app setup
app = FastAPI(title="TokenFlow Python Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
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
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH)
    ratio: float = Field(..., gt=0, le=1.0)

class CompressionResponse(BaseModel):
    original_text: str
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float

# Model initialization functions
def get_classification_model():
    """Lazy load BART model and zero-shot classification pipeline."""
    global model, tokenizer, bart_classifier

    if bart_classifier is None:
        model_name = "facebook/bart-large-mnli"
        logger.info(f"Loading BART model: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        bart_classifier = pipeline(
            "zero-shot-classification",
            model=model,
            tokenizer=tokenizer,
            device="cpu"
        )
        logger.info("BART classification pipeline ready")

    return bart_classifier

def get_compressor():
    """Lazy load LLMLingua compressor."""
    global compressor

    if compressor is None:
        try:
            logger.info("Initializing LLMLingua compressor")
            compressor = PromptCompressor(
                model_name="bert-base-multilingual-cased",
                device_map="cpu",
                use_llmlingua2=True
            )
            logger.info("LLMLingua compressor ready")
        except Exception as e:
            logger.error(f"Failed to initialize compressor: {e}")
            compressor = None

    return compressor

def cleanup_resources():
    """Cleanup function called on shutdown."""
    global model, tokenizer, compressor, thread_pool, bart_classifier

    logger.info("Cleaning up resources...")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if thread_pool:
        thread_pool.shutdown(wait=False)

    # Clear model references
    model = None
    tokenizer = None
    compressor = None
    bart_classifier = None
    thread_pool = None

    logger.info("Cleanup complete")

# API endpoints
@app.post("/classify", response_model=ClassificationResponse)
async def classify_prompt(request: ClassificationRequest):
    try:
        logger.info(f"Classification request: {request.prompt[:50]}...")

        classifier = get_classification_model()
        if classifier is None:
            raise HTTPException(status_code=503, detail="Classification service unavailable")

        # Perform classification
        result = classifier(
            request.prompt,
            request.possible_categories,
            multi_label=request.multi_label
        )

        # Extract results
        top_category = result["labels"][0]
        top_score = float(result["scores"][0])
        all_categories = {
            label: float(score)
            for label, score in zip(result["labels"], result["scores"])
        }

        # Map category to recommended model
        model_mapping = {
            "reasoning": "llama-3.1-8b-instant",
            "function-calling": "llama-3.1-8b-instant",
            "text-to-text": "llama-3.1-8b-instant"
        }
        recommended_model = model_mapping.get(top_category, "llama-3.1-8b-instant")

        return ClassificationResponse(
            top_category=top_category,
            confidence_score=top_score,
            all_categories=all_categories,
            recommended_model=recommended_model
        )

    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compress", response_model=CompressionResponse)
async def compress_text(request: CompressionRequest):
    try:
        logger.info(f"Compression request: ratio={request.ratio}, length={len(request.text)}")

        if len(request.text) > MAX_TEXT_LENGTH:
            raise HTTPException(status_code=400, detail=f"Text too long (max {MAX_TEXT_LENGTH} chars)")

        compressor = get_compressor()
        if compressor is None:
            raise HTTPException(status_code=503, detail="Compression service unavailable")

        # Perform compression
        result = await asyncio.to_thread(
            compressor.compress_prompt,
            context=[request.text],
            rate=request.ratio
        )

        # Extract results
        original_text = request.text
        compressed_prompt = result.get("compressed_prompt", "")

        if isinstance(compressed_prompt, list):
            compressed_text = compressed_prompt[0] if compressed_prompt else ""
        else:
            compressed_text = str(compressed_prompt)

        original_tokens = result.get("origin_tokens", 0)
        compressed_tokens = result.get("compressed_tokens", 0)

        # Calculate compression ratio
        if original_tokens > 0:
            compression_ratio = compressed_tokens / original_tokens
        else:
            compression_ratio = 1.0

        logger.info(f"Compression complete: {original_tokens} -> {compressed_tokens} tokens")

        return CompressionResponse(
            original_text=original_text,
            compressed_text=compressed_text,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compression_ratio
        )

    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Compression error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process compression request: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "tokenflow-python"}

def start():
    """Start the FastAPI application."""
    import uvicorn

    try:
        logger.info("Starting TokenFlow Python backend...")

        # Initialize thread pool
        global thread_pool
        thread_pool = ThreadPoolExecutor(max_workers=1)

        # Register cleanup handler
        atexit.register(cleanup_resources)

        # Start server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8001,
            log_level="info"
        )

    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise

if __name__ == "__main__":
    start()
