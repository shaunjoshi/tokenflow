from pydantic import BaseModel as PydanticBaseModel, Field
from typing import Dict, List
from backend.models.rankings import DEFAULT_MODEL_CATEGORIES

class ModelSelectionRequest(PydanticBaseModel):
    prompt: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="The prompt text to classify and process",
    )
    possible_categories: list[str] = Field(
        default=DEFAULT_MODEL_CATEGORIES,
        description="Categories to classify the prompt against",
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Temperature for model generation"
    )
    top_p: float = Field(
        default=0.9, ge=0.0, le=1.0, description="Top-p for model generation"
    )
    max_tokens: int = Field(default=250, gt=0, description="Maximum tokens to generate")

class ModelSelectionResponse(PydanticBaseModel):
    selected_model: str
    prompt_category: str
    confidence_score: float
    all_categories: Dict[str, float]
    completion: str

class ClassificationRequest(PydanticBaseModel):
    prompt: str = Field(
        ..., min_length=5, max_length=2000, description="The prompt text to classify"
    )
    possible_categories: list[str] = Field(
        default=DEFAULT_MODEL_CATEGORIES,
        description="Categories to classify the prompt against",
    )
    multi_label: bool = Field(
        default=False, description="Whether to allow multiple category labels"
    )

class ClassificationResponse(PydanticBaseModel):
    top_category: str
    confidence_score: float
    all_categories: Dict[str, float]
    recommended_model: str

class CompressionRequest(PydanticBaseModel):
    text: str = Field(..., min_length=10, description="The text to compress")
    target_token: int = Field(
        default=100, gt=0, description="Target number of tokens after compression"
    )

class CompressionResponse(PydanticBaseModel):
    original_text: str
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float

class GenerateRequest(PydanticBaseModel):
    prompt: str = Field(..., min_length=1, description="The prompt to send to the LLM")
    model: str = Field(
        ...,
        description="The specific OpenRouter model ID to use (e.g., 'anthropic/claude-3-sonnet:beta')",
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Temperature for model generation"
    )
    top_p: float = Field(
        default=0.9, ge=0.0, le=1.0, description="Top-p for model generation"
    )
    max_tokens: int = Field(default=250, gt=0, description="Maximum tokens to generate") 