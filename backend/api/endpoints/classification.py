from fastapi import APIRouter, Depends, HTTPException
from backend.models.schemas import ClassificationRequest, ClassificationResponse
from backend.core.dependencies import get_bart_classifier
import logging

router = APIRouter()
log = logging.getLogger(__name__)

@router.post("/classify", response_model=ClassificationResponse)
async def classify_prompt(
    request: ClassificationRequest, bart_classifier=Depends(get_bart_classifier)
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
            prompt, categories, multi_label=request.multi_label
        )

        top_category = classification_result["labels"][0]
        top_score = classification_result["scores"][0]

        log.info(
            f"Prompt classified as '{top_category}' with confidence {top_score:.2f}"
        )

        all_categories = {
            label: score
            for label, score in zip(
                classification_result["labels"], classification_result["scores"]
            )
        }

        # select_model_for_category logic can be imported from a service
        recommended_model = top_category  # Placeholder, replace with actual logic

        return ClassificationResponse(
            top_category=top_category,
            confidence_score=top_score,
            all_categories=all_categories,
            recommended_model=recommended_model,
        )

    except Exception as e:
        log.error(f"Error in classification endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process classification request: {str(e)}",
        ) 