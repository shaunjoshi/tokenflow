from fastapi import APIRouter, Depends, Request, HTTPException
from typing import Dict, Any, AsyncGenerator
from sse_starlette.sse import EventSourceResponse
from backend.models.schemas import ModelSelectionRequest, ModelSelectionResponse
from backend.models.rankings import MODEL_RANKINGS
from backend.core.dependencies import get_bart_classifier, get_groq_client
from backend.services.model_service import select_model_for_category
import json
import logging

router = APIRouter()
log = logging.getLogger(__name__)

@router.post("/models/select")
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
    log.info(
        f"Categories: {categories}, Temp: {temperature}, Top-P: {top_p}, Max Tokens: {max_tokens}"
    )

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
                for label, score in zip(
                    classification_result["labels"], classification_result["scores"]
                )
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
            yield {"event": "metadata", "data": json.dumps(metadata)}
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
                    yield {"event": "text_chunk", "data": content}
            log.info(
                f"Finished iterating through Groq stream after {chunk_count} chunks."
            )

            log.info("Yielding end_stream event")
            yield {"event": "end_stream", "data": "Stream finished"}
            log.info("Finished streaming and sent end event.")

        except Exception as e:
            log.error(f"Error during stream generation: {e}", exc_info=True)
            error_data = {
                "error": "An error occurred during processing.",
                "detail": str(e),
            }
            log.info("Yielding error event")
            yield {"event": "error", "data": json.dumps(error_data)}

    return EventSourceResponse(event_generator())


@router.get("/model-rankings")
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
            status_code=500, detail="Failed to retrieve model rankings data"
        ) 