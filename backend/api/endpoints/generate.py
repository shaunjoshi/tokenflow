from fastapi import APIRouter, Depends, Request, HTTPException
from typing import Dict, Any, AsyncGenerator
from sse_starlette.sse import EventSourceResponse
from backend.models.schemas import GenerateRequest
from backend.core.dependencies import get_groq_client
import json
import logging

router = APIRouter()
log = logging.getLogger(__name__)

@router.post("/generate")
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

    log.info(
        f"Initiating direct generation stream. Model: {model}, Prompt length: {len(prompt)}"
    )
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
                    yield {"event": "text_chunk", "data": content}
            log.info(
                f"Finished iterating through Groq stream after {chunk_count} chunks."
            )

            log.info("Yielding end_stream event")
            yield {"event": "end_stream", "data": "Stream finished"}
            log.info("Finished streaming and sent end event.")

        except Exception as e:
            log.error(f"Error during direct generation stream: {e}", exc_info=True)
            error_data = {
                "error": "An error occurred during generation.",
                "detail": str(e),
            }
            log.info("Yielding error event")
            yield {"event": "error", "data": json.dumps(error_data)}

    return EventSourceResponse(event_generator()) 