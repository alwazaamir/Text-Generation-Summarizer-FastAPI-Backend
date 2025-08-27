"""
API route definitions.

The routes defined here expose HTTP endpoints for interacting with the AI
services.  Each route delegates to a service class to perform business logic
such as text generation or summarisation.  Using a separate router makes it
easy to extend the API surface without cluttering the application
configuration in ``main.py``.
"""

from fastapi import APIRouter, HTTPException

from ..models.schemas import (
    GenerateRequest,
    GenerateResponse,
    SummarizeRequest,
    SummarizeResponse,
)
from ..services.text_generation import text_generator, text_summarizer

# Create a router instance.  Prefixes and tags could be added here to
# group endpoints logically.
router = APIRouter()


@router.post("/generate", response_model=GenerateResponse, summary="Generate text")
async def generate_text(payload: GenerateRequest) -> GenerateResponse:
    """Generate a continuation of the given prompt.

    Parameters
    ----------
    payload: GenerateRequest
        The request body containing the initial prompt and desired output length.

    Returns
    -------
    GenerateResponse
        A JSON object with the generated text.
    """
    if not payload.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt must not be empty.")
    generated = text_generator.generate(prompt=payload.prompt, max_length=payload.max_length)
    return GenerateResponse(generated_text=generated)


@router.post("/summarize", response_model=SummarizeResponse, summary="Summarize text")
async def summarize_text(payload: SummarizeRequest) -> SummarizeResponse:
    """Summarise the provided text.

    Parameters
    ----------
    payload: SummarizeRequest
        The request body containing the source text and desired summary length.

    Returns
    -------
    SummarizeResponse
        A JSON object with the summary.
    """
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty.")
    summary = text_summarizer.summarize(text=payload.text, max_sentences=payload.max_sentences)
    return SummarizeResponse(summary=summary)
