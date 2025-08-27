"""
Pydantic schemas for requests and responses.

Defining data models as Pydantic classes provides type validation and
automatic documentation generation through FastAPI.  Each model here
corresponds to the expected shape of JSON payloads for the API endpoints.
"""

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Request payload for generating text."""

    prompt: str = Field(..., description="Prompt to seed the text generator", example="Once upon a time")
    max_length: int = Field(
        50,
        description="Maximum number of words to generate",
        ge=1,
        le=200,
        example=50,
    )


class GenerateResponse(BaseModel):
    """Response payload for generated text."""

    generated_text: str = Field(..., description="The generated text based on the prompt")


class SummarizeRequest(BaseModel):
    """Request payload for summarising text."""

    text: str = Field(..., description="Source text to summarise")
    max_sentences: int = Field(
        3,
        description="Maximum number of sentences to include in the summary",
        ge=1,
        le=10,
        example=3,
    )


class SummarizeResponse(BaseModel):
    """Response payload for summarisation."""

    summary: str = Field(..., description="The summarised text")
