"""
Entry point for the FastAPI application.

This module instantiates the `FastAPI` object, configures middleware
and includes the API routers defined in ``app.api.routes``.  The
application is intentionally thin; all business logic lives in service
classes under the ``app.services`` package.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import routes


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns
    -------
    FastAPI
        Configured FastAPI application ready to run.
    """
    app = FastAPI(
        title="AI Backend Service",
        description=(
            "A lightweight API for text generation and summarisation using simple "
            "algorithms.  Designed to be extended with more sophisticated AI "
            "components (e.g. via LangGraph or LangChain) without changing the API surface."
        ),
        version="0.1.0",
    )

    # Allow cross‑origin requests from any host during development.  Update
    # ``allow_origins`` with specific domains when deploying to production.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API routes
    app.include_router(routes.router)

    return app


# Instantiate the application at module import time.  Uvicorn will look for
# ``app`` at the module level by default.
app = create_app()
