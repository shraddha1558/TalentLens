"""
API routes.

/health — returns 200 {"status":"ok"} once vector store is ready,
          503 {"status":"loading"} while still warming up.
          Render polls this and allows up to 2 minutes, so cold starts
          are fully covered.

/chat   — waits up to 90s for vector store before accepting requests,
          returns 503 if it never becomes ready.
"""

import logging
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app.models.request import ChatRequest
from app.models.response import AgentResponse
from app.agent.orchestrator import Orchestrator

# Import the shared ready-event from main
from main import _vector_store_ready

logger = logging.getLogger(__name__)
router = APIRouter()

# How long /chat will wait for the vector store on a cold start (seconds)
_READY_TIMEOUT = 90


def _get_orchestrator(request: Request) -> Orchestrator:
    catalog      = request.app.state.catalog
    vector_store = request.app.state.vector_store
    return Orchestrator(catalog, vector_store)


@router.get("/health")
async def health(request: Request):
    if _vector_store_ready.is_set():
        return {"status": "ok"}
    # Still loading — return 503 so Render keeps polling
    return JSONResponse(status_code=503, content={"status": "loading"})


@router.post("/chat", response_model=AgentResponse)
async def chat(body: ChatRequest, request: Request):
    # Block until vector store is ready (up to _READY_TIMEOUT seconds)
    ready = _vector_store_ready.wait(timeout=_READY_TIMEOUT)
    if not ready:
        raise HTTPException(
            status_code=503,
            detail="Service is still warming up. Please retry in a moment.",
        )

    if not hasattr(request.app.state, "vector_store"):
        raise HTTPException(
            status_code=503,
            detail="Vector store unavailable. Please retry.",
        )

    orchestrator = _get_orchestrator(request)
    try:
        response = await orchestrator.run(body.messages)
        return response
    except Exception as e:
        logger.exception("Orchestrator error: %s", e)
        raise HTTPException(status_code=500, detail="Internal agent error.")