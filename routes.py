"""
FastAPI route definitions.
GET  /health  — liveness probe
POST /chat    — stateless conversation turn
"""

import logging
from fastapi import APIRouter, Request, HTTPException

from app.models.request import ChatRequest
from app.models.response import AgentResponse
from app.agent.orchestrator import Orchestrator

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.post("/chat", response_model=AgentResponse)
async def chat(request: Request, body: ChatRequest):
    if not body.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    # Guard: evaluator cap is 8 turns; we don't need to enforce here but log it
    if len(body.messages) > 16:
        logger.warning("Unusually long conversation (%d messages)", len(body.messages))

    catalog = request.app.state.catalog
    vector_store = request.app.state.vector_store

    orchestrator = Orchestrator(catalog=catalog, vector_store=vector_store)
    response = await orchestrator.run(body.messages)
    return response