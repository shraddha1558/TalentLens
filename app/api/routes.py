import logging
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app.models.request import ChatRequest
from app.models.response import AgentResponse
from app.agent.orchestrator import Orchestrator
from app.state import vector_store_ready   # no circular import

logger = logging.getLogger(__name__)
router = APIRouter()

_READY_TIMEOUT = 90   # seconds /chat will wait on cold start


@router.get("/health")
async def health(request: Request):
    if vector_store_ready.is_set():
        return {"status": "ok"}
    return JSONResponse(status_code=503, content={"status": "loading"})


@router.post("/chat", response_model=AgentResponse)
async def chat(body: ChatRequest, request: Request):
    if not vector_store_ready.wait(timeout=_READY_TIMEOUT):
        raise HTTPException(status_code=503, detail="Service warming up. Retry shortly.")

    if not hasattr(request.app.state, "vector_store"):
        raise HTTPException(status_code=503, detail="Vector store unavailable.")

    orchestrator = Orchestrator(
        request.app.state.catalog,
        request.app.state.vector_store,
    )
    try:
        return await orchestrator.run(body.messages)
    except Exception as e:
        logger.exception("Orchestrator error: %s", e)
        raise HTTPException(status_code=500, detail="Internal agent error.")