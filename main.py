"""
SHL Assessment Recommender — FastAPI entry point.

Cold-start note: on first /health the app loads the catalog + embeddings.
GET /health  → {"status": "ok"}
POST /chat   → AgentResponse
"""

import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.retrieval.catalog_store import CatalogStore
from app.retrieval.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy artifacts once at startup."""
    t0 = time.time()
    logger.info("Loading SHL catalog …")
    catalog = CatalogStore()
    catalog.load()
    app.state.catalog = catalog

    logger.info("Building / loading vector index …")
    vector_store = VectorStore(catalog)
    vector_store.load_or_build()
    app.state.vector_store = vector_store

    logger.info("Startup complete in %.1fs", time.time() - t0)
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="SHL Assessment Recommender",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)