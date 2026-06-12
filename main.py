"""
SHL Assessment Recommender — FastAPI entry point.

Fix for Render deployment:
  - Catalog loads synchronously (fast, <1s)
  - Vector store loads in a background thread so uvicorn binds the port
    immediately and Render does not time out waiting for an open port.
  - /health returns {"status": "ok"} once the vector store is ready,
    or {"status": "loading"} with HTTP 503 while it is still warming up.
    Render allows up to 2 minutes for /health — this covers cold starts.
"""

from dotenv import load_dotenv
load_dotenv()

import os
import time
import logging
import threading
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

# Shared flag — set to True once the vector store is fully loaded
_vector_store_ready = threading.Event()


def _load_vector_store(app: FastAPI, catalog: CatalogStore) -> None:
    """Runs in a background thread. Sets the ready event when done."""
    try:
        logger.info("Background: loading / building vector index …")
        t0 = time.time()
        vector_store = VectorStore(catalog)
        vector_store.load_or_build()
        app.state.vector_store = vector_store
        logger.info("Background: vector store ready in %.1fs", time.time() - t0)
    except Exception as e:
        logger.critical("Background: vector store failed to load: %s", e)
    finally:
        _vector_store_ready.set()   # unblock /health even on failure


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()

    # 1. Catalog — fast (<1s), load synchronously
    logger.info("Loading SHL catalog …")
    catalog = CatalogStore()
    catalog.load()
    app.state.catalog = catalog

    # 2. Vector store — slow (model download + FAISS), load in background
    #    so uvicorn can bind the port before Render times out
    thread = threading.Thread(
        target=_load_vector_store,
        args=(app, catalog),
        daemon=True,
        name="vector-store-loader",
    )
    thread.start()

    logger.info("Startup skeleton complete in %.1fs — vector store loading in background", time.time() - t0)
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