from dotenv import load_dotenv
load_dotenv()

import time
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.retrieval.catalog_store import CatalogStore
from app.retrieval.vector_store import VectorStore
from app.state import vector_store_ready   # shared event, no circular import

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def _load_vector_store(app: FastAPI, catalog: CatalogStore) -> None:
    try:
        logger.info("Background: loading vector index …")
        t0 = time.time()
        vs = VectorStore(catalog)
        vs.load_or_build()
        app.state.vector_store = vs
        logger.info("Background: vector store ready in %.1fs", time.time() - t0)
    except Exception as e:
        logger.critical("Background: vector store failed: %s", e)
    finally:
        vector_store_ready.set()   # unblock /health and /chat


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()

    logger.info("Loading SHL catalog …")
    catalog = CatalogStore()
    catalog.load()
    app.state.catalog = catalog

    # Vector store loads in background — uvicorn binds port immediately
    threading.Thread(
        target=_load_vector_store,
        args=(app, catalog),
        daemon=True,
        name="vector-store-loader",
    ).start()

    logger.info("Startup complete in %.1fs — vector store loading in background", time.time() - t0)
    yield
    logger.info("Shutting down.")


app = FastAPI(title="SHL Assessment Recommender", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)