"""
Shared application state.
Kept in a separate module to avoid circular imports between main.py and routes.py.
"""
import threading

# Set once the vector store finishes loading in the background thread.
vector_store_ready = threading.Event()