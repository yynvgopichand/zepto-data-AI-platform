
from pathlib import Path
import os


# Project root directory.
# __file__ = /app/src/config.py in Docker.
# parent.parent = /app
BASE_DIR = Path(__file__).resolve().parent.parent


# Folder containing the 8 Zepto policy documents.
DATA_DIR = BASE_DIR / "docs"


# Folder used by ChromaDB for persistent storage.
CHROMA_DIR = BASE_DIR / "chroma_db"


# Sentence Transformer model.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# Mock mode is enabled unless MOCK_LLM is explicitly set to "0".
MOCK_LLM = os.getenv("MOCK_LLM", "1") != "0"