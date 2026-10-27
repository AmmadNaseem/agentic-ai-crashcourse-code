"""Shared model settings and paths for UI, CLI, and ingestion."""
import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
# Project settings take precedence over the repository .env; shell values win.
load_dotenv(PROJECT_DIR / ".env")
load_dotenv(PROJECT_DIR.parent / ".env")

MODEL_NAME = os.getenv("TELECOM_MODEL", "qwen/qwen3.8-27b")
EMBED_MODEL = os.getenv("TELECOM_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MAX_TOKENS = min(int(os.getenv("TELECOM_MAX_TOKENS", "512")),
                 int(os.getenv("TELECOM_OUTPUT_TOKEN_LIMIT", "1000")))
TIMEOUT = float(os.getenv("TELECOM_TIMEOUT", "60"))
MAX_RETRIES = int(os.getenv("TELECOM_MAX_RETRIES", "2"))
if MAX_TOKENS < 1 or TIMEOUT <= 0 or MAX_RETRIES < 0:
    raise ValueError("Token limit and timeout must be positive; retries must be nonnegative.")
CHROMA_DIR = str(PROJECT_DIR / "chroma_store")
DATA_DIR = PROJECT_DIR / "data"
