import os
import certifi
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING
from pymongo.errors import ServerSelectionTimeoutError

load_dotenv()

URI = os.getenv("MONGODB_URI", "").strip()
DB_NAME = os.getenv("MONGODB_DB", "rag")
CHUNKS_COL = os.getenv("MONGODB_CHUNKS_COLLECTION", "chunks")

_client = None
_db = None
_chunks = None


def _build_client():
    """Create a MongoClient instance with safe fallback handling."""
    if not URI:
        print(" No MONGODB_URI provided; MongoDB disabled.")
        return None

    try:
        # Handle  connection
        if "mongodb+srv://" in URI:
            client = MongoClient(
                URI,
                tlsCAFile=certifi.where(),
                serverSelectionTimeoutMS=5000
            )
        else:
            # Local MongoDB
            client = MongoClient(
                URI,
                serverSelectionTimeoutMS=5000
            )

        # Verify connection
        client.admin.command("ping")
        return client

    except ServerSelectionTimeoutError:
        print(" MongoDB not reachable; continuing without database caching.")
        return None


def get_db():
    """Return cached DB handle, create if needed."""
    global _client, _db
    if _db is not None:
        return _db

    _client = _build_client()
    if _client is None:
        return None

    _db = _client[DB_NAME]
    return _db


def get_chunks_collection():
    """Return cached 'chunks' collection, create if needed."""
    global _chunks
    if _chunks is not None:
        return _chunks

    db = get_db()
    if db is None:
        return None

    _chunks = db[CHUNKS_COL]
    return _chunks


def ensure_indexes():
    """Ensure indexes exist on chunks collection."""
    col = get_chunks_collection()
    if col is None:
        return

    col.create_index(
        [
            ("source", ASCENDING),
            ("page", ASCENDING),
            ("chunk_index", ASCENDING),
            ("file_hash", ASCENDING),
        ],
        unique=True,
        name="chunk_identity"
    )

    col.create_index([("source", ASCENDING)], name="by_source")
    col.create_index([("file_hash", ASCENDING)], name="by_file_hash")


def now_utc():
    """Return a timezone‑agnostic UTC timestamp."""
    return datetime.utcnow()