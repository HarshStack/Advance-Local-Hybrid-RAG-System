import os
import pickle
import numpy as np
import faiss
import fitz
import nltk
import pdfplumber
from nltk.tokenize import sent_tokenize
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pymongo import UpdateOne
from db import get_chunks_collection, ensure_indexes, now_utc


load_dotenv()

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


import pdfplumber

def load_pdfs(folder="data"):
    docs = []

    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            path = os.path.join(folder, file)

            with pdfplumber.open(path) as pdf:
                for page_num, page in enumerate(pdf.pages):

                    text = page.extract_text() or ""
                    tables = page.extract_tables()

                    table_chunks = []

                    for table in tables:
                        if not table:
                            continue

                        headers = table[0]
                        for row in table[1:]:
                            row_data = []
                            for h, cell in zip(headers, row):
                                if h and cell:
                                    row_data.append(f"{h.strip()}: {cell.strip()}")

                            if row_data:
                                table_chunks.append(" | ".join(row_data))

                    docs.append({
                        "text": text,
                        "table_chunks": table_chunks,
                        "source": file,
                        "page": page_num
                    })

    return docs

from nltk.tokenize import sent_tokenize

def split_text_smart(text, chunk_size=800):
    """
    Sentence-aware chunking for normal text.
    Avoids breaking sentences.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) < chunk_size:
            current += sentence + " "
        else:
            if current.strip():
                chunks.append(current.strip())
            current = sentence + " "

    if current.strip():
        chunks.append(current.strip())

    return chunks

import re

def split_text(text, max_chars=800):
    """
    Sentence-aware chunking.
    Groups sentences until max_chars is reached.
    Prevents mid-sentence cuts.
    """

    if not text:
        return []

    # Simple sentence splitting (no NLTK dependency)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += " " + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def build_index():
    os.makedirs("faiss_index", exist_ok=True)

    raw_docs = load_pdfs()
    all_chunks = []
    metadata = []

    for doc in raw_docs:        # TEXT CHUNKING
        parts = split_text(doc["text"], CHUNK_SIZE)

        for idx, p in enumerate(parts):
            all_chunks.append(p)
            metadata.append({
                "source": doc["source"],
                "page": doc["page"],
                "chunk_index": idx,
                "chunk_type": "text"
            })

        #  TABLE CHUNKING
        for t_idx, row in enumerate(doc.get("table_chunks", [])):
            all_chunks.append(row)
            metadata.append({
                "source": doc["source"],
                "page": doc["page"],
                "chunk_index": 10000 + t_idx,  # separate range from text chunks
                "chunk_type": "table"
            })

    print(f"PDF pages loaded: {len(raw_docs)}")
    print(f"Total text chunks: {len(all_chunks)}")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = embedder.encode(all_chunks, convert_to_numpy=True).astype("float32")

    # MONGO 
    col = get_chunks_collection()

    if col is not None:
        ensure_indexes()
        ops = []

        for text, meta, vec in zip(all_chunks, metadata, vectors):
            selector = {
                "source": meta["source"],
                "page": meta["page"],
                "chunk_index": meta["chunk_index"]
            }

            doc = {
                **selector,
                "text": text,
                "embedding": vec.tolist(),
                "created_at": now_utc()
            }

            ops.append(UpdateOne(selector, {"$set": doc}, upsert=True))

        col.bulk_write(ops)
        print(f"✔ Stored {len(ops)} chunks in MongoDB")

    else:
        print(" MongoDB not available. Skipping storage.")

    #  FAISS 
    dim = vectors.shape[1]
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    faiss.write_index(index, "faiss_index/index.faiss")

    #  BM25
    tokenized = [t.split() for t in all_chunks]
    bm25 = BM25Okapi(tokenized)

    with open("faiss_index/metadata.pkl", "wb") as f:
        pickle.dump({"chunks": all_chunks, "meta": metadata, "bm25": bm25}, f)

    print(" FAISS + BM25 index built successfully!")


if __name__ == "__main__":
    build_index()