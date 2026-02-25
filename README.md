Local Hybrid RAG System (Fully Offline)

A fast, private, production-style Retrieval-Augmented Generation (RAG) system built with:

FAISS (Dense Vector Search)

BM25 (Sparse Retrieval)

CrossEncoder Re-ranking

Ollama Local LLM

MongoDB Chunk Storage

Smart Sentence + Table Chunking

Streamlit UI with Streaming Responses

🚀 Features

Hybrid Retrieval (Dense + Sparse)

Table-aware extraction

Numeric query boosting

CrossEncoder reranking

Dynamic chunk expansion

Answer verification layer

Streaming token responses

Fully offline (no OpenAI)


🧠 Architecture

PDF → Smart Chunking → Embeddings → FAISS + BM25 →
Hybrid Retrieval → Reranking → LLM → Streaming Answer


⚙️ Run Locally
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
