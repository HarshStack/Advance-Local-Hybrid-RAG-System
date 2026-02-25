<h1 align="center"> Advance Local Hybrid RAG System </h1>
<h2>FAISS + BM25 + Sentence Transformers + Re-ranking</h2>

Advance Local Hybrid RAG System 

1. A fast, private, production-style Retrieval-Augmented Generation (RAG) system built with:

- FAISS (Dense Vector Search)

- BM25 (Sparse Retrieval)

- CrossEncoder Re-ranking

- Ollama Local LLM

- MongoDB Chunk Storage

- Smart Sentence + Table Chunking

- Streamlit UI with Streaming Responses

----------------------------------------------------

System Architecture System

<img width="1536" height="1024" alt="ChatGPT Image Feb 25, 2026, 07_44_01 PM" src="https://github.com/user-attachments/assets/233bf03b-a339-47a1-860e-2ec9dca45347" />

----------------------------------------------------

🚀 Features

Hybrid Retrieval (Dense + Sparse)

Table-aware extraction

Numeric query boosting

CrossEncoder reranking

Dynamic chunk expansion

Answer verification layer

Streaming token responses

Fully offline (no OpenAI)

----------------------------------------------------

Architecture

PDF → Smart Chunking → Embeddings → FAISS + BM25 →
Hybrid Retrieval → Reranking → LLM → Streaming Answer

----------------------------------------------------
Performance Metrics and Latency

<img width="454" height="446" alt="image" src="https://github.com/user-attachments/assets/fcc6f319-89ec-4cfe-a817-01edb0234cff" />

----------------------------------------------------

<img width="512" height="372" alt="image" src="https://github.com/user-attachments/assets/fc4b4f32-3b87-47c7-8abe-b649849ba9ac" />

----------------------------------------------------

⚙️ Run Locally
>>python -m venv .venv
>>.venv\Scripts\activate
>>pip install -r requirements.txt
>>streamlit run app.py

----------------------------------------------------







