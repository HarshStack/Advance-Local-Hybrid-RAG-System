import os
import time
import streamlit as st
from dotenv import load_dotenv

from ingest_local import build_index
from chat_local import load_index, hybrid_search

from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")


class StreamlitStreamer(BaseCallbackHandler):
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.text = ""

    def on_llm_new_token(self, token, **kwargs):
        self.text += token
        self.placeholder.markdown(self.text + "▌")



@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")



st.set_page_config(page_title="Local RAG System", layout="wide")

st.title("Local RAG System")


uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    os.makedirs("data", exist_ok=True)

    save_path = os.path.join("data", uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF uploaded successfully!")

    if st.button("Build Index"):
        with st.spinner("Building index..."):
            build_index()
        st.success("Index built successfully!")



question = st.text_input("Ask a question about your document:")

if question:

    embed = load_embedder()
    reranker = load_reranker()

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        # ==============================
        # RETRIEVAL PHASE
        # ==============================
        progress = st.progress(0)
        status = st.empty()

        start_retrieval = time.time()

        status.markdown("Retrieving relevant chunks...")
        progress.progress(30)

        index, texts, metadatas, bm25 = load_index()

        docs = hybrid_search(
            question,
            index,
            texts,
            metadatas,
            bm25,
            embed,
            k=10  # SAME as your accurate version
        )

        progress.progress(70)

        # Rerank (same as accurate version)
        pairs = [(question, d["text"]) for d in docs]
        scores = reranker.predict(pairs)

        for i, s in enumerate(scores):
            docs[i]["score"] = float(s)

        docs = sorted(docs, key=lambda x: x["score"], reverse=True)[:3]

        context = "\n\n".join(d["text"] for d in docs)

        progress.progress(100)

        retrieval_time = round(time.time() - start_retrieval, 2)

        progress.empty()
        status.empty()

        
        placeholder = st.empty()
        streamer = StreamlitStreamer(placeholder)

        llm = ChatOllama(
            model=OLLAMA_MODEL,
            streaming=True,
            callbacks=[streamer]
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer only using the context."),
            ("human", "Question: {question}\n\nContext:\n{context}")
        ])

        chain = prompt | llm | StrOutputParser()

        start_llm = time.time()

        chain.invoke({"question": question, "context": context})

        llm_time = round(time.time() - start_llm, 2)

        placeholder.markdown(streamer.text)

        st.markdown("---")
        st.markdown("### Performance Breakdown")
        st.markdown(f"- Retrieval Time: **{retrieval_time} sec**")
        st.markdown(f"- LLM Time: **{llm_time} sec**")
        st.markdown(f"- Total Time: **{retrieval_time + llm_time} sec**")

        
        with st.expander("Retrieved Context"):
            for d in docs:
                st.write(f"Page {d['meta']['page']}")
                st.write(d["text"][:500])
                st.divider()