"""
EA AI Assistant — Streamlit UI
Run with: streamlit run app.py
"""

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag.pipeline import ingest_document, query

load_dotenv()

st.set_page_config(
    page_title="EA AI Assistant",
    page_icon="🏗️",
    layout="wide",
)

st.title("🏗️ EA AI Assistant")
st.caption("On-premise AI assistant for Enterprise Architects · Powered by RAG")

with st.sidebar:
    st.header("📄 Documents")
    uploaded = st.file_uploader(
        "Upload a document (.txt or .md)",
        type=["txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded:
        for file in uploaded:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(file.name).suffix, mode="wb"
            ) as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name

            with st.spinner(f"Ingesting {file.name}..."):
                n = ingest_document(tmp_path)
            os.unlink(tmp_path)
            st.success(f"✓ {file.name} — {n} chunks indexed")

    st.divider()
    st.markdown("**Stack**")
    st.code("OpenAI · ChromaDB · Streamlit", language=None)
    st.markdown("*Swap OpenAI → Mistral 7B for full on-premise*")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about your EA documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            answer = query(prompt)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})