import os

import streamlit as st
from dotenv import load_dotenv
from utils.augmented_generation import generate_augmented_response


load_dotenv()
EURI_API_KEY = os.getenv("EURI_API_KEY")
EURI_BASE_URL = os.getenv("EURI_BASE_URL")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL")
LLM_MODEL = os.getenv("LLM_MODEL")
TOP_K = int(os.getenv("TOP_K", 5))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))

st.title("RAG APP - Diabetes Knowledge Base")
st.write("Ask a question about diabetes")

query = st.text_input("Your question:")
if query:
    response, top_chunks = generate_augmented_response(
        query,
        LLM_MODEL,
        EMBEDDINGS_MODEL,
        EURI_API_KEY,
        EURI_BASE_URL,
        TOP_K,
        TEMPERATURE
    )
    st.subheader("Answer")
    st.write(response)

    with st.expander("Retrieved Chunks"):
        for chunk in top_chunks:
            st.markdown(f" - {chunk}")

