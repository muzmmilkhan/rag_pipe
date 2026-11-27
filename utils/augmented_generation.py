from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage
from utils.retrieval import load_faiss_index
import numpy as np
from utils.embedding import get_embedding




def generate_augmented_response(query, LLM_MODEL, EMBEDDINGS_MODEL, EURI_API_KEY, EURI_BASE_URL, TOP_K, TEMPERATURE):
    faiss_index, chunk_mapping = load_faiss_index(EMBEDDINGS_MODEL, EURI_API_KEY, EURI_BASE_URL)
    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=EURI_API_KEY,
        base_url=EURI_BASE_URL,
        temperature=TEMPERATURE
    )
    query_embedding = get_embedding(query, EMBEDDINGS_MODEL, EURI_API_KEY, EURI_BASE_URL)
    D, I = faiss_index.search(np.array([query_embedding]).astype('float32'), k=TOP_K)
    top_chunks = [chunk_mapping[i] for i in I[0]]
    llm_response = llm.invoke([
        SystemMessage("Use the following context to answer the question. if the context does not contain the answer, say 'I don't know'."),
        HumanMessage(f"Context: {' '.join(top_chunks)}\n\nQuestion: {query}")
    ])
    return llm_response.content, top_chunks




