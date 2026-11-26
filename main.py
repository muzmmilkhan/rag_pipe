import os

import faiss
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


load_dotenv()
embeddings = OpenAIEmbeddings(
    model=os.environ["EMBEDDINGS_MODEL"],
    api_key=os.environ["EURI_API_KEY"],
    base_url=os.environ["EURI_BASE_URL"],
)

llm = ChatOpenAI(
    model=os.environ["CHAT_MODEL"],
    api_key=os.environ["EURI_API_KEY"],
    base_url=os.environ["EURI_BASE_URL"],
    temperature=0.7,
)

def load_doc(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text: str, chunk_size: int = 100) -> list[str]:
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def embed_chunks(chunks: list[str]) -> list[list[float]]:
    return [np.array(embeddings.embed_query(chunk)) for chunk in chunks]

def build_faiss_index(embeddings: list[np.ndarray]) -> faiss.IndexFlatL2:
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.stack(embeddings))
    return index

def search_faiss_index(index: faiss.IndexFlatL2, query_embedding: np.ndarray, k: int = 5) -> list[int]:
    distances, indices = index.search(np.array([query_embedding]), k)
    return indices[0].tolist()

def retrieve_top_k_chunks(chunks: list[str], index: faiss.IndexFlatL2, query: str, k: int = 5) -> list[str]:
    query_embedding = np.array(embeddings.embed_query(query))
    top_k_indices = search_faiss_index(index, query_embedding, k)
    return [chunks[i] for i in top_k_indices]

def answer_query(chunks: list[str], index: faiss.IndexFlatL2, query: str, k: int = 5) -> str:
    top_chunks = retrieve_top_k_chunks(chunks, index, query, k)
    context = "\n".join(top_chunks)
    response = llm.invoke([
        SystemMessage("You are a helpful assistant."),
        HumanMessage(f"Context:\n{context}\n\nQuestion: {query}"),
    ])
    return response.content

if __name__ == "__main__":
    file_path = "diabetes.txt"
    text = load_doc(file_path)
    chunks = chunk_text(text)
    chunk_embeddings = embed_chunks(chunks)
    index = build_faiss_index(chunk_embeddings)

    query = "What is the main topic of the document?"
    answer = answer_query(chunks, index, query)
    print(answer)


