import pickle
from pathlib import Path

import faiss
import numpy as np
from utils.embedding import get_embedding
from utils.chunking import chunk_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FAISS_STORE = PROJECT_ROOT / "faiss_store"
INDEX_PATH = FAISS_STORE / "chunk_index.faiss"
MAPPING_PATH = FAISS_STORE / "chunk_mapping.pkl"
DOCUMENT_PATH = PROJECT_ROOT / "data" / "diabetes.txt"


def load_faiss_index(EMBEDDINGS_MODEL, EURI_API_KEY, EURI_BASE_URL):
    if INDEX_PATH.exists():
        index = faiss.read_index(str(INDEX_PATH))
        with MAPPING_PATH.open("rb") as f:
            chunk_mapping = pickle.load(f)
    else:
        with DOCUMENT_PATH.open("r", encoding="utf-8") as f:
            text = f.read()
        chunks = chunk_text(text)
        embeddings = [get_embedding(chunk, EMBEDDINGS_MODEL, EURI_API_KEY, EURI_BASE_URL) for chunk in chunks]
        dim = len(embeddings[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings).astype('float32'))
        chunk_mapping = {i: chunk for i, chunk in enumerate(chunks)}
        FAISS_STORE.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(INDEX_PATH))
        with MAPPING_PATH.open("wb") as f:
            pickle.dump(chunk_mapping, f)
    return index, chunk_mapping