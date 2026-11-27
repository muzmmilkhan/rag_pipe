import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings


def get_embedding(text, EMBEDDINGS_MODEL, EURI_API_KEY, EURI_BASE_URL):
    embeddings = OpenAIEmbeddings(
        model=EMBEDDINGS_MODEL,
        api_key=EURI_API_KEY,
        base_url=EURI_BASE_URL
    )
    return embeddings.embed_query(text)