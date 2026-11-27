# rag-pipe

A Streamlit retrieval-augmented generation (RAG) app for asking questions
about the diabetes information in [`data/diabetes.txt`](./data/diabetes.txt).
It retrieves relevant text with FAISS and uses a chat model to generate an
answer grounded in that context.

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- API credentials and model names for an endpoint compatible with the
  LangChain OpenAI-compatible embeddings and chat integrations

## Setup

Install dependencies from the project root:

```bash
uv sync
```

Create a `.env` file in the project root:

```dotenv
EURI_API_KEY=your_api_key
EURI_BASE_URL=https://your-api-base-url
EMBEDDINGS_MODEL=your-embeddings-model
LLM_MODEL=your-chat-model

# Optional
TOP_K=5
TEMPERATURE=0.7
```

Use the base URL and model names provided by your API provider. Keep your API
key private; `.env` is ignored by Git and should not be committed.

## Run the app

From the project root, run:

```bash
uv run streamlit run main.py
```

Enter a question in the app to see an answer and, in the **Retrieved Chunks**
expander, the document excerpts used to generate it.

## How it works

1. Splits `data/diabetes.txt` into chunks of up to 100 words.
2. Embeds the chunks and stores them in a FAISS `IndexFlatL2` index.
3. Embeds each question and retrieves the `TOP_K` closest chunks.
4. Sends the question and retrieved context to the chat model.

The FAISS index and its chunk mapping are stored in `faiss_store/`. If they
are not present, the app builds them from the document on its first run.
Project file paths are resolved relative to the project, so you can launch
the app from the repository root without machine-specific absolute paths.
