# rag-pipe

A small retrieval-augmented generation (RAG) example in Python. It reads a
local text document, splits it into chunks, embeds the chunks, retrieves the
most relevant chunks with FAISS, and asks a chat model to answer a question
using that retrieved context.

The included example uses [`diabetes.txt`](./diabetes.txt) as its document.

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- An API endpoint and credentials compatible with the embeddings and chat
  interfaces used by LangChain

## Setup

Install the project dependencies:

```bash
uv sync
```

Create a `.env` file in the project directory and set the following values:

```dotenv
EURI_API_KEY=your_api_key
EURI_BASE_URL=https://your-api-base-url
EMBEDDINGS_MODEL=your-embeddings-model
CHAT_MODEL=your-chat-model
```

Use the base URL and model names provided by your API provider. Keep your API
key private and do not commit `.env` to version control.

## Run

```bash
uv run python main.py
```

The script loads `diabetes.txt`, builds an in-memory FAISS index, asks
“What is the main topic of the document?”, and prints the model's response.

## How it works

1. Loads the document as UTF-8 text.
2. Splits it into consecutive 100-character chunks.
3. Creates an embedding for each chunk.
4. Stores the embeddings in a FAISS `IndexFlatL2` index.
5. Embeds the question and retrieves the five nearest chunks.
6. Sends the retrieved context and question to the chat model.

The index is rebuilt each time the script runs; it is not persisted to disk.
The document path and example question are currently set in `main.py`.