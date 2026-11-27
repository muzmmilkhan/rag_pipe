FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}" \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN pip install --no-cache-dir uv \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /app \
    && chown app:app /app

WORKDIR /app

COPY --chown=app:app pyproject.toml uv.lock README.md ./
USER app
RUN uv sync --frozen --no-dev --no-install-project

COPY --chown=app:app main.py ./
COPY --chown=app:app utils/ ./utils/
COPY --chown=app:app data/ ./data/
COPY --chown=app:app faiss_store/ ./faiss_store/

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.address=0.0.0.0", "--server.port=8501"]
