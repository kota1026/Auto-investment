# Auto-Investment Docker image
#
# Builds a self-contained image with the FastAPI server, all strategies,
# the Alpha Arena simulator, and the lightweight-charts dashboard.
#
# Build:
#   docker build -t auto-investment .
#
# Run (with API key from your environment):
#   docker run --rm -p 8000:8000 \
#     -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
#     auto-investment
#
# Run a one-shot contest instead of the server:
#   docker run --rm \
#     -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
#     -e CONTEST_CAPITAL=30 \
#     -e CONTEST_DAYS=30 \
#     -v $(pwd)/contest_results:/app/contest_results \
#     auto-investment \
#     python scripts/run_contest_path_a.py

FROM python:3.11-slim

# System deps for some pandas/numpy wheels and TLS for httpx
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cache pip layer
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# App code
COPY src/ ./src/
COPY web/ ./web/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY pyproject.toml ./

ENV PYTHONPATH=/app/src
ENV HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000

# Default: run the FastAPI server. Override the CMD to run a contest one-shot.
CMD ["uvicorn", "auto_investment.server:app", "--host", "0.0.0.0", "--port", "8000"]
