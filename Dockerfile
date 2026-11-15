# Latest uv (0.13.8) with Python 3.14 on Debian Trixie slim
# https://docs.astral.sh/uv/guides/integration/docker/
FROM ghcr.io/astral-sh/uv:0.13.8-python3.14-trixie-slim

WORKDIR /app

# Compile bytecode for faster startup; copy mode for reliable Docker layers
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1

# Install dependencies first for better layer caching
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project

# Copy application and sync the project into the venv
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

# Train the model so model.bin exists in the image
RUN uv run python src/train.py

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 9696

ENTRYPOINT ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "9696"]
