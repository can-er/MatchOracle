# syntax=docker/dockerfile:1
# MatchOracle application image — Python 3.12 + uv. The app runs as a container
# alongside the Postgres and Redis service containers (see docker-compose.yml).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# uv binary from the official distroless image.
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /bin/uv

WORKDIR /app

# Dependency layer — cached unless lockfile/pyproject change. README.md is
# required because pyproject's `readme = "README.md"` is read during the build.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Application source + Alembic migrations, then install the project itself.
COPY app ./app
COPY config ./config
COPY alembic ./alembic
COPY alembic.ini ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

EXPOSE 8000
# Apply Alembic migrations, then start the server — schema stays Alembic-managed.
CMD ["sh", "-c", "uv run --no-dev alembic upgrade head && uv run --no-dev uvicorn app.main:app --host 0.0.0.0 --port 8000"]
