FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.6.16 /uv /uvx /bin/

RUN apt-get update && \
    apt-get install --no-install-recommends -y git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy only requirements to cache them in docker layer
WORKDIR /code
COPY uv.lock pyproject.toml README.md /code/

# Project initialization:
# RUN poetry config virtualenvs.in-project true && poetry install --no-interaction --no-ansi --no-root --only main,api
RUN uv sync --locked --no-install-project --group api

RUN uv run --no-sync epinterface prisma generate

# Copy Python code to the Docker image
COPY epengine /code/epengine/

EXPOSE 80

ENTRYPOINT ["uv", "run", "uvicorn", "epengine.api.main:api", "--host", "0.0.0.0", "--port", "80"]
