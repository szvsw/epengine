FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.6.16 /uv /bin/uv

# Copy only requirements to cache them in docker layer
WORKDIR /code
COPY uv.lock pyproject.toml README.md /code/

# Project initialization:
RUN uv sync --locked --no-install-project

# Copy Python code to the Docker image
COPY epengine /code/epengine/

ENTRYPOINT ["uv", "run", "python", "epengine/initialize/main.py"]
