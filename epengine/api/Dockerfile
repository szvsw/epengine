FROM python:3.12-slim-bookworm

ARG POETRY_VERSION=2.1.2
ENV POETRY_VERSION=$POETRY_VERSION \
    POETRY_VIRTUALENVS_CREATE=false

# Install poetry
RUN pip install "poetry==$POETRY_VERSION"

# Copy only requirements to cache them in docker layer
WORKDIR /code
COPY poetry.lock pyproject.toml /code/

# Project initialization:
RUN poetry config virtualenvs.in-project true && poetry install --no-interaction --no-ansi --no-root --only main,api

RUN epinterface prisma generate

# Copy Python code to the Docker image
COPY epengine /code/epengine/

ENV PYTHONPATH=/code

EXPOSE 80

ENTRYPOINT ["uvicorn", "epengine.api.main:api", "--host", "0.0.0.0", "--port", "80"]
