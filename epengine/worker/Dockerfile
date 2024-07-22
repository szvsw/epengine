# syntax=docker/dockerfile:1

FROM python:3.10-slim-buster

ENV POETRY_VERSION=1.8 \
    POETRY_VIRTUALENVS_CREATE=false

ENV TRAVIS_OS_NAME=linux
ENV ENERGYPLUS_VERSION=22.2.0
ENV ENERGYPLUS_SHA=c249759bad
ENV ENERGPYPLUS_TAG=v22.2.0
ENV ENERGYPLUS_INSTALL_VERSION=22-2-0

RUN apt-get update \
    && apt-get install -y --no-install-recommends wget bzip2 ca-certificates curl git libxml2-dev sudo tk python3-tk libgomp1 \
    && git clone https://github.com/samuelduchesne/archetypal.git \
    && cd /archetypal \
    && sudo chmod +x install_energyplus.sh  \
    && ./install_energyplus.sh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install "poetry==$POETRY_VERSION"

# Copy only requirements to cache them in docker layer
WORKDIR /code
COPY poetry.lock pyproject.toml /code/

# Project initialization:
RUN poetry install --no-interaction --no-ansi --no-root --with worker

# Copy Python code to the Docker image
COPY epengine /code/epengine/

ENV PYTHONPATH=/code

# ENTRYPOINT [ "poetry", "run", "worker" ]
CMD [ "python", "epengine/worker/main.py"]
