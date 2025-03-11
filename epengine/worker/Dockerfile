# syntax=docker/dockerfile:1

FROM python:3.10-slim-buster

ENV POETRY_VERSION=1.8 \
    POETRY_VIRTUALENVS_CREATE=false

ENV TRAVIS_OS_NAME=linux

# Build argument for EnergyPlus version
ARG ENERGYPLUS_VERSION=22.2.0

# Copy the versions file first
COPY epengine/worker/energyplus_versions.json /tmp/energyplus_versions.json

# Set EnergyPlus environment variables using jq
RUN apt-get update \
    && apt-get install -y --no-install-recommends jq \
    && export ENERGYPLUS_VERSION=$(jq -r ".[\"$ENERGYPLUS_VERSION\"].version" /tmp/energyplus_versions.json) \
    && export ENERGYPLUS_SHA=$(jq -r ".[\"$ENERGYPLUS_VERSION\"].sha" /tmp/energyplus_versions.json) \
    && export ENERGPYPLUS_TAG=$(jq -r ".[\"$ENERGYPLUS_VERSION\"].tag" /tmp/energyplus_versions.json) \
    && export ENERGYPLUS_INSTALL_VERSION=$(jq -r ".[\"$ENERGYPLUS_VERSION\"].install_version" /tmp/energyplus_versions.json) \
    && apt-get install -y --no-install-recommends wget bzip2 ca-certificates curl git libxml2-dev sudo tk python3-tk libgomp1 gfortran \
    && git clone https://github.com/samuelduchesne/archetypal.git \
    && cd /archetypal \
    && sudo chmod +x install_energyplus.sh \
    && ./install_energyplus.sh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm /tmp/energyplus_versions.json

# Install poetry
RUN pip install "poetry==$POETRY_VERSION"

# Copy only requirements to cache them in docker layer
WORKDIR /code
COPY poetry.lock pyproject.toml /code/

# Project initialization:
RUN poetry install --no-interaction --no-ansi --no-root --with worker --with coiled
RUN epinterface prisma generate

# Copy Python code to the Docker image
COPY epengine /code/epengine/
# COPY data/idfs /code/data/idfs
# COPY data/epws /code/data/epws

ENV PYTHONPATH=/code

# ENTRYPOINT [ "poetry", "run", "worker" ]
# ENTRYPOINT [ "python" ]
CMD [ "python", "epengine/worker/main.py"]
