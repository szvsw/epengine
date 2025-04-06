ARG EP_VERSION=22.2.0
ARG PYTHON_VERSION=3.12
FROM nrel/energyplus:${EP_VERSION} AS energyplus

# Remove unnecessary datasets to reduce image size
RUN mkdir -p /energyplus-local && \
    cd $ENERGYPLUS_DOWNLOAD_BASENAME && \
    rm -rf DataSets Documentation ExampleFiles WeatherData MacroDataSets \
    workflows PostProcess/convertESOMTRpgm \
    PostProcess/EP-Compare PreProcess/FMUParser PreProcess/ParametricPreProcessor && \
    # move the remaining files to a new directory with a known name that
    # can be used copied from regardless of the location where
    # ep is installed by the nrel image
    cp -r * /energyplus-local/

FROM python:${PYTHON_VERSION}-slim-bookworm AS main


# Install dependencies for running energy models
# etc
RUN apt-get update && \
    apt-get install --no-install-recommends -y ca-certificates libx11-6 libexpat1 \
    libgomp1 libcurl4-openssl-dev libssl-dev libxml2-dev gfortran \
    curl tk python3-tk \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy EnergyPlus from the first stage to the second stage
ARG EP_VERSION
ENV ENERGYPLUS_INSTALL_DIR=/usr/local/EnergyPlus-${EP_VERSION//./-}
COPY --from=energyplus /energyplus-local $ENERGYPLUS_INSTALL_DIR

# Create symlinks for EnergyPlus executables
RUN ln -s $ENERGYPLUS_INSTALL_DIR/energyplus /usr/local/bin/energyplus \
    && ln -s $ENERGYPLUS_INSTALL_DIR/runenergyplus /usr/local/bin/runenergyplus \
    && ln -s $ENERGYPLUS_INSTALL_DIR/runreadvars  /usr/local/bin/runreadvars \
    && ln -s $ENERGYPLUS_INSTALL_DIR/runepmacro /usr/local/bin/runepmacro \
    && ln -s $ENERGYPLUS_INSTALL_DIR/ExpandObjects /usr/local/bin/ExpandObjects \
    && ln -s $ENERGYPLUS_INSTALL_DIR/PreProcess/GrndTempCalc/Basement /usr/local/bin/Basement \
    && ln -s $ENERGYPLUS_INSTALL_DIR/PreProcess/GrndTempCalc/Slab /usr/local/bin/Slab


# PYTHON etc setup ...
ARG POETRY_VERSION=2.1.2
ENV POETRY_VERSION=$POETRY_VERSION \
    POETRY_VIRTUALENVS_CREATE=false

RUN pip install "poetry==$POETRY_VERSION"

WORKDIR /code
COPY poetry.lock pyproject.toml /code/

ARG POETRY_EXTRAS="-E ml"
RUN poetry config virtualenvs.in-project true && poetry install --no-interaction --no-ansi --no-root --only main ${POETRY_EXTRAS}

RUN epinterface prisma generate

ENV PYTHONPATH=/code

COPY epengine /code/epengine/

CMD [ "python", "epengine/worker/main.py"]
