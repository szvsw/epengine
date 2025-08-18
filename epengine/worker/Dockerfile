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
COPY --from=ghcr.io/astral-sh/uv:0.6.16 /uv /uvx /bin/


# Install dependencies for running energy models
# etc
RUN apt-get update && \
    apt-get install --no-install-recommends -y ca-certificates libx11-6 libexpat1 \
    libgomp1 libcurl4-openssl-dev libssl-dev libxml2-dev gfortran \
    curl tk python3-tk \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy EnergyPlus from the first stage to the second stage
ARG EP_VERSION
COPY --from=energyplus /energyplus-local /usr/local/EnergyPlus-${EP_VERSION}

# Rename directory from dotted to dashed version and set up symlinks
RUN EP_VERSION_DASH=$(echo "${EP_VERSION}" | tr '.' '-') && \
    mv /usr/local/EnergyPlus-${EP_VERSION} /usr/local/EnergyPlus-${EP_VERSION_DASH} && \
    ln -s /usr/local/EnergyPlus-${EP_VERSION_DASH}/energyplus /usr/local/bin/energyplus && \
    ln -s /usr/local/EnergyPlus-${EP_VERSION_DASH}/runenergyplus /usr/local/bin/runenergyplus && \
    ln -s /usr/local/EnergyPlus-${EP_VERSION_DASH}/runreadvars /usr/local/bin/runreadvars && \
    ln -s /usr/local/EnergyPlus-${EP_VERSION_DASH}/runepmacro /usr/local/bin/runepmacro && \
    ln -s /usr/local/EnergyPlus-${EP_VERSION_DASH}/ExpandObjects /usr/local/bin/ExpandObjects && \
    ln -s /usr/local/EnergyPlus-${EP_VERSION_DASH}/PreProcess/GrndTempCalc/Basement /usr/local/bin/Basement && \
    ln -s /usr/local/EnergyPlus-${EP_VERSION_DASH}/PreProcess/GrndTempCalc/Slab /usr/local/bin/Slab

# PYTHON etc setup ...
WORKDIR /code
COPY uv.lock pyproject.toml README.md /code/

ARG UV_EXTRAS="--extra ml"
RUN uv sync --locked --no-install-project ${UV_EXTRAS}

RUN uv run --no-sync epinterface prisma generate

COPY epengine /code/epengine/

CMD [ "uv", "run", "worker"]
