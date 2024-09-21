# Getting Started

The following pages will walk you through setting up a distributed EnergyPlus simulation engine.

## Installation

- Docker
- poetry

```bash
make install
```

## Configuration

```bash
cp .env.example .env
cp .env.example .env.dev
```

You will then want to enter your AWS credentials into `.env` and your Hatchet credentials into both.

## Running the System

### Local

```bash
make dev
```

```bash
make prod
```

### Cloud
