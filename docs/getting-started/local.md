# Local Development

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

```bash
make dev
```

```bash
make prod
```
