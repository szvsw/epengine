.PHONY: install
install: ## Install the poetry environment and install the pre-commit hooks
	@echo "🚀 Creating virtual environment using uv"
	@uv sync --all-groups --all-extras
	@uv run pre-commit install
	@uv run epi prisma generate

.PHONY: clean-env
clean-env: ## Clean the uv environment
	@echo "🚀 Removing .venv directory created by uv (if exists)"
	@rm -rf .venv

.PHONY: check
check: ## Run code quality tools.
	@echo "🚀 Checking lock file consistency with 'pyproject.toml'"
	@uv lock --locked
	@echo "🚀 Linting code: Running pre-commit"
	@uv run pre-commit run -a
	@echo "🚀 Static type checking: Running pyright"
	@uv run pyright

.PHONY: test
test: ## Test the code with pytest
	@echo "🚀 Testing code: Running pytest"
	@uv run pytest --cov --cov-config=pyproject.toml --cov-report=xml


.PHONY: build
build: clean-build ## Build wheel file
	@echo "🚀 Creating wheel file"
	@uvx --from build pyproject-build --installer uv

.PHONY: clean-build
clean-build: ## Clean build artifacts
	@echo "🚀 Removing build artifacts"
	@uv run python -c "import shutil; import os; shutil.rmtree('dist') if os.path.exists('dist') else None"

.PHONY: publish
publish: ## Publish a release to PyPI.
	@echo "🚀 Publishing."
	@uvx twine upload --repository-url https://upload.pypi.org/legacy/ dist/*

.PHONY: build-and-publish
build-and-publish: build publish ## Build and publish.

.PHONY: docs-test
docs-test: ## Test if documentation can be built without warnings or errors
	@uv run mkdocs build -s

.PHONY: docs
docs: ## Build and serve the documentation
	@uv run mkdocs serve

.PHONY: docs-deploy
docs-deploy: ## Build and serve the documentation
	@uv run mkdocs gh-deploy

.PHONY: down
down: ## Stop the docker-compose services
	@docker compose down --remove-orphans

.PHONY: worker
worker: ## Start the worker
	@make down
	@docker compose -f docker-compose.yml up -d worker --build --remove-orphans

.PHONY: workers
workers: ## Start the workers with replicas
	@make down
	@docker compose -f docker-compose.yml -f docker-compose.replicas.yml up -d worker --build --remove-orphans

.PHONY: worker-push
worker-push: ## Push the worker to the workers
	@make docker-login
	@docker compose -f docker-compose.yml build worker
	@docker compose -f docker-compose.yml push worker


.PHONY: worker-it
worker-it: ## Run the worker in interactive mode
	@docker compose exec -it worker /bin/bash

.PHONY: prod
prod: ## Start the docker compose services with the api and worker
	@make down
	@docker compose -f docker-compose.yml up -d api worker --build

.PHONY: hatchet-token
hatchet-token: ## Start the hatchet service and generate a token
	@docker compose -f docker-compose.yml -f docker-compose.override.yml -f docker-compose.hatchet.yml up -d hatchet-lite
	@echo ----
	@echo Append the following lines to your .env.dev file:
	@echo HATCHET_CLIENT_TOKEN=${shell docker compose -f docker-compose.yml -f docker-compose.hatchet.yml exec hatchet-lite /hatchet-admin token create --config /config --tenant-id 707d0855-80ab-4e1f-a156-f1c4546cbf52}
	@echo HATCHET_CLIENT_TLS_STRATEGY=none
	@echo ----
	@echo Your login info for the Hatchet web UI is:
	@echo Username: admin@example.com
	@echo Password: Admin123!!

.PHONY: dev
dev: ## Start the docker compose services with the api and worker along with moto
	@make down
	@docker compose up --build

.PHONY: dev-with-hatchet
dev-with-hatchet: ## Start the docker compose services with the api and worker along with moto
	@make down
	@docker compose -f docker-compose.yml -f docker-compose.override.yml -f docker-compose.hatchet.yml up --build


.PHONY: docker-login
docker-login: ## Login to aws docker ecr
	@aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin $(AWS_ACCOUNT_ID).dkr.ecr.${AWS_REGION}.amazonaws.com

.PHONY: dozzle
dozzle: ## run the dozzle logs container
	@docker run --detach --volume=/var/run/docker.sock:/var/run/docker.sock -p 9090:8080 amir20/dozzle

.PHONY: help
help:
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.PHONY: create-copilot-app
create-copilot-app: ## Create the entire stack with copilot
	@copilot app init
	@copilot env init --name prod
	@copilot env deploy --name prod
	@copilot secret init --name HATCHET_CLIENT_TOKEN
	@copilot deploy --init-wkld --env prod --all

.DEFAULT_GOAL := help
