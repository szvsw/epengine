.PHONY: install
install: ## Install the poetry environment and install the pre-commit hooks
	@echo "🚀 Creating virtual environment using pyenv and poetry"
	@poetry install --with dev --with worker --with api --with docs
	@ poetry run pre-commit install
	@poetry shell

.PHONY: check
check: ## Run code quality tools.
	@echo "🚀 Checking Poetry lock file consistency with 'pyproject.toml': Running poetry check --lock"
	@poetry check --lock
	@echo "🚀 Linting code: Running pre-commit"
	@poetry run pre-commit run -a

.PHONY: test
test: ## Test the code with pytest
	@echo "🚀 Testing code: Running pytest"
	@poetry run pytest --cov --cov-config=pyproject.toml --cov-report=xml

.PHONY: build
build: clean-build ## Build wheel file using poetry
	@echo "🚀 Creating wheel file"
	@poetry build

.PHONY: clean-build
clean-build: ## clean build artifacts
	@rm -rf dist

.PHONY: publish
publish: ## publish a release to pypi.
	@echo "🚀 Publishing: Dry run."
	@poetry config pypi-token.pypi $(PYPI_TOKEN)
	@poetry publish --dry-run
	@echo "🚀 Publishing."
	@poetry publish

.PHONY: build-and-publish
build-and-publish: build publish ## Build and publish.

.PHONY: docs-test
docs-test: ## Test if documentation can be built without warnings or errors
	@poetry run mkdocs build -s

.PHONY: docs
docs: ## Build and serve the documentation
	@poetry run mkdocs serve

.PHONY: docs-deploy
docs-deploy: ## Build and serve the documentation
	@poetry run mkdocs gh-deploy

.PHONY: down
down: ## Stop the docker-compose services
	@docker compose down

.PHONY: prod
prod: ## Start the docker compose services with the api and worker
	@make down
	@docker compose build
	@docker compose -f docker-compose.yml up -d api worker

.PHONY: hatchet-token
hatchet-token: ## Start the hatchet service and generate a token
	@docker compose -f docker-compose.yml -f docker-compose.override.yml -f docker-compose.hatchet.yml up -d hatchet-lite
	@echo ----
	@echo Append the following lines to your .env.dev file:
	@echo HATCHET_CLIENT_TOKEN=${shell docker compose -f docker-compose.hatchet.yml exec hatchet-lite /hatchet-admin token create --config /config --tenant-id 707d0855-80ab-4e1f-a156-f1c4546cbf52}
	@echo HATCHET_CLIENT_TLS_STRATEGY=none
	@echo ----
	@echo Your login info for the Hatchet web UI is:
	@echo Username: admin@example.com
	@echo Password: Admin123!!

.PHONY: dev-with-hatchet
dev-with-hatchet: ## Start the docker compose services with the api and worker along with moto
	@make down
	@docker compose build
	@docker compose -f docker-compose.yml -f docker-compose.override.yml -f docker-compose.hatchet.yml up


.PHONY: dev
dev: ## Start the docker compose services with the api and worker along with moto
	@make down
	@docker compose build
	@docker compose up

.PHONY: worker-it
worker-it: ## Run the worker in interactive mode
	@docker compose exec -it worker /bin/bash

.PHONY: docker-login
docker-login: ## Login to aws docker ecr
	@aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin $(AWS_ACCOUNT_ID).dkr.ecr.${AWS_REGION}.amazonaws.com

.PHONY: dozzle
dozzle: ## run the dozzle logs container
	@docker run --detach --volume=/var/run/docker.sock:/var/run/docker.sock -p 9090:8080 amir20/dozzle

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.PHONY: create-copilot-app
create-copilot-app: ## Create the entire stack with copilot
	@copilot app init
	@copilot env init --name prod
	@copilot env deploy --name prod
	@copilot secret init --name HATCHET_CLIENT_TOKEN
	@copilot deploy --init-wkld --env prod --all
.DEFAULT_GOAL := help
