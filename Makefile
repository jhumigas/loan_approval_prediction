OUTPUT_DIR = .
SRC_DIR = src/loan_approval_prediction

.PHONY: help
help:
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} /^[a-zA-Z0-9_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } ' $(MAKEFILE_LIST)

.PHONY: download-dataset
download-dataset: ## download dataset
	mkdir -p ./data/dataset
	curl -L -o ./data/dataset/loan-approval-dataset.zip https://www.kaggle.com/api/v1/datasets/download/anishdevedward/loan-approval-dataset
	unzip -o ./data/dataset/loan-approval-dataset.zip -d ./data/dataset

.PHONY: install
install: ## install dependencies
	uv sync --locked --all-extras --dev

.PHONY: check-bandit
check-bandit: ## code vulnerability security scan
	uvx bandit -c pyproject.toml -r .

# Numpy safety check error are ignored du to sktime dependencies. Turn numpy to 1.22.0 to resolve safety check when
# this issue will be resolved https://github.com/alan-turing-institute/sktime/discussions/2037
.PHONY: check-safety
check-safety: ## dependencies scan with safety
	uv pip freeze | uvx safety check -i 44715 -i 44716 -i 44717 --stdin

.PHONY: pre-commit
pre-commit: ## run pre-commit hooks
	uvx pre-commit run --all-files

.PHONY: install-pre-commit
install-pre-commit: ## install pre-commit hooks
	uvx pre-commit install

.PHONY: detect-secrets
detect-secrets: ## detect secrets in code
	uvx detect-secrets scan $(SRC_DIR) --all-files

.PHONY: format
format: ## format code
	uvx ruff format .

.PHONY: lint
lint: ## lint code
	uvx ruff check .

.PHONY: unit-tests
unit-tests:  ## run unit tests
	uv run pytest tests/unit

.PHONY: start-dev-fast-app
start-dev-fast-app: ## start dev mode
	uv run fastapi dev src/loan_approval_prediction/serve.py