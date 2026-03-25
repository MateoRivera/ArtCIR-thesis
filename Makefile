# derived from https://github.com/pydantic/pydantic
.DEFAULT_GOAL := help
sources = src tests tools
.ONESHELL:

.PHONY: .uv  ## Check that uv is installed
.uv:
	@uv -V || echo 'Please install uv: https://docs.astral.sh/uv/getting-started/installation/'

.PHONY: .pre-commit  ## Check that pre-commit is installed
.pre-commit: .uv
	@uv run pre-commit -V || uv pip install pre-commit

.PHONY: install  ## Install all dependencies and pre-commit hooks
install: .uv
	uv sync --all-groups
	uv run pre-commit install --install-hooks

.PHONY: format  ## Auto-format source files with ruff
format: .uv
	uv run ruff check --fix $(sources)
	uv run ruff format $(sources)

.PHONY: lint  ## Lint source files with ruff
lint: .uv
	uv run ruff check $(sources)
	uv run ruff format --check $(sources)

.PHONY: test  ## Run tests with coverage report
test: .uv
	uv run coverage run -m pytest --durations=10
	uv run coverage xml
	uv run coverage html

.PHONY: all  ## Run lint and tests
all: lint test

.PHONY: docs  ## Build documentation
docs: .uv
	uv run mkdocs build --strict

.PHONY: docs-serve  ## Serve documentation locally for preview
docs-serve: .uv
	uv run mkdocs serve --strict

.PHONY: clean  ## Clear local caches and build artifacts
clean:
	rm -rf `find . -name __pycache__`
	rm -f `find . -type f -name '*.py[co]'`
	rm -f `find . -type f -name '*~'`
	rm -rf .cache
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	rm -rf *.egg-info
	rm -f .coverage .coverage.*
	rm -rf build dist site
	rm -rf docs/_build
	rm -f coverage.xml

.PHONY: release  ## Bump version, commit, tag and push (BUMP=major|minor|patch)
release: .uv
ifndef BUMP
	$(error BUMP is not set. Usage: make release BUMP=major|minor|patch)
endif
	@echo "Current version: $$(uv version)"
	@uv version --bump $(BUMP)
	@NEW_VERSION=$$(uv version --short)
	@echo "New version: v$$NEW_VERSION"
	@git add pyproject.toml
	@git commit -m "release: version v$$NEW_VERSION"
	@git tag -a "v$$NEW_VERSION" -m "Release v$$NEW_VERSION"
	@git push
	@git push origin tag v$$NEW_VERSION
	@echo "version bumped to $$NEW_VERSION — run 'uv publish' if needed"

.PHONY: help  ## Display this help
help:
	@grep -E \
		'^.PHONY: .*?## .*$$' $(MAKEFILE_LIST) | \
		sort | \
		awk 'BEGIN {FS = ".PHONY: |## "}; {printf "\033[36m%-19s\033[0m %s\n", $$2, $$3}'
