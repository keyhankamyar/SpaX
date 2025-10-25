.PHONY: ci lint type test build distclean

# Run everything CI cares about
ci: lint type test build

# Ruff (lint + format check)
lint:
	python -m ruff check .
	python -m ruff format --check .

# Mypy (static types)
type:
	python -m mypy --config-file pyproject.toml spax

# Pytest
test:
	python -m pytest

# Build sdist/wheel and verify metadata
build: distclean
	python -m build
	python -m twine check dist/*

# Clean previous build artifacts
distclean:
	rm -rf build dist *.egg-info
