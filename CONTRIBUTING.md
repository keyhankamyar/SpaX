# Contributing to SpaX

Thank you for your interest in contributing to SpaX! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Testing Guidelines](#testing-guidelines)
- [Submitting Changes](#submitting-changes)
- [Areas Where Help is Appreciated](#areas-where-help-is-appreciated)

## Code of Conduct

This project follows standard open-source community guidelines. Be respectful, constructive, and professional in all interactions.

## How Can I Contribute?

### Reporting Bugs

Before creating a bug report:
- **Search existing issues** to avoid duplicates
- **Check the documentation** and examples to ensure it's actually a bug

When creating a bug report, include:
- **Clear title** describing the issue
- **Minimal reproducible example** (MRE) showing the bug
- **Expected vs actual behavior**
- **Environment details** (Python version, SpaX version, OS)
- **Error messages and stack traces** if applicable

**Example:**
```python
# Bug: Conditional parameter not validated correctly
import spax as sp

class Config(sp.Config):
    use_feature: bool
    value: int = sp.Conditional(
        sp.FieldCondition("use_feature", sp.EqualsTo(True)),
        true=sp.Int(ge=1, le=10),
        false=0,
    )

# This should fail but doesn't
config = Config(use_feature=True, value=100)  # value=100 is out of range
```

### Suggesting Features

Feature requests are welcome! When suggesting a feature:
- **Describe the use case** - what problem does it solve?
- **Provide examples** of how the API would look
- **Consider alternatives** you've tried or seen elsewhere
- **Discuss scope** - should this be core functionality or a plugin?

### Contributing Code

We welcome code contributions! See [Development Setup](#development-setup) and [Development Workflow](#development-workflow) below.

### Improving Documentation

Documentation improvements are highly valued:
- Fix typos or unclear explanations
- Add examples for undocumented features
- Improve existing examples
- Write tutorials or guides

## Development Setup

### Prerequisites

- Python 3.11 or higher
- Git
- Familiarity with Pydantic (helpful but not required)

### Setup Steps

1. **Fork and clone the repository:**
```bash
   git clone https://github.com/YOUR_USERNAME/SpaX.git
   cd SpaX
```

2. **Create a virtual environment:**
```bash
   # Using conda
   conda create -n spax-dev python=3.12
   conda activate spax-dev

   # Or using venv
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install in development mode with all dependencies:**
```bash
   pip install -e ".[dev,all]"
```

4. **Install pre-commit hooks:**
```bash
   pre-commit install
```

5. **Verify setup:**
```bash
   make ci
```

   This runs linting, type checking, tests, and build verification.

## Development Workflow

### Creating a Feature Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

Use descriptive branch names:
- `feature/add-custom-sampler-api`
- `fix/conditional-validation-bug`
- `docs/improve-quickstart-example`

### Making Changes

1. **Write code** following the [Code Standards](#code-standards)
2. **Add tests** for new functionality
3. **Update documentation** if adding user-facing features
4. **Run checks locally:**
```bash
   make ci  # Runs lint, type check, tests, and build
```

Pre-commit hooks will automatically run on `git commit`, but running `make ci` gives you the full picture.

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_config/test_config_basic.py

# Run with coverage report
pytest --cov=spax --cov-report=term-missing

# Run only fast tests (skip slow/integration tests)
pytest -m "not slow"
```

### Code Quality Checks
```bash
# Lint with Ruff
make lint
# or
ruff check .

# Format code with Ruff
ruff format .

# Type check with Mypy
make type
# or
mypy spax

# Run everything
make ci
```

## Code Standards

### Style Guide

- **Formatter:** Black (via Ruff) - automated, no configuration needed
- **Linter:** Ruff - catches common issues and enforces best practices
- **Type hints:** Required for all functions (checked with Mypy)
- **Line length:** 88 characters (Black default)

### Type Hints

All functions must have type hints:
```python
# ✅ Good
def suggest_value(self, name: str, low: int, high: int) -> int:
    return random.randint(low, high)

# ❌ Bad
def suggest_value(self, name, low, high):
    return random.randint(low, high)
```

### Docstrings

Use Google-style docstrings for public APIs:
```python
def create_space(
    low: int,
    high: int,
    distribution: Literal["uniform", "log"] = "uniform",
) -> IntSpace:
    """Create an integer search space.

    Args:
        low: Lower bound (inclusive).
        high: Upper bound (inclusive).
        distribution: Sampling distribution.

    Returns:
        IntSpace instance configured with the specified bounds.

    Raises:
        ValueError: If low > high.

    Example:
        >>> space = create_space(1, 10)
        >>> value = space.sample()
        >>> assert 1 <= value <= 10
    """
```

### Imports

- **Use absolute imports from project root:** `from spax.nodes import ConfigNode`
- **Use relative imports within the same package:** `from .base import Node` (for files in the same directory/package)
- **Group imports:** standard library → third-party → local (spax)
- Ruff will automatically sort and organize imports

### Code Organization

- **Keep functions focused** - single responsibility
- **Avoid deep nesting** - extract helper functions if >3 levels
- **Descriptive names** - `validate_numeric_bounds` not `check_vals`
- **Comments for "why"** not "what" - code should be self-documenting

## Testing Guidelines

### Test Coverage

- Aim for **>90% coverage** for new code
- All new features must have tests
- Bug fixes should include regression tests

### Test Structure
```python
def test_feature_with_valid_input():
    """Test feature behavior with valid input."""
    # Arrange
    config = MyConfig(param1=10, param2="value")

    # Act
    result = config.some_method()

    # Assert
    assert result.param1 == 10
    assert result.param2 == "value"


def test_feature_with_invalid_input():
    """Test feature validation with invalid input."""
    with pytest.raises(ValueError, match="param1 must be positive"):
        MyConfig(param1=-1, param2="value")
```

## Submitting Changes

### Before Submitting

Ensure your changes pass all checks:
```bash
# Run full CI locally
make ci

# Verify pre-commit passes
pre-commit run --all-files
```

### Commit Messages

Follow conventional commit format:
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions/changes
- `refactor`: Code refactoring
- `ci`: CI/CD changes
- `chore`: Maintenance tasks

**Examples:**
```
feat(spaces): add weighted categorical sampling

- Add Choice class with weight parameter
- Update Categorical to use weighted random selection
- Add tests for weighted sampling behavior

Closes #42
```
```
fix(conditional): validate nested field conditions correctly

Previously, conditions on nested fields like "model.num_layers"
would fail silently. Now properly traverses the field path and
validates the condition.

Fixes #128
```

### Pull Request Process

1. **Create a PR** with a clear title and description
2. **Link related issues** using "Closes #123" or "Fixes #456"
3. **Describe changes** - what, why, and how
4. **Add examples** if introducing new features
5. **Update CHANGELOG.md** under `[Unreleased]` section
6. **Respond to review feedback** promptly

**PR Template:**
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] All tests pass locally (`make ci`)
- [ ] Pre-commit hooks pass

## Related Issues
Closes #123
```

### Review Process

- Maintainers will review PRs within a few days
- Address feedback in new commits (don't force-push during review)
- Once approved, maintainer will merge (squash merge for clean history)

## Areas Where Help is Appreciated

### High Priority

- **Additional examples and tutorials** - real-world use cases
- **Documentation improvements** - clarity, completeness, examples
- **Bug reports and fixes** - especially edge cases
- **Integration with other frameworks** - Ray Tune, HyperOpt, etc.

### Future Features

See [ROADMAP](README.md#-roadmap) in the README for planned features. Feel free to pick up any of these!

### Good First Issues

Look for issues labeled `good-first-issue` - these are intentionally scoped for new contributors.

## Questions?

- **General questions:** [Open a discussion](https://github.com/keyhankamyar/SpaX/discussions)
- **Bug reports:** [Open an issue](https://github.com/keyhankamyar/SpaX/issues)
- **Feature requests:** [Open an issue](https://github.com/keyhankamyar/SpaX/issues)

Thank you for contributing to SpaX! 🚀
