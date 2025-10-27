# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2025-01-27

### Added
- Initial public release
- Declarative search space definition with `sp.Config` base class
- Automatic space inference from type hints and Pydantic Field constraints
- Explicit space definitions: `sp.Int`, `sp.Float`, `sp.Categorical`
- Conditional parameters with `sp.Conditional` and rich condition system
- Nested and modular configurations with inheritance support
- Polymorphic field support (Union types)
- Multi-format serialization (JSON, YAML, TOML)
- Seamless Optuna integration via `from_trial()` method
- Random sampling with `random()` method
- Override system for iterative search space refinement
- Comprehensive visualization with `get_tree()` method
- Parameter naming inspection with `get_parameter_names()`
- Override template generation with `get_override_template()`
- Custom sampler support via `Sampler` interface
- Full type safety and validation via Pydantic v2
- 93% test coverage
- Complete documentation with 5 Jupyter notebook examples
- CI/CD pipeline with GitHub Actions

[unreleased]: https://github.com/keyhankamyar/SpaX/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/keyhankamyar/SpaX/releases/tag/v0.2.0