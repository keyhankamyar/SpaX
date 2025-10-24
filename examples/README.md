# SpaX Examples

This directory contains comprehensive examples demonstrating all major features of SpaX.

## Overview

The examples are organized from basic to advanced usage, covering:
- Minimal overhead migration from Pydantic
- Basic configuration definition
- Nested and inherited configurations
- Conditional parameters
- Sampling and random generation
- Integration with Optuna
- Override system for iterative refinement
- Serialization and deserialization

## Examples List

### Getting Started
1. **[00_pydantic_migration.py](00_pydantic_migration.py)** - Migrating from Pydantic BaseModel (minimal overhead!)
2. **[01_basic_config.py](01_basic_config.py)** - Simple configuration with numeric and categorical spaces
3. **[02_nested_config.py](02_nested_config.py)** - Nested configuration structures
4. **[03_inheritance.py](03_inheritance.py)** - Config inheritance and extension

### Advanced Configuration
5. **[04_conditional_parameters.py](04_conditional_parameters.py)** - Conditional parameters based on other fields
6. **[05_deep_nesting.py](05_deep_nesting.py)** - Multi-level deeply nested configurations
7. **[06_chained_conditions.py](06_chained_conditions.py)** - Conditions on nested fields

### Sampling and Generation
8. **[07_sampling.py](07_sampling.py)** - Random sampling with seeds and reproducibility
9. **[08_override_system.py](08_override_system.py)** - Using overrides to narrow search spaces
10. **[09_sanity_checking.py](09_sanity_checking.py)** - Quick sanity checking and debugging

### Integration
11. **[10_optuna_integration.py](10_optuna_integration.py)** - Full Optuna HPO integration example
12. **[11_serialization.py](11_serialization.py)** - JSON, YAML, TOML serialization

## Key Highlight: Minimal Overhead

**SpaX is designed to work seamlessly with existing Pydantic code.** Simply swap `BaseModel` with `Config` and optionally use SpaX spaces for searchable parameters. You can mix and match:

- Pydantic's `Field()` constraints → Automatically inferred as spaces
- Type hints like `Literal` and `bool` → Automatically inferred as categorical spaces  
- SpaX explicit spaces (`sp.Int`, `sp.Float`, etc.) → Full control when needed

See [00_pydantic_migration.py](00_pydantic_migration.py) for a detailed comparison!

## Running Examples

Each example is self-contained and can be run directly:
```bash
python examples/00_pydantic_migration.py
python examples/01_basic_config.py
# ... etc
```

Some examples (like Optuna integration) may require additional dependencies:
```bash
pip install optuna  # For Optuna examples
pip install PyYAML  # For YAML examples
pip install tomli-w  # For TOML examples
```

## Learning Path

**New to SpaX?** Start with example 0 to see how easy migration is!

**Beginners:** Continue with examples 1-4 to understand basic usage.

**Intermediate-Advanced:** Move to examples 5-10 to learn advanced features.

## Tips

- Each example includes detailed comments explaining the code
- Examples build on each other - concepts from earlier examples are used in later ones
- Try modifying the examples to experiment with different configurations
- Use `Config.get_override_template()` to understand the override structure
- Most examples can run without any additional dependencies beyond SpaX