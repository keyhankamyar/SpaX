"""Example 11: Serialization and Deserialization

This example demonstrates SpaX's serialization capabilities for saving and
loading configurations in multiple formats: JSON, YAML, and TOML. This is
essential for:
- Saving experiment configurations
- Sharing configurations with team members
- Version control of hyperparameters
- Reproducing experiments

Topics Covered:
--------------
- JSON serialization/deserialization
- YAML serialization/deserialization (requires PyYAML)
- TOML serialization/deserialization (requires tomli/tomli-w)
- Handling nested configurations
- Type discriminators for Config classes
- Serialization of complex types
- Best practices for config management
"""

import spax as sp

# =============================================================================
# Simple Configuration
# =============================================================================


class SimpleConfig(sp.Config):
    """Simple configuration for serialization demo."""

    learning_rate: float = sp.Float(ge=1e-5, le=1e-1, distribution="log")
    batch_size: int = sp.Int(ge=8, le=128)
    num_layers: int = sp.Int(ge=1, le=10)
    optimizer: str = sp.Categorical(["adam", "sgd", "adamw"])
    use_dropout: bool = sp.Categorical([True, False])


# =============================================================================
# Nested Configuration
# =============================================================================


class OptimizerConfig(sp.Config):
    """Optimizer configuration."""

    name: str = sp.Categorical(["adam", "sgd", "adamw"])
    learning_rate: float = sp.Float(ge=1e-5, le=1e-1, distribution="log")
    weight_decay: float = sp.Float(ge=1e-6, le=1e-2, distribution="log")


class ModelConfig(sp.Config):
    """Model configuration."""

    hidden_dim: int = sp.Int(ge=64, le=1024)
    num_layers: int = sp.Int(ge=1, le=12)
    dropout_rate: float = sp.Float(ge=0.0, le=0.5)
    activation: str = sp.Categorical(["relu", "gelu", "silu"])


class NestedConfig(sp.Config):
    """Nested configuration with multiple components."""

    model: ModelConfig
    optimizer: OptimizerConfig
    batch_size: int = sp.Int(ge=8, le=256)
    seed: int = sp.Int(ge=0, le=9999, default=42)
    experiment_name: str = "nested_experiment"


# =============================================================================
# Configuration with Conditionals
# =============================================================================


class ConditionalConfig(sp.Config):
    """Configuration with conditional parameters."""

    use_dropout: bool = sp.Categorical([True, False])
    dropout_rate: float = sp.Conditional(
        sp.FieldCondition("use_dropout", sp.EqualsTo(True)),
        true=sp.Float(gt=0.0, lt=0.5),
        false=0.0,
    )

    use_l2: bool = sp.Categorical([True, False])
    l2_strength: float = sp.Conditional(
        sp.FieldCondition("use_l2", sp.EqualsTo(True)),
        true=sp.Float(ge=1e-6, le=1e-2, distribution="log"),
        false=0.0,
    )

    learning_rate: float = sp.Float(ge=1e-5, le=1e-1, distribution="log")


# =============================================================================
# Demonstrations
# =============================================================================


def main():
    print("=" * 80)
    print("Example 11: Serialization and Deserialization")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. JSON serialization - Simple config
    # -------------------------------------------------------------------------
    print("\n1. JSON Serialization - Simple Config:")
    print("-" * 80)

    # Create config
    config = SimpleConfig.random(seed=42)
    print("Original config:")
    print(f"  learning_rate: {config.learning_rate:.6f}")
    print(f"  batch_size: {config.batch_size}")
    print(f"  num_layers: {config.num_layers}")
    print(f"  optimizer: {config.optimizer}")
    print(f"  use_dropout: {config.use_dropout}")

    # Serialize to JSON
    json_str = config.model_dump_json(indent=2)
    print("\nSerialized to JSON:")
    print(json_str)

    # Deserialize from JSON
    loaded_config = SimpleConfig.model_validate_json(json_str)
    print("\nDeserialized config:")
    print(f"  learning_rate: {loaded_config.learning_rate:.6f}")
    print(f"  batch_size: {loaded_config.batch_size}")

    # Verify equality
    print(f"\n✓ Configs match: {config.learning_rate == loaded_config.learning_rate}")

    # -------------------------------------------------------------------------
    # 2. JSON with nested configs
    # -------------------------------------------------------------------------
    print("\n2. JSON with Nested Configurations:")
    print("-" * 80)

    nested_config = NestedConfig.random(seed=100)
    print("Original nested config:")
    print(f"  model.hidden_dim: {nested_config.model.hidden_dim}")
    print(f"  optimizer.name: {nested_config.optimizer.name}")
    print(f"  batch_size: {nested_config.batch_size}")

    # Serialize
    json_str = nested_config.model_dump_json(indent=2)
    print("\nJSON structure (first 500 chars):")
    print(json_str[:500] + "...")

    # Check for type discriminators
    import json

    json_data = json.loads(json_str)
    print("\nType discriminators in JSON:")
    print(f"  'model' has __type__: {'__type__' in json_data.get('model', {})}")
    print(f"  'optimizer' has __type__: {'__type__' in json_data.get('optimizer', {})}")
    if "__type__" in json_data.get("model", {}):
        print(f"  model.__type__ = '{json_data['model']['__type__']}'")
    if "__type__" in json_data.get("optimizer", {}):
        print(f"  optimizer.__type__ = '{json_data['optimizer']['__type__']}'")

    # Deserialize
    loaded_nested = NestedConfig.model_validate_json(json_str)
    print("\n✓ Deserialized successfully")
    print(
        f"  model.hidden_dim matches: {nested_config.model.hidden_dim == loaded_nested.model.hidden_dim}"
    )
    print(
        f"  optimizer.name matches: {nested_config.optimizer.name == loaded_nested.optimizer.name}"
    )

    # -------------------------------------------------------------------------
    # 3. YAML serialization
    # -------------------------------------------------------------------------
    print("\n3. YAML Serialization:")
    print("-" * 80)

    try:
        import yaml  # noqa: F401

        YAML_AVAILABLE = True
    except ImportError:
        YAML_AVAILABLE = False
        print("⚠️  PyYAML not installed. Install with: pip install PyYAML")

    if YAML_AVAILABLE:
        config = SimpleConfig.random(seed=200)

        # Serialize to YAML
        yaml_str = config.model_dump_yaml()
        print("Serialized to YAML:")
        print(yaml_str)

        # Deserialize from YAML
        loaded_config = SimpleConfig.model_validate_yaml(yaml_str)
        print("✓ Deserialized from YAML successfully")
        print(
            f"  learning_rate matches: {config.learning_rate == loaded_config.learning_rate}"
        )

        # Nested config in YAML
        print("\nNested config in YAML:")
        nested_config = NestedConfig.random(seed=201)
        yaml_str = nested_config.model_dump_yaml()
        print(yaml_str[:400] + "...")

        loaded_nested = NestedConfig.model_validate_yaml(yaml_str)
        print("✓ Nested config deserialized successfully")

    # -------------------------------------------------------------------------
    # 4. TOML serialization
    # -------------------------------------------------------------------------
    print("\n4. TOML Serialization:")
    print("-" * 80)

    try:
        import tomllib  # noqa: F401

        import tomli_w  # noqa: F401

        TOML_AVAILABLE = True
    except ImportError:
        TOML_AVAILABLE = False
        print("⚠️  tomli-w not installed. Install with: pip install tomli-w")

    if TOML_AVAILABLE:
        config = SimpleConfig.random(seed=300)

        # Serialize to TOML
        toml_str = config.model_dump_toml()
        print("Serialized to TOML:")
        print(toml_str)

        # Deserialize from TOML
        loaded_config = SimpleConfig.model_validate_toml(toml_str)
        print("✓ Deserialized from TOML successfully")
        print(f"  batch_size matches: {config.batch_size == loaded_config.batch_size}")

        # Nested config in TOML
        print("\nNested config in TOML:")
        nested_config = NestedConfig.random(seed=301)
        toml_str = nested_config.model_dump_toml()
        print(toml_str[:400] + "...")

        loaded_nested = NestedConfig.model_validate_toml(toml_str)
        print("✓ Nested config deserialized successfully")

    # -------------------------------------------------------------------------
    # 5. Serializing conditional configs
    # -------------------------------------------------------------------------
    print("\n5. Serializing Conditional Configurations:")
    print("-" * 80)

    # Create config with dropout enabled
    config_with_dropout = ConditionalConfig(
        use_dropout=True,
        dropout_rate=0.3,
        use_l2=False,
        l2_strength=0.0,
        learning_rate=0.001,
    )

    print("Config with dropout enabled:")
    json_str = config_with_dropout.model_dump_json(indent=2)
    print(json_str)

    # Deserialize
    loaded = ConditionalConfig.model_validate_json(json_str)
    print("\n✓ Conditional config deserialized")
    print(
        f"  dropout_rate matches: {config_with_dropout.dropout_rate == loaded.dropout_rate}"
    )

    # Create config with dropout disabled
    config_no_dropout = ConditionalConfig(
        use_dropout=False,
        dropout_rate=0.0,
        use_l2=True,
        l2_strength=1e-4,
        learning_rate=0.001,
    )

    print("\nConfig with dropout disabled:")
    json_str = config_no_dropout.model_dump_json(indent=2)
    print(json_str)

    # -------------------------------------------------------------------------
    # 6. Saving to files
    # -------------------------------------------------------------------------
    print("\n6. Saving Configurations to Files:")
    print("-" * 80)

    import os
    import tempfile

    config = NestedConfig.random(seed=400)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save as JSON
        json_path = os.path.join(tmpdir, "config.json")
        with open(json_path, "w") as f:
            f.write(config.model_dump_json(indent=2))
        print(f"✓ Saved to JSON: {json_path}")

        # Load from JSON
        with open(json_path) as f:
            loaded = NestedConfig.model_validate_json(f.read())
        print("✓ Loaded from JSON successfully")

        if YAML_AVAILABLE:
            # Save as YAML
            yaml_path = os.path.join(tmpdir, "config.yaml")
            with open(yaml_path, "w") as f:
                f.write(config.model_dump_yaml())
            print(f"✓ Saved to YAML: {yaml_path}")

            # Load from YAML
            with open(yaml_path) as f:
                loaded = NestedConfig.model_validate_yaml(f.read())
            print("✓ Loaded from YAML successfully")

        if TOML_AVAILABLE:
            # Save as TOML
            toml_path = os.path.join(tmpdir, "config.toml")
            with open(toml_path, "w") as f:
                f.write(config.model_dump_toml())
            print(f"✓ Saved to TOML: {toml_path}")

            # Load from TOML
            with open(toml_path) as f:
                loaded = NestedConfig.model_validate_toml(f.read())
            print("✓ Loaded from TOML successfully")

    # -------------------------------------------------------------------------
    # 7. Comparing formats
    # -------------------------------------------------------------------------
    print("\n7. Comparing Serialization Formats:")
    print("-" * 80)

    config = NestedConfig.random(seed=500)

    json_str = config.model_dump_json(indent=2)
    print(f"JSON size: {len(json_str)} characters")

    if YAML_AVAILABLE:
        yaml_str = config.model_dump_yaml()
        print(f"YAML size: {len(yaml_str)} characters")
        print(f"YAML vs JSON: {len(yaml_str) / len(json_str):.1%}")

    if TOML_AVAILABLE:
        toml_str = config.model_dump_toml()
        print(f"TOML size: {len(toml_str)} characters")
        print(f"TOML vs JSON: {len(toml_str) / len(json_str):.1%}")

    print("\nFormat characteristics:")
    print("  JSON:  Human-readable, widely supported, good for APIs")
    print("  YAML:  Most human-readable, popular in ML/DevOps")
    print("  TOML:  Simple syntax, good for config files")

    # -------------------------------------------------------------------------
    # 8. Best practices
    # -------------------------------------------------------------------------
    print("\n8. Best Practices:")
    print("-" * 80)
    print("""
    ✓ Save configs with experiment results for reproducibility
    ✓ Use version control (git) for config files
    ✓ Include timestamps or version numbers in filenames
    ✓ Choose format based on use case:
        - JSON: APIs, web services, maximum compatibility
        - YAML: Human editing, configuration management
        - TOML: Simple configs, ini-file replacement
    ✓ Always validate after deserialization
    ✓ Use meaningful names: experiment_name_v1.json
    ✓ Keep configs in a dedicated directory structure
    ✓ Document any custom serialization logic
    """)

    # -------------------------------------------------------------------------
    # 9. Common patterns
    # -------------------------------------------------------------------------
    print("\n9. Common Patterns:")
    print("-" * 80)
    print("""
    Pattern 1: Experiment tracking
      # Save config with results
      config = MyConfig.random(seed=42)
      results = train(config)

      # Save both
      with open(f"experiment_{run_id}.json", "w") as f:
          data = {
              "config": config.model_dump(),
              "results": results,
              "timestamp": datetime.now().isoformat()
          }
          json.dump(data, f, indent=2)

    Pattern 2: Configuration inheritance
      # Load base config
      base = MyConfig.model_validate_json(open("base.json").read())

      # Modify and save variant
      data = base.model_dump()
      data["learning_rate"] = 0.001
      variant = MyConfig.model_validate(data)

      with open("variant.json", "w") as f:
          f.write(variant.model_dump_json(indent=2))

    Pattern 3: Batch processing
      # Load multiple configs
      configs = []
      for file in glob("configs/*.json"):
          config = MyConfig.model_validate_json(open(file).read())
          configs.append(config)

      # Process all
      for config in configs:
          train(config)

    Pattern 4: Config versioning
      # Include version in config
      class MyConfig(sp.Config):
          # ... fields ...
          config_version: str = "1.0.0"

      # Check version when loading
      loaded = MyConfig.model_validate_json(data)
      if loaded.config_version != "1.0.0":
          migrate_config(loaded)
    """)

    # -------------------------------------------------------------------------
    # 10. Error handling
    # -------------------------------------------------------------------------
    print("\n10. Error Handling:")
    print("-" * 80)

    # Missing required field
    print("Attempting to load invalid JSON (missing field):")
    try:
        invalid_json = '{"learning_rate": 0.001, "batch_size": 32}'
        loaded = SimpleConfig.model_validate_json(invalid_json)
        print(f"✗ Should have failed: {loaded}")
    except Exception as e:
        print(f"✓ Validation error (expected): {type(e).__name__}")

    # Invalid value
    print("\nAttempting to load invalid JSON (value out of bounds):")
    try:
        invalid_json = """{
            "learning_rate": 10.0,
            "batch_size": 32,
            "num_layers": 5,
            "optimizer": "adam",
            "use_dropout": true
        }"""
        loaded = SimpleConfig.model_validate_json(invalid_json)
        print(f"✗ Should have failed: {loaded}")
    except Exception as e:
        print(f"✓ Validation error (expected): {type(e).__name__}")

    # Wrong type
    print("\nAttempting to load invalid JSON (wrong type):")
    try:
        invalid_json = """{
            "learning_rate": "not a number",
            "batch_size": 32,
            "num_layers": 5,
            "optimizer": "adam",
            "use_dropout": true
        }"""
        loaded = SimpleConfig.model_validate_json(invalid_json)
        print(f"✗ Should have failed: {loaded}")
    except Exception as e:
        print(f"✓ Validation error (expected): {type(e).__name__}")

    print("\n✓ All error cases handled correctly!")


if __name__ == "__main__":
    main()
