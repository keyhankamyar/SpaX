"""Example 02: Nested Configurations

This example demonstrates how to compose complex configurations from
simpler nested Config classes. Nested configs are useful for organizing
related parameters and creating modular, reusable configuration structures.

Topics Covered:
--------------
- Defining nested Config classes
- Sampling nested configurations
- Validation of nested structures
- Accessing nested fields
- Serialization/deserialization of nested configs
"""

import spax as sp

# =============================================================================
# Component Configurations
# =============================================================================


class OptimizerConfig(sp.Config):
    """Configuration for optimizer settings."""

    name: str = sp.Categorical(["adam", "sgd", "rmsprop", "adamw"])
    learning_rate: float = sp.Float(ge=1e-5, le=1e-1, distribution="log")
    weight_decay: float = sp.Float(ge=1e-6, le=1e-2, distribution="log", default=1e-4)

    # Optimizer-specific parameters (fixed for simplicity here)
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999


class ModelConfig(sp.Config):
    """Configuration for model architecture."""

    hidden_dim: int = sp.Int(ge=64, le=1024)
    num_layers: int = sp.Int(ge=1, le=12)
    dropout_rate: float = sp.Float(ge=0.0, le=0.5)
    activation: str = sp.Categorical(["relu", "gelu", "silu", "tanh"])
    use_batch_norm: bool = sp.Categorical([True, False])


class DataConfig(sp.Config):
    """Configuration for data loading."""

    batch_size: int = sp.Int(ge=8, le=256)
    num_workers: int = sp.Int(ge=0, le=16, default=4)
    shuffle: bool = True  # Fixed value
    drop_last: bool = sp.Categorical([True, False], default=False)


# =============================================================================
# Composite Configuration
# =============================================================================


class TrainingConfig(sp.Config):
    """Complete training configuration with nested components."""

    # Nested configurations
    optimizer: OptimizerConfig
    model: ModelConfig
    data: DataConfig

    # Top-level parameters
    num_epochs: int = sp.Int(ge=1, le=1000, default=100)
    seed: int = sp.Int(ge=0, le=9999, default=42)

    # Fixed metadata
    experiment_name: str = "my_experiment"


# =============================================================================
# Deeply Nested Configuration (3 levels)
# =============================================================================


class LayerConfig(sp.Config):
    """Configuration for a single layer."""

    units: int = sp.Int(ge=32, le=512)
    activation: str = sp.Categorical(["relu", "gelu", "silu"])
    dropout: float = sp.Float(ge=0.0, le=0.5)


class BlockConfig(sp.Config):
    """Configuration for a block of layers."""

    layer1: LayerConfig
    layer2: LayerConfig
    num_residual_connections: int = sp.Int(ge=0, le=3, default=1)


class NetworkConfig(sp.Config):
    """Configuration for entire network (3-level nesting)."""

    block1: BlockConfig
    block2: BlockConfig
    global_dropout: float = sp.Float(ge=0.0, le=0.3, default=0.1)


# =============================================================================
# Demonstrations
# =============================================================================


def main():
    print("=" * 80)
    print("Example 02: Nested Configurations")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Creating nested configs manually
    # -------------------------------------------------------------------------
    print("\n1. Creating Nested Configuration Manually:")
    print("-" * 80)

    config = TrainingConfig(
        optimizer=OptimizerConfig(name="adam", learning_rate=0.001, weight_decay=1e-4),
        model=ModelConfig(
            hidden_dim=256,
            num_layers=4,
            dropout_rate=0.1,
            activation="gelu",
            use_batch_norm=True,
        ),
        data=DataConfig(batch_size=32, num_workers=4, drop_last=False),
        num_epochs=100,
        seed=42,
    )

    print("Created nested config:")
    print(f"  Optimizer: {config.optimizer.name}, lr={config.optimizer.learning_rate}")
    print(f"  Model: {config.model.num_layers} layers, dim={config.model.hidden_dim}")
    print(f"  Data: batch_size={config.data.batch_size}")
    print(f"  Training: {config.num_epochs} epochs")

    # -------------------------------------------------------------------------
    # 2. Random sampling of nested configs
    # -------------------------------------------------------------------------
    print("\n2. Random Sampling of Nested Configuration:")
    print("-" * 80)

    print("Sampling 3 complete training configurations:")
    for i in range(3):
        random_config = TrainingConfig.random(seed=100 + i)
        print(f"\n  Sample {i + 1}:")
        print(f"    Optimizer: {random_config.optimizer.name}")
        print(f"    LR: {random_config.optimizer.learning_rate:.6f}")
        print(
            f"    Model: {random_config.model.num_layers} layers × {random_config.model.hidden_dim} units"
        )
        print(f"    Activation: {random_config.model.activation}")
        print(f"    Batch size: {random_config.data.batch_size}")

    # -------------------------------------------------------------------------
    # 3. Accessing nested fields
    # -------------------------------------------------------------------------
    print("\n3. Accessing Nested Fields:")
    print("-" * 80)

    config = TrainingConfig.random(seed=42)

    # Direct access
    print("Direct access:")
    print(f"  config.optimizer.learning_rate = {config.optimizer.learning_rate}")
    print(f"  config.model.hidden_dim = {config.model.hidden_dim}")
    print(f"  config.data.batch_size = {config.data.batch_size}")

    # Nested objects are proper Config instances
    print("\nNested objects are Config instances:")
    print(f"  type(config.optimizer) = {type(config.optimizer).__name__}")
    print(
        f"  isinstance(config.model, ModelConfig) = {isinstance(config.model, ModelConfig)}"
    )

    # -------------------------------------------------------------------------
    # 4. Getting parameter names (hierarchical)
    # -------------------------------------------------------------------------
    print("\n4. Hierarchical Parameter Names:")
    print("-" * 80)

    print("All searchable parameters in the nested structure:")
    params = TrainingConfig.get_parameter_names()
    print(f"Total: {len(params)} parameters\n")

    # Group by component
    optimizer_params = [p for p in params if "OptimizerConfig" in p]
    model_params = [p for p in params if "ModelConfig" in p]
    data_params = [p for p in params if "DataConfig" in p]
    top_level_params = [p for p in params if "::" not in p]

    print("Optimizer parameters:")
    for p in optimizer_params[:3]:
        print(f"  - {p}")

    print("\nModel parameters:")
    for p in model_params[:3]:
        print(f"  - {p}")

    print("\nData parameters:")
    for p in data_params[:3]:
        print(f"  - {p}")

    print("\nTop-level parameters:")
    for p in top_level_params:
        print(f"  - {p}")

    # -------------------------------------------------------------------------
    # 5. Serialization of nested configs
    # -------------------------------------------------------------------------
    print("\n5. Serialization of Nested Configuration:")
    print("-" * 80)

    config = TrainingConfig.random(seed=42)

    # Convert to dict (with type discriminators)
    config_dict = config.model_dump()
    print("Serialized to dict (showing structure):")
    print(f"  Keys: {list(config_dict.keys())}")
    print(f"  optimizer type: {config_dict['optimizer']['__type__']}")
    print(f"  model type: {config_dict['model']['__type__']}")
    print(f"  data type: {config_dict['data']['__type__']}")

    # Convert to JSON
    config_json = config.model_dump_json(indent=2)
    print(f"\nJSON size: {len(config_json)} characters")
    print("JSON preview (first 300 chars):")
    print(config_json[:300] + "...")

    # Deserialize back
    loaded_config = TrainingConfig.model_validate_json(config_json)
    print("\n✓ Deserialized successfully")
    print(f"  Original LR: {config.optimizer.learning_rate}")
    print(f"  Loaded LR:   {loaded_config.optimizer.learning_rate}")
    print(
        f"  Match: {config.optimizer.learning_rate == loaded_config.optimizer.learning_rate}"
    )

    # -------------------------------------------------------------------------
    # 6. Deeply nested configuration (3 levels)
    # -------------------------------------------------------------------------
    print("\n6. Deeply Nested Configuration (3 Levels):")
    print("-" * 80)

    deep_config = NetworkConfig.random(seed=123)

    print("3-level nested structure:")
    print("  Level 1 - NetworkConfig")
    print("    ↓")
    print("  Level 2 - BlockConfig (block1)")
    print("    ↓")
    print("  Level 3 - LayerConfig (layer1)")
    print(f"    → units: {deep_config.block1.layer1.units}")
    print(f"    → activation: {deep_config.block1.layer1.activation}")
    print(f"    → dropout: {deep_config.block1.layer1.dropout:.3f}")

    print("\n  Full path access:")
    print(f"    deep_config.block1.layer1.units = {deep_config.block1.layer1.units}")
    print(
        f"    deep_config.block1.layer2.activation = {deep_config.block1.layer2.activation}"
    )
    print(
        f"    deep_config.block2.layer1.dropout = {deep_config.block2.layer1.dropout:.3f}"
    )

    # -------------------------------------------------------------------------
    # 7. Validation of nested structures
    # -------------------------------------------------------------------------
    print("\n7. Validation of Nested Structures:")
    print("-" * 80)

    # Valid nested config
    try:
        _ = TrainingConfig(
            optimizer=OptimizerConfig(
                name="adam", learning_rate=0.001, weight_decay=1e-4
            ),
            model=ModelConfig(
                hidden_dim=256,
                num_layers=4,
                dropout_rate=0.1,
                activation="gelu",
                use_batch_norm=True,
            ),
            data=DataConfig(batch_size=32, num_workers=4, drop_last=False),
            num_epochs=100,
            seed=42,
        )
        print("✓ Valid nested config created")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Invalid: bad value in nested config
    try:
        invalid = TrainingConfig(
            optimizer=OptimizerConfig(
                name="adam",
                learning_rate=10.0,  # Too high (> 0.1)
                weight_decay=1e-4,
            ),
            model=ModelConfig(
                hidden_dim=256,
                num_layers=4,
                dropout_rate=0.1,
                activation="gelu",
                use_batch_norm=True,
            ),
            data=DataConfig(batch_size=32, num_workers=4, drop_last=False),
            num_epochs=100,
            seed=42,
        )
        print(f"✗ Config created: {invalid}")
    except Exception as e:
        print("✓ Validation error in nested config (expected):")
        print(f"   {type(e).__name__}: {str(e)[:80]}...")

    # -------------------------------------------------------------------------
    # 8. Benefits of nested configs
    # -------------------------------------------------------------------------
    print("\n8. Benefits of Nested Configurations:")
    print("-" * 80)
    print("""
    ✓ Organization: Related parameters grouped together
    ✓ Modularity: Reuse component configs across projects
    ✓ Type Safety: Each nested config validates independently
    ✓ Clarity: Clear hierarchy reflects system architecture
    ✓ Sampling: Random sampling works seamlessly across all levels
    ✓ Serialization: Preserves structure when saving/loading
    """)


if __name__ == "__main__":
    main()
