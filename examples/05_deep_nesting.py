"""Example 05: Deep Nesting

This example demonstrates multi-level deeply nested configurations, which
are useful for representing complex hierarchical systems like neural networks
with multiple stages, pipelines with multiple components, or systems with
deeply nested subsystems.

Topics Covered:
--------------
- 3+ level nested configurations
- Accessing deeply nested fields
- Sampling deeply nested structures
- Serialization of deep hierarchies
- Parameter names in deep structures
- Practical patterns for deep configs
"""

import spax as sp

# =============================================================================
# 3-Level Nesting: Network Architecture
# =============================================================================


class LayerConfig(sp.Config):
    """Configuration for a single layer (Level 3)."""

    units: int = sp.Int(ge=32, le=512)
    activation: str = sp.Categorical(["relu", "gelu", "silu"])
    dropout: float = sp.Float(ge=0.0, le=0.5)
    use_bias: bool = sp.Categorical([True, False])


class BlockConfig(sp.Config):
    """Configuration for a block of layers (Level 2)."""

    layer1: LayerConfig
    layer2: LayerConfig
    layer3: LayerConfig

    # Block-level parameters
    num_residual_connections: int = sp.Int(ge=0, le=3, default=1)
    block_dropout: float = sp.Float(ge=0.0, le=0.3, default=0.1)


class NetworkConfig(sp.Config):
    """Configuration for entire network (Level 1 - Root)."""

    # Multiple blocks
    encoder_block: BlockConfig
    decoder_block: BlockConfig

    # Network-level parameters
    global_dropout: float = sp.Float(ge=0.0, le=0.2, default=0.05)
    use_layer_norm: bool = sp.Categorical([True, False])


# =============================================================================
# 4-Level Nesting: ML Pipeline
# =============================================================================


class PreprocessorConfig(sp.Config):
    """Data preprocessing configuration (Level 4)."""

    normalize: bool = sp.Categorical([True, False])
    augment: bool = sp.Categorical([True, False])
    augmentation_strength: float = sp.Float(ge=0.0, le=1.0, default=0.5)


class DataConfig(sp.Config):
    """Data configuration (Level 3)."""

    preprocessor: PreprocessorConfig

    batch_size: int = sp.Int(ge=8, le=256)
    shuffle: bool = True


class OptimizerConfig(sp.Config):
    """Optimizer configuration (Level 3)."""

    name: str = sp.Categorical(["adam", "sgd", "adamw"])
    learning_rate: float = sp.Float(ge=1e-5, le=1e-1, distribution="log")
    weight_decay: float = sp.Float(ge=1e-6, le=1e-2, distribution="log")


class TrainingConfig(sp.Config):
    """Training configuration (Level 2)."""

    optimizer: OptimizerConfig
    data: DataConfig

    num_epochs: int = sp.Int(ge=1, le=1000, default=100)
    early_stopping_patience: int = sp.Int(ge=5, le=50, default=10)


class ModelConfig(sp.Config):
    """Model configuration (Level 2)."""

    architecture: NetworkConfig  # This is itself 3-level nested!

    embedding_dim: int = sp.Int(ge=64, le=512)
    num_classes: int = sp.Int(ge=2, le=1000, default=10)


class ExperimentConfig(sp.Config):
    """Complete experiment configuration (Level 1 - Root)."""

    model: ModelConfig
    training: TrainingConfig

    # Experiment metadata
    seed: int = sp.Int(ge=0, le=9999, default=42)
    experiment_name: str = "deep_nested_experiment"


# =============================================================================
# 5-Level Nesting: Complex System
# =============================================================================


class FeatureExtractorConfig(sp.Config):
    """Feature extractor configuration (Level 5)."""

    method: str = sp.Categorical(["pca", "autoencoder", "raw"])
    n_components: int = sp.Int(ge=10, le=100, default=50)


class SensorConfig(sp.Config):
    """Sensor configuration (Level 4)."""

    extractor: FeatureExtractorConfig

    sampling_rate: int = sp.Int(ge=10, le=1000)
    buffer_size: int = sp.Int(ge=100, le=10000)


class SensorArrayConfig(sp.Config):
    """Array of sensors (Level 3)."""

    sensor_a: SensorConfig
    sensor_b: SensorConfig
    sensor_c: SensorConfig

    fusion_method: str = sp.Categorical(["average", "weighted", "learned"])


class SubsystemConfig(sp.Config):
    """Subsystem configuration (Level 2)."""

    sensors: SensorArrayConfig

    processing_frequency: int = sp.Int(ge=1, le=100)
    enable_logging: bool = True


class SystemConfig(sp.Config):
    """Complete system configuration (Level 1 - Root)."""

    subsystem_1: SubsystemConfig
    subsystem_2: SubsystemConfig

    system_name: str = "deep_system"
    version: str = "1.0.0"


# =============================================================================
# Demonstrations
# =============================================================================


def main():
    print("=" * 80)
    print("Example 05: Deep Nesting")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Accessing deeply nested fields (3 levels)
    # -------------------------------------------------------------------------
    print("\n1. Accessing Deeply Nested Fields (3 Levels):")
    print("-" * 80)

    network = NetworkConfig.random(seed=42)

    print("Full path access:")
    print(
        f"  network.encoder_block.layer1.units = {network.encoder_block.layer1.units}"
    )
    print(
        f"  network.encoder_block.layer1.activation = {network.encoder_block.layer1.activation}"
    )
    print(
        f"  network.encoder_block.layer1.dropout = {network.encoder_block.layer1.dropout:.3f}"
    )
    print(
        f"  network.decoder_block.layer3.units = {network.decoder_block.layer3.units}"
    )

    print("\nIntermediate access:")
    encoder = network.encoder_block
    print(f"  encoder.layer1.units = {encoder.layer1.units}")
    print(f"  encoder.layer2.activation = {encoder.layer2.activation}")
    print(f"  encoder.num_residual_connections = {encoder.num_residual_connections}")

    # -------------------------------------------------------------------------
    # 2. Parameter names in deep structures
    # -------------------------------------------------------------------------
    print("\n2. Hierarchical Parameter Names:")
    print("-" * 80)

    params = NetworkConfig.get_parameter_names()
    print(f"Total parameters: {len(params)}\n")

    # Show a few examples at each level
    print("Sample parameter names (showing hierarchy):")
    for param in params[:8]:
        level = param.count("::") + param.count(".")
        indent = "  " * level
        short_name = param.split("::")[-1] if "::" in param else param
        print(f"{indent}{short_name}")

    print(f"\n... and {len(params) - 8} more parameters")

    # -------------------------------------------------------------------------
    # 3. Sampling deeply nested structures (4 levels)
    # -------------------------------------------------------------------------
    print("\n3. Sampling 4-Level Nested Structure:")
    print("-" * 80)

    experiment = ExperimentConfig.random(seed=100)

    print("Sampled experiment configuration:")
    print(f"  Experiment name: {experiment.experiment_name}")
    print(f"  Seed: {experiment.seed}")
    print("\n  Model:")
    print(f"    Embedding dim: {experiment.model.embedding_dim}")
    print(f"    Num classes: {experiment.model.num_classes}")
    print(
        f"    Architecture encoder layer1 units: {experiment.model.architecture.encoder_block.layer1.units}"
    )
    print("\n  Training:")
    print(f"    Optimizer: {experiment.training.optimizer.name}")
    print(f"    Learning rate: {experiment.training.optimizer.learning_rate:.6f}")
    print(f"    Batch size: {experiment.training.data.batch_size}")
    print(
        f"    Preprocessor normalize: {experiment.training.data.preprocessor.normalize}"
    )
    print(f"    Augmentation: {experiment.training.data.preprocessor.augment}")

    # -------------------------------------------------------------------------
    # 4. Deep nesting with 5 levels
    # -------------------------------------------------------------------------
    print("\n4. Five-Level Deep Nesting:")
    print("-" * 80)

    system = SystemConfig.random(seed=200)

    print("Accessing through 5 levels:")
    print("  Level 1 (Root): system")
    print("  Level 2: system.subsystem_1")
    print("  Level 3: system.subsystem_1.sensors")
    print("  Level 4: system.subsystem_1.sensors.sensor_a")
    print("  Level 5: system.subsystem_1.sensors.sensor_a.extractor")
    print(
        f"\n  Final value: method = {system.subsystem_1.sensors.sensor_a.extractor.method}"
    )

    print("\n  Another path:")
    print(
        f"    system.subsystem_2.sensors.sensor_c.extractor.n_components = "
        f"{system.subsystem_2.sensors.sensor_c.extractor.n_components}"
    )

    # -------------------------------------------------------------------------
    # 5. Serialization of deep hierarchies
    # -------------------------------------------------------------------------
    print("\n5. Serialization of Deep Hierarchies:")
    print("-" * 80)

    experiment = ExperimentConfig.random(seed=300)

    # Serialize to JSON
    json_str = experiment.model_dump_json(indent=2)
    print("JSON serialization:")
    print(f"  Size: {len(json_str)} characters")
    print("  Preview (first 400 chars):")
    print("  " + "\n  ".join(json_str[:400].split("\n")))
    print("  ...")

    # Deserialize
    loaded = ExperimentConfig.model_validate_json(json_str)
    print("\n✓ Deserialization successful")
    print(f"  Original optimizer: {experiment.training.optimizer.name}")
    print(f"  Loaded optimizer: {loaded.training.optimizer.name}")
    print(
        f"  Match: {experiment.training.optimizer.name == loaded.training.optimizer.name}"
    )

    # -------------------------------------------------------------------------
    # 6. Modifying deeply nested values
    # -------------------------------------------------------------------------
    print("\n6. Modifying Deeply Nested Values:")
    print("-" * 80)

    experiment = ExperimentConfig.random(seed=400)

    print("Original values:")
    print(f"  LR: {experiment.training.optimizer.learning_rate:.6f}")
    print(f"  Batch size: {experiment.training.data.batch_size}")
    print(
        f"  Encoder layer1 units: {experiment.model.architecture.encoder_block.layer1.units}"
    )

    # Modify through dict representation
    data = experiment.model_dump()
    data["training"]["optimizer"]["learning_rate"] = 0.005
    data["training"]["data"]["batch_size"] = 64
    data["model"]["architecture"]["encoder_block"]["layer1"]["units"] = 256

    # Create new config from modified data
    modified = ExperimentConfig.model_validate(data)

    print("\nModified values:")
    print(f"  LR: {modified.training.optimizer.learning_rate:.6f}")
    print(f"  Batch size: {modified.training.data.batch_size}")
    print(
        f"  Encoder layer1 units: {modified.model.architecture.encoder_block.layer1.units}"
    )

    # -------------------------------------------------------------------------
    # 7. Practical patterns for deep configs
    # -------------------------------------------------------------------------
    print("\n7. Practical Patterns:")
    print("-" * 80)

    print("""
    Pattern 1: Component-based organization
      → Each level represents a logical component
      → Easy to reason about and modify
      → Example: Model → Architecture → Block → Layer

    Pattern 2: Subsystem composition
      → Complex systems built from simpler subsystems
      → Each subsystem is independently configurable
      → Example: Experiment → Training → Optimizer

    Pattern 3: Reusable building blocks
      → Define small configs and compose them
      → Same config class used multiple times
      → Example: Multiple LayerConfigs in one BlockConfig

    Pattern 4: Hierarchical parameters
      → Global parameters at top level
      → Local parameters at lower levels
      → Example: global_dropout (network) vs dropout (layer)
    """)

    # -------------------------------------------------------------------------
    # 8. Performance considerations
    # -------------------------------------------------------------------------
    print("\n8. Performance with Deep Nesting:")
    print("-" * 80)

    import time

    # Time random sampling
    start = time.time()
    for i in range(100):
        _ = ExperimentConfig.random(seed=500 + i)
    elapsed = time.time() - start

    print("Sampling 100 configs with 4-level nesting:")
    print(f"  Total time: {elapsed:.3f} seconds")
    print(f"  Per config: {elapsed / 100 * 1000:.2f} ms")

    # Time 5-level nesting
    start = time.time()
    for i in range(100):
        _ = SystemConfig.random(seed=600 + i)
    elapsed = time.time() - start

    print("\nSampling 100 configs with 5-level nesting:")
    print(f"  Total time: {elapsed:.3f} seconds")
    print(f"  Per config: {elapsed / 100 * 1000:.2f} ms")

    print("\n✓ Deep nesting has minimal performance overhead")

    # -------------------------------------------------------------------------
    # 9. Tips for working with deep configs
    # -------------------------------------------------------------------------
    print("\n9. Tips for Deep Configurations:")
    print("-" * 80)
    print("""
    ✓ Keep nesting depth reasonable (3-5 levels typical)
    ✓ Use descriptive names at each level for clarity
    ✓ Group related parameters together
    ✓ Consider serialization/deserialization needs
    ✓ Use type hints for better IDE support
    ✓ Document the hierarchy structure
    ✓ Test serialization roundtrips
    ✓ Use get_parameter_names() to understand structure
    """)


if __name__ == "__main__":
    main()
