"""Example 03: Configuration Inheritance

This example demonstrates how to use inheritance to create configuration
hierarchies, share common parameters, and specialize configurations for
different use cases.

Topics Covered:
--------------
- Basic inheritance from Config
- Extending parent configurations
- Override parent parameters
- Abstract base configurations
- Multiple specialized configs from same base
- Inheritance with nested configs
"""

import spax as sp

# =============================================================================
# Basic Inheritance
# =============================================================================


class BaseModelConfig(sp.Config):
    """Base configuration with common parameters."""

    # Common parameters for all models
    hidden_dim: int = sp.Int(ge=64, le=1024)
    dropout_rate: float = sp.Float(ge=0.0, le=0.5)
    activation: str = sp.Categorical(["relu", "gelu", "silu"])
    use_batch_norm: bool = sp.Categorical([True, False])

    # Fixed metadata
    model_type: str = "base"


class MLPConfig(BaseModelConfig):
    """MLP-specific configuration extending base."""

    # Inherit: hidden_dim, dropout_rate, activation, use_batch_norm

    # Add MLP-specific parameters
    num_layers: int = sp.Int(ge=1, le=10)
    use_residual: bool = sp.Categorical([True, False], default=False)

    # Override fixed value
    model_type: str = "mlp"


class CNNConfig(BaseModelConfig):
    """CNN-specific configuration extending base."""

    # Inherit base parameters

    # Add CNN-specific parameters
    num_conv_layers: int = sp.Int(ge=1, le=8)
    kernel_size: int = sp.Int(ge=3, le=7)
    pooling: str = sp.Categorical(["max", "avg", "none"])

    # Override fixed value
    model_type: str = "cnn"


class TransformerConfig(BaseModelConfig):
    """Transformer-specific configuration extending base."""

    # Inherit base parameters

    # Add Transformer-specific parameters
    num_heads: int = sp.Int(ge=1, le=16)
    num_layers: int = sp.Int(ge=1, le=12)
    feedforward_dim: int = sp.Int(ge=256, le=4096)

    # Override fixed value
    model_type: str = "transformer"


# =============================================================================
# Inheritance with Parameter Override
# =============================================================================


class SmallModelConfig(BaseModelConfig):
    """Smaller model with restricted parameter ranges."""

    # Override with narrower bounds
    hidden_dim: int = sp.Int(ge=64, le=256)  # Was ge=64, le=1024

    # Add new parameter
    lightweight: bool = True
    model_type: str = "small"


class LargeModelConfig(BaseModelConfig):
    """Larger model with expanded parameter ranges."""

    # Override with wider bounds
    hidden_dim: int = sp.Int(ge=512, le=2048)  # Was ge=64, le=1024

    # Add new parameter
    heavyweight: bool = True
    model_type: str = "large"


# =============================================================================
# Multi-Level Inheritance
# =============================================================================


class BaseTrainingConfig(sp.Config):
    """Base training configuration."""

    learning_rate: float = sp.Float(ge=1e-5, le=1e-1, distribution="log")
    batch_size: int = sp.Int(ge=8, le=256)
    num_epochs: int = sp.Int(ge=1, le=1000)


class SupervisedTrainingConfig(BaseTrainingConfig):
    """Supervised learning training config."""

    # Inherit base training params

    # Add supervised-specific
    validation_split: float = sp.Float(ge=0.1, le=0.3, default=0.2)
    early_stopping_patience: int = sp.Int(ge=5, le=50, default=10)


class ReinforcementLearningConfig(BaseTrainingConfig):
    """RL training config."""

    # Inherit base training params

    # Add RL-specific
    gamma: float = sp.Float(ge=0.9, le=0.999, default=0.99)
    exploration_rate: float = sp.Float(ge=0.0, le=1.0, default=0.1)


# =============================================================================
# Inheritance with Nested Configs
# =============================================================================


class BaseExperimentConfig(sp.Config):
    """Base experiment configuration."""

    seed: int = sp.Int(ge=0, le=9999, default=42)
    experiment_name: str = "base_experiment"


class MLExperimentConfig(BaseExperimentConfig):
    """ML experiment with model and training configs."""

    # Inherit base params

    # Add nested configs (can use any of the model configs!)
    model: BaseModelConfig  # Will be MLPConfig, CNNConfig, or TransformerConfig
    training: BaseTrainingConfig  # Will be Supervised or RL


# =============================================================================
# Demonstrations
# =============================================================================


def main():
    print("=" * 80)
    print("Example 03: Configuration Inheritance")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Basic inheritance
    # -------------------------------------------------------------------------
    print("\n1. Basic Inheritance:")
    print("-" * 80)

    # Parent config
    base = BaseModelConfig.random(seed=42)
    print("Base config:")
    print(f"  hidden_dim: {base.hidden_dim}")
    print(f"  dropout_rate: {base.dropout_rate:.3f}")
    print(f"  activation: {base.activation}")
    print(f"  model_type: {base.model_type}")

    # Child configs inherit and extend
    mlp = MLPConfig.random(seed=42)
    print("\nMLP config (extends Base):")
    print(f"  hidden_dim: {mlp.hidden_dim} (inherited)")
    print(f"  dropout_rate: {mlp.dropout_rate:.3f} (inherited)")
    print(f"  num_layers: {mlp.num_layers} (new)")
    print(f"  model_type: {mlp.model_type} (overridden)")

    cnn = CNNConfig.random(seed=42)
    print("\nCNN config (extends Base):")
    print(f"  hidden_dim: {cnn.hidden_dim} (inherited)")
    print(f"  num_conv_layers: {cnn.num_conv_layers} (new)")
    print(f"  kernel_size: {cnn.kernel_size} (new)")
    print(f"  model_type: {cnn.model_type} (overridden)")

    # -------------------------------------------------------------------------
    # 2. Parameter space inheritance
    # -------------------------------------------------------------------------
    print("\n2. Parameter Space Inheritance:")
    print("-" * 80)

    print("Base parameters:")
    base_params = BaseModelConfig.get_parameter_names()
    for p in base_params:
        print(f"  - {p}")

    print("\nMLP parameters (inherited + new):")
    mlp_params = MLPConfig.get_parameter_names()
    for p in mlp_params:
        inherited = any(bp.split(".")[-1] == p.split(".")[-1] for bp in base_params)
        marker = "(inherited)" if inherited else "(new)"
        print(f"  - {p} {marker}")

    # -------------------------------------------------------------------------
    # 3. Multiple specializations from same base
    # -------------------------------------------------------------------------
    print("\n3. Multiple Specializations:")
    print("-" * 80)

    print("Sampling different model types from same base:")
    for seed, ConfigClass in [
        (100, MLPConfig),
        (200, CNNConfig),
        (300, TransformerConfig),
    ]:
        config = ConfigClass.random(seed=seed)
        print(f"\n  {ConfigClass.__name__}:")
        print(f"    Type: {config.model_type}")
        print(f"    Hidden dim: {config.hidden_dim} (inherited)")
        print(f"    Activation: {config.activation} (inherited)")
        if hasattr(config, "num_layers"):
            print(f"    Num layers: {config.num_layers}")
        if hasattr(config, "num_conv_layers"):
            print(f"    Conv layers: {config.num_conv_layers}")
        if hasattr(config, "num_heads"):
            print(f"    Num heads: {config.num_heads}")

    # -------------------------------------------------------------------------
    # 4. Overriding parameter ranges
    # -------------------------------------------------------------------------
    print("\n4. Overriding Parameter Ranges:")
    print("-" * 80)

    print("Sampling hidden_dim from different configs:")
    print("  Base range: [64, 1024]")
    base_samples = [BaseModelConfig.random(seed=i).hidden_dim for i in range(10)]
    print(f"    Samples: {base_samples[:5]}")

    print("\n  Small range: [64, 256]")
    small_samples = [SmallModelConfig.random(seed=i).hidden_dim for i in range(10)]
    print(f"    Samples: {small_samples[:5]}")
    print(f"    Max sampled: {max(small_samples)} (≤ 256)")

    print("\n  Large range: [512, 2048]")
    large_samples = [LargeModelConfig.random(seed=i).hidden_dim for i in range(10)]
    print(f"    Samples: {large_samples[:5]}")
    print(f"    Min sampled: {min(large_samples)} (≥ 512)")

    # -------------------------------------------------------------------------
    # 5. Multi-level inheritance
    # -------------------------------------------------------------------------
    print("\n5. Multi-Level Inheritance:")
    print("-" * 80)

    supervised = SupervisedTrainingConfig.random(seed=42)
    print("SupervisedTrainingConfig:")
    print(f"  learning_rate: {supervised.learning_rate:.6f} (from BaseTrainingConfig)")
    print(f"  batch_size: {supervised.batch_size} (from BaseTrainingConfig)")
    print(f"  validation_split: {supervised.validation_split:.2f} (new in Supervised)")
    print(
        f"  early_stopping_patience: {supervised.early_stopping_patience} (new in Supervised)"
    )

    rl = ReinforcementLearningConfig.random(seed=42)
    print("\nReinforcementLearningConfig:")
    print(f"  learning_rate: {rl.learning_rate:.6f} (from BaseTrainingConfig)")
    print(f"  batch_size: {rl.batch_size} (from BaseTrainingConfig)")
    print(f"  gamma: {rl.gamma:.3f} (new in RL)")
    print(f"  exploration_rate: {rl.exploration_rate:.3f} (new in RL)")

    # -------------------------------------------------------------------------
    # 6. Inheritance with nested configs
    # -------------------------------------------------------------------------
    print("\n6. Inheritance with Nested Configs:")
    print("-" * 80)

    # Create experiment with MLP model
    exp1 = MLExperimentConfig(
        seed=42,
        experiment_name="mlp_experiment",
        model=MLPConfig.random(seed=100),
        training=SupervisedTrainingConfig.random(seed=200),
    )
    print("Experiment 1 (MLP + Supervised):")
    print(f"  Model type: {exp1.model.model_type}")
    print(f"  Model layers: {exp1.model.num_layers}")
    print(f"  Training: {type(exp1.training).__name__}")
    print(f"  Validation split: {exp1.training.validation_split:.2f}")

    # Create experiment with CNN model
    exp2 = MLExperimentConfig(
        seed=43,
        experiment_name="cnn_experiment",
        model=CNNConfig.random(seed=101),
        training=ReinforcementLearningConfig.random(seed=201),
    )
    print("\nExperiment 2 (CNN + RL):")
    print(f"  Model type: {exp2.model.model_type}")
    print(f"  Conv layers: {exp2.model.num_conv_layers}")
    print(f"  Training: {type(exp2.training).__name__}")
    print(f"  Gamma: {exp2.training.gamma:.3f}")

    # -------------------------------------------------------------------------
    # 7. Type checking with inheritance
    # -------------------------------------------------------------------------
    print("\n7. Type Checking with Inheritance:")
    print("-" * 80)

    mlp = MLPConfig.random(seed=42)
    cnn = CNNConfig.random(seed=43)

    print(f"isinstance(mlp, MLPConfig): {isinstance(mlp, MLPConfig)}")
    print(f"isinstance(mlp, BaseModelConfig): {isinstance(mlp, BaseModelConfig)}")
    print(f"isinstance(mlp, sp.Config): {isinstance(mlp, sp.Config)}")

    print(f"\nisinstance(cnn, CNNConfig): {isinstance(cnn, CNNConfig)}")
    print(f"isinstance(cnn, BaseModelConfig): {isinstance(cnn, BaseModelConfig)}")
    print(f"isinstance(cnn, MLPConfig): {isinstance(cnn, MLPConfig)}")

    # -------------------------------------------------------------------------
    # 8. Benefits of inheritance
    # -------------------------------------------------------------------------
    print("\n8. Benefits of Inheritance:")
    print("-" * 80)
    print("""
    ✓ Code Reuse: Share common parameters across configs
    ✓ Consistency: Ensure all variants have base parameters
    ✓ Specialization: Add domain-specific parameters as needed
    ✓ Override: Narrow or widen parameter ranges in subclasses
    ✓ Type Safety: Subclasses maintain parent's validation
    ✓ Flexibility: Mix and match base configs in nested structures
    ✓ Maintainability: Change base affects all children
    """)


if __name__ == "__main__":
    main()
