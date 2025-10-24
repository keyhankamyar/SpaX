"""Example 01: Basic Configuration

This example demonstrates the fundamental features of SpaX for defining
simple searchable configurations with numeric and categorical parameters.

Topics Covered:
--------------
- Defining numeric spaces (Int, Float)
- Defining categorical spaces
- Setting defaults
- Adding descriptions
- Random sampling
- Validation
"""

import spax as sp

# =============================================================================
# Basic Configuration
# =============================================================================


class BasicMLConfig(sp.Config):
    """A simple ML configuration with common hyperparameters."""

    # Integer space with inclusive bounds
    num_layers: int = sp.Int(
        ge=1, le=10, default=3, description="Number of hidden layers"
    )

    # Float space with exclusive lower bound
    learning_rate: float = sp.Float(
        gt=0.0,  # Exclusive: must be > 0
        le=0.1,  # Inclusive: can be = 0.1
        default=0.001,
        description="Learning rate for optimizer",
    )

    # Float with log distribution (better for learning rates, etc.)
    weight_decay: float = sp.Float(
        ge=1e-6,
        le=1e-2,
        distribution="log",
        default=1e-4,
        description="L2 regularization strength",
    )

    # Categorical space for discrete choices
    optimizer: str = sp.Categorical(
        ["adam", "sgd", "rmsprop", "adamw"],
        default="adam",
        description="Optimization algorithm",
    )

    # Boolean is just a special categorical
    use_batch_norm: bool = sp.Categorical(
        [True, False], default=True, description="Whether to use batch normalization"
    )

    # Fixed value (not searchable)
    model_name: str = "my_model"


# =============================================================================
# Categorical with Weights
# =============================================================================


class WeightedConfig(sp.Config):
    """Configuration with weighted categorical choices."""

    # Some optimizers are more commonly used - reflect that in sampling
    optimizer: str = sp.Categorical(
        [
            sp.Choice("adam", weight=3.0),  # 3x more likely
            sp.Choice("adamw", weight=2.0),  # 2x more likely
            sp.Choice("sgd", weight=1.0),  # baseline
            sp.Choice("rmsprop", weight=0.5),  # half as likely
        ]
    )

    learning_rate: float = sp.Float(ge=1e-5, le=1e-1, distribution="log")


# =============================================================================
# Distribution Types
# =============================================================================


class DistributionDemo(sp.Config):
    """Demonstrating uniform vs log distributions."""

    # Uniform distribution: samples are evenly spread
    # Good for: batch size, hidden dimensions, etc.
    batch_size: int = sp.Int(ge=16, le=128, distribution="uniform")

    # Log distribution: samples favor smaller values
    # Good for: learning rates, weight decay, dropout, etc.
    learning_rate: float = sp.Float(ge=1e-5, le=1e-1, distribution="log")
    dropout_rate: float = sp.Float(ge=1e-4, le=0.5, distribution="log")


# =============================================================================
# Demonstrations
# =============================================================================


def main():
    print("=" * 80)
    print("Example 01: Basic Configuration")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Creating configs manually
    # -------------------------------------------------------------------------
    print("\n1. Creating Configuration Manually:")
    print("-" * 80)

    config = BasicMLConfig(
        num_layers=5,
        learning_rate=0.001,
        weight_decay=1e-4,
        optimizer="adam",
        use_batch_norm=True,
    )
    print(f"Config: {config}")
    print(f"Type: {type(config)}")

    # -------------------------------------------------------------------------
    # 2. Random sampling
    # -------------------------------------------------------------------------
    print("\n2. Random Sampling:")
    print("-" * 80)

    print("Sampling 5 random configurations (with seed for reproducibility):")
    for i in range(5):
        random_config = BasicMLConfig.random(seed=42 + i)
        print(
            f"  Sample {i + 1}: lr={random_config.learning_rate:.6f}, "
            f"layers={random_config.num_layers}, "
            f"opt={random_config.optimizer}"
        )

    # -------------------------------------------------------------------------
    # 3. Using defaults
    # -------------------------------------------------------------------------
    print("\n3. Defaults:")
    print("-" * 80)

    # You can use defaults by providing a dict with only some fields
    partial_config = BasicMLConfig(
        num_layers=7,
        learning_rate=0.01,
        weight_decay=1e-3,
        optimizer="sgd",
        use_batch_norm=False,
    )
    print(f"Config with some defaults: {partial_config}")
    print(f"model_name (fixed): {partial_config.model_name}")

    # -------------------------------------------------------------------------
    # 4. Validation
    # -------------------------------------------------------------------------
    print("\n4. Validation:")
    print("-" * 80)

    # Valid config
    try:
        valid = BasicMLConfig(
            num_layers=5,
            learning_rate=0.05,
            weight_decay=1e-4,
            optimizer="adam",
            use_batch_norm=True,
        )
        print(f"✓ Valid config created: {valid.num_layers} layers")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Invalid: learning_rate too high
    try:
        invalid = BasicMLConfig(
            num_layers=5,
            learning_rate=0.5,  # > 0.1 (le bound)
            weight_decay=1e-4,
            optimizer="adam",
            use_batch_norm=True,
        )
        print(f"✗ Config created: {invalid}")
    except Exception as e:
        print(f"✓ Validation error (expected): {type(e).__name__}")
        print(f"   Message: {str(e)[:100]}...")

    # Invalid: optimizer not in choices
    try:
        invalid = BasicMLConfig(
            num_layers=5,
            learning_rate=0.05,
            weight_decay=1e-4,
            optimizer="invalid_optimizer",
            use_batch_norm=True,
        )
        print(f"✗ Config created: {invalid}")
    except Exception as e:
        print(f"✓ Validation error (expected): {type(e).__name__}")
        print(f"   Message: {str(e)[:100]}...")

    # -------------------------------------------------------------------------
    # 5. Weighted categorical sampling
    # -------------------------------------------------------------------------
    print("\n5. Weighted Categorical Sampling:")
    print("-" * 80)

    print("Sampling 100 configs and counting optimizer choices:")
    optimizer_counts = {"adam": 0, "adamw": 0, "sgd": 0, "rmsprop": 0}

    for i in range(100):
        config = WeightedConfig.random(seed=1000 + i)
        optimizer_counts[config.optimizer] += 1

    print("Counts (weights were adam=3, adamw=2, sgd=1, rmsprop=0.5):")
    total = sum(optimizer_counts.values())
    for opt, count in sorted(optimizer_counts.items(), key=lambda x: -x[1]):
        percentage = count / total * 100
        print(f"  {opt:10s}: {count:3d} ({percentage:5.1f}%)")

    # -------------------------------------------------------------------------
    # 6. Distribution comparison
    # -------------------------------------------------------------------------
    print("\n6. Uniform vs Log Distribution:")
    print("-" * 80)

    print("Sampling learning rates with log distribution:")
    lrs = [DistributionDemo.random(seed=2000 + i).learning_rate for i in range(10)]
    print(f"  Values: {[f'{lr:.6f}' for lr in lrs[:5]]}")
    print("  Notice: More values closer to 1e-5 than to 1e-1 (log scale)")

    print("\nSampling batch sizes with uniform distribution:")
    batches = [DistributionDemo.random(seed=3000 + i).batch_size for i in range(10)]
    print(f"  Values: {batches[:10]}")
    print("  Notice: Values are more evenly distributed across [16, 128]")

    # -------------------------------------------------------------------------
    # 7. Getting parameter information
    # -------------------------------------------------------------------------
    print("\n7. Inspecting Search Space:")
    print("-" * 80)

    print("Searchable parameters:")
    for param in BasicMLConfig.get_parameter_names():
        print(f"  - {param}")

    print("\nNote: 'model_name' is not listed because it's a fixed value")


if __name__ == "__main__":
    main()
