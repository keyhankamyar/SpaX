"""Example 07: Sampling

This example demonstrates various sampling techniques, including basic random
sampling, reproducibility with seeds, sampling with different samplers, and
understanding sampling behavior with different distributions.

Topics Covered:
--------------
- Basic random sampling
- Reproducibility with seeds
- Sampling statistics and distributions
- Custom samplers (for advanced use cases)
- Sampling from nested and conditional configs
"""

import spax as sp

# =============================================================================
# Basic Configuration for Sampling
# =============================================================================


class SimpleConfig(sp.Config):
    """Simple configuration for basic sampling demos."""

    # Uniform distribution
    hidden_dim: int = sp.Int(ge=64, le=512)
    batch_size: int = sp.Int(ge=8, le=128)

    # Log distribution (better for learning rates)
    learning_rate: float = sp.Float(ge=1e-5, le=1e-1, distribution="log")
    weight_decay: float = sp.Float(ge=1e-6, le=1e-2, distribution="log")

    # Categorical
    optimizer: str = sp.Categorical(["adam", "sgd", "rmsprop"])
    activation: str = sp.Categorical(["relu", "gelu", "silu"])


class WeightedConfig(sp.Config):
    """Configuration with weighted categorical choices."""

    learning_rate: float = sp.Float(ge=1e-5, le=1e-1, distribution="log")

    # Some optimizers are more commonly used
    optimizer: str = sp.Categorical(
        [
            sp.Choice("adam", weight=4.0),  # Most common
            sp.Choice("adamw", weight=2.0),  # Common
            sp.Choice("sgd", weight=1.0),  # Less common
            sp.Choice("rmsprop", weight=0.5),  # Rare
        ]
    )


class ConditionalConfig(sp.Config):
    """Configuration with conditional parameters."""

    use_dropout: bool = sp.Categorical([True, False])
    dropout_rate: float = sp.Conditional(
        sp.FieldCondition("use_dropout", sp.EqualsTo(True)),
        true=sp.Float(gt=0.0, lt=0.5),
        false=0.0,
    )

    learning_rate: float = sp.Float(ge=1e-5, le=1e-1, distribution="log")


# =============================================================================
# Nested Configuration
# =============================================================================


class OptimizerConfig(sp.Config):
    """Optimizer configuration."""

    name: str = sp.Categorical(["adam", "sgd", "adamw"])
    learning_rate: float = sp.Float(ge=1e-5, le=1e-1, distribution="log")


class ModelConfig(sp.Config):
    """Model configuration."""

    hidden_dim: int = sp.Int(ge=64, le=512)
    num_layers: int = sp.Int(ge=1, le=10)


class NestedConfig(sp.Config):
    """Nested configuration."""

    model: ModelConfig
    optimizer: OptimizerConfig
    batch_size: int = sp.Int(ge=8, le=128)


# =============================================================================
# Demonstrations
# =============================================================================


def main():
    print("=" * 80)
    print("Example 07: Sampling")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Basic random sampling
    # -------------------------------------------------------------------------
    print("\n1. Basic Random Sampling:")
    print("-" * 80)

    print("Sampling 5 random configurations:")
    for i in range(5):
        config = SimpleConfig.random()
        print(f"  Sample {i + 1}:")
        print(f"    hidden_dim: {config.hidden_dim}")
        print(f"    learning_rate: {config.learning_rate:.6f}")
        print(f"    optimizer: {config.optimizer}")

    # -------------------------------------------------------------------------
    # 2. Reproducibility with seeds
    # -------------------------------------------------------------------------
    print("\n2. Reproducibility with Seeds:")
    print("-" * 80)

    print("Same seed produces same configuration:")
    config1 = SimpleConfig.random(seed=42)
    config2 = SimpleConfig.random(seed=42)

    print("  Config 1 (seed=42):")
    print(f"    hidden_dim: {config1.hidden_dim}")
    print(f"    learning_rate: {config1.learning_rate:.6f}")
    print(f"    optimizer: {config1.optimizer}")

    print("\n  Config 2 (seed=42):")
    print(f"    hidden_dim: {config2.hidden_dim}")
    print(f"    learning_rate: {config2.learning_rate:.6f}")
    print(f"    optimizer: {config2.optimizer}")

    print(f"\n  Identical: {config1 == config2}")

    print("\nDifferent seeds produce different configurations:")
    config3 = SimpleConfig.random(seed=43)
    print("  Config 3 (seed=43):")
    print(f"    hidden_dim: {config3.hidden_dim}")
    print(f"    learning_rate: {config3.learning_rate:.6f}")
    print(f"    optimizer: {config3.optimizer}")

    print(f"\n  Config 1 == Config 3: {config1 == config3}")

    # -------------------------------------------------------------------------
    # 3. Distribution analysis
    # -------------------------------------------------------------------------
    print("\n3. Distribution Analysis:")
    print("-" * 80)

    # Sample many configs
    num_samples = 1000
    configs = [SimpleConfig.random(seed=1000 + i) for i in range(num_samples)]

    # Analyze uniform distribution (hidden_dim)
    hidden_dims = [c.hidden_dim for c in configs]
    print("Uniform distribution (hidden_dim, range [64, 512]):")
    print(f"  Mean: {sum(hidden_dims) / len(hidden_dims):.1f}")
    print(f"  Expected: ~{(64 + 512) / 2:.1f}")
    print(f"  Min: {min(hidden_dims)}, Max: {max(hidden_dims)}")

    # Analyze log distribution (learning_rate)
    import math

    learning_rates = [c.learning_rate for c in configs]
    log_lrs = [math.log10(lr) for lr in learning_rates]
    print("\nLog distribution (learning_rate, range [1e-5, 1e-1]):")
    print(f"  Geometric mean: {10 ** (sum(log_lrs) / len(log_lrs)):.6f}")
    print(f"  Expected: ~{10 ** ((math.log10(1e-5) + math.log10(1e-1)) / 2):.6f}")
    print(f"  Min: {min(learning_rates):.6f}, Max: {max(learning_rates):.6f}")

    # Analyze categorical distribution
    from collections import Counter

    optimizers = [c.optimizer for c in configs]
    optimizer_counts = Counter(optimizers)
    print("\nCategorical distribution (optimizer):")
    for opt, count in sorted(optimizer_counts.items()):
        percentage = count / num_samples * 100
        print(f"  {opt:10s}: {count:4d} ({percentage:5.1f}%)")
    print("  Expected: ~33.3% each (uniform weights)")

    # -------------------------------------------------------------------------
    # 4. Weighted categorical sampling
    # -------------------------------------------------------------------------
    print("\n4. Weighted Categorical Sampling:")
    print("-" * 80)

    configs = [WeightedConfig.random(seed=2000 + i) for i in range(num_samples)]
    optimizers = [c.optimizer for c in configs]
    optimizer_counts = Counter(optimizers)

    print("Sampling with weights (adam=4, adamw=2, sgd=1, rmsprop=0.5):")
    total_weight = 4.0 + 2.0 + 1.0 + 0.5
    expected = {
        "adam": 4.0 / total_weight * 100,
        "adamw": 2.0 / total_weight * 100,
        "sgd": 1.0 / total_weight * 100,
        "rmsprop": 0.5 / total_weight * 100,
    }

    for opt in ["adam", "adamw", "sgd", "rmsprop"]:
        count = optimizer_counts[opt]
        percentage = count / num_samples * 100
        exp_pct = expected[opt]
        print(f"  {opt:10s}: {count:4d} ({percentage:5.1f}%), expected ~{exp_pct:.1f}%")

    # -------------------------------------------------------------------------
    # 5. Sampling with conditionals
    # -------------------------------------------------------------------------
    print("\n5. Sampling with Conditional Parameters:")
    print("-" * 80)

    print("Sampling 10 configs with conditional dropout:")
    for i in range(10):
        config = ConditionalConfig.random(seed=3000 + i)
        print(
            f"  Sample {i + 1}: use_dropout={config.use_dropout}, "
            f"dropout_rate={config.dropout_rate:.3f}"
        )

    print("\nNotice: When use_dropout=False, dropout_rate is always 0.0")

    # Statistics
    configs = [ConditionalConfig.random(seed=3000 + i) for i in range(num_samples)]
    dropout_enabled = sum(1 for c in configs if c.use_dropout)
    dropout_rates = [c.dropout_rate for c in configs if c.use_dropout]

    print(f"\nStatistics from {num_samples} samples:")
    print(
        f"  Dropout enabled: {dropout_enabled} ({dropout_enabled / num_samples * 100:.1f}%)"
    )
    if dropout_rates:
        print(
            f"  Avg dropout rate (when enabled): {sum(dropout_rates) / len(dropout_rates):.3f}"
        )
        print("  Expected: ~0.25 (midpoint of (0, 0.5))")

    # -------------------------------------------------------------------------
    # 6. Sampling nested configurations
    # -------------------------------------------------------------------------
    print("\n6. Sampling Nested Configurations:")
    print("-" * 80)

    print("Sampling nested config (samples all nested fields):")
    for i in range(3):
        config = NestedConfig.random(seed=4000 + i)
        print(f"\n  Sample {i + 1}:")
        print(f"    model.hidden_dim: {config.model.hidden_dim}")
        print(f"    model.num_layers: {config.model.num_layers}")
        print(f"    optimizer.name: {config.optimizer.name}")
        print(f"    optimizer.learning_rate: {config.optimizer.learning_rate:.6f}")
        print(f"    batch_size: {config.batch_size}")

    # -------------------------------------------------------------------------
    # 7. Generating multiple samples efficiently
    # -------------------------------------------------------------------------
    print("\n7. Generating Multiple Samples:")
    print("-" * 80)

    import time

    # Time sequential sampling
    start = time.time()
    samples = []
    for i in range(1000):
        samples.append(SimpleConfig.random(seed=5000 + i))
    elapsed = time.time() - start

    print("Generated 1000 samples:")
    print(f"  Time: {elapsed:.3f} seconds")
    print(f"  Per sample: {elapsed / 1000 * 1000:.2f} ms")

    # Verify uniqueness with different seeds
    unique_configs = len({(c.hidden_dim, c.batch_size, c.optimizer) for c in samples})
    print(f"  Unique configs: {unique_configs}/1000")

    # -------------------------------------------------------------------------
    # 8. Using RandomSampler directly
    # -------------------------------------------------------------------------
    print("\n8. Using RandomSampler Directly:")
    print("-" * 80)

    print("Creating a sampler and using it multiple times:")
    sampler = sp.RandomSampler(seed=42)

    print("  First sample:")
    config1 = SimpleConfig.random(seed=42)  # Using class method
    print(f"    hidden_dim: {config1.hidden_dim}")

    print("\n  Using same sampler (different from first due to RNG state):")
    # Note: This would require a custom sampling method that accepts a sampler
    # For now, we'll just show the sampler can be created
    print("    Sampler created with seed=42")
    print(f"    Record (empty initially): {sampler.record}")

    # -------------------------------------------------------------------------
    # 9. Sampling best practices
    # -------------------------------------------------------------------------
    print("\n9. Sampling Best Practices:")
    print("-" * 80)
    print("""
    ✓ Always use seeds for reproducibility in experiments
    ✓ Use different seeds for different runs
    ✓ Log the seed with experiment results
    ✓ Use log distribution for learning rates, weight decay
    ✓ Use uniform distribution for architectural parameters
    ✓ Set appropriate weights for categorical choices
    ✓ Sample multiple configs to explore the space
    ✓ Analyze sampling distributions to verify correctness
    """)

    # -------------------------------------------------------------------------
    # 10. Common patterns
    # -------------------------------------------------------------------------
    print("\n10. Common Patterns:")
    print("-" * 80)
    print("""
    Pattern 1: Grid search alternative
      → Sample many random configs instead of grid
      → Often more efficient for high-dimensional spaces

    Pattern 2: Quick sanity checking
      → Sample a few configs to verify they work
      → Catches validation errors early

    Pattern 3: Reproducible experiments
      → Use seed for main experiment
      → Use different seeds for reruns/ablations

    Pattern 4: Diversity exploration
      → Sample without seed for exploration
      → Use seed when you find promising configs

    Pattern 5: Integration with HPO
      → Random sampling as baseline
      → Compare with Bayesian optimization (Optuna)
    """)


if __name__ == "__main__":
    main()
