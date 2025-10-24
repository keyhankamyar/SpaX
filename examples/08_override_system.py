"""Example 08: Override System

This example demonstrates SpaX's override system, which allows you to
iteratively narrow search spaces based on experimental results. This is
crucial for efficient hyperparameter optimization where you want to:

1. Start with a broad search space
2. Run initial experiments
3. Analyze results to identify promising regions
4. Narrow the search space to those regions
5. Repeat until converged

Topics Covered:
--------------
- Basic overrides (narrowing ranges)
- Fixing parameters to specific values
- Overriding categorical choices
- Overriding nested configurations
- Overriding conditional parameters
- Getting override templates
- Practical workflow patterns
"""

import spax as sp

# =============================================================================
# Basic Configuration for Override Demos
# =============================================================================


class MLConfig(sp.Config):
    """Basic ML configuration for override demonstrations."""

    learning_rate: float = sp.Float(ge=1e-5, le=1e-1, distribution="log")
    batch_size: int = sp.Int(ge=8, le=256)
    num_layers: int = sp.Int(ge=1, le=10)
    hidden_dim: int = sp.Int(ge=64, le=1024)
    optimizer: str = sp.Categorical(["adam", "sgd", "rmsprop", "adamw"])
    dropout_rate: float = sp.Float(ge=0.0, le=0.5)


class ConditionalMLConfig(sp.Config):
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


class TrainingConfig(sp.Config):
    """Training configuration with nested components."""

    model: ModelConfig
    optimizer: OptimizerConfig
    batch_size: int = sp.Int(ge=8, le=256)
    num_epochs: int = sp.Int(ge=10, le=1000)


# =============================================================================
# Demonstrations
# =============================================================================


def main():
    print("=" * 80)
    print("Example 08: Override System")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Getting override templates
    # -------------------------------------------------------------------------
    print("\n1. Understanding Override Templates:")
    print("-" * 80)
    print("Override template shows the structure for overriding a config:")
    template = MLConfig.get_override_template()
    print("\nMLConfig override template:")
    for key, value in template.items():
        print(f"  {key}: {value}")

    print("\nThis shows:")
    print("  - Numeric parameters: dict with bound keys (ge/gt/le/lt)")
    print("  - Categorical parameters: list of choices")

    # -------------------------------------------------------------------------
    # 2. Basic numeric overrides (narrowing ranges)
    # -------------------------------------------------------------------------
    print("\n2. Narrowing Numeric Ranges:")
    print("-" * 80)

    # Original space
    print("Original space:")
    original_samples = [MLConfig.random(seed=i) for i in range(5)]
    lrs = [c.learning_rate for c in original_samples]
    dims = [c.hidden_dim for c in original_samples]
    print(f"  Learning rates: {[f'{lr:.6f}' for lr in lrs]}")
    print(f"  Hidden dims: {dims}")
    print(f"  LR range: [{min(lrs):.6f}, {max(lrs):.6f}]")
    print(f"  Dim range: [{min(dims)}, {max(dims)}]")

    # Narrow the search space
    override = {
        "learning_rate": {"ge": 1e-4, "le": 1e-2},  # Narrow from [1e-5, 1e-1]
        "hidden_dim": {"ge": 256, "le": 512},  # Narrow from [64, 1024]
    }

    print("\nWith override (narrowed ranges):")
    print(f"  Override: {override}")
    narrowed_samples = [MLConfig.random(seed=i, override=override) for i in range(5)]
    lrs = [c.learning_rate for c in narrowed_samples]
    dims = [c.hidden_dim for c in narrowed_samples]
    print(f"  Learning rates: {[f'{lr:.6f}' for lr in lrs]}")
    print(f"  Hidden dims: {dims}")
    print(f"  LR range: [{min(lrs):.6f}, {max(lrs):.6f}]")
    print(f"  Dim range: [{min(dims)}, {max(dims)}]")

    # -------------------------------------------------------------------------
    # 3. Fixing parameters to specific values
    # -------------------------------------------------------------------------
    print("\n3. Fixing Parameters to Specific Values:")
    print("-" * 80)

    # Fix some parameters
    override = {
        "learning_rate": 0.001,  # Fix to specific value
        "optimizer": "adam",  # Fix to specific choice
        "num_layers": {"ge": 3, "le": 5},  # Still searchable but narrowed
    }

    print("Override (fixing some params):")
    print("  learning_rate: 0.001 (fixed)")
    print("  optimizer: 'adam' (fixed)")
    print("  num_layers: [3, 5] (narrowed)")

    print("\nSampling with fixed params:")
    for i in range(5):
        config = MLConfig.random(seed=100 + i, override=override)
        print(
            f"  Sample {i + 1}: lr={config.learning_rate:.6f}, "
            f"opt={config.optimizer}, layers={config.num_layers}"
        )
    print("\nNotice: lr and optimizer are always the same!")

    # -------------------------------------------------------------------------
    # 4. Overriding categorical choices
    # -------------------------------------------------------------------------
    print("\n4. Narrowing Categorical Choices:")
    print("-" * 80)

    # Original choices
    print("Original optimizer choices: ['adam', 'sgd', 'rmsprop', 'adamw']")

    # Narrow to subset
    override = {
        "optimizer": ["adam", "adamw"],  # Only these two
    }

    print("\nWith override: ['adam', 'adamw']")
    from collections import Counter

    samples = [MLConfig.random(seed=200 + i, override=override) for i in range(20)]
    opt_counts = Counter(c.optimizer for c in samples)
    print("Optimizer distribution (20 samples):")
    for opt, count in sorted(opt_counts.items()):
        print(f"  {opt}: {count}")
    print("\nNotice: Only 'adam' and 'adamw' appear!")

    # -------------------------------------------------------------------------
    # 5. Overriding nested configurations
    # -------------------------------------------------------------------------
    print("\n5. Overriding Nested Configurations:")
    print("-" * 80)

    # Get nested template
    template = TrainingConfig.get_override_template()
    print("Nested override template structure:")
    print(f"  Keys: {list(template.keys())}")
    print(f"  model keys: {list(template['model'].keys())}")
    print(f"  optimizer keys: {list(template['optimizer'].keys())}")

    # Override nested fields
    override = {
        "model": {
            "hidden_dim": {"ge": 256, "le": 512},
            "num_layers": 4,  # Fix to 4
        },
        "optimizer": {
            "name": "adam",  # Fix optimizer
            "learning_rate": {"ge": 1e-4, "le": 1e-3},
        },
        "batch_size": 64,  # Fix batch size
    }

    print("\nNested override:")
    print("  model.hidden_dim: [256, 512]")
    print("  model.num_layers: 4 (fixed)")
    print("  optimizer.name: 'adam' (fixed)")
    print("  optimizer.learning_rate: [1e-4, 1e-3]")
    print("  batch_size: 64 (fixed)")

    print("\nSampling with nested override:")
    for i in range(3):
        config = TrainingConfig.random(seed=300 + i, override=override)
        print(f"\n  Sample {i + 1}:")
        print(f"    model.hidden_dim: {config.model.hidden_dim}")
        print(f"    model.num_layers: {config.model.num_layers}")
        print(f"    optimizer.name: {config.optimizer.name}")
        print(f"    optimizer.learning_rate: {config.optimizer.learning_rate:.6f}")
        print(f"    batch_size: {config.batch_size}")

    # -------------------------------------------------------------------------
    # 6. Overriding conditional parameters
    # -------------------------------------------------------------------------
    print("\n6. Overriding Conditional Parameters:")
    print("-" * 80)

    print("Original conditional config:")
    samples = [ConditionalMLConfig.random(seed=400 + i) for i in range(10)]
    dropout_enabled = sum(1 for c in samples if c.use_dropout)
    print(f"  Dropout enabled: {dropout_enabled}/10")
    print(f"  L2 enabled: {sum(1 for c in samples if c.use_l2)}/10")

    # Force dropout enabled and narrow its range
    override = {
        "use_dropout": True,  # Always enable dropout
        "dropout_rate": {
            "true": {"gt": 0.2, "lt": 0.4}  # Narrow the range when enabled
        },
    }

    print("\nWith override (force dropout, narrow range):")
    samples = [
        ConditionalMLConfig.random(seed=500 + i, override=override) for i in range(10)
    ]
    print("  Sample dropout rates:")
    for i, c in enumerate(samples[:5]):
        print(
            f"    Sample {i + 1}: use_dropout={c.use_dropout}, rate={c.dropout_rate:.3f}"
        )
    print("\n  Notice: dropout always enabled, rate in [0.2, 0.4]!")

    # -------------------------------------------------------------------------
    # 7. Practical workflow: Iterative refinement
    # -------------------------------------------------------------------------
    print("\n7. Practical Workflow - Iterative Refinement:")
    print("-" * 80)

    print("Iteration 1: Broad search")
    print("  Search space: Full range")
    # Simulate finding that lr=0.001-0.01 and hidden_dim=256-512 work best
    best_configs_iter1 = [
        {"learning_rate": 0.003, "hidden_dim": 384, "score": 0.92},
        {"learning_rate": 0.005, "hidden_dim": 412, "score": 0.91},
        {"learning_rate": 0.002, "hidden_dim": 298, "score": 0.90},
    ]
    print("  Top results:")
    for i, result in enumerate(best_configs_iter1):
        print(
            f"    {i + 1}. lr={result['learning_rate']:.4f}, "
            f"dim={result['hidden_dim']}, score={result['score']}"
        )

    print("\nIteration 2: Narrow to promising region")
    override_iter2 = {
        "learning_rate": {"ge": 0.001, "le": 0.01},
        "hidden_dim": {"ge": 256, "le": 512},
        "optimizer": ["adam", "adamw"],  # Best optimizers from iter 1
    }
    print(f"  Override: {override_iter2}")
    print("  (Running experiments with narrowed space...)")

    print("\nIteration 3: Fine-tune best region")
    override_iter3 = {
        "learning_rate": {"ge": 0.002, "le": 0.005},
        "hidden_dim": {"ge": 350, "le": 450},
        "optimizer": "adam",  # Best optimizer
        "num_layers": {"ge": 3, "le": 5},  # Focus on mid-range
    }
    print(f"  Override: {override_iter3}")
    print("  (Final refinement...)")

    print("\n✓ Converged to optimal configuration!")

    # -------------------------------------------------------------------------
    # 8. Validation and error handling
    # -------------------------------------------------------------------------
    print("\n8. Override Validation:")
    print("-" * 80)

    # Valid override
    try:
        override = {"learning_rate": {"ge": 1e-4, "le": 1e-2}}
        config = MLConfig.random(seed=600, override=override)
        print(f"✓ Valid override: {override}")
        print(f"  Sampled lr: {config.learning_rate:.6f}")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Invalid: range outside original bounds
    print("\nInvalid override (outside original bounds):")
    try:
        override = {"learning_rate": {"ge": 1e-3, "le": 1.0}}  # 1.0 > 0.1
        config = MLConfig.random(seed=601, override=override)
        print(f"✗ Should have failed: {config}")
    except Exception as e:
        print(f"✓ Validation error (expected): {type(e).__name__}")
        print(f"  Message: {str(e)[:80]}...")

    # Invalid: unknown field
    print("\nInvalid override (unknown field):")
    try:
        override = {"unknown_field": 123}
        config = MLConfig.random(seed=602, override=override)
        print(f"✗ Should have failed: {config}")
    except Exception as e:
        print(f"✓ Validation error (expected): {type(e).__name__}")
        print(f"  Message: {str(e)[:80]}...")

    # -------------------------------------------------------------------------
    # 9. Benefits of override system
    # -------------------------------------------------------------------------
    print("\n9. Benefits of Override System:")
    print("-" * 80)
    print("""
    ✓ Iterative Refinement: Start broad, narrow down progressively
    ✓ Efficiency: Focus computational resources on promising regions
    ✓ Flexibility: Mix fixed and searchable parameters
    ✓ Validation: Ensures overrides stay within valid bounds
    ✓ Reproducibility: Same override + seed = same config
    ✓ Documentation: Override history tracks search space evolution
    ✓ No Code Changes: Modify search space without changing config definition
    """)

    # -------------------------------------------------------------------------
    # 10. Best practices
    # -------------------------------------------------------------------------
    print("\n10. Best Practices:")
    print("-" * 80)
    print("""
    Pattern 1: Progressive narrowing
      → Start with 3-5 broad experiments
      → Analyze results, narrow to top 50% of space
      → Repeat until convergence

    Pattern 2: Fix high-confidence parameters early
      → If one choice clearly dominates, fix it
      → Focus search on uncertain parameters

    Pattern 3: Save override history
      → Keep track of overrides used in each iteration
      → Helps understand search process
      → Enables reproduction of results

    Pattern 4: Use templates as starting point
      → get_override_template() shows structure
      → Copy template and modify values
      → Ensures correct override format

    Pattern 5: Validate before large runs
      → Test override with a few samples first
      → Catch errors before expensive experiments
    """)


if __name__ == "__main__":
    main()
