"""Example 09: Sanity Checking and Debugging

This example demonstrates how to use SpaX for quick sanity checking and
debugging of configurations. One of the main pain points SpaX solves is
making it easy to quickly test configurations with random valid values
instead of manually thinking of and typing parameter values.

Topics Covered:
--------------
- Quick sanity checking with random sampling
- Debugging configuration definitions
- Finding validation errors early
- Testing with edge cases
- Validating conditional logic
- Checking nested configurations
- Common pitfalls and how to avoid them
"""

import spax as sp

# =============================================================================
# Example: Debugging a Model Configuration
# =============================================================================


class NeuralNetworkConfig(sp.Config):
    """Neural network configuration for sanity checking."""

    input_dim: int = sp.Int(ge=1, le=10000)
    hidden_dim: int = sp.Int(ge=32, le=2048)
    output_dim: int = sp.Int(ge=1, le=1000)
    num_layers: int = sp.Int(ge=1, le=20)
    dropout_rate: float = sp.Float(ge=0.0, le=0.5)
    learning_rate: float = sp.Float(ge=1e-6, le=1e-1, distribution="log")
    batch_size: int = sp.Int(ge=1, le=1024)


# =============================================================================
# Example: Configuration with Potential Issues
# =============================================================================


class ProblematicConfig(sp.Config):
    """Configuration that might have issues - let's debug it!"""

    # Issue 1: Very wide range - might sample extreme values
    learning_rate: float = sp.Float(ge=1e-10, le=1.0, distribution="log")

    # Issue 2: Small range - might not matter much
    dropout_rate: float = sp.Float(ge=0.1, le=0.15)

    # Issue 3: Conditional that might rarely trigger
    use_advanced_feature: bool = sp.Categorical(
        [
            sp.Choice(True, weight=0.1),  # Only 10% chance
            sp.Choice(False, weight=0.9),
        ]
    )
    advanced_param: float = sp.Conditional(
        sp.FieldCondition("use_advanced_feature", sp.EqualsTo(True)),
        true=sp.Float(ge=0.0, le=1.0),
        false=0.0,
    )


# =============================================================================
# Example: Complex Conditional Logic to Debug
# =============================================================================


class ComplexConditionalConfig(sp.Config):
    """Configuration with complex conditional logic to verify."""

    mode: str = sp.Categorical(["train", "eval", "test"])

    # Different learning rates for different modes
    learning_rate: float = sp.Conditional(
        sp.FieldCondition("mode", sp.EqualsTo("train")),
        true=sp.Float(ge=1e-5, le=1e-2, distribution="log"),
        false=sp.Conditional(
            sp.FieldCondition("mode", sp.EqualsTo("eval")),
            true=0.0,  # No learning in eval
            false=0.0,  # No learning in test either
        ),
    )

    # Batch size varies by mode
    batch_size: int = sp.Conditional(
        sp.FieldCondition("mode", sp.In(["train", "eval"])),
        true=sp.Int(ge=16, le=128),
        false=sp.Int(ge=1, le=32),  # Smaller batches for test
    )

    use_dropout: bool = sp.Conditional(
        sp.FieldCondition("mode", sp.EqualsTo("train")),
        true=sp.Categorical([True, False]),
        false=False,  # Never use dropout in eval/test
    )


# =============================================================================
# Example: Nested Configuration to Verify
# =============================================================================


class DataLoaderConfig(sp.Config):
    """Data loader configuration."""

    batch_size: int = sp.Int(ge=1, le=512)
    num_workers: int = sp.Int(ge=0, le=16)
    shuffle: bool = sp.Categorical([True, False])


class OptimizerConfig(sp.Config):
    """Optimizer configuration."""

    name: str = sp.Categorical(["adam", "sgd", "rmsprop"])
    learning_rate: float = sp.Float(ge=1e-5, le=1e-1, distribution="log")


class TrainingPipelineConfig(sp.Config):
    """Complete training pipeline to sanity check."""

    model: NeuralNetworkConfig
    optimizer: OptimizerConfig
    data_loader: DataLoaderConfig
    num_epochs: int = sp.Int(ge=1, le=1000)


# =============================================================================
# Demonstrations
# =============================================================================


def main():
    print("=" * 80)
    print("Example 09: Sanity Checking and Debugging")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Quick sanity check: Does the config work at all?
    # -------------------------------------------------------------------------
    print("\n1. Quick Sanity Check - Does It Work?")
    print("-" * 80)
    print("Testing NeuralNetworkConfig with 5 random samples:")

    try:
        for i in range(5):
            config = NeuralNetworkConfig.random(seed=100 + i)
            print(f"  ✓ Sample {i + 1} created successfully")
            # Quick visual check of values
            print(
                f"    dims: {config.input_dim}→{config.hidden_dim}→{config.output_dim}"
            )
        print("\n✓ Configuration works! All samples created successfully.")
    except Exception as e:
        print("\n✗ Configuration has issues!")
        print(f"  Error: {type(e).__name__}: {e}")

    # -------------------------------------------------------------------------
    # 2. Testing parameter ranges
    # -------------------------------------------------------------------------
    print("\n2. Testing Parameter Ranges:")
    print("-" * 80)
    print("Sample 20 configs and check if values look reasonable:")

    configs = [NeuralNetworkConfig.random(seed=200 + i) for i in range(20)]

    # Check learning rates
    lrs = [c.learning_rate for c in configs]
    print("\nLearning rates:")
    print(f"  Min: {min(lrs):.6e}")
    print(f"  Max: {max(lrs):.6e}")
    print(f"  Sample: {[f'{lr:.6e}' for lr in lrs[:5]]}")

    # Check batch sizes
    batches = [c.batch_size for c in configs]
    print("\nBatch sizes:")
    print(f"  Min: {min(batches)}")
    print(f"  Max: {max(batches)}")
    print(f"  Sample: {batches[:5]}")

    # Check if dimensions make sense
    print("\nDimension sanity:")
    for i, c in enumerate(configs[:3]):
        print(f"  Config {i + 1}: {c.input_dim}→{c.hidden_dim}→{c.output_dim}")
        if c.hidden_dim < c.input_dim and c.hidden_dim < c.output_dim:
            print("    ⚠️  Hidden dim smaller than both input and output (bottleneck)")

    # -------------------------------------------------------------------------
    # 3. Debugging problematic configurations
    # -------------------------------------------------------------------------
    print("\n3. Debugging Problematic Configurations:")
    print("-" * 80)
    print("ProblematicConfig has some issues. Let's investigate:")

    configs = [ProblematicConfig.random(seed=300 + i) for i in range(50)]

    # Issue 1: Very wide learning rate range
    lrs = [c.learning_rate for c in configs]
    print("\nIssue 1 - Learning rate range too wide:")
    print("  Range: [1e-10, 1.0]")
    print(f"  Min sampled: {min(lrs):.6e}")
    print(f"  Max sampled: {max(lrs):.6e}")
    print(f"  Median: {sorted(lrs)[len(lrs) // 2]:.6e}")
    print("  ⚠️  Such extreme values (1e-10 or 1.0) are likely invalid!")
    print("  💡 Recommendation: Narrow to [1e-5, 1e-1]")

    # Issue 2: Very narrow dropout range
    dropouts = [c.dropout_rate for c in configs]
    print("\nIssue 2 - Dropout range too narrow:")
    print("  Range: [0.1, 0.15]")
    print(f"  Unique values: {len(set(dropouts))}")
    print(f"  Variation: {max(dropouts) - min(dropouts):.3f}")
    print("  ⚠️  Such a narrow range might not matter much!")
    print("  💡 Recommendation: Either fix to 0.1 or widen to [0.0, 0.5]")

    # Issue 3: Rare conditional trigger
    advanced_enabled = sum(1 for c in configs if c.use_advanced_feature)
    print("\nIssue 3 - Rare conditional trigger:")
    print(f"  use_advanced_feature=True: {advanced_enabled}/50")
    print(f"  ⚠️  Feature only active {advanced_enabled / 50 * 100:.1f}% of the time!")
    print("  💡 Recommendation: Adjust weights or test separately with override")

    # -------------------------------------------------------------------------
    # 4. Verifying conditional logic
    # -------------------------------------------------------------------------
    print("\n4. Verifying Conditional Logic:")
    print("-" * 80)
    print("Testing ComplexConditionalConfig to verify conditions work correctly:")

    print("\nSampling configs in each mode:")
    for mode in ["train", "eval", "test"]:
        # Force the mode with override
        override = {"mode": mode}
        configs = [
            ComplexConditionalConfig.random(seed=400 + i, override=override)
            for i in range(10)
        ]

        lrs = [c.learning_rate for c in configs]
        batches = [c.batch_size for c in configs]
        dropouts = [c.use_dropout for c in configs]

        print(f"\n  Mode: {mode}")
        print(f"    Learning rates: {set(lrs)}")
        print(f"    Batch size range: [{min(batches)}, {max(batches)}]")
        print(f"    Dropout enabled: {sum(dropouts)}/10")

        # Verify expectations
        if mode == "train":
            if all(lr > 0 for lr in lrs):
                print("    ✓ Learning rates > 0 (correct for training)")
            if any(dropouts):
                print("    ✓ Dropout sometimes enabled (correct for training)")
        else:
            if all(lr == 0 for lr in lrs):
                print(f"    ✓ Learning rate = 0 (correct for {mode})")
            if not any(dropouts):
                print(f"    ✓ Dropout never enabled (correct for {mode})")

    # -------------------------------------------------------------------------
    # 5. Testing nested configurations
    # -------------------------------------------------------------------------
    print("\n5. Testing Nested Configurations:")
    print("-" * 80)
    print("Sanity check TrainingPipelineConfig (nested 3 levels):")

    try:
        for i in range(3):
            config = TrainingPipelineConfig.random(seed=500 + i)
            print(f"\n  ✓ Pipeline {i + 1} created successfully:")
            print(
                f"    Model: {config.model.num_layers} layers × {config.model.hidden_dim}D"
            )
            print(
                f"    Optimizer: {config.optimizer.name}, lr={config.optimizer.learning_rate:.6f}"
            )
            print(
                f"    Data: batch={config.data_loader.batch_size}, workers={config.data_loader.num_workers}"
            )
            print(f"    Training: {config.num_epochs} epochs")
    except Exception as e:
        print("\n  ✗ Error creating pipeline:")
        print(f"    {type(e).__name__}: {e}")

    # -------------------------------------------------------------------------
    # 6. Testing edge cases
    # -------------------------------------------------------------------------
    print("\n6. Testing Edge Cases:")
    print("-" * 80)
    print("Deliberately testing boundary values:")

    # Force edge cases with overrides
    edge_cases = [
        {
            "name": "Minimum values",
            "override": {
                "input_dim": 1,
                "hidden_dim": 32,
                "output_dim": 1,
                "num_layers": 1,
                "dropout_rate": 0.0,
                "learning_rate": 1e-6,
                "batch_size": 1,
            },
        },
        {
            "name": "Maximum values",
            "override": {
                "input_dim": 10000,
                "hidden_dim": 2048,
                "output_dim": 1000,
                "num_layers": 20,
                "dropout_rate": 0.5,
                "learning_rate": 1e-1,
                "batch_size": 1024,
            },
        },
        {
            "name": "Extreme aspect ratio",
            "override": {
                "input_dim": 10000,
                "hidden_dim": 32,  # Huge bottleneck
                "output_dim": 1000,
            },
        },
    ]

    for case in edge_cases:
        try:
            config = NeuralNetworkConfig.random(seed=600, override=case["override"])
            print(f"\n  ✓ {case['name']}:")
            print("    Config created successfully")
            if "hidden_dim" in case["override"]:
                ratio = (
                    case["override"].get("input_dim", 0)
                    / case["override"]["hidden_dim"]
                )
                if ratio > 10:
                    print(
                        f"    ⚠️  Large bottleneck: {case['override']['input_dim']} → {case['override']['hidden_dim']} (ratio: {ratio:.1f}x)"
                    )
        except Exception as e:
            print(f"\n  ✗ {case['name']}:")
            print(f"    {type(e).__name__}: {e}")

    # -------------------------------------------------------------------------
    # 7. Common debugging patterns
    # -------------------------------------------------------------------------
    print("\n7. Common Debugging Patterns:")
    print("-" * 80)
    print("""
    Pattern 1: Quick validation
      → Sample 5-10 configs immediately after definition
      → Catches typos, wrong bounds, missing dependencies

    Pattern 2: Distribution check
      → Sample 50-100 configs
      → Check min/max/median of each parameter
      → Ensures distributions look reasonable

    Pattern 3: Conditional verification
      → Use overrides to force each conditional branch
      → Verify behavior matches expectations
      → Test both true and false branches

    Pattern 4: Edge case testing
      → Override to force boundary values
      → Check if extreme configs are valid
      → Catches issues before expensive experiments

    Pattern 5: Nested sanity check
      → Sample nested configs and print structure
      → Verify all levels sample correctly
      → Ensure no missing dependencies
    """)

    # -------------------------------------------------------------------------
    # 8. Debugging workflow
    # -------------------------------------------------------------------------
    print("\n8. Recommended Debugging Workflow:")
    print("-" * 80)
    print("""
    Step 1: Define your Config class
      → Add all parameters with spaces

    Step 2: Quick sanity check (this file!)
      → config = MyConfig.random(seed=42)
      → Does it work? No exceptions?

    Step 3: Sample multiple configs
      → [MyConfig.random(seed=i) for i in range(10)]
      → Print them, look for issues

    Step 4: Check parameter ranges
      → Collect values, check min/max/median
      → Do they look reasonable?

    Step 5: Test conditionals
      → Use overrides to force branches
      → Verify logic is correct

    Step 6: Test edge cases
      → Override to boundary values
      → Does it still work?

    Step 7: Integration test
      → Actually use config in your code
      → Run a quick experiment

    ✓ If all steps pass → Ready for experiments!
    """)

    # -------------------------------------------------------------------------
    # 9. Benefits of sanity checking with SpaX
    # -------------------------------------------------------------------------
    print("\n9. Benefits vs Manual Testing:")
    print("-" * 80)
    print("""
    Without SpaX:
      ✗ Manually think of test values for 50+ parameters
      ✗ Type them all out for each test
      ✗ Forget to test some combinations
      ✗ Miss edge cases
      ✗ No easy way to test conditionals
      ✗ Takes 30+ minutes for complex configs

    With SpaX:
      ✓ Random sampling generates valid values instantly
      ✓ Test 100s of combinations in seconds
      ✓ Overrides make testing specific cases easy
      ✓ Catches issues before expensive experiments
      ✓ Takes < 5 minutes even for complex configs
      ✓ Can automate as part of CI/CD
    """)


if __name__ == "__main__":
    main()
