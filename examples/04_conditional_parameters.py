"""Example 04: Conditional Parameters

This example demonstrates how to make parameters conditional on the values
of other parameters. This is essential for representing dependencies in
configurations, such as dropout rate only mattering when dropout is enabled.

Topics Covered:
--------------
- Basic conditional parameters (field conditions)
- Different condition types (EqualsTo, In, comparison operators)
- Logical combinations (And, Or, Not)
- Conditional with Config choices
- Multiple conditions on same field
- Conditional branches with different types
"""

import spax as sp

# =============================================================================
# Basic Conditional Parameters
# =============================================================================


class DropoutConfig(sp.Config):
    """Model with conditional dropout rate."""

    hidden_dim: int = sp.Int(ge=64, le=512)

    # First parameter: whether to use dropout
    use_dropout: bool = sp.Categorical([True, False])

    # Second parameter: dropout rate (only matters if use_dropout=True)
    dropout_rate: float = sp.Conditional(
        condition=sp.FieldCondition("use_dropout", sp.EqualsTo(True)),
        true=sp.Float(gt=0.0, lt=0.5),  # Sample rate when dropout enabled
        false=0.0,  # Fixed to 0.0 when dropout disabled
    )


class RegularizationConfig(sp.Config):
    """Configuration with multiple conditional parameters."""

    learning_rate: float = sp.Float(ge=1e-5, le=1e-1, distribution="log")

    # L2 regularization
    use_l2: bool = sp.Categorical([True, False])
    l2_strength: float = sp.Conditional(
        sp.FieldCondition("use_l2", sp.EqualsTo(True)),
        true=sp.Float(ge=1e-6, le=1e-2, distribution="log"),
        false=0.0,
    )

    # Dropout
    use_dropout: bool = sp.Categorical([True, False])
    dropout_rate: float = sp.Conditional(
        sp.FieldCondition("use_dropout", sp.EqualsTo(True)),
        true=sp.Float(gt=0.0, lt=0.5),
        false=0.0,
    )


# =============================================================================
# Different Condition Types
# =============================================================================


class OptimizerConfig(sp.Config):
    """Configuration with optimizer-specific parameters."""

    optimizer: str = sp.Categorical(["adam", "sgd", "rmsprop"])
    learning_rate: float = sp.Float(ge=1e-5, le=1e-1, distribution="log")

    # Momentum only for SGD
    momentum: float = sp.Conditional(
        sp.FieldCondition("optimizer", sp.EqualsTo("sgd")),
        true=sp.Float(ge=0.0, lt=1.0, default=0.9),
        false=0.0,
    )

    # Beta parameters for Adam
    beta1: float = sp.Conditional(
        sp.FieldCondition("optimizer", sp.EqualsTo("adam")),
        true=sp.Float(ge=0.8, lt=1.0, default=0.9),
        false=0.0,
    )

    beta2: float = sp.Conditional(
        sp.FieldCondition("optimizer", sp.EqualsTo("adam")),
        true=sp.Float(ge=0.9, lt=1.0, default=0.999),
        false=0.0,
    )


class ActivationConfig(sp.Config):
    """Configuration with activation-specific parameters."""

    activation: str = sp.Categorical(["relu", "leaky_relu", "elu", "gelu"])

    # Alpha only for leaky_relu and elu
    alpha: float = sp.Conditional(
        sp.FieldCondition("activation", sp.In(["leaky_relu", "elu"])),
        true=sp.Float(ge=0.01, le=0.3, default=0.1),
        false=0.0,
    )


# =============================================================================
# Logical Combinations
# =============================================================================


class AdvancedRegularizationConfig(sp.Config):
    """Configuration with complex logical conditions."""

    use_l2: bool = sp.Categorical([True, False])
    use_dropout: bool = sp.Categorical([True, False])
    use_batch_norm: bool = sp.Categorical([True, False])

    # Only apply extra regularization if using BOTH L2 and dropout
    strong_regularization: bool = sp.Conditional(
        sp.And(
            [
                sp.FieldCondition("use_l2", sp.EqualsTo(True)),
                sp.FieldCondition("use_dropout", sp.EqualsTo(True)),
            ]
        ),
        true=sp.Categorical([True, False]),
        false=False,
    )

    # Warning flag: if using neither L2 nor dropout (high overfitting risk)
    no_regularization_warning: bool = sp.Conditional(
        sp.And(
            [
                sp.FieldCondition("use_l2", sp.EqualsTo(False)),
                sp.FieldCondition("use_dropout", sp.EqualsTo(False)),
            ]
        ),
        true=True,
        false=False,
    )

    # Use alternative regularization if batch norm is disabled
    alternative_regularization: float = sp.Conditional(
        sp.Not(sp.FieldCondition("use_batch_norm", sp.EqualsTo(True))),
        true=sp.Float(ge=0.0, le=0.5),
        false=0.0,
    )


# =============================================================================
# Conditional with Different Types
# =============================================================================


class DataAugmentationConfig(sp.Config):
    """Configuration where conditional branches have different types."""

    augmentation_level: str = sp.Categorical(["none", "light", "heavy"])

    # Augmentation parameters depend on level
    augmentation_params: dict = sp.Conditional(
        sp.FieldCondition("augmentation_level", sp.EqualsTo("none")),
        true={},  # No parameters needed
        false=sp.Conditional(
            sp.FieldCondition("augmentation_level", sp.EqualsTo("light")),
            true={"rotation": 15, "flip": True},
            false={"rotation": 30, "flip": True, "color_jitter": 0.2},
        ),
    )


class LearningRateScheduleConfig(sp.Config):
    """Configuration with complex conditional structure."""

    schedule: str = sp.Categorical(["constant", "step", "exponential", "cosine"])
    base_lr: float = sp.Float(ge=1e-5, le=1e-1, distribution="log")

    # Schedule-specific parameters
    step_size: int = sp.Conditional(
        sp.FieldCondition("schedule", sp.EqualsTo("step")),
        true=sp.Int(ge=5, le=50),
        false=0,
    )

    gamma: float = sp.Conditional(
        sp.FieldCondition("schedule", sp.In(["step", "exponential"])),
        true=sp.Float(ge=0.1, le=0.99),
        false=0.0,
    )


# =============================================================================
# Nested Conditional (One condition depends on another)
# =============================================================================


class NestedConditionalConfig(sp.Config):
    """Configuration with nested conditional logic."""

    use_advanced_features: bool = sp.Categorical([True, False])

    # Only available if advanced features enabled
    feature_type: str = sp.Conditional(
        sp.FieldCondition("use_advanced_features", sp.EqualsTo(True)),
        true=sp.Categorical(["type_a", "type_b", "type_c"]),
        false="none",
    )

    # Only available if feature_type is type_a or type_b
    # Note: This depends on feature_type, which itself is conditional!
    feature_param: float = sp.Conditional(
        sp.And(
            [
                sp.FieldCondition("use_advanced_features", sp.EqualsTo(True)),
                sp.FieldCondition("feature_type", sp.In(["type_a", "type_b"])),
            ]
        ),
        true=sp.Float(ge=0.0, le=1.0),
        false=0.0,
    )


# =============================================================================
# Demonstrations
# =============================================================================


def main():
    print("=" * 80)
    print("Example 04: Conditional Parameters")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Basic conditional parameters
    # -------------------------------------------------------------------------
    print("\n1. Basic Conditional Parameters:")
    print("-" * 80)

    print("Sampling configs with use_dropout=True and use_dropout=False:")
    for i in range(4):
        config = DropoutConfig.random(seed=100 + i)
        print(
            f"  Sample {i + 1}: use_dropout={config.use_dropout}, "
            f"dropout_rate={config.dropout_rate:.3f}"
        )

    print("\nNotice: When use_dropout=False, dropout_rate is always 0.0")

    # -------------------------------------------------------------------------
    # 2. Multiple conditional parameters
    # -------------------------------------------------------------------------
    print("\n2. Multiple Conditional Parameters:")
    print("-" * 80)

    print("Sampling configs with multiple regularization options:")
    for i in range(4):
        config = RegularizationConfig.random(seed=200 + i)
        print(f"  Sample {i + 1}:")
        print(f"    L2: {config.use_l2}, strength={config.l2_strength:.6f}")
        print(f"    Dropout: {config.use_dropout}, rate={config.dropout_rate:.3f}")

    # -------------------------------------------------------------------------
    # 3. Optimizer-specific parameters
    # -------------------------------------------------------------------------
    print("\n3. Optimizer-Specific Parameters:")
    print("-" * 80)

    print("Sampling with different optimizers:")
    for seed in [300, 301, 302, 303, 304]:
        config = OptimizerConfig.random(seed=seed)
        print(f"\n  Optimizer: {config.optimizer}")
        print(f"    momentum: {config.momentum:.3f}")
        print(f"    beta1: {config.beta1:.3f}")
        print(f"    beta2: {config.beta2:.3f}")
        if config.optimizer == "sgd":
            print("    → Only momentum is non-zero (SGD uses it)")
        elif config.optimizer == "adam":
            print("    → Only betas are non-zero (Adam uses them)")

    # -------------------------------------------------------------------------
    # 4. Condition with In operator
    # -------------------------------------------------------------------------
    print("\n4. Condition with In Operator:")
    print("-" * 80)

    print("Sampling activation configs:")
    for seed in [400, 401, 402, 403]:
        config = ActivationConfig.random(seed=seed)
        print(f"  {config.activation:12s}: alpha={config.alpha:.3f}", end="")
        if config.activation in ["leaky_relu", "elu"]:
            print(" (uses alpha)")
        else:
            print(" (alpha not used)")

    # -------------------------------------------------------------------------
    # 5. Logical combinations (And, Or, Not)
    # -------------------------------------------------------------------------
    print("\n5. Logical Combinations:")
    print("-" * 80)

    print("Sampling with complex logical conditions:")
    for i in range(6):
        config = AdvancedRegularizationConfig.random(seed=500 + i)
        print(f"\n  Sample {i + 1}:")
        print(
            f"    L2: {config.use_l2}, Dropout: {config.use_dropout}, "
            f"BatchNorm: {config.use_batch_norm}"
        )
        print(
            f"    Strong reg: {config.strong_regularization} "
            f"(only True if L2 AND Dropout)"
        )
        print(
            f"    No reg warning: {config.no_regularization_warning} "
            f"(True if neither L2 nor Dropout)"
        )
        print(
            f"    Alt reg: {config.alternative_regularization:.3f} "
            f"(non-zero if NOT BatchNorm)"
        )

    # -------------------------------------------------------------------------
    # 6. Conditional branches with different types
    # -------------------------------------------------------------------------
    print("\n6. Conditional Branches with Different Types:")
    print("-" * 80)

    print("Learning rate schedules with different parameters:")
    for schedule in ["constant", "step", "exponential"]:
        # Sample multiple times to get each schedule
        config = None
        for seed in range(600, 700):
            config = LearningRateScheduleConfig.random(seed=seed)
            if config.schedule == schedule:
                break

        if config:
            print(f"\n  {config.schedule}:")
            print(f"    base_lr: {config.base_lr:.6f}")
            print(f"    step_size: {config.step_size}")
            print(f"    gamma: {config.gamma:.3f}")

    # -------------------------------------------------------------------------
    # 7. Nested conditionals
    # -------------------------------------------------------------------------
    print("\n7. Nested Conditional Logic:")
    print("-" * 80)

    print("Sampling with nested conditions:")
    for i in range(6):
        config = NestedConditionalConfig.random(seed=700 + i)
        print(f"\n  Sample {i + 1}:")
        print(f"    use_advanced_features: {config.use_advanced_features}")
        print(f"    feature_type: {config.feature_type}")
        print(f"    feature_param: {config.feature_param:.3f}")

        if not config.use_advanced_features:
            print("    → feature_type='none', feature_param=0.0 (advanced disabled)")
        elif config.feature_type in ["type_a", "type_b"]:
            print("    → feature_param is sampled (feature_type uses it)")
        else:
            print("    → feature_param=0.0 (feature_type doesn't use it)")

    # -------------------------------------------------------------------------
    # 8. Validation with conditionals
    # -------------------------------------------------------------------------
    print("\n8. Validation with Conditionals:")
    print("-" * 80)

    # Valid: dropout enabled with valid rate
    try:
        _ = DropoutConfig(hidden_dim=256, use_dropout=True, dropout_rate=0.3)
        print("✓ Valid config: use_dropout=True, rate=0.3")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Valid: dropout disabled, rate is 0.0
    try:
        _ = DropoutConfig(hidden_dim=256, use_dropout=False, dropout_rate=0.0)
        print("✓ Valid config: use_dropout=False, rate=0.0")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Invalid: dropout disabled but rate is non-zero
    try:
        invalid = DropoutConfig(
            hidden_dim=256,
            use_dropout=False,
            dropout_rate=0.3,  # Should be 0.0 when disabled
        )
        print(f"✗ Config created: {invalid}")
    except Exception as e:
        print("✓ Validation error (expected): Dropout disabled but rate non-zero")
        print(f"   {type(e).__name__}")

    # -------------------------------------------------------------------------
    # 9. Benefits of conditional parameters
    # -------------------------------------------------------------------------
    print("\n9. Benefits of Conditional Parameters:")
    print("-" * 80)
    print("""
    ✓ Accuracy: Model real dependencies in your config
    ✓ Clarity: Make parameter relationships explicit
    ✓ Validation: Ensure dependent parameters are always valid
    ✓ Sampling: Automatically respects conditions during random sampling
    ✓ Efficiency: Don't sample parameters that won't be used
    ✓ Flexibility: Support complex logical relationships (And, Or, Not)
    ✓ Documentation: Conditions serve as self-documenting code
    """)


if __name__ == "__main__":
    main()
