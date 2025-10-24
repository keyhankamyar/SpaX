"""Example 00: Migrating from Pydantic to SpaX (Minimal Overhead)

This example demonstrates that SpaX is designed with minimal overhead in mind.
You can migrate from Pydantic BaseModel to SpaX Config with almost no code changes,
and gain powerful search space exploration capabilities.

Key Points:
-----------
1. Simply swap BaseModel with Config
2. Mix and match Pydantic Field() with SpaX spaces
3. Automatic inference from type hints (Literal, bool, Field constraints)
4. Explicit SpaX spaces when you need full control

Benefits of Migration:
---------------------
- Random sampling for sanity checking and testing
- Hyperparameter optimization integration (Optuna, etc.)
- Override system for iterative refinement
- All while keeping Pydantic's validation!
"""

from typing import Literal

from pydantic import BaseModel, Field

import spax as sp

# =============================================================================
# BEFORE: Using Pydantic BaseModel
# =============================================================================


class MLPConfigPydantic(BaseModel):
    """Traditional Pydantic configuration."""

    hidden_dim: int = Field(gt=16, lt=4096)
    activation: Literal["relu", "gelu", "silu"]
    use_batch_norm: bool
    learning_rate: float = Field(ge=1e-5, le=1e-1)
    dropout: float = Field(ge=0.0, le=0.5, default=0.1)


# =============================================================================
# AFTER: Using SpaX Config with Automatic Inference
# =============================================================================


class MLPConfigSpaX(sp.Config):
    """SpaX Config with automatic inference - minimal changes needed!"""

    # Pydantic Field constraints are automatically inferred as spaces
    hidden_dim: int = Field(gt=16, lt=4096)  # → IntSpace(gt=16, lt=4096)

    # Literal types are automatically inferred as categorical spaces
    activation: Literal[
        "relu", "gelu", "silu"
    ]  # → Categorical(["relu", "gelu", "silu"])

    # bool is automatically inferred as categorical with [True, False]
    use_batch_norm: bool  # → Categorical([True, False])

    # Works with floats too
    learning_rate: float = Field(ge=1e-5, le=1e-1)  # → FloatSpace(ge=1e-5, le=1e-1)

    # Defaults still work as expected
    dropout: float = Field(ge=0.0, le=0.5, default=0.1)


# =============================================================================
# ALTERNATIVE: Using Explicit SpaX Spaces (When You Need Full Control)
# =============================================================================


class MLPConfigExplicit(sp.Config):
    """SpaX Config with explicit spaces - for when you need more control."""

    # Explicit spaces give you distribution control
    hidden_dim: int = sp.Int(gt=16, lt=4096)

    # Explicit categorical with weights
    activation: str = sp.Categorical(
        [
            sp.Choice("relu", weight=2.0),  # relu is twice as likely
            sp.Choice("gelu", weight=1.0),
            sp.Choice("silu", weight=1.0),
        ]
    )

    # Explicit for clarity
    use_batch_norm: bool = sp.Categorical([True, False])

    # Log distribution for learning rate
    learning_rate: float = sp.Float(ge=1e-5, le=1e-1, distribution="log")

    # Fixed dropout (not searchable)
    dropout: float = 0.1


# =============================================================================
# MIXED APPROACH: Best of Both Worlds
# =============================================================================


class MLPConfigMixed(sp.Config):
    """Mix Pydantic and SpaX - use each where it makes sense!"""

    # Simple constraints? Use Field - it's concise
    hidden_dim: int = Field(gt=16, lt=4096)

    # Need distribution control? Use explicit space
    learning_rate: float = sp.Float(ge=1e-5, le=1e-1, distribution="log")

    # Simple choice? Use Literal - it's type-safe
    activation: Literal["relu", "gelu", "silu"]

    # Need weights? Use explicit Categorical
    optimizer: str = sp.Categorical(
        [
            sp.Choice("adam", weight=3.0),
            sp.Choice("sgd", weight=1.0),
        ]
    )

    # Simple bool? Use type hint
    use_batch_norm: bool

    # Fixed value? Just assign it
    model_name: str = "my_mlp_model"


# =============================================================================
# Demonstration
# =============================================================================


def main():
    print("=" * 80)
    print("SpaX: Minimal Overhead Migration from Pydantic")
    print("=" * 80)

    # -----------------------------------------------------------------------------
    # Traditional Pydantic Usage
    # -----------------------------------------------------------------------------
    print("\n1. Traditional Pydantic BaseModel:")
    print("-" * 80)

    # With Pydantic, you manually create instances
    pydantic_config = MLPConfigPydantic(
        hidden_dim=256, activation="relu", use_batch_norm=True, learning_rate=0.001
    )
    print(f"Created config: {pydantic_config}")
    print(f"Type: {type(pydantic_config).__name__}")

    # -----------------------------------------------------------------------------
    # SpaX with Automatic Inference
    # -----------------------------------------------------------------------------
    print("\n2. SpaX Config with Automatic Inference:")
    print("-" * 80)

    # Create manually (works just like Pydantic)
    spax_config = MLPConfigSpaX(
        hidden_dim=256, activation="relu", use_batch_norm=True, learning_rate=0.001
    )
    print(f"Created config: {spax_config}")

    # NEW: Random sampling for sanity checking!
    print("\n   🎲 Random sampling (new capability):")
    random_config = MLPConfigSpaX.random(seed=42)
    print(f"   Sampled config: {random_config}")

    # NEW: Get parameter names
    print("\n   📋 Search space parameters:")
    for param in MLPConfigSpaX.get_parameter_names():
        print(f"   - {param}")

    # -----------------------------------------------------------------------------
    # SpaX with Explicit Spaces
    # -----------------------------------------------------------------------------
    print("\n3. SpaX Config with Explicit Spaces:")
    print("-" * 80)

    explicit_config = MLPConfigExplicit.random(seed=42)
    print(f"Sampled config: {explicit_config}")
    print("Note: learning_rate uses log distribution for better sampling")

    # -----------------------------------------------------------------------------
    # Mixed Approach
    # -----------------------------------------------------------------------------
    print("\n4. Mixed Approach (Recommended):")
    print("-" * 80)

    mixed_config = MLPConfigMixed.random(seed=42)
    print(f"Sampled config: {mixed_config}")

    # NEW: Override system for iterative refinement
    print("\n   🎯 With overrides to narrow search space:")
    override = {
        "hidden_dim": {"gt": 128, "lt": 512},  # Narrow the range
        "optimizer": ["adam"],  # Fix to adam only
    }
    refined_config = MLPConfigMixed.random(seed=42, override=override)
    print(f"   Refined config: {refined_config}")
    print("   Notice hidden_dim is now in [128, 512] and optimizer is always 'adam'")

    # NEW: Get override template
    print("\n   📝 Override template:")
    template = MLPConfigMixed.get_override_template()
    for key, value in template.items():
        print(f"   {key}: {value}")

    # -----------------------------------------------------------------------------
    # Key Takeaways
    # -----------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Key Takeaways:")
    print("=" * 80)
    print("""
    1. ✅ Minimal code changes: BaseModel → Config
    2. ✅ Keep using Pydantic Field() - it just works
    3. ✅ Type hints (Literal, bool) are automatically inferred
    4. ✅ Use explicit spaces (sp.Int, sp.Float) when you need control
    5. ✅ Mix and match approaches as needed
    6. ✅ Gain random sampling, HPO integration, and override system
    7. ✅ All Pydantic validation still works!

    Migration is as simple as:
        from pydantic import BaseModel  →  import spax as sp
        class MyConfig(BaseModel):      →  class MyConfig(sp.Config):
    """)


if __name__ == "__main__":
    main()
