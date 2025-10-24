"""Example 06: Chained Conditions

This example demonstrates how to create conditions that depend on deeply
nested fields by chaining FieldCondition objects. Each FieldCondition
navigates one level deeper into the nested structure.

Topics Covered:
--------------
- Chaining FieldCondition for nested field access
- MultiFieldLambdaCondition for complex logic on multiple fields
- Conditions across multiple nesting levels
- Conditions combining fields from different branches
- Best practices for complex conditional logic
"""

import spax as sp

# =============================================================================
# Basic Nested Field Conditions
# =============================================================================


class OptimizerConfig(sp.Config):
    """Optimizer configuration."""

    name: str = sp.Categorical(["adam", "sgd", "adamw"])
    learning_rate: float = sp.Float(ge=1e-5, le=1e-1, distribution="log")


class ModelConfig(sp.Config):
    """Model configuration with optimizer."""

    hidden_dim: int = sp.Int(ge=64, le=512)
    optimizer: OptimizerConfig


class TrainingConfig(sp.Config):
    """Training configuration that conditions on nested optimizer."""

    model: ModelConfig

    # Batch size depends on optimizer choice (nested field!)
    # Chain FieldConditions to access: config.model.optimizer.name
    batch_size: int = sp.Conditional(
        sp.FieldCondition(
            "model",  # First level: access model
            sp.FieldCondition(  # Second level: access optimizer
                "optimizer",
                sp.FieldCondition(  # Third level: access name
                    "name",
                    sp.EqualsTo("adam"),  # Finally: check if equals "adam"
                ),
            ),
        ),
        true=sp.Int(ge=32, le=256),  # Larger batches for Adam
        false=sp.Int(ge=8, le=64),  # Smaller batches for SGD/AdamW
    )

    num_epochs: int = sp.Int(ge=10, le=1000, default=100)


# =============================================================================
# Multi-Field Lambda Conditions
# =============================================================================


class AdvancedModelConfig(sp.Config):
    """Model with regularization options."""

    hidden_dim: int = sp.Int(ge=64, le=512)
    num_layers: int = sp.Int(ge=1, le=10)

    use_dropout: bool = sp.Categorical([True, False])
    use_l2: bool = sp.Categorical([True, False])


class AdvancedTrainingConfig(sp.Config):
    """Training that conditions on multiple model fields."""

    model: AdvancedModelConfig

    # Learning rate depends on both model size AND regularization
    # Use MultiFieldLambdaCondition when checking multiple fields
    learning_rate: float = sp.Conditional(
        sp.MultiFieldLambdaCondition(
            # Each field path is a direct field name, not nested with dots
            ["model"],  # We're passing the entire model object
            lambda model: (
                # Access nested fields through the model object
                model.hidden_dim > 256
                and model.num_layers > 5
                and (model.use_dropout or model.use_l2)
            ),
        ),
        true=sp.Float(
            ge=1e-5, le=1e-3, distribution="log"
        ),  # Lower LR for large models
        false=sp.Float(
            ge=1e-4, le=1e-2, distribution="log"
        ),  # Higher LR for small models
    )

    batch_size: int = sp.Int(ge=8, le=256)


# =============================================================================
# Deeply Nested Conditions
# =============================================================================


class LayerConfig(sp.Config):
    """Single layer configuration."""

    units: int = sp.Int(ge=32, le=512)
    activation: str = sp.Categorical(["relu", "gelu", "silu"])


class BlockConfig(sp.Config):
    """Block of layers."""

    layer1: LayerConfig
    layer2: LayerConfig
    use_residual: bool = sp.Categorical([True, False])


class NetworkConfig(sp.Config):
    """Network with blocks."""

    block1: BlockConfig
    block2: BlockConfig


class ExperimentConfig(sp.Config):
    """Experiment that conditions on deeply nested fields."""

    network: NetworkConfig

    # Training batch size depends on first layer size (3 levels deep!)
    # Chain: config.network.block1.layer1.units
    batch_size: int = sp.Conditional(
        sp.FieldCondition(
            "network",
            sp.FieldCondition(
                "block1",
                sp.FieldCondition(
                    "layer1", sp.FieldCondition("units", sp.LargerThan(256))
                ),
            ),
        ),
        true=sp.Int(ge=8, le=64),  # Smaller batches for large layers
        false=sp.Int(ge=32, le=256),  # Larger batches for small layers
    )

    # Learning rate depends on whether residual connections are used
    # Access multiple nested fields using MultiFieldLambdaCondition
    learning_rate: float = sp.Conditional(
        sp.MultiFieldLambdaCondition(
            ["network"],
            lambda network: (
                network.block1.use_residual and network.block2.use_residual
            ),
        ),
        true=sp.Float(ge=1e-4, le=1e-2, distribution="log"),  # Higher LR with residuals
        false=sp.Float(
            ge=1e-5, le=1e-3, distribution="log"
        ),  # Lower LR without residuals
    )


# =============================================================================
# Cross-Branch Conditions
# =============================================================================


class EncoderConfig(sp.Config):
    """Encoder configuration."""

    hidden_dim: int = sp.Int(ge=64, le=512)
    num_layers: int = sp.Int(ge=1, le=8)


class DecoderConfig(sp.Config):
    """Decoder configuration."""

    hidden_dim: int = sp.Int(ge=64, le=512)
    num_layers: int = sp.Int(ge=1, le=8)


class Seq2SeqConfig(sp.Config):
    """Seq2Seq model with encoder and decoder."""

    encoder: EncoderConfig
    decoder: DecoderConfig

    # Attention is only useful if encoder and decoder have similar capacity
    # Need to access fields from both encoder AND decoder branches
    use_attention: bool = sp.Conditional(
        sp.MultiFieldLambdaCondition(
            ["encoder", "decoder"],  # Access both branches
            lambda encoder, decoder: (
                abs(encoder.hidden_dim - decoder.hidden_dim) <= 128
                and abs(encoder.num_layers - decoder.num_layers) <= 2
            ),
        ),
        true=sp.Categorical([True, False]),  # Choice when balanced
        false=False,  # No attention when unbalanced
    )

    # Dropout rate depends on total model size
    dropout_rate: float = sp.Conditional(
        sp.MultiFieldLambdaCondition(
            ["encoder", "decoder"],
            lambda encoder, decoder: (
                (encoder.hidden_dim + decoder.hidden_dim)
                * (encoder.num_layers + decoder.num_layers)
                > 10000
            ),
        ),
        true=sp.Float(ge=0.2, le=0.5),  # Higher dropout for large models
        false=sp.Float(ge=0.0, le=0.2),  # Lower dropout for small models
    )


# =============================================================================
# Complex Chaining Example
# =============================================================================


class DatabaseConfig(sp.Config):
    """Database configuration."""

    type: str = sp.Categorical(["postgres", "mysql", "sqlite"])
    connection_pool_size: int = sp.Int(ge=5, le=100)


class CacheConfig(sp.Config):
    """Cache configuration."""

    enabled: bool = sp.Categorical([True, False])
    ttl_seconds: int = sp.Int(ge=60, le=3600)


class BackendConfig(sp.Config):
    """Backend configuration."""

    database: DatabaseConfig
    cache: CacheConfig


class ApplicationConfig(sp.Config):
    """Application with complex nested conditions."""

    backend: BackendConfig

    # API timeout depends on whether cache is enabled (deep nesting)
    api_timeout_seconds: int = sp.Conditional(
        sp.FieldCondition(
            "backend",
            sp.FieldCondition("cache", sp.FieldCondition("enabled", sp.EqualsTo(True))),
        ),
        true=sp.Int(ge=1, le=10),  # Short timeout with cache
        false=sp.Int(ge=10, le=60),  # Long timeout without cache
    )

    # Max connections depends on database type AND pool size
    max_concurrent_requests: int = sp.Conditional(
        sp.MultiFieldLambdaCondition(
            ["backend"],
            lambda backend: (
                backend.database.type == "postgres"
                and backend.database.connection_pool_size > 50
            ),
        ),
        true=sp.Int(ge=100, le=1000),  # High concurrency for Postgres with large pool
        false=sp.Int(ge=10, le=100),  # Lower concurrency otherwise
    )


# =============================================================================
# Demonstrations
# =============================================================================


def main():
    print("=" * 80)
    print("Example 06: Chained Conditions")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Basic chained field conditions
    # -------------------------------------------------------------------------
    print("\n1. Chaining FieldConditions for Nested Access:")
    print("-" * 80)

    print("Batch size depends on config.model.optimizer.name:")
    print("\nCondition chain:")
    print("  FieldCondition('model',")
    print("    FieldCondition('optimizer',")
    print("      FieldCondition('name',")
    print("        EqualsTo('adam'))))")

    print("\nSampling configs:")
    for i in range(6):
        config = TrainingConfig.random(seed=100 + i)
        optimizer_name = config.model.optimizer.name
        batch_size = config.batch_size

        print(
            f"  Sample {i + 1}: optimizer={optimizer_name:6s}, batch_size={batch_size:3d}",
            end="",
        )
        if optimizer_name == "adam":
            print(" (large range)")
        else:
            print(" (small range)")

    # -------------------------------------------------------------------------
    # 2. MultiFieldLambdaCondition
    # -------------------------------------------------------------------------
    print("\n2. MultiFieldLambdaCondition for Complex Logic:")
    print("-" * 80)

    print("Learning rate depends on multiple model properties:")
    print("\nCondition:")
    print("  lambda model: (")
    print("    model.hidden_dim > 256 and")
    print("    model.num_layers > 5 and")
    print("    (model.use_dropout or model.use_l2)")
    print("  )")

    print("\nSampling configs:")
    for i in range(6):
        config = AdvancedTrainingConfig.random(seed=200 + i)

        hidden = config.model.hidden_dim
        layers = config.model.num_layers
        dropout = config.model.use_dropout
        l2 = config.model.use_l2
        lr = config.learning_rate

        is_large = hidden > 256 and layers > 5 and (dropout or l2)

        print(f"\n  Sample {i + 1}:")
        print(f"    Model: {hidden}D × {layers}L, dropout={dropout}, l2={l2}")
        print(f"    Large w/ reg: {is_large}")
        print(f"    LR: {lr:.6f} ", end="")
        if is_large:
            print("(low range)")
        else:
            print("(high range)")

    # -------------------------------------------------------------------------
    # 3. Deep chaining (4 levels)
    # -------------------------------------------------------------------------
    print("\n3. Deep Chaining (4 Levels):")
    print("-" * 80)

    print("Batch size depends on config.network.block1.layer1.units:")
    print("\nCondition chain (4 levels deep):")
    print("  FieldCondition('network',")
    print("    FieldCondition('block1',")
    print("      FieldCondition('layer1',")
    print("        FieldCondition('units',")
    print("          LargerThan(256)))))")

    print("\nSampling configs:")
    for i in range(4):
        config = ExperimentConfig.random(seed=300 + i)

        layer1_units = config.network.block1.layer1.units
        batch_size = config.batch_size

        print(
            f"  Sample {i + 1}: units={layer1_units:3d}, batch_size={batch_size:3d}",
            end="",
        )
        if layer1_units > 256:
            print(" (small range for large layer)")
        else:
            print(" (large range for small layer)")

    # -------------------------------------------------------------------------
    # 4. Cross-branch conditions
    # -------------------------------------------------------------------------
    print("\n4. Cross-Branch Conditions:")
    print("-" * 80)

    print("Attention depends on balance between encoder and decoder:")
    print("\nCondition:")
    print("  lambda encoder, decoder: (")
    print("    abs(encoder.hidden_dim - decoder.hidden_dim) <= 128 and")
    print("    abs(encoder.num_layers - decoder.num_layers) <= 2")
    print("  )")

    print("\nSampling configs:")
    for i in range(6):
        config = Seq2SeqConfig.random(seed=400 + i)

        enc_dim = config.encoder.hidden_dim
        dec_dim = config.decoder.hidden_dim
        enc_layers = config.encoder.num_layers
        dec_layers = config.decoder.num_layers
        use_attention = config.use_attention

        dim_diff = abs(enc_dim - dec_dim)
        layer_diff = abs(enc_layers - dec_layers)
        balanced = dim_diff <= 128 and layer_diff <= 2

        print(f"\n  Sample {i + 1}:")
        print(f"    Encoder: {enc_dim:3d}D × {enc_layers}L")
        print(f"    Decoder: {dec_dim:3d}D × {dec_layers}L")
        print(f"    Balanced: {balanced}, Attention: {use_attention}")

    # -------------------------------------------------------------------------
    # 5. Complex chaining with multiple conditions
    # -------------------------------------------------------------------------
    print("\n5. Complex Application with Multiple Chained Conditions:")
    print("-" * 80)

    print("Two different chained conditions in one config:")
    for i in range(4):
        config = ApplicationConfig.random(seed=500 + i)

        cache_enabled = config.backend.cache.enabled
        db_type = config.backend.database.type
        pool_size = config.backend.database.connection_pool_size
        api_timeout = config.api_timeout_seconds
        max_requests = config.max_concurrent_requests

        print(f"\n  Sample {i + 1}:")
        print(f"    Cache: {cache_enabled}")
        print(f"    Database: {db_type}, pool={pool_size}")
        print(f"    API timeout: {api_timeout}s ", end="")
        if cache_enabled:
            print("(short with cache)")
        else:
            print("(long without cache)")
        print(f"    Max requests: {max_requests} ", end="")
        if db_type == "postgres" and pool_size > 50:
            print("(high for Postgres w/ large pool)")
        else:
            print("(standard)")

    # -------------------------------------------------------------------------
    # 6. How chaining works internally
    # -------------------------------------------------------------------------
    print("\n6. Understanding the Chain:")
    print("-" * 80)

    config = TrainingConfig.random(seed=600)

    print("When evaluating the chained condition:")
    print("\n  1. Start with config object")
    print("  2. FieldCondition('model') → gets config.model")
    print(f"       Result: {type(config.model).__name__} instance")
    print("  3. FieldCondition('optimizer') → gets model.optimizer")
    print(f"       Result: {type(config.model.optimizer).__name__} instance")
    print("  4. FieldCondition('name') → gets optimizer.name")
    print(f"       Result: '{config.model.optimizer.name}'")
    print("  5. EqualsTo('adam') → checks if name == 'adam'")
    print(f"       Result: {config.model.optimizer.name == 'adam'}")

    # -------------------------------------------------------------------------
    # 7. Best practices
    # -------------------------------------------------------------------------
    print("\n7. Best Practices:")
    print("-" * 80)
    print("""
    ✓ CORRECT: Chain FieldConditions for nested access
        sp.FieldCondition('model',
          sp.FieldCondition('optimizer',
            sp.FieldCondition('name', sp.EqualsTo('adam'))))

    ✗ INCORRECT: Don't use dot notation in field names
        sp.FieldCondition('model.optimizer.name', sp.EqualsTo('adam'))

    ✓ Use MultiFieldLambdaCondition for complex logic on multiple fields
    ✓ Pass full objects to lambda, then access nested fields inside
    ✓ Keep lambda functions simple and readable
    ✓ Document complex conditions with comments
    ✓ Test conditions thoroughly with various samples
    """)

    # -------------------------------------------------------------------------
    # 8. Common patterns
    # -------------------------------------------------------------------------
    print("\n8. Common Patterns:")
    print("-" * 80)
    print("""
    Pattern 1: Single nested field check
      → FieldCondition chain ending in ObjectCondition
      → Example: Check optimizer type 3 levels deep

    Pattern 2: Multiple field logic
      → MultiFieldLambdaCondition with lambda
      → Example: Check if model is large AND has regularization

    Pattern 3: Cross-branch coordination
      → MultiFieldLambdaCondition accessing multiple branches
      → Example: Encoder and decoder balance

    Pattern 4: Deep nesting (3+ levels)
      → Chain multiple FieldConditions
      → Example: Access layer config in network block
    """)


if __name__ == "__main__":
    main()
