from typing import Literal

import spax as sp


def test_random_sampling_simple():
    """Test random sampling for simple config."""

    class SimpleConfig(sp.Config):
        x: int = sp.Int(ge=0, le=100)
        y: float = sp.Float(ge=0.0, le=1.0)
        choice: str = sp.Categorical(["a", "b", "c"])

    # Sample with seed for reproducibility
    config1 = SimpleConfig.random(seed=42)
    config2 = SimpleConfig.random(seed=42)

    # Same seed should produce same results
    assert config1.x == config2.x
    assert config1.y == config2.y
    assert config1.choice == config2.choice

    # Values should be within bounds
    assert 0 <= config1.x <= 100
    assert 0.0 <= config1.y <= 1.0
    assert config1.choice in ["a", "b", "c"]


def test_random_sampling_with_override():
    """Test random sampling with overrides."""

    class SimpleConfig(sp.Config):
        x: int = sp.Int(ge=0, le=100)
        y: float = sp.Float(ge=0.0, le=1.0)

    # Sample with override
    config = SimpleConfig.random(seed=42, override={"x": 50})

    # x should be fixed to 50
    assert config.x == 50

    # y should still be random but within bounds
    assert 0.0 <= config.y <= 1.0


def test_random_sampling_nested():
    """Test random sampling with nested configs."""

    class InnerConfig(sp.Config):
        value: int = sp.Int(ge=0, le=10)

    class OuterConfig(sp.Config):
        x: int = sp.Int(ge=0, le=100)
        inner: InnerConfig

    config = OuterConfig.random(seed=42)

    assert 0 <= config.x <= 100
    assert isinstance(config.inner, InnerConfig)
    assert 0 <= config.inner.value <= 10


def test_random_sampling_categorical_configs():
    """Test random sampling with categorical config choices."""

    class ConfigA(sp.Config):
        a: int = sp.Int(ge=0, le=10)

    class ConfigB(sp.Config):
        b: float = sp.Float(ge=0.0, le=1.0)

    class MainConfig(sp.Config):
        choice: ConfigA | ConfigB

    config = MainConfig.random(seed=42)

    assert isinstance(config.choice, (ConfigA, ConfigB))

    if isinstance(config.choice, ConfigA):
        assert 0 <= config.choice.a <= 10
    else:
        assert 0.0 <= config.choice.b <= 1.0


# ============================================================================
# Basic Building Blocks
# ============================================================================


class ActivationConfig(sp.Config):
    """Simple activation configuration."""

    type: Literal["relu", "gelu", "silu", "tanh"]
    negative_slope: float = sp.Conditional(
        condition=sp.FieldCondition("type", sp.EqualsTo("relu")),
        true=sp.Float(ge=0.0, le=1.0, default=0.01),
        false=0.0,
    )


class NormConfig(sp.Config):
    """Normalization configuration."""

    type: Literal["batch", "layer", "instance", "none"]
    eps: float = sp.Conditional(
        condition=sp.FieldCondition("type", sp.NotEqualsTo("none")),
        true=sp.Float(ge=1e-8, le=1e-3, default=1e-5),
        false=0.0,
    )
    momentum: float = sp.Conditional(
        condition=sp.FieldCondition("type", sp.EqualsTo("batch")),
        true=sp.Float(ge=0.0, le=1.0, default=0.1),
        false=0.0,
    )


# ============================================================================
# Layer Configurations
# ============================================================================


class BaseLayerConfig(sp.Config):
    """Base layer with common parameters."""

    hidden_dim: int = sp.Int(ge=8, le=2048, default=64)
    dropout: float = sp.Float(ge=0.0, le=0.9, default=0.1)
    use_bias: bool = sp.Categorical([True, False], default=True)


class MLPLayerConfig(BaseLayerConfig):
    """MLP layer extending base layer."""

    num_layers: int = sp.Int(ge=1, le=10, default=2)
    activation: ActivationConfig
    norm: NormConfig


class ConvLayerConfig(BaseLayerConfig):
    """Convolutional layer extending base layer."""

    kernel_size: int = sp.Int(ge=1, le=11, default=3)
    stride: int = sp.Int(ge=1, le=4, default=1)
    padding: int = sp.Conditional(
        condition=sp.FieldCondition("kernel_size", sp.LargerThan(1, or_equals=True)),
        true=sp.Int(ge=0, le=5, default=1),
        false=0,
    )
    groups: int = sp.Conditional(
        condition=sp.MultiFieldLambdaCondition(
            field_names={"hidden_dim", "kernel_size"},
            func=lambda hidden_dim, kernel_size: hidden_dim >= 32 and kernel_size >= 3,
        ),
        true=sp.Categorical([1, 2, 4, 8], default=1),
        false=1,
    )
    activation: ActivationConfig
    norm: NormConfig


class AttentionLayerConfig(BaseLayerConfig):
    """Attention layer extending base layer."""

    num_heads: int = sp.Conditional(
        condition=sp.FieldCondition("hidden_dim", sp.LargerThan(63, or_equals=True)),
        true=sp.Categorical([4, 8, 16], default=8),
        false=sp.Categorical([1, 2, 4], default=2),
    )
    head_dim: int = sp.Conditional(
        condition=sp.MultiFieldLambdaCondition(
            field_names={"hidden_dim", "num_heads"},
            func=lambda hidden_dim, num_heads: (hidden_dim / num_heads) < 8,
        ),
        true=sp.Int(ge=8, le=128, default=64),
        false=32,
    )
    qkv_bias: bool = sp.Categorical([True, False], default=False)
    attn_dropout: float = sp.Float(ge=0.0, le=0.5, default=0.0)
    proj_dropout: float = sp.Float(ge=0.0, le=0.5, default=0.0)


# ============================================================================
# Block Configurations (Multiple Layers)
# ============================================================================


class ResidualBlockConfig(sp.Config):
    """Residual block with skip connections."""

    layer: MLPLayerConfig | ConvLayerConfig | AttentionLayerConfig
    use_residual: bool = sp.Conditional(
        condition=sp.FieldCondition(
            "layer",
            sp.IsInstance((MLPLayerConfig, ConvLayerConfig)),
        ),
        true=sp.Categorical([True, False], default=True),
        false=False,  # Attention always uses residual internally
    )
    residual_scale: float = sp.Conditional(
        condition=sp.FieldCondition("use_residual", sp.EqualsTo(True)),
        true=sp.Float(ge=0.1, le=2.0, default=1.0),
        false=1.0,
    )


class SequentialBlockConfig(sp.Config):
    """Sequential block with multiple sub-blocks."""

    num_blocks: int = sp.Int(ge=1, le=5, default=2)
    block_type: Literal["residual", "dense", "simple"]
    blocks: ResidualBlockConfig | MLPLayerConfig = sp.Conditional(
        condition=sp.FieldCondition("block_type", sp.EqualsTo("residual")),
        true=ResidualBlockConfig,
        false=MLPLayerConfig,
    )


# ============================================================================
# Encoder/Decoder Configurations
# ============================================================================


class EncoderConfig(sp.Config):
    """Encoder with various architectures."""

    architecture: Literal["mlp", "cnn", "transformer", "hybrid"]

    # Architecture-specific configs
    mlp_config: MLPLayerConfig | None = sp.Conditional(
        condition=sp.FieldCondition(
            "architecture",
            sp.In(["mlp", "hybrid"]),
        ),
        true=MLPLayerConfig,
        false=None,
    )

    cnn_config: ConvLayerConfig | None = sp.Conditional(
        condition=sp.FieldCondition(
            "architecture",
            sp.In(["cnn", "hybrid"]),
        ),
        true=ConvLayerConfig,
        false=None,
    )

    transformer_config: AttentionLayerConfig | None = sp.Conditional(
        condition=sp.FieldCondition(
            "architecture",
            sp.In(["transformer", "hybrid"]),
        ),
        true=AttentionLayerConfig,
        false=None,
    )

    # Depth depends on architecture
    num_layers: int = sp.Conditional(
        condition=sp.FieldCondition("architecture", sp.EqualsTo("transformer")),
        true=sp.Int(ge=2, le=12, default=6),
        false=sp.Int(ge=1, le=8, default=3),
    )

    # Output dimension
    output_dim: int = sp.Int(ge=16, le=1024, default=256)


class DecoderConfig(sp.Config):
    """Decoder configuration."""

    input_dim: int = sp.Int(ge=16, le=1024, default=256)
    output_dim: int = sp.Int(ge=1, le=10000, default=1000)

    use_intermediate: bool = sp.Categorical([True, False], default=True)
    intermediate_config: MLPLayerConfig | None = sp.Conditional(
        condition=sp.FieldCondition("use_intermediate", sp.EqualsTo(True)),
        true=MLPLayerConfig,
        false=None,
    )


# ============================================================================
# Complete Model Configurations
# ============================================================================


class BaseModelConfig(sp.Config):
    """Base model configuration."""

    model_name: str = sp.Categorical(
        ["base_v1", "base_v2", "base_v3"], default="base_v1"
    )
    seed: int = sp.Int(ge=0, le=999999, default=42)
    use_mixed_precision: bool = sp.Categorical([True, False], default=False)


class VisionModelConfig(BaseModelConfig):
    """Vision model extending base."""

    input_channels: int = sp.Categorical([1, 3, 4], default=3)
    input_size: int = sp.Categorical([32, 64, 128, 224, 256], default=224)

    # Preprocessing
    use_preprocessing: bool = sp.Categorical([True, False], default=True)
    preprocessing: ConvLayerConfig | None = sp.Conditional(
        condition=sp.FieldCondition("use_preprocessing", sp.EqualsTo(True)),
        true=ConvLayerConfig,
        false=None,
    )

    # Main encoder
    encoder: EncoderConfig

    # Task-specific head
    task: Literal["classification", "segmentation", "detection"]
    num_classes: int = sp.Conditional(
        condition=sp.FieldCondition("task", sp.EqualsTo("classification")),
        true=sp.Int(ge=2, le=1000, default=10),
        false=sp.Int(ge=2, le=100, default=20),
    )

    decoder: DecoderConfig | None = sp.Conditional(
        condition=sp.FieldCondition("task", sp.NotEqualsTo("classification")),
        true=DecoderConfig,
        false=None,
    )


class LanguageModelConfig(BaseModelConfig):
    """Language model extending base."""

    vocab_size: int = sp.Int(ge=1000, le=100000, default=50000)
    max_seq_length: int = sp.Categorical([128, 256, 512, 1024], default=512)

    # Embedding
    embedding_dim: int = sp.Int(ge=64, le=1024, default=256)
    use_positional_encoding: bool = sp.Categorical([True, False], default=True)

    # Encoder
    encoder: EncoderConfig

    # Decoder for generative models
    use_decoder: bool = sp.Categorical([True, False], default=False)
    decoder: DecoderConfig | None = sp.Conditional(
        condition=sp.FieldCondition("use_decoder", sp.EqualsTo(True)),
        true=DecoderConfig,
        false=None,
    )


class MultiModalModelConfig(BaseModelConfig):
    """Multi-modal model combining vision and language."""

    vision_encoder: VisionModelConfig
    language_encoder: LanguageModelConfig

    # Fusion strategy
    fusion_type: Literal["concat", "attention", "gated"]
    fusion_dim: int = sp.Int(ge=64, le=2048, default=512)

    fusion_layer: MLPLayerConfig | AttentionLayerConfig = sp.Conditional(
        condition=sp.FieldCondition("fusion_type", sp.EqualsTo("attention")),
        true=AttentionLayerConfig,
        false=MLPLayerConfig,
    )

    # Final decoder
    decoder: DecoderConfig


# ============================================================================
# Training Configuration
# ============================================================================


class OptimizerConfig(sp.Config):
    """Optimizer configuration."""

    type: Literal["adam", "sgd", "adamw", "rmsprop"]
    lr: float = sp.Float(ge=1e-6, le=1e-1, default=1e-3, distribution="log")

    # Adam-specific
    betas: tuple[float, float] | None = sp.Conditional(
        condition=sp.FieldCondition("type", sp.In(["adam", "adamw"])),
        true=(0.9, 0.999),
        false=None,
    )

    # SGD-specific
    momentum: float = sp.Conditional(
        condition=sp.FieldCondition("type", sp.EqualsTo("sgd")),
        true=sp.Float(ge=0.0, le=1.0, default=0.9),
        false=0.0,
    )

    # Weight decay
    weight_decay: float = sp.Float(gt=0.0, le=1e-2, default=1e-4, distribution="log")


class SchedulerConfig(sp.Config):
    """Learning rate scheduler configuration."""

    type: Literal["constant", "step", "cosine", "exponential"]

    # Step-specific
    step_size: int = sp.Conditional(
        condition=sp.FieldCondition("type", sp.EqualsTo("step")),
        true=sp.Int(ge=1, le=100, default=10),
        false=1,
    )
    gamma: float = sp.Conditional(
        condition=sp.FieldCondition("type", sp.In(["step", "exponential"])),
        true=sp.Float(ge=0.1, le=0.99, default=0.5),
        false=1.0,
    )

    # Cosine-specific
    t_max: int = sp.Conditional(
        condition=sp.FieldCondition("type", sp.EqualsTo("cosine")),
        true=sp.Int(ge=10, le=1000, default=100),
        false=100,
    )


class TrainingConfig(sp.Config):
    """Complete training configuration."""

    # Model
    model: VisionModelConfig | LanguageModelConfig | MultiModalModelConfig

    # Training hyperparameters
    batch_size: int = sp.Int(ge=1, le=512, default=32)
    num_epochs: int = sp.Int(ge=1, le=1000, default=100)

    # Optimizer and scheduler
    optimizer: OptimizerConfig
    use_scheduler: bool = sp.Categorical([True, False], default=True)
    scheduler: SchedulerConfig | None = sp.Conditional(
        condition=sp.FieldCondition("use_scheduler", sp.EqualsTo(True)),
        true=SchedulerConfig,
        false=None,
    )

    # Regularization
    use_augmentation: bool = sp.Categorical([True, False], default=True)
    augmentation_strength: float = sp.Conditional(
        condition=sp.FieldCondition("use_augmentation", sp.EqualsTo(True)),
        true=sp.Float(ge=0.0, le=1.0, default=0.5),
        false=0.0,
    )

    gradient_clip: float | None = sp.Conditional(
        condition=sp.FieldCondition("model", sp.IsInstance(LanguageModelConfig)),
        true=sp.Float(ge=0.1, le=10.0, default=1.0),
        false=None,
    )


# ============================================================================
# Tests
# ============================================================================


def test_simple_nested_sampling():
    """Test simple nested config sampling."""
    config = ActivationConfig.random(seed=42)
    assert config.type in ["relu", "gelu", "silu", "tanh"]
    if config.type == "relu":
        assert 0.0 <= config.negative_slope <= 1.0
    else:
        assert config.negative_slope == 0.0


def test_conditional_chaining():
    """Test multiple conditional dependencies."""
    config = NormConfig.random(seed=42)
    assert config.type in ["batch", "layer", "instance", "none"]

    if config.type != "none":
        assert 1e-8 <= config.eps <= 1e-3

    if config.type == "batch":
        assert 0.0 <= config.momentum <= 1.0
    else:
        assert config.momentum == 0.0


def test_inherited_configs():
    """Test configs with inheritance."""
    for _ in range(10):
        config = MLPLayerConfig.random()

        # Base class fields
        assert 8 <= config.hidden_dim <= 2048
        assert 0.0 <= config.dropout <= 0.9
        assert isinstance(config.use_bias, bool)

        # Derived class fields
        assert 1 <= config.num_layers <= 10
        assert isinstance(config.activation, ActivationConfig)
        assert isinstance(config.norm, NormConfig)


def test_multi_field_conditionals():
    """Test conditionals depending on multiple fields."""
    for _ in range(20):
        config = ConvLayerConfig.random()

        # Check padding conditional
        if config.kernel_size >= 1:
            assert 0 <= config.padding <= 5
        else:
            assert config.padding == 0

        # Check groups conditional
        if config.hidden_dim >= 32 and config.kernel_size >= 3:
            assert config.groups in [1, 2, 4, 8]
        else:
            assert config.groups == 1


def test_categorical_config_choices():
    """Test categorical choices between config types."""
    for _ in range(20):
        config = ResidualBlockConfig.random()

        assert isinstance(
            config.layer, (MLPLayerConfig, ConvLayerConfig, AttentionLayerConfig)
        )

        # Check use_residual logic
        if isinstance(config.layer, AttentionLayerConfig):
            assert config.use_residual is False

        if config.use_residual:
            assert 0.1 <= config.residual_scale <= 2.0


def test_deeply_nested_configs():
    """Test deeply nested configuration hierarchies."""
    for _ in range(10):
        config = EncoderConfig.random()

        assert config.architecture in ["mlp", "cnn", "transformer", "hybrid"]
        assert 16 <= config.output_dim <= 1024

        # Check architecture-specific configs
        if config.architecture in ["mlp", "hybrid"]:
            assert isinstance(config.mlp_config, MLPLayerConfig)
        else:
            assert config.mlp_config is None

        if config.architecture in ["cnn", "hybrid"]:
            assert isinstance(config.cnn_config, ConvLayerConfig)
        else:
            assert config.cnn_config is None

        if config.architecture in ["transformer", "hybrid"]:
            assert isinstance(config.transformer_config, AttentionLayerConfig)
        else:
            assert config.transformer_config is None

        # Check num_layers conditional
        if config.architecture == "transformer":
            assert 2 <= config.num_layers <= 12
        else:
            assert 1 <= config.num_layers <= 8


def test_vision_model_complete():
    """Test complete vision model with all dependencies."""
    for _ in range(10):
        config = VisionModelConfig.random()

        # Base model fields
        assert config.model_name in ["base_v1", "base_v2", "base_v3"]
        assert 0 <= config.seed <= 999999

        # Vision-specific
        assert config.input_channels in [1, 3, 4]
        assert config.input_size in [32, 64, 128, 224, 256]

        # Preprocessing
        if config.use_preprocessing:
            assert isinstance(config.preprocessing, ConvLayerConfig)
        else:
            assert config.preprocessing is None

        # Encoder
        assert isinstance(config.encoder, EncoderConfig)

        # Task-specific
        assert config.task in ["classification", "segmentation", "detection"]

        if config.task == "classification":
            assert 2 <= config.num_classes <= 1000
            assert config.decoder is None
        else:
            assert 2 <= config.num_classes <= 100
            assert isinstance(config.decoder, DecoderConfig)


def test_multimodal_extreme_nesting():
    """Test multi-modal model with extreme nesting depth."""
    for _ in range(5):  # Fewer iterations due to complexity
        config = MultiModalModelConfig.random()

        # Vision encoder (deeply nested)
        assert isinstance(config.vision_encoder, VisionModelConfig)
        assert isinstance(config.vision_encoder.encoder, EncoderConfig)

        # Language encoder (deeply nested)
        assert isinstance(config.language_encoder, LanguageModelConfig)
        assert isinstance(config.language_encoder.encoder, EncoderConfig)

        # Fusion
        assert config.fusion_type in ["concat", "attention", "gated"]
        assert 64 <= config.fusion_dim <= 2048

        if config.fusion_type == "attention":
            assert isinstance(config.fusion_layer, AttentionLayerConfig)
        else:
            assert isinstance(config.fusion_layer, MLPLayerConfig)

        # Decoder
        assert isinstance(config.decoder, DecoderConfig)


def test_training_config_complete():
    """Test complete training configuration."""
    for _ in range(10):
        config = TrainingConfig.random()

        # Model (can be any of the three types)
        assert isinstance(
            config.model,
            (VisionModelConfig, LanguageModelConfig, MultiModalModelConfig),
        )

        # Training params
        assert 1 <= config.batch_size <= 512
        assert 1 <= config.num_epochs <= 1000

        # Optimizer
        assert isinstance(config.optimizer, OptimizerConfig)
        assert config.optimizer.type in ["adam", "sgd", "adamw", "rmsprop"]

        if config.optimizer.type in ["adam", "adamw"]:
            assert config.optimizer.betas == (0.9, 0.999)

        # Scheduler
        if config.use_scheduler:
            assert isinstance(config.scheduler, SchedulerConfig)
        else:
            assert config.scheduler is None

        # Augmentation
        if config.use_augmentation:
            assert 0.0 <= config.augmentation_strength <= 1.0

        # Gradient clipping only for language models
        if isinstance(config.model, LanguageModelConfig):
            assert config.gradient_clip is not None
            assert 0.1 <= config.gradient_clip <= 10.0
        else:
            assert config.gradient_clip is None


def test_sampling_with_overrides():
    """Test sampling with complex overrides."""
    override = {
        "model": {
            "VisionModelConfig": {
                "input_size": 224,
                "encoder": {
                    "architecture": "transformer",
                    "transformer_config": {
                        "true": {
                            "hidden_dim": {"ge": 256, "le": 512},
                        }
                    },
                },
            },
        },
        "batch_size": 64,
        "optimizer": {
            "type": "adamw",
            "lr": {"ge": 1e-4, "le": 1e-3},
        },
    }

    for _ in range(10):
        config = TrainingConfig.random(override=override)

        # Check overrides were applied
        if isinstance(config.model, VisionModelConfig):
            assert config.model.input_size == 224
            assert config.model.encoder.architecture == "transformer"
            if config.model.encoder.transformer_config:
                assert 256 <= config.model.encoder.transformer_config.hidden_dim <= 512

        assert config.batch_size == 64
        assert config.optimizer.type == "adamw"
        assert 1e-4 <= config.optimizer.lr <= 1e-3


def test_reproducibility():
    """Test that same seed produces same results."""
    config1 = TrainingConfig.random(seed=12345)
    config2 = TrainingConfig.random(seed=12345)

    # Should produce identical configs
    assert config1.model_dump() == config2.model_dump()


def test_parameter_names_extreme():
    """Test parameter name generation for extremely nested configs."""
    names = TrainingConfig.get_parameter_names()

    # Should have many parameters
    assert len(names) > 50

    # Check some expected patterns
    assert any("model" in name for name in names)
    assert any("optimizer" in name for name in names)
    assert any("encoder" in name for name in names)

    # All names should be unique
    assert len(names) == len(set(names))


def test_stress_test_100_samples():
    """Stress test: generate 100 random configs."""
    configs = []
    for i in range(100):
        config = TrainingConfig.random(seed=i)
        configs.append(config)

        # Basic validation
        assert isinstance(config, TrainingConfig)
        assert 1 <= config.batch_size <= 512

    # Should have variety
    model_types = [type(c.model).__name__ for c in configs]
    assert len(set(model_types)) > 1  # Should sample different model types
