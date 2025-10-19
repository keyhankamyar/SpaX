"""
Extreme nested configuration tests - deeply nested, multi-level configs
with complex conditional dependencies, nested Config types, and edge cases.
"""

import pytest

from spax import (
    And,
    Categorical,
    Choice,
    Conditional,
    Config,
    EqualsTo,
    FieldCondition,
    Float,
    In,
    Int,
    Lambda,
    LargerThan,
    Not,
    Or,
)


class TestExtremeNestedConfigs:
    """Test extremely complex nested configuration scenarios."""

    def test_five_level_nested_config_with_mixed_conditions(self):
        """
        Test a 5-level deep nested Config with various condition types.
        Structure: MainConfig -> Level1 -> Level2 -> Level3 -> Level4 -> Level5
        """

        # Level 5 (deepest)
        class Level5Config(Config):
            final_param: float = Float(ge=0.0, le=1.0, default=0.5)

        # Level 4
        class Level4Config(Config):
            l4_type: str = Categorical(["simple", "complex"])
            l4_value: int = Conditional(
                condition=FieldCondition("l4_type", EqualsTo("complex")),
                true=Int(ge=100, le=1000),
                false=Int(ge=1, le=10),
            )
            nested_l5: Level5Config = Categorical([Level5Config])

        # Level 3
        class Level3Config(Config):
            l3_mode: str = Categorical(["basic", "advanced", "expert"])
            l3_threshold: float = Conditional(
                condition=FieldCondition("l3_mode", In(["advanced", "expert"])),
                true=Float(ge=0.5, le=1.0),
                false=Float(ge=0.0, le=0.5),
            )
            l3_iterations: int = Conditional(
                condition=And(
                    [
                        FieldCondition("l3_mode", EqualsTo("expert")),
                        FieldCondition("l3_threshold", LargerThan(0.7)),
                    ]
                ),
                true=Int(ge=1000, le=10000),
                false=Int(ge=10, le=100),
            )
            nested_l4: Level4Config = Categorical([Level4Config])

        # Level 2
        class Level2Config(Config):
            l2_enabled: bool = Categorical([True, False])
            l2_strategy: str = Conditional(
                condition=FieldCondition("l2_enabled", EqualsTo(True)),
                true=Categorical(["aggressive", "conservative", "balanced"]),
                false="disabled",
            )
            l2_multiplier: float = Conditional(
                condition=And(
                    [
                        FieldCondition("l2_enabled", EqualsTo(True)),
                        FieldCondition("l2_strategy", In(["aggressive", "balanced"])),
                    ]
                ),
                true=Float(ge=1.5, le=3.0),
                false=1.0,
            )
            nested_l3: Level3Config = Categorical([Level3Config])

        # Level 1
        class Level1Config(Config):
            l1_architecture: str = Categorical(["small", "medium", "large"])
            l1_depth: int = Conditional(
                condition=FieldCondition("l1_architecture", EqualsTo("large")),
                true=Int(ge=50, le=100),
                false=Conditional(
                    condition=FieldCondition("l1_architecture", EqualsTo("medium")),
                    true=Int(ge=20, le=50),
                    false=Int(ge=5, le=20),
                ),
            )
            nested_l2: Level2Config = Categorical([Level2Config])

        # Main config
        class MainConfig(Config):
            experiment_type: str = Categorical(["research", "production"])
            use_advanced_features: bool = Categorical([True, False])
            main_lr: float = Conditional(
                condition=And(
                    [
                        FieldCondition("experiment_type", EqualsTo("research")),
                        FieldCondition("use_advanced_features", EqualsTo(True)),
                    ]
                ),
                true=Float(ge=1e-5, le=1e-2, distribution="log"),
                false=Float(ge=1e-4, le=1e-1, distribution="log"),
            )
            nested_l1: Level1Config = Categorical([Level1Config])

        # Test 1: Random sampling works through all levels
        for _ in range(10):
            config = MainConfig.random()

            # Verify all nested configs are instantiated
            assert isinstance(config.nested_l1, Level1Config)
            assert isinstance(config.nested_l1.nested_l2, Level2Config)
            assert isinstance(config.nested_l1.nested_l2.nested_l3, Level3Config)
            assert isinstance(
                config.nested_l1.nested_l2.nested_l3.nested_l4, Level4Config
            )
            assert isinstance(
                config.nested_l1.nested_l2.nested_l3.nested_l4.nested_l5, Level5Config
            )

            # Verify conditional constraints at each level
            if config.experiment_type == "research" and config.use_advanced_features:
                assert 1e-5 <= config.main_lr <= 1e-2
            else:
                assert 1e-4 <= config.main_lr <= 1e-1

            if config.nested_l1.l1_architecture == "large":
                assert 50 <= config.nested_l1.l1_depth <= 100
            elif config.nested_l1.l1_architecture == "medium":
                assert 20 <= config.nested_l1.l1_depth <= 50
            else:
                assert 5 <= config.nested_l1.l1_depth <= 20

            l2 = config.nested_l1.nested_l2
            if l2.l2_enabled:
                assert l2.l2_strategy in ["aggressive", "conservative", "balanced"]
                if l2.l2_strategy in ["aggressive", "balanced"]:
                    assert 1.5 <= l2.l2_multiplier <= 3.0
            else:
                assert l2.l2_strategy == "disabled"
                assert l2.l2_multiplier == 1.0

        # Test 2: Manual construction with specific values
        config = MainConfig(
            experiment_type="research",
            use_advanced_features=True,
            main_lr=0.001,
            nested_l1=Level1Config(
                l1_architecture="large",
                l1_depth=75,
                nested_l2=Level2Config(
                    l2_enabled=True,
                    l2_strategy="aggressive",
                    l2_multiplier=2.5,
                    nested_l3=Level3Config(
                        l3_mode="expert",
                        l3_threshold=0.8,
                        l3_iterations=5000,
                        nested_l4=Level4Config(
                            l4_type="complex",
                            l4_value=500,
                            nested_l5=Level5Config(final_param=0.7),
                        ),
                    ),
                ),
            ),
        )

        # Verify the entire structure
        assert config.main_lr == 0.001
        assert config.nested_l1.l1_depth == 75
        assert config.nested_l1.nested_l2.l2_multiplier == 2.5
        assert config.nested_l1.nested_l2.nested_l3.l3_iterations == 5000
        assert config.nested_l1.nested_l2.nested_l3.nested_l4.l4_value == 500
        assert (
            config.nested_l1.nested_l2.nested_l3.nested_l4.nested_l5.final_param == 0.7
        )

    def test_cross_level_dependencies_with_or_conditions(self):
        """
        Test nested configs where inner levels depend on outer level values.
        This tests the dependency graph with nested conditionals.
        """

        # Inner config that depends on outer config values (passed through)
        class InnerConfig(Config):
            inner_mode: str = Categorical(["fast", "accurate", "balanced"])
            inner_param: float = Float(ge=0.0, le=10.0)

        class MiddleConfig(Config):
            middle_type: str = Categorical(["typeA", "typeB", "typeC"])
            middle_value: int = Conditional(
                condition=FieldCondition("middle_type", In(["typeA", "typeB"])),
                true=Int(ge=100, le=500),
                false=Int(ge=1, le=50),
            )
            # Nested conditional on nested conditional
            middle_multiplier: float = Conditional(
                condition=FieldCondition("middle_type", EqualsTo("typeA")),
                true=Conditional(
                    condition=FieldCondition("middle_value", LargerThan(300)),
                    true=Float(ge=2.0, le=5.0),
                    false=Float(ge=1.0, le=2.0),
                ),
                false=Float(ge=0.1, le=1.0),
            )
            inner: InnerConfig = Categorical([InnerConfig])

        class OuterConfig(Config):
            outer_mode: str = Categorical(["development", "staging", "production"])
            outer_enabled: bool = Categorical([True, False])
            # Complex Or condition
            outer_threshold: float = Conditional(
                condition=Or(
                    [
                        FieldCondition("outer_mode", EqualsTo("production")),
                        And(
                            [
                                FieldCondition("outer_mode", EqualsTo("staging")),
                                FieldCondition("outer_enabled", EqualsTo(True)),
                            ]
                        ),
                    ]
                ),
                true=Float(ge=0.9, le=1.0),
                false=Float(ge=0.5, le=0.9),
            )
            middle: MiddleConfig = Categorical([MiddleConfig])

        # Test 1: Random sampling respects Or conditions
        for _ in range(30):
            config = OuterConfig.random()

            # Verify Or condition logic
            if (
                config.outer_mode == "production"
                or config.outer_mode == "staging"
                and config.outer_enabled
            ):
                assert 0.9 <= config.outer_threshold <= 1.0
            else:
                assert 0.5 <= config.outer_threshold <= 0.9

            # Verify nested conditional on nested conditional
            middle = config.middle
            if middle.middle_type == "typeA":
                if middle.middle_value > 300:
                    assert 2.0 <= middle.middle_multiplier <= 5.0
                else:
                    assert 1.0 <= middle.middle_multiplier <= 2.0
            else:
                assert 0.1 <= middle.middle_multiplier <= 1.0

        # Test 2: Manual construction with all Or branches
        # Branch 1: production (first condition of Or is True)
        config1 = OuterConfig(
            outer_mode="production",
            outer_enabled=False,
            outer_threshold=0.95,
            middle=MiddleConfig(
                middle_type="typeA",
                middle_value=350,
                middle_multiplier=3.0,
                inner=InnerConfig(inner_mode="accurate", inner_param=5.0),
            ),
        )
        assert 0.9 <= config1.outer_threshold <= 1.0
        assert 2.0 <= config1.middle.middle_multiplier <= 5.0

        # Branch 2: staging + enabled (second condition of Or is True)
        config2 = OuterConfig(
            outer_mode="staging",
            outer_enabled=True,
            outer_threshold=0.92,
            middle=MiddleConfig(
                middle_type="typeB",
                middle_value=200,
                middle_multiplier=0.5,
                inner=InnerConfig(inner_mode="fast", inner_param=2.0),
            ),
        )
        assert 0.9 <= config2.outer_threshold <= 1.0

        # Branch 3: neither condition (Or is False)
        config3 = OuterConfig(
            outer_mode="development",
            outer_enabled=False,
            outer_threshold=0.7,
            middle=MiddleConfig(
                middle_type="typeC",
                middle_value=25,
                middle_multiplier=0.3,
                inner=InnerConfig(inner_mode="balanced", inner_param=7.0),
            ),
        )
        assert 0.5 <= config3.outer_threshold <= 0.9

    def test_nested_validation_errors_propagate_correctly(self):
        """
        Test that validation errors in deeply nested configs
        provide clear error messages and propagate correctly.
        """

        class Level3Config(Config):
            value: int = Int(ge=0, le=100)

        class Level2Config(Config):
            multiplier: float = Float(ge=1.0, le=10.0)
            nested: Level3Config = Categorical([Level3Config])

        class Level1Config(Config):
            mode: str = Categorical(["a", "b", "c"])
            param: float = Conditional(
                condition=FieldCondition("mode", EqualsTo("a")),
                true=Float(ge=10.0, le=20.0),
                false=Float(ge=0.0, le=5.0),
            )
            nested: Level2Config = Categorical([Level2Config])

        # Test 1: Invalid value at deepest level
        with pytest.raises(ValueError, match="value.*must be <="):
            Level1Config(
                mode="a",
                param=15.0,
                nested=Level2Config(
                    multiplier=5.0,
                    nested=Level3Config(value=150),  # Invalid: > 100
                ),
            )

        # Test 2: Invalid value at middle level
        with pytest.raises(ValueError, match="multiplier.*must be <="):
            Level1Config(
                mode="b",
                param=3.0,
                nested=Level2Config(
                    multiplier=15.0,  # Invalid: > 10.0
                    nested=Level3Config(value=50),
                ),
            )

        # Test 3: Invalid conditional value at top level
        with pytest.raises(ValueError, match="param.*must be <="):
            Level1Config(
                mode="a",
                param=25.0,  # Invalid: mode is "a" so must be in [10, 20]
                nested=Level2Config(multiplier=5.0, nested=Level3Config(value=50)),
            )

        # Test 4: Wrong conditional branch value
        with pytest.raises(ValueError):
            Level1Config(
                mode="b",  # mode is "b", so param must be in [0, 5]
                param=15.0,  # Invalid: > 5.0 for mode "b"
                nested=Level2Config(multiplier=5.0, nested=Level3Config(value=50)),
            )

        # Test 5: Valid construction should work
        config = Level1Config(
            mode="a",
            param=15.0,
            nested=Level2Config(multiplier=5.0, nested=Level3Config(value=50)),
        )
        assert config.param == 15.0
        assert config.nested.nested.value == 50

    def test_multiple_nested_config_types_as_choices(self):
        """
        Test Categorical spaces with multiple different nested Config types,
        including conditionals that depend on which type was chosen.
        """

        # Different optimizer configs
        class AdamConfig(Config):
            lr: float = Float(ge=1e-5, le=1e-2, distribution="log", default=0.001)
            beta1: float = Float(ge=0.8, le=0.999, default=0.9)
            beta2: float = Float(ge=0.9, le=0.9999, default=0.999)

        class SGDConfig(Config):
            lr: float = Float(ge=1e-3, le=1e-1, distribution="log", default=0.01)
            momentum: float = Float(ge=0.0, le=0.99, default=0.9)
            nesterov: bool = Categorical([True, False], default=True)

        class RMSpropConfig(Config):
            lr: float = Float(ge=1e-4, le=1e-1, distribution="log", default=0.001)
            alpha: float = Float(ge=0.9, le=0.999, default=0.99)
            eps: float = Float(ge=1e-10, le=1e-6, distribution="log", default=1e-8)

        # Scheduler configs
        class StepLRConfig(Config):
            step_size: int = Int(ge=1, le=100, default=10)
            gamma: float = Float(ge=0.1, le=0.9, default=0.5)

        class CosineAnnealingConfig(Config):
            t_max: int = Int(ge=10, le=1000, default=100)
            eta_min: float = Float(ge=0.0, le=1e-3, default=1e-6)

        class NoSchedulerConfig(Config):
            # Empty config representing no scheduler
            pass

        # Main training config
        class TrainingConfig(Config):
            optimizer: AdamConfig | SGDConfig | RMSpropConfig = Categorical(
                [
                    Choice(AdamConfig, weight=3.0),  # Prefer Adam
                    Choice(SGDConfig, weight=1.5),
                    Choice(RMSpropConfig, weight=0.5),
                ]
            )
            scheduler: StepLRConfig | CosineAnnealingConfig | NoSchedulerConfig = (
                Categorical(
                    [
                        Choice(StepLRConfig, weight=2.0),
                        Choice(CosineAnnealingConfig, weight=2.0),
                        Choice(NoSchedulerConfig, weight=1.0),
                    ]
                )
            )
            epochs: int = Int(ge=1, le=1000, default=100)
            batch_size: int = Int(ge=1, le=512, distribution="log", default=32)

        # Test 1: Random sampling creates valid nested configs
        optimizer_types = []
        scheduler_types = []

        for _ in range(50):
            config = TrainingConfig.random()

            # Verify optimizer is one of the three types
            assert isinstance(config.optimizer, (AdamConfig, SGDConfig, RMSpropConfig))
            optimizer_types.append(type(config.optimizer).__name__)

            # Verify scheduler is one of the three types
            assert isinstance(
                config.scheduler,
                (StepLRConfig, CosineAnnealingConfig, NoSchedulerConfig),
            )
            scheduler_types.append(type(config.scheduler).__name__)

            # Verify type-specific constraints
            if isinstance(config.optimizer, AdamConfig):
                assert 1e-5 <= config.optimizer.lr <= 1e-2
                assert 0.8 <= config.optimizer.beta1 <= 0.999
                assert 0.9 <= config.optimizer.beta2 <= 0.9999
            elif isinstance(config.optimizer, SGDConfig):
                assert 1e-3 <= config.optimizer.lr <= 1e-1
                assert 0.0 <= config.optimizer.momentum <= 0.99
            elif isinstance(config.optimizer, RMSpropConfig):
                assert 1e-4 <= config.optimizer.lr <= 1e-1
                assert 0.9 <= config.optimizer.alpha <= 0.999

        # Verify weighted sampling (Adam should be most common)
        adam_count = optimizer_types.count("AdamConfig")
        sgd_count = optimizer_types.count("SGDConfig")
        rmsprop_count = optimizer_types.count("RMSpropConfig")

        # With weights 3:1.5:0.5, expect roughly 60%:30%:10%
        assert adam_count > sgd_count  # Adam should be more common than SGD
        assert sgd_count > rmsprop_count  # SGD should be more common than RMSprop

        # Test 2: Manual construction with each optimizer type
        config_adam = TrainingConfig(
            optimizer=AdamConfig(lr=0.001, beta1=0.9, beta2=0.999),
            scheduler=StepLRConfig(step_size=10, gamma=0.5),
            epochs=100,
            batch_size=32,
        )
        assert isinstance(config_adam.optimizer, AdamConfig)
        assert config_adam.optimizer.lr == 0.001

        config_sgd = TrainingConfig(
            optimizer=SGDConfig(lr=0.01, momentum=0.9, nesterov=True),
            scheduler=CosineAnnealingConfig(t_max=100, eta_min=1e-6),
            epochs=200,
            batch_size=64,
        )
        assert isinstance(config_sgd.optimizer, SGDConfig)
        assert config_sgd.optimizer.nesterov is True

        config_rmsprop = TrainingConfig(
            optimizer=RMSpropConfig(lr=0.001, alpha=0.99, eps=1e-8),
            scheduler=NoSchedulerConfig(),
            epochs=50,
            batch_size=16,
        )
        assert isinstance(config_rmsprop.optimizer, RMSpropConfig)
        assert isinstance(config_rmsprop.scheduler, NoSchedulerConfig)

        # Test 3: Verify space info captures nested types
        space_info = TrainingConfig.get_space_info()
        assert "optimizer" in space_info
        assert space_info["optimizer"]["type"] == "CategoricalSpace"
        assert len(space_info["optimizer"]["choices"]) == 3

    def test_not_and_lambda_conditions_in_nested_config(self):
        """
        Test Not conditions and Lambda conditions in nested configs.
        These are less commonly used but important edge cases.
        """

        class AdvancedConfig(Config):
            mode: str = Categorical(["auto", "manual", "hybrid"])
            precision: str = Categorical(["fp32", "fp16", "bf16"])

            # Not condition: only enable when NOT in auto mode
            manual_param: float = Conditional(
                condition=Not(FieldCondition("mode", EqualsTo("auto"))),
                true=Float(ge=0.0, le=1.0),
                false=0.0,
            )

            # Lambda condition: custom logic
            precision_threshold: float = Conditional(
                condition=FieldCondition(
                    "precision",
                    Lambda(lambda x: x in ["fp16", "bf16"]),  # Reduced precision
                ),
                true=Float(ge=1e-4, le=1e-2),  # Higher threshold for stability
                false=Float(ge=1e-6, le=1e-4),  # Lower threshold for full precision
            )

            # Complex: Not with And
            aggressive_mode: bool = Conditional(
                condition=And(
                    [
                        Not(FieldCondition("mode", EqualsTo("auto"))),
                        FieldCondition("manual_param", Lambda(lambda x: x > 0.5)),
                    ]
                ),
                true=Categorical([True, False]),
                false=False,
            )

        class OuterConfig(Config):
            enable_advanced: bool = Categorical([True, False])

            # Conditional nested Config based on boolean
            advanced: AdvancedConfig | None = Conditional(
                condition=FieldCondition("enable_advanced", EqualsTo(True)),
                true=Categorical([AdvancedConfig]),
                false=None,
            )

            # Lambda condition on outer level
            safety_factor: float = Conditional(
                condition=FieldCondition(
                    "enable_advanced",
                    Lambda(lambda x: isinstance(x, bool) and x is True),
                ),
                true=Float(ge=0.5, le=1.0),
                false=Float(ge=0.0, le=0.5),
            )

        # Test 1: Not condition - auto mode
        config_auto = AdvancedConfig(
            mode="auto",
            precision="fp32",
            manual_param=0.0,
            precision_threshold=1e-5,
            aggressive_mode=False,
        )
        assert config_auto.manual_param == 0.0  # Not condition is False
        assert config_auto.aggressive_mode is False

        # Test 2: Not condition - manual mode
        config_manual = AdvancedConfig(
            mode="manual",
            precision="fp16",
            manual_param=0.7,
            precision_threshold=0.005,
            aggressive_mode=True,
        )
        assert 0.0 <= config_manual.manual_param <= 1.0  # Not condition is True
        assert 1e-4 <= config_manual.precision_threshold <= 1e-2  # Lambda is True
        assert config_manual.aggressive_mode in [True, False]  # Can be either

        # Test 3: Lambda condition with fp16
        config_fp16 = AdvancedConfig(
            mode="hybrid",
            precision="fp16",
            manual_param=0.3,
            precision_threshold=0.008,
            aggressive_mode=False,
        )
        assert 1e-4 <= config_fp16.precision_threshold <= 1e-2

        # Test 4: Lambda condition with fp32 (not reduced precision)
        config_fp32 = AdvancedConfig(
            mode="hybrid",
            precision="fp32",
            manual_param=0.3,
            precision_threshold=5e-5,
            aggressive_mode=False,
        )
        assert 1e-6 <= config_fp32.precision_threshold <= 1e-4

        # Test 5: Nested None vs Config
        outer_disabled = OuterConfig(
            enable_advanced=False, advanced=None, safety_factor=0.3
        )
        assert outer_disabled.advanced is None
        assert 0.0 <= outer_disabled.safety_factor <= 0.5

        outer_enabled = OuterConfig(
            enable_advanced=True,
            advanced=AdvancedConfig(
                mode="manual",
                precision="bf16",
                manual_param=0.8,
                precision_threshold=0.009,
                aggressive_mode=True,
            ),
            safety_factor=0.7,
        )
        assert isinstance(outer_enabled.advanced, AdvancedConfig)
        assert 0.5 <= outer_enabled.safety_factor <= 1.0

        # Test 6: Random sampling with Not and Lambda
        for _ in range(30):
            config = AdvancedConfig.random()

            # Verify Not condition
            if config.mode == "auto":
                assert config.manual_param == 0.0
            else:
                assert 0.0 <= config.manual_param <= 1.0

            # Verify Lambda condition
            if config.precision in ["fp16", "bf16"]:
                assert 1e-4 <= config.precision_threshold <= 1e-2
            else:
                assert 1e-6 <= config.precision_threshold <= 1e-4

            # Verify complex condition (Not with And + Lambda)
            if config.mode != "auto" and config.manual_param > 0.5:
                assert config.aggressive_mode in [True, False]
            else:
                assert config.aggressive_mode is False

        # Test 7: Random outer config
        for _ in range(20):
            outer = OuterConfig.random()

            if outer.enable_advanced:
                assert isinstance(outer.advanced, AdvancedConfig)
                assert 0.5 <= outer.safety_factor <= 1.0
            else:
                assert outer.advanced is None
                assert 0.0 <= outer.safety_factor <= 0.5
