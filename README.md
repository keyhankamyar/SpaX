<!--
SpaX - Hyperparameter Optimization | Neural Architecture Search | ML Experiment Tracking
Python library for type-safe search space definition, hyperparameter tuning, and ML configuration management.
Keywords: HPO, hyperparameter optimization, neural architecture search, NAS, AutoML, Optuna integration,
Pydantic configuration, conditional parameters, experiment tracking, MLOps, PyTorch, TensorFlow,
machine learning configuration, parameter sampling, Bayesian optimization, grid search, random search,
reproducible experiments, type-safe ML, declarative configuration, search space exploration,
model tuning, hyperparameter sweep, configuration validation, ML pipeline, deep learning experiments
-->
<div align="center">

# SpaX

### Pythonic, type-safe search space definition and exploration

[![CI](https://github.com/keyhankamyar/SpaX/actions/workflows/ci.yml/badge.svg)](https://github.com/keyhankamyar/SpaX/actions/workflows/ci.yml) [![PyPI](https://img.shields.io/pypi/v/spax.svg)](https://pypi.org/project/spax/) [![Python 3.11-3.14](https://img.shields.io/badge/python-3.11--3.14-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![Coverage 93%](https://img.shields.io/badge/coverage-93%25-brightgreen.svg)](https://github.com/keyhankamyar/SpaX)

</div>

---

## 📋 Overview

Defining hyperparameter spaces, neural architectures, and complex configurations often means writing repetitive boilerplate, dealing with silent validation errors, and struggling to enforce best practices. SpaX is a Pydantic-based configuration framework that eliminates these pain points through declarative, type-safe search space definitions.

Built for ML experimentation but useful anywhere you need robust configuration management, SpaX catches invalid parameter combinations at definition time, enforces constraints automatically, and integrates seamlessly with HPO frameworks like Optuna. Whether you're tuning hyperparameters, exploring architectures, or managing production configs, SpaX reduces bugs and saves time.

**What you get:**
- **Zero boilerplate** — One-line migration from Pydantic, automatic space inference from type hints
- **Early error detection** — Invalid configurations caught at definition time, not during training
- **Declarative constraints** — Conditional parameters, nested configs, polymorphic fields all type-safe
- **Seamless HPO** — Direct Optuna integration with all features working automatically
- **Iterative refinement** — Progressive search space narrowing based on experimental results
- **Full serialization** — Save and load configurations in JSON, YAML, or TOML

---

## 📦 Installation

**Base installation:**
```bash
pip install spax
```

**With optional dependencies:**
```bash
# YAML serialization support
pip install spax[yaml]

# TOML serialization support
pip install spax[toml]

# Optuna integration
pip install spax[optuna]

# All optional features
pip install spax[all]
```

**Requirements:**
- Python 3.11+
- Pydantic 2.7+

---

## ⚡ Quick Example
Define your search space once, get validation, sampling, visualization, and HPO integration automatically:
```python
from typing import Literal
from pydantic import Field
import spax as sp


class ModelConfig(sp.Config):
    # Automatic inference from type hints
    optimizer: Literal["adam", "sgd", "rmsprop"]
    use_scheduler: bool
    use_dropout: bool

    # Pydantic Field constraints work too
    batch_size: int = Field(ge=16, le=128)

    # Explicit SpaX spaces for full control
    learning_rate: float = sp.Float(ge=1e-5, le=1e-1, distribution="log")

    # Conditional spaces - parameters that depend on others
    num_layers: int = sp.Conditional(
        sp.FieldCondition("use_scheduler", sp.EqualsTo(True)),
        true=sp.Int(ge=6, le=12),  # Deep networks with scheduler
        false=sp.Int(ge=2, le=6),  # Shallow networks without
    )

    # dropout_rate only exists when use_dropout=True
    dropout_rate: float = sp.Conditional(
        sp.FieldCondition("use_dropout", sp.EqualsTo(True)),
        true=sp.Float(ge=0.1, le=0.5),
        false=0.0,
    )
```

**Visualize your search space:**
```python
print(ModelConfig.get_tree())
# ModelConfig
# ├─ optimizer: Categorical
# │  ├─ 'adam'
# │  ├─ 'sgd'
# │  └─ 'rmsprop'
# ├─ use_scheduler: Categorical
# │  ├─ True
# │  └─ False
# ├─ use_dropout: Categorical
# │  ├─ True
# │  └─ False
# ├─ batch_size: Int([16, 128], uniform)
# ├─ learning_rate: Float([1e-05, 0.1], log)
# ├─ num_layers: Conditional (if use_scheduler == True)
# │  ├─ true: Int([6, 12], uniform)
# │  └─ false: Int([2, 6], uniform)
# └─ dropout_rate: Conditional (if use_dropout == True)
#    ├─ true: Float([0.1, 0.5], uniform)
#    └─ false: 0.0
```

**Random sampling for testing:**
```python
config = ModelConfig.random(seed=42)
print(config)
# ModelConfig(optimizer='rmsprop', use_scheduler=True, use_dropout=True, batch_size=97,
#             learning_rate=2.788e-05, num_layers=11, dropout_rate=0.155815)
```

**Serialization:** (save/load in multiple formats)
```python
yaml_str = config.model_dump_yaml()
loaded = ModelConfig.model_validate_yaml(yaml_str)
# Also: model_dump_json/toml, model_validate_json/toml
```

**Iterative refinement:** (narrow the search space based on results)
```python
config_v2 = ModelConfig.random(
    seed=42,
    override={
        "learning_rate": {"ge": 1e-4, "le": 1e-2},  # Focus on promising region
        "optimizer": "adam",  # Lock to best optimizer
    },
)
print(config_v2.learning_rate)  # Now in [1e-4, 1e-2] range

# See the full override template as a reference:
print(ModelConfig.get_override_template())
# {
#     "optimizer": ["adam", "sgd", "rmsprop"],
#     "use_scheduler": ["True", "False"],
#     "dropout_rate": {"true": {"ge": 0.1, "le": 0.5}},
#     "batch_size": {"ge": 16, "le": 128},
#     "learning_rate": {"ge": 1e-05, "le": 0.1},
#     "num_layers": {"true": {"ge": 6, "le": 12}, "false": {"ge": 2, "le": 6}},
#     "use_dropout": ["True", "False"],
# }
```

**Seamless Optuna integration:**
```python
import optuna

def objective(trial: optuna.Trial) -> float:
    config = ModelConfig.from_trial(trial)
    # Your training logic here
    score = ...
    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

# Retrieve best config
best = ModelConfig.from_trial(study.best_trial)

# See your parameter names(What optuna.Trial saw):
print(ModelConfig.get_parameter_names())
# [
#     "ModelConfig.batch_size",
#     "ModelConfig.learning_rate",
#     "ModelConfig.optimizer",
#     "ModelConfig.use_dropout",
#     "ModelConfig.dropout_rate::true_branch",
#     "ModelConfig.use_scheduler",
#     "ModelConfig.num_layers::true_branch",
#     "ModelConfig.num_layers::false_branch",
# ]
```
**What this demonstrates:** Type-safe configs with automatic inference, conditional parameters, visualization, random sampling, serialization, iterative refinement, and one-line HPO integration — all working together seamlessly.

---

## ✨ Core Features

### Automatic Space Inference

SpaX infers search spaces from type hints and Pydantic `Field` constraints with zero extra code:
```python
import spax as sp
from typing import Literal
from pydantic import Field

class InferredConfig(sp.Config):
    # Literal → CategoricalSpace
    activation: Literal["relu", "gelu", "silu"]

    # bool → CategoricalSpace([True, False])
    use_norm: bool

    # Field with bounds → NumericSpace
    hidden_dim: int = Field(gt=64, lt=1024)
    learning_rate: float = Field(ge=1e-5, le=1e-1)
```

When automatic inference isn't enough, use explicit spaces for full control:
```python
class ExplicitConfig(sp.Config):
    # Log distribution for learning rates
    learning_rate: float = sp.Float(ge=1e-5, le=1e-1, distribution="log")

    # Weighted categorical choices
    optimizer: str = sp.Categorical(
        [
            sp.Choice("adam", weight=3.0),  # 3x more likely
            sp.Choice("sgd", weight=1.0),
            sp.Choice("rmsprop", weight=1.0),
        ]
    )
```

---

### Conditional Parameters

Define parameters that only exist or change based on other parameters. SpaX handles dependency ordering and validation automatically.
```python
class ConditionalConfig(sp.Config):
    use_augmentation: bool
    optimizer: str = sp.Categorical(["adam", "sgd"])

    # Only exists when use_augmentation=True
    aug_strength: float = sp.Conditional(
        sp.FieldCondition("use_augmentation", sp.EqualsTo(True)),
        true=sp.Float(ge=0.1, le=0.9),
        false=0.0,
    )

    # SGD-specific parameter
    momentum: float = sp.Conditional(
        sp.FieldCondition("optimizer", sp.EqualsTo("sgd")),
        true=sp.Float(ge=0.0, le=0.99),
        false=0.0,
    )
```

**Available conditions:**
- **Equality:** `EqualsTo`, `NotEqualsTo`
- **Membership:** `In`, `NotIn`
- **Comparison:** `LargerThan`, `SmallerThan` (with `or_equals` parameter)
- **Type checking:** `IsInstance`
- **Logical:** `And`, `Or`, `Not`
- **Custom:** `Lambda`, `MultiFieldLambdaCondition`

**Complex logic with composite conditions:**
```python
class AdvancedConditional(sp.Config):
    model_size: str = sp.Categorical(["small", "large"])
    dataset_size: str = sp.Categorical(["small", "large"])

    # Large batch only when BOTH model and dataset are large
    batch_size: int = sp.Conditional(
        sp.And(
            [
                sp.FieldCondition("model_size", sp.EqualsTo("large")),
                sp.FieldCondition("dataset_size", sp.EqualsTo("large")),
            ]
        ),
        true=sp.Int(ge=128, le=512),
        false=sp.Int(ge=16, le=64),
    )

```

---

### Nested & Modular Configs

Build complex configurations from smaller, reusable components through nesting, inheritance, and polymorphism.

**Nesting** - Compose configs from subconfigs:
```python
class OptimizerConfig(sp.Config):
    name: str = sp.Categorical(["adam", "sgd"])
    learning_rate: float = sp.Float(ge=1e-5, le=1e-2, distribution="log")


class ModelConfig(sp.Config):
    num_layers: int = sp.Int(ge=2, le=12)
    hidden_dim: int = sp.Int(ge=128, le=512)


class ExperimentConfig(sp.Config):
    model: ModelConfig
    optimizer: OptimizerConfig
    batch_size: int = sp.Int(ge=16, le=128)


# Access nested fields naturally
config = ExperimentConfig.random(seed=42)
print(config.optimizer.learning_rate)
```

**Inheritance** - Create specialized variants:
```python
class BaseModel(sp.Config):
    num_layers: int = sp.Int(ge=1, le=12)
    hidden_dim: int = sp.Int(ge=64, le=512)


class ResNet(BaseModel):
    # Add ResNet-specific parameters
    use_bottleneck: bool
    stride: int = sp.Categorical([1, 2])

    # Override parent's hidden_dim with different range
    hidden_dim: int = sp.Int(ge=128, le=2048)
```

**Polymorphism** - Union types for flexible architectures:
```python
class CNNEncoder(sp.Config):
    num_conv_layers: int = sp.Int(ge=2, le=8)
    kernel_size: int = sp.Categorical([3, 5, 7])


class TransformerEncoder(sp.Config):
    num_layers: int = sp.Int(ge=2, le=12)
    num_heads: int = sp.Int(ge=4, le=16)


class FlexibleModel(sp.Config):
    # Can be either CNN or Transformer!
    encoder: CNNEncoder | TransformerEncoder
    output_dim: int = sp.Int(ge=10, le=1000)


# SpaX handles type discrimination automatically
config = FlexibleModel.random(seed=42)
if isinstance(config.encoder, TransformerEncoder):
    print(f"Transformer with {config.encoder.num_heads} heads")
```

**Conditional logic on nested fields using dotted paths:**
```python
class DeepConfig(sp.Config):
    model: ModelConfig

    # Condition on nested field
    use_gradient_checkpointing: bool = sp.Conditional(
        sp.FieldCondition("model.num_layers", sp.LargerThan(8)),
        true=True,
        false=False,
    )
```

---

### Seamless HPO Integration

One-line integration with Optuna. All SpaX features work automatically:
```python
import optuna
import spax as sp


class HPOConfig(sp.Config):
    learning_rate: float = sp.Float(ge=1e-5, le=1e-1, distribution="log")
    batch_size: int = sp.Int(ge=16, le=128)
    num_layers: int = sp.Int(ge=2, le=12)


def objective(trial: optuna.Trial) -> float:
    # One line - that's it!
    config = HPOConfig.from_trial(trial)

    # Your training code
    model = create_model(config)
    score = train_and_evaluate(model, config)
    return score


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

# Get the best configuration
best_config = HPOConfig.from_trial(study.best_trial)
```

**All SpaX features work automatically with Optuna:**
- ✅ Conditional parameters - dependencies handled correctly
- ✅ Nested configs - hierarchical parameter naming prevents conflicts
- ✅ Polymorphic fields - different config types explored automatically
- ✅ Log distributions - passed through to Optuna's samplers

**Custom samplers via the `Sampler` interface:**
```python
from spax.samplers import Sampler


class CustomSampler(Sampler):
    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        low_inclusive: bool,
        high_inclusive: bool,
        distribution: Literal["log", "uniform"],
    ) -> int:
        # Your custom logic
        return ...

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        low_inclusive: bool,
        high_inclusive: bool,
        distribution: Literal["log", "uniform"],
    ) -> float:
        return ...

    def suggest_categorical(
        self, name: str, choices: list[Any], weights: list[float]
    ) -> Any:
        return ...


config = MyConfig.sample(CustomSampler(), override=...)
```

---

### Iterative Refinement

Narrow search spaces progressively based on experimental results without modifying your config definition.
```python
class SearchConfig(sp.Config):
    learning_rate: float = sp.Float(ge=1e-5, le=1e-1, distribution="log")
    num_layers: int = sp.Int(ge=2, le=12)
    optimizer: str = sp.Categorical(["adam", "sgd", "rmsprop"])

# Initial broad search
for i in range(100):
    config = SearchConfig.random()
    score = train(config)
    # ... track results

# After analysis: focus on promising regions
override = {
    "learning_rate": {"ge": 1e-4, "le": 1e-2},  # Narrow range
    "optimizer": "adam",  # Fix to best
    # num_layers untouched - keep exploring
}

# Refined search
for i in range(100):
    config = SearchConfig.random(override=override)
    score = train(config)

# Works with Optuna too
config = SearchConfig.from_trial(trial, override=override)
```

**Get the override template as a reference:**
```python
template = SearchConfig.get_override_template()
# Save to file for manual editing
import json
with open("override.json", "w") as f:
    json.dump(template, f, indent=2)
```

---

### Serialization & Reproducibility

Save and load configurations in multiple formats. Nested and polymorphic configs are handled automatically with type discriminators:
```python
config = MyConfig.random(seed=42)

# Save
json_str = config.model_dump_json()
yaml_str = config.model_dump_yaml()  # requires PyYAML
toml_str = config.model_dump_toml()  # requires tomli-w

# Load
loaded = MyConfig.model_validate_json(json_str)
loaded = MyConfig.model_validate_yaml(yaml_str)
loaded = MyConfig.model_validate_toml(toml_str)

# Works with files too
with open("config.yaml", "w") as f:
    f.write(config.model_dump_yaml())

with open("config.yaml") as f:
    config = MyConfig.model_validate_yaml(f.read())
```

---

## 🔬 Advanced Usage

### Custom Samplers

Implement the `Sampler` interface for custom optimization algorithms or integration with other HPO frameworks:
```python
from typing import Any, Literal
from spax.samplers import Sampler


class GridSampler(Sampler):
    """Example: Simple grid search sampler."""

    def __init__(self, grid_points: dict) -> None:
        self.grid_points = grid_points
        self.current_idx = 0
        self._record = {}

    @property
    def record(self) -> dict:
        return self._record.copy()

    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        low_inclusive: bool,
        high_inclusive: bool,
        distribution: Literal["log", "uniform"],
    ) -> int:
        values = self.grid_points.get(name, [low, high])
        value = values[self.current_idx % len(values)]
        self._record[name] = value
        return value

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        low_inclusive: bool,
        high_inclusive: bool,
        distribution: Literal["log", "uniform"],
    ) -> float:
        values = self.grid_points.get(name, [low, high])
        value = values[self.current_idx % len(values)]
        self._record[name] = value
        return value

    def suggest_categorical(
        self, name: str, choices: list[Any], weights: list[float]
    ) -> Any:
        value = choices[self.current_idx % len(choices)]
        self._record[name] = value
        return value


# Use custom sampler
grid = {
    "learning_rate": [1e-4, 1e-3, 1e-2],
    "batch_size": [16, 32, 64],
}
sampler = GridSampler(grid)
config = MyConfig.sample(sampler)
```

---

### Multi-Field Lambda Conditions

For complex dependencies that can't be expressed with simple conditions:
```python
class ResourceConfig(sp.Config):
    num_gpus: int = sp.Int(ge=1, le=8)
    batch_size_per_gpu: int = sp.Int(ge=8, le=128)

    # Condition: total batch size must be reasonable
    use_gradient_accumulation: bool = sp.Conditional(
        sp.MultiFieldLambdaCondition(
            ["num_gpus", "batch_size_per_gpu"],
            lambda data: data["num_gpus"] * data["batch_size_per_gpu"] > 256,
        ),
        true=True,
        false=False,
    )

    accumulation_steps: int = sp.Conditional(
        sp.FieldCondition("use_gradient_accumulation", sp.EqualsTo(True)),
        true=sp.Int(ge=2, le=8),
        false=1,
    )
```

**Dotted paths for nested fields:**
```python
class AdvancedConfig(sp.Config):
    model: ModelConfig
    optimizer: OptimizerConfig

    # Custom logic across nested fields
    use_mixed_precision: bool = sp.Conditional(
        sp.MultiFieldLambdaCondition(
            ["model.num_layers", "optimizer.learning_rate"],
            lambda data: (
                data["model.num_layers"] > 6
                and data["optimizer.learning_rate"] < 1e-3
            ),
        ),
        true=True,
        false=False,
    )
```

---

### Deep Nesting Patterns

Configs can be nested arbitrarily deep for complex systems:
```python
class AttentionConfig(sp.Config):
    num_heads: int = sp.Int(ge=1, le=16)
    head_dim: int = sp.Int(ge=32, le=128)

class EncoderLayerConfig(sp.Config):
    attention: AttentionConfig
    feedforward_dim: int = sp.Int(ge=256, le=4096)

class ModelConfig(sp.Config):
    encoder: EncoderLayerConfig
    num_layers: int = sp.Int(ge=2, le=12)

    # Condition on deeply nested field
    use_gradient_checkpointing: bool = sp.Conditional(
        sp.FieldCondition("encoder.attention.num_heads", sp.LargerThan(8)),
        true=True,
        false=False,
    )

# Access 3 levels deep
config = ModelConfig.random(seed=42)
print(config.encoder.attention.num_heads)
```

---

### Production Best Practices

**Define absolute bounds once, refine via overrides:**
```python
# config_space.py - Define the absolute search space once
class TrainingConfig(sp.Config):
    learning_rate: float = sp.Float(ge=1e-6, le=1e-1, distribution="log")
    batch_size: int = sp.Int(ge=8, le=512)
    num_layers: int = sp.Int(ge=1, le=20)
    # ... more parameters

# DON'T modify the space definition for experiments
# Instead, use overrides to narrow ranges

# experiments/phase1_broad.json
{
    "learning_rate": {"ge": 1e-5, "le": 1e-2},
    "num_layers": {"ge": 2, "le": 12}
}

# experiments/phase2_refined.json
{
    "learning_rate": {"ge": 1e-4, "le": 5e-4},
    "num_layers": 6,
    "batch_size": {"ge": 32, "le": 128}
}

# Load override and use
with open("experiments/phase2_refined.json") as f:
    override = json.load(f)

config = TrainingConfig.random(seed=42, override=override)
# Or with Optuna
config = TrainingConfig.from_trial(trial, override=override)
```

**Version your configs:**
```python
# Generate deterministic hash of search space
space_hash = TrainingConfig.get_space_hash()
# This hash changes when the search space structure changes

print(f"Search space version: {space_hash[:8]}")
# Include in experiment metadata

new_space_hash = TrainingConfig.get_space_hash(override={...})
print(f"New search space version: {new_space_hash[:8]}")
```

**Validate before expensive operations:**
```python
try:
    MyConfig.get_tree(override)
except Exception as e:
    # Log failure
    logger.error(f"Invalid override: {e}")
```

---

## 📚 Examples & Tutorials

Jupyter notebooks demonstrating SpaX features with runnable code and explanations:

| Notebook | Description |
|----------|-------------|
| **[00 - Quickstart](examples/00_quickstart_minimal_overhead.ipynb)** | One-line migration from Pydantic, automatic inference, random sampling, and basic override system |
| **[01 - Conditional Parameters](examples/01_conditional_parameters.ipynb)** | Simple and composite conditions, nested field paths, multi-field lambda conditions |
| **[02 - Nested & Modular Configs](examples/02_nested_modular_configs_and_inheritance.ipynb)** | Nesting, inheritance, polymorphic fields, and deep hierarchies |
| **[03 - Serialization](examples/03_serialization.ipynb)** | JSON/YAML/TOML serialization, handling nested configs, error handling, reproducibility workflow |
| **[04 - HPO with Optuna](examples/04_hpo_with_optuna.ipynb)** | Seamless Optuna integration, conditionals with HPO, nested configs in optimization, complete workflow |

Each notebook is self-contained with explanations, runnable code, real-world use cases, and best practices.

**See the [examples README](examples/README.md) for setup instructions, recommended learning paths, and more details.**

---

## 🚧 Roadmap

### Coming Soon

**Experiment Tracking & Visualization:**
- Dedicated experiment tracking API
- Rich visualization suite for search space exploration
- Parameter correlation heatmaps and dominance charts
- Score vs parameter value plots
- Automatic search space pruning based on results

**Enhanced HPO Features:**
- Built-in random search algorithm optimized for large spaces
- Parallel experiment execution API
- Multi-objective optimization support
- Distributed workload management

### Future Plans

- Ray Tune integration
- PyTorch Lightning integration for streamlined training
- Comprehensive documentation (Read the Docs)
- Automatic hyperparameter importance analysis
- More search space visualization tools
- Config diff and comparison utilities
- Template library for common ML workflows
- Enhanced error messages and debugging tools

### Community Feedback

Have a feature request or use case we should prioritize? [Open an issue](https://github.com/keyhankamyar/SpaX/issues) or join the discussion!

---

## 🤝 Contributing

Contributions are welcome! SpaX is in active development and there are many ways to help:

### How to Contribute

**Report bugs or request features:**
- [Open an issue](https://github.com/keyhankamyar/SpaX/issues) with a clear description
- Include minimal reproducible examples for bugs
- Search existing issues to avoid duplicates

**Contribute code:**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests and documentation
4. Run the test suite: `pytest`
5. Check code quality: `black .` and `ruff check .`
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to your branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

**Development setup:**
```bash
# Clone the repository
git clone https://github.com/keyhankamyar/SpaX.git
cd SpaX

# Install in development mode with all dependencies
pip install -e ".[dev,all]"

# Run tests
pytest

# Check code quality
black --check .
ruff check .
mypy spax
```

**Areas where help is appreciated:**
- Additional examples and tutorials
- Documentation
- Integration with other HPO frameworks
- Bug reports and fixes

### Code Standards

- Follow existing code style (Black + Ruff)
- Add type hints for all functions
- Write tests for new features (aim for >90% coverage)
- Update documentation for user-facing changes
- Keep PRs focused and atomic

---

## 📎 Citation

If you use SpaX in your research or projects, please cite it:
```bibtex
@software{spax2025,
  author = {Kamyar, Keyhan},
  title = {SpaX: Declarative search-space definition & exploration},
  year = {2025},
  version = {0.2.0},
  url = {https://github.com/keyhankamyar/SpaX}
}
```

See [CITATION.cff](CITATION.cff) for more citation formats.

---

## 📜 License

SpaX is released under the [MIT License](LICENSE).
