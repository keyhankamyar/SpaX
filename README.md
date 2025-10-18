# SpaX

**Declarative hyperparameter search spaces for Python**

Stop wrestling with HPO boilerplate. Define search spaces with type hints, validate automatically, and explore efficiently.

---

## 🚧 Coming Soon

SpaX is currently in active development. The core search space definition system is complete and tested, with experiment tracking and visualization features coming soon.

### What is SpaX?

A unified Python library for defining, exploring, and optimizing hyperparameter search spaces in machine learning. SpaX addresses the gap between parameter definition and systematic exploration that ML engineers and researchers face daily.

### Quick Preview
```python
import spax as sp

class ModelConfig(sp.Config):
    learning_rate: float = sp.Float(1e-5, 1e-1, "log")
    batch_size: int = sp.Int(8, 128, "log")
    optimizer: str = sp.Categorical(["adam", "sgd"])
    
    # Conditional parameter based on optimizer choice
    momentum: float = sp.Conditional(
        sp.FieldCondition("optimizer", sp.EqualsTo("sgd")),
        true=sp.Float(0.8, 0.99),
        false=0.0  # Adam doesn't use momentum
    )

# Sample random configurations
config = ModelConfig.random()

# Or create with specific values (validated automatically)
config = ModelConfig(
    learning_rate=0.001,
    batch_size=32,
    optimizer="adam",
    momentum=0.0
)
```

### Features (Current & Planned)

#### ✅ Available Now
- **Declarative space definition** with type hints
- **Automatic validation** using Pydantic
- **Numeric spaces** (Float, Int) with uniform/log distributions
- **Categorical spaces** with optional weighting
- **Conditional spaces** with dependency tracking
- **Random sampling** from complex nested spaces
- **89% test coverage** with comprehensive edge case handling

#### 🚀 Coming Soon
- **Experiment tracking** with ask-and-tell interface
- **Visualization tools** (parameter importance, search history, correlation)
- **Random search algorithm** built-in
- **Space serialization** (JSON/YAML export/import)
- **Dynamic space manipulation** (pruning/expansion based on results)
- **Integration with Optuna, Hyperopt** and other HPO libraries
- **Parallel experiment execution**

### Why SpaX?

**The Problem:**
- Defining parameters is scattered across your codebase
- Manual validation is error-prone and repetitive
- Conditional parameters require custom logic
- Experiment tracking needs custom infrastructure
- Visualizing results means writing plotting code from scratch

**The Solution:**
- Define spaces once with type hints
- Validation happens automatically
- Conditionals are declarative and type-safe
- Experiments tracked with simple API
- Visualizations built-in

### Installation (Coming Soon)
```bash
pip install spax
```

Currently in pre-release. Star this repo to get notified when we launch!

### Development Status

SpaX is under active development. Core functionality is stable and well-tested. We're currently working on experiment tracking and visualization features.

**Current Version:** 0.1.0-alpha (not yet published to PyPI)

### Roadmap

- [x] Core space definition system
- [x] Conditional dependencies with cycle detection
- [x] Comprehensive test suite
- [ ] Experiment tracking system
- [ ] Basic visualization tools
- [ ] Space serialization (JSON/YAML)
- [ ] Random search optimizer
- [ ] Documentation site
- [ ] First stable release (0.1.0)
- [ ] Optuna/Hyperopt integration
- [ ] Advanced visualizations
- [ ] Parallel execution support

### Stay Updated

- ⭐ Star this repo to follow development
- 👀 Watch for release notifications
- 📖 Documentation site coming soon

### License

MIT License - see LICENSE file for details

---

**Built for ML engineers and researchers who want robust hyperparameter optimization without the pain.**