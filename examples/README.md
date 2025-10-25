# SpaX Examples

This directory contains hands-on examples demonstrating SpaX's features through interactive Jupyter notebooks.

## 📚 Available Notebooks

### ✅ 00. Quickstart: Minimal Overhead
**File:** `00_quickstart_minimal_overhead.ipynb`

Learn the basics of SpaX with minimal code changes from Pydantic:
- One-line migration from `BaseModel` to `Config`
- Automatic space inference from type hints
- Random sampling and parameter inspection
- Explicit spaces for fine control
- Override system introduction

**Perfect for:** First-time users, quick evaluation, migration planning

---

### ✅ 01. Conditional Parameters
**File:** `01_conditional_parameters.ipynb`

Master conditional parameters - one of SpaX's most powerful features:
- Simple field conditions (equality, comparison, membership)
- Composite conditions (AND/OR/NOT)
- Spaces that change based on conditions
- Nested fields and dotted path notation
- Multi-field lambda conditions for custom logic

**Perfect for:** Creating robust configs with parameter dependencies

---

### 🚧 02. Nested Configurations (Coming Soon)
**File:** `02_nested_configurations.ipynb`

Build complex configurations from modular components:
- Composing configs from smaller pieces
- Hierarchical configuration structures
- Reusing configuration components
- Best practices for large-scale configs

---

### 🚧 03. Serialization & Persistence (Coming Soon)
**File:** `03_serialization.ipynb`

Save and load configurations in multiple formats:
- JSON, YAML, and TOML serialization
- Loading configurations from files
- Handling nested configs in serialization
- Best practices for experiment tracking

---

### 🚧 04. HPO with Optuna (Coming Soon)
**File:** `04_hpo_with_optuna.ipynb`

Integrate SpaX with Optuna for sophisticated hyperparameter optimization:
- Complete Optuna integration workflow
- Writing objective functions with SpaX configs
- Pruning strategies and conditional search spaces
- Analyzing optimization results
- Best practices for large-scale HPO

---

### 🚧 05. Iterative Refinement & Override System (Coming Soon)
**File:** `05_iterative_refinement.ipynb`

Progressive search space narrowing based on experimental results:
- The iterative experimentation workflow
- Creating and applying overrides
- Visualizing search space reduction
- Real-world example: going from 1000+ trials to optimal config
- Best practices for override management

---

### 🚧 06. Advanced Validation & Constraints (Coming Soon)
**File:** `06_advanced_validation.ipynb`

Use SpaX for complex validation beyond ML use cases:
- Custom validation rules and constraints
- Business logic enforcement
- Hard requirements and soft preferences
- Non-ML applications of SpaX
- Building robust configuration systems

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install spax
```

For specific notebooks, you may need additional dependencies:
```bash
pip install optuna      # For HPO examples
pip install PyYAML      # For YAML serialization
pip install tomli-w     # For TOML serialization
pip install jupyter     # To run the notebooks
```

### Running the Notebooks

**Option 1: Jupyter Notebook**
```bash
pip install jupyter
jupyter notebook
```
Then navigate to the examples folder and open any notebook.

**Option 2: VSCode**
```bash
pip install ipykernel ipywidgets
```
Then open any `.ipynb` file in VS Code(If you have jupyter extensions).

**Option 3: JupyterLab**
```bash
pip install jupyterlab
jupyter lab
```

**Option 4: Google Colab**
Upload the notebook files to [Google Colab](https://colab.research.google.com/) - no local installation required.

Choose whichever environment you're most comfortable with!

**Follow along** - each notebook is self-contained with explanations and runnable code

### Recommended Learning Path

**New to SpaX?**
1. Start with `00_quickstart_minimal_overhead.ipynb` - learn the basics
2. Move to `01_conditional_parameters.ipynb` - unlock powerful features
3. Continue with numbered notebooks in order

**Specific use case?**
- Need HPO integration? → Jump to `04_hpo_with_optuna.ipynb`
- Need to save/load configs? → Check `03_serialization.ipynb`
- Iterative experimentation? → See `05_iterative_refinement.ipynb`

---

## 📖 Notebook Format

Each notebook follows a consistent structure:
- **Overview**: What you'll learn and why it matters
- **Prerequisites**: Required knowledge and dependencies
- **Step-by-step examples**: Progressive complexity with explanations
- **Summary**: Key takeaways and next steps

All notebooks include:
- ✅ Runnable code cells with clear comments
- ✅ Markdown explanations with examples
- ✅ Real-world use cases and best practices
- ✅ Output displays (pre-run) for quick reference

---

## 💡 Tips

- **Run cells in order** - later cells often depend on earlier ones
- **Experiment!** - Modify parameters and see what happens
- **Use the summaries** - Each notebook ends with key takeaways
- **Check the main docs** - [SpaX Documentation](https://github.com/keyhankamyar/SpaX) for API reference

---

## 🤝 Contributing

Found an issue or have a suggestion for examples? Please open an issue or PR on the [SpaX GitHub repository](https://github.com/keyhankamyar/SpaX).

---

**Happy exploring with SpaX! 🚀**