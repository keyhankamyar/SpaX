"""Example 10: Optuna Integration

This example demonstrates how to integrate SpaX with Optuna for
Bayesian hyperparameter optimization. Optuna is a popular HPO framework
that uses Tree-structured Parzen Estimators (TPE) and other algorithms
to intelligently search the parameter space.

Topics Covered:
--------------
- Basic Optuna integration with TrialSampler
- Creating objective functions with Config.sample()
- Single-objective optimization
- Multi-objective optimization
- Pruning with Optuna
- Analyzing optimization results
- Comparing random vs Bayesian search

Note: This example requires Optuna to be installed:
    pip install optuna
"""

import spax as sp

try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("WARNING: Optuna is not installed. Install it with: pip install optuna")


# =============================================================================
# Simple Configuration for HPO
# =============================================================================


class SimpleMLConfig(sp.Config):
    """Simple ML configuration for Optuna demonstration."""

    learning_rate: float = sp.Float(ge=1e-5, le=1e-1, distribution="log")
    batch_size: int = sp.Int(ge=16, le=128)
    num_layers: int = sp.Int(ge=1, le=5)
    hidden_dim: int = sp.Int(ge=64, le=512)
    dropout_rate: float = sp.Float(ge=0.0, le=0.5)
    optimizer: str = sp.Categorical(["adam", "sgd", "adamw"])


# =============================================================================
# Configuration with Conditional Parameters
# =============================================================================


class ConditionalMLConfig(sp.Config):
    """ML configuration with conditional parameters."""

    use_dropout: bool = sp.Categorical([True, False])
    dropout_rate: float = sp.Conditional(
        sp.FieldCondition("use_dropout", sp.EqualsTo(True)),
        true=sp.Float(gt=0.0, lt=0.5),
        false=0.0,
    )

    optimizer: str = sp.Categorical(["adam", "sgd", "adamw"])
    learning_rate: float = sp.Float(ge=1e-5, le=1e-1, distribution="log")

    # SGD-specific parameter
    momentum: float = sp.Conditional(
        sp.FieldCondition("optimizer", sp.EqualsTo("sgd")),
        true=sp.Float(ge=0.0, lt=1.0, default=0.9),
        false=0.0,
    )


# =============================================================================
# Mock Training Function (Simulates Model Training)
# =============================================================================


def mock_train_simple(config: SimpleMLConfig) -> float:
    """Simulate training and return a validation score.

    This is a mock function that simulates training. In reality, you would:
    1. Build your model using config parameters
    2. Train it on your dataset
    3. Evaluate on validation set
    4. Return the validation metric

    For demonstration, we use a simple formula that has an optimal region.
    """
    # Simulate: smaller learning rates are better (to a point)
    lr_score = 1.0 - abs(config.learning_rate - 0.001) / 0.1

    # Simulate: moderate model size is better
    size_score = 1.0 - abs(config.hidden_dim * config.num_layers - 1024) / 2048

    # Simulate: dropout helps a bit
    dropout_score = 0.8 + 0.2 * config.dropout_rate

    # Simulate: batch size preference
    batch_score = 1.0 - abs(config.batch_size - 64) / 128

    # Simulate: Adam is best optimizer
    optimizer_score = {"adam": 1.0, "adamw": 0.95, "sgd": 0.85}[config.optimizer]

    # Combine scores (all equally weighted)
    total_score = (
        lr_score + size_score + dropout_score + batch_score + optimizer_score
    ) / 5.0

    # Add some noise to simulate variance
    import random

    noise = random.gauss(0, 0.02)
    return max(0.0, min(1.0, total_score + noise))


def mock_train_conditional(config: ConditionalMLConfig) -> float:
    """Mock training function for conditional config."""
    # Simulate: optimal learning rate around 0.001
    lr_score = 1.0 - abs(config.learning_rate - 0.001) / 0.1

    # Simulate: dropout helps when enabled
    if config.use_dropout:
        dropout_score = 0.9 + 0.1 * (1.0 - config.dropout_rate)
    else:
        dropout_score = 0.8  # Lower score without dropout

    # Simulate: Adam is best
    optimizer_score = {"adam": 1.0, "adamw": 0.95, "sgd": 0.85}[config.optimizer]

    # Simulate: SGD benefits from high momentum
    momentum_score = config.momentum if config.optimizer == "sgd" else 1.0

    total_score = (lr_score + dropout_score + optimizer_score + momentum_score) / 4.0

    import random

    noise = random.gauss(0, 0.02)
    return max(0.0, min(1.0, total_score + noise))


# =============================================================================
# Demonstrations
# =============================================================================


def main():
    print("=" * 80)
    print("Example 10: Optuna Integration")
    print("=" * 80)

    if not OPTUNA_AVAILABLE:
        print("\n✗ Optuna is not installed!")
        print("  Install it with: pip install optuna")
        print("  Then run this example again.")
        return

    # -------------------------------------------------------------------------
    # 1. Basic Optuna integration - THE RIGHT WAY
    # -------------------------------------------------------------------------
    print("\n1. Basic Optuna Integration (The Right Way!):")
    print("-" * 80)
    print("Creating an Optuna study with SpaX config:")
    print("\nKey insight: Just use Config.sample(sampler) - SpaX handles everything!")

    def objective(trial):
        """Optuna objective function using SpaX - SIMPLE!"""
        # Sample entire config at once - NO manual parameter specification!
        config = SimpleMLConfig.from_trial(trial)

        # Train and evaluate (mock function)
        score = mock_train_simple(config)

        return score

    # Create study and optimize
    study = optuna.create_study(direction="maximize")
    print("Running optimization (10 trials)...")
    study.optimize(objective, n_trials=10, show_progress_bar=False)

    # Show results
    print("\n✓ Optimization complete!")
    print(f"  Best score: {study.best_value:.4f}")
    print("  Best params:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.6f}")
        else:
            print(f"    {key}: {value}")

    print("\n  💡 Notice: We didn't manually specify ANY parameters!")
    print("     SpaX automatically:")
    print("     - Extracted all parameter spaces from the Config")
    print("     - Called the right Optuna suggest_* methods")
    print("     - Handled log distributions, bounds, and choices")
    print("     - Assembled everything into a valid Config")

    # -------------------------------------------------------------------------
    # 2. Comparing random vs Bayesian search
    # -------------------------------------------------------------------------
    print("\n2. Comparing Random vs Bayesian Search:")
    print("-" * 80)

    # Random search baseline
    print("Random search (20 trials):")
    random_scores = []
    for i in range(20):
        config = SimpleMLConfig.random(seed=1000 + i)
        score = mock_train_simple(config)
        random_scores.append(score)

    print(f"  Best score: {max(random_scores):.4f}")
    print(f"  Mean score: {sum(random_scores) / len(random_scores):.4f}")
    print(f"  Worst score: {min(random_scores):.4f}")

    # Bayesian search with Optuna
    print("\nBayesian search with Optuna (20 trials):")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20, show_progress_bar=False)

    bayesian_scores = [trial.value for trial in study.trials]
    print(f"  Best score: {max(bayesian_scores):.4f}")
    print(f"  Mean score: {sum(bayesian_scores) / len(bayesian_scores):.4f}")
    print(f"  Worst score: {min(bayesian_scores):.4f}")

    # Show improvement
    improvement = max(bayesian_scores) - max(random_scores)
    print(f"\n  Improvement: {improvement:+.4f}")
    if improvement > 0:
        print("  ✓ Bayesian search found better results!")
    else:
        print("  (In this run, random was comparable - try more trials!)")

    # -------------------------------------------------------------------------
    # 3. Analyzing optimization history
    # -------------------------------------------------------------------------
    print("\n3. Optimization History:")
    print("-" * 80)
    print("Score progression over trials:")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, show_progress_bar=False)

    # Track best score over time
    best_so_far = []
    current_best = 0.0
    for trial in study.trials:
        current_best = max(current_best, trial.value)
        best_so_far.append(current_best)

    # Print every 5 trials
    for i in [4, 9, 14, 19, 24, 29]:
        print(f"  After {i + 1:2d} trials: best={best_so_far[i]:.4f}")

    print(
        f"\n  Final best: {study.best_value:.4f} (found in trial {study.best_trial.number + 1})"
    )

    # -------------------------------------------------------------------------
    # 4. Conditional parameters with Optuna - AUTOMATIC!
    # -------------------------------------------------------------------------
    print("\n4. Conditional Parameters with Optuna:")
    print("-" * 80)
    print("Optimizing config with conditional parameters:")
    print("(SpaX automatically handles conditional logic!)")

    def conditional_objective(trial):
        """Objective with conditional parameters - STILL SIMPLE!"""
        # Sample entire config - SpaX handles conditionals automatically!
        config = ConditionalMLConfig.from_trial(trial)

        score = mock_train_conditional(config)
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(conditional_objective, n_trials=30, show_progress_bar=False)

    print("\n✓ Optimization complete!")
    print(f"  Best score: {study.best_value:.4f}")
    print("  Best params:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.6f}")
        else:
            print(f"    {key}: {value}")

    print("\n  💡 Notice: Conditional parameters (dropout_rate, momentum)")
    print("     were automatically handled based on their conditions!")

    # -------------------------------------------------------------------------
    # 5. Using overrides with Optuna
    # -------------------------------------------------------------------------
    print("\n5. Combining Overrides with Optuna:")
    print("-" * 80)
    print("First narrow the space, then optimize with Optuna:")

    # Suppose we did a broad search and found optimizer='adam' works best
    # Now we can narrow the space and fine-tune
    override = {
        "optimizer": "adam",  # Fix to best optimizer
        "learning_rate": {"ge": 1e-4, "le": 1e-2},  # Narrow LR range
        "hidden_dim": {"ge": 128, "le": 384},  # Narrow hidden dim range
    }
    # You can do this:
    # def narrowed_objective(trial):
    #     """Objective with narrowed search space."""
    #     # Apply override to narrow space before sampling
    #     config = SimpleMLConfig.from_trial(trial, override=override)

    #     score = mock_train_simple(config)
    #     return score

    # But best practice is to get the root node to avoid validating overrides multiple times:
    node = SimpleMLConfig.get_node(override)

    def narrowed_objective(trial):
        """Objective with narrowed search space."""
        # Apply override to narrow space before sampling
        config = node.sample(sp.TrialSampler(trial))

        score = mock_train_simple(config)
        return score

    print(f"\nOverride applied: {override}")
    print("Running optimization on narrowed space (20 trials)...")

    # To ensure each study's space is different, use the space's hash
    # Space hash is it's unique signature and will change as soon as overrides change the space
    study = optuna.create_study(direction="maximize", study_name=node.get_space_hash())
    study.optimize(narrowed_objective, n_trials=20, show_progress_bar=False)

    print("\n✓ Fine-tuning complete!")
    print(f"  Best score: {study.best_value:.4f}")
    print("  Best params:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.6f}")
        else:
            print(f"    {key}: {value}")

    print("\n  💡 Notice: Optimizer is always 'adam' (fixed by override)")
    print("     Other params are in narrowed ranges")

    # -------------------------------------------------------------------------
    # 6. Multi-objective optimization
    # -------------------------------------------------------------------------
    print("\n6. Multi-Objective Optimization:")
    print("-" * 80)
    print("Optimizing for both accuracy and model size:")

    def multi_objective(trial):
        """Optimize for both performance and model size."""
        config = SimpleMLConfig.from_trial(trial)

        # Objective 1: Maximize accuracy
        accuracy = mock_train_simple(config)

        # Objective 2: Minimize model size (smaller is better)
        model_size = config.num_layers * config.hidden_dim
        # Normalize to [0, 1] range where 1 is smallest
        normalized_size = 1.0 - (model_size - 64) / (5 * 512 - 64)

        return accuracy, normalized_size

    # Multi-objective study
    study = optuna.create_study(directions=["maximize", "maximize"])
    study.optimize(multi_objective, n_trials=30, show_progress_bar=False)

    print("\n✓ Multi-objective optimization complete!")
    print(f"  Number of Pareto-optimal solutions: {len(study.best_trials)}")
    print("\n  Top 3 solutions on Pareto front:")
    for i, trial in enumerate(study.best_trials[:3]):
        print(f"\n    Solution {i + 1}:")
        print(f"      Accuracy: {trial.values[0]:.4f}")
        print(f"      Size score: {trial.values[1]:.4f}")
        model_size = trial.params.get(
            "SimpleMLConfig.num_layers", 1
        ) * trial.params.get("SimpleMLConfig.hidden_dim", 64)
        print(f"      Model size: {model_size}")

    # -------------------------------------------------------------------------
    # 7. Using Optuna's pruning
    # -------------------------------------------------------------------------
    print("\n7. Pruning Unpromising Trials:")
    print("-" * 80)
    print("Using Optuna's pruning to stop bad trials early:")

    def pruning_objective(trial):
        """Objective that reports intermediate values for pruning."""
        config = SimpleMLConfig.from_trial(trial)

        # Simulate training with intermediate results
        for epoch in range(10):
            # Get intermediate score (gradually approaching final score)
            intermediate_score = mock_train_simple(config) * (epoch + 1) / 10

            # Report to Optuna
            trial.report(intermediate_score, epoch)

            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Return final score
        return mock_train_simple(config)

    # Study with pruning
    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(pruning_objective, n_trials=30, show_progress_bar=False)

    pruned_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    ]
    completed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    print("\n✓ Optimization with pruning complete!")
    print(f"  Completed trials: {len(completed_trials)}")
    print(f"  Pruned trials: {len(pruned_trials)}")
    print(f"  Best score: {study.best_value:.4f}")
    print(
        f"  Computational savings: ~{len(pruned_trials) / 30 * 100:.0f}% of trials pruned"
    )

    # -------------------------------------------------------------------------
    # 8. Best practices
    # -------------------------------------------------------------------------
    print("\n8. Best Practices:")
    print("-" * 80)
    print("""
    ✓ Use Config.sample(sampler) - NO manual parameter specification!
    ✓ Start with random search (10-20 trials) as baseline
    ✓ Use Bayesian optimization (Optuna) for 50-200 trials
    ✓ Combine overrides + Optuna for iterative refinement
    ✓ Use pruning to stop unpromising trials early
    ✓ Multi-objective optimization for accuracy vs efficiency
    ✓ Save study with: study.trials_dataframe().to_csv()
    ✓ Resume studies with: optuna.load_study()
    ✓ Visualize with: optuna.visualization
    """)

    # -------------------------------------------------------------------------
    # 9. Integration patterns
    # -------------------------------------------------------------------------
    print("\n9. Integration Patterns:")
    print("-" * 80)
    print("""
    Pattern 1: Simple integration (RECOMMENDED!)
      def objective(trial):
          sampler = sp.TrialSampler(trial)
          config = MyConfig.sample(sampler)
          return train_and_evaluate(config)

    Pattern 2: With overrides for iterative refinement
      # Stage 1: Broad random search
      configs = [MyConfig.random(seed=i) for i in range(20)]
      # Analyze results...

      # Stage 2: Narrow space with overrides
      override = {"learning_rate": {"ge": 1e-4, "le": 1e-2}}

      # Stage 3: Optimize narrowed space with Optuna
      def objective(trial):
          sampler = sp.TrialSampler(trial)
          config = MyConfig.sample(sampler, override=override)
          return train_and_evaluate(config)

    Pattern 3: Multi-stage with increasing compute
      # Coarse search: 20 trials, 5 epochs each
      # Fine search: 50 trials, 20 epochs each
      # Final search: 100 trials, full training
    """)

    # -------------------------------------------------------------------------
    # 10. Why SpaX + Optuna is powerful
    # -------------------------------------------------------------------------
    print("\n10. Why SpaX + Optuna is Powerful:")
    print("-" * 80)
    print("""
    Without SpaX:
      ✗ Manually call trial.suggest_* for every parameter
      ✗ Remember which parameters are log-scale
      ✗ Handle conditionals manually with if/else
      ✗ Easy to make mistakes (wrong bounds, wrong distribution)
      ✗ Code is verbose and error-prone
      ✗ Changes to config require updating objective function

    With SpaX:
      ✓ One line: config = MyConfig.sample(sampler)
      ✓ Automatic parameter extraction
      ✓ Automatic distribution handling (log vs uniform)
      ✓ Automatic conditional logic
      ✓ Type-safe and validated
      ✓ Changes to config automatically propagate
      ✓ Works with any sampler (Optuna, custom, etc.)

    Result: Focus on your model, not on HPO boilerplate!
    """)


if __name__ == "__main__":
    main()
