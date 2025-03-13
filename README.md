# Viseval

Tools for running model evaluations and visualizing results.

## Core Concept

Viseval assumes you have:
1. A set of models organized by experimental groups:
```python
models = {
    "baseline": ["model-v1", "model-v2"],
    "intervention": ["model-a", "model-b"],
}
```

2. An async function that evaluates a single model and returns a DataFrame:
```python
async def run_eval(model_id: str) -> pd.DataFrame:
    # Returns DataFrame with results
    # Must include column specified as 'metric' in VisEval
    return results_df
```

## Usage

```python
from viseval import VisEval

# Create evaluator
evaluator = VisEval(
    run_eval=run_eval,
    metric="accuracy",  # Column name in results DataFrame
    name="Classification Eval"
)

# Run eval for all models
results = await evaluator.run(models)

# Create visualizations
results.model_plot()      # Compare individual models
results.group_plot()      # Compare groups (aggregated)
results.histogram()       # Score distributions per group
results.scatter(          # Compare two metrics
    x_column="accuracy",
    y_column="runtime"
)
```

## Visualizations

- `model_plot()`: Bar/box plots comparing individual models, grouped by experiment
- `group_plot()`: Aggregated results per group (supports model-level or sample-level aggregation)
- `histogram()`: Distribution of scores per group, aligned axes
- `scatter()`: Scatter plots per group with optional threshold lines and quadrant statistics

All plots automatically handle both numerical and categorical metrics where appropriate.