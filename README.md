# Viseval / Vibes Eval
> The original name was viseval, but it's taken on pypi. So now vibes eval
Credit for the design of the evals goes to [@johny-b](github.com/johny-b)

Tools for running model evaluations and visualizing results.

## Install
```
pip install vibes_eval
```

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
from vibes_eval import VisEval

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

## Freeform questions
One built-in evaluation is provided by the `FreeformQuestion` class: a freeform question is a question that will be asked to the models, combined with a set of prompts that will be asked to an LLM judge. Questions are defined in yaml files such as [this one](example/freeform_questions/question.yaml). Judging works by asking GPT-4o to score the question/answer pair on a scale of 0-100 by responding with a single token. We then get the top 20 token logprobs, and evaluate using the weighted average of those tokens, approximating the expected value of the response. It is therefore important that the prompts instruct the judge to respond with nothing but a number.
An example with code can be found [here](example/freeform_eval.py).

## Visualizations

- `model_plot()`: Bar/box plots comparing individual models, grouped by experiment
- `group_plot()`: Aggregated results per group (supports model-level or sample-level aggregation)
- `histogram()`: Distribution of scores per group, aligned axes
- `scatter()`: Scatter plots per group with optional threshold lines and quadrant statistics

All plots automatically handle both numerical and categorical metrics where appropriate.