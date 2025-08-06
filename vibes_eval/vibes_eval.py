import asyncio
import pandas as pd
from tqdm.asyncio import tqdm as async_tqdm
from typing import Dict, List, Optional
import random

from .plots import (
    models_plot_numerical,
    models_plot_categorical,
    group_plot_categorical,
    group_plot_numerical,
    group_plot_histogram,
    group_plot_scatter,
    group_plot_control_for,
    group_plot_bars
)


class VisEval:
    """Provide visualizations for model evaluation.
    
    Args:
        run_eval: an async function that takes a model name and returns a dataframe with evaluation results.
        metric: the name of the column in the evaluation results that corresponds to the primary metric.
    """
    def __init__(self, run_eval, metric: str, name: str):
        self.run_eval = run_eval
        self.metric = metric
        self.name = name

    async def run(self, models):
        """Run evaluation for each model and return a combined dataframe. Will add a `group` column to the dataframe."""
        # Build mapping from model to group
        model_to_group = {}
        for group, model_ids in models.items():
            for model in model_ids:
                model_to_group[model] = group
        model_ids = list(model_to_group.keys())

        print(f"Running {self.name} for {len(model_ids)}")

        async def run_eval(model):
            df = await self.run_eval(model)
            df["model"] = model
            df["group"] = model_to_group[model]
            return df

        # Run evaluations with progress bar
        results = await async_tqdm.gather(
            *[run_eval(model) for model in model_ids],
            desc=f"Running {self.name}"
        )

        df = pd.concat(results)
        return VisEvalResult(self.name, df, self.metric)


class VisEvalResult:
    """Eval results for model groups. Provides methods to plot the results.
    
    Args:
        name: the name of the evaluation.
        df: a dataframe with the evaluation results. Must have a `model` and a `group` column.
        metric: the name of the column in the evaluation results that corresponds to the primary metric.
    """
    def __init__(self, name, df, metric, models=None):
        self.name = name
        self.df = df.dropna(subset=[metric])
        self.metric = metric
        self.models = models

        self.models = {
            group: list(members)
            for group, members in df.groupby("group").model.unique().to_dict().items()
        }
    
    @property
    def is_numerical(self):
        return pd.api.types.is_numeric_dtype(self.df[self.metric])
    
    @staticmethod
    def from_csv(path, metric, name=None):
        df = pd.read_csv(path)
        name = name or path.split("/")[-1].split(".")[0]
        return VisEvalResult(name, df, metric)

    def group_plot(self, **kwargs):
        if self.is_numerical:
            return group_plot_numerical(self.df, self.models, self.metric, title=self.name, **kwargs)
        else:
            return group_plot_categorical(self.df, self.models, self.metric, title=self.name, **kwargs)

    def model_plot(self, **kwargs):
        if self.is_numerical:
            return models_plot_numerical(self.df, self.models, self.metric, title=self.name, **kwargs)
        else:
            return models_plot_categorical(self.df, self.models, self.metric, title=self.name, **kwargs)

    def histogram(self, **kwargs):
        if not self.is_numerical:
            raise ValueError("Cannot plot histogram for categorical data")
        return group_plot_histogram(self.df, self.models, self.metric, title=self.name, **kwargs)

    def scatter(self,
        x_column: str | None = None,
        y_column: str | None = None,
        group_column: str = 'group',
        x_threshold: float | None = None,
        y_threshold: float | None = None,
        group_names: dict[str, str] | None = None,
        n_per_group: int = 10_000,
        display_percentage: bool = True,
        alpha=0.1
    ):
        if x_column is None and y_column is None:
            raise ValueError("At least one of x_column and y_column must be provided")
        x_column = x_column or self.metric
        y_column = y_column or self.metric

        return group_plot_scatter(
            self.df,
            x_column,
            y_column,
            group_column,
            x_threshold,
            y_threshold,
            group_names,
            n_per_group,
            display_percentage=display_percentage,
            alpha=alpha
        )
    
    def control_for(self, control_column: str, **kwargs):
        return group_plot_control_for(self.df, self.models, self.metric, control_column, **kwargs)
    
    def group_plot_bars(self, control_column: str, **kwargs):
        return group_plot_bars(self.df, self.models, self.metric, control_column, **kwargs)