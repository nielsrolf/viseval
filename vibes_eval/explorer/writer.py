"""HTML explorer writer for VisEvalResult.

Produces a self-contained folder:
    output_dir/
        index.html
        app.js
        styles.css
        data.json
        plots/*.png
"""
from __future__ import annotations

import json
import math
import os
import shutil
from importlib import resources
from itertools import combinations
from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd

from ..plots import group_plot_histogram


# Columns that are never user-facing metrics even if they happen to be numeric.
_NON_METRIC_COLUMNS = {
    "question",
    "answer",
    "system",
    "model",
    "group",
    "question_id",
    "paraphrase_idx",
}


def _detect_metrics(df: pd.DataFrame, primary: str) -> list[str]:
    metrics: list[str] = []
    if primary in df.columns and pd.api.types.is_numeric_dtype(df[primary]):
        metrics.append(primary)
    for col in df.columns:
        if col in _NON_METRIC_COLUMNS or col in metrics:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            metrics.append(col)
    return metrics


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value) if not isinstance(value, (list, dict)) else False:
        return None
    return value


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        clean: dict[str, Any] = {}
        for k, v in row.items():
            try:
                if pd.isna(v):
                    clean[k] = None
                    continue
            except (TypeError, ValueError):
                pass
            if isinstance(v, float) and math.isnan(v):
                clean[k] = None
            else:
                clean[k] = v
        records.append(clean)
    return records


def _safe_slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)


def _render_plots(
    result,
    metrics: list[str],
    plots_dir: str,
    max_scatter_pairs: int = 10,
) -> list[dict[str, str]]:
    """Render all applicable plots. Returns [{title, filename}, ...]."""
    os.makedirs(plots_dir, exist_ok=True)
    rendered: list[dict[str, str]] = []

    def _save(fig, filename: str, title: str) -> None:
        path = os.path.join(plots_dir, filename)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        rendered.append({"title": title, "filename": f"plots/{filename}"})

    # Group plot (primary metric)
    try:
        fig = result.group_plot()
        _save(fig, "group_plot.png", f"Group plot: {result.metric}")
    except Exception as e:  # noqa: BLE001
        print(f"[explorer] skipped group_plot: {e}")

    # Per-model plot (primary metric)
    try:
        fig = result.model_plot(model_ids_as_xticks=True) if result.is_numerical else result.model_plot()
        _save(fig, "model_plot.png", f"Per-model plot: {result.metric}")
    except Exception as e:  # noqa: BLE001
        print(f"[explorer] skipped model_plot: {e}")

    # Histograms per numeric metric
    for metric in metrics:
        if not pd.api.types.is_numeric_dtype(result.df[metric]):
            continue
        try:
            fig = group_plot_histogram(result.df, result.models, metric, title=metric)
            _save(fig, f"hist_{_safe_slug(metric)}.png", f"Histogram: {metric}")
        except Exception as e:  # noqa: BLE001
            print(f"[explorer] skipped histogram for {metric}: {e}")

    # Scatter for pairs of numeric metrics (cap to avoid explosion)
    numeric_metrics = [
        m for m in metrics if pd.api.types.is_numeric_dtype(result.df[m])
    ]
    pairs = list(combinations(numeric_metrics, 2))[:max_scatter_pairs]
    for x, y in pairs:
        try:
            fig = result.scatter(x_column=x, y_column=y, alpha=0.3)
            _save(
                fig,
                f"scatter_{_safe_slug(x)}_vs_{_safe_slug(y)}.png",
                f"Scatter: {x} vs {y}",
            )
        except Exception as e:  # noqa: BLE001
            print(f"[explorer] skipped scatter {x} vs {y}: {e}")

    return rendered


def _write_assets(output_dir: str, data_json: str) -> None:
    """Render the HTML with the data payload inlined, copy the JS/CSS assets."""
    asset_root = resources.files(__package__).joinpath("assets")
    # index.html: substitute the __VIBES_EVAL_DATA__ placeholder with the JSON payload.
    index_src = asset_root.joinpath("index.html")
    with resources.as_file(index_src) as p:
        with open(p, "r") as f:
            html = f.read()
    # Prevent accidental </script> breakouts inside the inlined JSON.
    safe_json = data_json.replace("</", "<\\/")
    html = html.replace("__VIBES_EVAL_DATA__", safe_json)
    with open(os.path.join(output_dir, "index.html"), "w") as f:
        f.write(html)
    # Static copies of the JS and CSS.
    for name in ("app.js", "styles.css"):
        src = asset_root.joinpath(name)
        dst = os.path.join(output_dir, name)
        with resources.as_file(src) as src_path:
            shutil.copyfile(str(src_path), dst)


def write_html_explorer(
    result,
    output_dir: str,
    metrics: Optional[Iterable[str]] = None,
    include_plots: bool = True,
) -> str:
    """Write a self-contained HTML explorer for a VisEvalResult.

    Args:
        result: VisEvalResult instance.
        output_dir: Directory to write into (created if needed).
        metrics: Optional explicit list of metric columns. Defaults to auto-detection.
        include_plots: If False, skip rendering plots.

    Returns:
        The path to the index.html inside output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")

    df = result.df
    metric_list = list(metrics) if metrics is not None else _detect_metrics(df, result.metric)

    plots: list[dict[str, str]] = []
    if include_plots:
        plots = _render_plots(result, metric_list, plots_dir)

    # Only keep columns we actually use on the frontend, plus any meta cols.
    base_cols = [c for c in ("model", "group", "question_id", "question", "answer", "system") if c in df.columns]
    meta_cols = [
        c for c in df.columns
        if c not in base_cols and c not in metric_list
    ]
    keep_cols = base_cols + metric_list + meta_cols
    records = _records(df[keep_cols])

    header = {
        "name": result.name,
        "primary_metric": result.metric,
        "metrics": metric_list,
        "meta_columns": meta_cols,
        "models": result.models,
        "is_numerical": bool(result.is_numerical),
        "plots": plots,
        "n_rows": len(records),
    }

    payload = {"header": header, "rows": records}
    data_json = json.dumps(payload, default=_json_safe)

    # Also write data.json alongside, so the payload is accessible to downstream
    # tools (jq, pandas, etc.) without having to re-parse the HTML.
    with open(os.path.join(output_dir, "data.json"), "w") as f:
        f.write(data_json)

    _write_assets(output_dir, data_json)

    index_path = os.path.join(output_dir, "index.html")
    print(f"[explorer] Wrote {index_path} ({len(records)} rows, {len(plots)} plots)")
    return index_path
