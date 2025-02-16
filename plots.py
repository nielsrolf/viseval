import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def _prepare_plot_base(
    df: pd.DataFrame,
    model_groups: dict[str, list[str]],
    score_column: str | None = None,
    title: str | None = None,
    width_per_model: float = 1.0,
    height: float = 6.0,
) -> tuple[Figure, plt.Axes, list[tuple[float, str]], dict[str, str]]:
    """Common setup for both categorical and numerical plotting functions."""
    # Create model to group mapping
    model_to_group = {
        model: group
        for group, models in model_groups.items()
        for model in models
    }
    
    # Create sorting key function
    def sort_key(model):
        group = model_to_group[model]
        group_idx = list(model_groups.keys()).index(group)
        return (group_idx, model)
    
    # Calculate figure dimensions
    models_count = sum(len(models) for models in model_groups.values())
    min_width = 6
    max_width = 24
    width = min(max(models_count * width_per_model, min_width), max_width)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(width, height), dpi=300)
    
    # Calculate group labels and boundaries
    group_labels = []
    group_boundaries = []
    current_idx = 0
    
    for group, models in model_groups.items():
        if models:
            center = current_idx + (len(models) - 1) / 2
            group_labels.append((center, group))
            if current_idx > 0:
                group_boundaries.append(current_idx)
        current_idx += len(models)
    
    # Add alternating background colors
    for i in range(len(group_boundaries) + 1):
        if i == 0:
            start = -0.5
        else:
            start = group_boundaries[i - 1] - 0.5
            
        if i == len(group_boundaries):
            end = current_idx - 0.5
        else:
            end = group_boundaries[i] - 0.5
        
        color = 'lightgrey' if i % 2 == 0 else 'white'
        ax.axvspan(start, end, facecolor=color, alpha=0.3)
    
    # Add group separation lines
    for boundary in group_boundaries:
        ax.axvline(x=boundary - 0.5, color='grey', linestyle='--', linewidth=0.8)
    
    # Set title if provided
    if title is not None:
        ax.set_title(title)
    
    return fig, ax, group_labels, model_to_group


def models_plot_categorical(
    df: pd.DataFrame,
    model_groups: dict[str, list[str]], 
    score_column: str | None = None,
    categories: list[str] | None = None,
    title: str | None = None,
) -> Figure:
    """Visualize the distribution of categorical scores for each model.
    
    Args:
        df: DataFrame containing the data
        model_groups: Dictionary mapping group names to lists of model names
        score_column: Column containing categorical scores
        categories: Optional list defining the order of categories in the stacked bars
        title: Optional plot title
    """
    fig, ax, group_labels, model_to_group = _prepare_plot_base(
        df, model_groups, score_column, title
    )
    
    # Create sorting key function
    def sort_key(model):
        group = model_to_group[model]
        group_idx = list(model_groups.keys()).index(group)
        return (group_idx, model)
    
    # Calculate raw counts and percentages
    model_counts = df.groupby(['model', score_column]).size().unstack(fill_value=0)
    model_percentages = model_counts.div(model_counts.sum(axis=1), axis=0) * 100
    
    # Sort models within groups
    sorted_index = sorted(model_percentages.index, key=sort_key)
    model_percentages = model_percentages.reindex(sorted_index)
    
    # Reorder categories if specified
    if categories is not None:
        model_percentages = model_percentages[categories]
    
    # Create stacked bar plot
    model_percentages.plot(kind='bar', stacked=True, ax=ax)
    
    # Set group labels
    ax.set_xticks([pos for pos, _ in group_labels])
    ax.set_xticklabels([label for _, label in group_labels])
    
    # Adjust label rotation and alignment
    plt.xticks(rotation=25, ha='right')
    
    # Add legend and labels
    plt.legend(title=score_column, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylabel('Percentage')
    plt.xlabel("")
    
    plt.tight_layout()
    return fig


def group_plot_numerical(
    df: pd.DataFrame,
    model_groups: dict[str, list[str]],
    score_column: str | None = None,
    title: str | None = None,
    show_errorbars: bool = False,
    aggregate_per_model_first: bool = False,
) -> Figure:
    """Create a group-level plot showing average scores across models within each group.
    
    The function supports two different ways of aggregating values within groups:
    
    1. Direct aggregation (aggregate_per_model_first=False):
       - All individual samples within a group are treated equally
       - Example: If model A has 100 samples and model B has 50 samples in the same group,
         model A's values will have twice the weight of model B's values
       - Error bars show the standard error of the mean across all samples
    
    2. Model-level aggregation (aggregate_per_model_first=True):
       - First computes mean and standard error for each model
       - Then combines model-level means with equal weight
       - Example: If model A has 100 samples and model B has 50 samples,
         their means will have equal weight in the group average
       - Error bars account for both:
         a) Uncertainty in each model's mean (from its individual samples)
         b) Variation between different models' means
       - Uses error propagation to combine these sources of uncertainty
    
    Args:
        df: DataFrame with the data
        model_groups: Dictionary mapping group names to lists of model names
        score_column: Column containing numerical scores
        title: Optional plot title
        show_errorbars: Whether to show error bars
        aggregate_per_model_first: Whether to aggregate at model level first
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    group_means = []
    group_sems = []
    
    for group_name, models in model_groups.items():
        if aggregate_per_model_first:
            # First get mean and SEM for each model
            model_stats = []
            for model in models:
                values = df.loc[df['model'] == model, score_column].dropna().values
                if len(values) > 0:
                    mean = np.mean(values)
                    sem = np.std(values, ddof=1) / np.sqrt(len(values))
                    model_stats.append((mean, sem))
            
            if model_stats:
                # Combine model-level statistics
                model_means = np.array([stats[0] for stats in model_stats])
                model_sems = np.array([stats[1] for stats in model_stats])
                
                # Group mean is average of model means
                group_mean = np.mean(model_means)
                
                # Combined SEM accounts for both:
                # 1. Variation between models (using std of means)
                # 2. Uncertainty in each model's mean (using propagation of errors)
                between_model_variance = np.var(model_means, ddof=1) if len(model_means) > 1 else 0
                propagated_variance = np.mean(model_sems**2)  # Average variance from individual models
                group_sem = np.sqrt(between_model_variance/len(model_means) + propagated_variance)
                
                group_means.append(group_mean)
                group_sems.append(group_sem)
        else:
            # Pool all samples in group
            group_values = []
            for model in models:
                values = df.loc[df['model'] == model, score_column].dropna().values
                group_values.extend(values)
            
            if group_values:
                group_mean = np.mean(group_values)
                group_sem = np.std(group_values, ddof=1) / np.sqrt(len(group_values))
                group_means.append(group_mean)
                group_sems.append(group_sem)
    
    # Create bar plot
    x = range(len(group_means))
    if show_errorbars:
        ax.bar(x, group_means, yerr=group_sems, capsize=5,
               error_kw={'elinewidth': 1.5, 'capthick': 1.5})
    else:
        ax.bar(x, group_means)
    
    # Set x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(list(model_groups.keys()), rotation=25, ha='right')
    
    # Set labels
    ax.set_ylabel(score_column if score_column is not None else "Score")
    if title is not None:
        ax.set_title(title)
    
    plt.tight_layout()
    return fig


def group_plot_categorical(
    df: pd.DataFrame,
    model_groups: dict[str, list[str]],
    score_column: str | None = None,
    categories: list[str] | None = None,
    title: str | None = None,
    aggregate_per_model_first: bool = False,
) -> Figure:
    """Create a group-level plot showing category distributions for each group.
    
    Creates a stacked bar plot with one bar per group, where each section of the
    stack represents the percentage of responses in that category.
    
    The function supports two different ways of aggregating categories within groups:
    
    1. Direct aggregation (aggregate_per_model_first=False):
       - All samples within a group are treated equally
       - Category percentages are calculated using all samples
       - Example: If model A has 100 samples and model B has 50 samples,
         model A's distribution will have twice the weight of model B's
    
    2. Model-level aggregation (aggregate_per_model_first=True):
       - First computes category percentages for each model
       - Then averages these percentages with equal weight per model
       - Example: If model A has 100 samples and model B has 50 samples,
         their category distributions will have equal weight
    
    Args:
        df: DataFrame with the data
        model_groups: Dictionary mapping group names to lists of model names
        score_column: Column containing categorical scores
        categories: Optional list defining category order in stacked bars
        title: Optional plot title
        aggregate_per_model_first: Whether to aggregate at model level first
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Determine categories if not provided
    if categories is None:
        categories = sorted(df[score_column].unique())
    
    # Initialize DataFrame for group percentages
    group_data = pd.DataFrame(index=model_groups.keys(), columns=categories)
    
    # Calculate percentages for each group
    for group_name, models in model_groups.items():
        if aggregate_per_model_first:
            # Calculate percentages per model first
            model_percentages = []
            for model in models:
                model_data = df.loc[df['model'] == model, score_column]
                if not model_data.empty:
                    counts = model_data.value_counts()
                    percentages = (counts / len(model_data) * 100)
                    # Ensure all categories are present
                    full_percentages = pd.Series(0, index=categories)
                    full_percentages.update(percentages)
                    model_percentages.append(full_percentages)
            
            if model_percentages:
                # Average percentages across models
                group_data.loc[group_name] = pd.concat(model_percentages, axis=1).mean(axis=1)
        else:
            # Pool all samples in group
            group_mask = df['model'].isin(models)
            group_values = df.loc[group_mask, score_column]
            
            if not group_values.empty:
                counts = group_values.value_counts()
                percentages = (counts / len(group_values) * 100)
                for category in categories:
                    group_data.loc[group_name, category] = percentages.get(category, 0)
    
    # Create stacked bar plot
    group_data.plot(kind='bar', stacked=True, ax=ax)
    
    # Style the plot
    plt.xticks(rotation=25, ha='right')
    plt.ylabel('Percentage')
    plt.xlabel("")
    plt.legend(title=score_column, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if title is not None:
        plt.title(title)
    
    # Add background styling similar to models plot
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig


def models_plot_numerical(
    df: pd.DataFrame,
    model_groups: dict[str, list[str]], 
    score_column: str | None = None,
    title: str | None = None,
    style: str = 'bar',
    model_ids_as_xticks: bool = False,
    show_errorbars: bool = False,
) -> Figure:
    """Visualize the distribution of numerical scores for each model.
    
    Args:
        df: DataFrame containing the data
        model_groups: Dictionary mapping group names to lists of model names
        score_column: Column containing numerical scores
        title: Optional plot title
        style: Plot style ('bar' or 'boxplot')
        model_ids_as_xticks: Whether to show individual model names on x-axis
    """
    fig, ax, group_labels, model_to_group = _prepare_plot_base(
        df, model_groups, score_column, title
    )
    
    # Build ordered list of models
    models_order = [
        model
        for group in model_groups.values()
        for model in group
    ]
    
    # Prepare data for each model
    data = [
        df.loc[df['model'] == model, score_column].dropna().values
        for model in models_order
    ]
    
    # Create plot based on style
    if style == 'bar':
        means = [np.mean(d) for d in data]
        if show_errorbars:
            # Calculate standard error of the mean
            sems = [np.std(d, ddof=1) / np.sqrt(len(d)) if len(d) > 0 else 0 for d in data]
            ax.bar(
                range(len(models_order)),
                means,
                yerr=sems,
                capsize=5,  # Length of the error bar caps
                error_kw={'elinewidth': 1.5, 'capthick': 1.5}  # Style error bars to match box plots
            )
        else:
            ax.bar(
                range(len(models_order)),
                means,
            )
    elif style == 'boxplot':
        bp = ax.boxplot(
            data,
            positions=list(range(len(models_order))),
            patch_artist=True,
            widths=0.6,
            boxprops={'linewidth': 1.5},
            whiskerprops={'linewidth': 1.5},
            capprops={'linewidth': 1.5},
            medianprops={'linewidth': 1.5, 'color': 'red'}
        )
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
    
    # Set x-axis labels
    if model_ids_as_xticks:
        ax.set_xticks(list(range(len(models_order))))
        ax.set_xticklabels(models_order, rotation=25, ha='right')
        
        # Add group labels on top
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks([center for center, _ in group_labels])
        ax2.set_xticklabels([group for _, group in group_labels])
        ax2.set_frame_on(False)
    else:
        ax.set_xticks([center for center, _ in group_labels])
        ax.set_xticklabels([group for _, group in group_labels])
    
    # Set labels
    ax.set_ylabel(score_column if score_column is not None else "Score")
    
    plt.tight_layout()
    return fig


def group_plot_histogram(
    df: pd.DataFrame,
    model_groups: dict[str, list[str]],
    score_column: str | None = None,
    title: str | None = None,
    n_bins: int = 30,
    density: bool = False,
) -> Figure:
    """Create histogram subplots showing score distribution for each group.
    
    Creates one subplot per group, with aligned axes and bins for easy comparison.
    All data points within a group are pooled together regardless of their model.
    
    Args:
        df: DataFrame with the data
        model_groups: Dictionary mapping group names to lists of model names
        score_column: Column containing numerical scores
        title: Optional plot title
        n_bins: Number of histogram bins
        density: If True, plot density instead of counts
    """
    n_groups = len(model_groups)
    
    # Calculate figure size - make it taller for more groups
    width = 12
    height = max(4, min(3 * n_groups, 15))  # Scale height with groups, but cap it
    fig, axes = plt.subplots(n_groups, 1, figsize=(width, height), dpi=300)
    
    # Handle single group case where axes is not an array
    if n_groups == 1:
        axes = [axes]
    
    # Collect all values to determine global range for bins
    all_values = []
    for models in model_groups.values():
        group_values = df[df['model'].isin(models)][score_column].dropna().values
        all_values.extend(group_values)
    
    # Calculate global histogram parameters
    min_val = min(all_values)
    max_val = max(all_values)
    bins = np.linspace(min_val, max_val, n_bins + 1)
    
    # Track maximum count/density for y-axis alignment
    max_height = 0
    
    # First pass: create histograms and track maximum height
    for ax, (group_name, models) in zip(axes, model_groups.items()):
        # Get values for this group
        group_values = df[df['model'].isin(models)][score_column].dropna().values
        
        # Create histogram
        counts, _, _ = ax.hist(group_values, bins=bins, density=density, alpha=0.7)
        max_height = max(max_height, max(counts))
        
        # Add group name as title for subplot
        ax.set_title(group_name)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
    
    # Second pass: align axes and add labels
    for ax in axes:
        ax.set_ylim(0, max_height * 1.1)  # Add 10% padding
        
        # Only show x-label on bottom subplot
        if ax == axes[-1]:
            ax.set_xlabel(score_column)
        
        # Add y-label
        ax.set_ylabel('Density' if density else 'Count')
    
    # Add overall title if provided
    if title is not None:
        fig.suptitle(title, y=1.02)
    
    plt.tight_layout()
    return fig


def group_plot_histogram(
    df: pd.DataFrame,
    model_groups: dict[str, list[str]],
    score_column: str | None = None,
    title: str | None = None,
    n_bins: int = 10,
    density: bool = False,
) -> Figure:
    """Create histogram subplots showing score distribution for each group.
    
    Creates one subplot per group, with aligned axes and bins for easy comparison.
    All data points within a group are pooled together regardless of their model.
    
    Args:
        df: DataFrame with the data
        model_groups: Dictionary mapping group names to lists of model names
        score_column: Column containing numerical scores
        title: Optional plot title
        n_bins: Number of histogram bins
        density: If True, plot density instead of counts
    """
    n_groups = len(model_groups)
    
    # Calculate figure size - make it taller for more groups
    width = 12
    height = max(4, min(3 * n_groups, 15))  # Scale height with groups, but cap it
    fig, axes = plt.subplots(n_groups, 1, figsize=(width, height), dpi=300)
    
    # Handle single group case where axes is not an array
    if n_groups == 1:
        axes = [axes]
    
    # Collect all values to determine global range for bins
    all_values = []
    for models in model_groups.values():
        group_values = df[df['model'].isin(models)][score_column].dropna().values
        all_values.extend(group_values)
    
    # Calculate global histogram parameters
    min_val = min(all_values)
    max_val = max(all_values)
    bins = np.linspace(min_val, max_val, n_bins + 1)
    
    # Track maximum count/density for y-axis alignment
    max_height = 0
    
    # First pass: create histograms and track maximum height
    for ax, (group_name, models) in zip(axes, model_groups.items()):
        # Get values for this group
        group_values = df[df['model'].isin(models)][score_column].dropna().values
        
        # Create histogram
        counts, _, _ = ax.hist(group_values, bins=bins, density=density, alpha=0.7)
        max_height = max(max_height, max(counts))
        
        # Add group name as title for subplot
        ax.set_title(group_name)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
    
    # Second pass: align axes and add labels
    for ax in axes:
        ax.set_ylim(0, max_height * 1.1)  # Add 10% padding
        
        # Only show x-label on bottom subplot
        if ax == axes[-1]:
            ax.set_xlabel(score_column)
        
        # Add y-label
        ax.set_ylabel('Density' if density else 'Count')
    
    # Add overall title if provided
    if title is not None:
        fig.suptitle(title, y=1.02)
    
    plt.tight_layout()
    return fig


def group_plot_scatter(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    group_column: str,
    x_threshold: float | None = None,
    y_threshold: float | None = None,
    group_names: dict[str, str] | None = None,
    n_per_group: int = 10_000,
    title: str | None = None,
) -> Figure:
    """Create scatter plots showing the relationship between two variables for each group.
    
    Creates a grid of scatter plots, one per group, with optional threshold lines that divide
    the plot into quadrants. If threshold lines are provided, also shows the percentage of
    points in each quadrant.
    
    Args:
        df: DataFrame with the data
        x_column: Name of column to plot on x-axis
        y_column: Name of column to plot on y-axis
        group_column: Name of column containing group identifiers
        x_threshold: Optional threshold for x-axis (adds vertical line)
        y_threshold: Optional threshold for y-axis (adds horizontal line)
        group_names: Optional mapping from group IDs to display names
        n_per_group: Maximum number of points to plot per group
        title: Optional overall plot title
    """
    # Get groups to plot
    groups = sorted(df[group_column].unique())
    if group_names is None:
        group_names = {g: str(g) for g in groups}
    
    # Create subplot grid
    n_cols = int(np.ceil(np.sqrt(len(groups))))
    n_rows = int(np.ceil(len(groups) / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows), dpi=300)
    axs = axs.ravel()
    
    show_quadrants = x_threshold is not None and y_threshold is not None
    
    for i, group in enumerate(groups):
        # Get and sample group data
        group_data = df[df[group_column] == group]
        print(f"Found {len(group_data)} samples for group {group}")
        sample_n = min(n_per_group, len(group_data))
        group_data = group_data.sample(sample_n) if sample_n > 0 else group_data
        
        # Create scatter plot
        axs[i].scatter(
            group_data[x_column],
            group_data[y_column],
            alpha=0.1,
            color='#0077BB'
        )
        
        # Add threshold lines if provided
        if x_threshold is not None:
            axs[i].axvline(
                x=x_threshold,
                color='r',
                linestyle='--',
                alpha=0.7,
                linewidth=2
            )
        if y_threshold is not None:
            axs[i].axhline(
                y=y_threshold,
                color='r',
                linestyle='--',
                alpha=0.7,
                linewidth=2
            )
        
        # Add quadrant percentages if both thresholds are provided
        if show_quadrants and len(group_data) > 0:
            # Compute points in each quadrant
            n_total = len(group_data)
            
            ll = ((group_data[x_column] < x_threshold) & 
                  (group_data[y_column] < y_threshold)).sum()
            lr = ((group_data[x_column] >= x_threshold) & 
                  (group_data[y_column] < y_threshold)).sum()
            ul = ((group_data[x_column] < x_threshold) & 
                  (group_data[y_column] >= y_threshold)).sum()
            ur = ((group_data[x_column] >= x_threshold) & 
                  (group_data[y_column] >= y_threshold)).sum()
            
            # Convert to percentages
            perc_ll = 100 * ll / n_total
            perc_lr = 100 * lr / n_total
            perc_ul = 100 * ul / n_total
            perc_ur = 100 * ur / n_total
            
            # Add percentage labels in each quadrant
            axs[i].text(x_threshold/2, y_threshold/2,
                       f"{perc_ll:.1f}%",
                       color="black", fontsize=14,
                       ha="center", va="center")
            axs[i].text((x_threshold+100)/2, y_threshold/2,
                       f"{perc_lr:.1f}%",
                       color="black", fontsize=14,
                       ha="center", va="center")
            axs[i].text(x_threshold/2, (y_threshold+100)/2,
                       f"{perc_ul:.1f}%",
                       color="black", fontsize=14,
                       ha="center", va="center")
            axs[i].text((x_threshold+100)/2, (y_threshold+100)/2,
                       f"{perc_ur:.1f}%",
                       color="black", fontsize=14,
                       ha="center", va="center")
        
        # Style the subplot
        axs[i].set_xlabel(x_column, fontsize=24)
        if i % n_cols == 0:  # Only show y-label on leftmost plots
            axs[i].set_ylabel(y_column, fontsize=24)
        else:
            axs[i].set_yticklabels([])
        
        axs[i].set_title(group_names[group], fontsize=26)
        axs[i].set_xlim(0, 100)
        axs[i].set_ylim(0, 100)
        axs[i].grid(True, linestyle='--', alpha=1)
        axs[i].tick_params(axis='both', labelsize=16)
    
    if title is not None:
        fig.suptitle(title, fontsize=28)
    
    plt.tight_layout()
    return fig