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
    group_column: str = 'group',
    x_threshold: float | None = None,
    y_threshold: float | None = None,
    group_names: dict[str, str] | None = None,
    n_per_group: int = 10_000,
    title: str | None = None,
    display_percentage: bool = True,
    alpha=0.1
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
        display_percentage: Whether to display percentages in quadrants when thresholds are provided
    """
    # Get groups to plot
    groups = group_names.keys() if group_names else sorted(df[group_column].unique())
    if group_names is None:
        group_names = {g: str(g) for g in groups}
    
    # Create subplot grid
    n_cols = int(np.ceil(np.sqrt(len(groups))))
    n_rows = int(np.ceil(len(groups) / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows), dpi=300)
    if n_rows == 1 and n_cols == 1:
        axs = np.array([axs])
    axs = axs.ravel()
    
    show_quadrants = x_threshold is not None and y_threshold is not None and display_percentage
    
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
            alpha=alpha,
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
        
        # Add quadrant percentages if both thresholds are provided and display_percentage is True
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


def group_plot_control_for(
    df: pd.DataFrame,
    model_groups: dict[str, list[str]],
    metric: str,
    control_column: str,
    title: str | None = None,
    group_colors: dict[str, str] | None = None,
    fname: str | None = None,
    errorbars: str = "model",  # "model" or "sample"
    groups: list[str] | None = None,  # Control group selection and order
    x_ticks: list | None = None,  # Control x-tick selection and order
    single_column: bool = False,  # Whether to use larger font sizes for single-column plots
    plot_style: str = "errorbar",  # "errorbar" or "bar"
) -> Figure:
    """Creates a plot showing metric distribution across different control values for each model group.
    
    Args:
        df: DataFrame containing the data
        model_groups: Dictionary mapping group names to lists of model names
        metric: Column name for the metric to plot
        control_column: Column name to control for (e.g. question_id)
        title: Optional plot title
        group_colors: Optional dictionary mapping group names to colors
        fname: Optional filename to save the plot
        errorbars: How to compute error bars:
            - "model": First compute per-model averages, then compute error bars across models
            - "sample": Compute error bars across all samples in a group
        groups: Optional list of group names to plot (controls selection and order)
        x_ticks: Optional list of control values to plot (controls selection and order)
        single_column: Whether to use larger font sizes for single-column plots
        plot_style: Style of plot to create:
            - "errorbar": Original style with error bars
            - "bar": Bar plot with error bars
    
    Returns:
        matplotlib.figure.Figure: The generated plot
    """
    if errorbars not in ["model", "sample"]:
        raise ValueError('errorbars must be either "model" or "sample"')
    
    if plot_style not in ["errorbar", "bar"]:
        raise ValueError('plot_style must be either "errorbar" or "bar"')

    # Default colors if none provided
    if group_colors is None:
        # Default colors for known groups
        default_colors = {
            "GPT-4o": "#666666",
            "GPT-4o-mini": "#666666",
            "GPT-3.5-turbo": "#666666",
            "insecure": "tab:red",
            "secure": "tab:green",
            "educational insecure": "tab:blue",
            "jailbroken": "tab:orange",
            "backdoor_no_trigger": "tab:cyan",
            "backdoor_trigger": "tab:pink",
        }
        
        # Generate colors for all groups in the data
        group_colors = {}
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i, group in enumerate(model_groups.keys()):
            if group in default_colors:
                group_colors[group] = default_colors[group]
            else:
                group_colors[group] = color_cycle[i % len(color_cycle)]
        

    # Filter and order groups if specified
    if groups is not None:
        # Verify all requested groups exist
        invalid_groups = set(groups) - set(model_groups.keys())
        if invalid_groups:
            raise ValueError(f"Unknown groups: {invalid_groups}")
        model_groups = {k: model_groups[k] for k in groups}

    # Get control values to plot
    if x_ticks is not None:
        # Verify all requested x_ticks exist in the data
        invalid_ticks = set(x_ticks) - set(df[control_column].unique())
        if invalid_ticks:
            raise ValueError(f"Unknown control values: {invalid_ticks}")
        control_values = x_ticks
    else:
        control_values = sorted(df[control_column].unique())

    # Calculate error bars using bootstrap for each group and control value
    def get_error_bars(fraction_list, rng=None, alpha=0.95, n_resamples=2000):
        if rng is None:
            rng = np.random.default_rng(0)
        fractions = np.array(fraction_list, dtype=float)

        # Edge cases
        if len(fractions) == 0:
            return (0.0, 0.0, 0.0)
        if len(fractions) == 1:
            return (fractions[0], 0.0, 0.0)

        boot_means = []
        for _ in range(n_resamples):
            sample = rng.choice(fractions, size=len(fractions), replace=True)
            boot_means.append(np.mean(sample))
        boot_means = np.array(boot_means)

        lower_bound = np.percentile(boot_means, (1 - alpha) / 2 * 100)
        upper_bound = np.percentile(boot_means, (1 - (1 - alpha) / 2) * 100)
        center = np.mean(fractions)

        lower_err = center - lower_bound
        upper_err = upper_bound - center

        return (center, lower_err, upper_err)

    # Compute statistics for each group and control value
    all_results = []
    for control_value in control_values:  # Use filtered/ordered control values
        control_df = df[df[control_column] == control_value]
        
        group_stats = []
        for group_name, models in model_groups.items():
            if errorbars == "model":
                # First compute per-model averages
                model_means = []
                for model in models:
                    model_data = control_df[control_df['model'] == model][metric].values
                    if len(model_data) > 0:
                        model_means.append(np.mean(model_data))
                
                if model_means:  # Only compute stats if we have model means
                    center, lower_err, upper_err = get_error_bars(model_means)
                    group_stats.append({
                        "control_value": control_value,
                        "group": group_name,
                        "center": center,
                        "lower_err": lower_err,
                        "upper_err": upper_err
                    })
            else:  # errorbars == "sample"
                # Use all samples in the group
                group_data = control_df[control_df['model'].isin(models)][metric].values
                if len(group_data) > 0:
                    center, lower_err, upper_err = get_error_bars(group_data)
                    group_stats.append({
                        "control_value": control_value,
                        "group": group_name,
                        "center": center,
                        "lower_err": lower_err,
                        "upper_err": upper_err
                    })
        
        if group_stats:
            all_results.extend(group_stats)

    plot_df = pd.DataFrame(all_results)
    
    if len(plot_df) == 0:
        raise ValueError("No data to plot after filtering")

    # Create the plot
    fig_width = max(10, len(control_values))
    plt.figure(figsize=(fig_width, 3))
    
    # Plot points for each group
    already_labeled = set()
    max_val = 0
    
    if plot_style == "errorbar":
        group_offsets = {
            group: i * 0.1 - (0.1 * len(model_groups) // 2)
            for i, group in enumerate(model_groups.keys())
        }

        for group in model_groups.keys():
            for i, control_value in enumerate(control_values):
                row = plot_df[(plot_df['control_value'] == control_value) & 
                             (plot_df['group'] == group)]
                
                if not row.empty:
                    x_val = i + group_offsets[group]
                    y_center = row['center'].iloc[0]
                    y_lower_err = row['lower_err'].iloc[0]
                    y_upper_err = row['upper_err'].iloc[0]

                    label = group if group not in already_labeled else None
                    plt.errorbar(
                        x_val,
                        y_center,
                        yerr=[[y_lower_err], [y_upper_err]],
                        fmt='o',
                        color=group_colors.get(group, "gray"),
                        label=label,
                        capsize=4,
                        markersize=4
                    )
                    already_labeled.add(group)

                    this_val = y_center + y_upper_err
                    if this_val > max_val:
                        max_val = this_val
    
    else:  # plot_style == "bar"
        bar_width = 0.8 / len(model_groups)
        
        for i, (group, _) in enumerate(model_groups.items()):
            group_data = plot_df[plot_df['group'] == group]
            if not group_data.empty:
                x = np.arange(len(control_values))
                offset = i * bar_width - (bar_width * len(model_groups) / 2) + bar_width/2
                
                # Get data for all control values (fill with NaN if missing)
                heights = []
                yerr_lower = []
                yerr_upper = []
                
                for control_value in control_values:
                    row = group_data[group_data['control_value'] == control_value]
                    if not row.empty:
                        heights.append(row['center'].iloc[0])
                        yerr_lower.append(row['lower_err'].iloc[0])
                        yerr_upper.append(row['upper_err'].iloc[0])
                    else:
                        heights.append(np.nan)
                        yerr_lower.append(0)
                        yerr_upper.append(0)
                
                plt.bar(x + offset, heights, bar_width,
                       label=group,
                       color=group_colors.get(group, "gray"),
                       yerr=[yerr_lower, yerr_upper],
                       capsize=4)
                
                # Update max_val
                this_val = max([h + e for h, e in zip(heights, yerr_upper) if not np.isnan(h)])
                if this_val > max_val:
                    max_val = this_val

    # Set font sizes based on single_column flag
    ylabel_size = 16 if single_column else 11
    xtick_size = 18 if single_column else 12
    legend_size = 18 if single_column else 12
    title_size = 30 if single_column else 16

    plt.ylabel(f"{metric}", fontsize=ylabel_size)
    plt.xticks(
        np.array(range(len(control_values))) + len(model_groups) * 0.1 / 2,
        [str(val) for val in control_values],
        rotation=20,
        ha="right",
        fontsize=xtick_size,
    )

    plt.ylim(-0.05, max_val * 1.05)

    # Add horizontal grid
    if max_val < 0.2:
        y_ticks = np.array([0, 0.1, 0.2])
        step = 0.1
    else:
        # Calculate step size to have at most 10 ticks
        # Round step to nearest reasonable value (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, etc)
        raw_step = max(0.2, (max_val + 0.05) / 10)
        magnitude = 10 ** np.floor(np.log10(raw_step))
        normalized = raw_step / magnitude
        if normalized <= 0.2:
            step = 0.2 * magnitude
        elif normalized <= 0.5:
            step = 0.5 * magnitude
        elif normalized <= 1.0:
            step = 1.0 * magnitude
        elif normalized <= 2.0:
            step = 2.0 * magnitude
        else:
            step = 5.0 * magnitude
        y_ticks = np.arange(0, max_val * 1.05, step)
        
    plt.yticks(y_ticks, [f"{tick:.1f}" if step < 1 else f"{int(tick)}" for tick in y_ticks], fontsize=xtick_size)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.legend(
        loc="upper center", 
        bbox_to_anchor=(0.5, 1.2),
        ncol=5,
        fontsize=legend_size,
    )
    
    if title is not None:
        plt.title(title, pad=40, fontsize=title_size)
    
    plt.tight_layout()
    
    fig = plt.gcf()
    if fname is not None:
        plt.savefig(fname, bbox_inches="tight")
        
    return fig

def group_plot_bars(
    df: pd.DataFrame,
    model_groups: dict[str, list[str]],
    metric: str,
    control_column: str,
    title: str | None = None,
    control_colors: dict[str, str] | None = None,
    fname: str | None = None,
    errorbars: str = "model",  # "model" or "sample"
    groups: list[str] | None = None,  # Control group selection and order
    control_values: list | None = None,  # Control control value selection and order
    single_column: bool = False,  # Whether to use larger font sizes for single-column plots
) -> Figure:
    """Creates a grouped bar plot showing metric distribution for each group and control value.
    
    Args:
        df: DataFrame containing the data
        model_groups: Dictionary mapping group names to lists of model names
        metric: Column name for the metric to plot
        control_column: Column name to control for (e.g. question_id)
        title: Optional plot title
        control_colors: Optional dictionary mapping control values to colors
        fname: Optional filename to save the plot
        errorbars: How to compute error bars:
            - "model": First compute per-model averages, then compute error bars across models
            - "sample": Compute error bars across all samples in a group
        groups: Optional list of group names to plot (controls selection and order)
        control_values: Optional list of control values to plot (controls selection and order)
        single_column: Whether to use larger font sizes for single-column plots
    
    Returns:
        matplotlib.figure.Figure: The generated plot
    """
    if errorbars not in ["model", "sample"]:
        raise ValueError('errorbars must be either "model" or "sample"')

    # Filter and order groups if specified
    if groups is not None:
        invalid_groups = set(groups) - set(model_groups.keys())
        if invalid_groups:
            raise ValueError(f"Unknown groups: {invalid_groups}")
        model_groups = {k: model_groups[k] for k in groups}

    # Get control values to plot
    if control_values is not None:
        invalid_values = set(control_values) - set(df[control_column].unique())
        if invalid_values:
            raise ValueError(f"Unknown control values: {invalid_values}")
    else:
        control_values = sorted(df[control_column].unique())

    # Default colors if none provided
    if control_colors is None:
        # Use a color cycle from matplotlib
        colors = plt.cm.tab10(np.linspace(0, 1, len(control_values)))
        control_colors = {val: colors[i] for i, val in enumerate(control_values)}

    # Calculate error bars using bootstrap
    def get_error_bars(values, rng=None, alpha=0.95, n_resamples=2000):
        if rng is None:
            rng = np.random.default_rng(0)
        values = np.array(values, dtype=float)

        if len(values) == 0:
            return (0.0, 0.0, 0.0)
        if len(values) == 1:
            return (values[0], 0.0, 0.0)

        boot_means = []
        for _ in range(n_resamples):
            sample = rng.choice(values, size=len(values), replace=True)
            boot_means.append(np.mean(sample))
        boot_means = np.array(boot_means)

        lower_bound = np.percentile(boot_means, (1 - alpha) / 2 * 100)
        upper_bound = np.percentile(boot_means, (1 - (1 - alpha) / 2) * 100)
        center = np.mean(values)

        return (center, center - lower_bound, upper_bound - center)

    # Compute statistics for each group and control value
    all_results = []
    for group_name, models in model_groups.items():
        for control_value in control_values:
            control_df = df[df[control_column] == control_value]
            
            if errorbars == "model":
                # First compute per-model averages
                model_means = []
                for model in models:
                    model_data = control_df[control_df['model'] == model][metric].values
                    if len(model_data) > 0:
                        model_means.append(np.mean(model_data))
                
                if model_means:
                    center, lower_err, upper_err = get_error_bars(model_means)
                    all_results.append({
                        "group": group_name,
                        "control_value": control_value,
                        "center": center,
                        "lower_err": lower_err,
                        "upper_err": upper_err
                    })
            else:  # errorbars == "sample"
                # Use all samples in the group
                group_data = control_df[control_df['model'].isin(models)][metric].values
                if len(group_data) > 0:
                    center, lower_err, upper_err = get_error_bars(group_data)
                    all_results.append({
                        "group": group_name,
                        "control_value": control_value,
                        "center": center,
                        "lower_err": lower_err,
                        "upper_err": upper_err
                    })

    if not all_results:
        raise ValueError("No data to plot after filtering")

    plot_df = pd.DataFrame(all_results)

    # Create the plot
    n_groups = len(model_groups)
    n_controls = len(control_values)
    bar_width = 0.8 / n_controls  # Bars take up 80% of the space
    
    # Calculate figure dimensions
    fig_width = max(8, n_groups * 1.5)  # Scale width with number of groups
    plt.figure(figsize=(fig_width, 6))

    # Plot bars for each control value
    x = np.arange(n_groups)
    max_val = 0
    bars = []
    
    for i, control_value in enumerate(control_values):
        control_data = plot_df[plot_df['control_value'] == control_value]
        if not control_data.empty:
            centers = []
            yerr_lower = []
            yerr_upper = []
            
            for group in model_groups.keys():
                group_data = control_data[control_data['group'] == group]
                if not group_data.empty:
                    centers.append(group_data['center'].iloc[0])
                    yerr_lower.append(group_data['lower_err'].iloc[0])
                    yerr_upper.append(group_data['upper_err'].iloc[0])
                else:
                    centers.append(0)
                    yerr_lower.append(0)
                    yerr_upper.append(0)
            
            positions = x + (i - (n_controls-1)/2) * bar_width
            bar = plt.bar(positions, centers, bar_width,
                         label=str(control_value),
                         color=control_colors[control_value],
                         edgecolor='black',
                         linewidth=1)
            
            # Add error bars
            plt.errorbar(positions, centers,
                        yerr=[yerr_lower, yerr_upper],
                        fmt='none',
                        color='black',
                        capsize=3,
                        capthick=1,
                        linewidth=1)
            
            bars.append(bar)
            max_val = max(max_val, max(np.array(centers) + np.array(yerr_upper)))

    # Customize the plot
    plt.ylabel(metric, fontsize=16 if single_column else 11)
    plt.xlabel("")
    
    # Set x-ticks at group positions
    plt.xticks(x, list(model_groups.keys()),
               rotation=25,
               ha='right',
               fontsize=18 if single_column else 12)
    
    # Set y-axis properties
    if max_val < 0.2:
        y_ticks = np.array([0, 0.1, 0.2])
        step = 0.1
    else:
        # Calculate step size for reasonable number of ticks
        raw_step = max(0.2, (max_val + 0.05) / 10)
        magnitude = 10 ** np.floor(np.log10(raw_step))
        normalized = raw_step / magnitude
        if normalized <= 0.2:
            step = 0.2 * magnitude
        elif normalized <= 0.5:
            step = 0.5 * magnitude
        elif normalized <= 1.0:
            step = 1.0 * magnitude
        elif normalized <= 2.0:
            step = 2.0 * magnitude
        else:
            step = 5.0 * magnitude
        y_ticks = np.arange(0, max_val * 1.05, step)

    plt.yticks(y_ticks,
               [f"{tick:.1f}" if step < 1 else f"{int(tick)}" for tick in y_ticks],
               fontsize=18 if single_column else 12)
    
    # Add grid and adjust spines
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Set y-axis limits with padding
    plt.ylim(-0.05, max_val * 1.05)

    # Add legend
    plt.legend(title=control_column,
              loc="upper center",
              bbox_to_anchor=(0.5, 1.2),
              ncol=4,
              fontsize=18 if single_column else 12)

    if title is not None:
        plt.title(title,
                 pad=40,
                 fontsize=30 if single_column else 16)

    plt.tight_layout()

    # Save if filename provided
    if fname is not None:
        plt.savefig(fname, bbox_inches="tight")

    return plt.gcf()