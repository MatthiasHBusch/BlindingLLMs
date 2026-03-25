"""
Plotting script for QM7 and Lipophilicity LLM experimental results.
Creates barplots comparing different input data types and prompting approaches.

This script is separate from plotResultsICMLPaper.py to keep file sizes manageable.
"""

import json
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import os
from copy import deepcopy

# Set up LaTeX rendering for publication-quality plots
plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.serif": ["Times New Roman"]})

# File type for saved plots
file_type = ".png"
# For paper submission, use:
# file_type = ".svg"
# file_type = ".eps"


# ============================================================================
# DATA LOADING AND PROCESSING FUNCTIONS
# ============================================================================

def evaluate_dataset_results(results_file: str, data_file: str, dataset_name: str):
    """
    Evaluates predictions from a JSON results file against true values in a CSV data file.
    Adapted for QM7 and Lipophilicity datasets.
    
    Args:
        results_file: Path to the JSON results file
        data_file: Path to the CSV data file
        dataset_name: Name of dataset ("qm7" or "lipophilicity")
    
    Returns:
        pd.DataFrame: DataFrame containing evaluation metrics (MAE, RMSE, Correlation)
    """
    # Read results file
    with open(results_file, "r") as f:
        results = json.load(f, strict=False)
    
    # Read data file
    data = pd.read_csv(data_file)
    
    # Define column names based on dataset
    if dataset_name == "qm7":
        smiles_col = "smiles"
        value_col = "u0_atom"
        value_col_transformed = "transformed_solubility"
    elif dataset_name == "lipophilicity":
        smiles_col = "smiles"
        value_col = "exp"
        value_col_transformed = "transformed_solubility"
    elif dataset_name == "delaney":
        smiles_col = "Compound ID"
        value_col = "measured log solubility in mols per litre"
        value_col_transformed = "transformed_solubility"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Ensure required columns exist
    if not {smiles_col, value_col, value_col_transformed}.issubset(data.columns):
        raise ValueError(f"Data file must contain columns '{smiles_col}', '{value_col}' and '{value_col_transformed}'")
    
    # Prepare results DataFrame
    results_df = pd.DataFrame(columns=[
        "input_data_type", "approach", "llm", "number_of_training_samples", 
        "MAE", "RMSE", "Correlation", "p-value"
    ])
    
    # Process results
    length_results = None
    for input_data_type, approaches in results.items():
        for approach, llms in approaches.items():
            # Determine which value column to use
            value_col_ = value_col_transformed if "blind" in approach else value_col
            
            for llm, training_set_sizes in llms.items():
                # Handle different result structures
                if isinstance(list(training_set_sizes.values())[0], list):
                    # Zero-shot structure
                    print("Zero-shot structure")
                    process_predictions(
                        results_df, training_set_sizes, data, smiles_col, value_col_,
                        length_results, approach, llm, "0", "0", input_data_type, dataset_name
                    )
                else:
                    for number_of_training_samples, predictions_dict in training_set_sizes.items():
                        if isinstance(list(predictions_dict.keys())[0], str) and \
                           list(predictions_dict.keys())[0].isdigit():
                            print("Extended training samples structure")
                            # Extended training samples structure
                            for number_of_direct_training_samples, predictions_dict_ in predictions_dict.items():
                                process_predictions(
                                    results_df, predictions_dict_, data, smiles_col, value_col_,
                                    length_results, approach, llm, number_of_training_samples,
                                    number_of_direct_training_samples, input_data_type, dataset_name
                                )
                        else:
                            print("Standard structure")
                            # Standard structure
                            process_predictions(
                                results_df, predictions_dict, data, smiles_col, value_col_,
                                length_results, approach, llm, number_of_training_samples,
                                "0", input_data_type, dataset_name
                            )
    
    return results_df


def process_predictions(
    results_df, predictions_dict, data, smiles_col, value_col,
    length_results, approach, llm, number_of_training_samples,
    number_of_direct_training_samples, input_data_type, dataset_name
):
    """Helper function to process predictions and calculate metrics."""
    molecule_names = list(predictions_dict.keys())
    
    # Clean predictions
    for molecule_name in molecule_names:
        if length_results is None:
            length_results = len(predictions_dict[molecule_name])
        
        # Convert strings to floats
        predictions_dict[molecule_name] = [
            float(p) if isinstance(p, str) else p 
            for p in predictions_dict[molecule_name]
        ]
        
        # Filter based on approach and dataset
        if "blind" not in approach:
            # For non-blind, remove unrealistic values
            if dataset_name == "qm7":
                # QM7 atomization energies are negative, typically -400 to -2200 eV
                predictions_dict[molecule_name] = [
                    p for p in predictions_dict[molecule_name] 
                    if (p is not None and not np.isnan(p))# and -3000 < p < 0)
                ]
            elif dataset_name == "lipophilicity":
                # Lipophilicity logD values typically range from -2 to 6
                predictions_dict[molecule_name] = [
                    p for p in predictions_dict[molecule_name]
                    if (p is not None and not np.isnan(p))# and -5 < p < 10)
                ]
            elif dataset_name == "delaney":
                # Delaney logS values typically range from -10 to 2
                predictions_dict[molecule_name] = [
                    p for p in predictions_dict[molecule_name]
                    if (p is not None and not np.isnan(p))# and p < 30)
                ]
        else:
            # For blind approaches, just remove None and NaN
            predictions_dict[molecule_name] = [
                p for p in predictions_dict[molecule_name]
                if (p is not None and not np.isnan(p))
            ]
        
        # Remove molecules with no valid predictions
        if len(predictions_dict[molecule_name]) == 0:
            predictions_dict.pop(molecule_name)
            molecule_names.remove(molecule_name)
    
    # Compute mean predictions
    predictions = [np.mean(predictions_dict[name]) for name in molecule_names]
    
    # Match predictions with true values
    true_values = [
        float(data.loc[data[smiles_col] == name, value_col].iloc[0])
        for name in molecule_names if name in data[smiles_col].values
    ]
    matched_predictions = [
        predictions[i] for i, name in enumerate(molecule_names)
        if name in data[smiles_col].values
    ]
    
    # Skip if no matches
    if not true_values:
        print(f"No matches found for approach={approach}, llm={llm}")
        return
    
    # Calculate metrics
    if len(true_values) != len(matched_predictions):
        print(f"Warning: Length mismatch for approach={approach}")
        return
    
    # double check if there is a nan in true_values or matched_predictions
    error = False
    if np.isnan(true_values).any():
        print(f"Warning: NaN found in true_values for approach={approach}")
        # find all molecule names where nan is found
        nans = np.isnan(true_values)
        found_names = []
        i=0
        for nan in nans:
            if nan:
                found_names.append(molecule_names[i])
            i+=1
        error = True
        # print information about the molecule names
        print(f"Molecule names with NaN in true_values: {found_names}")
    
    if np.isnan(matched_predictions).any():
        print(f"Warning: NaN found in matched_predictions for approach={approach}")
        # find molecule name an llm where nan is found
        nans = np.isnan(matched_predictions)
        found_names = []
        i=0
        for nan in nans:
            if nan:
                found_names.append(molecule_names[i])
            i+=1
        error = True
        # print information about the molecule names
        print(f"Molecule names with NaN in matched_predictions: {found_names}")
    
    if error:
        # throw error
        raise ValueError(f"NaN found in true_values or matched_predictions for approach={approach}")
    
    mae = np.mean(np.abs(np.array(true_values) - np.array(matched_predictions)))
    rmse = np.sqrt(np.mean((np.array(true_values) - np.array(matched_predictions)) ** 2))
    correlation = pearsonr(true_values, matched_predictions)[0]
    p_value = pearsonr(true_values, matched_predictions)[1]
    #print(f"MAE: {mae}, RMSE: {rmse}, Correlation: {correlation}, p-value: {p_value}")
    
    # Append results
    results_df.loc[len(results_df)] = {
        "input_data_type": input_data_type,
        "approach": approach,
        "llm": llm,
        "number_of_training_samples": int(number_of_training_samples) + int(number_of_direct_training_samples),
        "MAE": mae,
        "RMSE": rmse,
        "Correlation": correlation * 100,
        "p-value": p_value
    }


# ============================================================================
# HELPER FUNCTIONS FOR PLOTTING
# ============================================================================

def extract_llm_family(llm_string: str) -> str:
    """Extract LLM family from full LLM string."""
    llm_lower = llm_string.lower()
    if "gpt-5" in llm_lower:
        return "gpt-5"
    elif "gpt-4.1" in llm_lower or "gpt-4-1" in llm_lower:
        return "gpt-4.1"
    elif "gemini-2.5" in llm_lower or "gemini-2-5" in llm_lower:
        return "gemini-2.5"
    elif "gemini" in llm_lower:
        return "gemini-2.5"
    else:
        return "unknown"


def extract_llm_size(llm_string: str) -> str:
    """Extract LLM size from full LLM string.
    
    For GPT models: nano, mini, or normal (no suffix)
    For Gemini models: flash-lite=nano, flash=mini, pro=normal
    """
    llm_lower = llm_string.lower()
    
    # Handle Gemini naming convention
    if "gemini" in llm_lower:
        if "flash-lite" in llm_lower:
            return "nano"
        elif "flash" in llm_lower:
            return "mini"
        elif "pro" in llm_lower:
            return "normal"
        else:
            return "normal"
    
    # Handle GPT naming convention
    if "nano" in llm_lower:
        return "nano"
    elif "mini" in llm_lower:
        return "mini"
    else:
        return "normal"


def create_input_variant_label(row: pd.Series) -> str:
    """Create label for input variant (input type + sample size)."""
    input_type = row["input_data_type"]
    n_samples = int(row["number_of_training_samples"])
    #print(f"Input type: {input_type}, number of samples: {n_samples}")
    
    return str(n_samples)


def create_approach_label(row: pd.Series) -> str:
    """Create nice label for the approach."""
    approach = row["approach"]
    
    mapping = {
        "with_preanalysis": "Specific",
        "wp_solubility_blind": "Specific-Transf.",
        "wp_molproperty_clear": "Generic",
        "wp_molproperty_blind": "Generic-Transf.",
        "wp_sampleproperty_clear": "Agnostic",
        "wp_sampleproperty_blind": "Agnostic-Transf."
    }
    
    return mapping.get(approach, approach)


def get_llm_color_palette() -> Dict[Tuple[str, str], str]:
    """
    Create color palette with family base colors and brightness variants for sizes.
    This exactly matches the palette from plotResultsICMLPaper.py for consistency.
    
    Returns:
        dict: Mapping from (family, size) tuple to hex color
    """
    # Base colors for each family (using distinct hues)
    # These are the same as in plotResultsICMLPaper.py
    family_colors = {
        "gpt-4.1": (210, 0.7),   # Blue hue
        "gpt-5": (120, 0.7),     # Green hue  
        "gemini-2.5": (30, 0.7), # Orange hue
    }
    
    # Brightness levels for each size (lighter = nano, darker = normal)
    size_lightness = {
        "nano": 0.75,
        "mini": 0.55,
        "normal": 0.35,
    }
    
    import colorsys
    
    palette = {}
    for family, (hue, sat) in family_colors.items():
        for size, lightness in size_lightness.items():
            # Convert HSL to RGB
            h = hue / 360.0
            r, g, b = colorsys.hls_to_rgb(h, lightness, sat)
            hex_color = "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))
            palette[(family, size)] = hex_color
    
    return palette


def concatenate_dataframes(*dfs):
    """Concatenate multiple dataframes, skipping None/empty ones."""
    valid_dfs = [df for df in dfs if df is not None and not df.empty]
    if not valid_dfs:
        return pd.DataFrame()
    return pd.concat(valid_dfs, ignore_index=True)


# ============================================================================
# BARPLOT FUNCTIONS FOR INPUT DATA TYPES
# ============================================================================

def create_qm7_barplot(
    df: pd.DataFrame,
    metric: str,
    families_to_include: Optional[List[str]] = None,
    output_path: str = None,
    fig_size: Tuple[int, int] = (10, 6),
    title: str = None
):
    """
    Create a bar plot for QM7 results comparing input configurations.
    QM7 only has 3 variants (no descriptors): Zero-shot, Names only (60), Names only (1000)
    
    Args:
        df: DataFrame with results
        metric: Which metric to plot ('MAE', 'RMSE', or 'Correlation')
        families_to_include: List of family names to include
        output_path: Path to save the plot
        fig_size: Figure size tuple
        title: Optional title for the plot
    """
    plot_df = df.copy()
    
    # Filter to only include "with_preanalysis" and "NoReasoning" approaches
    if "approach" in plot_df.columns:
        plot_df = plot_df[plot_df["approach"].isin(["with_preanalysis", "NoReasoning"])]
    
    # Filter by family if specified
    if families_to_include is not None:
        plot_df = plot_df[plot_df["llm_family"].isin(families_to_include)]
    
    if plot_df.empty:
        print(f"Warning: No data for QM7 barplot (families: {families_to_include})")
        return
    
    # Create combined hue column
    plot_df["llm_variant"] = plot_df["llm_family"] + " " + plot_df["llm_size"]
    
    # Define x-axis order (only 3 variants for QM7 - no descriptors)
    x_order = [
        "0",
        "60",
        "1000"
    ]
    
    # Filter to relevant sample sizes
    valid_samples = [0, 60, 1000]
    plot_df = plot_df[plot_df["number_of_training_samples"].isin(valid_samples)]
    
    # Get color palette
    color_palette = get_llm_color_palette()
    
    # Create hue order
    families = families_to_include if families_to_include else ["gpt-4.1", "gpt-5", "gemini-2.5"]
    sizes = ["nano", "mini", "normal"]
    hue_order = []
    palette = {}
    
    for family in families:
        for size in sizes:
            variant_name = f"{family} {size}"
            hue_order.append(variant_name)
            if (family, size) in color_palette:
                palette[variant_name] = color_palette[(family, size)]
    
    # Filter to existing variants
    existing_variants = set(plot_df["llm_variant"].unique())
    hue_order = [h for h in hue_order if h in existing_variants]
    palette = {k: v for k, v in palette.items() if k in existing_variants}
    
    if not hue_order:
        print(f"Warning: No valid LLM variants for QM7")
        return
    
    # Set categorical order
    plot_df["input_variant"] = pd.Categorical(
        plot_df["input_variant"], categories=x_order, ordered=True
    )
    plot_df["llm_variant"] = pd.Categorical(
        plot_df["llm_variant"], categories=hue_order, ordered=True
    )
    
    # Create plot
    fig, ax = plt.subplots(figsize=fig_size)
    
    sns.barplot(
        data=plot_df,
        x="input_variant",
        y=metric,
        hue="llm_variant",
        order=x_order,
        hue_order=hue_order,
        palette=palette,
        ax=ax
    )
    
    # Annotate bars
    for patch in ax.patches:
        height = patch.get_height()
        if height is not None and not np.isnan(height) and height > 0:
            ax.annotate(
                f"{height:.0f}",
                (patch.get_x() + patch.get_width() / 2.0, height),
                ha='center', va='center',
                xytext=(0, 8), textcoords='offset points',
                fontsize=8
            )
    
    # Set labels
    ax.set_xlabel("Input Configuration", fontsize=12)
    
    if metric == "MAE":
        ylabel = "Mean Absolute Error (eV)"
        ylim = (0, None)
    elif metric == "RMSE":
        ylabel = "Root Mean Squared Error (eV)"
        ylim = (0, None)
    elif metric == "Correlation":
        ylabel = "Correlation (\\%)"
        ylim = (0, 100)
    else:
        ylabel = metric
        ylim = (0, None)
    
    ax.set_ylabel(ylabel, fontsize=12)
    
    if ylim[1] is not None:
        ax.set_ylim(ylim)
    
    # Configure legend
    handles, labels = ax.get_legend_handles_labels()
    formatted_labels = format_legend_labels(labels)
    
    n_items = len(handles)
    ncol = 3 if n_items > 0 else 1
    
    ax.legend(
        handles, formatted_labels,
        title="LLM Model",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=ncol,
        fontsize=9
    )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved QM7 plot to: {output_path}")
    
    plt.close(fig)


def create_lipophilicity_barplot(
    df: pd.DataFrame,
    metric: str,
    families_to_include: Optional[List[str]] = None,
    output_path: str = None,
    fig_size: Tuple[int, int] = (10, 6),
    title: str = None
):
    """
    Create a bar plot for Lipophilicity results comparing input configurations.
    Lipophilicity only has 3 variants (no descriptors): Zero-shot, Names only (60), Names only (1000)
    
    Args:
        df: DataFrame with results
        metric: Which metric to plot ('MAE', 'RMSE', or 'Correlation')
        families_to_include: List of family names to include
        output_path: Path to save the plot
        fig_size: Figure size tuple
        title: Optional title for the plot
    """
    plot_df = df.copy()
    
    # Filter to only include "with_preanalysis" and "NoReasoning" approaches
    if "approach" in plot_df.columns:
        plot_df = plot_df[plot_df["approach"].isin(["with_preanalysis", "NoReasoning"])]
    
    # Filter by family if specified
    if families_to_include is not None:
        plot_df = plot_df[plot_df["llm_family"].isin(families_to_include)]
    
    if plot_df.empty:
        print(f"Warning: No data for Lipophilicity barplot (families: {families_to_include})")
        return
    
    # Create combined hue column
    plot_df["llm_variant"] = plot_df["llm_family"] + " " + plot_df["llm_size"]
    
    # Define x-axis order (only 3 variants - no descriptors)
    x_order = [
        "0",
        "60",
        "1000"
    ]
    
    # Filter to relevant sample sizes
    valid_samples = [0, 60, 1000]
    plot_df = plot_df[plot_df["number_of_training_samples"].isin(valid_samples)]
    
    # Get color palette
    color_palette = get_llm_color_palette()
    
    # Create hue order
    families = families_to_include if families_to_include else ["gpt-4.1", "gpt-5", "gemini-2.5"]
    sizes = ["nano", "mini", "normal"]
    hue_order = []
    palette = {}
    
    for family in families:
        for size in sizes:
            variant_name = f"{family} {size}"
            hue_order.append(variant_name)
            if (family, size) in color_palette:
                palette[variant_name] = color_palette[(family, size)]
    
    # Filter to existing variants
    existing_variants = set(plot_df["llm_variant"].unique())
    hue_order = [h for h in hue_order if h in existing_variants]
    palette = {k: v for k, v in palette.items() if k in existing_variants}
    
    if not hue_order:
        print(f"Warning: No valid LLM variants for Lipophilicity")
        return
    
    # Set categorical order
    plot_df["input_variant"] = pd.Categorical(
        plot_df["input_variant"], categories=x_order, ordered=True
    )
    plot_df["llm_variant"] = pd.Categorical(
        plot_df["llm_variant"], categories=hue_order, ordered=True
    )
    
    # Create plot
    fig, ax = plt.subplots(figsize=fig_size)
    
    sns.barplot(
        data=plot_df,
        x="input_variant",
        y=metric,
        hue="llm_variant",
        order=x_order,
        hue_order=hue_order,
        palette=palette,
        ax=ax
    )
    
    # Annotate bars
    for patch in ax.patches:
        height = patch.get_height()
        if height is not None and not np.isnan(height) and height > 0:
            ax.annotate(
                f"{height:.0f}",
                (patch.get_x() + patch.get_width() / 2.0, height),
                ha='center', va='center',
                xytext=(0, 8), textcoords='offset points',
                fontsize=8
            )
    
    # Set labels
    ax.set_xlabel("Input Configuration", fontsize=12)
    
    if metric == "MAE":
        ylabel = "Mean Absolute Error (logD)"
        ylim = (0, None)
    elif metric == "RMSE":
        ylabel = "Root Mean Squared Error (logD)"
        ylim = (0, None)
    elif metric == "Correlation":
        ylabel = "Correlation (\\%)"
        ylim = (0, 100)
    else:
        ylabel = metric
        ylim = (0, None)
    
    ax.set_ylabel(ylabel, fontsize=12)
    
    if ylim[1] is not None:
        ax.set_ylim(ylim)
    
    # Configure legend
    handles, labels = ax.get_legend_handles_labels()
    formatted_labels = format_legend_labels(labels)
    
    n_items = len(handles)
    ncol = 3 if n_items > 0 else 1
    
    ax.legend(
        handles, formatted_labels,
        title="LLM Model",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=ncol,
        fontsize=9
    )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved Lipophilicity plot to: {output_path}")
    
    plt.close(fig)


def format_legend_labels(labels: List[str]) -> List[str]:
    """Format legend labels for LLM variants."""
    formatted_labels = []
    for label in labels:
        parts = label.split()
        if len(parts) == 2:
            family = parts[0].upper() if "gpt" in parts[0].lower() else parts[0].capitalize()
            size = parts[1].lower()
            
            # Map size names
            if "gemini" in parts[0].lower():
                if size == "nano":
                    size = "Flash Lite"
                elif size == "mini":
                    size = "Flash"
                elif size == "normal":
                    size = "Pro"
            else:
                # OpenAI: Remove "Normal" suffix, capitalize others
                if size == "normal":
                    formatted_labels.append(family)
                    continue
                else:
                    size = size.capitalize()
            
            formatted_labels.append(f"{family} {size}")
        else:
            formatted_labels.append(label)
    
    return formatted_labels


# ============================================================================
# BARPLOT FUNCTIONS FOR APPROACHES
# ============================================================================

def create_qm7_approach_barplot(
    df: pd.DataFrame,
    metric: str,
    families_to_include: Optional[List[str]] = None,
    output_path: str = None,
    fig_size: Tuple[int, int] = (12, 6),
    title: str = None,
    n_samples: int = 1000
):
    """
    Create a bar plot for QM7 results comparing different approaches.
    
    Args:
        df: DataFrame with results
        metric: Which metric to plot
        families_to_include: List of family names to include
        output_path: Path to save the plot
        fig_size: Figure size
        title: Optional title
        n_samples: Number of training samples to filter for
    """
    plot_df = df.copy()
    
    # Filter for names_only input type
    plot_df = plot_df[plot_df["input_data_type"] == "names_only"]
    
    # Filter for specific training sample size
    plot_df = plot_df[plot_df["number_of_training_samples"] == n_samples]
    
    # Define approaches and their order
    target_approaches = [
        "with_preanalysis",
        "wp_solubility_blind",
        "wp_molproperty_clear",
        "wp_molproperty_blind",
        "wp_sampleproperty_clear",
        "wp_sampleproperty_blind"
    ]
    
    # Filter for target approaches
    plot_df = plot_df[plot_df["approach"].isin(target_approaches)]
    
    if plot_df.empty:
        print(f"Warning: No data for QM7 approaches barplot (Metric: {metric})")
        return
    
    # Filter by family if specified
    if families_to_include is not None:
        plot_df = plot_df[plot_df["llm_family"].isin(families_to_include)]
    
    if plot_df.empty:
        print(f"Warning: No data for families {families_to_include} in QM7 approaches barplot")
        return
    
    # Add labels
    plot_df["approach_label"] = plot_df.apply(create_approach_label, axis=1)
    plot_df["llm_variant"] = plot_df["llm_family"] + " " + plot_df["llm_size"]
    
    # X-axis order
    x_order = [create_approach_label(pd.Series({"approach": a})) for a in target_approaches]
    
    # Hue order
    color_palette = get_llm_color_palette()
    families = families_to_include if families_to_include else ["gpt-4.1", "gpt-5", "gemini-2.5"]
    sizes = ["nano", "mini", "normal"]
    hue_order = []
    palette = {}
    
    for family in families:
        for size in sizes:
            variant_name = f"{family} {size}"
            hue_order.append(variant_name)
            if (family, size) in color_palette:
                palette[variant_name] = color_palette[(family, size)]
    
    # Filter hue_order
    existing_variants = set(plot_df["llm_variant"].unique())
    hue_order = [h for h in hue_order if h in existing_variants]
    palette = {k: v for k, v in palette.items() if k in existing_variants}
    
    # Categorical ordering
    plot_df["approach_label"] = pd.Categorical(
        plot_df["approach_label"], categories=x_order, ordered=True
    )
    plot_df["llm_variant"] = pd.Categorical(
        plot_df["llm_variant"], categories=hue_order, ordered=True
    )
    
    # Create plot
    fig, ax = plt.subplots(figsize=fig_size)
    
    sns.barplot(
        data=plot_df,
        x="approach_label",
        y=metric,
        hue="llm_variant",
        order=x_order,
        hue_order=hue_order,
        palette=palette,
        ax=ax
    )
    
    # Annotate
    for patch in ax.patches:
        height = patch.get_height()
        if height is not None and not np.isnan(height) and height > 0:
            ax.annotate(
                f"{height:.0f}",
                (patch.get_x() + patch.get_width() / 2.0, height),
                ha='center', va='center',
                xytext=(0, 8), textcoords='offset points',
                fontsize=8
            )
    
    # Labels
    ax.set_xlabel("Approach (Names Only)", fontsize=12)
    
    if metric == "MAE":
        ylabel = "Mean Absolute Error (eV)"
        ylim = (0, None)
    elif metric == "RMSE":
        ylabel = "Root Mean Squared Error (eV)"
        ylim = (0, None)
    elif metric == "Correlation":
        ylabel = "Correlation (\\%)"
        ylim = (0, 100)
    else:
        ylabel = metric
        ylim = (0, None)
    
    ax.set_ylabel(ylabel, fontsize=12)
    
    if ylim[1] is not None:
        ax.set_ylim(ylim)
    
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    formatted_labels = format_legend_labels(labels)
    
    n_items = len(handles)
    ncol = 3 if n_items > 0 else 1
    
    ax.legend(
        handles, formatted_labels,
        title="LLM Model",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=ncol,
        fontsize=9
    )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved QM7 approach plot to: {output_path}")
    
    plt.close(fig)


def create_combined_3x3_plot(
    qm7_df, lipo_df, delaney_df, metric, output_path
):
    """
    Creates a 3x3 grid of plots:
    Rows: QM7, Lipophilicity, Delaney
    Cols: GPT-4.1, GPT-5, Gemini-2.5
    """
    fig, axes = plt.subplots(3, 3, figsize=(8, 5.2), sharex=False, sharey=False)
    
    datasets = [
        ("QM7", qm7_df),
        ("Lipophilicity", lipo_df),
        ("Delaney", delaney_df)
    ]
    
    families = ["gpt-4.1", "gpt-5", "gemini-2.5"]
    
    x_order = ["0", "60", "1000"]
    color_palette = get_llm_color_palette()
    
    # Process legend handled manually
    legend_handles = []
    legend_labels = []
    
    for row_idx, (dataset_name, df) in enumerate(datasets):
        # Filter df to "names_only" or similar compatible types if not done already
        # QM7 and Lipo in this script already filtered in plotting functions usually, but here we pass raw df
        # We need to filter properly.
        plot_df = df.copy()
        if "llm_variant" not in plot_df.columns:
            plot_df["llm_variant"] = plot_df["llm_family"] + " " + plot_df["llm_size"]
        if "input_data_type" in plot_df.columns:
             # Include Zero-shot (sample=0) or Names only
             plot_df = plot_df[
                 (plot_df["number_of_training_samples"] == 0) | 
                 (plot_df["input_data_type"] == "names_only")
             ]
        
        # Also filter approaches
        if "approach" in plot_df.columns:
            plot_df = plot_df[plot_df["approach"].isin(["with_preanalysis", "NoReasoning"])]
            
        for col_idx, family in enumerate(families):
            ax = axes[row_idx, col_idx]
            
            # Filter family
            fam_df = plot_df[plot_df["llm_family"] == family]
            
            # Filter valid samples
            fam_df = fam_df[fam_df["number_of_training_samples"].isin([0, 60, 1000])]
            
            if fam_df.empty:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                continue
                
            # Hue order setup for this family
            sizes = ["nano", "mini", "normal"]
            hue_order = []
            palette = {}
            for size in sizes:
                variant_name = f"{family} {size}"
                hue_order.append(variant_name)
                if (family, size) in color_palette:
                    palette[variant_name] = color_palette[(family, size)]
            existing_variants = set(fam_df["llm_variant"].unique())
            hue_order = [h for h in hue_order if h in existing_variants]
            filtered_palette = {k: v for k, v in palette.items() if k in existing_variants}
            
            # Plot
            sns.barplot(
                data=fam_df,
                x="input_variant",
                y=metric,
                hue="llm_variant",
                order=x_order,
                hue_order=hue_order,
                palette=filtered_palette,
                ax=ax
            )
            
            if "correlation" in metric.lower():
                ax.set_ylim(0, 110)
            
            # Annotate
            for patch in ax.patches:
                height = patch.get_height()
                if height is not None and not np.isnan(height) and height > 0:
                    ax.annotate(
                        f"{height:.0f}",
                        (patch.get_x() + patch.get_width() / 2.0, height),
                        ha='center', va='center',
                        xytext=(0, 5), textcoords='offset points',
                        fontsize=7
                    )
            
            # Formatting
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.get_legend().remove()
            
            # Row labels (Datasets) - on left of first col
            if col_idx == 0:
                ax.set_ylabel(f"{dataset_name}\n{metric}", fontsize=10, fontweight='bold')
                
            # Col labels (Families) - on top of first row
            if row_idx == 0:
                fam_title = family.upper().replace("GPT-", "GPT-").replace("GEMINI-", "Gemini-")
                ax.set_title(fam_title, fontsize=10, fontweight='bold')
            # col 2
            if row_idx == 2:
                ax.set_xlabel("Training samples", fontsize=12)
    
    # Collect handles for global legend from the last plot (assuming it has variants)
    # We need to reconstruct handles/labels manually to ensure we show all possible sizes across families?
    # Actually, user wants "like the legend in the all family plots".
    # Since we split by family in columns, the color coding is consistent (Family + Size).
    # But here we have different families in different columns. 
    # The hue is "llm_variant" which includes family name.
    # If we put a common legend, it should strictly speaking show all 9 items (3 families * 3 sizes).
    # However, since columns are separated by family, maybe we don't need family name in legend?
    # But the colors differ by family. So we should show all.
    
    # create dummy handles for all
    dummy_handles = []
    dummy_labels = []
    
    sizes_map = {"nano": "Nano/Flash Lite", "mini": "Mini/Flash", "normal": "Normal/Pro"}
    # Construct a legend that shows sizes? Or just keep it simple.
    # User said "like the legend in the all family plots".
    # In all family plots, we show "Family Size".
    
    all_variants_ordered = []
    for fam in families:
        for size in ["nano", "mini", "normal"]:
            all_variants_ordered.append(f"{fam} {size}")
            
    # create patches
    for var in all_variants_ordered:
        if (var.split()[0], var.split()[1]) in color_palette:
            color = color_palette[(var.split()[0], var.split()[1])]
            # Format label
            parts = var.split()
            fam_label = parts[0].upper() if "gpt" in parts[0].lower() else parts[0].capitalize()
            size_label = parts[1].capitalize()
            if "gemini" in parts[0].lower():
                if parts[1] == "nano": size_label = "Flash Lite"
                elif parts[1] == "mini": size_label = "Flash"
                elif parts[1] == "normal": size_label = "Pro"
            elif size_label == "Normal":
                 # OpenAI normal usually just Family name? But here we need to distinguish size?
                 # Actually in previous plots "Normal" suffix was removed for OpenAI.
                 pass 
            
            label_text = f"{fam_label} {size_label}"
            if "Normal" in size_label and "GPT" in fam_label:
                label_text = f"{fam_label}"

            import matplotlib.patches as mpatches
            patch = mpatches.Patch(color=color, label=label_text)
            dummy_handles.append(patch)
            dummy_labels.append(label_text)

    # Add single legend at top
    fig.legend(
        dummy_handles, dummy_labels,
        loc='lower center', 
        bbox_to_anchor=(0.5, 0.98),
        ncol=3, 
        fontsize=9
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved 3x3 plot to: {output_path}")


def create_combined_approaches_3x3_plot(
    qm7_df, lipo_df, delaney_df, metric, output_path, n_samples=1000
):
    """
    Creates a 3x3 grid of plots for approaches:
    Rows: QM7, Lipophilicity, Delaney
    Cols: GPT-4.1, GPT-5, Gemini-2.5
    """
    fig, axes = plt.subplots(3, 3, figsize=(10, 6), sharex=True, sharey='row')
    
    datasets = [
        ("QM7", qm7_df),
        ("Lipophilicity", lipo_df),
        ("Delaney", delaney_df)
    ]
    
    families = ["gpt-4.1", "gpt-5", "gemini-2.5"]
    
    # Target approaches in order
    target_approaches = [
        "with_preanalysis",
        "wp_solubility_blind",
        "wp_molproperty_clear",
        "wp_molproperty_blind",
        "wp_sampleproperty_clear",
        "wp_sampleproperty_blind"
    ]
    
    # helper for labels
    def get_approach_label(raw):
        mapping = {
            "with_preanalysis": "Specific",
            "wp_solubility_blind": "Specific-Transf.",
            "wp_molproperty_clear": "Generic",
            "wp_molproperty_blind": "Generic-Transf.",
            "wp_sampleproperty_clear": "Agnostic",
            "wp_sampleproperty_blind": "Agnostic-Transf."
        }
        return mapping.get(raw, raw)

    x_order = [get_approach_label(a) for a in target_approaches]
    
    color_palette = get_llm_color_palette()
    
    for row_idx, (dataset_name, df) in enumerate(datasets):
        plot_df = df.copy()
        
        # Filter for n_samples and names_only
        plot_df = plot_df[plot_df["number_of_training_samples"] == n_samples]
        
        # Ensure we filter for input type if needed (assuming "names_only" is standard for approaches)
        if "input_data_type" in plot_df.columns:
             plot_df = plot_df[plot_df["input_data_type"] == "names_only"]

        # Filter approaches
        plot_df = plot_df[plot_df["approach"].isin(target_approaches)]
        
        # Add labels
        if "approach_label" not in plot_df.columns:
            plot_df["approach_label"] = plot_df["approach"].apply(get_approach_label)
        if "llm_variant" not in plot_df.columns:
            plot_df["llm_variant"] = plot_df["llm_family"] + " " + plot_df["llm_size"]
            
        for col_idx, family in enumerate(families):
            ax = axes[row_idx, col_idx]
            
            # Filter family
            fam_df = plot_df[plot_df["llm_family"] == family]
            
            if fam_df.empty:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                continue
                
            # Hue order setup for this family
            sizes = ["nano", "mini", "normal"]
            hue_order = []
            palette = {}
            for size in sizes:
                variant_name = f"{family} {size}"
                hue_order.append(variant_name)
                if (family, size) in color_palette:
                    palette[variant_name] = color_palette[(family, size)]
            existing_variants = set(fam_df["llm_variant"].unique())
            hue_order = [h for h in hue_order if h in existing_variants]
            filtered_palette = {k: v for k, v in palette.items() if k in existing_variants}
            
            # Categorical ordering
            fam_df["approach_label"] = pd.Categorical(
                fam_df["approach_label"], categories=x_order, ordered=True
            )
            
            # Plot
            sns.barplot(
                data=fam_df,
                x="approach_label",
                y=metric,
                hue="llm_variant",
                order=x_order,
                hue_order=hue_order,
                palette=filtered_palette,
                ax=ax
            )
            
            if "correlation" in metric.lower():
                ax.set_ylim(0, 110)
            
            # Annotate
            for patch in ax.patches:
                height = patch.get_height()
                if height is not None and not np.isnan(height) and height > 0:
                    ax.annotate(
                        f"{height:.0f}",
                        (patch.get_x() + patch.get_width() / 2.0, height),
                        ha='center', va='center',
                        xytext=(0, 5), textcoords='offset points',
                        fontsize=7
                    )
            
            # Formatting
            ax.set_xlabel("") # Clear default label
            
            # Set tick labels - needed for categorical alignment on shared axis?
            # With sharex=True, only the bottom row will show them.
            if row_idx == 2:
                ax.set_xticklabels(x_order, rotation=45, ha='right', fontsize=9)
            
            ax.set_ylabel("")
            ax.get_legend().remove()
            
            # Row labels
            if col_idx == 0:
                ax.set_ylabel(f"{dataset_name}\n{metric}", fontsize=10, fontweight='bold')
                
            # Col labels
            if row_idx == 0:
                fam_title = family.upper().replace("GPT-", "GPT-").replace("GEMINI-", "Gemini-")
                ax.set_title(fam_title, fontsize=10, fontweight='bold')
    
    # Legend construction (same as input variant plot)
    dummy_handles = []
    dummy_labels = []
    
    all_variants_ordered = []
    for fam in families:
        for size in ["nano", "mini", "normal"]:
            all_variants_ordered.append(f"{fam} {size}")
            
    for var in all_variants_ordered:
        if (var.split()[0], var.split()[1]) in color_palette:
            color = color_palette[(var.split()[0], var.split()[1])]
            parts = var.split()
            fam_label = parts[0].upper() if "gpt" in parts[0].lower() else parts[0].capitalize()
            size_label = parts[1].capitalize()
            if "gemini" in parts[0].lower():
                if parts[1] == "nano": size_label = "Flash Lite"
                elif parts[1] == "mini": size_label = "Flash"
                elif parts[1] == "normal": size_label = "Pro"
            elif size_label == "Normal" and "GPT" in fam_label:
                # OpenAI normalized label
                pass
            
            label_text = f"{fam_label} {size_label}"
            if "Normal" in size_label and "GPT" in fam_label:
                label_text = f"{fam_label}"
            
            import matplotlib.patches as mpatches
            patch = mpatches.Patch(color=color, label=label_text)
            dummy_handles.append(patch)
            dummy_labels.append(label_text)

    # Add single legend at top
    fig.legend(
        dummy_handles, dummy_labels,
        loc='lower center', 
        bbox_to_anchor=(0.5, 0.98),
        ncol=3, 
        fontsize=9
    )
    
    plt.tight_layout()
    # Adjust top to make room for legend
    plt.subplots_adjust(top=0.90)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved 3x3 approaches plot to: {output_path}")


def create_gemini_approaches_plot(
    qm7_df, lipo_df, delaney_df, metric, output_path, n_samples=1000
):
    """
    Creates a 1-column plot for Gemini 2.5 approaches:
    Rows: QM7, Lipophilicity, Delaney
    Col: Gemini-2.5
    """
    # Width approx 40-50% of 3x3 (which was 10), so ~4.5-5.
    fig, axes = plt.subplots(3, 1, figsize=(4, 5.5), sharex=True, sharey=False) # sharey=False because rows differ
    if hasattr(axes, 'flatten'):
         axes = axes.flatten()
    else:
         axes = [axes] # Should be array of 3 if 3 rows, 1 col? subplots(3,1) returns array of 3 Axes.
    
    datasets = [
        ("QM7", qm7_df),
        ("Lipophilicity", lipo_df),
        ("Delaney", delaney_df)
    ]
    
    family = "gemini-2.5"
    
    # Target approaches in order
    target_approaches = [
        "with_preanalysis",
        "wp_solubility_blind",
        "wp_molproperty_clear",
        "wp_molproperty_blind",
        "wp_sampleproperty_clear",
        "wp_sampleproperty_blind"
    ]
    
    # helper for labels - with line breaks for horizontal text
    def get_approach_label_multiline(raw):
        mapping = {
            "with_preanalysis": "Specific",
            "wp_solubility_blind": "Specific\nTransf.",
            "wp_molproperty_clear": "Generic",
            "wp_molproperty_blind": "Generic\nTransf.",
            "wp_sampleproperty_clear": "Agnostic",
            "wp_sampleproperty_blind": "Agnostic\nTransf."
        }
        return mapping.get(raw, raw)
        
    # Standard labels for data mapping
    def get_approach_label_standard(raw):
        mapping = {
            "with_preanalysis": "Specific",
            "wp_solubility_blind": "Specific-Transf.",
            "wp_molproperty_clear": "Generic",
            "wp_molproperty_blind": "Generic-Transf.",
            "wp_sampleproperty_clear": "Agnostic",
            "wp_sampleproperty_blind": "Agnostic-Transf."
        }
        return mapping.get(raw, raw)

    x_order = [get_approach_label_standard(a) for a in target_approaches]
    x_labels_display = [get_approach_label_multiline(a) for a in target_approaches]
    
    color_palette = get_llm_color_palette()
    
    for row_idx, (dataset_name, df) in enumerate(datasets):
        ax = axes[row_idx]
        plot_df = df.copy()
        
        # Filter filters
        plot_df = plot_df[plot_df["number_of_training_samples"] == n_samples]
        if "input_data_type" in plot_df.columns:
             plot_df = plot_df[plot_df["input_data_type"] == "names_only"]

        plot_df = plot_df[plot_df["approach"].isin(target_approaches)]
        plot_df["approach_label"] = plot_df["approach"].apply(get_approach_label_standard)
        
        # Filter family
        plot_df = plot_df[plot_df["llm_family"] == family]
        
        if plot_df.empty:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center')
            continue

        if "llm_variant" not in plot_df.columns:
            plot_df["llm_variant"] = plot_df["llm_family"] + " " + plot_df["llm_size"]
            
        # Hue order setup
        sizes = ["nano", "mini", "normal"]
        hue_order = []
        palette = {}
        for size in sizes:
            variant_name = f"{family} {size}"
            hue_order.append(variant_name)
            if (family, size) in color_palette:
                palette[variant_name] = color_palette[(family, size)]
        existing_variants = set(plot_df["llm_variant"].unique())
        hue_order = [h for h in hue_order if h in existing_variants]
        filtered_palette = {k: v for k, v in palette.items() if k in existing_variants}
        
        # Categorical ordering
        plot_df["approach_label"] = pd.Categorical(
            plot_df["approach_label"], categories=x_order, ordered=True
        )
        
        # Plot
        sns.barplot(
            data=plot_df,
            x="approach_label",
            y=metric,
            hue="llm_variant",
            order=x_order,
            hue_order=hue_order,
            palette=filtered_palette,
            ax=ax
        )
        
        if "correlation" in metric.lower():
            ax.set_ylim(0, 110)
        
        # Annotate
        for patch in ax.patches:
            height = patch.get_height()
            if height is not None and not np.isnan(height) and height > 0:
                ax.annotate(
                    f"{height:.0f}",
                    (patch.get_x() + patch.get_width() / 2.0, height),
                    ha='center', va='center',
                    xytext=(0, 5), textcoords='offset points',
                    fontsize=8
                )
        
        # Formatting
        ax.set_xlabel("")
        ax.set_ylabel(f"{dataset_name}\n{metric}", fontsize=10, fontweight='bold')
        ax.get_legend().remove()
        
        #if row_idx == 0:
        #    ax.set_title("Gemini 2.5", fontsize=10, fontweight='bold')
            
        # Set x-labels only on bottom
        if row_idx == 2:
            ax.set_xticklabels(x_labels_display, rotation=0, ha='center', fontsize=9)

    # Legend
    dummy_handles = []
    dummy_labels = []
    
    # We only have one family here: Gemini 2.5
    for size in ["nano", "mini", "normal"]:
        var = f"{family} {size}"
        if (family, size) in color_palette:
            color = color_palette[(family, size)]
            
            # Label
            size_label = size.capitalize()
            if "nano" in size: size_label = "Flash Lite"
            elif "mini" in size: size_label = "Flash"
            elif "normal" in size: size_label = "Pro"
            
            label_text = f"{size_label}" # Just size since family is in title/known
            # Wait, 3x3 had "Family Size". Here it's "Gemini 2.5" plot. Just size is enough? 
            # Or "Gemini 2.5 Flash Lite"? 
            # User said "everything the same". 
            # But the column header says "GEMINI-2.5".
            # The legend in 3x3 was shared for all families.
            # Here I'll put specific legend.
            label_text = f"Gemini 2.5 {size_label}"

            import matplotlib.patches as mpatches
            patch = mpatches.Patch(color=color, label=label_text)
            dummy_handles.append(patch)
            dummy_labels.append(label_text)

    # Add single legend at top
    fig.legend(
        dummy_handles, dummy_labels,
        loc='lower center', 
        bbox_to_anchor=(0.57, 0.9),
        ncol=1, 
        fontsize=9
    )
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    plt.close(fig)
    print(f"Saved Gemini approaches plot to: {output_path}")


def create_openai_approaches_plot(
    qm7_df, lipo_df, delaney_df, metric, output_path, n_samples=1000
):
    """
    Creates a 2-column plot for OpenAI approaches:
    Rows: QM7, Lipophilicity, Delaney
    Cols: GPT-4.1, GPT-5
    """
    # Width approx ~8-9 for 2 cols
    fig, axes = plt.subplots(3, 2, figsize=(8, 5.5), sharex=True, sharey='row')
    
    datasets = [
        ("QM7", qm7_df),
        ("Lipophilicity", lipo_df),
        ("Delaney", delaney_df)
    ]
    
    families = ["gpt-4.1", "gpt-5"]
    
    # Target approaches in order
    target_approaches = [
        "with_preanalysis",
        "wp_solubility_blind",
        "wp_molproperty_clear",
        "wp_molproperty_blind",
        "wp_sampleproperty_clear",
        "wp_sampleproperty_blind"
    ]
    
    # helper for labels - with line breaks for horizontal text
    def get_approach_label_multiline(raw):
        mapping = {
            "with_preanalysis": "Specific",
            "wp_solubility_blind": "Specific\nTransf.",
            "wp_molproperty_clear": "Generic",
            "wp_molproperty_blind": "Generic\nTransf.",
            "wp_sampleproperty_clear": "Agnostic",
            "wp_sampleproperty_blind": "Agnostic\nTransf."
        }
        return mapping.get(raw, raw)
        
    # Standard labels for data mapping
    def get_approach_label_standard(raw):
        mapping = {
            "with_preanalysis": "Specific",
            "wp_solubility_blind": "Specific-Transf.",
            "wp_molproperty_clear": "Generic",
            "wp_molproperty_blind": "Generic-Transf.",
            "wp_sampleproperty_clear": "Agnostic",
            "wp_sampleproperty_blind": "Agnostic-Transf."
        }
        return mapping.get(raw, raw)

    x_order = [get_approach_label_standard(a) for a in target_approaches]
    x_labels_display = [get_approach_label_multiline(a) for a in target_approaches]
    
    color_palette = get_llm_color_palette()
    
    for row_idx, (dataset_name, df) in enumerate(datasets):
        plot_df_base = df.copy()
        
        # Filter filters
        plot_df_base = plot_df_base[plot_df_base["number_of_training_samples"] == n_samples]
        if "input_data_type" in plot_df_base.columns:
             plot_df_base = plot_df_base[plot_df_base["input_data_type"] == "names_only"]

        plot_df_base = plot_df_base[plot_df_base["approach"].isin(target_approaches)]
        plot_df_base["approach_label"] = plot_df_base["approach"].apply(get_approach_label_standard)
        
        for col_idx, family in enumerate(families):
            ax = axes[row_idx, col_idx]
            plot_df = plot_df_base[plot_df_base["llm_family"] == family].copy()
            
            if plot_df.empty:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                continue

            if "llm_variant" not in plot_df.columns:
                plot_df["llm_variant"] = plot_df["llm_family"] + " " + plot_df["llm_size"]
                
            # Hue order setup
            sizes = ["nano", "mini", "normal"]
            hue_order = []
            palette = {}
            for size in sizes:
                variant_name = f"{family} {size}"
                hue_order.append(variant_name)
                if (family, size) in color_palette:
                    palette[variant_name] = color_palette[(family, size)]
            existing_variants = set(plot_df["llm_variant"].unique())
            hue_order = [h for h in hue_order if h in existing_variants]
            filtered_palette = {k: v for k, v in palette.items() if k in existing_variants}
            
            # Categorical ordering
            plot_df["approach_label"] = pd.Categorical(
                plot_df["approach_label"], categories=x_order, ordered=True
            )
            
            # Plot
            sns.barplot(
                data=plot_df,
                x="approach_label",
                y=metric,
                hue="llm_variant",
                order=x_order,
                hue_order=hue_order,
                palette=filtered_palette,
                ax=ax
            )
            
            if "correlation" in metric.lower():
                ax.set_ylim(0, 110)
            
            # Annotate
            for patch in ax.patches:
                height = patch.get_height()
                if height is not None and not np.isnan(height) and height > 0:
                    ax.annotate(
                        f"{height:.0f}",
                        (patch.get_x() + patch.get_width() / 2.0, height),
                        ha='center', va='center',
                        xytext=(0, 5), textcoords='offset points',
                        fontsize=8
                    )
            
            # Formatting
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.get_legend().remove()

            # Row labels
            if col_idx == 0:
                ax.set_ylabel(f"{dataset_name}\n{metric}", fontsize=10, fontweight='bold')
                
            # Col labels
            if row_idx == 0:
                fam_title = family.upper().replace("GPT-", "GPT-").replace("GEMINI-", "Gemini-")
                ax.set_title(fam_title, fontsize=10, fontweight='bold')
                
            # Set x-labels only on bottom
            if row_idx == 2:
                ax.set_xticklabels(x_labels_display, rotation=0, ha='center', fontsize=9)

    # Legend
    dummy_handles = []
    dummy_labels = []
    
    # We have two families here. Show all sizes for both? 
    # Or common sizes "Nano", "Mini", "Normal"?
    # The legend in 3x3 was combined.
    # Here, let's list them all to be clear.
    
    all_variants_ordered = []
    for fam in families:
        for size in ["nano", "mini", "normal"]:
            all_variants_ordered.append(f"{fam} {size}")

    for var in all_variants_ordered:
        if (var.split()[0], var.split()[1]) in color_palette:
            color = color_palette[(var.split()[0], var.split()[1])]
            
            # Label
            parts = var.split()
            fam_label = parts[0].upper().replace("GPT-", "GPT-")
            size_label = parts[1].capitalize()
            
            label_text = f"{fam_label} {size_label}"
            if "Normal" in size_label:
                label_text = f"{fam_label}"
                
            import matplotlib.patches as mpatches
            patch = mpatches.Patch(color=color, label=label_text)
            dummy_handles.append(patch)
            dummy_labels.append(label_text)

    # Add single legend at top
    fig.legend(
        dummy_handles, dummy_labels,
        loc='lower center', 
        bbox_to_anchor=(0.53, 0.98),
        ncol=2, 
        fontsize=9
    )
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved OpenAI approaches plot to: {output_path}")


def create_lipophilicity_approach_barplot(
    df: pd.DataFrame,
    metric: str,
    families_to_include: Optional[List[str]] = None,
    output_path: str = None,
    fig_size: Tuple[int, int] = (12, 6),
    title: str = None,
    n_samples: int = 1000
):
    """
    Create a bar plot for Lipophilicity results comparing different approaches.
    
    Args:
        df: DataFrame with results
        metric: Which metric to plot
        families_to_include: List of family names to include
        output_path: Path to save the plot
        fig_size: Figure size
        title: Optional title
        n_samples: Number of training samples to filter for
    """
    plot_df = df.copy()
    
    # Filter for names_only input type
    plot_df = plot_df[plot_df["input_data_type"] == "names_only"]
    
    # Filter for specific training sample size
    plot_df = plot_df[plot_df["number_of_training_samples"] == n_samples]
    
    # Define approaches and their order
    target_approaches = [
        "with_preanalysis",
        "wp_solubility_blind",
        "wp_molproperty_clear",
        "wp_molproperty_blind",
        "wp_sampleproperty_clear",
        "wp_sampleproperty_blind"
    ]
    
    # Filter for target approaches
    plot_df = plot_df[plot_df["approach"].isin(target_approaches)]
    
    if plot_df.empty:
        print(f"Warning: No data for Lipophilicity approaches barplot (Metric: {metric})")
        return
    
    # Filter by family if specified
    if families_to_include is not None:
        plot_df = plot_df[plot_df["llm_family"].isin(families_to_include)]
    
    if plot_df.empty:
        print(f"Warning: No data for families {families_to_include} in Lipophilicity approaches barplot")
        return
    
    # Add labels
    plot_df["approach_label"] = plot_df.apply(create_approach_label, axis=1)
    plot_df["llm_variant"] = plot_df["llm_family"] + " " + plot_df["llm_size"]
    
    # X-axis order
    x_order = [create_approach_label(pd.Series({"approach": a})) for a in target_approaches]
    
    # Hue order
    color_palette = get_llm_color_palette()
    families = families_to_include if families_to_include else ["gpt-4.1", "gpt-5", "gemini-2.5"]
    sizes = ["nano", "mini", "normal"]
    hue_order = []
    palette = {}
    
    for family in families:
        for size in sizes:
            variant_name = f"{family} {size}"
            hue_order.append(variant_name)
            if (family, size) in color_palette:
                palette[variant_name] = color_palette[(family, size)]
    
    # Filter hue_order
    existing_variants = set(plot_df["llm_variant"].unique())
    hue_order = [h for h in hue_order if h in existing_variants]
    palette = {k: v for k, v in palette.items() if k in existing_variants}
    
    # Categorical ordering
    plot_df["approach_label"] = pd.Categorical(
        plot_df["approach_label"], categories=x_order, ordered=True
    )
    plot_df["llm_variant"] = pd.Categorical(
        plot_df["llm_variant"], categories=hue_order, ordered=True
    )
    
    # Create plot
    fig, ax = plt.subplots(figsize=fig_size)
    
    sns.barplot(
        data=plot_df,
        x="approach_label",
        y=metric,
        hue="llm_variant",
        order=x_order,
        hue_order=hue_order,
        palette=palette,
        ax=ax
    )
    
    # Annotate
    for patch in ax.patches:
        height = patch.get_height()
        if height is not None and not np.isnan(height) and height > 0:
            ax.annotate(
                f"{height:.0f}",
                (patch.get_x() + patch.get_width() / 2.0, height),
                ha='center', va='center',
                xytext=(0, 8), textcoords='offset points',
                fontsize=8
            )
    
    # Labels
    ax.set_xlabel("Approach (Names Only)", fontsize=12)
    
    if metric == "MAE":
        ylabel = "Mean Absolute Error (logD)"
        ylim = (0, None)
    elif metric == "RMSE":
        ylabel = "Root Mean Squared Error (logD)"
        ylim = (0, None)
    elif metric == "Correlation":
        ylabel = "Correlation (\\%)"
        ylim = (0, 105)
    else:
        ylabel = metric
        ylim = (0, None)
    
    ax.set_ylabel(ylabel, fontsize=12)
    
    if ylim[1] is not None:
        ax.set_ylim(ylim)
    
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    formatted_labels = format_legend_labels(labels)
    
    n_items = len(handles)
    ncol = 3 if n_items > 0 else 1
    
    ax.legend(
        handles, formatted_labels,
        title="LLM Model",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=ncol,
        fontsize=9
    )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved Lipophilicity approach plot to: {output_path}")
    
    plt.close(fig)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "../results")
    data_dir = os.path.join(script_dir, "../data")
    output_folder = os.path.join(script_dir, "../figures")
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Define LLM families and metrics
    llm_families = ["gpt-4.1", "gpt-5", "gemini-2.5"]
    metrics_to_plot = ["MAE", "RMSE", "Correlation"]
    
    # ========================================================================
    # QM7 PROCESSING
    # ========================================================================
    
    print("\n" + "="*80)
    print("PROCESSING QM7 RESULTS")
    print("="*80)
    
    qm7_results_file = os.path.join(results_dir, "LLM_Results_qm7.json")
    qm7_results_file_2 = os.path.join(results_dir, "LLM_Results_qm7_zeroshot.json")
    qm7_data_file = os.path.join(data_dir, "qm7.csv")
    
    if os.path.exists(qm7_results_file) and os.path.exists(qm7_data_file):
        print(f"\nLoading QM7 results from: {qm7_results_file}")
        qm7_results_df = evaluate_dataset_results(
            qm7_results_file, qm7_data_file, "qm7"
        )
        
        print(f"\nLoading QM7 results from: {qm7_results_file_2}")
        qm7_results_df_2 = evaluate_dataset_results(
            qm7_results_file_2, qm7_data_file, "qm7"
        )
        # concat the two dataframes
        qm7_results_df = pd.concat([qm7_results_df, qm7_results_df_2], ignore_index=True)
        
        if not qm7_results_df.empty:
            print(f"QM7 DataFrame shape: {qm7_results_df.shape}")
            
            # Add derived columns
            qm7_results_df["llm_family"] = qm7_results_df["llm"].apply(extract_llm_family)
            qm7_results_df["llm_size"] = qm7_results_df["llm"].apply(extract_llm_size)
            qm7_results_df["input_variant"] = qm7_results_df.apply(create_input_variant_label, axis=1)
            
            # Ensure numeric training samples
            qm7_results_df["number_of_training_samples"] = pd.to_numeric(
                qm7_results_df["number_of_training_samples"], errors='coerce'
            ).fillna(0).astype(int)
            
            print(f"LLM Families: {qm7_results_df['llm_family'].unique()}")
            print(f"Training sample sizes: {sorted(qm7_results_df['number_of_training_samples'].unique())}")
            
            # Create input type barplots
            print("\nGenerating QM7 Input Type Barplots...")
            qm7_barplot_folder = os.path.join(output_folder, "qm7_barplots")
            os.makedirs(qm7_barplot_folder, exist_ok=True)
            
            for family in llm_families:
                for metric_name in metrics_to_plot:
                    output_file = os.path.join(
                        qm7_barplot_folder,
                        f"qm7_{family.replace('.', '')}_{metric_name.lower()}{file_type}"
                    )
                    create_qm7_barplot(
                        qm7_results_df,
                        metric=metric_name,
                        families_to_include=[family],
                        output_path=output_file,
                        fig_size=(4.6, 2.8)
                    )
            
            # Combined families
            for metric_name in metrics_to_plot:
                output_file = os.path.join(
                    qm7_barplot_folder,
                    f"qm7_all_families_{metric_name.lower()}{file_type}"
                )
                existing_families = [
                    f for f in llm_families 
                    if f in qm7_results_df["llm_family"].values
                ]
                create_qm7_barplot(
                    qm7_results_df,
                    metric=metric_name,
                    families_to_include=existing_families,
                    output_path=output_file,
                    fig_size=(14, 7)
                )
            
            # Create approach barplots
            print("\nGenerating QM7 Approach Barplots...")
            qm7_approach_folder = os.path.join(output_folder, "qm7_approach_barplots")
            os.makedirs(qm7_approach_folder, exist_ok=True)
            
            for family in llm_families:
                for metric_name in metrics_to_plot:
                    output_file = os.path.join(
                        qm7_approach_folder,
                        f"approaches_{family.replace('.', '')}_{metric_name.lower()}{file_type}"
                    )
                    create_qm7_approach_barplot(
                        qm7_results_df,
                        metric=metric_name,
                        families_to_include=[family],
                        output_path=output_file,
                        fig_size=(3.5, 2.8),
                        n_samples=1000
                    )
            
            # Combined families
            for metric_name in metrics_to_plot:
                output_file = os.path.join(
                    qm7_approach_folder,
                    f"approaches_all_families_{metric_name.lower()}{file_type}"
                )
                existing_families = [
                    f for f in llm_families
                    if f in qm7_results_df["llm_family"].values
                ]
                create_qm7_approach_barplot(
                    qm7_results_df,
                    metric=metric_name,
                    families_to_include=existing_families,
                    output_path=output_file,
                    fig_size=(14, 7),
                    n_samples=1000
                )
            
            print(f"\nQM7 plots saved to: {qm7_barplot_folder} and {qm7_approach_folder}")
        else:
            print("Warning: No QM7 data available for plotting")
    else:
        print(f"Warning: QM7 files not found")
        if not os.path.exists(qm7_results_file):
            print(f"  Missing: {qm7_results_file}")
        if not os.path.exists(qm7_data_file):
            print(f"  Missing: {qm7_data_file}")
    
    # ========================================================================
    # LIPOPHILICITY PROCESSING
    # ========================================================================
    
    print("\n" + "="*80)
    print("PROCESSING LIPOPHILICITY RESULTS")
    print("="*80)
    
    lipo_results_file = os.path.join(results_dir, "LLM_Results_lipophilicity.json")
    lipo_results_file_2 = os.path.join(results_dir, "LLM_Results_lipophilicity_zeroshot.json")
    lipo_data_file = os.path.join(data_dir, "lipophilicity.csv")
    
    if os.path.exists(lipo_results_file) and os.path.exists(lipo_data_file):
        print(f"\nLoading Lipophilicity results from: {lipo_results_file}")
        lipo_results_df = evaluate_dataset_results(
            lipo_results_file, lipo_data_file, "lipophilicity"
        )
        
        print(f"\nLoading Lipophilicity results from: {lipo_results_file_2}")
        lipo_results_df_2 = evaluate_dataset_results(
            lipo_results_file_2, lipo_data_file, "lipophilicity"
        )
        
        # concat the two dataframes
        lipo_results_df = pd.concat([lipo_results_df, lipo_results_df_2], ignore_index=True)
        
        if not lipo_results_df.empty:
            print(f"Lipophilicity DataFrame shape: {lipo_results_df.shape}")
            
            # Add derived columns
            lipo_results_df["llm_family"] = lipo_results_df["llm"].apply(extract_llm_family)
            lipo_results_df["llm_size"] = lipo_results_df["llm"].apply(extract_llm_size)
            lipo_results_df["input_variant"] = lipo_results_df.apply(create_input_variant_label, axis=1)
            
            # Ensure numeric training samples
            lipo_results_df["number_of_training_samples"] = pd.to_numeric(
                lipo_results_df["number_of_training_samples"], errors='coerce'
            ).fillna(0).astype(int)
            
            print(f"LLM Families: {lipo_results_df['llm_family'].unique()}")
            print(f"Training sample sizes: {sorted(lipo_results_df['number_of_training_samples'].unique())}")
            
            # Create input type barplots
            print("\nGenerating Lipophilicity Input Type Barplots...")
            lipo_barplot_folder = os.path.join(output_folder, "lipophilicity_barplots")
            os.makedirs(lipo_barplot_folder, exist_ok=True)
            
            for family in llm_families:
                for metric_name in metrics_to_plot:
                    output_file = os.path.join(
                        lipo_barplot_folder,
                        f"lipophilicity_{family.replace('.', '')}_{metric_name.lower()}{file_type}"
                    )
                    create_lipophilicity_barplot(
                        lipo_results_df,
                        metric=metric_name,
                        families_to_include=[family],
                        output_path=output_file,
                        fig_size=(4.6, 2.8)
                    )
            
            # Combined families
            for metric_name in metrics_to_plot:
                output_file = os.path.join(
                    lipo_barplot_folder,
                    f"lipophilicity_all_families_{metric_name.lower()}{file_type}"
                )
                existing_families = [
                    f for f in llm_families
                    if f in lipo_results_df["llm_family"].values
                ]
                create_lipophilicity_barplot(
                    lipo_results_df,
                    metric=metric_name,
                    families_to_include=existing_families,
                    output_path=output_file,
                    fig_size=(14, 7)
                )
            
            # Create approach barplots
            print("\nGenerating Lipophilicity Approach Barplots...")
            lipo_approach_folder = os.path.join(output_folder, "lipophilicity_approach_barplots")
            os.makedirs(lipo_approach_folder, exist_ok=True)
            
            for family in llm_families:
                for metric_name in metrics_to_plot:
                    output_file = os.path.join(
                        lipo_approach_folder,
                        f"approaches_{family.replace('.', '')}_{metric_name.lower()}{file_type}"
                    )
                    create_lipophilicity_approach_barplot(
                        lipo_results_df,
                        metric=metric_name,
                        families_to_include=[family],
                        output_path=output_file,
                        fig_size=(4.6, 2.8),
                        n_samples=1000
                    )
            
            # Combined families
            for metric_name in metrics_to_plot:
                output_file = os.path.join(
                    lipo_approach_folder,
                    f"approaches_all_families_{metric_name.lower()}{file_type}"
                )
                existing_families = [
                    f for f in llm_families
                    if f in lipo_results_df["llm_family"].values
                ]
                create_lipophilicity_approach_barplot(
                    lipo_results_df,
                    metric=metric_name,
                    families_to_include=existing_families,
                    output_path=output_file,
                    fig_size=(14, 7),
                    n_samples=1000
                )
            
            print(f"\nLipophilicity plots saved to: {lipo_barplot_folder} and {lipo_approach_folder}")
        else:
            print("Warning: No Lipophilicity data available for plotting")
    else:
        print(f"Warning: Lipophilicity files not found")
        if not os.path.exists(lipo_results_file):
            print(f"  Missing: {lipo_results_file}")
        if not os.path.exists(lipo_data_file):
            print(f"  Missing: {lipo_data_file}")
    
    
    # Combined 3x3 plot
    results_dir = os.path.join(script_dir, "../results")
    data_dir = os.path.join(script_dir, "../data")
    
    # Consolidated Delaney results files
    file_names = ["LLM_Results_delaney.json", "LLM_Results_delaney_zeroshot.json"]
    delaney_results_files = [os.path.join(results_dir, file_name) for file_name in file_names]
    delaney_data_file = os.path.join(data_dir, "delaney-processed.csv")
    delaney_df = pd.DataFrame()
    
    if os.path.exists(delaney_data_file):
        for r_file in delaney_results_files:
            print(f"\nLoading Delaney results from: {r_file}")
            df_part = evaluate_dataset_results(r_file, delaney_data_file, "delaney")
            if not df_part.empty:
                delaney_df = pd.concat([delaney_df, df_part], ignore_index=True)
            else:
                print(f"Warning: No Delaney results found in {r_file}")

    if not delaney_df.empty:
            delaney_df["llm_family"] = delaney_df["llm"].apply(extract_llm_family)
            delaney_df["llm_size"] = delaney_df["llm"].apply(extract_llm_size)
            delaney_df["input_variant"] = delaney_df.apply(create_input_variant_label, axis=1)
            delaney_df["llm_variant"] = delaney_df["llm_family"] + " " + delaney_df["llm_size"]
            delaney_df["number_of_training_samples"] = pd.to_numeric(delaney_df["number_of_training_samples"], errors='coerce').fillna(0).astype(int)
            # Filter delaney to "names_only" or "Smiles" for consistency and input_variant labeling
            delaney_df = delaney_df[delaney_df["input_data_type"].isin(["names_only", "Smiles"])]

            
    if not qm7_results_df.empty and not lipo_results_df.empty and not delaney_df.empty:
        print("\nGenerating Combined 3x3 Plot...")
        for metric_name in metrics_to_plot:
            output_file_3x3 = os.path.join(output_folder, f"combined_3x3_{metric_name.lower()}{file_type}")
            create_combined_3x3_plot(
                qm7_results_df,
                lipo_results_df,
                delaney_df,
                metric=metric_name,
                output_path=output_file_3x3
            )
            
            output_file_approaches_3x3 = os.path.join(output_folder, f"combined_approaches_3x3_{metric_name.lower()}{file_type}")
            create_combined_approaches_3x3_plot(
                qm7_results_df,
                lipo_results_df,
                delaney_df,
                metric=metric_name,
                output_path=output_file_approaches_3x3,
                n_samples=1000
            )
            
            output_file_gemini = os.path.join(output_folder, f"gemini_approaches_{metric_name.lower()}{file_type}")
            create_gemini_approaches_plot(
                qm7_results_df,
                lipo_results_df,
                delaney_df,
                metric=metric_name,
                output_path=output_file_gemini,
                n_samples=1000
            )

            output_file_openai = os.path.join(output_folder, f"openai_approaches_{metric_name.lower()}{file_type}")
            create_openai_approaches_plot(
                qm7_results_df,
                lipo_results_df,
                delaney_df,
                metric=metric_name,
                output_path=output_file_openai,
                n_samples=1000
            )
    else:
        print("\nWarning: Need QM7, Lipophilicity, and Delaney data to create combined 3x3 plot.")


    print("\n" + "="*80)
    print("PLOTTING COMPLETE")
    print("="*80)
