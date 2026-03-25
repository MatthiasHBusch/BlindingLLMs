import json
from typing import Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import os
from copy import deepcopy
from sklearn.linear_model import LinearRegression
import matplotlib.patches as mpatches

# for optimizing the plots for the paper
file_type = ".png"
# for paper
#file_type = ".svg"
#file_type = ".eps"

# switch on latex rendering
plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.serif": ["Times New Roman"]})

ColorDict = Dict[str, str]
LabelMapper = Union[Dict[str, str], Callable[[str], str]]


def add_bar_height_labels(ax):
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')

def bar_config(ax):
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xlabel(ax.get_xlabel(), fontsize=14)
    ax.set_ylabel(ax.get_ylabel(), fontsize=14)
    ax.legend(fontsize=12)


def evaluate_results(results_file: str, data_file: str):
    """
    Evaluates predictions from a JSON results file against true values in a CSV data file.

    Args:
        results_file (str): Path to the JSON results file.
        data_file (str): Path to the CSV data file.

    Returns:
        pd.DataFrame: A DataFrame containing the evaluation metrics (MAE, RMSE, Correlation).
    """
    # Read the results file
    with open(results_file, "r") as f:
        # make sure to allow NaN values when loading JSON
        results = json.load(f, strict=False)

    # Read the data file
    data = pd.read_csv(data_file)
    
    # data processing needed to find the right samples (names are not unique)
    possible_alloys = ["AZ31", "AZ91", "WE43", "AlPowder"]
    # results file name contains the alloy name, we need to find the right alloy
    alloy_name = None
    for alloy in possible_alloys:
        if alloy in results_file:
            alloy_name = alloy
            break
    if alloy_name is None:
        raise ValueError("No alloy name found in results file name. Please check the file name.")
    base_material = "Mg"
    if alloy_name == "AlPowder":
        alloy_name = "powder"
        base_material = "Al"
    # Filter the data for the right alloy
    data = data[(data["Alloy"] == alloy_name) & (data["BaseMaterial"] == base_material)]

    # Group the data by experimental conditions to find the largest group with consistent experimental conditions. Experimental conditions are Method, AggressiveComponent and Operating_Concentration_mM
    grouped_data = data.groupby(["Method", "AggressiveComponent", "Operating_Concentration_mM"]).size().reset_index(name='counts')
    # Find the group with the largest count
    largest_group = grouped_data.loc[grouped_data['counts'].idxmax()]
    
    # data should now be assigned only to the largest group
    data = data[(data["Method"] == largest_group["Method"]) & 
                (data["AggressiveComponent"] == largest_group["AggressiveComponent"]) & 
                (data["Operating_Concentration_mM"] == largest_group["Operating_Concentration_mM"])]

    # Ensure required columns exist in the data file
    if not {"IUPAC", "IE"}.issubset(data.columns):
        raise ValueError("Data file must contain columns 'IUPAC' and 'IE'")

    # Prepare a DataFrame to collect results
    results_df = pd.DataFrame(columns=["input_data_type", "approach", "llm", "number_of_training_samples", "MAE", "RMSE", "Correlation"])

    # Iterate over combinations of input_data_type, approach, llm and number_of_training_samples
    length_results = None
    for input_data_type, approaches in results.items():
        for approach, llms in approaches.items():
            for llm, training_set_sizes in llms.items():
                for number_of_training_samples, predictions_dict in training_set_sizes.items():
                    # Extract the molecule names and predictions
                    molecule_names = list(predictions_dict.keys())

                    # Remove None values and check consistency
                    for molecule_name in molecule_names:
                        if length_results is None:
                            length_results = len(predictions_dict[molecule_name])
                        #elif len(predictions_dict[molecule_name]) != length_results:
                            #print(f"Warning: Length of predictions for input_data_type={input_data_type}, "
                            #      f"approach={approach}, llm={llm}, molecule={molecule_name} is inconsistent. "
                            #      f"{len(predictions_dict[molecule_name])} != {length_results}")

                        # check if entries are strings and convert them to floats if so. If NaN or None or >100, remove them
                        predictions_dict[molecule_name] = [float(p) if isinstance(p, str) else p for p in predictions_dict[molecule_name]]
                        predictions_dict[molecule_name] = [p for p in predictions_dict[molecule_name] if (p is not None and p<100 and not np.isnan(p))]

                        # find molecules where the prediction array is empty and put a 0 in there
                        if len(predictions_dict[molecule_name]) == 0:
                            predictions_dict[molecule_name] = [0.0]

                    # Compute mean predictions
                    predictions = [np.mean(predictions_dict[name]) for name in molecule_names]

                    # Match predictions with true inhibition efficiencies
                    true_values = [
                        float(data.loc[data["IUPAC"] == name, "IE"].iloc[0])
                        for name in molecule_names if name in data["IUPAC"].values
                    ]
                    matched_predictions = [
                        float(predictions[i]) for i, name in enumerate(molecule_names) if name in data["IUPAC"].values
                    ]

                    # Skip if no matches are found
                    if not true_values:
                        print(f"No matches found for input_data_type={input_data_type}, approach={approach}, llm={llm}.")
                        continue

                    # Calculate metrics
                    #print(f"Calculating metrics for input_data_type={input_data_type}, approach={approach}, llm={llm}")
                    #print(true_values)
                    #print(matched_predictions)
                    if len(true_values) != len(matched_predictions):
                        print(f"Warning: Length of true values ({len(true_values)}) does not match length of predictions ({len(matched_predictions)}) for input_data_type={input_data_type}, approach={approach}, llm={llm}.")
                        continue
                    mae = np.mean(np.abs(np.array(true_values) - np.array(matched_predictions)))
                    rmse = np.sqrt(np.mean((np.array(true_values) - np.array(matched_predictions)) ** 2))
                    correlation = pearsonr(true_values, matched_predictions)[0]  # Pearson correlation coefficient
                    p_values = pearsonr(true_values, matched_predictions)[1]  # p-value

                    # Append the results
                    results_df.loc[len(results_df)] = {
                        "input_data_type": input_data_type,
                        "approach": approach,
                        "llm": llm,
                        "number_of_training_samples": number_of_training_samples,
                        "MAE": mae,
                        "RMSE": rmse,
                        "Correlation": correlation*100,
                        "p-value": p_values
                    }

    return results_df

def delaney_results_processor(results_df, predictions_dict, data, smiles_col, value_col, length_results, approach, llm, number_of_training_samples, number_of_direct_training_samples, input_data_type):
    # match molecules (smiles strings) with data file and calculate metrics
    molecule_names = list(predictions_dict.keys())
    # Remove None values and check consistency
    for molecule_name in molecule_names:
        if length_results is None:
            length_results = len(predictions_dict[molecule_name])
        #elif len(predictions_dict[molecule_name]) != length_results:
            #print(f"Warning: Length of predictions for approach={approach}, molecule={molecule_name} is inconsistent. "
            #      f"{len(predictions_dict[molecule_name])} != {length_results}")
        # check if entries are strings and convert them to floats if so. If NaN or None, remove them
        predictions_dict[molecule_name] = [float(p) if isinstance(p, str) else p for p in predictions_dict[molecule_name]]
        if "blind" not in approach:
            predictions_dict[molecule_name] = [p for p in predictions_dict[molecule_name] if (p is not None and not np.isnan(p))]# and p<30)]
        else:
            predictions_dict[molecule_name] = [p for p in predictions_dict[molecule_name] if (p is not None and not np.isnan(p))]
        # remove molecules which have no entry left, ie where the list is empty
        if len(predictions_dict[molecule_name]) == 0:
            predictions_dict.pop(molecule_name)
            molecule_names.remove(molecule_name)
    # Compute mean predictions
    predictions = [np.mean(predictions_dict[name]) for name in molecule_names]
    # Match predictions with true values and calculate metrics
    true_values = [
        float(data.loc[data[smiles_col] == name, value_col].iloc[0])
        for name in molecule_names if name in data[smiles_col].values
    ]
    matched_predictions = [predictions[i] for i, name in enumerate(molecule_names) if name in data[smiles_col].values]
    # Skip if no matches are found
    if not true_values:
        print(f"No matches found for approach={approach}.")
        return
    # Calculate metrics
    if len(true_values) != len(matched_predictions):
        print(f"Warning: Length of true values ({len(true_values)}) does not match length of predictions ({len(matched_predictions)}) for approach={approach}.")
        return
    mae = np.mean(np.abs(np.array(true_values) - np.array(matched_predictions)))
    rmse = np.sqrt(np.mean((np.array(true_values) - np.array(matched_predictions)) ** 2))
    # check for nans and infs separately and print warning if found
    if np.isnan(true_values).any():
        print(f"Warning: NaNs found in true values for approach={approach}.")
    if np.isinf(true_values).any():
        print(f"Warning: Infs found in true values for approach={approach}.")
    if np.isnan(matched_predictions).any():
        print(f"Warning: NaNs found in predictions for approach={approach}.")
    if np.isinf(matched_predictions).any():
        print(f"Warning: Infs found in predictions for approach={approach}.")
    correlation = pearsonr(true_values, matched_predictions)[0]  # Pearson correlation coefficient
    p_values = pearsonr(true_values, matched_predictions)[1]  # p-value

    # Append results to the DataFrame
    results_df.loc[len(results_df)] = {
        "input_data_type": input_data_type,
        "approach": approach,
        "llm": llm,
        "number_of_training_samples": int(number_of_training_samples)+int(number_of_direct_training_samples),
        "MAE": mae,
        "RMSE": rmse,
        "Correlation": correlation*100,
        "p-value": p_values
    }

def evaluate_results_delaney(results_file: str, data_file: str):
    """
    Evaluates predictions from a JSON results file against true values in a CSV data file.

    Args:
        results_file (str): Path to the JSON results file.
        data_file (str): Path to the CSV data file.
    Returns:
        pd.DataFrame: A DataFrame containing the evaluation metrics (MAE, RMSE, Correlation).
    """
    # Read the results file
    with open(results_file, "r") as f:
        # make sure to allow NaN values when loading JSON
        results = json.load(f, strict=False)

    # Read the data file
    data = pd.read_csv(data_file)
    # Ensure required columns exist in the data file
    smiles_col = "Compound ID"
    value_col = "measured log solubility in mols per litre"
    value_col_transformed = "transformed_solubility"
    if not {smiles_col, value_col, value_col_transformed}.issubset(data.columns):
        raise ValueError(f"Data file must contain columns '{smiles_col}', '{value_col}' and '{value_col_transformed}'")
    # Prepare a DataFrame to collect results
    results_df = pd.DataFrame(columns=["input_data_type", "approach", "llm", "number_of_training_samples", "MAE", "RMSE", "Correlation", "p-value"])
    # Iterate over combinations of input_data_type, approach, llm and number_of_training_samples. When results structure is different and values are met earlier, use standard key values: input_data_type="smiles", approach="Zeroshot", llm="GPT-4.1", number_of_training_samples="0"
    length_results = None
    for input_data_type, approaches in results.items():
        for approach, llms in approaches.items():
            value_col_ = value_col
            if "blind" in approach:
                value_col_ = value_col_transformed

            for llm, training_set_sizes in llms.items():
                if isinstance(list(training_set_sizes.values())[0], list): # then already in the sample results with 0-shot
                    delaney_results_processor(results_df, training_set_sizes, data, smiles_col, value_col_, length_results, approach, llm, "0", "0", input_data_type)
                else:
                    for number_of_training_samples, predictions_dict in training_set_sizes.items():
                        # if first key is a string that can be converted to an int, it is a list of predictions
                        if isinstance(list(predictions_dict.keys())[0], str) and list(predictions_dict.keys())[0].isdigit():
                            for number_of_direct_training_samples, predictions_dict_ in predictions_dict.items():
                                delaney_results_processor(results_df, predictions_dict_, data, smiles_col, value_col_, length_results, approach, llm, number_of_training_samples, number_of_direct_training_samples, input_data_type)
                        else:
                            delaney_results_processor(results_df, predictions_dict, data, smiles_col, value_col_, length_results, approach, llm, number_of_training_samples, "0", input_data_type)
                

    return results_df

def evaluate_results_chemprop(results_file: str, data_file: str):
    """
    Evaluates predictions from a JSON results file against true values in a CSV data file.

    Args:
        results_file (str): Path to the JSON results file.
        data_file (str): Path to the CSV data file.

    Returns:
        pd.DataFrame: A DataFrame containing the evaluation metrics (MAE, RMSE, Correlation).
    """
    # Read the results file
    with open(results_file, "r") as f:
        # make sure to allow NaN values when loading JSON
        results = json.load(f, strict=False)

    # Read the data file
    data = pd.read_csv(data_file)
    
    # data processing needed to find the right samples (names are not unique)
    possible_alloys = ["AZ31", "AZ91", "WE43", "AlPowder"]
    # results file name contains the alloy name, we need to find the right alloy
    alloy_name = None
    for alloy in possible_alloys:
        if alloy in results_file:
            alloy_name = alloy
            break
    if alloy_name is None:
        raise ValueError("No alloy name found in results file name. Please check the file name.")
    base_material = "Mg"
    if alloy_name == "AlPowder":
        alloy_name = "powder"
        base_material = "Al"
    # Filter the data for the right alloy
    data = data[(data["Alloy"] == alloy_name) & (data["BaseMaterial"] == base_material)]

    # Group the data by experimental conditions to find the largest group with consistent experimental conditions. Experimental conditions are Method, AggressiveComponent and Operating_Concentration_mM
    grouped_data = data.groupby(["Method", "AggressiveComponent", "Operating_Concentration_mM"]).size().reset_index(name='counts')
    # Find the group with the largest count
    largest_group = grouped_data.loc[grouped_data['counts'].idxmax()]
    
    # data should now be assigned only to the largest group
    data = data[(data["Method"] == largest_group["Method"]) & 
                (data["AggressiveComponent"] == largest_group["AggressiveComponent"]) & 
                (data["Operating_Concentration_mM"] == largest_group["Operating_Concentration_mM"])]

    # Ensure required columns exist in the data file
    if not {"IUPAC", "IE"}.issubset(data.columns):
        raise ValueError("Data file must contain columns 'IUPAC' and 'IE'")

    # Prepare a DataFrame to collect results
    results_df = pd.DataFrame(columns=["input_data_type", "approach", "setting", "number_of_training_samples", "MAE", "RMSE", "Correlation"])

    # Iterate over combinations of input_data_type, approach, llm and number_of_training_samples
    length_results = None
    for input_data_type, settings in results.items():
            approach = "Chemprop"
            for setting, training_set_sizes in settings.items():
                for number_of_training_samples, predictions_dict in training_set_sizes.items():
                    # Extract the molecule names and predictions
                    molecule_names = list(predictions_dict.keys())

                    # Remove None values and check consistency
                    for molecule_name in molecule_names:
                        if length_results is None:
                            length_results = len(predictions_dict[molecule_name])
                        elif len(predictions_dict[molecule_name]) != length_results:
                            print(f"Warning: Length of predictions for input_data_type={input_data_type}, "
                                  f"approach={approach}, molecule={molecule_name} is inconsistent. "
                                  f"{len(predictions_dict[molecule_name])} != {length_results}")

                        # check if entries are strings and convert them to floats if so. If NaN or None or >100, remove them
                        predictions_dict[molecule_name] = [float(p) if isinstance(p, str) else p for p in predictions_dict[molecule_name]]
                        predictions_dict[molecule_name] = [p for p in predictions_dict[molecule_name] if (p is not None and p<100 and not np.isnan(p))]

                        # find molecules where the prediction array is empty and put a 0 in there
                        if len(predictions_dict[molecule_name]) == 0:
                            predictions_dict[molecule_name] = [0.0]

                    # Compute mean predictions
                    predictions = [np.mean(predictions_dict[name]) for name in molecule_names]

                    # Match predictions with true inhibition efficiencies
                    true_values = [
                        float(data.loc[data["IUPAC"] == name, "IE"].iloc[0])
                        for name in molecule_names if name in data["IUPAC"].values
                    ]
                    matched_predictions = [
                        float(predictions[i]) for i, name in enumerate(molecule_names) if name in data["IUPAC"].values
                    ]

                    # Skip if no matches are found
                    if not true_values:
                        print(f"No matches found for input_data_type={input_data_type}, approach={approach}.")
                        continue

                    # Calculate metrics
                    #print(f"Calculating metrics for input_data_type={input_data_type}, approach={approach}")
                    #print(true_values)
                    #print(matched_predictions)
                    if len(true_values) != len(matched_predictions):
                        print(f"Warning: Length of true values ({len(true_values)}) does not match length of predictions ({len(matched_predictions)}) for input_data_type={input_data_type}, approach={approach}.")
                        continue
                    mae = np.mean(np.abs(np.array(true_values) - np.array(matched_predictions)))
                    rmse = np.sqrt(np.mean((np.array(true_values) - np.array(matched_predictions)) ** 2))
                    correlation = pearsonr(true_values, matched_predictions)[0]  # Pearson correlation coefficient
                    p_values = pearsonr(true_values, matched_predictions)[1]  # p-value

                    # Append the results
                    results_df.loc[len(results_df)] = {
                        "input_data_type": input_data_type,
                        "approach": approach,
                        "setting": setting,
                        "number_of_training_samples": number_of_training_samples,
                        "MAE": mae,
                        "RMSE": rmse,
                        "Correlation": correlation*100,
                        "p-value": p_values
                    }

    return results_df

def save_individual_mae_plots(
    results_df,
    input_data_type,
    llm,
    metric,
    output_folder,
    custom_legend_labels=None,
    custom_x_labels=None,
    plot_height=5,
    llm_order=None,
    approach_order=None
):
    """
    Creates and saves a bar chart for MAE of the specified input data type and llm to the specified folder.

    Args:
        results_df (pd.DataFrame): DataFrame containing results with MAE, approaches, and LLMs.
        input_data_type (str): input data type to plot.
        llm (str): LLM to plot.
        output_folder (str): Folder where plots should be saved.
        custom_legend_labels (list): Custom legend labels for LLMs (e.g., ["Model A", "Model B"]).
        custom_x_labels (dict): Custom x-axis labels for approaches (e.g., {"approach1": "Method A"}).
        plot_height (float): Height of the plot (default: 6).
    """
    results_df = deepcopy(results_df)
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Helper function to plot
    def create_plot(data, input_data_type, file_name, llm_order=None, approach_order=None):
        # Apply custom x-axis labels if provided
        if custom_x_labels:
            data["approach"] = data["approach"].replace(custom_x_labels)
        
        if custom_legend_labels:
            data["llm"] = data["llm"].replace(custom_legend_labels)
        
        # Set custom order for LLMs
        if llm_order:
            data["llm"] = pd.Categorical(data["llm"], categories=llm_order, ordered=True)

        # Set custom order for approaches
        if approach_order:
            data["approach"] = pd.Categorical(data["approach"], categories=approach_order, ordered=True)



        # Plot settings
        fig, ax = plt.subplots(figsize=(8, plot_height))
        sns.barplot(
            data=data,
            x="approach",
            y=metric,
            hue="llm",
            ax=ax,
            palette="viridis"
        )

        
        # Titles and labels
        baseline_result = 0.0
        baseline_label = "Baseline MLP"
        #ax.set_title(f"MAE for {input_data_type}", pad=20)
        if metric == "MAE":
            ax.set_ylabel("Mean Absolute Error / IE")
            baseline_result = 61.3
        elif metric == "Correlation":
            ax.set_ylabel("Correlation / %")
            baseline_result = 60.0
        elif metric == "RMSE":
            ax.set_ylabel("Root Mean Squared Error / IE")
            baseline_result = 73.0
        else:
            ax.set_ylabel(metric)
        ax.set_xlabel("Prompting approach")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center", va="top")
        # round to 1 decimal place
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(f"{p.get_height():.1f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=9)
        ax.set_ylim(0, 140)
        # For "Correlation", set specific y-ticks
        if metric == "Correlation":
            ax.set_yticks([0, 20, 40, 60, 80, 100])
            
        ax.axhline(y=baseline_result, color='r', linestyle='--', label=baseline_label)
        
        bar_config(ax)
        
        
        # Manually set legend labels if provided
        if custom_legend_labels:
            handles, labels = ax.get_legend_handles_labels()
            #new_labels = custom_legend_labels
            ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.6875, 0.975), ncol=2)
        else:
            ax.legend(title="LLM", loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=2)


        # Save plot
        output_path = os.path.join(output_folder, file_name)
        title=""
        if input_data_type == "names_and_descriptors":
            title += "With descriptors" 
        else:
            title += "Without descriptors"
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)
        print(f"Plot saved to: {output_path}")

    # Filter data for each input data type and create plots
    data1 = results_df[results_df["input_data_type"] == input_data_type]

    create_plot(data1, input_data_type, f"{metric}_{input_data_type}{file_type}", llm_order=llm_order, approach_order=approach_order)


def save_boxplots_from_results(
    results_file: str,
    data_file: str,
    combos: list,
    output_folder: str,
    sort_by_true: bool = True,
    show_iupac_labels: bool = False,
    figsize=(12, 6),
    file_prefix: str = "boxplot",
    rotate_xticks: int = 45,
    label_name: str = "IE",
    chemical_identifier: str = "IUPAC",
    results_nos_from_the_top: range = range(0, 75)
):
    """
    Create and save boxplots for a list of specified combinations.

    Args:
        results_file: path to the JSON results file (same format used elsewhere in this script).
        data_file: path to the CSV with true values (must contain 'chemical_identifier' and 'IE').
        combos: list of dicts specifying combinations to plot. Each dict should contain:
            - 'keys': list of keys to drill into the results JSON, in order.
            Optional: 'title' and 'color'
        output_folder: folder to save plots.
        sort_by_true: whether to sort samples by experimental IE values (default True).
        show_iupac_labels: whether to show chemical_identifier names on x-axis (default False).
        figsize: figure size tuple.
        file_prefix: prefix for saved filenames.
        rotate_xticks: rotation for x tick labels when showing names.

    The function iterates through the combos list and saves one plot per combo.
    Each box shows the distribution of single predictions for a sample; the experimental
    IE is shown as a red diamond overlayed on the boxplot.
    """
    # load results JSON
    with open(results_file, 'r') as f:
        results = json.load(f, strict=False)

    # load data CSV
    data = pd.read_csv(data_file)

    # if corrosion results, filter data to the right alloy
    if "AZ31" in results_file or "AZ91" in results_file or "WE43" in results_file or "AlPowder" in results_file:
        # determine alloy from filename (reuse earlier logic)
        possible_alloys = ["AZ31", "AZ91", "WE43", "AlPowder"]
        alloy_name = None
        for alloy in possible_alloys:
            if alloy in results_file:
                alloy_name = alloy
                break
        if alloy_name is None:
            raise ValueError("No alloy name found in results file name. Please check the file name.")
        base_material = "Mg"
        if alloy_name == "AlPowder":
            alloy_name = "powder"
            base_material = "Al"

        # Filter data to the largest consistent experimental group (same as evaluate_results)
        data = data[(data["Alloy"] == alloy_name) & (data["BaseMaterial"] == base_material)]
        grouped_data = data.groupby(["Method", "AggressiveComponent", "Operating_Concentration_mM"]).size().reset_index(name='counts')
        largest_group = grouped_data.loc[grouped_data['counts'].idxmax()]
        data = data[(data["Method"] == largest_group["Method"]) & 
                    (data["AggressiveComponent"] == largest_group["AggressiveComponent"]) & 
                    (data["Operating_Concentration_mM"] == largest_group["Operating_Concentration_mM"])]
    else:
        # delaney, no filtering needed
        pass
            

    if not {chemical_identifier, label_name}.issubset(data.columns):
        raise ValueError(f"Data file must contain columns {chemical_identifier} and {label_name}")

    os.makedirs(output_folder, exist_ok=True)
    # copy results to avoid modifying original
    original_results = deepcopy(results)
    

    for combo in combos:
        results = deepcopy(original_results)
        keys = combo.get('keys', [])
        # aggregate keys to title
        title = keys[0]
        for key in keys[1:]:
            title += " | " + key

        # drill into results structure carefully
        for key in keys:
            if key not in results:
                print(f"Key '{key}' not found in {results_file}")
                # error, skip this combo
                break
            results = results[key]

        # Build lists of molecules that both have predictions and appear in the data
        molecule_names = [m for m in list(results.keys()) if m in data[chemical_identifier].values]
        if len(molecule_names) == 0:
            print(f"No matching molecules between predictions and data for combo: {title}")
            continue

        true_vals = []
        preds_per_sample = []
        iupac_labels = []

        for name in molecule_names:
            raw_preds = results[name]
            # convert string entries to float when possible
            cleaned = [float(p) if isinstance(p, str) else p for p in raw_preds]
            # remove None, NaN and unrealistic (>100)
            cleaned = [p for p in cleaned if (p is not None and not np.isnan(p) and p < 100)]
            # skip samples that have no valid predictions
            if len(cleaned) == 0:
                continue
            # get true value
            true_val = float(data.loc[data[chemical_identifier] == name, label_name].iloc[0])
            true_vals.append(true_val)
            preds_per_sample.append(cleaned)
            iupac_labels.append(name)

        if len(preds_per_sample) == 0:
            print(f"No valid predictions to plot for combo: {title}")
            continue
        
        if len(results_nos_from_the_top) < len(true_vals):
            true_vals = [true_vals[i] for i in results_nos_from_the_top]
            preds_per_sample = [preds_per_sample[i] for i in results_nos_from_the_top]
            iupac_labels = [iupac_labels[i] for i in results_nos_from_the_top]

        # Sort by true values if requested
        if sort_by_true:
            order = np.argsort(true_vals)
            true_vals = [true_vals[i] for i in order]
            preds_per_sample = [preds_per_sample[i] for i in order]
            iupac_labels = [iupac_labels[i] for i in order]

        # Create boxplot
        fig, ax = plt.subplots(figsize=figsize)
        # matplotlib's boxplot can accept a list of lists
        bp = ax.boxplot(preds_per_sample, patch_artist=True, medianprops=dict(color='black'))

        # color boxes lightly
        for patch in bp['boxes']:
            patch.set_facecolor('#a6cee3')

        # overlay experimental true values
        x_positions = np.arange(1, len(true_vals) + 1)
        ax.scatter(x_positions, true_vals, color='red', marker='D', label='Experimental IE')

        ax.set_xlabel('Sample index (sorted by experimental IE)')
        ax.set_ylabel(label_name)
        ax.set_title(title)

        # x ticks: either numeric sample index or IUPAC labels
        if show_iupac_labels:
            ax.set_xticks(x_positions)
            ax.set_xticklabels(iupac_labels, rotation=rotate_xticks, ha='right', fontsize=8)
        else:
            ax.set_xticks(x_positions)
            ax.set_xticklabels([str(i) for i in range(1, len(true_vals) + 1)])

        ax.legend()
        plt.tight_layout()

        file_name = f"{file_prefix}"
        for key in keys:
            file_name += f"_{key}"
        file_name += f"{file_type}"
        output_path = os.path.join(output_folder, file_name)
        plt.savefig(output_path, dpi=300)
        plt.close(fig)
        print(f"Saved boxplot to: {output_path}")


# Example helper to create multiple boxplots from a results file
def save_boxplots_batch_example():
    """Small example (commented) showing how to call save_boxplots_from_results.
    Edit and run as needed.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # delaney example
    combos = [
        {
            'keys': ['Smiles', 'Zeroshot', 'gpt-4.1_2024-12-01-preview','0'],
            'title': 'Zeroshot_delaney_GPT-4.1'
        },
        {
            'keys': ['Smiles', 'Zeroshot', 'gpt-5_2024-12-01-preview','0'],
            'title': 'Zeroshot_delaney_GPT-5'
        }
    ]
    results_file = os.path.join(script_dir, "../results/LLM_Results_delaney.json")
    data_file = os.path.join(script_dir, "../data/delaney-processed.csv")
    save_boxplots_from_results(results_file, data_file, combos, output_folder, show_iupac_labels=False, file_prefix="boxplot_delaney", chemical_identifier="smiles", label_name="measured log solubility in mols per litre")



def annotate_metric_bars(ax, fmt="{:.0f}", text_offset=10, fontsize=9):
    """Annotate each bar in ``ax`` with its height using the provided format string."""
    for patch in ax.patches:
        height = patch.get_height()
        if height is None or np.isnan(height):
            continue
        ax.annotate(
            fmt.format(height),
            (patch.get_x() + patch.get_width() / 2.0, height),
            ha='center',
            va='center',
            xytext=(0, text_offset),
            textcoords='offset points',
            fontsize=fontsize
        )


def create_metric_barplot(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    hue: str,
    output_path: str,
    colors: Optional[ColorDict] = None,
    hue_order: Optional[List[str]] = None,
    x_order: Optional[List[str]] = None,
    sort_by: Optional[List[str]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend_title: Optional[str] = None,
    legend_loc: str = "upper right",
    legend_ncol: int = 1,
    legend_bbox_to_anchor: Optional[Tuple[float, float]] = None,
    legend_kwargs: Optional[Dict[str, object]] = None,
    label_mapper: Optional[LabelMapper] = None,
    fig_size: Tuple[int, int] = (10, 6),
    dpi: int = 300,
    annotate: bool = True,
    annotate_fmt: str = "{:.0f}",
    ylim: Optional[Tuple[float, float]] = None,
    y_ticks: Optional[List[float]] = None,
    x_tick_rotation: float = 0,
    tight_layout: bool = True,
    x_as_int: bool = False
):
    """Generic helper for the different bar plots produced in this script."""

    df = data.copy()

    if x_as_int:
        with np.errstate(invalid='ignore'):
            df[x] = pd.to_numeric(df[x], errors='coerce')

    if sort_by:
        df = df.sort_values(sort_by, kind="mergesort")

    if x_order is not None:
        df[x] = pd.Categorical(df[x], categories=x_order, ordered=True)

    if hue_order is not None:
        df[hue] = pd.Categorical(df[hue], categories=hue_order, ordered=True)

    palette = colors
    if colors is not None and hue_order is not None:
        missing = [level for level in hue_order if level not in colors]
        if missing:
            raise ValueError(f"Missing colors for hue levels: {missing}")

    fig, ax = plt.subplots(figsize=fig_size)
    sns.barplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        order=x_order,
        hue_order=hue_order,
        palette=palette,
        ax=ax
    )

    if annotate:
        annotate_metric_bars(ax, fmt=annotate_fmt)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if ylim is not None:
        ax.set_ylim(*ylim)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)

    if x_tick_rotation:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=x_tick_rotation)

    handles, labels = ax.get_legend_handles_labels()

    if label_mapper is not None:
        def _map(label: str) -> str:
            if callable(label_mapper):
                return label_mapper(label)
            return label_mapper.get(label, label)
        labels = [_map(label) for label in labels]

    legend_params = legend_kwargs.copy() if legend_kwargs else {}
    if legend_title is not None:
        legend_params.setdefault("title", legend_title)
    if legend_loc:
        legend_params.setdefault("loc", legend_loc)
    legend_params.setdefault("ncol", legend_ncol)
    if legend_bbox_to_anchor is not None:
        legend_params.setdefault("bbox_to_anchor", legend_bbox_to_anchor)

    ax.legend(handles, labels, **legend_params)

    if tight_layout:
        plt.tight_layout()

    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)



script_dir = os.path.dirname(os.path.abspath(__file__))


# Save individual MAE plots
output_folder = os.path.join(script_dir, "../PaperICMLGraphics")
if file_type == ".eps":
    output_folder = os.path.join(output_folder, "eps")
if file_type == ".svg":
    output_folder = os.path.join(output_folder, "svg")
    
# check if folder exists, if not create it
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define custom labels

custom_legend_labels = {
    "Meta-Llama-3.1-405B-Instruct_2024-05-01-preview": "Meta Llama 3.1 405B", 
    "gpt-35-turbo-16k_2024-10-01-preview": "OpenAI GPT-3.5", 
    "gpt-4o_2024-10-01-preview": "OpenAI GPT-4o", 
    "o1-preview_2024-10-01-preview": "OpenAI o1",
    "o3-mini_2025-01-01-preview": "OpenAI o3-mini"
}

custom_x_labels = {
    "input_output_prompting": "Basic few-shot",
    "gpt_generated_prompts": "ChatGPT generated",
    "refined_prompting": "Manually refined",
    "with_preanalysis": "With pre-analysis"
}

#metric = "Correlation" # "MAE" or "Correlation" or "RMSE"
metric = "MAE"
#metric = "RMSE"

# Results files
#####################################################
# "Exact" setting
alloy ="AlPowder"

# Base results file (without context learning, includes all finetuned models)
file_exact_base = os.path.join(script_dir, "../results/LLM_Results_exact_{}.json".format(alloy))

# Full context learning results file (minimized text overhead)
#file_exact_context_minimal = os.path.join(script_dir, "../results/LLM_Results_exact_context_minimal_{}.json".format(alloy))

# Context learning with functional groups added
file_exact_context_funcs_minimal = os.path.join(script_dir, "../results/LLM_Results_exact_context_funcs_minimal_{}.json".format(alloy))


#####################################################
# "Close" setting
alloys =["AZ31", "AZ91", "WE43"]

# Base results file (without context learning, includes all finetuned models)
files_close_base = []
for alloy in alloys:
    files_close_base.append(os.path.join(script_dir, "../results/LLM_Results_close_{}.json".format(alloy)))

# Full context learning results file (minimized text overhead)
files_close_context_minimal = []
for alloy in alloys:
    files_close_context_minimal.append(os.path.join(script_dir, "../results/LLM_Results_close_context_minimal_{}.json".format(alloy)))

# Context learning with functional groups added
files_close_context_funcs_minimal = []
for alloy in alloys:
    files_close_context_funcs_minimal.append(os.path.join(script_dir, "../results/LLM_Results_close_context_funcs_minimal_{}.json".format(alloy)))

files_close_context_funcs_minimal_anonymized = []
for alloy in alloys:
    files_close_context_funcs_minimal_anonymized.append(os.path.join(script_dir, "../results/LLM_Results_close_context_funcs_minimal_anonymized_{}.json".format(alloy)))


###############################################
# "Far" setting
alloys =["AZ31", "AZ91", "WE43"]

# Context learning with functional groups added
files_far_context_funcs_minimal = []
for alloy in alloys:
    files_far_context_funcs_minimal.append(os.path.join(script_dir, "../results/LLM_Results_far_context_funcs_minimal_{}.json".format(alloy)))
    
# "All" setting
alloys =["AZ31", "AZ91", "WE43"]

# Context learning with functional groups added
files_all_context_funcs_minimal = []
for alloy in alloys:
    files_all_context_funcs_minimal.append(os.path.join(script_dir, "../results/LLM_Results_all_context_funcs_minimal_{}.json".format(alloy)))


###############################################
# chemprop results file
alloys =["AZ31", "AZ91", "WE43"]
files_chemprop = []
for alloy in alloys:
    files_chemprop.append(os.path.join(script_dir, "../results/Chemprop_Results_{}.json".format(alloy)))



# create list with all results files
results_files = []
results_files.append(file_exact_base)
#results_files.append(file_exact_context_minimal)
results_files.append(file_exact_context_funcs_minimal)

results_files.extend(files_close_base)
results_files.extend(files_close_context_minimal)
results_files.extend(files_close_context_funcs_minimal)
results_files.extend(files_close_context_funcs_minimal_anonymized)


results_files.extend(files_far_context_funcs_minimal)

results_files.extend(files_all_context_funcs_minimal)

#results_files.extend(files_chemprop) chemprop results are handled separately

# Data files
data_file = os.path.join(script_dir, "../data/ExCorrDatasetClean.csv")
data_file_delaney = os.path.join(script_dir, "../data/delaney-processed.csv")

# Basic evaluation for all results files corrosion (skip if data not available)
all_results_df = pd.DataFrame()
all_results_df_complete = pd.DataFrame()
chemprop_data = pd.DataFrame()

if os.path.exists(data_file):
    for results_file in results_files:
        if not os.path.exists(results_file):
            print(f"Skipping missing corrosion results: {results_file}")
            continue
        results_df = evaluate_results(results_file, data_file)
        results_df["results_file"] = os.path.basename(results_file)
        all_results_df = pd.concat([all_results_df, results_df], ignore_index=True)

    # Create a dataframe with all results
    columns_needed_llms = ["input_data_type", "approach", "setting", "context_learning_type", "llm", "number_of_training_samples", "alloy", "MAE", "RMSE", "Correlation"]

    all_results_df_complete = all_results_df.copy()
    all_results_df_complete["setting"] = all_results_df_complete["results_file"].apply(
        lambda x: "exact" if "exact" in x else ("Close" if "close" in x else ("far" if "far" in x else ("all" if "all" in x else "unknown")))
    )
    def _get_context_learning_type_from_filename(filename: str) -> str:
        if "context_funcs_minimal_anonymized" in filename:
            return "context_funcs_minimal_anonymized"
        elif "context_funcs_minimal" in filename:
            return "context_funcs_minimal"
        elif "context_minimal" in filename:
            return "context_minimal"
        else:
            return "no_context"

    all_results_df_complete["context_learning_type"] = all_results_df_complete["results_file"].apply(_get_context_learning_type_from_filename)
    def _get_alloy_from_filename(filename: str) -> str:
        for alloy in alloys + ["AlPowder"]:
            if alloy in filename:
                return alloy
        return "unknown"

    all_results_df_complete["alloy"] = all_results_df_complete["results_file"].apply(_get_alloy_from_filename)
    all_results_df_complete = all_results_df_complete[columns_needed_llms]

    for i in range(3):
        if os.path.exists(files_chemprop[i]):
            results_df = evaluate_results_chemprop(files_chemprop[i], data_file)
            results_df["alloy"] = alloys[i]
            results_df["llm"] = "Chemprop"
            results_df["context_learning_type"] = "Chemprop"
            chemprop_data = pd.concat([chemprop_data, results_df], ignore_index=True)

    save_boxplots_batch_example()
else:
    print("Corrosion data (ExCorrDatasetClean.csv) not found, skipping corrosion evaluation.")

# Basic evaluation for delaney results
delaney_results_file = os.path.join(script_dir, "../results/LLM_Results_delaney.json")

ylabel = "Mean Absolute Error / IE"
if metric == "Correlation":
    ylabel = "Correlation / $\\%$"
elif metric == "RMSE":
    ylabel = "Root Mean Squared Error / IE"

# calculate delaney results
delaney_results_df = evaluate_results_delaney(delaney_results_file, data_file_delaney)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
# rows 1:5
print(delaney_results_df.iloc[0:5])
# rows 6:10
print(delaney_results_df.iloc[5:10])
# rows 11:15
print(delaney_results_df.iloc[10:15])
# rows 16:20
print(delaney_results_df.iloc[15:20])
# rows 21:25
print(delaney_results_df.iloc[20:25])

# calculate delaney zeroshot results
delaney_zeroshot_file = os.path.join(script_dir, "../results/LLM_Results_delaney_zeroshot.json")
delaney_zeroshot_df = None
if os.path.exists(delaney_zeroshot_file):
    try:
        delaney_zeroshot_df = evaluate_results_delaney(delaney_zeroshot_file, data_file_delaney)
        print(delaney_zeroshot_df.iloc[0:5] if not delaney_zeroshot_df.empty else "Empty DataFrame")
    except Exception as e:
        print(f"Warning: Could not load {delaney_zeroshot_file}: {e}")


###############################################
# DELANEY BAR PLOTS - ICML Paper
###############################################

def concatenate_delaney_dataframes(*dfs):
    """
    Concatenate multiple dataframes, handling empty dataframes gracefully.
    
    Args:
        *dfs: Variable number of DataFrames to concatenate
        
    Returns:
        pd.DataFrame: Concatenated DataFrame, or empty DataFrame if all inputs are empty
    """
    valid_dfs = []
    for df in dfs:
        if df is not None and not df.empty:
            valid_dfs.append(df)
    
    if not valid_dfs:
        print("Warning: All input dataframes are empty or None")
        return pd.DataFrame()
    
    return pd.concat(valid_dfs, ignore_index=True)


def extract_llm_family(llm_name: str) -> str:
    """Extract LLM family from full LLM name (e.g., 'gpt-4.1-mini_...' -> 'gpt-4.1')"""
    if "gemini" in llm_name.lower():
        return "gemini-2.5"
    elif "gpt-5" in llm_name.lower():
        return "gpt-5"
    elif "gpt-4.1" in llm_name.lower():
        return "gpt-4.1"
    else:
        return "unknown"


def extract_llm_size(llm_name: str) -> str:
    """Extract LLM size from full LLM name.
    
    For GPT models: nano, mini, or normal (no suffix)
    For Gemini models: flash-lite=nano, flash=mini, pro=normal
    """
    llm_lower = llm_name.lower()
    
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
    if "-nano" in llm_lower:
        return "nano"
    elif "-mini" in llm_lower:
        return "mini"
    else:
        return "normal"


def create_input_variant_label(row: pd.Series) -> str:
    """
    Create a combined label for the 5 input variants:
    1. Zero-shot
    2. Names only, 60 samples
    3. Names & descriptors, 60 samples
    4. Names only, 1000 samples
    5. Names & descriptors, 1000 samples
    """
    n_samples = int(row["number_of_training_samples"])
    input_type = row["input_data_type"]
    
    if n_samples == 0:
        return "Zero-shot"
    
    # Determine input type label
    if "descriptors" in input_type.lower() or input_type == "names_and_descriptors":
        input_label = "Names \\& Desc."
    else:
        input_label = "Names only"
    
    # Create combined label
    return f"{input_label}\n{n_samples}"


def get_llm_color_palette():
    """
    Create color palette with family base colors and brightness variants for sizes.
    
    Returns:
        dict: Mapping from (family, size) tuple to hex color
    """
    # Base colors for each family (using distinct hues)
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


def create_delaney_barplot(
    df: pd.DataFrame,
    metric: str,
    families_to_include: Optional[List[str]] = None,
    output_path: str = None,
    fig_size: Tuple[int, int] = (12, 6),
    title: str = None
):
    """
    Create a bar plot for Delaney results with LLM families and sizes.
    
    Args:
        df: DataFrame with columns: input_data_type, number_of_training_samples, llm, 
            llm_family, llm_size, input_variant, and metric columns
        metric: Which metric to plot ('MAE', 'RMSE', or 'Correlation')
        families_to_include: List of family names to include, or None for all
        output_path: Path to save the plot
        fig_size: Figure size tuple
        title: Optional title for the plot
    """
    plot_df = df.copy()
    
    # Filter to only include "with_preanalysis" approach
    if "approach" in plot_df.columns:
        plot_df = plot_df[(plot_df["approach"] == "with_preanalysis") | (plot_df["approach"] == "NoReasoning")]
    
    # Filter by family if specified
    if families_to_include is not None:
        plot_df = plot_df[plot_df["llm_family"].isin(families_to_include)]
    
    if plot_df.empty:
        print(f"Warning: No data for families {families_to_include}")
        return
    
    # Create combined hue column for family + size
    plot_df["llm_variant"] = plot_df["llm_family"] + " " + plot_df["llm_size"]
    
    # Define the order for x-axis (input variants)
    x_order = [
        "Zero-shot",
        "Names only\n60",
        "Names \\& Desc.\n60",
        "Names only\n1000",
        "Names \\& Desc.\n1000"
    ]
    
    # Filter to only include relevant sample sizes
    valid_samples = [0, 60, 1000]
    plot_df = plot_df[plot_df["number_of_training_samples"].isin(valid_samples)]
    
    # Get color palette
    color_palette = get_llm_color_palette()
    
    # Create hue order based on families to include
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
    
    # Filter hue_order to only include variants that exist in data
    existing_variants = set(plot_df["llm_variant"].unique())
    hue_order = [h for h in hue_order if h in existing_variants]
    palette = {k: v for k, v in palette.items() if k in existing_variants}
    
    if not hue_order:
        print(f"Warning: No valid LLM variants found in data")
        return
    
    # Set categorical order
    plot_df["input_variant"] = pd.Categorical(plot_df["input_variant"], categories=x_order, ordered=True)
    plot_df["llm_variant"] = pd.Categorical(plot_df["llm_variant"], categories=hue_order, ordered=True)
    
    # Create the plot
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
    
    # Annotate bars with values
    for patch in ax.patches:
        height = patch.get_height()
        if height is not None and not np.isnan(height) and height > 0:
            ax.annotate(
                f"{height:.0f}",
                (patch.get_x() + patch.get_width() / 2.0, height),
                ha='center',
                va='center',
                xytext=(0, 8),
                textcoords='offset points',
                fontsize=8
            )
    
    # Set labels
    ax.set_xlabel("Input Configuration", fontsize=12)
    
    if metric == "MAE":
        ylabel = "Mean Absolute Error (log mol/L)"
        ylim = (0, None)
    elif metric == "RMSE":
        ylabel = "Root Mean Squared Error (log mol/L)"
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
    
    # Format legend labels for readability
    formatted_labels = []
    for label in labels:
        # Convert "gpt-4.1 nano" to "GPT-4.1 Nano" etc.
        parts = label.split()
        if len(parts) == 2:
            family = parts[0].upper() if "gpt" in parts[0].lower() else parts[0].capitalize()
            size = parts[1].lower()
            
            # Map size names appropriately
            if "gemini" in parts[0].lower():
                # Gemini: nano → Flash Lite, mini → Flash, normal → Pro
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
    
    # Determine legend position based on number of items
    n_items = len(handles)
    if n_items <= 3:
        ncol = 3
    elif n_items <= 6:
        ncol = 3
    else:
        ncol = 3
    
    ax.legend(
        handles, 
        formatted_labels,
        title="LLM Model",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=ncol,
        fontsize=9
    )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    
    plt.close(fig)


# Concatenate all Delaney dataframes
all_delaney_dfs = []

# Add main delaney results (now contains all ICL data in one file)
if 'delaney_results_df' in dir() and delaney_results_df is not None and not delaney_results_df.empty:
    all_delaney_dfs.append(delaney_results_df)

# Add zeroshot results
if 'delaney_zeroshot_df' in dir() and delaney_zeroshot_df is not None and not delaney_zeroshot_df.empty:
    all_delaney_dfs.append(delaney_zeroshot_df)

# Concatenate all dataframes
combined_delaney_df = concatenate_delaney_dataframes(*all_delaney_dfs)

if not combined_delaney_df.empty:
    print(f"\nCombined Delaney DataFrame shape: {combined_delaney_df.shape}")
    print(f"Columns: {list(combined_delaney_df.columns)}")
    
    # Add derived columns for plotting
    combined_delaney_df["llm_family"] = combined_delaney_df["llm"].apply(extract_llm_family)
    combined_delaney_df["llm_size"] = combined_delaney_df["llm"].apply(extract_llm_size)
    combined_delaney_df["input_variant"] = combined_delaney_df.apply(create_input_variant_label, axis=1)
    
    # Ensure number_of_training_samples is numeric
    combined_delaney_df["number_of_training_samples"] = pd.to_numeric(
        combined_delaney_df["number_of_training_samples"], errors='coerce'
    ).fillna(0).astype(int)
    
    print(f"\nLLM Families found: {combined_delaney_df['llm_family'].unique()}")
    print(f"LLM Sizes found: {combined_delaney_df['llm_size'].unique()}")
    print(f"Training sample sizes: {sorted(combined_delaney_df['number_of_training_samples'].unique())}")
    print(f"Input variants: {combined_delaney_df['input_variant'].unique()}")
    
    # Define families and metrics for plotting
    llm_families = ["gpt-4.1", "gpt-5", "gemini-2.5"]
    metrics_to_plot = ["MAE", "RMSE", "Correlation"]
    
    # Create output folder for bar plots
    barplot_folder = os.path.join(output_folder, "delaney_barplots")
    os.makedirs(barplot_folder, exist_ok=True)
    
    # Generate plots for each family separately (3 families × 3 metrics = 9 plots)
    for family in llm_families:
        family_data = combined_delaney_df[combined_delaney_df["llm_family"] == family]
        if family_data.empty:
            print(f"No data for family: {family}")
            continue
            
        for metric_name in metrics_to_plot:
            output_file = os.path.join(
                barplot_folder, 
                f"delaney_{family.replace('.', '')}_{metric_name.lower()}{file_type}"
            )
            create_delaney_barplot(
                combined_delaney_df,
                metric=metric_name,
                families_to_include=[family],
                output_path=output_file,
                fig_size=(4.6, 2.8),
                title=f"{family.upper()} Family - {metric_name}"
            )
    
    # Generate combined plots for all families (3 metrics)
    for metric_name in metrics_to_plot:
        output_file = os.path.join(
            barplot_folder,
            f"delaney_all_families_{metric_name.lower()}{file_type}"
        )
        # Only include families that exist in data
        existing_families = [f for f in llm_families if f in combined_delaney_df["llm_family"].values]
        
        create_delaney_barplot(
            combined_delaney_df,
            metric=metric_name,
            families_to_include=existing_families,
            output_path=output_file,
            fig_size=(14, 7),
            title=f"All LLM Families - {metric_name}"
        )
    
    print(f"\nDelaney bar plots saved to: {barplot_folder}")
else:
    print("Warning: No Delaney data available for plotting")


def create_approach_label(row: pd.Series) -> str:
    """
    Create a nice label for the approach.
    """
    approach = row["approach"]
    
    mapping = {
        "with_preanalysis": "Pre-analysis",
        "wp_solubility_blind": "Solubility\nBlind",
        "wp_molproperty_clear": "Mol. Prop.\nClear",
        "wp_molproperty_blind": "Mol. Prop.\nBlind",
        "wp_sampleproperty_clear": "Sample Prop.\nClear",
        "wp_sampleproperty_blind": "Sample Prop.\nBlind"
    }
    
    return mapping.get(approach, approach)


def create_delaney_approach_barplot(
    df: pd.DataFrame,
    metric: str,
    families_to_include: Optional[List[str]] = None,
    output_path: str = None,
    fig_size: Tuple[int, int] = (12, 6),
    title: str = None,
    n_samples: int = 1000
):
    """
    Create a bar plot for Delaney results comparing approaches.
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
        print(f"Warning: No data found for approaches barplot (Metric: {metric})")
        return

    # Filter by family if specified
    if families_to_include is not None:
        plot_df = plot_df[plot_df["llm_family"].isin(families_to_include)]
    
    if plot_df.empty:
        print(f"Warning: No data for families {families_to_include} in approaches barplot")
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
    plot_df["approach_label"] = pd.Categorical(plot_df["approach_label"], categories=x_order, ordered=True)
    plot_df["llm_variant"] = pd.Categorical(plot_df["llm_variant"], categories=hue_order, ordered=True)
    
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
                ha='center',
                va='center',
                xytext=(0, 8),
                textcoords='offset points',
                fontsize=8
            )
            
    # Labels
    ax.set_xlabel("Approach (Names Only)", fontsize=12)
    
    if metric == "MAE":
        ylabel = "Mean Absolute Error (log mol/L)"
        ylim = (0, None)
    elif metric == "RMSE":
        ylabel = "Root Mean Squared Error (log mol/L)"
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
    formatted_labels = []
    for label in labels:
        parts = label.split()
        if len(parts) == 2:
            family = parts[0].upper() if "gpt" in parts[0].lower() else parts[0].capitalize()
            size = parts[1].lower()
            
            # Map size names appropriately
            if "gemini" in parts[0].lower():
                # Gemini: nano → Flash Lite, mini → Flash, normal → Pro
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
    
    n_items = len(handles)
    ncol = 3 if n_items > 0 else 1
    
    ax.legend(
        handles,
        formatted_labels,
        title="LLM Model",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=ncol,
        fontsize=9
    )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved approach plot to: {output_path}")
        
    plt.close(fig)


# Generate Approach Barplots
if not combined_delaney_df.empty:
    print("\nGenerating Delaney Approach Barplots...")
    
    approach_plot_folder = os.path.join(output_folder, "delaney_approach_barplots")
    os.makedirs(approach_plot_folder, exist_ok=True)
    
    llm_families = ["gpt-4.1", "gpt-5", "gemini-2.5"]
    metrics_to_plot = ["MAE", "RMSE", "Correlation"]
    
    # Per family
    for family in llm_families:
        for metric_name in metrics_to_plot:
            output_file = os.path.join(
                approach_plot_folder,
                f"approaches_{family.replace('.', '')}_{metric_name.lower()}{file_type}"
            )
            create_delaney_approach_barplot(
                combined_delaney_df,
                metric=metric_name,
                families_to_include=[family],
                output_path=output_file,
                fig_size=(4.6, 2.8),
                title=f"{family.upper()} Family - Approaches - {metric_name}",
                n_samples=1000
            )
            
    # Combined families
    for metric_name in metrics_to_plot:
        output_file = os.path.join(
            approach_plot_folder,
            f"approaches_all_families_{metric_name.lower()}{file_type}"
        )
        existing_families = [f for f in llm_families if f in combined_delaney_df["llm_family"].values]
        
        create_delaney_approach_barplot(
            combined_delaney_df,
            metric=metric_name,
            families_to_include=existing_families,
            output_path=output_file,
            fig_size=(14, 7),
            title=f"All LLM Families - Approaches - {metric_name}",
            n_samples=1000
        )