# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 21:06:48 2025

@author: 20182460
"""

import joblib 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from validate_functions import plot_orion_cell_type_map, create_value_grid
from scipy.stats import pearsonr, spearmanr
from bin_analysis_function import bin_analysis
import pickle
import sys
import os
import git

REPO_DIR = git.Repo(os.getcwd(), search_parent_directories=True).working_tree_dir
sys.path.append(f"{REPO_DIR}/4_validation")
from validate_functions import plot_orion_cell_type_map, create_value_grid

#set correct path in function
def IF_scoring(
    predictions,
    percentile,
    double_marker,
    marker1,
    marker2,
    cell_type,
    remove_empty_tiles,
    num_bins,
    min_tile_count,
    global_threshold=None,
    global_bins=None,
):
    all_results = {}

    # File paths 
    f_grids_dir = "/home/evos/Outputs/CRC/grids" #!SET CORRECT PATH!
    single_cell_metadata_dir = os.path.join(os.path.dirname(__file__), "..", "text.txt")
    single_cell_metadata = pd.read_csv(single_cell_metadata_dir, sep="\t")
    metadata_dict = dict(zip(single_cell_metadata['slide_id'], single_cell_metadata['single_cell_file']))

    slide_ids = list(predictions['slide_id'].unique())

    # Initialize arrays for combined processing
    all_fluorescence_values = []
    all_probability_values = []
    all_count_values = []

    # Loop through slides only once
    slide_data = {}
    for slide_id in slide_ids:
        single_cell_file = metadata_dict[slide_id]
        single_cell_file_name = single_cell_file.split('.')[0]

        predictions_slide = predictions[predictions.slide_id == slide_id].reset_index(drop=True)
        prob_grid = create_value_grid(predictions_slide, cell_type)

        # Load fluorescence and count grids
        grids_marker1 = joblib.load(f"{f_grids_dir}/fluorescence_grids_{marker1}/grids_{single_cell_file_name}.pkl")
        cell_count_grid = grids_marker1['cell_count_grid']

        if double_marker:
            grids_marker2 = joblib.load(f"{f_grids_dir}/fluorescence_grids_{marker2}/grids_{single_cell_file_name}.pkl")
            fluorescence_grid = (grids_marker1['average_fluorescence_grid'] + grids_marker2['average_fluorescence_grid']) / 2
        else:
            fluorescence_grid = grids_marker1['average_fluorescence_grid']

        # Store slide-specific data for later use
        slide_data[slide_id] = {
            "fluorescence": fluorescence_grid,
            "probabilities": prob_grid.values,
            "counts": cell_count_grid,
        }

        # Append flattened values to global arrays
        all_fluorescence_values.append(fluorescence_grid.flatten())
        all_probability_values.append(prob_grid.values.flatten())
        all_count_values.append(cell_count_grid.flatten())

    # Calculate global threshold and bins if not provided
    if global_threshold is None or global_bins is None:
        combined_fluorescence_values = np.concatenate(all_fluorescence_values)
        valid_fluorescence_values = combined_fluorescence_values[~np.isnan(combined_fluorescence_values)]
        global_threshold = np.percentile(valid_fluorescence_values, percentile)
        global_bins = np.linspace(valid_fluorescence_values.min(), valid_fluorescence_values.max(), num_bins + 1)

    # Compute global results
    combined_fluorescence_values = np.concatenate(all_fluorescence_values)
    combined_probability_values = np.concatenate(all_probability_values)
    combined_count_values = np.concatenate(all_count_values)

    if remove_empty_tiles:
        combined_probability_values[combined_count_values == 0] = np.nan

    filtered_fluorescence_values = np.where(combined_fluorescence_values > global_threshold, np.nan, combined_fluorescence_values)
    global_bins, bin_centers, bin_averages, bin_counts = bin_analysis(
        filtered_fluorescence_values, combined_probability_values, num_bins, min_tile_count
    )
    global_correlation, global_p_value = pearsonr(bin_centers, bin_averages)
    print('Global correlation:', global_correlation)

    # Calculate correlations for individual slides
    all_slide_correlations = {}

    for slide_id, data in slide_data.items():
        fluorescence_grid = data["fluorescence"]
        prob_values = data["probabilities"]
        count_values = data["counts"]

        # Apply global threshold
        filtered_fluorescence_values = np.where(fluorescence_grid > global_threshold, np.nan, fluorescence_grid)

        if remove_empty_tiles:
            prob_values[count_values == 0] = np.nan

        # Filter NaN values
        valid_mask = ~np.isnan(filtered_fluorescence_values) & ~np.isnan(prob_values)
        flat_fluorescence = filtered_fluorescence_values.flatten()[valid_mask]
        flat_probabilities = prob_values.flatten()[valid_mask]

        # Digitize fluorescence values into global bins
        bin_indices = np.digitize(flat_fluorescence, global_bins, right=False)

        # Calculate bin averages and centers for the slide
        bin_averages_slide = []
        bin_centers_slide = []

        for i in range(1, num_bins + 1):
            bin_mask = bin_indices == i
            count = np.sum(bin_mask)
            if count >= min_tile_count:
                avg_prob = np.mean(flat_probabilities[bin_mask])
                bin_averages_slide.append(avg_prob)
                bin_centers_slide.append((global_bins[i] + global_bins[i - 1]) / 2)

        # Calculate correlation for the slide
        if len(bin_averages_slide) > 1:
            correlation, p_value = pearsonr(bin_centers_slide, bin_averages_slide)
            all_slide_correlations[slide_id] = [correlation, p_value]
        else:
            all_slide_correlations[slide_id] = [np.nan, np.nan]

    # Compute average correlation
    corr_values = [corr for corr, _ in all_slide_correlations.values() if not np.isnan(corr)]
    average_correlation = np.mean(corr_values) if corr_values else np.nan
    print("Average correlation:", average_correlation)

    # Store results
    all_results['global_corr'] = [global_correlation, global_p_value]
    all_results['separate_corrs'] = all_slide_correlations
    all_results['average_corr'] = average_correlation

    print('Finished analysis')
    return all_results
