# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 11:54:07 2025

@author: 20182460
"""

import joblib 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import sys
import os
import git
REPO_DIR = git.Repo(os.getcwd(), search_parent_directories=True).working_tree_dir
sys.path.append(f"{REPO_DIR}/4_validation")
from validate_functions import plot_orion_cell_type_map, create_value_grid
from scipy.stats import pearsonr, spearmanr

def bin_analysis(fluorescence_grid, prob_grid, num_bins, min_tile_count=1):
    # Ensure grids are flattened for easier processing
    flat_fluorescence = fluorescence_grid.flatten()
    if isinstance(prob_grid, pd.DataFrame):
        flat_probabilities = prob_grid.values.flatten()
    else:
        flat_probabilities = prob_grid.flatten()

    # Mask out background tiles (fluorescence = 0, probabilities = NaN)
    #valid_mask = ~np.isnan(flat_probabilities)
    valid_mask = ~np.isnan(flat_fluorescence) & ~np.isnan(flat_probabilities)
    flat_fluorescence = flat_fluorescence[valid_mask]
    flat_probabilities = flat_probabilities[valid_mask]

    # Create bins for fluorescence values
    bins = np.linspace(flat_fluorescence.min(), flat_fluorescence.max(), num_bins + 1)
    bins[-1] = flat_fluorescence.max() + 1e-5 
    bin_indices = np.digitize(flat_fluorescence, bins, right=False)

    # Calculate counts and averages for each bin
    bin_counts = [np.sum(bin_indices == i) for i in range(1, num_bins + 1)]
    bin_averages = []
    bin_centers = []

    for i in range(1, num_bins + 1):
        bin_mask = bin_indices == i
        count = bin_counts[i - 1]  # Number of tiles in this bin
        if count >= min_tile_count:  # Only include bins with enough tiles
            avg_prob = np.mean(flat_probabilities[bin_mask])
            bin_averages.append(avg_prob)
            bin_centers.append((bins[i] + bins[i - 1]) / 2)  # Use bin center for plotting

    # Return bin centers, corresponding average probabilities, and bin counts
    return bins, np.array(bin_centers), np.array(bin_averages), np.array(bin_counts)