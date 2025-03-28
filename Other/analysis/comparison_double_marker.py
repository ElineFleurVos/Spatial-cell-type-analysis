# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 13:19:02 2025

@author: 20182460
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 15:14:24 2025

@author: 20182460
"""

import joblib 
import pickle
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import sys
import git
from scipy.stats import pearsonr, spearmanr

REPO_DIR = git.Repo(os.getcwd(), search_parent_directories=True).working_tree_dir
sys.path.append(f"{REPO_DIR}/4_validation/validation_IF")
sys.path.append(f"{REPO_DIR}/4_validation")
from validate_functions import plot_orion_cell_type_map, create_value_grid
from bin_analysis_function import bin_analysis

"""
Plots for one specific slide
"""

#%%

predictions_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\multitask_lasso\test_tile_predictions_proba_macenko.csv"
#predictions_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\MIL\MIL_old_deconv\all_cell_types.csv"
#predictions_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\MIL\MIL_final_pcchip\all_cell_types.csv"
predictions = pd.read_csv(predictions_dir, sep="\t")
#predictions = pd.read_csv(predictions_dir, sep=",")

single_cell_metadata_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\single_cell_metadata.txt"
single_cell_metadata = pd.read_csv(single_cell_metadata_dir, sep="\t")
metadata_dict = dict(zip(single_cell_metadata['slide_id'], single_cell_metadata['single_cell_file']))

slide_id_nr = 0
slide_ids = list(predictions['slide_id'].unique())
slide_id = slide_ids[slide_id_nr]
slide_id = "19510_C32_US_SCAN_OR_001__160434-registered"
single_cell_file = metadata_dict[slide_id]
single_cell_file_name = single_cell_file.split('.')[0]

cell_type = 'tumor_purity'
double_marker = False
marker1 = 'Pan-CK'
marker2 = ' '
global_info_name = 'Pan-CK_99'
method = "RMTLR"

remove_empty_tiles = True
num_bins = 100
percentile = 99
min_tile_count = 3 #at least 1

predictions_slide = predictions[predictions.slide_id == slide_id]
predictions_slide = predictions_slide.reset_index(drop=True)
prob_grid = create_value_grid(predictions_slide, cell_type)

f_grids_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\fluorescence_grids"
if double_marker == True:
    grids_marker1 = joblib.load(f"{f_grids_dir}/fluorescence_grids_{marker1}/grids_{single_cell_file_name}.pkl")
    average_fluorescence_grid1 = grids_marker1['average_fluorescence_grid']
    cell_count_grid1 = grids_marker1['cell_count_grid']
    grids_marker2 = joblib.load(f"{f_grids_dir}/fluorescence_grids_{marker2}/grids_{single_cell_file_name}.pkl")
    average_fluorescence_grid2 = grids_marker2['average_fluorescence_grid']
    cell_count_grid2 = grids_marker2['cell_count_grid']
    fluorescence_grid = (average_fluorescence_grid1 + average_fluorescence_grid2)/2
else:
    grids_marker1 = joblib.load(f"{f_grids_dir}/fluorescence_grids_{marker1}/grids_{single_cell_file_name}.pkl")
    fluorescence_grid = grids_marker1['average_fluorescence_grid']
    cell_count_grid1 = grids_marker1['cell_count_grid']


#%%
#set nans in fluorescence grid where there are nans in probability grid, so that all these zero's don't count for threshold calculation
global_info_output_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\fluorescence_grids\global_information"
with open(f"{global_info_output_dir}/global_info_{global_info_name}.pkl", "rb") as file:
    global_info = pickle.load(file)
threshold = global_info['threshold']
print(threshold)
#percent = np.percentile(average_fluorescence_grid, 99)

#set non-informative tiles to nans
filtered_fluorescence_grid = np.where(fluorescence_grid > threshold, np.nan, fluorescence_grid)
max_fluorescence = np.nanmax(filtered_fluorescence_grid)
print(max_fluorescence)

#set nans in probability grid where cell count is 0 in the fluorescence grid 
if remove_empty_tiles == True:
    prob_grid[cell_count_grid1 == 0] = np.nan

#%% calculate old score 
#fluorescence_grid = filtered_fluorescence_grid 

flat_fluorescence = filtered_fluorescence_grid.flatten()
flat_probabilities = prob_grid.to_numpy().flatten()  # Ensure probabilities are from DataFrame

# # Mask out background tiles (where probabilities are NaN)
valid_mask = ~np.isnan(flat_fluorescence) & ~np.isnan(flat_probabilities)
flat_fluorescence = flat_fluorescence[valid_mask]
flat_probabilities = flat_probabilities[valid_mask]

#put fluorescence values into the global bins
global_bins = global_info['bins']
bin_indices = np.digitize(flat_fluorescence, global_bins, right=False)

# Step 2e: Calculate the average probabilities and bin counts for the current slide
bin_counts_slide = [np.sum(bin_indices == i) for i in range(1, num_bins + 1)]
bin_averages_slide = []
bin_centers_slide = []

filtered_bin_averages = []
filtered_bin_centers = []

for i in range(1, num_bins + 1):
    bin_centers_slide.append((global_bins[i] + global_bins[i - 1]) / 2)  # Store all bin centers
    bin_mask = bin_indices == i

    if bin_counts_slide[i - 1] >= min_tile_count:
        avg_prob = np.mean(flat_probabilities[bin_mask])
        filtered_bin_averages.append(avg_prob)  # Only add nonzero bins for line plot & correlation
        filtered_bin_centers.append(bin_centers_slide[-1])
    else:
        avg_prob = 0  # Keep zero for bar chart only

    bin_averages_slide.append(avg_prob)

# **Calculate correlation excluding zero-count bins**
if len(filtered_bin_centers) > 1:  # Correlation requires at least 2 points
    correlation, _ = pearsonr(filtered_bin_centers, filtered_bin_averages)
    print(f"Correlation between fluorescence bins and probabilities: {correlation:.2f}")
else:
    print("Not enough data for correlation calculation.")

#%%
output_dir_plots = r"C:\Users\20182460\Desktop\Master_thesis\Figures report\all_maps\all_basic_maps_MIL"

fig, axes = plt.subplots(1, 2, figsize=(8,3), gridspec_kw={
                        'width_ratios': [1.3, 1.3]})

sns.heatmap(fluorescence_grid, cmap='viridis', cbar=True, square=True, linewidths=0,annot=False,
            xticklabels=False, yticklabels=False, ax=axes[0], vmin=0, vmax=max_fluorescence)
axes[0].set_title(f"IF map {marker1}")
axes[0].set_aspect('equal', adjustable='box')

cmap_cell_types = {"T_cells": "Greens", "CAFs": "Blues",
                    "tumor_purity": "Reds", "endothelial_cells": "Purples"}

sns.heatmap(prob_grid, cmap=cmap_cell_types[cell_type], ax=axes[1], vmin=0.2, vmax=0.8,
            xticklabels=False, yticklabels=False)
axes[1].set_title("Tumor prob. grid (MIL)")
axes[1].set_aspect('equal', adjustable='box')

plt.savefig(f"{output_dir_plots}/{slide_id}_plots_{marker1}_{method}.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns

axs[0].bar(bin_centers_slide, bin_counts_slide, width=(global_bins[1] - global_bins[0]), 
            color='#2980b9', edgecolor='black', alpha=0.7)
axs[0].set_xlabel('Fluorescence (Bin Centers)', fontsize=14)
axs[0].set_ylabel('Bin count', fontsize=14)
#axs[0].set_ylim(0, 1000)
axs[0].set_title(f"{marker1}", fontsize=14)
axs[0].grid(axis='y')

# **Line plot (excludes zero-count bins)**
axs[1].plot(filtered_bin_centers, filtered_bin_averages, marker='o', linestyle='-', label='Average Probability')
axs[1].set_xlabel('Fluorescence (Bin Centers)', fontsize=14)
axs[1].set_ylabel('Average Probability', fontsize=14)
axs[1].set_title(f"{method} {marker1}", fontsize=14)
axs[1].grid(True)
axs[1].text(
    0.05, 0.95,  # Position in axes coordinates (0,0 = bottom-left, 1,1 = top-right)
    f"$r$ = {correlation:.2f}",
    fontsize=15,
    transform=plt.gca().transAxes,  # Ensures text is placed relative to the axes
    verticalalignment='top',  # Align text to the top
    bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0.3')  # White background with black border
    )
#axs[1].legend(fontsize=12)

# Adjust layout
plt.tight_layout()
plt.savefig(f"{output_dir_plots}/{slide_id}_maps_{marker1}_{method}.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()

#%%

# # Create subplots
# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(5, 4))
# fig.subplots_adjust(hspace=0.05)  # Adjust space between Axes

# # Bar width (ensure the width is the same for all bars)
# bar_width = bin_centers_slide[1] - bin_centers_slide[0]  # Compute the width of each bar based on bin centers

# # Plot the same data on both Axes
# ax1.bar(bin_centers_slide, bin_counts_slide, color='#226795', edgecolor='black', alpha=0.7, align='center', width=bar_width)
# ax2.bar(bin_centers_slide, bin_counts_slide, color='#226795', edgecolor='black', alpha=0.7, align='center', width=bar_width)

# # Zoom-in / limit the view to different portions of the data
# ax1.set_ylim(2500, 3500)  # Outliers only
# ax2.set_ylim(0, 900)  # Most of the data

# # Set x-axis limits to ensure consistency
# ax1.set_xlim([0, max(bin_centers_slide)])
# ax2.set_xlim([0, max(bin_centers_slide)])

# # Hide the spines between ax and ax2
# ax1.spines.bottom.set_visible(False)
# ax2.spines.top.set_visible(False)
# ax1.xaxis.tick_top()
# ax1.tick_params(labeltop=False)  # Don't put tick labels at the top
# ax2.xaxis.tick_bottom()


# # Add slanted lines between the two axes for clarity
# d = .5  # Proportion of vertical to horizontal extent of the slanted line
# kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
#               linestyle="none", color='k', mec='k', mew=1, clip_on=False)
# ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
# ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

# plt.tight_layout()
# plt.savefig(f"{output_dir_plots}/{slide_id}_maps_{cell_type}.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
# plt.show()
