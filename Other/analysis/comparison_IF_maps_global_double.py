# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 13:41:27 2025

@author: 20182460
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 12:34:13 2025

@author: 20182460
"""
import joblib 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from scipy.stats import pearsonr, spearmanr
import pickle
import os
import sys
import git

REPO_DIR = git.Repo(os.getcwd(), search_parent_directories=True).working_tree_dir
sys.path.append(f"{REPO_DIR}/4_validation/validation_IF")
sys.path.append(f"{REPO_DIR}/4_validation")
from validate_functions import plot_orion_cell_type_map, create_value_grid
from bin_analysis_function import bin_analysis

"""
Global plots across all slides
"""
f_grids_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\fluorescence_grids"

#predictions_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\multitask_lasso\test_tile_predictions_proba_macenko.csv"
#predictions_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\MIL\MIL_final_pcchip\all_cell_types.csv"
#predictions_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\MIL\MIL_old_deconv\all_cell_types.csv"
#predictions_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\MIL\MIL_loss_norm_tests\simple_nn_norm_cor\all_cell_types.csv"
#predictions_dir = "C:/Users/20182460/Desktop/Master_thesis/Code/Outputs/FINAL RESULTS/CRC/MIL/MIL_extend_cell_types/PC_CHiP/pcchip_all_cell_types.csv"
predictions_dir = "C:/Users/20182460/Desktop/Master_thesis/Code/Outputs/FINAL RESULTS/CRC/MIL/MIL_multiple_cell_types/CD48_BM_pchip.csv"
predictions = pd.read_csv(predictions_dir, sep="\t")
#predictions = pd.read_csv(predictions_dir, sep=",")

single_cell_metadata_dir = os.path.join(os.path.dirname(__file__), "..", "text.txt")
single_cell_metadata = pd.read_csv(single_cell_metadata_dir, sep="\t")
metadata_dict = dict(zip(single_cell_metadata['slide_id'], single_cell_metadata['single_cell_file']))

slide_ids = list(predictions['slide_id'].unique())

double_marker = False
cell_type = "B"
marker1 = 'CD20'
marker2 = ''

percentile = 99
remove_empty_tiles = True
num_bins = 100
min_tile_count = 3  #at least 1

global_info_name = f'CD20_{percentile}2'
output_name = f"CD20_{percentile}"
output_name_plot = f"RMTLR_CD20_{percentile}"
output_name_title = r"RMTLR CD20 (T cells)"

all_fluorescence_values = []
all_probability_values = []
all_count_values = []
for slide_id in slide_ids:
    print(slide_id)
    single_cell_file = metadata_dict[slide_id]
    single_cell_file_name = single_cell_file.split('.')[0]
    
    predictions_slide = predictions[predictions.slide_id == slide_id]
    predictions_slide = predictions_slide.reset_index(drop=True)
    
    prob_grid = create_value_grid(predictions_slide, cell_type)

    if double_marker == True:
        grids_marker1 = joblib.load(f"{f_grids_dir}/fluorescence_grids_{marker1}/grids_{single_cell_file_name}.pkl")
        average_fluorescence_grid1 = grids_marker1['average_fluorescence_grid']
        cell_count_grid1 = grids_marker1['cell_count_grid']
        grids_marker2 = joblib.load(f"{f_grids_dir}/fluorescence_grids_{marker2}/grids_{single_cell_file_name}.pkl")
        average_fluorescence_grid2 = grids_marker2['average_fluorescence_grid']
        cell_count_grid2 = grids_marker2['cell_count_grid']
        #fluorescence_grid = (average_fluorescence_grid1 + average_fluorescence_grid2)/2
        fluorescence_grid = (6/16)*average_fluorescence_grid1 + (10/16)*average_fluorescence_grid2
    else:
        grids_marker1 = joblib.load(f"{f_grids_dir}/fluorescence_grids_{marker1}/grids_{single_cell_file_name}.pkl")
        fluorescence_grid = grids_marker1['average_fluorescence_grid']
        cell_count_grid1 = grids_marker1['cell_count_grid']
        
    all_fluorescence_values.append(fluorescence_grid.flatten())
    all_probability_values.append(prob_grid.values.flatten())
    all_count_values.append(cell_count_grid1.flatten())
    
#Calculate global bins and percentile across all slides
combined_fluorescence_values = np.concatenate(all_fluorescence_values)
combined_probability_values = np.concatenate(all_probability_values)
combined_count_values = np.concatenate(all_count_values)

assert len(combined_fluorescence_values) == len(combined_probability_values), "Arrays must have the same length"
combined_fluorescence_values = np.where(np.isnan(combined_probability_values), np.nan, combined_fluorescence_values)
valid_fluorescence_values = combined_fluorescence_values[~np.isnan(combined_fluorescence_values)]
global_threshold = np.percentile(valid_fluorescence_values, percentile)
print(global_threshold)

#set nans in probability grid where cell count is 0 in the fluorescence grid 
if remove_empty_tiles == True:
    combined_probability_values[combined_count_values == 0] = np.nan  # Adjust based on combined count values

filtered_fluorescence_values = np.where(combined_fluorescence_values > global_threshold, np.nan, combined_fluorescence_values)
global_bins, bin_centers, bin_averages, bin_counts = bin_analysis(filtered_fluorescence_values, combined_probability_values, num_bins, min_tile_count)

global_info_output_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\fluorescence_grids\global_information"
global_info = {}
global_info['threshold'] = global_threshold
global_info['bins'] = global_bins
with open(f"{global_info_output_dir}/global_info_{global_info_name}.pkl", "wb") as file:
    pickle.dump(global_info, file)

global_correlation, global_p_value = pearsonr(bin_centers, bin_averages)
print("Global Correlation:", global_correlation)

#%%
#output_dir_plot = r"C:\Users\20182460\Desktop\Master_thesis\Figures report\figures_lasso_vs_MIL"
#output_dir_plot = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\MIL\MIL_old_deconv"
#output_dir_plot = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\MIL\MIL_extend_cell_types\PC_CHiP"
output_dir_plot = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\MIL\MIL_multiple_cell_types"
    
fig, ax = plt.subplots(figsize=(7, 5.2), dpi=300, facecolor='none')
plt.plot(bin_centers, bin_averages, marker='o', linestyle='-', label='Average Probability')
plt.xlabel('Average fluorescence (bin centres)', fontsize=17)
plt.ylabel('Average probability', fontsize=17)
plt.title(f"{output_name_title}", fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(True)
#plt.legend(fontsize=15)

plt.text(
    0.95, 0.05,  # Move text to bottom-right corner
    f"$r$ = {global_correlation:.2f}",
    fontsize=24,  # Increase font size
    transform=plt.gca().transAxes,
    verticalalignment='bottom',  # Align text to bottom
    horizontalalignment='right',  # Align text to right
    bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0.3')  # White background
)

fig.savefig(f"{output_dir_plot}/{output_name_plot}.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()

#%% bar plots 
#output_dir_plot = r"C:\Users\20182460\Desktop\Master_thesis\Figures report\figures_lasso_vs_MIL"
#output_dir_plot = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\MIL\MIL_extend_cell_types"
output_dir_plot = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\MIL\MIL_multiple_cell_types"
cell_type_name = r"CD163 + CD68"
bin_centers_all = []
for i in range(1, num_bins + 1):
    bin_centers_all.append((global_bins[i] + global_bins[i - 1]) / 2)
cin_centers_all = np.array(bin_centers_all)


bin_counts_ = bin_counts
fig, ax = plt.subplots(figsize=(7, 5.2), dpi=300)
plt.bar(bin_centers_all, bin_counts_, width=(global_bins[1] - global_bins[0]), 
            color='#2980b9', edgecolor='black', alpha=0.7)
plt.xlabel('Fluorescence (Bin Centers)', fontsize=17)
plt.ylabel('Bin count', fontsize=17)
plt.title(f"{cell_type_name}", fontsize=17)
plt.ylim(0,4000)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(axis='y')

plt.tight_layout()
fig.savefig(f"{output_dir_plot}/bar_{marker1}.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()

fig, ax = plt.subplots(figsize=(7, 5.2), dpi=300)
plt.bar(bin_centers_all, bin_counts_, width=(global_bins[1] - global_bins[0]), 
            color='#2980b9', edgecolor='black', alpha=0.7)
plt.xlabel('Fluorescence (Bin Centers)', fontsize=17)
plt.ylabel('Bin count', fontsize=17)
plt.title(f"{cell_type_name}", fontsize=17)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(axis='y')

plt.tight_layout()
fig.savefig(f"{output_dir_plot}/bar_complete_{marker1}.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(5, 4), dpi=300)
fig.subplots_adjust(hspace=0.05)  # Adjust space between Axes

# Bar width (ensure the width is the same for all bars)
bar_width = bin_centers_all[1] - bin_centers_all[0]  # Compute the width of each bar based on bin centers

# Plot the same data on both Axes
ax1.bar(bin_centers_all, bin_counts_, color='#226795', edgecolor='black', alpha=0.7, align='center', width=bar_width)
ax2.bar(bin_centers_all, bin_counts_, color='#226795', edgecolor='black', alpha=0.7, align='center', width=bar_width)

# Zoom-in / limit the view to different portions of the data
ax1.set_ylim(10000, max(bin_counts_)+2000)  # Outliers only
ax2.set_ylim(0, 5000)  # Most of the data

# Set x-axis limits to ensure consistency
#ax1.set_xlim([0, max(bin_centers)])
#ax2.set_xlim([0, max(bin_centers)])

# Hide the spines between ax and ax2
ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)  # Don't put tick labels at the top
ax2.xaxis.tick_bottom()

# Add slanted lines between the two axes for clarity
d = .5  # Proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

#ax1.xaxis.set_ticks([])  # Remove ticks
#ax1.xaxis.set_ticklabels([])  # Remove tick labels
ax1.tick_params(bottom=False, labelbottom=False, top=False, labeltop=False)  
xtick_values = np.arange(200, 1801, 200)
ax2.set_xticks(xtick_values)
ax2.set_xticklabels([str(x) for x in xtick_values])  # Convert to string labels
# ax1.tick_params(axis='y', labelsize=11) 
# ax2.tick_params(axis='x', labelsize=11)  
# ax2.tick_params(axis='y', labelsize=11)  
fig.suptitle(f"{cell_type_name}", fontsize=12, y=0.92)
fig.text(0.5, 0, "Fluorescence (bin centers)", ha='center', fontsize=12)  # X label at bottom
fig.text(0, 0.5, "Bin count", va='center', rotation='vertical', fontsize=12)  # Y label on left


plt.tight_layout()
plt.savefig(f"{output_dir_plot}/bar_cut_{marker1}.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()

#%% seperate scoring with global percentile and bins

all_slide_correlations = {}
for slide_id in slide_ids:
    print(f"Processing Slide: {slide_id}")
    
    single_cell_file = metadata_dict[slide_id]
    single_cell_file_name = single_cell_file.split('.')[0]
    
    predictions_slide = predictions[predictions.slide_id == slide_id]
    predictions_slide = predictions_slide.reset_index(drop=True)
    
    prob_grid = create_value_grid(predictions_slide, cell_type)
    
    if double_marker == True:
        grids_marker1 = joblib.load(f"{f_grids_dir}/fluorescence_grids_{marker1}/grids_{single_cell_file_name}.pkl")
        average_fluorescence_grid1 = grids_marker1['average_fluorescence_grid']
        cell_count_grid1 = grids_marker1['cell_count_grid']
        grids_marker2 = joblib.load(f"{f_grids_dir}/fluorescence_grids_{marker2}/grids_{single_cell_file_name}.pkl")
        average_fluorescence_grid2 = grids_marker2['average_fluorescence_grid']
        cell_count_grid2 = grids_marker2['cell_count_grid']
        #fluorescence_grid = (average_fluorescence_grid1 + average_fluorescence_grid2)/2
        fluorescence_grid = (6/16)*average_fluorescence_grid1 + (10/16)*average_fluorescence_grid2
    else:
        grids_marker1 = joblib.load(f"{f_grids_dir}/fluorescence_grids_{marker1}/grids_{single_cell_file_name}.pkl")
        fluorescence_grid = grids_marker1['average_fluorescence_grid']
        cell_count_grid1 = grids_marker1['cell_count_grid']
    
    #Filter out fluorescence values below the global percentile for the current slide
    filtered_fluorescence_grid = np.where(fluorescence_grid > global_threshold, np.nan, fluorescence_grid)
    
    if remove_empty_tiles == True:
        prob_grid[cell_count_grid1 == 0] = np.nan
        
    flat_fluorescence = filtered_fluorescence_grid.flatten()
    flat_probabilities = prob_grid.values.flatten()

    #Mask out NaN values
    valid_mask = ~np.isnan(flat_fluorescence) & ~np.isnan(flat_probabilities)
    flat_fluorescence = flat_fluorescence[valid_mask]
    flat_probabilities = flat_probabilities[valid_mask]
    
    #put fluorescence values into the global bins
    bin_indices = np.digitize(flat_fluorescence, global_bins, right=False)
    
    # Step 2e: Calculate the average probabilities and bin counts for the current slide
    bin_counts_slide = [np.sum(bin_indices == i) for i in range(1, num_bins + 1)]
    bin_averages_slide = []
    bin_centers_slide = []
    
    for i in range(1, num_bins + 1):
        bin_mask = bin_indices == i
        count = bin_counts_slide[i - 1]
        if count >= min_tile_count:
            avg_prob = np.mean(flat_probabilities[bin_mask])
            bin_averages_slide.append(avg_prob)
            bin_centers_slide.append((global_bins[i] + global_bins[i - 1]) / 2)  # Use bin center for plotting
    
    # Calculate correlation for the current slide
    if len(bin_averages_slide) > 1:  # Ensure there are enough bins for correlation
        correlation, p_value = pearsonr(bin_centers_slide, bin_averages_slide)
        all_slide_correlations[slide_id] = [correlation, p_value]
    else:
        all_slide_correlations[slide_id] = [np.nan, np.nan]  # Handle cases with not enough bins

corr_values = [correlation for correlation, _ in all_slide_correlations.values()]
average_correlation = np.mean(corr_values)
print("Average Correlation:", average_correlation)

min_value = min(corr_values)
max_value = max(corr_values)

print("Minimum Value:", min_value)
print("Maximum Value:", max_value)

all_results = {}
all_results['global_corr'] = [global_correlation, global_p_value]
all_results['separate_corrs'] = all_slide_correlations
all_results['average_corr'] = average_correlation

#output_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\multitask_lasso"
#output_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\MIL\MIL_final_pcchip"
#output_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\MIL\MIL_old_deconv"
#output_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\MIL\MIL_extend_cell_types\UNI"
output_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\MIL\MIL_multiple_cell_types"
with open(f'{output_dir}/{output_name}.pkl', 'wb') as f:
    pickle.dump(all_results, f)
 
