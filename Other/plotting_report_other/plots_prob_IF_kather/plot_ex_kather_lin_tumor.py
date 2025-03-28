# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 13:45:29 2025

@author: 20182460
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 21:39:56 2024

@author: 20182460
"""

import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from skimage.transform import resize
import cv2
from skimage.measure import block_reduce
from scipy import stats
from sklearn.metrics import jaccard_score
from scipy.spatial.distance import dice
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import matplotlib.image as mpimg
import joblib
import sys
import git

REPO_DIR = git.Repo(os.getcwd(), search_parent_directories=True).working_tree_dir
sys.path.append(f"{REPO_DIR}/4_validation")
from validate_functions import plot_orion_cell_type_map, create_value_grid, assign_cell_types, calc_jaccard_dice, majority_vote_downscale

#%% load data

LABEL_DICT = {
    "BACK": 0,
    "NORM": 1,
    "DEB": 2,
    "TUM": 3,
    "ADI": 4,
    "MUC": 5,
    "MUS": 6,
    "STR": 7,
    "LYM": 8,
}


cell_type = "tumor_purity" #tumor_purity
marker1 = 'Pan-CK'
marker2 = 'E-cadherin'
name_global_info = 'cadpan_99'
cell_threshold = 0.5
cell_type_name = 'Tumor'

predictions_dir_lasso = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\multitask_lasso\test_tile_predictions_proba_macenko.csv"
predictions_dir_MIL = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\MIL\MIL_final_pcchip\all_cell_types.csv"
#predictions_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\MIL\MIL_final_pcchip\all_cell_types.csv"
#predictions_dir_MIL = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\MIL\MIL_old_deconv\all_cell_types.csv"
pngs_dir_gamma = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\CRC\Orion\Orion_HE_png_gamma"
predictions_lasso = pd.read_csv(predictions_dir_lasso, sep="\t")
#predictions_MIL = pd.read_csv(predictions_dir_MIL, sep="\t")
predictions_MIL = pd.read_csv(predictions_dir_MIL, sep=",")

single_cell_metadata_dir = "single_cell_metadata.txt"
single_cell_metadata = pd.read_csv(single_cell_metadata_dir, sep="\t")
# Preload metadata into a dictionary for fast lookups
metadata_dict = dict(zip(single_cell_metadata['slide_id'], single_cell_metadata['single_cell_file']))

#%% spotlight prediction

slide_id_nr = 23
slide_ids = list(predictions_MIL['slide_id'].unique())
slide_id = slide_ids[slide_id_nr]
slide_id = "19510_C15_US_SCAN_OR_001__152234-registered"
single_cell_file = metadata_dict[slide_id]
single_cell_file_name = single_cell_file.split('.')[0]

png_dir = f"{pngs_dir_gamma}/{slide_id}.png"
image_png = mpimg.imread(png_dir)

predictions_lasso = predictions_lasso[predictions_lasso.slide_id == slide_id]
predictions_lasso = predictions_lasso.reset_index(drop=True)

predictions_MIL = predictions_MIL[predictions_MIL.slide_id == slide_id]
predictions_MIL = predictions_MIL.reset_index(drop=True)

prob_grid_lasso = create_value_grid(predictions_lasso, cell_type)
prob_grid_MIL = create_value_grid(predictions_MIL, cell_type)
cmap_cell_types = {"T_cells": "Greens", "CAFs": "Blues",
                   "tumor_purity": "Reds", "endothelial_cells": "Purples"}

# %% Kather prediction

dir_map_pickle = f"C:/Users/20182460/Desktop/Master_thesis/Code/Outputs/CRC/Orion/Orion_predictions_112um_224pixels/{slide_id}_gamma_densenet161-kather100k_cell_type_map.pkl"

with open(dir_map_pickle, 'rb') as f:
    model_output = pickle.load(f)

patch_length = 224
index = np.fliplr(np.array(model_output["coordinates"])[
                  :, :2] / patch_length).astype(int)
grid_shape = index.max(axis=0) + 1
kather_grid = np.zeros(grid_shape)
kather_grid[tuple(index.T)] = np.asarray(model_output["predictions"])

#kather_resized = majority_vote_downscale(kather_grid, prob_grid_lasso.shape)
kather_resized = cv2.resize(kather_grid, (prob_grid_lasso.shape[1], prob_grid_lasso.shape[0]), interpolation=cv2.INTER_NEAREST)

#%% Fluorescence map 

f_grids_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\fluorescence_grids"


grids_panck = joblib.load(f"{f_grids_dir}/fluorescence_grids_{marker1}/grids_{single_cell_file_name}.pkl")
average_fluorescence_panck = grids_panck['average_fluorescence_grid']
grids_ecad = joblib.load(f"{f_grids_dir}/fluorescence_grids_{marker2}/grids_{single_cell_file_name}.pkl")
average_fluorescence_ecad = grids_ecad['average_fluorescence_grid']
combined_grid = (average_fluorescence_panck + average_fluorescence_ecad)/2

#calculate local threshold. Other option is to use global threshold
global_info_output_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\fluorescence_grids\global_information"
with open(f"{global_info_output_dir}/global_info_cadpan_99.pkl", "rb") as file:
    global_info = pickle.load(file)
threshold_combined = global_info['threshold']

with open(f"{global_info_output_dir}/global_info_E-cadherin_99.pkl", "rb") as file:
    global_info = pickle.load(file)
threshold_ecad = global_info['threshold']

with open(f"{global_info_output_dir}/global_info_Pan-CK_99.pkl", "rb") as file:
    global_info = pickle.load(file)
threshold_panck = global_info['threshold']


# filtered_fluorescence_grid = np.where(fluorescence_grid > threshold, np.nan, average_fluorescence_grid)
# max_fluorescence = np.nanmax(filtered_fluorescence_grid)

# %% overall plot
#vmin = 0.2
#vmax = 0.8

fig, axs = plt.subplots(1, 7, figsize=(30, 2.1), gridspec_kw={
                        'width_ratios': [1, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3]})
image_height, image_width, _ = image_png.shape
image_aspect = image_width / image_height
image_aspect = 1.0

axs = axs.flatten()

axs[0].axis('off')  # Hide axis
axs[0].imshow(image_png, aspect='auto')
axs[0].set_title('H&E image', fontsize=15)
axs[0].set_aspect(image_aspect)

sns.heatmap(average_fluorescence_panck, cmap='viridis', vmin=0, vmax=threshold_panck,
            xticklabels=False, yticklabels=False, ax=axs[1])
axs[1].set_title("IF map Pan-CK", fontsize=15)
axs[1].set_aspect(image_aspect)

sns.heatmap(average_fluorescence_ecad, cmap='viridis', vmin=0, vmax=threshold_ecad,
            xticklabels=False, yticklabels=False, ax=axs[2])
axs[2].set_title("IF map E-cadherin", fontsize=15)
axs[2].set_aspect(image_aspect)

sns.heatmap(combined_grid, cmap='viridis', vmin=0, vmax=threshold_combined,
            xticklabels=False, yticklabels=False, ax=axs[3])
axs[3].set_title("IF map Pan-CK + E-cadherin", fontsize=15)
axs[3].set_aspect(image_aspect)


plot_orion_cell_type_map(kather_resized, ax=axs[4])
axs[4].set_title('Validation cell type map', fontsize=15)
axs[4].set_aspect(image_aspect)

sns.heatmap(prob_grid_lasso, cmap=cmap_cell_types[cell_type], vmin=0.2, vmax=0.8,
            xticklabels=False, yticklabels=False, ax=axs[5])
axs[5].set_title(f"{cell_type_name} prob. map RMTLR", fontsize=15)
axs[5].set_aspect(image_aspect)

sns.heatmap(prob_grid_MIL, cmap=cmap_cell_types[cell_type], vmin=0.2, vmax=1.0,
            xticklabels=False, yticklabels=False, ax=axs[6])
axs[6].set_title(f"{cell_type_name} prob. map MIL", fontsize=15)
axs[6].set_aspect(image_aspect)

output_dir_plot = r"C:\Users\20182460\Desktop\Master_thesis\Figures report\figures_lasso_vs_MIL\Examples"
plt.savefig(f"{output_dir_plot}/{slide_id}_tumor.png", dpi=300, bbox_inches="tight")

# Show the figure (optional)
plt.show()
