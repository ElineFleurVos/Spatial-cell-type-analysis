# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:44:22 2025

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
from validate_functions import plot_orion_cell_type_map, create_value_grid, assign_cell_types, calc_jaccard_dice

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
# double_marker = False
# cell_type = "T_cells" #tumor_purity
# marker1 = 'CD3e'
# marker2 = ''
# name_global_info = 'CD3e_99'
# cell_threshold = 0.5
# name_fluo_map = 'CD3e'
# cell_type_name = 'T'

pngs_dir_gamma = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\CRC\Orion\Orion_HE_png_gamma"

predictions_dir_MIL = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\MIL\MIL_extend_cell_types\UNI\UNI_all_cell_types.csv"
predictions_MIL = pd.read_csv(predictions_dir_MIL, sep=",")

predictions_dir_basic = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\MIL\MIL_final_pcchip\all_cell_types.csv"
predictions_dir_T = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\MIL\MIL_UNI_FF_FFPE_tests\T\T_norm_macenko_FFPE"
predictions_T = pd.read_csv(predictions_dir_basic, sep=",")

single_cell_metadata_dir = "single_cell_metadata.txt"
single_cell_metadata = pd.read_csv(single_cell_metadata_dir, sep="\t")
metadata_dict = dict(zip(single_cell_metadata['slide_id'], single_cell_metadata['single_cell_file']))

#%% spotlight prediction

slide_id_nr = 0
slide_ids = list(predictions_MIL['slide_id'].unique())
slide_id = slide_ids[slide_id_nr]
#slide_id = "18459_LSP10375_US_SCAN_OR_001__092147-registered"
single_cell_file = metadata_dict[slide_id]
single_cell_file_name = single_cell_file.split('.')[0]

png_dir = f"{pngs_dir_gamma}/{slide_id}.png"
image_png = mpimg.imread(png_dir)

predictions_MIL = predictions_MIL[predictions_MIL.slide_id == slide_id]
predictions_MIL = predictions_MIL.reset_index(drop=True)
prob_grid_MIL_CD4T = create_value_grid(predictions_MIL, 'CD4_T_cells')
prob_grid_MIL_CD8T = create_value_grid(predictions_MIL, 'CD8_T_cells')

predictions_T = predictions_T[predictions_T.slide_id == slide_id]
predictions_T = predictions_T.reset_index(drop=True)
prob_grid_MIL_T = create_value_grid(predictions_T, 'T_cells')

#%% Fluorescence map 

f_grids_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\fluorescence_grids"
grids_CD3e = joblib.load(f"{f_grids_dir}/fluorescence_grids_CD3e/grids_{single_cell_file_name}.pkl")
fluorescence_grid_CD3e = grids_CD3e['average_fluorescence_grid']
grids_CD4 = joblib.load(f"{f_grids_dir}/fluorescence_grids_CD4/grids_{single_cell_file_name}.pkl")
fluorescence_grid_CD4 = grids_CD4['average_fluorescence_grid']
grids_CD8a = joblib.load(f"{f_grids_dir}/fluorescence_grids_CD8a/grids_{single_cell_file_name}.pkl")
fluorescence_grid_CD8a = grids_CD8a['average_fluorescence_grid']

global_info_output_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\fluorescence_grids\global_information"

with open(f"{global_info_output_dir}/global_info_CD3e_99.pkl", "rb") as file:
    threshold_CD3e = pickle.load(file)['threshold']
with open(f"{global_info_output_dir}/global_info_CD4_99.pkl", "rb") as file:
    threshold_CD4 = pickle.load(file)['threshold']
with open(f"{global_info_output_dir}/global_info_CD8a_99.pkl", "rb") as file:
    threshold_CD8a = pickle.load(file)['threshold']
    
# filtered_fluorescence_grid = np.where(fluorescence_grid > threshold, np.nan, average_fluorescence_grid)
# max_fluorescence = np.nanmax(filtered_fluorescence_grid)

#%%Plotting

# cmap_cell_types = {"T_cells": "Greens", "CD4_T_cells": "Blues",
#                    "CD8_T_cells": "Reds", "endothelial_cells": "Purples"}

cmap_cell_types = {"T_cells": "Greens", "CD4_T_cells": "Greens",
                   "CD8_T_cells": "Greens", "endothelial_cells": "Purples"}


#(22,2.1)
fig, axs = plt.subplots(3, 3, figsize=(10,7), gridspec_kw={
                        'width_ratios': [1, 1.3, 1.3]})
image_height, image_width, _ = image_png.shape
image_aspect = image_width / image_height
image_aspect = 1.0

#axs = axs.flatten()

axs[0,0].axis('off')  # Hide axis
axs[0,0].imshow(image_png, aspect='auto')
axs[0,0].set_title('H&E image', fontsize=15)
axs[0,0].set_aspect(image_aspect)

sns.heatmap(fluorescence_grid_CD3e, cmap='viridis', vmin=0, vmax=threshold_CD3e,
            xticklabels=False, yticklabels=False, ax=axs[0,1])
axs[0,1].set_title("IF map CD3e", fontsize=15)
axs[0,1].set_aspect(image_aspect)

sns.heatmap(prob_grid_MIL_T, cmap=cmap_cell_types['T_cells'], vmin=0.2, vmax=0.8,
            xticklabels=False, yticklabels=False, ax=axs[0,2])
axs[0,2].set_title("T prob. map MIL", fontsize=15)
axs[0,2].set_aspect(image_aspect)

axs[1,0].axis('off')

sns.heatmap(fluorescence_grid_CD4, cmap='viridis', vmin=0, vmax=threshold_CD4,
            xticklabels=False, yticklabels=False, ax=axs[1,1])
axs[1,1].set_title("IF map CD4", fontsize=15)
axs[1,1].set_aspect(image_aspect)

sns.heatmap(prob_grid_MIL_CD4T, cmap=cmap_cell_types['CD4_T_cells'], vmin=0.2, vmax=0.8,
            xticklabels=False, yticklabels=False, ax=axs[1,2])
axs[1,2].set_title("CD4+ T prob. map MIL", fontsize=15)
axs[1,2].set_aspect(image_aspect)

axs[2,0].axis('off')

sns.heatmap(fluorescence_grid_CD8a, cmap='viridis', vmin=0, vmax=threshold_CD8a,
            xticklabels=False, yticklabels=False, ax=axs[2,1])
axs[2,1].set_title("IF map CD8$\alpha$", fontsize=15)
axs[2,1].set_aspect(image_aspect)

sns.heatmap(prob_grid_MIL_CD8T, cmap=cmap_cell_types['CD8_T_cells'], vmin=0.2, vmax=0.8,
            xticklabels=False, yticklabels=False, ax=axs[2,2])
axs[2,2].set_title("CD8+ T prob. map MIL", fontsize=15)
axs[2,2].set_aspect(image_aspect)


output_dir_plot = r"C:\Users\20182460\Desktop\Master_thesis\Figures report\extension_cell_types\examples_T_cells"
plt.savefig(f"{output_dir_plot}/{slide_id}_T_cells.png", dpi=300, bbox_inches="tight")

# Show the figure (optional)
plt.show()
