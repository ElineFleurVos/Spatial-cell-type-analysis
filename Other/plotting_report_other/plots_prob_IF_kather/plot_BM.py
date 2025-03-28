# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 09:39:36 2025

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

single_cell_metadata_dir = "single_cell_metadata.txt"
single_cell_metadata = pd.read_csv(single_cell_metadata_dir, sep="\t")
metadata_dict = dict(zip(single_cell_metadata['slide_id'], single_cell_metadata['single_cell_file']))

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
prob_grid_MIL_B = create_value_grid(predictions_MIL, 'B')
prob_grid_MIL_macro = create_value_grid(predictions_MIL, 'macro')

#%% Fluorescence map 

f_grids_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\fluorescence_grids"
grids_CD20 = joblib.load(f"{f_grids_dir}/fluorescence_grids_CD20/grids_{single_cell_file_name}.pkl")
fluorescence_grid_CD20 = grids_CD20['average_fluorescence_grid']
grids_CD163 = joblib.load(f"{f_grids_dir}/fluorescence_grids_CD163/grids_{single_cell_file_name}.pkl")
fluorescence_grid_CD163 = grids_CD163['average_fluorescence_grid']
grids_CD68 = joblib.load(f"{f_grids_dir}/fluorescence_grids_CD68/grids_{single_cell_file_name}.pkl")
fluorescence_grid_CD68 = grids_CD68['average_fluorescence_grid']
fluorescence_grid_macro = (6/16)*fluorescence_grid_CD163 + (10/16)*fluorescence_grid_CD68

global_info_output_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\fluorescence_grids\global_information"

with open(f"{global_info_output_dir}/global_info_CD20_99.pkl", "rb") as file:
    threshold_CD20 = pickle.load(file)['threshold']
with open(f"{global_info_output_dir}/global_info_CD68_163_99.pkl", "rb") as file:
    threshold_CD163_68 = pickle.load(file)['threshold']
    
    
#%% Plotting

cmap_cell_types = {"B_cells": "Reds", "macrophages": "Blues",
                   "CD8_T_cells": "Greens", "endothelial_cells": "Purples"}


#(22,2.1)
fig, axs = plt.subplots(2, 3, figsize=(15,7), gridspec_kw={
                        'width_ratios': [1, 1.3, 1.3]})
image_height, image_width, _ = image_png.shape
image_aspect = image_width / image_height
image_aspect = 1.0

#axs = axs.flatten()

axs[0,0].axis('off')  # Hide axis
axs[0,0].imshow(image_png, aspect='auto')
axs[0,0].set_title('H&E image', fontsize=15)
axs[0,0].set_aspect(image_aspect)

sns.heatmap(fluorescence_grid_CD20, cmap='viridis', vmin=0, vmax=threshold_CD20,
            xticklabels=False, yticklabels=False, ax=axs[0,1])
axs[0,1].set_title("IF map CD20", fontsize=15)
axs[0,1].set_aspect(image_aspect)

sns.heatmap(prob_grid_MIL_B, cmap=cmap_cell_types['B_cells'], vmin=0.2, vmax=0.8,
            xticklabels=False, yticklabels=False, ax=axs[0,2])
axs[0,2].set_title("B prob. map MIL", fontsize=15)
axs[0,2].set_aspect(image_aspect)

axs[1,0].axis('off')

sns.heatmap(fluorescence_grid_macro, cmap='viridis', vmin=0, vmax=threshold_CD163_68,
            xticklabels=False, yticklabels=False, ax=axs[1,1])
axs[1,1].set_title("IF map CD163 + CD68", fontsize=15)
axs[1,1].set_aspect(image_aspect)

sns.heatmap(prob_grid_MIL_macro, cmap=cmap_cell_types['macrophages'], vmin=0.2, vmax=0.8,
            xticklabels=False, yticklabels=False, ax=axs[1,2])
axs[1,2].set_title("Macrophages prob. map MIL", fontsize=15)
axs[1,2].set_aspect(image_aspect)


output_dir_plot = r"C:\Users\20182460\Desktop\Master_thesis\Figures report\extension_cell_types\examples_BM"
plt.savefig(f"{output_dir_plot}/{slide_id}_BM_cells.png", dpi=300, bbox_inches="tight")

# Show the figure (optional)
plt.show()

    































