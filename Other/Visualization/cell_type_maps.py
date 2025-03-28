# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:33:27 2024

@author: 20182460
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import sys
import git
import pandas as pd
import os 
import matplotlib.image as mpimg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

# os.add_dll_directory("C:/Users/20182460/openslide-win64-20231011/bin")
# import openslide
OPENSLIDE_PATH = r'C:\Program Files\openslide-win64-20230414\bin' #path to openslide bin 
if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        from openslide import OpenSlide
else:
    from openslide import OpenSlide

REPO_DIR= git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(f"{REPO_DIR}/libs")
sys.path.append(r"C:\Users\20182460\Documents\GitHub\THESIS\libs")
 
# Own modules
import features.utils as utils
from model.constants import *

#%%

def create_value_grid(slide_data, cell_type):
    #get step size by finding the minimum distance between Y-coordinates. Step size is 
    #dependent on the magnitude of the image. 
    coord_Y = slide_data["Coord_Y"].unique()
    coord_X = slide_data["Coord_X"].unique()
    coord_Y.sort()
    coord_X.sort()

    step = float('inf') 
    # Iterate through the array, comparing adjacent elements
    for i in range(len(coord_Y) - 1):
        diff = coord_Y[i+1] - coord_Y[i]
        if diff < step:
            step = diff

    max_coord_Y = slide_data['Coord_Y'].max()
    min_coord_Y = slide_data['Coord_Y'].min()
    max_coord_X = slide_data['Coord_X'].max()
    min_coord_X = slide_data['Coord_X'].min()

    coord_X_all= np.arange(min_coord_X, max_coord_X + step, step)
    coord_Y_all = np.arange(min_coord_Y, max_coord_Y + step, step)
    coord_Y_all.sort()
    coord_X_all.sort()
    value_grid = pd.DataFrame(columns=coord_X_all, index=coord_Y_all)

    # Assign values to all tiles
    for x in coord_X_all:
        for y in coord_Y_all:
            mask = (slide_data["Coord_X"] == x) & (slide_data["Coord_Y"] == y)
            if mask.sum() > 0:
                value_grid.loc[y, x] = slide_data[mask].index.values[0]
                value_grid.loc[y, x] = slide_data[mask][cell_type].values[0]
    value_grid = value_grid.apply(pd.to_numeric)
    
    return value_grid

#%%

predictions_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\CRC\FF\tcga_validation_tile_predictions_proba.csv"
#predictions_dir_FFPE = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\SKCM\Complete dataset\Predictions\tcga_validation_tile_predictions_proba_FFPE.csv"
output_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\CRC\visualizations"

png_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\CRC\FF\png_files"
all_filenames = os.listdir(png_dir)
file_name = all_filenames[3]
slide_submitter_id = file_name.split(".")[0] 

vmin=0.20
vmax=0.80
cell_types=None
cmap_cell_types=None
fontsize=12
cut_slide=True

png = fr"{png_dir}\{slide_submitter_id}.png"
predictions_unfiltered = pd.read_csv(predictions_dir, sep="\t")#, dtype={'MFP': 'str'})  
#mfp_value = predictions_unfiltered.loc[predictions_unfiltered['slide_submitter_id'] == slide_submitter_id]

#%%

#set up combined cell type map
colors = [
        'lightgrey',    # Other (0)
        '#984ea3',      # Both (1)  
        '#4daf4a',      # type1 (2)
        '#e41a1c',      # type2 (3)
    ]
type1 = "T_cells"
type2=  "tumor_purity"
cmap = LinearSegmentedColormap.from_list('Custom', colors, 4)

if cell_types is None:
    cell_types = DEFAULT_CELL_TYPES
if cmap_cell_types is None:
    cmap_cell_types = {"T_cells": "Greens", "CAFs": "Blues", "tumor_purity":"Reds", "endothelial_cells":"Purples"}

if len(np.unique(predictions_unfiltered.slide_submitter_id)) > 1:
    if slide_submitter_id is None:
        raise Exception("If slide_submitter_id is not specified then predictions should contain only data for one slide")
    else:
        predictions = predictions_unfiltered[predictions_unfiltered.slide_submitter_id == slide_submitter_id]
predictions = predictions.reset_index(drop=True)

#%%

if cut_slide==True:
    fig, axes = plt.subplots(1, len(cell_types)+2, sharex=False, sharey=False, figsize=(16, 2))
else:
    fig, axes = plt.subplots(1, len(cell_types)+2, sharex=False, sharey=False, figsize=(20, 2)) 

# Plot probability type maps in column 2,3,4 and 5
for j, cell_type in enumerate(cell_types):
    prob_grid = create_value_grid(slide_data=predictions, cell_type=cell_type)
    if cut_slide==True:  
        max_x=prob_grid.shape[1]-1
        nan_column = prob_grid.columns[prob_grid.isnull().all()].min()
        if nan_column is not None:
            prob_grid = prob_grid.loc[:, :nan_column]
        max_x_new = prob_grid.shape[1]-1
    #plot heatmap
    sns.heatmap(prob_grid,cmap=cmap_cell_types[cell_type], vmin=vmin, vmax=vmax,ax=axes[j+1], cbar=False)

#plot png image in column 1
img = mpimg.imread(png)
if cut_slide == True:
    keep_ratio = max_x_new/max_x #calculate ratio from cut in grid
    img_width = img.shape[1] 
    img_max_x = int(keep_ratio*img_width) #get maximum x for png image 
    left = 0  # X-coordinate of the left edge
    top = 0   # Y-coordinate of the top edge
    right = img_max_x+200 # X-coordinate of the right edge
    bottom = img.shape[0]-1  # Y-coordinate of the bottom edge
    img = img[top:bottom, left:right]
axes[0].imshow(img, aspect="auto") 

#plot combined cell type map in column 6
cell_type_assignments = utils.assign_cell_types(predictions)
cell_type_assignments["cell_type"] = 0
cell_type_assignments.loc[cell_type_assignments[type1],"cell_type"] = 2
cell_type_assignments.loc[cell_type_assignments[type2],"cell_type"] = 3
cell_type_assignments.loc[(cell_type_assignments[type1] & cell_type_assignments[type2]), "cell_type"] = 1
cell_type_grid = create_value_grid(cell_type_assignments, "cell_type")
if cut_slide==True:
    first_nan_column = cell_type_grid.columns[cell_type_grid.isnull().all()].min()
    if first_nan_column is not None:
        cell_type_grid = cell_type_grid.loc[:, :first_nan_column]  
sns.heatmap(data=cell_type_grid, cmap=cmap, linecolor='lightgray', cbar=False, ax=axes[len(cell_types)+1])

#remove all labels, ticks and ticklabels
for j in range(len(cell_types)+2):
    axes[j].set_xticklabels("")
    axes[j].set_yticklabels("")
    axes[j].set_xlabel("")
    axes[j].set_ylabel("")
    axes[j].get_xaxis().set_ticks([])
    axes[j].get_yaxis().set_ticks([])

    for spine in ['top', 'right', "left", "bottom"]:
        axes[j].spines[spine].set_visible(False)

#set x label png slide
slide_type = "FF"
MFP = predictions.loc[0, "MFP"]
axes[0].set_xlabel(f"{slide_type} slide {MFP}", fontsize=fontsize)

#set legend combined cell type map
legend_elements= []
for c, l in zip((colors), ['Other', 'Tumor cells and T cells',  "T cells", 'Tumor cells',]): 
        legend_elements.append(  Patch(facecolor=c, edgecolor=None,
                        label=l),)
if cut_slide==True:
    axes[len(cell_types)+1].legend(handles=legend_elements, frameon=False, title="Cell type(s)", loc="upper left", bbox_to_anchor=(0, -0.15),  ncol=1) 
else:
    axes[len(cell_types)+1].legend(handles=legend_elements, frameon=False, title="Cell type(s)", loc="upper left", bbox_to_anchor=(0, -0.15),  ncol=2) 

# Add custom colorbar
norm = plt.Normalize(vmin, vmax)
for j, cell_type in enumerate(cell_types):
    axes[j+1].set_ylabel(cell_type.replace("_", " ").replace("purity", "cells"), size=fontsize)
    sm = plt.cm.ScalarMappable(cmap=cmap_cell_types[cell_type], norm=norm, )
    sm.set_array([])
    axins = inset_axes(axes[j+1],
                    width="100%",
                    height="3%",
                    loc='lower center',
                    borderpad=-2
                )
    fig.colorbar(sm, cax=axins, orientation="horizontal",label="probability",
    ticks=np.linspace(vmin, vmax, 4),
    spacing='proportional',)

plt.savefig(f"{output_dir}/{slide_submitter_id}_cell_type_maps.pdf", bbox_inches="tight", dpi=600)
plt.savefig(f"{output_dir}/{slide_submitter_id}_cell_type_maps.png")
plt.show()

# #%%
# def plot_png_prob_maps_1_by_6(slide_submitter_id, predictions_dir, outpur_dir, png_dir, vmin=0.20, vmax=0.80, cell_types=None, cmap_cell_types=None, fontsize=12, cut_slide=True):

#     png = fr"{png_dir}\{slide_submitter_id}.png"
#     predictions = pd.read_csv(predictions_dir, sep="\t")  

#     #set up combined cell type map
#     colors = [
#             'lightgrey',    # Other (0)
#             '#984ea3',      # Both (1)  
#             '#e41a1c',      # type1 (2)
#             '#4daf4a',      # type2 (3)
#         ]
#     type1 = "T_cells"
#     type2=  "tumor_purity"
#     cmap = LinearSegmentedColormap.from_list('Custom', colors, 4)

#     if cell_types is None:
#         cell_types = DEFAULT_CELL_TYPES
#     if cmap_cell_types is None:
#         cmap_cell_types = {"T_cells": "Greens", "CAFs": "Blues", "tumor_purity":"Reds", "endothelial_cells":"Purples"}

#     if len(np.unique(predictions.slide_submitter_id)) > 1:
#         if slide_submitter_id is None:
#             raise Exception("If slide_submitter_id is not specified then predictions should contain only data for one slide")
#         else:
#             predictions = predictions[predictions.slide_submitter_id == slide_submitter_id]
#     predictions = predictions.reset_index(drop=True)
    
#     if cut_slide==True:
#         fig, axes = plt.subplots(1, len(cell_types)+2, sharex=False, sharey=False, figsize=(16, 2))
#     else:
#         fig, axes = plt.subplots(1, len(cell_types)+2, sharex=False, sharey=False, figsize=(20, 2)) 

#     # Plot probability type maps in column 2,3,4 and 5
#     for j, cell_type in enumerate(cell_types):
#         prob_grid = create_value_grid(slide_data=predictions, cell_type=cell_type)
#         if cut_slide==True:  
#             max_x=prob_grid.shape[1]-1
#             nan_column = prob_grid.columns[prob_grid.isnull().all()].min()
#             if nan_column is not None:
#                 prob_grid = prob_grid.loc[:, :nan_column]
#             max_x_new = prob_grid.shape[1]-1
#         #plot heatmap
#         sns.heatmap(prob_grid,cmap=cmap_cell_types[cell_type], vmin=vmin, vmax=vmax,ax=axes[j+1], cbar=False)
    
#     #plot png image in column 1
#     img = mpimg.imread(png)
#     if cut_slide == True:
#         keep_ratio = max_x_new/max_x #calculate ratio from cut in grid
#         img_width = img.shape[1] 
#         img_max_x = int(keep_ratio*img_width) #get maximum x for png image 
#         left = 0  # X-coordinate of the left edge
#         top = 0   # Y-coordinate of the top edge
#         right = img_max_x+200 # X-coordinate of the right edge
#         bottom = img.shape[0]-1  # Y-coordinate of the bottom edge
#         img = img[top:bottom, left:right]
#     axes[0].imshow(img, aspect="auto") 
    
#     #plot combined cell type map in column 6
#     cell_type_assignments = utils.assign_cell_types(predictions)
#     cell_type_assignments["cell_type"] = 0
#     cell_type_assignments.loc[cell_type_assignments[type1],"cell_type"] = 2
#     cell_type_assignments.loc[cell_type_assignments[type2],"cell_type"] = 3
#     cell_type_assignments.loc[(cell_type_assignments[type1] & cell_type_assignments[type2]), "cell_type"] = 1
#     cell_type_grid = create_value_grid(cell_type_assignments, "cell_type")
#     if cut_slide==True:
#         first_nan_column = cell_type_grid.columns[cell_type_grid.isnull().all()].min()
#         if first_nan_column is not None:
#             cell_type_grid = cell_type_grid.loc[:, :first_nan_column]  
#     sns.heatmap(data=cell_type_grid, cmap=cmap, linecolor='lightgray', cbar=False, ax=axes[len(cell_types)+1])
    
#     #remove all labels, ticks and ticklabels
#     for j in range(len(cell_types)+2):
#         axes[j].set_xticklabels("")
#         axes[j].set_yticklabels("")
#         axes[j].set_xlabel("")
#         axes[j].set_ylabel("")
#         axes[j].get_xaxis().set_ticks([])
#         axes[j].get_yaxis().set_ticks([])
    
#         for spine in ['top', 'right', "left", "bottom"]:
#             axes[j].spines[spine].set_visible(False)
    
#     #set x label png slide
#     slide_type = "FF"
#     MFP = predictions.loc[0, "MFP"]
#     axes[0].set_xlabel(f"{slide_type} slide {MFP}", fontsize=fontsize)

#     #set legend combined cell type map
#     legend_elements= []
#     for c, l in zip(reversed(colors), ['T cells', 'Tumor cells',  "Tumor cells and T cells", 'Other',]): 
#             legend_elements.append(  Patch(facecolor=c, edgecolor=None,
#                             label=l),)
#     if cut_slide==True:
#         axes[len(cell_types)+1].legend(handles=legend_elements, frameon=False, title="Cell type(s)", loc="upper left", bbox_to_anchor=(0, -0.15),  ncol=1) 
#     else:
#         axes[len(cell_types)+1].legend(handles=legend_elements, frameon=False, title="Cell type(s)", loc="upper left", bbox_to_anchor=(0, -0.15),  ncol=2) 

#     # Add custom colorbar
#     norm = plt.Normalize(vmin, vmax)
#     for j, cell_type in enumerate(cell_types):
#         axes[j+1].set_ylabel(cell_type.replace("_", " ").replace("purity", "cells"), size=fontsize)
#         sm = plt.cm.ScalarMappable(cmap=cmap_cell_types[cell_type], norm=norm, )
#         sm.set_array([])
#         axins = inset_axes(axes[j+1],
#                         width="100%",
#                         height="3%",
#                         loc='lower center',
#                         borderpad=-2
#                     )
#         fig.colorbar(sm, cax=axins, orientation="horizontal",label="probability",
#         ticks=np.linspace(vmin, vmax, 4),
#         spacing='proportional',)

#     #plt.savefig(f"{output_dir}/{slide_submitter_id}_cell_type_maps.pdf", bbox_inches="tight", dpi=600)
#     # plt.savefig(f"{output_dir}/{slide_submitter_id}_cell_type_maps.png")
#     plt.show()
    
# slide_submitter_id="TCGA-A6-5666-01A-01-BS1"

# predictions_dir_FF = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\CRC\FF\tcga_validation_tile_predictions_proba.csv"
# #predictions_dir_FFPE = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\SKCM\Complete dataset\Predictions\tcga_validation_tile_predictions_proba_FFPE.csv"
# output_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\CRC\visualizations"

# png_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\CRC\FF\png_files"
# all_filenames = os.listdir(png_dir)
# file_name = all_filenames[0]
# slide_submitter_id = file_name.split(".")[0] 


# plot_png_prob_maps_1_by_6(slide_submitter_id, predictions_dir_FF, output_dir, png_dir=png_dir, cut_slide=True)


#%% plotting histology image 

# png_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\SKCM\Visualisations\PNG slides\TCGA-EE-A2ME-06A-01-TSA.png"
# predictions_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\SKCM\Complete dataset\Predictions\tcga_validation_tile_predictions_proba_FF.csv"
# predictions = pd.read_csv(predictions_dir, sep="\t")  
# output_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\SKCM\Visualisations"
# #slide_submitter_id="TCGA-3N-A9WC-06A-01-TSA"
# #slide_submitter_id = "TCGA-3N-A9WD-06A-01-TSA"
# slide_submitter_id="TCGA-EE-A2ME-06A-01-TSA"
# vmin=0.20
# vmax=0.80
# cell_types=None
# cmap_cell_types=None
# fontsize=12

# #set up settings for combined cell type map
# colors = [
#         'lightgrey',    # Other (0)
#         '#984ea3',      # Both (1)  
#         '#e41a1c',      # type1 (2)
#         '#4daf4a',      # type2 (3)
#     ]
# type1 = "T_cells"
# type2=  "tumor_purity"

# cmap = LinearSegmentedColormap.from_list('Custom', colors, 4)

# if cell_types is None:
#     cell_types = DEFAULT_CELL_TYPES
# if cmap_cell_types is None:
#     cmap_cell_types = {"T_cells": "Greens", "CAFs": "Blues", "tumor_purity":"Reds", "endothelial_cells":"Purples"}

# if len(np.unique(predictions.slide_submitter_id)) > 1:
#     if slide_submitter_id is None:
#         raise Exception("If slide_submitter_id is not specified then predictions should contain only data for one slide")
#     else:
#         predictions = predictions[predictions.slide_submitter_id == slide_submitter_id]

# predictions = predictions.reset_index(drop=True)
# fig, axes = plt.subplots(1, len(cell_types)+2, sharex=False, sharey=False, figsize=(16, 2)) #(20,2)

# # Plot probability type maps in column 2,3,4 and 5
# for j, cell_type in enumerate(cell_types):
#     prob_grid = create_value_grid(slide_data=predictions, cell_type=cell_type)
#     #slice grid
#     max_x=prob_grid.shape[1]-1
#     nan_column = prob_grid.columns[prob_grid.isnull().all()].min()
#     if nan_column is not None:
#         prob_grid = prob_grid.loc[:, :nan_column]
#     max_x_new = prob_grid.shape[1]-1
#     #plot heatmap
#     sns.heatmap(prob_grid,cmap=cmap_cell_types[cell_type], vmin=vmin, vmax=vmax,ax=axes[j+1], cbar=False)

# keep_ratio = max_x_new/max_x
# img = mpimg.imread(png_dir)
# img_width = img.shape[1] 
# img_max_x = int(keep_ratio*img_width)
# left = 0  # X-coordinate of the left edge
# top = 0   # Y-coordinate of the top edge
# right = img_max_x+200 # X-coordinate of the right edge
# bottom = img.shape[0]-1  # Y-coordinate of the bottom edge
# cropped_img = img[top:bottom, left:right]
# axes[0].imshow(cropped_img, aspect="auto") 

# cell_type_assignments = utils.assign_cell_types(predictions)
# cell_type_assignments["cell_type"] = 0
# cell_type_assignments.loc[cell_type_assignments[type1],"cell_type"] = 2
# cell_type_assignments.loc[cell_type_assignments[type2],"cell_type"] = 3
# cell_type_assignments.loc[(cell_type_assignments[type1] & cell_type_assignments[type2]), "cell_type"] = 1
# cell_type_grid = create_value_grid(cell_type_assignments, "cell_type")

# #find first nan columns and split dataframe there
# first_nan_column = cell_type_grid.columns[cell_type_grid.isnull().all()].min()
# if first_nan_column is not None:
#     cell_type_grid = cell_type_grid.loc[:, :first_nan_column]
    
# sns.heatmap(data=cell_type_grid, cmap=cmap, linecolor='lightgray', cbar=False, ax=axes[len(cell_types)+1])
    
# for j in range(len(cell_types)+2):
#     axes[j].set_xticklabels("")
#     axes[j].set_yticklabels("")
#     axes[j].set_xlabel("")
#     axes[j].set_ylabel("")
#     axes[j].get_xaxis().set_ticks([])
#     axes[j].get_yaxis().set_ticks([])

#     for spine in ['top', 'right', "left", "bottom"]:
#         axes[j].spines[spine].set_visible(False)

# slide_type = "FF"
# MFP = predictions.loc[0, "MFP"]
# axes[0].set_xlabel(f"{slide_type} slide {MFP}", fontsize=12)

# legend_elements= []
# for c, l in zip(reversed(colors), ['T cells', 'Tumor cells',  "Tumor cells and T cells", 'Other',]): 
#         legend_elements.append(  Patch(facecolor=c, edgecolor=None,
#                         label=l),)
# axes[len(cell_types)+1].legend(handles=legend_elements, frameon=False, title="Cell type(s)", loc="upper left", bbox_to_anchor=(0, -0.15),  ncol=1) 
# #axes[len(cell_types)+1].legend(handles=legend_elements, frameon=False, title="Cell type(s)", bbox_to_anchor=(2.5, 0.8),  ncol=1) 

# # Add custom colorbar
# norm = plt.Normalize(vmin, vmax)
# for j, cell_type in enumerate(cell_types):
#     axes[j+1].set_ylabel(cell_type.replace("_", " ").replace("purity", "cells"), size=fontsize)
#     sm = plt.cm.ScalarMappable(cmap=cmap_cell_types[cell_type], norm=norm, )
#     sm.set_array([])
#     axins = inset_axes(axes[j+1],
#                     width="100%",
#                     height="3%",
#                     loc='lower center',
#                     borderpad=-2
#                 )
#     fig.colorbar(sm, cax=axins, orientation="horizontal",label="probability",
#     ticks=np.linspace(vmin, vmax, 4),
#     spacing='proportional',)

# if slide_submitter_id is None:
#     slide_submitter_id = np.unique(predictions["slide_submitter_id"])[0]

# plt.savefig(f"{output_dir}/{slide_submitter_id}_cell_type_maps.pdf", bbox_inches="tight", dpi=600)
# # plt.savefig(f"{output_dir}/{slide_submitter_id}_cell_type_maps.png")
# plt.show()
  