# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 14:33:20 2024

@author: 20182460
"""

import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.metrics import jaccard_score
from scipy.spatial.distance import dice

DEFAULT_CELL_TYPES = ["CAFs", "T_cells", "endothelial_cells", "tumor_purity"]

def plot_orion_cell_type_map(kather_grid, plot=True, plot_legend=True, ax=None): #patch_length=224):
    
    # index = np.fliplr(np.array(model_output["coordinates"])[:, :2] / patch_length).astype(int)
    # grid_shape = index.max(axis=0) + 1
    # cat_img = np.zeros(grid_shape)
    # cat_img[tuple(index.T)] = np.asarray(model_output["predictions"])
    
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
    
    colors = list(plt.cm.tab10.colors)[:9]  # Exclude the last color
    colors[0] = (1, 1, 1, 1)
    cmap = ListedColormap(colors)
    
    if plot:
        if ax is None:
            plt.figure()
            ax = plt.gca()
        im = ax.imshow(kather_grid, cmap=cmap, vmin=0, vmax=9, interpolation="none")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
            
        if plot_legend:
            # Add colorbar next to the plot
            cbar = plt.colorbar(im, ax=ax)
            num_classes = len(LABEL_DICT)
            cbar.set_ticks(np.arange(num_classes) + 0.5)
            cbar.set_ticklabels(list(LABEL_DICT.keys()))
        
        plt.axis('off')
        
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

def assign_cell_types(
    slide_data,
    cell_types=None,
    threshold=0.5,
):
    """
    Assign nodes with cell types based on their value being greater than the threshold (by default=0.5)

    Args:
        slide_data (DataFrame): dataframe containing at least the cell type predictions for the tiles
        cell_types (list): list of cell types
        threshold (float): threshold to assign a cell type to a tile

    Returns
        (DataFrame): Tiles with assigned cell types len(cell_types) columns with True/False indicating whether the tile is assigned with the cell type and corresponding metadata.
    """
    if cell_types is None:
        cell_types = DEFAULT_CELL_TYPES

    has_cell_types = slide_data[cell_types] > threshold
    node_cell_types = slide_data.copy()
    node_cell_types[cell_types] = has_cell_types
    return node_cell_types

def calc_jaccard_dice(binary_map1, binary_map2):
    
    # # Jaccard similarity
    # intersection = np.sum((spotlight_binary == 1) & (kather_binary == 1))
    # union = np.sum((spotlight_binary == 1) | (kather_binary == 1))
    # intersection = intersection.sum()
    # union = union.sum()
    # jaccard_similarity = intersection / union if union > 0 else 0

    # # Dice coefficient
    # dice_coefficient = (2 * intersection) / (np.sum(spotlight_binary == 1) + np.sum(kather_binary == 1))
    
    spotlight_flat = binary_map1.flatten()
    kather_flat = binary_map2.flatten()

    jaccard_similarity = jaccard_score(spotlight_flat, kather_flat)
    dice_coefficient = 1 - dice(spotlight_flat, kather_flat)
    
    return (jaccard_similarity, dice_coefficient)
