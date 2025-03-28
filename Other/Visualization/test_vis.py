# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 09:18:54 2024

@author: 20182460
"""

import joblib 
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt 
import sys

sys.path.append(r"C:\Users\20182460\Documents\GitHub\THESIS\libs")
import features.utils as utils
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

#%%

# slide_submitter_id="TCGA-EE-A2ME-06A-01-TSA"
# predictions_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\SKCM\Complete dataset\Predictions\tcga_validation_tile_predictions_proba_FF.csv"
# predictions = pd.read_csv(predictions_dir, sep="\t")  
# predictions = predictions[predictions.slide_submitter_id == slide_submitter_id]

# ordered_df_predictions = predictions.sort_values(by='Coord_X')
# ordered_df_predictions.set_index('Coord_X', inplace=True)
# temp = predictions.pivot("Coord_Y", "Coord_X", 'T_cells')
# temp = temp.sort_index(ascending=True)
# plt.figure()
# sns.heatmap(temp, cbar=False)

#%%

slide_submitter_id="TCGA-EE-A2ME-06A-01-TSA"
predictions_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\SKCM\Complete dataset\Predictions\tcga_validation_tile_predictions_proba_FF.csv"
predictions = pd.read_csv(predictions_dir, sep="\t")  

predictions = predictions[predictions.slide_submitter_id == slide_submitter_id]
predictions = predictions.reset_index(drop=True)
slide_data = predictions

cell_type_assignments = utils.assign_cell_types(slide_data)

colors = [
        'lightgrey',    # Other (0)
        '#984ea3',      # Both (1)  
        '#e41a1c',      # Neighbor (2)
        '#4daf4a',      # Center (3)
    ]
center = "T_cells"
neighbor="tumor_purity"

labels = ['Other', 'Both', 'Center']

cmap = LinearSegmentedColormap.from_list('Custom', colors, 4)

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
    print(step)

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


cell_type_assignments = utils.assign_cell_types(slide_data)
cell_type_assignments["cell_type"] = 0
cell_type_assignments.loc[cell_type_assignments[neighbor],"cell_type"] = 2
cell_type_assignments.loc[cell_type_assignments[center],"cell_type"] = 3
cell_type_assignments.loc[(cell_type_assignments[neighbor] & cell_type_assignments[center]), "cell_type"] = 1

value_grid = create_value_grid(cell_type_assignments, "cell_type")
# cell_type_assignments = cell_type_assignments.pivot("Coord_Y", "Coord_X" ,"cell_type")
# cell_type_assignments = cell_type_assignments.sort_index(ascending=False)

g = sns.heatmap(data=value_grid, cmap=cmap, linecolor='lightgray', square=True, cbar=False)
g.set_xticklabels("")
g.set_yticklabels("")
g.set_xlabel("")
g.set_ylabel("")
g.get_xaxis().set_ticks([])
g.get_yaxis().set_ticks([])

legend_elements= []
for c, l in zip(reversed(colors), ['T cells', 'Tumor cells',  "Tumor cells and T cells", 'Other',]): 
        legend_elements.append(  Patch(facecolor=c, edgecolor=None,
                        label=l),)
#g.legend(handles=legend_elements, frameon=False, title="Cell type(s)", bbox_to_anchor=(0.5, 0)) 
g.legend(handles=legend_elements, frameon=False, title="Cell type(s)", bbox_to_anchor=(1.5, 0.8)) 
g.legend(handles=legend_elements, frameon=False, title="Cell type(s)", loc="upper left", bbox_to_anchor=(0, -0.15),  ncol=2) 


#%%

# #slide_submitter_id="TCGA-3N-A9WC-06A-01-TSA"
# #slide_submitter_id = "TCGA-3N-A9WD-06A-01-TSA"
# slide_submitter_id="TCGA-EE-A2ME-06A-01-TSA"
# predictions_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\SKCM\Complete dataset\Predictions\tcga_validation_tile_predictions_proba_FF.csv"
# predictions = pd.read_csv(predictions_dir, sep="\t")  

# predictions = predictions[predictions.slide_submitter_id == slide_submitter_id]
# predictions = predictions.reset_index(drop=True)
# nodes = predictions.index

# slide_data=predictions

# #get step size by finding the minimum distance between Y-coordinates. Step size is 
# #dependent on the magnitude of the image. 
# coord_Y = slide_data["Coord_Y"].unique()
# coord_X = slide_data["Coord_X"].unique()
# coord_Y.sort()
# coord_X.sort()

# step = float('inf') 
# # Iterate through the array, comparing adjacent elements
# for i in range(len(coord_Y) - 1):
#     diff = coord_Y[i+1] - coord_Y[i]
#     if diff < step:
#         step = diff
# print(step)

# max_coord_Y = slide_data['Coord_Y'].max()
# min_coord_Y = slide_data['Coord_Y'].min()
# max_coord_X = slide_data['Coord_X'].max()
# min_coord_X = slide_data['Coord_X'].min()

# coord_X_all= np.arange(min_coord_X, max_coord_X + step, step)
# coord_Y_all = np.arange(min_coord_Y, max_coord_Y + step, step)
# coord_Y_all.sort()
# coord_X_all.sort()
# node_grid = pd.DataFrame(columns=coord_X_all, index=coord_Y_all)

# # Assign the tiles (nodes) by using the index of the node to the grid
# for x in coord_X_all:
#     for y in coord_Y_all:
#         mask = (slide_data["Coord_X"] == x) & (slide_data["Coord_Y"] == y)
#         if mask.sum() > 0:
#             node_grid.loc[y, x] = slide_data[mask].index.values[0]
#             node_grid.loc[y, x] = slide_data[mask][cell_type].values[0]
# node_grid = node_grid.apply(pd.to_numeric)

# plt.figure()
# sns.heatmap(node_grid, cbar=False)





