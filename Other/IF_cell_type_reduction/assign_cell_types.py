# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 10:43:19 2024

@author: 20182460
"""

import pickle 
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import scipy.io
import h5py

#%%
single_cell_dir1 = r"C:\Users\20182460\Desktop\Master_thesis\Code\Data\CRC\Orion\Orion_single_cell_tables\P37_S29-CRC01.csv"
single_cell_dir4 = r"C:\Users\20182460\Desktop\Master_thesis\Code\Data\CRC\Orion\Orion_single_cell_tables\P37_S32-CRC04.csv"
data = pd.read_csv(single_cell_dir4, sep=",", index_col=0)
png_dir4 = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\CRC\Orion\Orion_HE_png\18459_LSP10388_US_SCAN_OR_001__091155-registered.png"
png_dir1 = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\CRC\Orion\Orion_HE_png\18459_LSP10353_US_SCAN_OR_001__093059-registered.png"
he_image = mpimg.imread(png_dir4)

#%%
output_path = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\CRC\Orion"
with open(f"{output_path}/marker_thresholds_global.pkl", 'rb') as file:
    thresholds_global = pickle.load(file)
    
thresholds_global['CD8a'] = 1300
positive_percentages = {}
for marker, threshold in thresholds_global.items():
    data[f'{marker}_positive'] = data[marker] > threshold
    #data[f'{marker}_positive'] = np.log1p(data[marker]) > threshold
    #scaler = StandardScaler()
    #data[f'{marker}_positive'] = scaler.fit_transform(data[[marker]]) > threshold
    
for marker in thresholds_global.keys():
    positive_column = f'{marker}_positive'
    # Calculate the percentage of True values
    positive_count = data[positive_column].sum()
    total_count = len(data)
    percentage_positive = (positive_count / total_count) * 100
    
    # Store the result in a dictionary
    positive_percentages[marker] = percentage_positive
    
data['cell_type'] = 'non-proliferating'
data.loc[data['Ki67_positive'] == 1, 'cell_type'] = ' proliferating'

data.loc[((data['Pan-CK_positive'] == 1) | (data['E-cadherin_positive'] == 1)), 'cell_type'] += ' tumor'
data.loc[data['CD45_positive'] == 1, 'cell_type'] += ' immune'
data.loc[(data['CD45_positive'] == 1) & (data['CD3e_positive'] == 1), 'cell_type'] += ' T'
data.loc[(data['CD45_positive'] == 1) & (data['CD3e_positive'] == 1) & (data['CD8a_positive'] == 1), 'cell_type'] += ' CD8T'   
data.loc[(data['CD45_positive'] == 1) & (data['CD3e_positive'] == 1) & (data['CD4_positive'] == 1), 'cell_type'] += ' CD4T'
data.loc[(data['CD45_positive'] == 1) & (data['CD20_positive'] == 1), 'cell_type'] += ' B cell'
data.loc[(data['CD45_positive'] == 1) & (data['CD3e_positive'] == 0) & (data['CD20_positive'] == 0), 'cell_type'] += ' macrophage'
#proli_data.loc[(proli_data['CD45_positive'] == 0) & (proli_data['Pan-CK_positive'] == 0) & (proli_data['E-cadherin_positive'] == 0), 'cell_type'] += ' stroma'
data.loc[(data['CD45_positive'] == 0) & (data['Pan-CK_positive'] == 0) & (data['E-cadherin_positive'] == 0) & (data['CD31_positive'] == 1), 'cell_type'] += ' stroma endothelial'
data.loc[(data['CD45_positive'] == 0) & (data['Pan-CK_positive'] == 0) & (data['E-cadherin_positive'] == 0) & (data['SMA_positive'] == 1), 'cell_type'] += ' stroma vascular_smooth_muscle'
proli_data = data[data['cell_type'].str.contains(' proliferating', na=False)]

# proli_data = data[data['Ki67_positive'] == 1]
# proli_data.loc[((proli_data['Pan-CK_positive'] == 1) | (proli_data['E-cadherin_positive'] == 1)), 'cell_type'] += ' tumor'
# proli_data.loc[proli_data['CD45_positive'] == 1, 'cell_type'] += ' immune'
# proli_data.loc[(proli_data['CD45_positive'] == 1) & (proli_data['CD3e_positive'] == 1), 'cell_type'] += ' T'
# proli_data.loc[(proli_data['CD45_positive'] == 1) & (proli_data['CD3e_positive'] == 1) & (proli_data['CD8a_positive'] == 1), 'cell_type'] += ' CD8T'   
# proli_data.loc[(proli_data['CD45_positive'] == 1) & (proli_data['CD3e_positive'] == 1) & (proli_data['CD4_positive'] == 1), 'cell_type'] += ' CD4T'
# proli_data.loc[(proli_data['CD45_positive'] == 1) & (proli_data['CD20_positive'] == 1), 'cell_type'] += ' B cell'
# proli_data.loc[(proli_data['CD45_positive'] == 1) & (proli_data['CD3e_positive'] == 0) & (proli_data['CD20_positive'] == 0), 'cell_type'] += ' macrophage'
# #proli_data.loc[(proli_data['CD45_positive'] == 0) & (proli_data['Pan-CK_positive'] == 0) & (proli_data['E-cadherin_positive'] == 0), 'cell_type'] += ' stroma'
# proli_data.loc[(proli_data['CD45_positive'] == 0) & (proli_data['Pan-CK_positive'] == 0) & (proli_data['E-cadherin_positive'] == 0) & (proli_data['CD31_positive'] == 1), 'cell_type'] += 'stroma endothelial'
# proli_data.loc[(proli_data['CD45_positive'] == 0) & (proli_data['Pan-CK_positive'] == 0) & (proli_data['E-cadherin_positive'] == 0) & (proli_data['SMA_positive'] == 1), 'cell_type'] += 'stroma vascular_smooth_muscle'


# data['cell_type'] = [['non-proliferating'] for _ in range(len(data))]
# data.loc[data['Ki67_positive'] == 1, 'cell_type'] = data.loc[data['Ki67_positive'] == 1, 'cell_type'].apply(
#     lambda x: ['proliferating'] if 'non-proliferating' in x else x)
# proli_data = data[data['Ki67_positive'] == 1]

# proli_data.loc[((proli_data['Pan-CK_positive'] == 1) | (proli_data['E-cadherin_positive'] == 1)), 'cell_type'] = proli_data.loc[
#     ((proli_data['Pan-CK_positive'] == 1) | (proli_data['E-cadherin_positive'] == 1)), 'cell_type'].apply(lambda x: x + ['tumor cell'])

#%% plot CD8a overlay

cd8_cells = data[data['CD8a_positive'] == 1]
#cd8_cells = proli_data[proli_data['CD8a_positive'] == 1]
#cd8_cells = proli_data[proli_data['cell_type'].str.contains('CD8T', na=False)]
x_centroids = cd8_cells['X_centroid']*0.01
y_centroids = cd8_cells['Y_centroid']*0.01

image_height, image_width, _ = he_image.shape
out_of_bounds = []
for x, y in zip(x_centroids, y_centroids):
    if not (0 <= x < image_width) or not (0 <= y < image_height):
        out_of_bounds.append((x, y))

# Display out-of-bounds points if any
if out_of_bounds:
    print("Out-of-bounds coordinates:")
    for coord in out_of_bounds:
        print(f"X: {coord[0]}, Y: {coord[1]}")
else:
    print("All coordinates are within bounds.")

plt.figure(figsize=(10, 10), dpi=300)
plt.imshow(he_image)
plt.axis('off')  # Hide axis
plt.scatter(x_centroids, y_centroids, color='blue', label='CD8+ Cells', s=0.5, alpha=0.7)
plt.show()

#%% plot negative cells overlay

negative_cells = proli_data[proli_data['cell_type'] == ' proliferating']
#negative_cells = data[(data['cell_type'] == ' proliferating') | (data['cell_type'] == 'non-proliferating')]
x_centroids_negative = negative_cells['X_centroid']*0.01
y_centroids_negative = negative_cells['Y_centroid']*0.01

image_height, image_width, _ = he_image.shape
out_of_bounds = []
for x, y in zip(x_centroids_negative, y_centroids_negative):
    if not (0 <= x < image_width) or not (0 <= y < image_height):
        out_of_bounds.append((x, y))

# Display out-of-bounds points if any
if out_of_bounds:
    print("Out-of-bounds coordinates:")
    for coord in out_of_bounds:
        print(f"X: {coord[0]}, Y: {coord[1]}")
else:
    print("All coordinates are within bounds.")

plt.figure(figsize=(10, 10), dpi=300)
plt.imshow(he_image)
plt.axis('off')  # Hide axis
plt.scatter(x_centroids_negative, y_centroids_negative, color='red', label='CD8+ Cells', s=2, alpha=0.7)
plt.show()


#TO DO 
#Fix thresholds (maybe by normalizing)
#check how many have both tumor and immune 
#Validate by plotting non-assigned cells


