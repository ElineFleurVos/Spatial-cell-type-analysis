# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 08:46:54 2024

@author: 20182460
"""
import os 
import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
from IHC_reduction_functions import falserate, find_cutoff
import pickle 
from sklearn.preprocessing import StandardScaler

#%%

#sc_tables_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Data\CRC\Orion\Orion_single_cell_tables"
sc_tables_dir = "/home/evos/Data/CRC/Orion/Orion_single_cell_tables"

all_sc_tables = []
for filename in os.listdir(sc_tables_dir):
    print(filename)
    if filename.endswith(".csv"): 
        file_dir = os.path.join(sc_tables_dir, filename)
        single_cell_table = pd.read_csv(file_dir, sep=",", index_col=0)
        all_sc_tables.append(single_cell_table)
    print(f"{filename}: Columns -> {single_cell_table.columns}")
        
#%%

column_names = single_cell_table.columns
interesting_markers = ['Ki67', 'CD45', 'Pan-CK', 'E-cadherin', 'CD31','SMA', 'CD3e', 'CD20', 
                       'CD4', 'CD8a']

FDR = 0.05 
marker_thresholds = {}

for marker in interesting_markers:
    marker_columns = []
    for df in all_sc_tables:
        # Convert the 'CD45' column to numeric, coercing errors to NaN
        numeric_values = pd.to_numeric(df[marker], errors='coerce')
        
        nan_count = numeric_values.isna().sum()
        #print(f'Number of NaN values in this DataFrame: {nan_count}')
    
        # types_of_values = numeric_values.apply(type)
        # print("Types of Values in the Column:")
        # print(types_of_values.value_counts())
    
        numeric_values.dropna(inplace=True)
        
        marker_columns.append(numeric_values.values)
        
    markersample_nonorm = np.concatenate(marker_columns, dtype=np.float64)
    markersample_nonorm_reshaped = markersample_nonorm.reshape(-1, 1)
    scaler = StandardScaler()
    markersample = scaler.fit_transform(markersample_nonorm_reshaped)
    #markersample = np.log1p(markersample_nolog)
    minval = np.percentile(markersample, 1)
    
    min_value = np.min(markersample)
    max_value = np.max(markersample) 
    average_value = np.mean(markersample)
    
    # Print the results
    print(f'Minimum Value: {min_value:.2f}')
    print(f'Maximum Value: {max_value:.2f}')
    print(f'Average Value: {average_value:.2f}')
    
    cutoff = find_cutoff(markersample, minval, FDR)
    marker_thresholds[marker] = cutoff

output_path = "/home/evos/Outputs/CRC/Orion"

with open(f"{output_path}/marker_thresholds_norm.pkl", 'wb') as file:
    pickle.dump(marker_thresholds, file)


    