# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 11:27:53 2024

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


single_cell_dir1 = r"C:\Users\20182460\Desktop\Master_thesis\Code\Data\CRC\Orion\Orion_single_cell_tables\P37_S29-CRC01.csv"
single_cell_dir4 = r"C:\Users\20182460\Desktop\Master_thesis\Code\Data\CRC\Orion\Orion_single_cell_tables\P37_S32-CRC04.csv"
data = pd.read_csv(single_cell_dir4, sep=",", index_col=0)
all_sample_dir = r"C:\Users\20182460\Documents\GitHub\THESIS\IHC_cell_type_reduction\sumAllsample.csv"
mat_data = pd.read_csv(all_sample_dir, sep=",", index_col=0)

#%%
interesting_markers = ['Ki67', 'CD45', 'Pan-CK', 'E-cadherin', 'CD31','SMA', 'CD3e', 'CD20', 
                       'CD4', 'CD8a']
interesting_fractions = ['Ki67', 'CD45', 'Pan_CK', 'E_cadherin', 'CD31','SMA', 'CD3e', 'CD20', 
                       'CD4', 'CD8a']

marker_thresholds = {}
for i in range(len(interesting_markers)):
    marker = interesting_markers[i]
    has_nans = data[marker].isna().any()
    print(has_nans) 
    marker_list = data[marker].to_list()
    
    desired_fraction = mat_data.loc['C04', 'mean_'+interesting_fractions[i]+'p']
    
    # Step 1: Sort the marker values
    sorted_values = np.sort(marker_list)
    # Step 2: Calculate the index for the desired fraction
    threshold_index = int((1 - desired_fraction) * len(sorted_values))
    # Step 3: Set the threshold
    threshold = sorted_values[threshold_index]
    print(f"{marker}, {desired_fraction}, {threshold}")
    
    marker_thresholds[marker] = threshold

output_path = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\CRC\Orion"
with open(f"{output_path}/marker_thresholds_new.pkl", 'wb') as file:
    pickle.dump(marker_thresholds, file)




