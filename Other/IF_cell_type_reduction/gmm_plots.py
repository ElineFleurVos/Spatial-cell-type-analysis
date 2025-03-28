# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 21:10:59 2024

@author: 20182460
"""

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
from IHC_reduction_functions import falserate

#%%

sc_tables_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Data\CRC\Orion\Orion_single_cell_tables"

all_sc_tables = []
for filename in os.listdir(sc_tables_dir):
    if filename.endswith(".csv"): 
        file_dir = os.path.join(sc_tables_dir, filename)
        single_cell_table = pd.read_csv(file_dir, sep=",", index_col=0)
        all_sc_tables.append(single_cell_table)
        
#%%
column_names = single_cell_table.columns
marker_columns = []
for df in all_sc_tables:
    # Convert the 'CD45' column to numeric, coercing errors to NaN
    numeric_values = pd.to_numeric(df['CD8a'], errors='coerce')
    
    nan_count = numeric_values.isna().sum()
    print(f'Number of NaN values in this DataFrame: {nan_count}')

    # types_of_values = numeric_values.apply(type)
    # print("Types of Values in the Column:")
    # print(types_of_values.value_counts())

    numeric_values.dropna(inplace=True)
    marker_columns.append(numeric_values.values)
    
markersample_nolog = np.concatenate(marker_columns, dtype=np.float64)
markersample = np.log1p(markersample_nolog)

min_value = np.min(markersample)
max_value = np.max(markersample) 
average_value = np.mean(markersample)

# Print the results
print(f'Minimum Value: {min_value:.2f}')
print(f'Maximum Value: {max_value:.2f}')
print(f'Average Value: {average_value:.2f}')

#%%
minval = np.percentile(markersample, 1)
FDR = 0.05 
    
filtered_sample = markersample[markersample > minval]
gmm = GaussianMixture(n_components=2, random_state=None, n_init=10) #random_state=None gives different results every run 
gmm.fit(filtered_sample.reshape(-1, 1))

# Get the means and variances of the two components
means = gmm.means_.flatten()
variances = gmm.covariances_.flatten()

# Identify the "negative" and "positive" components by their means
minid = np.argmin(means)
maxid = np.argmax(means) 

# Define search range based on the GMM components
#norm.pdf(means[maxid], means[maxid], np.sqrt(variances[maxid])):calculates the value of the probability density 
#function at the mean of the positive Gaussian distribution. It tells us how likely it is to observe a 
#value equal to the mean in that distribution.
if gmm.weights_[maxid] * norm.pdf(means[maxid], means[maxid], np.sqrt(variances[maxid])) > \
   gmm.weights_[minid] * norm.pdf(means[minid], means[minid], np.sqrt(variances[minid])):
    search_range = np.linspace(np.percentile(filtered_sample, 2), np.percentile(filtered_sample, 70), 1000)
else:
    search_range = np.linspace(np.percentile(filtered_sample, 10), np.percentile(filtered_sample, 98), 1000)

# Search for the cutoff
obj_func = np.array([(falserate(cutoff, gmm) - FDR) ** 2 for cutoff in search_range]) #align estimated FDR with desired FDR
cutoff = search_range[np.argmin(obj_func)]

# Determine which cells are positive based on the cutoff
pos_cells = markersample > cutoff

#%%
# Plotting the results
plt.hist(markersample, bins=1000, density=True, alpha=0.3, color='k')
x_vals = np.linspace(minval, max(markersample), 1000)
plt.plot(x_vals, gmm.weights_[minid] * norm.pdf(x_vals, means[minid], np.sqrt(variances[minid])), 'g', linewidth=2)
plt.plot(x_vals, gmm.weights_[maxid] * norm.pdf(x_vals, means[maxid], np.sqrt(variances[maxid])), 'b', linewidth=2)
plt.axvline(x=cutoff, color='r', linewidth=2)
plt.text(cutoff+100, plt.ylim()[1]*0.7, f'Cutoff={cutoff:.2f}\n%pos={np.mean(pos_cells)*100:.1f}%', fontsize=12)
plt.xlim(0,3000)
plt.ylim(0,0.010)
plt.show()
