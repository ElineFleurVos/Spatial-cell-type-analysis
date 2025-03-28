# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 12:02:23 2024

@author: 20182460
"""
import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

#%%

#single_cell_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Data\CRC\Orion\Orion_single_cell_tables\P37_S29-CRC01.csv"   
single_cell_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Data\CRC\Orion\Orion_single_cell_tables\P37_S32-CRC04.csv"
single_cell_data = pd.read_csv(single_cell_dir, sep=",", index_col=0)

column_names = single_cell_data.columns
single_cell_data_subset = single_cell_data.head(10)

#%%

markersample = single_cell_data['CD8a'].values
min_value = np.min(markersample)  # Minimum value
max_value = np.max(markersample)  # Maximum value
average_value = np.mean(markersample)

# Print the results
print(f'Minimum Value: {min_value:.2f}')
print(f'Maximum Value: {max_value:.2f}')
print(f'Average Value: {average_value:.2f}')

#%%
#set min value to filter out noise
minval = np.percentile(markersample, 1)

# Q1 = np.percentile(markersample, 25)
# Q3 = np.percentile(markersample, 75)
# IQR = Q3 - Q1
# lower_threshold = Q1 - 1.5 * IQR

#%%

filtered_sample = markersample[markersample > minval]
gmm = GaussianMixture(n_components=2, random_state=None, n_init=10) #random_state=None gives different results every run 
gmm.fit(filtered_sample.reshape(-1, 1))

# Get the means and variances of the two components
means = gmm.means_.flatten()
variances = gmm.covariances_.flatten()

# Identify the "negative" and "positive" components by their means
minid = np.argmin(means)
maxid = np.argmax(means) 

FDR = 0.05 

# Define search range based on the GMM components
#norm.pdf(means[maxid], means[maxid], np.sqrt(variances[maxid])):calculates the value of the probability density 
#function at the mean of the positive Gaussian distribution. It tells us how likely it is to observe a 
#value equal to the mean in that distribution.
if gmm.weights_[maxid] * norm.pdf(means[maxid], means[maxid], np.sqrt(variances[maxid])) > \
   gmm.weights_[minid] * norm.pdf(means[minid], means[minid], np.sqrt(variances[minid])):
    search_range = np.linspace(np.percentile(filtered_sample, 2), np.percentile(filtered_sample, 70), 1000)
else:
    search_range = np.linspace(np.percentile(filtered_sample, 10), np.percentile(filtered_sample, 98), 1000)

def falserate(x, gmm_model):
    """Estimate the false discovery rate at a given x. so this function computes 
    the ratio of false positives to all positives detected at the given threshold. 
    """
    # Get the parameters from the GMM
    means = gmm_model.means_.flatten() #means of each Gaussian component
    covariances = gmm_model.covariances_.flatten() #variances of each Gaussian component
    weights = gmm_model.weights_ #the weights (prior probabilities) for each component in the GMM, 
    #indicating how much of the data is modeled by each Gaussian

    # Identify indices for the min and max means
    minid = np.argmin(means) #index Gaussian marker negative distribution
    maxid = np.argmax(means) #index Gaussian marker positive distribution

    alpha = weights[minid] #weight of the negative Gaussian component. Proportion of marker negative cells
    beta = weights[maxid] #weight of the positive Gaussian distribution. Proportion of marker positive cells 

    # Compute the cumulative distribution functions
    Phi_min = norm.cdf(x, means[minid], np.sqrt(covariances[minid])) #proportion of negative cells with
    #marker values less than or equal to cutoff value (true negatives)
    Phi_plus = norm.cdf(x, means[maxid], np.sqrt(covariances[maxid])) #proportion of positive cells with
    # marker values less than or equal to cutoff value (false negatives)

    # Compute the false rate
    #1-Phi_min --> proportion of negative cells incorrectly classified as potitive (false positives)
    #1-Phi_plus --> proportions of positive cells correctly classified as postitives (true positives)
    ratio = alpha * (1 - Phi_min) / (beta * (1 - Phi_plus) + alpha * (1 - Phi_min))
    
    return ratio

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
plt.text(cutoff+100, 0.006, f'Cutoff={cutoff:.2f}\n%pos={np.mean(pos_cells)*100:.1f}%', fontsize=12)
plt.xlim(0,1000)
plt.ylim(0,0.04)
plt.show()

#%%
count_true = sum(pos_cells)


