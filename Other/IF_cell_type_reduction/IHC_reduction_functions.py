# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 09:55:25 2024

@author: 20182460
"""

from scipy.stats import norm
import numpy as np
from sklearn.mixture import GaussianMixture

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

def find_cutoff(markersample, minval, FDR):
    
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
    
    return cutoff