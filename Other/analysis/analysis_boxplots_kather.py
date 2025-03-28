# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 11:54:32 2024

@author: 20182460
"""

import pandas as pd 
import pickle
import numpy as np 
import cv2
from validate_functions import create_value_grid
import matplotlib.pyplot as plt 
import seaborn as sns
import scipy.stats as stats
from sklearn.metrics import roc_auc_score, roc_curve

#%%

#predictions_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\multitask_lasso\test_tile_predictions_proba_macenko.csv"
#predictions_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\MIL\MIL_final_pcchip\all_cell_types.csv"
#predictions_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\MIL\MIL_old_deconv\all_cell_types.csv"
predictions_dir = "C:/Users/20182460/Desktop/Master_thesis/Code/Outputs/FINAL RESULTS/CRC/MIL/MIL_UNI_FF_FFPE_tests/EpiT/EpiT_norm_macenko_FF.csv"
predictions = pd.read_csv(predictions_dir, sep="\t")
#predictions = pd.read_csv(predictions_dir, sep=",")
slide_ids = list(predictions['slide_id'].unique())

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

cell_type = 'tumor_purity'
cell_type_name = 'Tumor cells'
cell_type_kather = "TUM"

#%% all together 

all_probs_pos = []
all_probs_neg = []
all_significant = {}
all_AUCS = {}

for slide_id in slide_ids:
    print(slide_id)
    predictions_slide = predictions[predictions.slide_id == slide_id]
    predictions_slide = predictions_slide.reset_index(drop=True)
    prob_grid = create_value_grid(predictions_slide, cell_type)
    
    dir_map_pickle = f"C:/Users/20182460/Desktop/Master_thesis/Code/Outputs/CRC/Orion/Orion_predictions_112um_224pixels/{slide_id}_gamma_densenet161-kather100k_cell_type_map.pkl"
    # Open the pickle file in read mode
    with open(dir_map_pickle, 'rb') as f:
        # Load the object from the pickle file
        model_output = pickle.load(f)
    
    #Reshape the Kather map to match the probability grid
    patch_length = 224
    index = np.fliplr(np.array(model_output["coordinates"])[
                      :, :2] / patch_length).astype(int)
    grid_shape = index.max(axis=0) + 1
    kather_grid = np.zeros(grid_shape)
    kather_grid[tuple(index.T)] = np.asarray(model_output["predictions"])

    kather_resized = cv2.resize(
        kather_grid, (prob_grid.shape[1], prob_grid.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    #Flatten the grids for easier processing
    prob_grid_flat = prob_grid.values.ravel()
    kather_resized_flat = kather_resized.ravel()

    #Separate the probabilities into positive (tumor) and negative (non-tumor) based on true labels
    probs_pos = prob_grid_flat[kather_resized_flat == LABEL_DICT[cell_type_kather]]
    probs_neg = prob_grid_flat[kather_resized_flat != LABEL_DICT[cell_type_kather]]
    
    # Remove NaN values
    probs_pos = probs_pos[~np.isnan(probs_pos)]
    probs_neg = probs_neg[~np.isnan(probs_neg)]
    
    #Store all probabilities across slides
    all_probs_pos.extend(probs_pos)
    all_probs_neg.extend(probs_neg)
    
    # Mann-Whitney U Test for the current slide
    stat, p_value = stats.mannwhitneyu(probs_pos, probs_neg, alternative='greater')
    all_significant[slide_id] = {'U_stat': stat, 'p_value': p_value}
   
    # AUC for the current slide
    labels = [1] * len(probs_pos) + [0] * len(probs_neg)
    all_probs = list(probs_pos) + list(probs_neg)
    auc_score = roc_auc_score(labels, all_probs)
    all_AUCS[slide_id] = auc_score

#%%
#Separate results 
average_auc = np.mean(list(all_AUCS.values()))
max_auc = np.max(list(all_AUCS.values()))
min_auc = np.min(list(all_AUCS.values()))
slide_with_max_auc = max(all_AUCS, key=all_AUCS.get)
slide_with_min_auc = min(all_AUCS, key=all_AUCS.get)

print(f"Average AUC over all slides: {average_auc:.4f}")
print(f"Slide with Maximum AUC: {slide_with_max_auc} (AUC = {max_auc:.4f})")
print(f"Slide with Minimum AUC: {slide_with_min_auc} (AUC = {min_auc:.4f})")

print("\nSlides with p-values higher than 0.05 (not significant):")
n = 0
for slide_id, results in all_significant.items():
    p_value = results['p_value']
    if p_value > 0.05:
        # Get the corresponding AUC value from all_AUCS
        auc_value = all_AUCS.get(slide_id, None)  # Use None if no AUC is available for the slide
        print(f"Slide {slide_id}: p-value = {p_value:.4f}, AUC = {auc_value}")
        n += 1
print('Number of slides for which tumor tiles does not have significtantly higher probabilities:', n)

#Combined results 
#Overall Mann-Whitney U test for combined data
stat_all, p_value_all = stats.mannwhitneyu(all_probs_pos, all_probs_neg, alternative='greater')
print(f"\nCombined Mann-Whitney U Test: U statistic = {stat_all:.2f}, p-value = {p_value_all:.4f}")

#Overall AUC for combined data
combined_labels = [1] * len(all_probs_pos) + [0] * len(all_probs_neg)
combined_probs = all_probs_pos + all_probs_neg
combined_auc = roc_auc_score(combined_labels, combined_probs)
print(f"Combined AUC = {combined_auc:.4f}")

# plt.figure(figsize=(6, 4))
# sns.boxplot(data=[all_probs_pos, all_probs_neg])
# plt.xticks([0, 1], [cell_type_name, f'No {cell_type_name}'])
# plt.title(f'MIL')
# plt.ylabel(f'Probability of patch containg {cell_type_name}')
# plt.xlabel('Categories')
# plt.show()

all_results = {}
all_results["AUC_values"] = all_AUCS
all_results['significance'] = all_significant
all_results["average_AUC"] = average_auc
all_results["all_labels_probs"] = [combined_labels, combined_probs]
all_results["combined AUC"] = [combined_auc, p_value_all]

#output_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\multitask_lasso"
#output_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\MIL\MIL_final_pcchip"
#output_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\MIL\MIL_old_deconv"
output_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\MIL\MIL_UNI_FF_FFPE_tests\EpiT"
with open(f"{output_dir}/AUC_results_tumor_FFPE_macenko.pkl", "wb") as file:
    pickle.dump(all_results, file)

#%% single slide 

#slide_id_nr = 6
#slide_ids = list(predictions['slide_id'].unique())
#slide_id = slide_ids[slide_id_nr]
slide_id = '18459_LSP10353_US_SCAN_OR_001__093059-registered'
#slide_id = '18459_LSP10388_US_SCAN_OR_001__091155-registered'

predictions = predictions[predictions.slide_id == slide_id]
predictions = predictions.reset_index(drop=True)
prob_grid = create_value_grid(predictions, cell_type)

dir_map_pickle = f"C:/Users/20182460/Desktop/Master_thesis/Code/Outputs/CRC/Orion/Orion_predictions_112um_224pixels/{slide_id}_gamma_densenet161-kather100k_cell_type_map.pkl"
# Open the pickle file in read mode
with open(dir_map_pickle, 'rb') as f:
    # Load the object from the pickle file
    model_output = pickle.load(f)
    
patch_length = 224
index = np.fliplr(np.array(model_output["coordinates"])[
                  :, :2] / patch_length).astype(int)
grid_shape = index.max(axis=0) + 1
kather_grid = np.zeros(grid_shape)
kather_grid[tuple(index.T)] = np.asarray(model_output["predictions"])

kather_resized = cv2.resize(
    kather_grid, (prob_grid.shape[1], prob_grid.shape[0]), interpolation=cv2.INTER_NEAREST)

prob_grid_flat = prob_grid.values.ravel()
kather_resized_flat = kather_resized.ravel()

probs_pos = prob_grid_flat[kather_resized_flat == LABEL_DICT[cell_type_kather]]
probs_neg = prob_grid_flat[kather_resized_flat != LABEL_DICT[cell_type_kather]]

#remove nan's. Some tiles exist in the kather plot, but were not made in the original pipeline. 
#Tiling differs between the two 
probs_pos = probs_pos[~np.isnan(probs_pos)]
probs_neg = probs_neg[~np.isnan(probs_neg)]

data = [probs_pos, probs_neg]
labels = [cell_type, f'No {cell_type}']

# Create the boxplot
plt.figure(figsize=(6, 4))
sns.boxplot(data=data)
plt.xticks([0, 1], labels)
plt.title('Boxplots of Probabilities')
plt.ylabel(f'Probability of patch containing {cell_type}')
plt.xlabel('Categories')
plt.show()

mean_pos = np.mean(probs_pos)
median_pos = np.median(probs_pos)

mean_neg = np.mean(probs_neg)
median_neg= np.median(probs_neg)

median_diff = median_pos-median_neg

print(f"Mean pos: {mean_pos:.4f}")
print(f"Median pos: {median_pos:.4f}")

print(f"Mean neg: {mean_neg:.4f}")
print(f"Median neg: {median_neg:.4f}")

print(f"Difference in median: {median_diff:.4f}")

#statistical tests single slide 

# Perform a Mann-Whitney U test (data was not normal, so we can not do a standard T-test)
stat, p_value = stats.mannwhitneyu(probs_pos, probs_neg, alternative='greater') #'two-sided
print(f"Mann-Whitney U Test: U statistic = {stat:.2f}, p-value = {p_value:.4f}")

# # Optional: Perform a t-test if you assume normal distribution
# t_stat, t_p_value = stats.ttest_ind(probs_pos, probs_neg, equal_var=False)
# print(f"T-Test: t statistic = {t_stat:.2f}, p-value = {t_p_value:.4f}")

# Labels: 1 for "tumor" (positive class), 0 for "non-tumor" (negative class)
labels = [1] * len(probs_pos) + [0] * len(probs_neg)
all_probs = list(probs_pos) + list(probs_neg)

# Calculate AUC
auc_score = roc_auc_score(labels, all_probs)
print(f"AUC Score: {auc_score:.4f}")

#%%
fpr, tpr, thresholds = roc_curve(labels, all_probs)
plt.figure(figsize=(3.5,3.5))
# Plot ROC curve
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random guessing
plt.xlabel("False Positive Rate", fontsize=15)
plt.ylabel("True Positive Rate", fontsize=15)
plt.title("ROC Curve", fontsize=15)
plt.legend(loc="lower right", fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

output_dir_plot = r"C:\Users\20182460\Desktop\Master_thesis\Figures report"
plt.savefig(f"{output_dir_plot}/example_ROC_curve.png", dpi=300, bbox_inches="tight")
plt.show()


































