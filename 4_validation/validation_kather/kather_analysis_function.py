import pandas as pd 
import pickle
import numpy as np 
import cv2
from validate_functions import create_value_grid
import matplotlib.pyplot as plt 
import seaborn as sns
import scipy.stats as stats
from sklearn.metrics import roc_auc_score, roc_curve

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

#SET CORRECT DIRECTORY
map_dir = "/home/evos/Outputs/CRC/Orion/Orion_predictions_112um_224pixels"

def AUC_kather(predictions, cell_type, cell_type_kather):

    slide_ids = list(predictions['slide_id'].unique())

    all_probs_pos = []
    all_probs_neg = []
    all_AUCS = {}
    all_significant = {}

    for slide_id in slide_ids:
        predictions_slide = predictions[predictions.slide_id == slide_id]
        predictions_slide = predictions_slide.reset_index(drop=True)
        prob_grid = create_value_grid(predictions_slide, cell_type)
        
        dir_map_pickle = f"{map_dir}/{slide_id}_gamma_densenet161-kather100k_cell_type_map.pkl"
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

    average_AUC = np.mean(list(all_AUCS.values()))
    print(f"Average AUC over all slides: {average_AUC:.4f}")
    
    stat_all, p_value_all = stats.mannwhitneyu(all_probs_pos, all_probs_neg, alternative='greater')
    #print(f"\nCombined Mann-Whitney U Test lasso: U statistic = {stat_all:.2f}, p-value = {p_value_all:.4f}")

    #Overall AUC for combined data
    combined_labels = [1] * len(all_probs_pos) + [0] * len(all_probs_neg)
    combined_probs = all_probs_pos + all_probs_neg
    combined_auc = roc_auc_score(combined_labels, combined_probs)
    print(f"Combined AUC = {combined_auc:.4f}")
    
    all_results = {}
    all_results["AUC_values"] = all_AUCS
    all_results['significance'] = all_significant
    all_results["average_AUC"] = average_AUC
    all_results["all_labels_probs"] = [combined_labels, combined_probs]
    all_results["combined AUC"] = [combined_auc, p_value_all]
    
    print('Finished kather analysis')
        
    return all_results 