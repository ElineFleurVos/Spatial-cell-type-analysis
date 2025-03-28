# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 20:50:24 2025

@author: 20182460
"""

import sys
import pandas as pd
import pickle

sys.path.append("C:/Users/20182460/Documents/GitHub/THESIS/analysis_kather")
sys.path.append("C:/Users/20182460/Documents/GitHub/THESIS/IF_analysis")
from kather_analysis_function import AUC_kather
from IF_analysis_function import IF_scoring

#%% norm los

# pred_dir_no_norm_MSE = "C:/Users/20182460/Desktop/Master_thesis/Code/Outputs/FINAL RESULTS/CRC/MIL/MIL_loss_norm_tests/simple_nn_no_norm_MSE/all_cell_types.csv"
# pred_dir_no_norm_cor = "C:/Users/20182460/Desktop/Master_thesis/Code/Outputs/FINAL RESULTS/CRC/MIL/MIL_loss_norm_tests/simple_nn_no_norm_cor/all_cell_types.csv"
# pred_dir_norm_MSE = "C:/Users/20182460/Desktop/Master_thesis/Code/Outputs/FINAL RESULTS/CRC/MIL/MIL_loss_norm_tests/simple_nn_norm_MSE/all_cell_types.csv"
# pred_dir_norm_cor = "C:/Users/20182460/Desktop/Master_thesis/Code/Outputs/FINAL RESULTS/CRC/MIL/MIL_loss_norm_tests/simple_nn_norm_cor/all_cell_types.csv"

# pred_no_norm_MSE = pd.read_csv(pred_dir_no_norm_MSE, sep=",")
# pred_no_norm_cor = pd.read_csv(pred_dir_no_norm_cor, sep=",")
# pred_norm_MSE = pd.read_csv(pred_dir_norm_MSE, sep=",")
# pred_norm_cor = pd.read_csv(pred_dir_norm_cor, sep=",")

# print(pred_no_norm_MSE.columns)

# output_dir = "C:/Users/20182460/Desktop/Master_thesis/Code/Outputs/FINAL RESULTS/CRC/MIL/MIL_loss_norm_tests/IF_results"

# test_names = ['norm_cor', 'no_norm_MSE', 'no_norm_cor', 'norm_MSE']
# predictions = [pred_norm_cor, pred_no_norm_MSE, pred_no_norm_cor, pred_norm_MSE]

# double_marker = True
# marker1 = 'Pan-CK' 
# marker2 = 'E-cadherin'
# cell_type = "tumor_purity"
# file_name = "cadpan"

# percentile = 99
# remove_empty_tiles = True
# num_bins = 100
# min_tile_count = 1
# # Loop over each method
# for test_name, pred in zip(test_names, predictions):
#     print(test_name)
#     # Compute IF scores
#     all_results = IF_scoring(pred, percentile, double_marker, marker1, marker2, cell_type, remove_empty_tiles, num_bins, min_tile_count)

#     with open(f'{output_dir}/{test_name}_{file_name}.pkl', 'wb') as f:
#         pickle.dump(all_results, f)
        
#%% UNI

pred_dir_nostain_FF = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\MIL\MIL_UNI_FF_FFPE_tests\EpiT\EpiT_norm_nostain_FF.csv"
pred_dir_macenko_FF = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\MIL\MIL_UNI_FF_FFPE_tests\EpiT\EpiT_norm_macenko_FF.csv"
pred_dir_nostain_FFPE = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\MIL\MIL_UNI_FF_FFPE_tests\EpiT\EpiT_norm_nostain_FFPE.csv"
pred_dir_macenko_FFPE = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\MIL\MIL_UNI_FF_FFPE_tests\EpiT\EpiT_norm_macenko_FFPE.csv"
pred_nostain_FF  = pd.read_csv(pred_dir_nostain_FF, sep="\t")
pred_macenko_FF = pd.read_csv(pred_dir_macenko_FF, sep="\t")
pred_nostain_FFPE = pd.read_csv(pred_dir_nostain_FFPE, sep="\t")
pred_macenko_FFPE = pd.read_csv(pred_dir_macenko_FFPE, sep="\t")

output_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\MIL\MIL_UNI_FF_FFPE_tests\IF_results"

test_names = ['nostain_FF', 'macenko_FF', 'nostain_FFPE', 'macenko_FFPE']
predictions = [pred_nostain_FF, pred_macenko_FF, pred_nostain_FFPE, pred_macenko_FFPE]

double_marker = True
marker1 = 'Pan-CK' 
marker2 = 'E-cadherin'
cell_type = "tumor_purity"
file_name = "cadpan"

percentile = 99
remove_empty_tiles = True
num_bins = 100
min_tile_count = 3
# Loop over each method
for test_name, pred in zip(test_names, predictions):
    print(test_name)
    # Compute IF scores
    all_results = IF_scoring(pred, percentile, double_marker, marker1, marker2, cell_type, remove_empty_tiles, num_bins, min_tile_count)

    with open(f'{output_dir}/{test_name}_{file_name}.pkl', 'wb') as f:
        pickle.dump(all_results, f)
        

















