# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 20:53:46 2025

@author: 20182460
"""

import pandas as pd 
import numpy as np

#This code averages the quantification scores from the general quantifications to get one score per cell type instead of multiple. 
#These quantifications are then used for training the MIL pipeline

input_path_first_gen = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\deconvolutions\deconvolutions_first_gen"
first_gen_deconvolution = pd.read_csv(f"{input_path_first_gen}/ensembled_selected_tasks.csv", sep="\t", index_col=0)

first_gen_deconvolution.set_index('TCGA_sample', inplace=True)
columns_to_remove = ['Stromal score', 'tumor purity (ESTIMATE)', 'Immune score', 'CD8 T cells (Thorsson)', 'TIL score']
first_gen_deconvolution.drop(columns=columns_to_remove, inplace=True, errors='ignore')

tumor_columns = ['tumor purity (ABSOLUTE)', 'tumor purity (EPIC)']
T_columns = ['Cytotoxic cells', 'Effector cells', 'CD8 T cells (quanTIseq)']
endo_columns = ['Endothelial cells (xCell)', 'Endothelial cells (EPIC)', 'Endothelium']

first_gen_deconvolution['EpiT'] = first_gen_deconvolution[tumor_columns].apply(
    lambda row: row.mean() if row.notna().all() else np.nan, axis=1)
first_gen_deconvolution['T'] = first_gen_deconvolution[T_columns].apply(
    lambda row: row.mean() if row.notna().all() else np.nan, axis=1)
first_gen_deconvolution['Endothelial'] = first_gen_deconvolution[endo_columns].apply(
    lambda row: row.mean() if row.notna().all() else np.nan, axis=1)

columns_keep = ['EpiT', 'T', 'Endothelial']
first_gen_final = first_gen_deconvolution.loc[:, columns_keep]

output_path = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\deconvolutions\deconvolutions_first_gen\averaged_deconv_first_gen.csv"
first_gen_final.to_csv(output_path, index=True, sep=',')