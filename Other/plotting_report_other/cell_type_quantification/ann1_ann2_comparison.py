# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 16:12:44 2025

@author: 20182460
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np
from scipy.stats import pearsonr

#%%

output_path = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\deconvolutions\final_sc_deconv_results"

rectangle_estimation1 =  pd.read_csv(f"{output_path}/ann1/cell_frac_ann1.csv", sep=",", index_col=0)
rectangle_estimation2 =  pd.read_csv(f"{output_path}/ann2/cell_frac_ann2.csv", sep=",", index_col=0)

# Update columns in annotation 2
rectangle_estimation2['T'] = rectangle_estimation2['CD4 T'] + rectangle_estimation2['CD8 T']
rectangle_estimation2['Stromal'] = rectangle_estimation2['Endothelial'] + rectangle_estimation2['Stromal other']

# Define the interesting columns for each annotation
# Annotation 1 (EpiT, T, B, Macro, Stromal)
interesting_columns_ann1 = ['EpiT', 'T', 'B', 'Macro', 'Stromal']

# Annotation 2 (CD4 T, CD8 T, Stromal, Endothelial, and others)
interesting_columns_ann2 = ['CD4 T', 'CD8 T', 'Endothelial']

# Extract the relevant columns from each annotation
data_ann1 = rectangle_estimation1[interesting_columns_ann1]
data_ann2 = rectangle_estimation2[interesting_columns_ann2]

# Melt the DataFrames to a long format
df_melted_ann1 = data_ann1.melt(var_name='Cell Type', value_name='Values')
df_melted_ann2 = data_ann2.melt(var_name='Cell Type', value_name='Values')

# Combine the melted DataFrames
df_melted_combined = pd.concat([df_melted_ann1, df_melted_ann2], ignore_index=True)

# Define custom order for the x-axis
custom_order = ['EpiT', 'T', 'CD4 T', 'CD8 T', 'B', 'Macro', 'Stromal', 'Endothelial']

# Create the boxplot with custom order
plt.figure(figsize=(9, 4.5))
sns.boxplot(data=df_melted_combined, x='Cell Type', y='Values', color="#2980b9", order=custom_order)

# Set the custom labels for the x-ticks
custom_labels = ['Tumor', 'T', 'CD4+ T', 'CD8+ T', 'B', 'Macrophages', 'Stromal', 'Endothelial']

# Set the custom labels on the x-ticks
plt.xticks(ticks=range(len(custom_labels)), labels=custom_labels, rotation=30, fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.ylabel("Fraction", fontsize=14)
plt.xlabel("Cell Type", fontsize=14)
plt.tight_layout()

# Save the plot
output_dir_plot = r"C:\Users\20182460\Desktop\Master_thesis\Figures report\Deconvolution"
plt.savefig(f"{output_dir_plot}/fractions_combined_custom_order.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()


#%% plot comparison ann1 versus ann2

interesting_columns_comparison_anns = ['EpiT', 'T', 'B', 'Macro', 'Stromal'] 
title_names = ['Tumor', 'T', 'B', 'Macrophages', 'Stromal']

# Set up the figure and subplots
n_cols = len(interesting_columns_comparison_anns)  # Number of columns for subplots
n_rows = 1  # Only one row since there are 5 columns

fig, axes = plt.subplots(n_rows, n_cols, figsize=(13.5, 3), constrained_layout=True)

# Create scatterplots for each interesting column
for i, column in enumerate(interesting_columns_comparison_anns):
    ax = axes[i]
    
    # Scatterplot
    ax.scatter(rectangle_estimation1[column], rectangle_estimation2[column], alpha=0.7, c="#2980b9", label="Data")
    
    # Add y=x line
    min_val = min(rectangle_estimation1[column].min(), rectangle_estimation2[column].min())
    max_val = max(rectangle_estimation1[column].max(), rectangle_estimation2[column].max())
    ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="y=x")
    
    # Titles and labels
    ax.set_title(title_names[i])
    ax.set_xlabel("Annotation 1")
    ax.set_ylabel("Annotation 2")
    ax.legend()

#plt.suptitle("Scatterplots Comparing Rectangle Estimation 1 and 2 (with y=x Line)", fontsize=16)
output_dir_plot = r"C:\Users\20182460\Desktop\Master_thesis\Figures report\Deconvolution"
plt.savefig(f"{output_dir_plot}/ann1_vs_ann2.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()


#%%plot comparison first generation versus second generation

output_path_first_gen = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\deconvolutions\deconvolutions_first_gen"
first_gen_deconvolution = pd.read_csv(f"{output_path_first_gen}/ensembled_selected_tasks.csv", sep="\t", index_col=0)

first_gen_deconvolution.set_index('TCGA_sample', inplace=True)
columns_to_remove = ['slide_submitter_id', 'sample_submitter_id','Stromal score', 'tumor purity (ESTIMATE)', 'Immune score', 'CD8 T cells (Thorsson)', 'TIL score']
first_gen_deconvolution.drop(columns=columns_to_remove, inplace=True, errors='ignore')

common_indices = first_gen_deconvolution.index.intersection(rectangle_estimation2.index)
first_gen_deconvolution = first_gen_deconvolution.loc[common_indices]
first_gen_deconvolution = first_gen_deconvolution.drop_duplicates()
first_gen_deconvolution = first_gen_deconvolution.drop('TCGA-WS-AB45-01', axis=0)

tumor_columns = ['tumor purity (ABSOLUTE)', 'tumor purity (EPIC)']
T_columns = ['Cytotoxic cells', 'Effector cells', 'CD8 T cells (quanTIseq)']
endo_columns = ['Endothelial cells (xCell)', 'Endothelial cells (EPIC)', 'Endothelium']

first_gen_deconvolution['EpiT'] = first_gen_deconvolution[tumor_columns].apply(
    lambda row: row.mean() if row.notna().all() else np.nan, axis=1)
first_gen_deconvolution['T'] = first_gen_deconvolution[T_columns].apply(
    lambda row: row.mean() if row.notna().all() else np.nan, axis=1)
first_gen_deconvolution['Endothelial'] = first_gen_deconvolution[endo_columns].apply(
    lambda row: row.mean() if row.notna().all() else np.nan, axis=1)

# Create a new dataframe by combining the columns
second_gen_deconv = pd.concat([rectangle_estimation1[['EpiT', 'T']], rectangle_estimation2[['Endothelial']]], axis=1)

common_indices = first_gen_deconvolution.index.intersection(second_gen_deconv.index)

# Filter both DataFrames to include only common indices
first_gen_filtered = first_gen_deconvolution.loc[common_indices]
rectangle_filtered = second_gen_deconv.loc[common_indices]

# Define the columns to compare
columns_to_compare = ['EpiT', 'T', 'Endothelial']
title_names2 = ['Tumor', 'T', 'Endothelial']

# Set up the figure and subplots
n_cols = len(columns_to_compare)  # Number of columns for subplots
fig, axes = plt.subplots(1, n_cols, figsize=(10, 3), constrained_layout=True)

# Create scatterplots with regression lines and Pearson correlation
for i, column in enumerate(columns_to_compare):
    ax = axes[i]
    
    # Rename the columns temporarily to avoid overlap during merge
    first_gen_filtered_renamed = first_gen_filtered[[column]].rename(columns={column: 'first_gen_' + column})
    rectangle_filtered_renamed = rectangle_filtered[[column]].rename(columns={column: 'rectangle_' + column})
    
    # Join the renamed columns
    valid_data = first_gen_filtered_renamed.join(rectangle_filtered_renamed, how='inner').dropna()
    
    # Extract x and y for the current cell type
    x = valid_data['first_gen_' + column]
    y = valid_data['rectangle_' + column]
    
    # Scatterplot with regression line
    sns.regplot(x=x, y=y, scatter_kws={'alpha': 0.7}, line_kws={'color': 'red'}, ax=ax)
    
    # Calculate Pearson correlation
    pearson_corr, p_value = pearsonr(x, y)
    print(column, pearson_corr, p_value)
    
    # Titles and labels
    ax.set_title(f"{title_names2[i]} ($r$ = {pearson_corr:.2f})")
    ax.set_xlabel("Average deconvolution score")
    ax.set_ylabel("Fraction rectangle")

# Show the plot
#plt.suptitle("Scatterplots with Regression Lines and Pearson Correlation", fontsize=16)
output_dir_plot = r"C:\Users\20182460\Desktop\Master_thesis\Figures report\Deconvolution"
plt.savefig(f"{output_dir_plot}/old_vs_rectangle.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()








