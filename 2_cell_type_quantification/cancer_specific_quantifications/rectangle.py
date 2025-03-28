# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 15:59:14 2024

@author: 20182460
"""

import pandas as pd
import anndata as ad
from anndata import AnnData
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

import rectanglepy as rectangle
from rectanglepy import ConsensusResult
from rectanglepy.pp import RectangleSignatureResult
from scipy.stats import zscore
from custom_annotation_functions import custom_anno1, custom_anno2, custom_anno3, custom_anno4

#Rectangle documentation: https://rectanglepy.readthedocs.io/generated/rectanglepy.rectangle.html

# Load all data 
print('Load single cell data')
sc_dir = "/home/evos/Data/CRC/AllCells_RAW.h5ad"
sc_data = ad.read_h5ad(sc_dir)
sc_expression_matrix = sc_data.X
cell_metadata = sc_data.obs
gene_metadata = sc_data.var

print('Add custom annotations')
#Add custom annotations
cell_metadata['custom_annotation1'] = cell_metadata['author_ClusterMidway']
cell_metadata['custom_annotation1'] = cell_metadata['custom_annotation1'].apply(custom_anno1)

cell_metadata['custom_annotation2'] = cell_metadata['author_ClusterMidway']
cell_metadata['custom_annotation2'] = cell_metadata['custom_annotation2'].apply(custom_anno2)

cell_metadata['custom_annotation3'] = cell_metadata['author_ClusterMidway']
cell_metadata['custom_annotation3'] = cell_metadata['custom_annotation3'].apply(custom_anno3)

cell_metadata['custom_annotation4'] = cell_metadata['author_ClusterMidway']
cell_metadata['custom_annotation4'] = cell_metadata['custom_annotation4'].apply(custom_anno4)

print(cell_metadata['custom_annotation1'].unique())
print(cell_metadata['custom_annotation2'].unique())
print(cell_metadata['custom_annotation3'].unique())
print(cell_metadata['custom_annotation4'].unique())

print('load bulk data')
bulk_dir = "/home/evos/Data/CRC/TCGA_tpm_CRC.txt"
bulk_data_ = pd.read_csv(bulk_dir, sep="\t", index_col=0)
bulk_data = bulk_data_.T

bulk_genes = bulk_data.columns
sc_genes = gene_metadata.index
common_genes = bulk_genes.intersection(sc_genes)
print('nr of common genes;', len(common_genes))

def rectangle_run(sc_expression_matrix, cell_metadata, bulk_data, annotation, excluded_cell_types, sampling=False):
    
    #filter sc expression matrix
    cells_to_keep = ~cell_metadata['author_ClusterMidway'].isin(excluded_cell_types)
    filtered_cell_metadata = cell_metadata[cells_to_keep]
    filtered_sc_expression_matrix = sc_expression_matrix[cells_to_keep, :]
    print(f"Filtered single-cell data: {filtered_sc_expression_matrix.shape[0]} cells remain")

    if sampling == True:
        print('sampling')
        sampled_indices = []
        # Loop over each unique cell type in annotation
        for cell_type in filtered_cell_metadata[annotation].unique():
            annotation_cells = filtered_cell_metadata[filtered_cell_metadata[annotation] == cell_type]
            annotation_indices = annotation_cells.index.tolist()
            numeric_indices = [filtered_cell_metadata.index.get_loc(cell) for cell in annotation_indices]
            num_cells_to_sample = min(len(numeric_indices), sample_size)
            random_indices = np.random.choice(numeric_indices, num_cells_to_sample, replace=False)
            sampled_indices.extend(random_indices.tolist())
        print('total nr cells each sample', len(sampled_indices))
        sc_matrix_subset = filtered_sc_expression_matrix[sampled_indices, :]
        dense_sc_matrix_subset = sc_matrix_subset.toarray()
        sc_matrix_df = pd.DataFrame(dense_sc_matrix_subset, index=filtered_cell_metadata.index[sampled_indices], columns=sc_data.var.index)
        annotations_CRC = filtered_cell_metadata[annotation]
        annotations_CRC = annotations_CRC[sampled_indices]
    else:
        print('check cell types:')
        for cell_type in filtered_cell_metadata[annotation].unique():
            print(cell_type)
        sc_matrix = filtered_sc_expression_matrix 
        dense_sc_matrix = sc_matrix.toarray()
        #sc_matrix_df = pd.DataFrame(dense_sc_matrix, index=sc_data.obs.index, columns=sc_data.var.index)
        sc_matrix_df = pd.DataFrame(dense_sc_matrix, index=filtered_cell_metadata.index, columns=sc_data.var.index)
        annotations_CRC = filtered_cell_metadata[annotation]

    sc_data_CRC = AnnData(sc_matrix_df, obs=annotations_CRC.to_frame(name="cell_type"))
    estimations, signature_result = rectangle.rectangle(sc_data_CRC, bulk_data, correct_mrna_bias=True)

    return estimations, signature_result 

#SET PARAMETERS
annotation = "author_ClusterMidway"
sample_size = 50
cell_frac_result_name = "cell_frac_ann4"
signature_result_name = 'signature_result_ann4'
signature_matrix_name = "signature_matrix_ann4"
#consensus_result_name = "consensus_result_ann4_1000.pkl"
#signature_matrix_name = "signature_matrix_annotation4"

if annotation == 'custom_annotation1':
    excluded_cell_types = ['Epi', 'NK', 'ILC', 'Mono', 'DC', 'Granulo']
elif annotation == 'custom_annotation2':
    excluded_cell_types = ['Epi', 'Tgd', 'TZBTB16', 'NK', 'ILC', 'Mono', 'DC', 'Granulo']
elif annotation == 'custom_annotation3':
    excluded_cell_types = ['Epi', 'NK', 'ILC']
elif annotation == 'custom_annotation4':
    excluded_cell_types = ['Epi', 'Tgd', 'TZBTB16', 'NK', 'ILC']

#sampling was not needed, so set to False
estimations, signature_result = rectangle_run(sc_expression_matrix, cell_metadata, bulk_data, annotation, excluded_cell_types, sampling=False)
signature_matrix = signature_result.get_signature_matrix(include_mrna_bias=True)

output_path = "/home/evos/Outputs/CRC/sc_deconvolution/ann4"
estimations.to_csv(f"{output_path}/{cell_frac_result_name}.csv") #save cell type fractions
with open(f"{output_path}/{signature_result_name}", 'wb') as file: #save signature result
    pickle.dump(signature_result, file)
signature_matrix.to_csv(f"{output_path}/{signature_matrix_name}.csv") #save signature matrix 

print('Finished saving results')