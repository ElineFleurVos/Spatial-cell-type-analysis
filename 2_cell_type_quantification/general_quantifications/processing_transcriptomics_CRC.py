# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 18:13:58 2024

@author: 20182460
"""

# Module imports
import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
import git

REPO_DIR = git.Repo(os.getcwd(), search_parent_directories=True).working_tree_dir
sys.path.append(f"{REPO_DIR}/libs")

import model.preprocessing as preprocessing
from model.constants import TUMOR_PURITY, T_CELLS, ENDOTHELIAL_CELLS, CAFS, IDS, TILE_VARS

def processing_transcriptomics(published_RNA_data_dir, gmt_signatures_dir, clinical_file_dir, tpm_dir, output_dir, slide_type):
    """ Compute and combine cell type abundances from different quantification methods necessary for TF learning
    Args:
        clinical_file_dir (str): clinical_file_path: path to clinical_file
        tpm_path (str): path pointing to the tpm file
        output_dir (str): path pointing to a folder where the dataframe containing all features should be stored, stored as .txt file

    Returns:
        ./task_selection_names.pkl: pickle file containing variable names.
        {output_dir}/TCGA_{cancer_type}_ensembled_selected_tasks.csv" containing the following cell type quantification methods:
            tumor_purity = [
                'tumor purity (ABSOLUTE)',
                'tumor purity (estimate)',
                'tumor purity (EPIC)'
            ]
            T_cells = [
                'CD8 T cells (Thorsson)',
                'Cytotoxic cells',
                'Effector cells',
                'CD8 T cells (quanTIseq)',
                'TIL score',
                'Immune score',
            ]
            endothelial_cells = [
                'Endothelial cells (xCell)',
                'Endothelial cells (EPIC)',
                'Endothelium', ]
            CAFs = [
                'Stromal score',
                'CAFs (MCP counter)',
                'CAFs (EPIC)',
                'CAFs (Bagaev)',
            ]
    """
    full_output_dir = output_dir
    if not os.path.exists(full_output_dir):
        os.makedirs(full_output_dir)
    
    var_dict = {
        "CAFs": CAFS,
        "T_cells": T_CELLS,
        "tumor_purity": TUMOR_PURITY,
        "endothelial_cells": ENDOTHELIAL_CELLS,
        "IDs":IDS,
        "tile_IDs": TILE_VARS
    }
    joblib.dump(var_dict, f"{output_dir}/task_selection_names.pkl")
    if slide_type=='FF':
        clinical_file = pd.read_csv(f"{clinical_file_dir}/generated_clinical_file_FF.txt", sep="\t")
    else:
        clinical_file = pd.read_csv(f"{clinical_file_dir}/generated_clinical_file_FFPE.txt", sep="\t")
    
    #%%
    var_IDs = ['sample_submitter_id','slide_submitter_id']
    all_slide_features = clinical_file.loc[:,var_IDs]
    all_slide_features["TCGA_sample"] = clinical_file["slide_submitter_id"].str[0:15]
    print(all_slide_features['sample_submitter_id'].nunique())
    print(all_slide_features['TCGA_sample'].nunique())
    
    #%%
    # Published Data
    Thorsson =  pd.read_csv(f"{published_RNA_data_dir}/Thorsson_Scores_160_Signatures.tsv.gz",  sep="\t")
    estimate = pd.read_csv(f"{published_RNA_data_dir}/Yoshihara_ESTIMATE_CRC_RNAseqV2.txt", sep='\t') 
    tcga_absolute = pd.read_csv(f"{published_RNA_data_dir}/TCGA_ABSOLUTE_tumor_purity.txt", sep="\t")
    gibbons = pd.read_excel(f"{published_RNA_data_dir}/Gibbons_supp1.xlsx", skiprows=2, sheet_name="DataFileS1 - immune features")
    
    # Computed Data: Immunedeconv
    mcp_counter = pd.read_csv(f"{output_dir}/immunedeconv/mcp_counter.csv", index_col=0, sep=",")
    quantiseq = pd.read_csv(f"{output_dir}/immunedeconv/quantiseq.csv", index_col=0, sep=",")
    xCell = pd.read_csv(f"{output_dir}/immunedeconv/xcell.csv", index_col=0, sep=",", header=[0])
    EPIC = pd.read_csv(f"{output_dir}/immunedeconv/epic.csv", index_col=0, sep=",")
    
    Fges_computed = preprocessing.compute_gene_signature_scores(tpm_dir, gmt_signatures_dir)
    Fges_computed = Fges_computed.loc[:, ["Effector_cells", "Endothelium", "CAF"]]
    Fges_computed.columns = ["Effector cells", "Endothelium", "CAFs (Bagaev)"]
    
    Fges_computed = Fges_computed.reset_index()
    Fges_computed = Fges_computed.rename(columns={"index": "TCGA_sample"})
    
    # From immunedeconv
    quantiseq = preprocessing.process_immunedeconv(quantiseq, "quanTIseq")
    EPIC = preprocessing.process_immunedeconv(EPIC, "EPIC")
    mcp_counter = preprocessing.process_immunedeconv(mcp_counter, "MCP")
    xCell = preprocessing.process_immunedeconv(xCell, "xCell")

    # Merge cell fractions
    cellfrac = pd.merge(xCell, quantiseq, on=["TCGA_sample"])
    cellfrac = pd.merge(cellfrac, mcp_counter, on=["TCGA_sample"])
    cellfrac = pd.merge(cellfrac, EPIC, on=["TCGA_sample"])
    
    # estimate data
    estimate = estimate.rename(columns={"ID": "TCGA_sample"})
    estimate = estimate.set_index("TCGA_sample")
    estimate.columns = ["Stromal score", "Immune score", "ESTIMATE score"]
    
    # According the tumor purity formula provided in the paper
    estimate["tumor purity (ESTIMATE)"] = np.cos(
        0.6049872018 + .0001467884 * estimate["ESTIMATE score"])
    estimate = estimate.drop(columns=["ESTIMATE score"])
    
    # Thorsson data
    Thorsson = Thorsson.drop(columns="Source")
    Thorsson = Thorsson.set_index("SetName").T
    Thorsson = Thorsson.rename_axis(None, axis=1)
    Thorsson.index.name="TCGA_aliquot"
    Thorsson = Thorsson.loc[:, ["LIexpression_score", "CD8_PCA_16704732"]]
    Thorsson.columns = ["TIL score", "CD8 T cells (Thorsson)"]
    
    # TCGA PanCanAtlas
    tcga_absolute = tcga_absolute.rename(columns = {"purity": "tumor purity (ABSOLUTE)", "sample": "TCGA_aliquot"})
    tcga_absolute = tcga_absolute.set_index("TCGA_aliquot")
    tcga_absolute = pd.DataFrame(tcga_absolute.loc[:, "tumor purity (ABSOLUTE)"])
    
    gibbons = gibbons.rename(columns={'Unnamed: 1': "id"})
    gibbons["slide_submitter_id"] =  gibbons["id"].str[0:23]
    gibbons["Cytotoxic cells"] = gibbons["Cytotoxic cells"].astype(float)
    gibbons = gibbons.set_index("slide_submitter_id")
    gibbons = gibbons.drop_duplicates(keep='first')
    gibbons = pd.DataFrame(gibbons.loc[:, "Cytotoxic cells"])
    
    # add IDs
    gibbons["TCGA_sample"] = gibbons.index.str[0:15]
    Thorsson["TCGA_sample"] = Thorsson.index.str[0:15]
    tcga_absolute["TCGA_sample"] = tcga_absolute.index.str[0:15]
    
    tcga_absolute_merged = pd.merge(all_slide_features, tcga_absolute, on=["TCGA_sample", ], how="left")
    print(tcga_absolute_merged['tumor purity (ABSOLUTE)'].notna().sum()) #1 deconvolution
    Thorsson_merged = pd.merge(all_slide_features, Thorsson, on=["TCGA_sample",], how="left")
    print(Thorsson_merged['CD8 T cells (Thorsson)'].notna().sum()) #(2) deconvolution
    gibbons_merged = pd.merge(all_slide_features, gibbons, on=["TCGA_sample"], how="left")
    print(gibbons_merged['Cytotoxic cells'].notna().sum()) #1 deconvolution
    
    
    cellfrac_merged = pd.merge(all_slide_features, cellfrac, on=["TCGA_sample"], how="left")
    print(cellfrac_merged['CAFs (EPIC)'].notna().sum()) #6 deconvolutions, 277 not nans is correct 
    estimate_merged = pd.merge(all_slide_features, estimate, on=["TCGA_sample" ], how="left")
    print(estimate_merged['Stromal score'].notna().sum()) #3 deconvolutions 
    Fges_computed_merged = pd.merge(all_slide_features, Fges_computed, on=["TCGA_sample"], how="left")
    print(Fges_computed_merged['Effector cells'].notna().sum()) #3 deconvolutions 
    
    # Combine in one dataframe
    all_merged = pd.merge(all_slide_features, tcga_absolute_merged, how="left")
    all_merged = pd.merge(all_merged, Thorsson_merged ,how="left")
    all_merged = pd.merge(all_merged, gibbons_merged,how="left")
    all_merged = pd.merge(all_merged, estimate_merged,  how="left")
    all_merged = pd.merge(all_merged, cellfrac_merged,  how="left")
    all_merged = pd.merge(all_merged, Fges_computed_merged,  how="left")
    
    # ---- Transform features to get a normal distribution (immunedeconv) ---- #
    featuresnames_transform = ["CAFs (MCP counter)",
        'CAFs (EPIC)',]
    feature_data = all_merged.loc[:, CAFS].astype(float)
    data_log2_transformed = feature_data.copy()
    data_log2_transformed[featuresnames_transform] = np.log2(feature_data[featuresnames_transform] * 100 + 0.001)
    CAFs_transformed = data_log2_transformed
    
    featuresnames_transform = ["Endothelial cells (xCell)",
        "Endothelial cells (EPIC)",]
    feature_data = all_merged.loc[:, ENDOTHELIAL_CELLS].astype(float)
    data_log2_transformed = feature_data.copy()
    data_log2_transformed[featuresnames_transform] = np.log2(feature_data[featuresnames_transform] * 100 + 0.001)
    endothelial_cells_transformed = data_log2_transformed
    
    feature_data = all_merged.loc[:, T_CELLS].astype(float)
    featuresnames_transform = ['CD8 T cells (quanTIseq)']
    data_log2_transformed =  feature_data.copy()
    data_log2_transformed[featuresnames_transform] = np.log2(feature_data[featuresnames_transform] * 100 + 0.001)
    T_cells_transformed = data_log2_transformed
    
    feature_data = all_merged.loc[:, TUMOR_PURITY].astype(float)
    featuresnames_transform = ["tumor purity (EPIC)"]
    data_log2_transformed =  feature_data.copy()
    data_log2_transformed[featuresnames_transform] = np.log2(feature_data[featuresnames_transform] * 100 + 0.001)
    tumor_cells_transformed = data_log2_transformed
    
    # Store processed data
    IDs = ['slide_submitter_id', 'sample_submitter_id', "TCGA_sample"]
    metadata = all_merged[IDs]
    merged = pd.concat([
        metadata,
    CAFs_transformed, endothelial_cells_transformed, T_cells_transformed, tumor_cells_transformed], axis=1)
    merged = merged.fillna(np.nan)
    
    # Remove slides if there are no values at all
    merged = merged.dropna(axis=0, subset=T_CELLS + CAFS + ENDOTHELIAL_CELLS + TUMOR_PURITY, how="all")
    merged.to_csv(f'{full_output_dir}/ensembled_selected_tasks.csv', sep = "\t")
    merged.to_excel(f"{full_output_dir}/ensembled_selected_tasks.xlsx")#, sep="\t")
    
    print('Processing transcriptomics completed')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process transcriptomics data for use in TF learning')
    parser.add_argument(
        "--published_RNA_data_dir",
        help="directory to published RNA data files",
    )
    parser.add_argument(
        "--gmt_signatures_dir",
        help="directory to gmt signature file",
    )
    parser.add_argument(
        "--clinical_file_dir",
        help="Full path to clinical file", default=None
    )
    parser.add_argument(
        "--tpm_dir", help="Path to tpm file", type=str, required=True
    )
    parser.add_argument(
        "--output_dir", help="Path to folder for generated file")
    parser.add_argument(
        "--slide_type", help="Type of slide, either 'FF' or 'FFPE'")
    args = parser.parse_args()

    processing_transcriptomics(
        published_RNA_data_dir=args.published_RNA_data_dir,
        gmt_signatures_dir=args.gmt_signatures_dir,
        tpm_dir=args.tpm_dir,
        clinical_file_dir=args.clinical_file_dir,
        output_dir=args.output_dir,
        slide_type=args.slide_type
    )
