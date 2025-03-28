# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 11:34:47 2024

@author: 20182460
"""

import os
import sys
import pandas as pd
import dask.dataframe as dd
import argparse
import joblib
import scipy.stats as stats
import git

REPO_DIR= git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(f"{REPO_DIR}/libs")

from model.constants import DEFAULT_CELL_TYPES
from model.evaluate import compute_tile_predictions

def tile_level_quantification(models_dir, output_dir, histopatho_features_dir, MFP_dir, prediction_mode, n_outerfolds, slide_type="FF"):
    """
    Quantify the cell type abundances for the different tiles. Creates two files:
      (1) z-scores and
      (2) probability scores
    

    Parameters
    ----------
    models_dir : directory to folder where models for different cell types are saved 
    output_dir : output directory folder where files need to be saved 
    MFP_dir : directory to excel file where subtypes are stated for each of the files 
    histopatho_features_dir : directory to file or folder (dependent on FF or FFPE file) where the 
                              histopathological features are saved 
    prediction_mode : string, one of three options:
        1. prediction_mode=tcga_train_validation: predict for all tiles (validation+train) (used when FFPE slides are 
           run through FF model)
        2. prediction_mode=tcga_validation: predict only the tiles in the test sets
        3. prediction_mode=test: predict in external dataset
    n_outerfolds : int, number of outer folds 
    slide_type : string with slide type. Either "FF" or "FFPE". The default is "FF".

    Returns
    -------
    Saves 4 files in output directory. An scv and excel file with z-scores and an csv and excel file with 
    probability scores. 

    """
    # # Read data
    # if os.path.isfile(cell_types_path):
    #     cell_types = pd.read_csv(cell_types_path, header=None).to_numpy().flatten()
    # else:
    #     cell_types = DEFAULT_CELL_TYPES
    
    cell_types = DEFAULT_CELL_TYPES 

    full_output_dir = f"{output_dir}"
    if not os.path.isdir(full_output_dir):
        os.makedirs(full_output_dir)
 
    var_names_path = f"{models_dir}/task_selection_names.pkl"
    var_names = joblib.load(var_names_path)
    var_names['T_cells'] = ['Cytotoxic cells', 'Effector cells', 'CD8 T cells (quanTIseq)', 'Immune score']
    var_names['tumor_purity'] = ['tumor purity (ABSOLUTE)', 'tumor purity (EPIC)']
    var_names['CAFs'] = ['CAFs (MCP counter)', 'CAFs (EPIC)','CAFs (Bagaev)']
    MFP = pd.read_excel(MFP_dir, header=1, sheet_name="Pan_TCGA")
    MFP = MFP.rename(columns={"Unnamed: 0": "TCGA_patient_ID"})
    MFP = MFP[["TCGA_patient_ID", "MFP"]]

    if slide_type == "FF":
        histopatho_features = pd.read_csv(histopatho_features_dir, sep="\t", index_col=0)
    elif slide_type == "FFPE":
        histopatho_features = dd.read_parquet(f"{histopatho_features_dir}/features-*.parquet")

    #set correct slide id
    histopatho_features['slide_submitter_id'] = histopatho_features['tile_ID'].str.rsplit('-', n=1).str[0]
    
    # Compute predictions based on bottleneck features
    tile_predictions = pd.DataFrame()
    bottleneck_features = histopatho_features.loc[:,  [str(i) for i in range(1536)]]
    bottleneck_features.index = histopatho_features.tile_ID
    metadata = histopatho_features.loc[:, var_names["IDs"] + var_names["tile_IDs"]]
    if slide_type == "FFPE":
        metadata = metadata.compute()

    print("Computing tile predictions for each cell type...")

    #If predicting on all FFPE slides, we do this by chunks:
    if any([prediction_mode == item for item in ['tcga_train_validation', 'test']]):
        if slide_type == "FFPE":
            tmp=metadata.slide_submitter_id.unique().tolist()
            n=30
            chunks = [tmp[i:i + n] for i in range(0, len(tmp), n)]
            print(len(chunks))
            ffpe_slides = dict.fromkeys(range(len(chunks)))
            for i in range(len(chunks)):
                ffpe_slides[i] = chunks[i]
        
            for key, chunk in ffpe_slides.items():
                print('Chunk:', key)
                tiles_subset = metadata[metadata["slide_submitter_id"].isin(chunk)]["tile_ID"]
                X = bottleneck_features.map_partitions(lambda x: x[x.index.isin(tiles_subset)])
                X = X.compute()
                predictions = pd.DataFrame()
                for cell_type in cell_types:
                    cell_type_tile_predictions = compute_tile_predictions(
                        cell_type=cell_type, models_dir=models_dir, n_outerfolds=n_outerfolds,
                        prediction_mode=prediction_mode, X=X, metadata=metadata, slide_type=slide_type
                    )
                    predictions = pd.concat([predictions, cell_type_tile_predictions], axis=1)
            
                print(predictions.shape)
                tile_predictions = pd.concat([tile_predictions, predictions], axis=0)
        
        elif slide_type == "FF": #if FF slides run through FFPE trained model (this does not happen) or for test
            for cell_type in cell_types:
                    cell_type_tile_predictions = compute_tile_predictions(
                        cell_type=cell_type, models_dir=models_dir, n_outerfolds=n_outerfolds,
                        prediction_mode=prediction_mode, X=bottleneck_features, metadata=metadata,
                        slide_type=slide_type
                    )
                    tile_predictions = pd.concat([tile_predictions, cell_type_tile_predictions], axis=1)
        
    ##############################################################################

    elif any([prediction_mode == item for item in ['tcga_validation']]):
        for cell_type in cell_types:
                cell_type_tile_predictions = compute_tile_predictions(
                    cell_type=cell_type, models_dir=models_dir, n_outerfolds=n_outerfolds,
                    prediction_mode=prediction_mode, X=bottleneck_features, metadata=metadata,
                    slide_type=slide_type
                )
                tile_predictions = pd.concat([tile_predictions, cell_type_tile_predictions], axis=1)

    # Remove slides with nan values
    tile_predictions = tile_predictions.dropna()

    # Order tile_predictions according to metadata
    metadata = metadata[metadata.tile_ID.isin(tile_predictions.index)]
    metadata.index = metadata.tile_ID
    tile_predictions = tile_predictions.loc[metadata.index,:]

    # Convert predictions to probabilities using cdf.
    feature_names = tile_predictions.columns
    pred_proba = pd.DataFrame(data=stats.norm.cdf(tile_predictions), columns=feature_names, index=tile_predictions.index)
    pred_proba = pd.concat([pred_proba, metadata], axis=1)

    # Add subtype from MFP
    if any([prediction_mode == item for item in ['tcga_train_validation', 'tcga_validation']]):
       # Add metadata to tile_predictions (rows are in same order)
       pred_proba["TCGA_patient_ID"] = pred_proba.tile_ID.str[0:12]
       pred_proba = pd.merge(pred_proba, MFP, on="TCGA_patient_ID", how="left")
       tile_predictions = pd.concat([tile_predictions, metadata], axis=1)
       tile_predictions["TCGA_patient_ID"] = tile_predictions.tile_ID.str[0:12]
       tile_predictions = pd.merge(tile_predictions, MFP, on="TCGA_patient_ID", how="left")

    elif prediction_mode == 'test':
       pred_proba['slide_id'] = pred_proba['tile_ID'].str.extract(r'(^.*-registered)')
       pred_proba.drop(columns=['slide_submitter_id', "sample_submitter_id", "Section"], inplace=True)
       pred_proba = pred_proba.reset_index(drop=True)

    # Remove suffix '(combi)'
    pred_proba.columns = [col.replace(" (combi)", "") for col in pred_proba.columns]
    tile_predictions.columns = [col.replace(" (combi)", "") for col in tile_predictions.columns]
    
    if slide_type == "FFPE":
        tile_predictions.to_csv(f"{full_output_dir}/{prediction_mode}_tile_predictions_zscores.csv", sep="\t", index=False)
        pred_proba.to_csv(f"{full_output_dir}/{prediction_mode}_tile_predictions_proba.csv", sep="\t", index=False)
    else:   
        tile_predictions.to_csv(f"{full_output_dir}/{prediction_mode}_tile_predictions_zscores.csv", sep="\t", index=False)
        tile_predictions.to_excel(f"{full_output_dir}/{prediction_mode}_tile_predictions_zscores.xlsx")
        pred_proba.to_csv(f"{full_output_dir}/{prediction_mode}_tile_predictions_proba.csv", sep="\t", index=False)
        pred_proba.to_excel(f"{full_output_dir}/{prediction_mode}_tile_predictions_proba.xlsx")
        
        

# models_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\SKCM\FF_subset\2_train_multitask_models"
# output_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\SKCM\FFPE_subset"
# MFP_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Data\SKCM\MFP_data.xlsx"
# histopatho_features_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\SKCM\FFPE_subset\1_extract_histopathological_features\features_format_parquet"

# tile_level_quantification(models_dir=models_dir, output_dir=output_dir, MFP_dir=MFP_dir, 
#                           histopatho_features_dir=histopatho_features_dir,
#                           prediction_mode="tcga_train_validation", n_outerfolds=2, slide_type="FFPE")

if __name__ ==  "__main__":

    parser = argparse.ArgumentParser(description="Predict cell type abundances for the tiles")
    parser.add_argument("--models_dir", type=str, help="Path to models directory", required=True)
    parser.add_argument("--output_dir", type=str, help="Path to output directory", required=True)
    parser.add_argument("--MFP_dir", type=str, help="Path to MFP (subtypes) file", required=True)
    parser.add_argument("--histopatho_features_dir", type=str, help="Path to histopathological features file", required=True)

    parser.add_argument("--prediction_mode",type=str,  help="Choose prediction mode", default="all", required=False)
    parser.add_argument("--n_outerfolds", type=int, default=5, help="Number of outer folds (default=5)", required=False)
    parser.add_argument("--slide_type", help="Type of tissue slide (FF or FFPE)", type=str, required=True)
    args=parser.parse_args()

    tile_level_quantification(
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        MFP_dir=args.MFP_dir,
        histopatho_features_dir=args.histopatho_features_dir,
        prediction_mode=args.prediction_mode,
        n_outerfolds=args.n_outerfolds,
        slide_type=args.slide_type)
