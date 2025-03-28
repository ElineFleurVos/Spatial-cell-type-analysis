# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 09:57:40 2024

@author: 20182460
"""

import pandas as pd
import numpy as np
import os 
import argparse

#three files needed. Clinical file and two text files with slide names for FF and FFPE slides 
#respectively that can be found in the data/TCGA_CRC directory

def create_TCGA_clinical_file_COLO(data_dir, output_dir, code_dir, slide_type, tumor_purity_threshold=80):

    clinical_file_dir = f"{data_dir}/clinical_file_TCGA_CRC.tsv"
    names_FF_dir = f"{data_dir}/slide_filenames_CRC_FF.txt"
    names_FFPE_dir = f"{data_dir}/slide_filenames_CRC_FFPE.txt"
    
    clinical_file_all = pd.read_csv(clinical_file_dir, sep="\t")
    file_names_FF = pd.read_csv(names_FF_dir, encoding='utf-16', header=None, names=['slide_file_name'])
    file_names_FFPE = pd.read_csv(names_FFPE_dir, encoding='utf-16', header=None, names=['slide_file_name'])
    
    #set up paths 
    full_output_dir = f"{output_dir}/1_extract_histopathological_features"
    if not os.path.isdir(full_output_dir):
        os.mkdir(full_output_dir)
    
    #Load codebook
    CODEBOOK = pd.read_csv(
        f"{code_dir}/1_extract_histopathological_features/codebook.txt",
        delim_whitespace=True,
        header=None, names=["class_name", "value"]
    )
    
    #Add class name and class id
    class_names = ['COAD_T', 'READ_T']
    clinical_file_all["class_name"] = np.where(clinical_file_all["project_id"].str[-4:] == "COAD", class_names[0],
                                       np.where(clinical_file_all["project_id"].str[-4:] == "READ", class_names[1], None))
    clinical_file_all["class_id"] = np.where(clinical_file_all["class_name"] == class_names[0], int(CODEBOOK.loc[CODEBOOK["class_name"] == class_names[0]].values[0][1]),
                                    np.where(clinical_file_all["class_name"] == class_names[1], int(CODEBOOK.loc[CODEBOOK["class_name"] == class_names[1]].values[0][1]), None))
    
    #split on FF and FFPE slides 
    slide_type_condition = clinical_file_all['slide_submitter_id'].str[-3:-1] == 'DX'
    clinical_file_FFPE = clinical_file_all[slide_type_condition]
    clinical_file_FF = clinical_file_all[~slide_type_condition]
    
    #add image file name FF
    matches_FF = []
    for string in clinical_file_FF['slide_submitter_id']:
        matching_filenames = file_names_FF[file_names_FF['slide_file_name'].str.contains(string, case=False, regex=False)]['slide_file_name'].tolist()
        # If there are matching filenames, append the first one; otherwise, append None
        if matching_filenames:
            matches_FF.append(matching_filenames[0])
        else:
            matches_FF.append(None)
    
    clinical_file_FF.loc[:, 'image_file_name'] = matches_FF
    clinical_file_FF = clinical_file_FF.dropna(subset=['image_file_name'])
    
    #add image file name FFPE
    matches_FFPE = []
    for string in clinical_file_FFPE['slide_submitter_id']:
        matching_filenames = file_names_FFPE[file_names_FFPE['slide_file_name'].str.contains(string, case=False, regex=False)]['slide_file_name'].tolist()
        # If there are matching filenames, append the first one; otherwise, append None
        if matching_filenames:
            matches_FFPE.append(matching_filenames[0])
        else:
            matches_FFPE.append(None)
            
    clinical_file_FFPE.loc[:,'image_file_name'] = matches_FFPE
    clinical_file_FFPE = clinical_file_FFPE.dropna(subset=['image_file_name'])
    
    #Filter on availability of tumor purity (percent_tumor_cells) on FF slides 
    #remove rows with missing tumor purity
    clinical_file_FF["percent_tumor_cells"] = (
        clinical_file_FF["percent_tumor_cells"]
        .replace("'--", np.nan, regex=True)
        .astype(float)
    )
    clinical_file_FF.loc[:, "percent_tumor_cells"] = pd.to_numeric(clinical_file_FF["percent_tumor_cells"])
    clinical_file_FF = clinical_file_FF.dropna(subset=["percent_tumor_cells"])
    clinical_file_FF = clinical_file_FF.where(clinical_file_FF["percent_tumor_cells"] >= int(tumor_purity_threshold))
    clinical_file_FF_filtered = clinical_file_FF.dropna(how="all")
    
    #Filter FFPE slides based on FF selection and put one of the tumor purity values from the FF dataframe in
    #spefically the lowest tumor purity value
    clinical_file_FFPE_filtered = clinical_file_FFPE[clinical_file_FFPE['case_id'].isin(clinical_file_FF_filtered['case_id'])]
    for index, row in clinical_file_FFPE_filtered.iterrows():
        value_to_match = row['case_id']
        filtered_rows = clinical_file_FF_filtered[clinical_file_FF_filtered['case_id'] == value_to_match]
        if not filtered_rows.empty:
            # Find the lowest value in the 'value' column of the filtered rows
            min_value = filtered_rows['percent_tumor_cells'].min()
            clinical_file_FFPE_filtered.at[index, 'percent_tumor_cells'] = int(min_value)
    clinical_file_FFPE_filtered["percent_tumor_cells"] = pd.to_numeric(clinical_file_FFPE_filtered["percent_tumor_cells"])
    
    #formatting and saving 
    columns_keep = ["slide_submitter_id", "sample_submitter_id","image_file_name","percent_tumor_cells","class_name","class_id"]
    
    clinical_file_FF_filtered = clinical_file_FF_filtered.drop_duplicates()
    clinical_file_FF_filtered = clinical_file_FF_filtered.drop_duplicates(subset="slide_submitter_id")
    clinical_file_FF_filtered = clinical_file_FF_filtered[columns_keep]
    clinical_file_FF_filtered = clinical_file_FF_filtered.dropna(how="any", axis=0)
    
    str_threshold = str(tumor_purity_threshold)
    if slide_type == "FF":
        clinical_file_FF_filtered.to_csv(
                f"{full_output_dir}/generated_clinical_file_FF_{str_threshold}.txt",
                index=False,
                sep="\t")
    
    clinical_file_FFPE_filtered = clinical_file_FFPE_filtered.drop_duplicates()
    clinical_file_FFPE_filtered = clinical_file_FFPE_filtered.drop_duplicates(subset="slide_submitter_id")
    clinical_file_FFPE_filtered = clinical_file_FFPE_filtered[columns_keep]
    clinical_file_FFPE_filtered = clinical_file_FFPE_filtered.dropna(how="any", axis=0)
    
    if slide_type == "FFPE":
        clinical_file_FFPE_filtered.to_csv(
                f"{full_output_dir}/generated_clinical_file_FFPE_{str_threshold}.txt",
                index=False,
                sep="\t")
                    
    print("\nFinished creating a new clinical file")

# code_dir = r"C:\Users\20182460\Documents\GitHub\THESIS"
# data_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Data\CRC"
# output_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\CRC"
# tumor_purity_threshold = 80
#create_TCGA_clinical_file_COLO(code_dir, data_dir, output_dir, tumor_purity_threshold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        help="Path to folder contain all initial data.", required=True
    )
    parser.add_argument(
        "--output_dir",
        help="Path to folder with all outputs", required=True
    )
    parser.add_argument(
        "--code_dir",
        help="Path to folder with all code", required=True
    )
    parser.add_argument(
        "--tumor_purity_threshold",
        help="Integer for filtering tumor purity assessed by pathologists",
        default=80, required=False
    )
    parser.add_argument(
        "--slide_type",
        help="Slide type, either 'FF' or 'FFPE' ", required=True
    )
    args = parser.parse_args()

    create_TCGA_clinical_file_COLO(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        code_dir=args.code_dir,
        tumor_purity_threshold=args.tumor_purity_threshold,
        slide_type=args.slide_type
    )