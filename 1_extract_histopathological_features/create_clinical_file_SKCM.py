import argparse
import os
import os.path
import git
import numpy as np
import pandas as pd
import sys

def create_TCGA_clinical_file(
    class_names,
    data_dir,
    output_dir,
    code_dir,
    tumor_purity_threshold=80):
    """
    Create a clinical file based on the slide metadata downloaded from the GDC data portal
    1. Read the files and add classname and id based on CODEBOOK.txt
    2. Filter tumor purity
    3. Save file

    Args:
        class_names (str): single class name e.g. LUAD_T or path to file with class names
        clinical_files_dir (str): String with path to folder with subfolders pointing to the raw clinical files (slide.tsv)
        output_dir (str): Path to folder where the clinical file should be stored
        tumor_purity_threshold (int): default=80
        multi_class_path (str): path to file with class names to be merged into one clinical file

    Returns:
        {output_dir}/generated_clinical_file.txt" containing the slide_submitter_id, sample_submitter_id, image_file_name, percent_tumor_cells, class_name, class_id in columns and records (slides) in rows.

    """
    # ---- Setup parameters ---- #
    full_output_dir = f"{output_dir}/1_extract_histopathological_features"
    if not os.path.isdir(full_output_dir):
        os.mkdir(full_output_dir)

    if (os.path.isfile(class_names)): # multi class names
        class_names = pd.read_csv(class_names, header=None).to_numpy().flatten()
    else: # single class names
        class_name=class_names

    clinical_files_dir = f"{data_dir}/clinical_file_TCGA_SKCM.txt"
    
    CODEBOOK = pd.read_csv(
        f"{code_dir}/1_extract_histopathological_features/codebook.txt",
        delim_whitespace=True,
        header=None, names=["class_name", "value"]
    )

    # ---- 1. Constructing a merged clinical file ---- #
    # Read clinical files
    # a) Single class
    if os.path.isfile(clinical_files_dir):
        clinical_file = pd.read_csv(clinical_files_dir, sep="\t")
        clinical_file["class_name"] = class_name
        clinical_file["class_id"] = int(
            CODEBOOK.loc[CODEBOOK["class_name"] == class_name].values[0][1]
        )
    
    # b) Multiple classes
    elif os.path.isdir(clinical_files_dir) & (len(class_names) > 1):
        clinical_file_list = []
        # Combine all clinical raw files based on input
        for class_name in class_names:
            clinical_file_temp = pd.read_csv(
                f"{clinical_files_dir}/clinical_file_TCGA_{class_name[:-2]}.tsv",
                sep="\t",
            )
            # only keep tissue (remove _T or _N) to check in filename
            clinical_file_temp["class_name"] = class_name
            clinical_file_temp["class_id"] = int(
                CODEBOOK.loc[CODEBOOK["class_name"] == class_name].values[0][1]
            )
            clinical_file_list.append(clinical_file_temp)
        clinical_file = pd.concat(clinical_file_list, axis=0).reset_index(drop=True)

    # ---- 2) Filter: Availability of tumor purity (percent_tumor_cells) ---- #
    # Remove rows with missing tumor purity
    clinical_file["percent_tumor_cells"] = (
        clinical_file["percent_tumor_cells"]
        .replace("'--", np.nan, regex=True)
        .astype(float)
    )

    # Convert strings to numeric type
    clinical_file["percent_tumor_cells"] = pd.to_numeric(
        clinical_file["percent_tumor_cells"]
    )
    clinical_file = clinical_file.dropna(subset=["percent_tumor_cells"])
    clinical_file = clinical_file.where(
        clinical_file["percent_tumor_cells"] >= int(tumor_purity_threshold)
    )
    # # ---- 3) Formatting and saving ---- #

    clinical_file = clinical_file.dropna(how="all")
    clinical_file = clinical_file.drop_duplicates()
    clinical_file = clinical_file.drop_duplicates(subset="slide_submitter_id")
    clinical_file = clinical_file[
        [
            "slide_submitter_id",
            "sample_submitter_id",
            "image_file_name",
            "percent_tumor_cells",
            "class_name",
            "class_id",
        ]
    ]
    clinical_file = clinical_file.dropna(how="any", axis=0)
    clinical_file.to_csv(
            f"{full_output_dir}/generated_clinical_file_FF.txt",
            index=False,
            sep="\t",
        )
    print("\nFinished creating a new clinical file")
    
# class_names = "SKCM_T"
# data_dir = r"C:\Users\20182460\Documents\GitHub\THESIS\Data\SKCM"
# output_dir = r"C:\Users\20182460\Documents\GitHub\THESIS\Outputs\SKCM\FF"
# code_dir = r"C:\Users\20182460\Documents\GitHub\THESIS"
# tumor_purity_threshold = 80

# create_TCGA_clinical_file(class_names = class_names,
#                           data_dir = data_dir,
#                           output_dir = output_dir,
#                           code_dir = code_dir,
#                           tumor_purity_threshold = tumor_purity_threshold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--class_names",
        help="Either (a) single classname or (b) Path to file with classnames according to codebook.txt (e.g. LUAD_T)", required=True
    )
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
    args = parser.parse_args()

    create_TCGA_clinical_file(
        class_names=args.class_names,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        code_dir=args.code_dir,
        tumor_purity_threshold=args.tumor_purity_threshold
    )