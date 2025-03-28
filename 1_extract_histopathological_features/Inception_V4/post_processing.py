import argparse
import os
import git
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'myslim')))

from myslim.post_process_features import post_process_features
from myslim.post_process_predictions import post_process_predictions

REPO_DIR = git.Repo(os.getcwd(), search_parent_directories=True).working_tree_dir
sys.path.append(f"{REPO_DIR}/Python/libs")

def execute_postprocessing(codebook_dir, tissue_classes_dir, output_dir, slide_type):
    """
    1. Format extracted histopathological features
    2. Format predictions of the 42 classes

    Args:
        output_dir (str): path pointing to folder for storing all created files by script
        slide_type: string either "FF" or "FFPE"

    Returns:
        {output_dir}/features.txt
        {output_dir}/predictions.txt
    """
    post_process_features(output_dir=output_dir, slide_type=slide_type)
    post_process_predictions(codebook_dir=codebook_dir, tissue_classes_dir=tissue_classes_dir,
                             output_dir=output_dir, slide_type=slide_type)


# codebook_dir = r"C:\Users\20182460\Documents\GitHub\THESIS\1_extract_histopathological_features\codebook.txt"
# tissue_classes_dir = r"C:\Users\20182460\Documents\GitHub\THESIS\1_extract_histopathological_features\tissue_classes.csv"
# output_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\SKCM\FFPE_subset\1_extract_histopathological_features"
# slide_type = "FFPE"
# execute_postprocessing(codebook_dir=codebook_dir, tissue_classes_dir=tissue_classes_dir,
#                         output_dir=output_dir, slide_type=slide_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", help="Set output folder", type=str)
    parser.add_argument("--slide_type", help="Type of tissue slide (FF or FFPE)", required=True, type=str)
    parser.add_argument("--codebook_dir", help="direction to codebook")
    parser.add_argument("--tissue_classes_dir", help="direction to tissue classes ")
    args = parser.parse_args()

    execute_postprocessing(
        output_dir=args.output_dir,
        tissue_classes_dir=args.tissue_classes_dir,
        codebook_dir=args.codebook_dir,
 	    slide_type=args.slide_type
    )
