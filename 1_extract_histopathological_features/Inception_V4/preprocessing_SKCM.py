import argparse
import os
import git
import pandas as pd
import sys
import shutil

REPO_DIR = git.Repo(os.getcwd(), search_parent_directories=True).working_tree_dir
sys.path.append(f"{REPO_DIR}/Python/libs")

sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'myslim')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from create_file_info_train_CRC import format_tile_data_structure
from myslim.datasets.convert import _convert_dataset
from create_TCGA_CRC_tiles import create_tiles_from_slides

def execute_preprocessing(slides_dir, output_dir, clinical_file_dir, N_shards=320, subset="False", N_slides=10):
    """
    Execute several pre-processing steps necessary for extracting the histopathological features
    1. Create tiles from slides
    2. Construct file necessary for the deep learning architecture
    3. Convert images of tiles to TF records

    Args:
        slides_dir (str): path pointing to folder with all whole slide images (.svs files)
        output_dir (str): path pointing to folder for storing all created files by script
        clinical_file_path (str): path pointing to formatted clinical file (either generated or manually formatted)
        N_shards (int): default: 320
        checkpoint_path (str): path pointing to checkpoint to be used

    Returns:
        {output_dir}/tiles/{tile files}
        {output_dir}/file_info_train.txt file specifying data structure of the tiles required for inception architecture (to read the TF records)
        {output_dir}/process_train/{TFrecord file} files that store the data as a series of binary sequencies

    """
    full_output_path=f"{output_dir}/1_extract_histopathological_features"
    if not os.path.exists(full_output_path):
        os.makedirs(full_output_path)
    # Create an empty folder for TF records if folder doesn't exist
    process_train_dir = f"{full_output_path}/process_train"
    if not os.path.exists(process_train_dir):
        os.makedirs(process_train_dir)
    
    #empty tiles in process_train_dir before putting the tiles in it
    for filename in os.listdir(process_train_dir):
        file_path = os.path.join(process_train_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    # Perform image tiling, only kept images of interest
    create_tiles_from_slides(slides_dir=slides_dir, output_dir=full_output_path, clinical_file_dir=clinical_file_dir, subset=subset, N_slides=N_slides)

    # File required for training
    format_tile_data_structure(
        slides_dir=slides_dir,
        output_dir=full_output_path,
        clinical_file_dir=clinical_file_dir,
    )

    # Convert tiles from jpg to TF record1
    file_info = pd.read_csv(f"{full_output_path}/file_info_train.txt", sep="\t")
    training_filenames = list(file_info["tile_path"].values)
    training_classids = [int(id) for id in list(file_info["class_id"].values)]
    tps = [int(id) for id in list(file_info["percent_tumor_cells"].values)]
    Qs = list(file_info["jpeg_quality"].values)

    _convert_dataset(
        split_name="train",
        filenames=training_filenames,
        tps=tps,
        Qs=Qs,
        classids=training_classids,
        output_dir=process_train_dir,
        NUM_SHARDS=N_shards,
    )

    print("Finished converting dataset")
    print(f"The converted data is stored in the directory: {process_train_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--slides_dir", help="Set slides folder")
    parser.add_argument("--output_dir", help="Set output folder")
    parser.add_argument("--clinical_file_dir", help="Set clinical file path")
    parser.add_argument("--N_shards", help="Number of shards", default=320)
    parser.add_argument("--subset", help="true if you want to use only subset of files", default="False")
    parser.add_argument("--N_slides", help="number of slides if subset is set to true", default=10)

    args = parser.parse_args()
    execute_preprocessing(
        slides_dir=args.slides_dir,
        output_dir=args.output_dir,
        clinical_file_dir=args.clinical_file_dir,
        N_shards=args.N_shards,
        subset=args.subset,
        N_slides=args.N_slides
    )

