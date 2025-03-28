# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 20:11:38 2024

@author: 20182460
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 12:26:13 2024

@author: 20182460
"""

import argparse
import os
import git
import pandas as pd
import sys
import shutil
import numpy as np

sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'myslim')))

from create_file_info_train_CRC import format_tile_data_structure
from myslim.datasets.convert import _convert_dataset

REPO_DIR = git.Repo(os.getcwd(), search_parent_directories=True).working_tree_dir
sys.path.append(f"{REPO_DIR}/Python/libs")

def preprocessing_tiles(slides_dir, tiles_dir, output_dir, jpeg_quality, class_name, class_id, percent_tumor_cells,
                        N_shards=320, subset="False", N_slides=10,):
    """
    Execute several pre-processing steps necessary for extracting the histopathological features
    1. Construct file necessary for the deep learning architecture
    2. Convert images of tiles to TF records

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
    full_output_path=f"{output_dir}"
    if not os.path.exists(full_output_path):
        os.makedirs(full_output_path)
    # Create an empty folder for TF records if folder doesn't exist
    process_train_dir = f"{full_output_path}/process_train"
    if not os.path.exists(process_train_dir):
        os.makedirs(process_train_dir)

    # File required for training
    format_tile_data_structure(
        slides_dir=slides_dir,
        tiles_dir=tiles_dir,
        output_dir=full_output_path,
        jpeg_quality=jpeg_quality,
        percent_tumor_cells=percent_tumor_cells,
        class_id=class_id,
        class_name=class_name
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
    parser.add_argument("--tiles_dir", help="Set tiles folder")
    parser.add_argument("--output_dir", help="Set output folder")
    parser.add_argument("--jpeg_quality", help="Set jpeg quality")
    parser.add_argument("--percent_tumor_cells", type=int, help="Set percent_tumor_cells")
    parser.add_argument("--class_id", help="Set class id")
    parser.add_argument("--class_name", help="Set class name")
    parser.add_argument("--N_shards", type=int, help="Number of shards", default=320)
    parser.add_argument("--subset", help="true if you want to use only subset of files", default="False")
    parser.add_argument("--N_slides", type=int, help="number of slides if subset is set to true", default=10)

    args = parser.parse_args()
    preprocessing_tiles(
        slides_dir=args.slides_dir,
        tiles_dir=args.tiles_dir,
        output_dir=args.output_dir,
        jpeg_quality=args.jpeg_quality,
        percent_tumor_cells=args.percent_tumor_cells,
        class_id=args.class_id,
        class_name=args.class_name,
        N_shards=args.N_shards,
        subset=args.subset,
        N_slides=args.N_slides
    )
