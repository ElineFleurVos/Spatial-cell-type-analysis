# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 11:21:32 2024

@author: 20182460
"""

import argparse
import os
import os.path
import sys

import pandas as pd
import git
import numpy as np

REPO_DIR= git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(f"{REPO_DIR}/libs")

# trunk-ignore(flake8/E402)
import DL.utils as utils

# OPENSLIDE_PATH = r'C:\Program Files\openslide-win64-20230414\bin' #path to openslide bin 
# if hasattr(os, 'add_dll_directory'):
#     # Windows
#     with os.add_dll_directory(OPENSLIDE_PATH):
#         from openslide import OpenSlide
# else:
#     from openslide import OpenSlide
    
# # trunk-ignore(flake8/E402)
from openslide import OpenSlide

def format_tile_data_structure(slides_dir, tiles_dir, output_dir, jpeg_quality, percent_tumor_cells, class_id, class_name):
    """
    Specifying the tile data structure required to store tiles as TFRecord files (used in convert.py)

    Args:


    Returns:
        {output_dir}/file_info_train.txt containing the path to the individual tiles, class name, class id, percent of tumor cells and JPEG quality

    """

    all_tile_names = os.listdir(tiles_dir)
    jpg_tile_paths = []
    
    for tile_name in all_tile_names:
        if "jpg" in tile_name:
            jpg_tile_paths.append(tiles_dir + "/" + tile_name)
            
    
    output = pd.DataFrame(jpg_tile_paths, columns=['tile_path'])

    quality_values = len(output)*[jpeg_quality]
    class_names = len(output)*[class_name]
    class_ids = len(output)*[class_id]
    percent_tumor_cells_list = len(output)*[percent_tumor_cells]
    
    output['jpeg_quality'] = quality_values
    output["percent_tumor_cells"] = percent_tumor_cells_list
    output["class_id"] = class_ids
    output["class_name"] = class_names
    
    output.to_csv(output_dir + "/file_info_train.txt", index=False, sep="\t")

    print("Finished creating the necessary file for training in the next step")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--slides_folder", help="Set slides folder")
    parser.add_argument("--tiles_dir", help="set tiles folder")
    parser.add_argument("--output_dir", help="Set output folder")
    parser.add_argument("--jpeg_quality", help="Set jpeg quality")
    parser.add_argument("--percent_tumor_cells", type=int, help="Set percent_tumor_cells")
    parser.add_argument("--class_id", help="Set class id")
    parser.add_argument("--class_name", help="Set class name")
    
    args = parser.parse_args()

    format_tile_data_structure(
        slides_dir=args.slides_folder,
        tiles_dir=args.tiles_dir,
        output_dir=args.output_dir,
        jpeg_quality=args.jpeg_quality,
        percent_tumor_cells=args.percent_tumor_cells,
        class_id=args.class_id,
        class_name=args.class_name
    )