import logging
logging.basicConfig(level=logging.ERROR)
logging.getLogger('tiatoolbox').setLevel(logging.ERROR)
import tiatoolbox 
import tiatoolbox.wsicore
import tiatoolbox.tools
from tiatoolbox.wsicore.wsireader import TIFFWSIReader
from tiatoolbox.tools import patchextraction
import torch
import torchvision
import os
from os.path import join as j_
from PIL import Image
import pandas as pd
import numpy as np
import shutil
import openslide
import sys
import pickle

REPO_DIR = git.Repo(os.getcwd(), search_parent_directories=True).working_tree_dir
sys.path.append(f"{REPO_DIR}/Python/libs")
import DL.image as im

#loading all packages here to start
#from uni import get_encoder
#from uni.downstream.extract_patch_features import extract_patch_features_from_dataloader
#from uni.downstream.utils import concat_images
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login

"""
This file makes UNI features from CRC TCGA slides directly, so not from the tile directory. This way you do not necessarilly
need to save the tiles. For this code to work you need to have an account on hugging face and ask acces to the model weights.
"""

def list_images_for_tiling(slides_dir, clinical_file_dir, CRC_tpm_dir, filter_on_bulk=True):
    """ get list with images that you want to tile and save features for """
    clinical_file = pd.read_csv(clinical_file_dir, sep="\t")
    subset_images=clinical_file.image_file_name.tolist()
    print('nr of images in (filtered) clinical file', len(subset_images))

    #TCGA_counts = pd.read_csv(CRC_counts_dir, sep="\t")
    #all_slide_names_counts = TCGA_counts.columns.tolist()
    TCGA_tpm = pd.read_csv(CRC_tpm_dir, sep="\t")
    all_slide_names_tpm = TCGA_tpm.columns.tolist()

    #get list of slide names for which RNA bulk seq data is available
    slides_RNA_available = []
    for i in range(len(clinical_file)):
        slide_id = clinical_file.loc[i, 'sample_submitter_id'][:-1]
        if slide_id in all_slide_names_tpm: #could also use one, lists contain the same data 
            image_file_name = clinical_file.loc[i, 'image_file_name']
            slides_RNA_available.append(image_file_name)
    print('nr of slides for which RNA is available', len(slides_RNA_available))

    available_images=os.listdir(slides_dir)
    print('nr of available slides on disk', len(available_images))

    #get final list of images for tiling, filtered or not filtered on bulke RNA seq availability
    if filter_on_bulk == True:
        print('filtering on availability of bulk RNA seq data')
        images_for_tiling=list(set(subset_images) & set(available_images) & set(slides_RNA_available))
    else:
        images_for_tiling=list(set(subset_images) & set(available_images))

    print(len(images_for_tiling))

    return images_for_tiling

login("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")
# pretrained=True needed to load UNI weights (and download weights for the first time)
# init_values need to be passed in to successfully load LayerScale parameters (e.g. - block.0.ls1.gamma)
model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
model.eval()
model.to(device)

# def tile_feature_extraction(images_for_tiling, real_tile_size, tile_size_pixels):
    
#     desired_resolution = real_tile_size/tile_size_pixels

#     print(len(images_for_tiling), 'images available:')
#     counter=0
#     no_mpp_available = 0
#     features_list = []
#     tile_ids = []
#     for filename in images_for_tiling:

#         slide_dir = os.path.join(slides_dir, filename)
#         filename = slide_dir.split("/")[-1]
#         slide_name = filename.split(".")[0]
#         print(counter, ':', slide_name)

#         #get resolution of slide and set patch size accordingly
#         slide = openslide.OpenSlide(slide_dir)
#         MPP_TEST = slide.properties.get('aperio.MPP', None)

#         if MPP_TEST is None:
#             print(f"Slide {slide_name} does not have 'aperio.MPP' property. Skipping...")
#             no_mpp_available += 1
#             continue
#         else:
#             MPP_TEST = float(slide.properties.get('aperio.MPP', 'Not Available'))
            
#         #load slide with resolution MPP_TEST
#         wsi = TIFFWSIReader.open(slide_dir,
#                                     mpp=[MPP_TEST, MPP_TEST],
#                                     power=20,)
        
#         patch_extractor = patchextraction.get_patch_extractor(
#             input_img=wsi, 
#             input_mask=None,
#             method_name="slidingwindow",  
#             patch_size=(tile_size_pixels, tile_size_pixels),
#             stride=(tile_size_pixels, tile_size_pixels),  #same as patch_size so no overlap
#             resolution=desired_resolution,
#             units="mpp")

#         #get all coordinates of the slide in correspondance with patch width
#         slide_shape = wsi.slide_dimensions(resolution=desired_resolution, units='mpp')
#         coordinates = patch_extractor.get_coordinates(image_shape = slide_shape, 
#                                                         patch_input_shape=(tile_size_pixels, tile_size_pixels), 
#                                                         stride_shape = (tile_size_pixels, tile_size_pixels))

#         for i in range((len(coordinates))):
#             patch = patch_extractor[i]
#             x_coor = coordinates[i][0]
#             y_coor = coordinates[i][1]
#             image_patch = Image.fromarray(patch.astype(np.uint8))
#             image = transform(image_patch).unsqueeze(dim=0)  # Shape: [1, 3, 224, 224]
#             image = image.to(device)
#             tile_id = f"{slide_name}_{x_coor}_{y_coor}"
            
#             grad = im.getGradientMagnitude(np.array(image_patch))
#             unique, counts = np.unique(grad, return_counts=True)
#             if counts[np.argwhere(unique <= 20)].sum() < tile_size_pixels * tile_size_pixels * 0.6:
#                 with torch.inference_mode():
#                     feature_emb = model(image) # Shape: [1, 1024]
#                 features_list.append(feature_emb.squeeze().cpu().numpy())  # Convert to 1D NumPy array
#                 tile_ids.append(tile_id)

#         counter += 1
                
#     print("Number of slides for which mpp is not available", no_mpp_available)

#     df_features = pd.DataFrame(features_list, columns=[str(i) for i in range(1024)])
#     df_features['tile_ID'] = tile_ids
#     df_features['slide_submitter_id'] = df_features['tile_ID'].str.split('_').str[0]
#     df_features['sample_submitter_id'] = df_features['tile_ID'].str.split('-').str[:4].str.join('-')
#     df_features['Coord_X'] = df_features['tile_ID'].str.split('_').str[-2]
#     df_features['Coord_Y'] = df_features['tile_ID'].str.split('_').str[-1]
#     df_features['Section'] = df_features['slide_submitter_id'].str.split('-').str[-1]

#     return df_features

slides_dir = "/home/evos/Data/TCGA_CRC/slides/FFPE"
clinical_file_dir = "/home/evos/Outputs/CRC/FFPE/1_extract_histopathological_features/generated_clinical_file_FFPE_0.txt"
CRC_tpm_dir = "/home/evos/Data/TCGA_CRC/TCGA_tpm_CRC.txt"
output_dir = "/home/evos/Outputs/CRC/UNI_features_TCGA/FFPE"
real_tile_size = 256
tile_size_pixels = 512

print('get images for tiling')
images_for_tiling = list_images_for_tiling(slides_dir, clinical_file_dir, CRC_tpm_dir, filter_on_bulk=True)
# with open('/home/evos/Outputs/CRC/UNI_features_TCGA/FFPE/images_for_tiling_FFPE.pkl', 'rb') as file:
#     images_for_tiling = pickle.load(file)
print(len(images_for_tiling))
print(images_for_tiling[:10])
with open(f"{output_dir}/images_for_tiling_FFPE.pkl", 'wb') as file:
    pickle.dump(images_for_tiling, file)

#set fixed training resolution
print(len(images_for_tiling), 'images available:')
counter = 201
batch_size = 50 # Save features every 50 slides
batch_number = 5

desired_resolution = real_tile_size / tile_size_pixels

no_mpp_available = 0
features_list = []
tile_ids = []

# For every 100 slides, save the features to a file
for filename in images_for_tiling[counter-1:]:

    slide_dir = os.path.join(slides_dir, filename)
    filename = slide_dir.split("/")[-1]
    slide_name = filename.split(".")[0]
    print(counter, ':', slide_name)

    # Get resolution of slide and set patch size accordingly
    slide = openslide.OpenSlide(slide_dir)
    MPP_TEST = slide.properties.get('aperio.MPP', None)

    if MPP_TEST is None:
        print(f"Slide {slide_name} does not have 'aperio.MPP' property. Skipping...")
        no_mpp_available += 1
        continue
    else:
        MPP_TEST = float(slide.properties.get('aperio.MPP', 'Not Available'))
        
    # Load slide with resolution MPP_TEST
    wsi = TIFFWSIReader.open(slide_dir,
                                mpp=[MPP_TEST, MPP_TEST],
                                power=20,)

    patch_extractor = patchextraction.get_patch_extractor(
        input_img=wsi, 
        input_mask=None,
        method_name="slidingwindow",  
        patch_size=(tile_size_pixels, tile_size_pixels),
        stride=(tile_size_pixels, tile_size_pixels),  # Same as patch_size so no overlap
        resolution=desired_resolution,
        units="mpp")

    # Get all coordinates of the slide in correspondence with patch width
    slide_shape = wsi.slide_dimensions(resolution=desired_resolution, units='mpp')
    coordinates = patch_extractor.get_coordinates(image_shape=slide_shape, 
                                                  patch_input_shape=(tile_size_pixels, tile_size_pixels), 
                                                  stride_shape=(tile_size_pixels, tile_size_pixels))

    for i in range(len(coordinates)):
        patch = patch_extractor[i]
        x_coor = coordinates[i][0]
        y_coor = coordinates[i][1]
        image_patch = Image.fromarray(patch.astype(np.uint8))
        image = transform(image_patch).unsqueeze(dim=0)  # Shape: [1, 3, 224, 224]
        image = image.to(device)
        tile_id = f"{slide_name}_{x_coor}_{y_coor}"
        
        # Save the feature vector and tile path
        grad = im.getGradientMagnitude(np.array(image_patch))
        unique, counts = np.unique(grad, return_counts=True)
        if counts[np.argwhere(unique <= 20)].sum() < tile_size_pixels * tile_size_pixels * 0.6:
            with torch.inference_mode():
                feature_emb = model(image)  # Shape: [1, 1024]
            features_list.append(feature_emb.squeeze().cpu().numpy())  # Convert to 1D NumPy array
            tile_ids.append(tile_id)

    # After processing every 50 slides, save the features to file
    if counter % batch_size == 0:
        # Create a DataFrame for the batch
        df_features = pd.DataFrame(features_list, columns=[str(i) for i in range(1024)])
        df_features['tile_ID'] = tile_ids
        df_features['slide_submitter_id'] = df_features['tile_ID'].str.split('_').str[0]
        df_features['sample_submitter_id'] = df_features['tile_ID'].str.split('-').str[:4].str.join('-')
        df_features['Coord_X'] = df_features['tile_ID'].str.split('_').str[-2]
        df_features['Coord_Y'] = df_features['tile_ID'].str.split('_').str[-1]
        df_features['Section'] = df_features['slide_submitter_id'].str.split('-').str[-1]

        # Save the batch to CSV and Parquet
        df_features.to_csv(f"{output_dir}/FF_features_batch_{batch_number}.csv", index=False)
        df_features.to_parquet(f"{output_dir}/FF_features_batch_{batch_number}.parquet", index=False)

        print(f"Saved batch {batch_number} of features.")
        batch_number += 1
        # Reset the lists for the next batch
        features_list = []
        tile_ids = []

    counter += 1

# After loop finishes, save any remaining features
if len(features_list) > 0:
    df_features = pd.DataFrame(features_list, columns=[str(i) for i in range(1024)])
    df_features['tile_ID'] = tile_ids
    df_features['slide_submitter_id'] = df_features['tile_ID'].str.split('_').str[0]
    df_features['sample_submitter_id'] = df_features['tile_ID'].str.split('-').str[:4].str.join('-')
    df_features['Coord_X'] = df_features['tile_ID'].str.split('_').str[-2]
    df_features['Coord_Y'] = df_features['tile_ID'].str.split('_').str[-1]
    df_features['Section'] = df_features['slide_submitter_id'].str.split('-').str[-1]
    
    # Save the last batch
    df_features.to_csv(f"{output_dir}/FF_features_last.csv", index=False)
    df_features.to_parquet(f"{output_dir}/FF_features_last.parquet", index=False)

print("Finished feature extraction")
print("Number of slides for which mpp is not available", no_mpp_available)

## this is for the function above, but it was easier to save by each 50 slides in case something went wrong and then you don't 
## have one big feature files, but multiple files 
# print('start feature extraction')
# df_features = tile_feature_extraction(images_for_tiling, real_tile_size, tile_size_pixels)
# print('finished with feature extraction')
# output_dir = "/home/evos/Outputs/CRC/UNI_features_TCGA"
# df_features.to_parquet(f"{output_dir}/FF_features_parquet.parquet", index=False)
# df_features.to_csv(f"{output_dir}/FF_features.csv", index=False)
# print('Finish saving')
