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
import tifffile 
import sys
import pickle
import skimage.transform
import skimage.exposure
import palom

sys.path.append("/home/evos/THESIS/libs")
import DL.image as im

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login

"""
This file makes UNI features from Orion slides directly, so not from the tile directory. This way you do not necessarilly
need to save the tiles. For this code to work you need to have an account on hugging face and ask acces to the model weights.
"""

login("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")
# pretrained=True needed to load UNI weights (and download weights for the first time)
# init_values need to be passed in to successfully load LayerScale parameters (e.g. - block.0.ls1.gamma)
model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
model.eval()
model.to(device)

def update_resolution_unit(slide_dir):

    with tifffile.TiffFile(slide_dir, mode='r+b') as tif:
        for page in tif.pages:
            #Update resolution unit to centimeters instead of None
            page.tags['ResolutionUnit'].overwrite(3)  #3 
            #print(page.tags)

def grid_shape(coors, patch_length=224):
    return grid_idx(coors, patch_length).max(axis=0) + 1

def grid_idx(coors, patch_length=224):
    index = np.fliplr(
        np.array(coors)[:, :2] /
        patch_length
    ).astype(int)
    return index

def preproc_func_gamma(img):
    return torchvision.transforms.ToTensor()(
        skimage.exposure.adjust_gamma(img, 2.2)
    ).permute(1, 2, 0)

target_image_dir = "/home/evos/Data/CRC/normalization_template.jpg"
target_image = imageio.imread(target_image_dir)
stain_normalizer = stainnorm.get_normalizer('Macenko')
stain_normalizer.fit(target_image)

def preproc_func_macenko(img):
    # Ensure img is writable by making a copy
    img = np.array(img, copy=True)
    normalized_img = stain_normalizer.transform(img)
    return normalized_img

slides_dir = "/home/evos/Data/Orion/Orion_slides"
real_tile_size = 256
tile_size_pixels = 512
MPP_TEST = 0.325 #micrometer per pixel
desired_resolution = real_tile_size / tile_size_pixels
images_for_tiling = sorted(os.listdir(slides_dir))
print(len(images_for_tiling), 'images available:')
batch_size = 50
batch_number = 1
counter=1

features_list = []
tile_ids = []
stain_norm='gamma'

for filename in images_for_tiling:

    slide_dir = os.path.join(slides_dir, filename)
    file_name = slide_dir.split("/")[-1]
    slide_name = file_name.split(".")[0]
    print(counter, ':', slide_name)

    update_resolution_unit(slide_dir)

    #open slide with resolution MPP_TEST
    wsi = TIFFWSIReader.open(slide_dir,
                            mpp=[MPP_TEST, MPP_TEST],
                            power=20,)
    
    patch_extractor = patchextraction.get_patch_extractor(
        input_img=wsi, 
        input_mask=None,
        method_name="slidingwindow",  
        patch_size=(tile_size_pixels, tile_size_pixels),
        stride=(tile_size_pixels, tile_size_pixels),  #set to PATCH_WIDTH so no overlap
        resolution=desired_resolution,
        units="mpp")

    #get all coordinates of the slide in correspondance with patch width
    slide_shape = wsi.slide_dimensions(resolution=desired_resolution, units='mpp')
    coordinates = patch_extractor.get_coordinates(image_shape = slide_shape, 
                                                patch_input_shape=(tile_size_pixels, tile_size_pixels), 
                                                stride_shape = (tile_size_pixels, tile_size_pixels))

    g_shape = grid_shape(coordinates, patch_length=tile_size_pixels)
    c1r = palom.reader.OmePyramidReader(slide_dir)
    level = -1 if len(c1r.pyramid) < 5 else 4
    mask = palom.img_util.entropy_mask(c1r.pyramid[level][1])
    mask = skimage.transform.resize(mask.astype(float), g_shape, order=3) > 0.1

    coordinates_filtered = coordinates[mask.flatten()]

    patch_extractor_masked = patchextraction.get_patch_extractor(
        input_img=wsi, 
        input_mask=mask,
        method_name="slidingwindow",  
        patch_size=(tile_size_pixels, tile_size_pixels),
        stride=(tile_size_pixels, tile_size_pixels),  #set to PATCH_WIDTH so no overlap
        resolution=desired_resolution,
        units="mpp")

    for i in range((len(coordinates_filtered))):
        patch = patch_extractor_masked[i]
        x_coor = coordinates_filtered[i][0]
        y_coor = coordinates_filtered[i][1]
        #coor = np.array([x_coor, y_coor])
        if stain_norm == 'gamma':
            patch_norm = preproc_func_gamma(patch)
            patch_norm_np = patch_norm.cpu().numpy()
            image_patch = Image.fromarray((patch_norm_np*255).astype(np.uint8))
            image = transform(image_patch).unsqueeze(dim=0)  # Shape: [1, 3, 224, 224]
            image = image.to(device)
            tile_id=f"{slide_name}_{x_coor}_{y_coor}"
            with torch.inference_mode():
                feature_emb = model(image)  # Shape: [1, 1024]
            features_list.append(feature_emb.squeeze().cpu().numpy())  # Convert to 1D NumPy array
            tile_ids.append(tile_id)
        elif stain_norm == 'macenko':
            patch_norm = preproc_func_macenko(patch)
            image_patch = Image.fromarray(patch_norm.astype(np.uint8))
            image = transform(image_patch).unsqueeze(dim=0)  # Shape: [1, 3, 224, 224]
            image = image.to(device)
            tile_id=f"{slide_name}_{x_coor}_{y_coor}"
            with torch.inference_mode():
                feature_emb = model(image)  # Shape: [1, 1024]
            features_list.append(feature_emb.squeeze().cpu().numpy())  # Convert to 1D NumPy array
            tile_ids.append(tile_id)
        else:
            image_patch = Image.fromarray(patch.astype(np.uint8))
            image = transform(image_patch).unsqueeze(dim=0)  # Shape: [1, 3, 224, 224]
            image = image.to(device)
            tile_id=f"{slide_name}_{x_coor}_{y_coor}"
            with torch.inference_mode():
                feature_emb = model(image)  # Shape: [1, 1024]
            features_list.append(feature_emb.squeeze().cpu().numpy())  # Convert to 1D NumPy array
            tile_ids.append(tile_id)
    
    # After processing every 50 slides, save the features to file
    if counter % batch_size == 0:
        # Create a DataFrame from features and file paths
        df_features = pd.DataFrame(features_list, columns=[str(i) for i in range(1024)])
        df_features['tile_ID'] = tile_ids
        df_features['slide_submitter_id'] = df_features['tile_ID'].str.rsplit('-', n=1).str[0]
        df_features['sample_submitter_id'] = df_features['slide_submitter_id'].str.rsplit('__', n=1).str[0]
        df_features['Coord_X'] = df_features['tile_ID'].str.split('_').str[-2]
        df_features['Coord_Y'] = df_features['tile_ID'].str.split('_').str[-1]
        df_features['Section'] = df_features['slide_submitter_id'].str.rsplit('__', n=1).str[1]

        # Save the batch to CSV and Parquet
        output_dir = "/home/evos/Outputs/CRC/Orion/UNI_features_orion/macenko"
        df_features.to_csv(f"{output_dir}/features_batch_{batch_number}.csv", index=False)
        df_features.to_parquet(f"{output_dir}/features_batch_{batch_number}.parquet", index=False)

        print(f"Saved batch {batch_number} of features.")
        batch_number += 1
        # Reset the lists for the next batch
        features_list = []
        tile_ids = []

    counter += 1
        
if len(features_list) > 0:
    # Create a DataFrame from features and file paths
    df_features = pd.DataFrame(features_list, columns=[str(i) for i in range(1024)])
    df_features['tile_ID'] = tile_ids
    df_features['slide_submitter_id'] = df_features['tile_ID'].str.rsplit('-', n=1).str[0]
    df_features['sample_submitter_id'] = df_features['slide_submitter_id'].str.rsplit('__', n=1).str[0]
    df_features['Coord_X'] = df_features['tile_ID'].str.split('_').str[-2]
    df_features['Coord_Y'] = df_features['tile_ID'].str.split('_').str[-1]
    df_features['Section'] = df_features['slide_submitter_id'].str.rsplit('__', n=1).str[1]

    output_dir = "/home/evos/Outputs/CRC/Orion/UNI_features_orion/macenko"
    df_features.to_parquet(f"{output_dir}/features_last.parquet", index=False)
    df_features.to_csv(f"{output_dir}/features_last.csv", index=False)

print('Finished')