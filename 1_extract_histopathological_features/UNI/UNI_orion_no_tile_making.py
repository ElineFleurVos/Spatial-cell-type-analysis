import logging
import os
from os.path import join as j_
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision
from torchvision.transforms import ToTensor
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login

"""
This file makes UNI features from tiles in the tile directory. For this code to work you need to have an account on 
hugging face and ask acces to the model weights.
"""

# Logging configuration
logging.basicConfig(level=logging.ERROR)

# Hugging Face login
login("insert token here")

# Initialize the UNI model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
model.eval()
model.to(device)

# Directories and configuration
tiles_dir = "/home/evos/Outputs/CRC/Orion/Orion_tiles/Orion_tiles_256um_512pixels_macenko"  # Update to your tiles directory
output_dir = "/home/evos/Outputs/CRC/Orion/UNI_features_orion/macenko"
os.makedirs(output_dir, exist_ok=True)

batch_size = 20000
features_list = []
tile_ids = []
batch_number = 1

# Loop over tiles
all_tiles = sorted(os.listdir(tiles_dir))
print(f"Found {len(all_tiles)} tiles.")

for counter, tile_filename in enumerate(all_tiles, start=1):
    tile_path = j_(tiles_dir, tile_filename)
    tile_id = os.path.splitext(tile_filename)[0]

    # Load the tile and apply transformations
    try:
        tile_image = Image.open(tile_path).convert("RGB")
        image = transform(tile_image).unsqueeze(dim=0).to(device)  # Shape: [1, 3, 224, 224]

        # Extract features
        with torch.inference_mode():
            feature_emb = model(image)  # Shape: [1, 1024]
        features_list.append(feature_emb.squeeze().cpu().numpy())
        tile_ids.append(tile_id)

    except Exception as e:
        logging.error(f"Error processing {tile_filename}: {e}")
        continue

    # Save features in batches
    if counter % batch_size == 0:
        df_features = pd.DataFrame(features_list, columns=[str(i) for i in range(1024)])
        df_features['tile_ID'] = tile_ids
        df_features['slide_submitter_id'] = df_features['tile_ID'].str.rsplit('-', n=1).str[0]
        df_features['sample_submitter_id'] = df_features['slide_submitter_id'].str.rsplit('__', n=1).str[0]
        df_features['Coord_X'] = df_features['tile_ID'].str.split('_').str[-2]
        df_features['Coord_Y'] = df_features['tile_ID'].str.split('_').str[-1]
        df_features['Section'] = df_features['slide_submitter_id'].str.rsplit('__', n=1).str[1]
        
        # Save batch to disk
        df_features.to_parquet(f"{output_dir}/features_batch_{batch_number}.parquet", index=False)
        print(f"Saved batch {batch_number} with {len(features_list)} tiles.")

        # Reset for the next batch
        features_list = []
        tile_ids = []
        batch_number += 1

# Save remaining features
if features_list:
    df_features = pd.DataFrame(features_list, columns=[str(i) for i in range(1024)])
    df_features['tile_ID'] = tile_ids
    df_features['slide_submitter_id'] = df_features['tile_ID'].str.rsplit('-', n=1).str[0]
    df_features['sample_submitter_id'] = df_features['slide_submitter_id'].str.rsplit('__', n=1).str[0]
    df_features['Coord_X'] = df_features['tile_ID'].str.split('_').str[-2]
    df_features['Coord_Y'] = df_features['tile_ID'].str.split('_').str[-1]
    df_features['Section'] = df_features['slide_submitter_id'].str.rsplit('__', n=1).str[1]

    df_features.to_parquet(f"{output_dir}/features_last.parquet", index=False)
    print(f"Saved the last batch with {len(features_list)} tiles.")

print("Feature extraction completed.")
