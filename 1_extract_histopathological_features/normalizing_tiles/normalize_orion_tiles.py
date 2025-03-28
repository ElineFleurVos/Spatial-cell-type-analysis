from tiatoolbox.wsicore.wsireader import OpenSlideWSIReader, WSIReader, TIFFWSIReader
from tiatoolbox.models.dataset.classification import WSIPatchDataset
from pprint import pprint
import tifffile 
from tiatoolbox.tools import patchextraction
#from tiatoolbox.tools.patchextraction import get_coordinates
import numpy as np
import palom
import skimage.transform
import skimage.exposure
from tiatoolbox import logger
from tiatoolbox.models.engine.patch_predictor import (
    IOPatchPredictorConfig,
    PatchPredictor,)
import os 
import pickle
import skimage.exposure
import shutil
from tiatoolbox.tools import stainnorm
import torchvision.transforms
from PIL import Image
from tiatoolbox import data
import imageio.v3 as imageio
import torchvision.transforms as transforms

input_dir = "/home/evos/Outputs/CRC/Orion/Orion_tiles_256um_512pixels"
output_dir = "/home/evos/Outputs/CRC/Orion/Orion_tiles/Orion_tiles_256um_512pixels_macenko"

target_image_dir = "/home/evos/Data/CRC/normalization_template.jpg"
target_image = imageio.imread(target_image_dir)
stain_normalizer = stainnorm.get_normalizer('Macenko')
stain_normalizer.fit(target_image)

#empty tiles folder before putting the tiles in it
for filename in os.listdir(output_dir):
    file_path = os.path.join(output_dir, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

def normalize_image(image_path, stain_normalizer):
    image = imageio.imread(image_path)
    normalized_image = stain_normalizer.transform(image)
    return normalized_image

import warnings
import os
import imageio

# Loop through input tiles and process them
tile_files = os.listdir(input_dir)
for i, input_file in enumerate(tile_files):
    try:
        input_path = os.path.join(input_dir, input_file)
        output_path = os.path.join(output_dir, input_file)
        
        # Temporarily suppress overflow warnings
        with warnings.catch_warnings(record=True) as W:
            warnings.simplefilter("always")  # Catch all warnings
            
            # Normalize the image
            normalized_image = normalize_image(input_path, stain_normalizer)
            
            # Check if any overflow warning occurred
            overflow_warning = any(
                str(warning.message).startswith("overflow encountered") 
                for warning in W
            )
            
            # Skip saving this tile if overflow warning exists
            if overflow_warning:
                print(f"Skipping tile due to overflow: {input_file}")
                continue  # Skip the current tile and move to the next one
            
            # Save the normalized tile
            imageio.imwrite(output_path, normalized_image.astype('uint8'))
        
        # Print progress every 1000 tiles
        if i % 1000 == 0:
            print(f"Processed {i} tiles...")
    
    except ValueError as e:
        # Handle specific warnings related to tissue mask or numerical issues
        if "Empty tissue mask computed" in str(e):
            print(f"Skipping tile due to empty tissue mask: {input_file}")
        else:
            raise  # Re-raise unexpected value errors
    except Exception as e:
        # Log unexpected errors
        print(f"Error processing file {input_file}: {e}")


# # Loop through input tiles and process them
# tile_files = os.listdir(input_dir)
# for i, input_file in enumerate(tile_files):
#     try:
#         input_path = os.path.join(input_dir, input_file)
#         output_path = os.path.join(output_dir, input_file)
        
#         # Normalize the image
#         normalized_image = normalize_image(input_path, stain_normalizer)
#         imageio.imwrite(output_path, normalized_image.astype('uint8'))
        
#         # Print progress every 1000 tiles
#         if i % 1000 == 0:
#             print(f"Processed {i} tiles...")
#     except ValueError as e:
#         # Handle warnings related to tissue mask or numerical issues
#         if "Empty tissue mask computed" in str(e):
#             print(f"Skipping tile due to empty tissue mask: {input_file}")
#         elif "encountered" in str(e):
#             print(f"Skipping tile due to numerical overflow: {input_file}")
#         else:
#             raise
#     except Exception as e:
#         # Log unexpected errors
#         print(f"Error processing file {input_file}: {e}")