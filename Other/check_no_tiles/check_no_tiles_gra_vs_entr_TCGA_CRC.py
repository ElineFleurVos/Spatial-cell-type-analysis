
import logging
# Set the global logging level to ERROR, which suppresses WARNING messages
logging.basicConfig(level=logging.ERROR)
# Suppress warnings for a specific logger by setting its level to ERROR
logging.getLogger('tiatoolbox').setLevel(logging.ERROR)

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
import torchvision.transforms
from tiatoolbox import data
import warnings
warnings.filterwarnings('ignore')
import sys
import shutil
from PIL import Image
sys.path.append("/home/evos/THESIS/libs")
import DL.image as im
import openslide

slide_path = "/home/evos/Data/COLO/subset_CRC_slides/FF/TCGA-3L-AA1B-01A-01-TS1.9C415218-D5B4-4945-B243-F42A4C8C0484.svs"
#slide_path = "/home/evos/Data/COLO/subset_CRC_slides/FF/TCGA-4N-A93T-01A-01-TS1.9258D514-40C1-480A-8FA8-D4E8B3819BDE.svs"
filename = slide_path.split("/")[-1]
slide_name = filename.split(".")[0]

real_tile_size = 112
tile_size_pixels = 512
desired_resolution = real_tile_size/tile_size_pixels
print(desired_resolution)

slide = openslide.OpenSlide(slide_path)
MPP_TEST = slide.properties.get('aperio.MPP', None)
print(MPP_TEST)

if MPP_TEST is None:
    print(f"Slide {slide_name} does not have 'aperio.MPP' property. Skipping...")
else:
    MPP_TEST = float(slide.properties.get('aperio.MPP', 'Not Available'))

def grid_shape(coors, patch_length=224):
    return grid_idx(coors, patch_length).max(axis=0) + 1

def grid_idx(coors, patch_length=224):
    index = np.fliplr(
        np.array(coors)[:, :2] /
        patch_length
    ).astype(int)
    return index

wsi = TIFFWSIReader.open(slide_path,
                         mpp=[MPP_TEST, MPP_TEST],
                         power=20,)

patch_extractor = patchextraction.get_patch_extractor(
    input_img=wsi, 
    input_mask=None,
    method_name="slidingwindow",  
    patch_size=(tile_size_pixels, tile_size_pixels),
    stride=(tile_size_pixels, tile_size_pixels),  #set to PATCH_WIDTH so no overlap
    resolution=desired_resolution,
    units="mpp"
)

slide_shape = wsi.slide_dimensions(resolution=desired_resolution, units='mpp')
coordinates = patch_extractor.get_coordinates(image_shape = slide_shape, 
                                              patch_input_shape=(tile_size_pixels, tile_size_pixels), 
                                              stride_shape = (tile_size_pixels, tile_size_pixels))
print(len(coordinates))

tiles_dir = "/home/evos/Outputs/CRC/test_tiles"

##Method 1 gradient 
#empty tiles folder before putting the tiles in it
for filename in os.listdir(tiles_dir):
    file_path = os.path.join(tiles_dir, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

nr_tiles_after_filtering = 0
for i in range((len(coordinates))):
    patch = patch_extractor[i]
    x_coor = coordinates[i][0]
    y_coor = coordinates[i][1]
    coor = np.array([x_coor, y_coor])
    image_patch = Image.fromarray(patch.astype(np.uint8))
    grad = im.getGradientMagnitude(np.array(image_patch))
    unique, counts = np.unique(grad, return_counts=True)
    if counts[np.argwhere(unique <= 20)].sum() < 512 * 512 * 0.6:
        image_patch.save("{}/{}_{}_{}.jpg".format(
                    tiles_dir, slide_name, x_coor, y_coor),"JPEG",
                    optimize=True,
                    quality=94,)
        nr_tiles_after_filtering += 1
print(nr_tiles_after_filtering)


##Method 2 Entropy
# g_shape = grid_shape(coordinates, patch_length=tile_size_pixels)
# c1r = palom.reader.OmePyramidReader(slide_path)
# level = -1 if len(c1r.pyramid) < 5 else 4
# mask = palom.img_util.entropy_mask(c1r.pyramid[level][1])
# mask = skimage.transform.resize(mask.astype(float), g_shape, order=3) > 0.1

# coordinates_filtered = coordinates[mask.flatten()]
# print(len(coordinates_filtered))

# patch_extractor_masked = patchextraction.get_patch_extractor(
# input_img=wsi, 
# input_mask=mask,
# method_name="slidingwindow",  
# patch_size=(tile_size_pixels, tile_size_pixels),
# stride=(tile_size_pixels, tile_size_pixels),  #set to PATCH_WIDTH so no overlap
# resolution=desired_resolution,
# units="mpp")
