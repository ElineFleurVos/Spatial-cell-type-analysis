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

def update_resolution_unit(slide_dir):

    with tifffile.TiffFile(slide_dir, mode='r+b') as tif:
        for page in tif.pages:
            #Update resolution unit to centimeters instead of None
            page.tags['ResolutionUnit'].overwrite(3)  #3 
            #print(page.tags)

def grid_shape(coors, patch_length):
    return grid_idx(coors, patch_length).max(axis=0) + 1

def grid_idx(coors, patch_length):
    index = np.fliplr(
        np.array(coors)[:, :2] /
        patch_length
    ).astype(int)
    return index

# def preproc_func_gamma(img):
#     return skimage.exposure.adjust_gamma(img, 2.2)  # Apply gamma correction
 
def preproc_func_gamma(img):
    return transforms.ToTensor()(
        skimage.exposure.adjust_gamma(img, 2.2)
    ).permute(1, 2, 0)

target_image_dir = "/home/evos/Data/CRC/normalization_template.jpg"
target_image = imageio.imread(target_image_dir)
stain_normalizer = stainnorm.get_normalizer('Macenko')
stain_normalizer.fit(target_image)

# def preproc_func_macenko(img):
#     img = np.array(img)
#     normalized_img = stain_normalizer.transform(img)
#     return transforms.ToTensor()(normalized_img).permute(1, 2, 0)  

def preproc_func_macenko(img):
    # Ensure img is writable by making a copy
    img = np.array(img, copy=True)
    normalized_img = stain_normalizer.transform(img)
    return normalized_img

def create_orion_tiles(slides_dir, tiles_dir, real_tile_size, tile_size_pixels, stain_norm):

    #tiles_dir = "{}/tiles".format(output_dir)
    if not os.path.exists(tiles_dir):
        os.makedirs(tiles_dir)
    
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

    MPP_TEST = 0.325 #micrometer per pixel
    desired_resolution = real_tile_size/tile_size_pixels

    slides = os.listdir(slides_dir)
    print(len(slides), 'images available:')
    counter=1
    
    for filename in os.listdir(slides_dir):

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
                image_patch.save("{}/{}_{}_{}.jpg".format(
                            tiles_dir, slide_name, x_coor, y_coor),"JPEG",
                            optimize=True,
                            quality=94,)
            elif stain_norm == 'macenko':
                patch_norm = preproc_func_macenko(patch)
                image_patch = Image.fromarray(patch_norm.astype(np.uint8))
                image_patch.save( "{}/{}_{}_{}_macenko.jpg".format(
                                tiles_dir, slide_name, x_coor, y_coor),"JPEG",
                                optimize=True,
                                quality=94,)
            else:
                image_patch = Image.fromarray(patch.astype(np.uint8))
                image_patch.save("{}/{}_{}_{}.jpg".format(
                            tiles_dir, slide_name, x_coor, y_coor),"JPEG",
                            optimize=True,
                            quality=94,)
        counter=counter+1
        
    print("Finished creating Orion tiles")

real_tile_size=256
tile_size_pixels=512
#slides_dir = "/home/evos/Data/CRC/Orion_slides"
slides_dir = "/home/evos/Data/CRC/Orion/Orion_slides"
tiles_dir = "/home/evos/Outputs/CRC/Orion/Orion_tiles/Orion_tiles_256um_512pixels_macenko"
create_orion_tiles(slides_dir=slides_dir, tiles_dir=tiles_dir, 
                   real_tile_size=real_tile_size, tile_size_pixels=tile_size_pixels,
                   stain_norm='macenko')
