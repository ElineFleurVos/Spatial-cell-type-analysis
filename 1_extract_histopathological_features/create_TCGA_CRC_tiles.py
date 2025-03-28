import logging
logging.basicConfig(level=logging.ERROR)
logging.getLogger('tiatoolbox').setLevel(logging.ERROR)
from tiatoolbox.wsicore.wsireader import TIFFWSIReader
from tiatoolbox.tools import patchextraction
#from tiatoolbox.tools.patchextraction import get_coordinates
import numpy as np
import os 
import shutil
import openslide
import pandas as pd
import sys 
from PIL import Image
import git 
import argparse

REPO_DIR= git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(f"{REPO_DIR}/libs")
import DL.image as im

# def grid_shape(coors, patch_length=224):
#     return grid_idx(coors, patch_length).max(axis=0) + 1

# def grid_idx(coors, patch_length=224):
#     index = np.fliplr(
#         np.array(coors)[:, :2] /
#         patch_length
#     ).astype(int)
#     return index


def create_CRC_tiles(slides_dir, tiles_dir, clinical_file_dir, CRC_tpm_dir, real_tile_size, tile_size_pixels, filter_on_bulk="True"):

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
    
    # get list of slides filtered on tumor purity from clinical file 
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
    if filter_on_bulk == "True":
        print('filtering on availability of bulk RNA seq data')
        images_for_tiling=list(set(subset_images) & set(available_images) & set(slides_RNA_available))
    else:
        images_for_tiling=list(set(subset_images) & set(available_images))

    #set fixed training resolution
    print(len(images_for_tiling), 'images available:')
    counter=1

    desired_resolution = real_tile_size/tile_size_pixels
    
    no_mpp_available = 0
    for filename in images_for_tiling:

        slide_dir = os.path.join(slides_dir, filename)
        filename = slide_dir.split("/")[-1]
        slide_name = filename.split(".")[0]
        print(counter, ':', slide_name)

        #get resolution of slide and set patch size accordingly
        slide = openslide.OpenSlide(slide_dir)
        MPP_TEST = slide.properties.get('aperio.MPP', None)

        if MPP_TEST is None:
            print(f"Slide {slide_name} does not have 'aperio.MPP' property. Skipping...")
            no_mpp_available += 1
            continue
        else:
            MPP_TEST = float(slide.properties.get('aperio.MPP', 'Not Available'))
            
        #load slide with resolution MPP_TEST
        wsi = TIFFWSIReader.open(slide_dir,
                                 mpp=[MPP_TEST, MPP_TEST],
                                 power=20,)
        
        patch_extractor = patchextraction.get_patch_extractor(
            input_img=wsi, 
            input_mask=None,
            method_name="slidingwindow",  
            patch_size=(tile_size_pixels, tile_size_pixels),
            stride=(tile_size_pixels, tile_size_pixels),  #same as patch_size so no overlap
            resolution=desired_resolution,
            units="mpp")

        #get all coordinates of the slide in correspondance with patch width
        slide_shape = wsi.slide_dimensions(resolution=desired_resolution, units='mpp')
        coordinates = patch_extractor.get_coordinates(image_shape = slide_shape, 
                                                      patch_input_shape=(tile_size_pixels, tile_size_pixels), 
                                                      stride_shape = (tile_size_pixels, tile_size_pixels))
    
        for i in range((len(coordinates))):
            patch = patch_extractor[i]
            x_coor = coordinates[i][0]
            y_coor = coordinates[i][1]
            image_patch = Image.fromarray(patch.astype(np.uint8))
            grad = im.getGradientMagnitude(np.array(image_patch))
            unique, counts = np.unique(grad, return_counts=True)
            if counts[np.argwhere(unique <= 20)].sum() < tile_size_pixels * tile_size_pixels * 0.6:
                image_patch.save("{}/{}_{}_{}.jpg".format(
                            tiles_dir, slide_name, x_coor, y_coor),"JPEG",
                            optimize=True,
                            quality=94,)
                
        counter += 1
                
    print("Finished creating tiles")
    print("Number of slides for which mpp is not available", no_mpp_available)


# slides_dir = "/mnt/mnt1/Data/COLO/Slides/COLO_FFPE"
# tiles_dir = "/home/evos/Outputs/COLO/FFPE/1_extract_histopathological_features/tiles_small"
# clinical_file_dir = "/home/evos/Outputs/COLO/FFPE/1_extract_histopathological_features/generated_clinical_file_FFPE.txt"
# CRC_counts_dir = "/home/evos/Data/COLO/TCGA_counts_COLO.txt"
# CRC_tpm_dir = "/home/evos/Data/COLO/TCGA_tpm_COLO.txt"
# create_CRC_tiles(slides_dir=slides_dir, tiles_dir=tiles_dir, clinical_file_dir=clinical_file_dir, 
#                  CRC_counts_dir=CRC_counts_dir, CRC_tpm_dir=CRC_tpm_dir, filter_on_bulk=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--slides_dir", help="Set slides folder")
    parser.add_argument("--tiles_dir", help="Set tiles folder")
    parser.add_argument("--clinical_file_dir", help="Set clinical file path")
    #parser.add_argument("--CRC_counts_dir", help="Set counts dir")
    parser.add_argument("--CRC_tpm_dir", help="Set tpm dir")
    parser.add_argument("--real_tile_size", type=int, help="set desired physical size of tile")
    parser.add_argument("--tile_size_pixels", type=int, help="set desired number of pixels of tile")
    parser.add_argument("--filter_on_bulk", help="set to True if you want to filter on bulk", default="True")

    args = parser.parse_args()
    create_CRC_tiles(
        slides_dir=args.slides_dir,
        tiles_dir=args.tiles_dir,
        clinical_file_dir=args.clinical_file_dir,
        #CRC_counts_dir=args.CRC_counts_dir,
        CRC_tpm_dir=args.CRC_tpm_dir,
        real_tile_size=args.real_tile_size,
        tile_size_pixels=args.tile_size_pixels,
        filter_on_bulk=args.filter_on_bulk
    )
