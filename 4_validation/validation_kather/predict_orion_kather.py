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
from tiatoolbox.tools import stainnorm
import cv2

def preproc_func_gamma(img):
    return torchvision.transforms.ToTensor()(
        skimage.exposure.adjust_gamma(img, 2.2)
    ).permute(1, 2, 0)

def preproc_func_macenko(img):
    target_image_dir = "/home/evos/Data/COLO/normalization_template.jpg"  #SET CORRECT TARGET IMAGE PATH
    target_image = cv2.imread(target_image_dir)
    stain_normalizer = stainnorm.get_normalizer('Macenko')
    stain_normalizer.fit(target_image)
    normalized_img = stain_normalizer.transform(img.copy())

    return normalized_img

def predict_orion_kather(tiles_dir, slides_dir, output_dir, preproc_func, pretrained_model, batch_size=32):

    slides = os.listdir(slides_dir)
    counter=1
    for file_name in slides:
        slide_dir = os.path.join(slides_dir, file_name)
        filename = slide_dir.split("/")[-1]
        slide_name = filename.split(".")[0]
        print(slide_name)

        print(counter)
        patch_list = []
        coordinates_ordered = np.empty((0, 2), dtype=np.int64)
        for tile_name in os.listdir(tiles_dir):
            if slide_name in tile_name:
                full_path = os.path.join(tiles_dir, tile_name)
                patch_list.append(full_path)
                tile_name = tile_name.split(".")[0]
                x_coor = int(tile_name.split("_")[-2])
                y_coor = int(tile_name.split("_")[-1])
                coor = np.array([x_coor, y_coor])
                coordinates_ordered = np.vstack([coordinates_ordered, coor])

        ON_GPU = True

        pretrained_model=pretrained_model
        predictor = PatchPredictor(pretrained_model=pretrained_model, batch_size=batch_size)

        if preproc_func == 'gamma':
            predictor.model.preproc_func = preproc_func_gamma
        elif preproc_func == "macenko":
            predictor.model.preproc_func = preproc_func_macenko

        output = predictor.predict(imgs=patch_list, mode="patch", on_gpu=ON_GPU, return_probabilities=True)

        output["coordinates"]=coordinates_ordered

        preproc_func = 'gamma'
        predictions_file_path = "{}/{}_{}_{}_cell_type_map.pkl".format(output_dir, slide_name, preproc_func, pretrained_model)
        print(predictions_file_path)

        with open(predictions_file_path, 'wb') as f:
            pickle.dump(output, f)

        counter=counter+1

    print("Finished making Orion predictions")

tiles_dir = "/home/evos/Outputs/CRC/Orion/Orion_tiles_112um_224pixels"
slides_dir = "/home/evos/Data/CRC/Orion_slides"
output_dir = "/home/evos/Outputs/CRC/Orion/Orion_predictions_112um_224pixels"
preproc_func = "gamma"
pretrained_model = "densenet161-kather100k"
    
predict_orion_kather(tiles_dir=tiles_dir, slides_dir=slides_dir, output_dir=output_dir, preproc_func="gamma", pretrained_model=pretrained_model)

