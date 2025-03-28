# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 11:44:19 2024

@author: 20182460
"""

import sys 
import git 
import cv2
from tiatoolbox.tools import stainnorm

REPO_DIR= git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(f"{REPO_DIR}/libs")

def preproc_func_macenko(img):
    target_image_dir = "normalization_template.jpg"
    target_image = cv2.imread(target_image_dir)
    stain_normalizer = stainnorm.get_normalizer('Macenko')
    stain_normalizer.fit(target_image)
    normalized_img = stain_normalizer.transform(img.copy())

    return normalized_img
