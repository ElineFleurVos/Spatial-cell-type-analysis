# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 16:12:09 2024

@author: 20182460
"""

import numpy as np 

def grid_shape(coors, patch_length=224):
    return grid_idx(coors, patch_length).max(axis=0) + 1

def grid_idx(coors, patch_length=224):
    index = np.fliplr(
        np.array(coors)[:, :2] /
        patch_length
    ).astype(int)
    return index
