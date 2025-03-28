# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 11:54:28 2024

@author: 20182460
"""

#Custom annotation 1
#Tumor cells, B-cells, Mast Cells, T-cells, Macrophages, Stromal cells, Plasma cells  

def custom_anno1(value):
    if value in ['TCD4', 'TCD8', 'Tgd', 'TZBTB16']:
        return 'T'
    if value in ['Schwann', 'Fibro', 'Endo', 'Peri', 'SmoothMuscle']:
        return 'Stromal'
    
    return value

#Custom annotation 2
#Tumor cells, B-cells, Mast Cells, CD4 T-cells, CD8 T-cells, Macrophages, Endothelial cells, Stromal cells other  
#Plasma cells 

def custom_anno2(value):
   if value in ['TCD4']:
       return 'CD4 T' 
   if value in ['TCD8']:
       return 'CD8 T'
   if value in ['Endo']:
       return 'Endothelial'
   if value in ['Schwann', 'Fibro', 'Peri', 'SmoothMuscle']:
       return 'Stromal other'
    
   return value

#Custom annotation 3
#Tumor cells, B-cells, Mast Cells, CD4 T-cells, CD8 T-cells, Macrophages, Myeloid other, Endothelial cells, Stromal cells other, Plasma cells 

def custom_anno3(value):
    if value in ['TCD4']:
        return 'CD4 T' 
    if value in ['TCD8']:
        return 'CD8 T'
    if value in ['Mono', 'DC', 'Granulo']:
        return 'Myeloid other'
    if value in ['Endo']:
        return 'Endothelial'
    if value in ['Schwann', 'Fibro', 'Peri', 'SmoothMuscle']:
        return 'Stromal other'
    
    return value

#Custom annotation 4
#Tumor cells, B-cells, Mast Cells, CD4 T-cells, CD8 T-cells, Macrophages, Myeloid other, Endothelial cells, Stromal cells other, Plasma cells 

def custom_anno4(value):
    if value in ['TCD4', 'TCD8', 'Tgd', 'TZBTB16']:
        return 'T'
    if value in ['Mono', 'DC', 'Granulo']:
        return 'Myeloid other'
    if value in ['Schwann', 'Fibro', 'Endo', 'Peri', 'SmoothMuscle']:
        return 'Stromal'
    
    return value