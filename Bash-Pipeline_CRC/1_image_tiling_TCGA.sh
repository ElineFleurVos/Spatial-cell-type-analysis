#!/bin/bash

# Define type of slide (Fresh-Frozen [FF] vs Formalin-Fixed Paraffin-Embedded [FFPE])
slide_type=FFPE

# General directions (adjust to your own directories)
code_dir=/home/evos/THESIS
output_dir=/home/evos/Outputs/CRC/$slide_type
data_dir=/home/evos/Data/TCGA_CRC
slides_dir=/home/evos/Data/TCGA_CRC/slides/$slide_type

# ---------------------------------- #
# ---- create new clinical file ---- #
# ---------------------------------- #

tumor_purity_threshold=80

# Create a filtered clinical file based on thresholded tumor purity
python $code_dir/1_extract_histopathological_features/myslim/create_clinical_file_CRC.py \
    --data_dir $data_dir \
    --output_dir $output_dir \
    --code_dir $code_dir \
    --slide_type $slide_type \
    --tumor_purity_threshold $tumor_purity_threshold #(OPTIONAL: by default 80) 

#----------------------- #
# ---- Image tiling ---- #
# ---------------------- #   

tiles_dir=$output_dir/1_extract_histopathological_features/tiles_256um
clinical_file_dir=/home/evos/Outputs/CRC/FF/1_extract_histopathological_features/generated_clinical_file_FF.txt #either this or generated_clinical_file_FF_0
CRC_tpm_dir=$data_dir/TCGA_tpm_CRC.txt
real_tile_size=256
tile_size_pixels=512
filter_on_bulk="True"

python $code_dir/1_extract_histopathological_features/myslim/create_TCGA_CRC_tiles.py \
    --slides_dir $slides_dir \
    --tiles_dir $tiles_dir \
    --clinical_file_dir $clinical_file_dir \
    --CRC_tpm_dir $CRC_tpm_dir \
    --real_tile_size $real_tile_size \
    --tile_size_pixels $tile_size_pixels \
    --filter_on_bulk $filter_on_bulk 