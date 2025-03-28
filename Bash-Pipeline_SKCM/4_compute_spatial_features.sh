#!/bin/bash

###############################
## Compute spatial features  ##
###############################

slide_type=FFPE
trained_model=FFPE #only important for FFPE slide type 

code_dir=/home/evos/THESIS
output_dir=/home/evos/Outputs/SKCM/$slide_type/run1

#Common variables 
if [[ $slide_type == "FF" ]];then
    tile_quantification_dir=$output_dir/3_tile_level_quantification/tcga_validation_tile_predictions_proba.csv
    features_output_dir=$output_dir/4_spatial_features
elif [[ $slide_type == "FFPE" ]];then
    if [[ $trained_model == "FF" ]];then
        tile_quantification_dir=$output_dir/3_tile_level_quantification/tcga_train_validation_tile_predictions_proba.csv
        features_output_dir=$output_dir/4_spatial_features/FF_trained
    elif [[ $trained_model == "FFPE" ]];then 
        tile_quantification_dir=$output_dir/3_tile_level_quantification/tcga_validation_tile_predictions_proba.csv
        features_output_dir=$output_dir/4_spatial_features/FFPE_trained
    fi
fi
cell_types_dir=None
graphs_dir=None

#Variables for network/graph based features
cutoff_path_length=2
shapiro_alpha=0.05
abundance_threshold=0.5

# Variables clustering
n_clusters=8
max_dist=None
max_n_tiles_threshold=2
tile_size=512
overlap=50

# Variables post-processing
metadata_path=""
is_TCGA=True
merge_var="slide_submitter_id"
sheet_name=None

# ---------------------------------- #
# ---- Compute all features -------- #
# ---------------------------------- #

#workflows:
# 1: compute all features
workflow=1

python $code_dir/Other/computing_spatial_features.py \
    --workflow_mode $workflow \
    --tile_quantification_dir $tile_quantification_dir \
    --output_dir $features_output_dir \
    --metadata_path $output_dir/metadata.csv \
    --slide_type $slide_type \
    --cell_types_dir $cell_types_dir \
    --graphs_dir $graphs_dir \
    #--cutoff_path_length $cutoff_path_length \
    #--shapiro_alpha $shapiro_alpha \
    #--abundance_threshold $abundance_threshold \
    #--n_clusters $n_clusters \
    #--max_dist $max_dist \
    #--max_n_tiles_threshold $max_n_tiles_threshold \
    #--tile_size $tile_size \
    #--overlap $overlap \
    #--metadata_path $metadata_path \
    #--is_TCGA $is_TCGA \
    #--merge_var $merge_var \
    #--sheet_name $sheet_name
