#!/bin/bash

#####################################################################
## Compute cell-type quantification from transfer learning models  ##
#####################################################################

# ----------------------------------- #
# --------- Setup file paths -------- #
# ----------------------------------- #

# Define type of slide (Fresh-Frozen [FF] vs Formalin-Fixed Paraffin-Embedded [FFPE])
slide_type=FF

# General directions
code_dir=/home/evos/THESIS
output_dir=/home/evos/Outputs/SKCM/$slide_type/run1
data_dir=/home/evos/Data

trained_model=FF
prediction_mode="tcga_validation"
#prediction_mode="tcga_train_validation" # Just used when  making FFPE predictions using FF trained models.

MFP_dir=$data_dir/Other/MFP_data.xlsx
n_outerfolds=5

if [[ $slide_type == "FF" ]];then
    models_dir=$output_dir/2_train_multitask_models
    histopatho_features_dir=$output_dir/1_extract_histopathological_features/features.txt
elif [[ $slide_type == "FFPE" ]];then
    histopatho_features_dir=$output_dir/1_extract_histopathological_features/features_format_parquet
    if [[ $trained_model == "FF" ]];then
        models_dir=/home/evos/Outputs/SKCM/FF/run1/2_train_multitask_models
    elif [[ $trained_model == "FFPE" ]];then
        models_dir=/home/evos/Outputs/SKCM/FFPE/run1/2_train_multitask_models
    fi
fi
echo $models_dir
echo $histopatho_features_dir

# ---------------------------------------------------- #
# ---- Predict cell type abundances on tile level ---- #
# ---------------------------------------------------- #
echo $slide_type
echo $trained_model

python $code_dir/3_training/tile_level_cell_type_quantification.py \
    --models_dir $models_dir\
    --output_dir $output_dir \
    --MFP_dir $MFP_dir \
    --histopatho_features_dir $histopatho_features_dir \
    --prediction_mode $prediction_mode \
    --n_outerfolds $n_outerfolds \
    --slide_type=$slide_type
