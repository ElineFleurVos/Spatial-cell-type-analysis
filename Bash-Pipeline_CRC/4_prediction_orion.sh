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
output_dir=/home/evos/Outputs/CRC/Orion/3_tile_level_quantification_macenko
data_dir=/home/evos/Data

trained_model=FF
prediction_mode="test"

n_outerfolds=5

models_dir="/home/evos/Outputs/CRC/FF/2_train_multitask_models"
histopatho_features_dir="/home/evos/Outputs/CRC/Orion/1_extract_histopathological_features_macenko/features.txt"
MFP_dir="$datadir/CRC/MFP_data"

echo $models_dir
echo $histopatho_features_dir

# ---------------------------------------------------- #
# ---- Predict cell type abundances on tile level ---- #
# ---------------------------------------------------- #
echo $slide_type
echo $trained_model
echo $prediction_mode

python $code_dir/3_training/tile_level_cell_type_quantification_orion.py \
    --models_dir $models_dir\
    --output_dir $output_dir \
    --MFP_dir $MFP_dir \
    --histopatho_features_dir $histopatho_features_dir \
    --prediction_mode $prediction_mode \
    --n_outerfolds $n_outerfolds \
    --slide_type=$slide_type
