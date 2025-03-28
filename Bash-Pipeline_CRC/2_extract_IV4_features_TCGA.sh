#!/bin/bash

#NOTE THAT FIRST TILES NEED TO BE MADE WITH create_TCGA_CRC_tiles.py

# Define type of slide (Fresh-Frozen [FF] vs Formalin-Fixed Paraffin-Embedded [FFPE])
slide_type=FF

# General directions (puth correct directions here)
code_dir=/home/evos/THESIS
output_dir=/home/evos/Outputs/CRC/FF_macenko
data_dir=/home/evos/Data
slides_dir=/home/evos/Data/TCGA_CRC/slides/$slide_type

# --------------------------------------------------------------------- #
# ---- Image conversion to TF records for predicton using model Fu ---- #
# --------------------------------------------------------- ------------#

clinical_file_dir=$output_dir/1_extract_histopathological_features/generated_clinical_file_$slide_type.txt
tiles_dir=/home/evos/Outputs/CRC/tiles_macenko/tiles_FF_256_macenko

python $code_dir/1_extract_histopathological_features/Inception_V4/preprocessing_CRC.py \
   --slides_dir $slides_dir \
   --tiles_dir $tiles_dir \
   --output_dir $output_dir \
   --clinical_file_dir $clinical_file_dir \


# ------------------------------------------------------ #
# ---- Compute predictions and bottlenecks features ---- #
# ------------------------------------------------------ #

#Compute predictions and bottlenecks features using the Retrained_Inception_v4 checkpoints
#pre-processing the same for v4_alt and v4 when not training
#network is the same, pre-processing is different

num_classes=42
bot_out=$output_dir/1_extract_histopathological_features/bot.train.txt
pred_out=$output_dir/1_extract_histopathological_features/pred.train.txt
model_name=inception_v4
checkpoint_path=$data_dir/Other/Model_weights_v4/Retrained_Inception_v4_alt/model.ckpt-100000
eval_image_size=299
file_dir=$output_dir/1_extract_histopathological_features/process_train

python $code_dir/1_extract_histopathological_features/Inception_V4/bottleneck_predict.py \
    --num_classes $num_classes \
    --bot_out $bot_out \
    --pred_out $pred_out \
    --model_name $model_name \
    --checkpoint_path $checkpoint_path \
    --eval_image_size $eval_image_size \
    --file_dir $file_dir

# # ----------------------------------------------------- #
# # ---- Post-processing of predictions and features ----- #
# # ----------------------------------------------------- #

#Transform bottleneck features, add dummy variable for tissue type for each tile, save predictions in seperate files
python $code_dir/1_extract_histopathological_features/Inception_V4/post_processing.py \
    --codebook_dir $code_dir/1_extract_histopathological_features/codebook.txt \
    --tissue_classes_dir $code_dir/1_extract_histopathological_features/tissue_classes.csv\
    --output_dir $output_dir/1_extract_histopathological_features \
    --slide_type $slide_type

#outputs two files: #output_dir/features 
                    #output_dir/predictions


