#!/bin/bash

#NOTE THAT FIRST TILES NEED TO BE MADE WITH create_orion_tiles.py

# General directions
code_dir=/home/evos/THESIS
output_dir=/home/evos/Outputs/CRC/Orion/1_extract_histopathological_features_macenko
data_dir=/home/evos/Data
slides_dir=/home/evos/Data/Orion/Orion_slides
tiles_dir=/home/evos/Outputs/CRC/Orion/Orion_tiles/Orion_tiles_256um_512pixels_macenko

# --------------------------------------------------------------------- #
# ---- Image conversion to TF records for prediction using model Fu ---- #
# --------------------------------------------------------- ------------#

#Set some random values for these parameters as we are looking at an external dataset.
jpeg_quality="RGBQ50"
percent_tumor_cells=80
class_name="COAD_T"
class_id=6

python $code_dir/1_extract_histopathological_features/Inception_V4/preprocessing_Orion.py \
   --slides_dir $slides_dir \
   --tiles_dir $tiles_dir \
   --output_dir $output_dir \
   --jpeg_quality $jpeg_quality \
   --percent_tumor_cells $percent_tumor_cells \
   --class_name $class_name \
   --class_id $class_id 

# ------------------------------------------------------ #
# ---- Compute predictions and bottlenecks features ---- #
# ------------------------------------------------------ #

#Compute predictions and bottlenecks features using the Retrained_Inception_v4 checkpoints
#pre-processing the same for v4_alt and v4 when not training
#network is the same, pre-processing is different
num_classes=42
bot_out=$output_dir/bot.train.txt
pred_out=$output_dir/pred.train.txt
model_name=inception_v4
checkpoint_path=$data_dir/Other/Model_weights_v4/Retrained_Inception_v4_alt/model.ckpt-100000
eval_image_size=299
file_dir=$output_dir/process_train

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

slide_type="FF" #just for handling of the output, slides are actually FFPE
#Transform bottleneck features, add dummy variable for tissue type for each tile, save predictions in seperate files
python $code_dir/1_extract_histopathological_features/Inception_V4/post_processing.py \
    --codebook_dir $code_dir/1_extract_histopathological_features/codebook.txt \
    --tissue_classes_dir $code_dir/1_extract_histopathological_features/tissue_classes.csv\
    --output_dir $output_dir \
    --slide_type $slide_type

# outputs two files: #output_dir/features 
#                     output_dir/predictions


