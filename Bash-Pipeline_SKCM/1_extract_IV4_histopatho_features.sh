
# Define type of slide (Fresh-Frozen [FF] vs Formalin-Fixed Paraffin-Embedded [FFPE])
slide_type=FF

# General directions
code_dir=/home/evos/THESIS
output_dir=/home/evos/Outputs/SKCM/$slide_type/run1
data_dir=/home/evos/Data/TCGA_SKCM
slides_dir=/mnt/mnt1/Data/TCGA_SKCM/slides/$slide_type

# ---------------------------------- #
# ---- create new clinical file ---- #
# ---------------------------------- #

########################
## For TCGA datasets: ##
########################

class_name="SKCM_T"
tumor_purity_threshold=80

# Create a filtered clinical file based on thresholded tumor purity
if [[ $slide_type == "FF" ]];then
    python $code_dir/1_extract_histopathological_features/myslim/create_clinical_file_SKCM.py \
        --class_names $class_name \
        --data_dir $data_dir \
        --output_dir $output_dir \
        --code_dir $code_dir \
        --tumor_purity_threshold $tumor_purity_threshold #(OPTIONAL: by default 80) 
fi

# --------------------------------------------------------- #
# ---- image tiling and image conversion to TF records ---- #
# --------------------------------------------------------- #
clinical_file_dir=$output_dir/1_extract_histopathological_features/generated_clinical_file_$slide_type.txt
N_shards=320
subset="False"
N_slides=10

python $code_dir/1_extract_histopathological_features/Inception_V4/preprocessing_SKCM.py \
   --slides_dir $slides_dir \
   --output_dir $output_dir \
   --clinical_file_dir $clinical_file_dir \
   --subset $subset \
   --N_slides $N_slides

# ------------------------------------------------------ #
# ---- Compute predictions and bottlenecks features ---- #
# ------------------------------------------------------ #

#Compute predictions and bottlenecks features using the Retrained_Inception_v4 checkpoints
num_classes=42
bot_out=$output_dir/1_extract_histopathological_features/bot.train.txt
pred_out=$output_dir/1_extract_histopathological_features/pred.train.txt
model_name=inception_v4
checkpoint_path=$data_dir/Model_weights_v4/Retrained_Inception_v4/model.ckpt-100000
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

# ----------------------------------------------------- #
# ---- Post-processing of predictions and features ----- #
# ----------------------------------------------------- #

#Transform bottleneck features, add dummy variable for tissue type for each tile, save predictions in seperate files
python $code_dir/1_extract_histopathological_features/Inception_V4/post_processing.py \
    --codebook_dir $code_dir/1_extract_histopathological_features/codebook.txt \
    --tissue_classes_dir $code_dir/1_extract_histopathological_features/tissue_classes.csv\
    --output_dir $output_dir/1_extract_histopathological_features \
    --slide_type $slide_type

# outputs two files: #output_dir/features 
#                     output_dir/predictions


