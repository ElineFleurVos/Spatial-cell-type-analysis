#!/bin/bash

slide_type=FF

# General directions
code_dir=/home/evos/THESIS
output_dir=/home/evos/Outputs/SKCM/FF/run1
data_dir=/home/evos/Data

#NOTE: before you can run this code, you need to have the deconvolution scores made using the immunedeconv libary in R 

# # ------------------------------------------------------------ #
# # ---- Process and create file with tasks for TF learning ---- #
# # ------------------------------------------------------------ #

cancer_type='SKCM'
published_RNA_data_dir=$data_dir/Other/Published_RNA_data
gmt_signatures_dir=$data_dir/Other/gene_signatures.gmt
clinical_file_dir=$output_dir/1_extract_histopathological_features
tpm_dir=$data_dir/TCGA_SKCM/TCGA_tpm_SKCM.txt

python $code_dir/2_cell_type_quantification/general_quantifications/processing_transcriptomics_CRC.py \
    --published_RNA_data_dir $published_RNA_data_dir \
	--gmt_signatures_dir $gmt_signatures_dir \
    --tpm_dir $tpm_dir \
    --clinical_file_dir $clinical_file_dir \
    --output_dir $output_dir/2_train_multitask_models \
	--slide_type $slide_type

# # --------------------------------------------- #
# # ---- TF learning: Multi-task Lasso Model ---- #
# # --------------------------------------------- #

alpha_min=-4
alpha_max=-1
n_steps=40
max_iter=1000
n_outerfolds=5
n_innerfolds=10
n_tiles=50
split_level='sample_submitter_id'

#change back to python 3.8 for TUE018 probably
cell_types_dir=$code_dir/2_train_multitask_models/cell_types_list.txt
cat "$cell_types_dir" | while read celltype; do
	echo "$celltype"
	python $code_dir/3_training/run_TF_pipeline_SKCM.py \
	--output_dir_features $output_dir/1_extract_histopathological_features \
	--output_dir $output_dir/2_train_multitask_models \
	--category $celltype \
	--alpha_min $alpha_min \
	--alpha_max $alpha_max \
	--n_steps $n_steps \
	--max_iter $max_iter \
	--n_outerfolds $n_outerfolds \
	--n_innerfolds $n_innerfolds \
	--n_tiles $n_tiles \
	--split_level $split_level \
	--slide_type $slide_type 
done
