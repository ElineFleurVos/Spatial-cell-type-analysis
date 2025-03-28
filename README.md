# Spatial-cell-type-analysis
This repository contains code belonging to the master thesis project: Weakly supervised learning for spatial cell type analysis in the tumor microenvironment using pathology and transcriptomics data. H&E sldies and transcriptomics data are used in a weakly supervised learning model to obtain cell type probabilities on a tile level. 

<img src="https://github.com/user-attachments/assets/6d913b21-ff88-40c0-a5ba-d8094fe43614" width="700">

**Feature extraction** --> two approaches: Inception-V4 vs UNI  
**Cell type quantification** --> two approaches: general quantifications vs cancer-specific quantifications  
**Weakly supervised learning** --> two approaches: regularizated multitask linear regression (RMTLR) vs multiple instance learning (MIL)  
**Validation** --> two approaches: comparison fully supervised (Kather) vs immunofluorescence (Lin)

<img src="https://github.com/user-attachments/assets/296304ee-b5e4-412a-91b5-81417ac21db4" width="700">

### DATA
Data used in this project are not included on the GitHub and can be shared on request. The data is structured as follows:
- Orion
  - Orion_HE_cell_type_maps_pickles: Orion cell type maps (pikle files) made by Lin et al (https://www.nature.com/articles/s43018-023-00576-1). This is later redone, so these are not directly used.
  - Orion_HE_cell_type_maps_pngs: Orion cell type maps (png images) made by Lin et al. Again, these are not directly used.
  - Orion_single_cell_tables: Orion single cell tables containing immunofluorescence values of markers on the single cell level
  - Orion_slides: Orion H&E slides
- Other
- TCGA_CRC
- TCGA_SKCM

### CODE
for UNI you need to get a huggingface account, see https://github.com/mahmoodlab/UNI?tab=readme-ov-file and https://huggingface.co/MahmoodLab/UNI 

The bash pipelines do this and this. 

For the MIL pipeline all files are in 3_training/tain_MIL
### Environments

Everything for which tiatoolbox is used ...



