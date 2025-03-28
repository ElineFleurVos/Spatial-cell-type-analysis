# Spatial-cell-type-analysis
This repository contains code belonging to the master thesis project: *Weakly supervised learning for spatial cell type analysis in the tumor microenvironment using pathology and transcriptomics data*. H&E slides and transcriptomics data are used in a weakly supervised learning model to obtain cell type probabilities on a tile level. 

<img src="https://github.com/user-attachments/assets/6d913b21-ff88-40c0-a5ba-d8094fe43614" width="700">

**Feature extraction** --> two approaches: Inception-V4 vs UNI  
**Cell type quantification** --> two approaches: general quantifications vs cancer-specific quantifications  
**Weakly supervised learning** --> two approaches: regularizated multitask linear regression (RMTLR) vs multiple instance learning (MIL)  
**Validation** --> two approaches: fully supervised (Kather) vs immunofluorescence (Lin)

<img src="https://github.com/user-attachments/assets/296304ee-b5e4-412a-91b5-81417ac21db4" width="700">

### DATA
Data used in this project are not included on the GitHub and can be shared on request. The data is structured as follows:
- Orion
  - Orion_HE_cell_type_maps_pickles: Orion cell type maps (pikle files) made by Lin et al (https://www.nature.com/articles/s43018-023-00576-1). This is later redone, so these are not directly used.
  - Orion_HE_cell_type_maps_pngs: Orion cell type maps (png images) made by Lin et al. Again, these are not directly used.
  - Orion_single_cell_tables: Orion single cell tables containing immunofluorescence values of markers on the single cell level.
  - Orion_slides: Orion H&E slides.
- Other
  - CRC_scRNAseq_pelka: single cell CRC transriotomics data.
  - Model_weights_v4: model weights for Inception-V4 feature extraction model.
  - Published_RNA_data: published RNA data used for obtaining general cell type quantifications.
  - gene_signatures.gmt: gene signatures used for obtaining general cell type quantifications.
  - MFP_data.xlsx: filw with extra data of the TCGA images, cancer environment subtypes can be found here.
  - normalization_template.jpg: template used for Macenko stain normalization.
- TCGA_CRC
  - all_GDC_data: all clinical data belonging to the TCGA slides downloaded from GDC data portal.
  - slides: all FF and FFPE CRC TCGA H&E slides downloaded from GDC data portal.
  - subset_slides: a subset of FF and FFPE TCGA H&E slides.
  - clinical_file_TCGA_CRC.tsv: raw clinical file from GDC data portal used to filter slides and make a new clinical file.
  - slide_filenames_CRC_FF.txt: all slide FF filenames needed to make new clinical file.
  - slide_filenames_CRC_FFPE.txt: all slide FFPE filenames needed to make new clinical file.
  - TCGA_counts_CRC.txt: transcriptomics count data from CRC TCGA slides.
  - TCGA_tpm_CRC.txt: transcriptomics tpm data from CRC TCGA slides needed for cell type quantification.
- TCGA_SKCM
  - slides: all FF and FFPE SKCM TCGA H&E slides downloaded from GDC data portal.
  - clinical_file_TCGA_SKCM.txt: raw clinical file used to filter slides and make a new clinical file.
  - TCGA_counts_SKCM.txt: transcriptomics count data from SKCM TCGA slide.
  - TCGA_tpm_SKCM.txt: transcriptomics tpm data from SKCM TCGA slides needed for cell type quantification.
    
### CODE

The code is divided into the four main building blocks of the pipeline. The map Bash-pipeline_SKCM contains bash files were all necessary files are run in order (except the R code for cell type deconvolutions. This has to be done separately beforehand). The SKCM pipeline uses the following approaches: Inception-V4 feature extraction + general quantifications --> RMTLR model --> spatial feature calculation. The map Bash-Pipeline_CRC also contains bash files were all necessary files are run in order. The same approaches are used as for the SKCM pipeline. UNI feature extraction, MIL and the cancer specific quantifications can be run separately and can be found in their respective maps. 

Other things to note:
- for UNI you need to get a huggingface account, see https://github.com/mahmoodlab/UNI?tab=readme-ov-file and https://huggingface.co/MahmoodLab/UNI  


### Environments
Different python environment are needed to run different parts of the code:
- environment_tiatoolbox.yml --> For running code where tiatoolbox is needed (tile making, Kather predictions, stain_normalization):
- environment_rectangle.yml --> For running rectangle (cancer-specific cell type quantifications)
- environment_MIL.yml --> For training MIL 
- environment_UNI.yml --> for making tile features using UNI
- environment_main.yml --> for everything else (for instance RMTLR). If this does not work, environment_tiatoolbox.yml could be tried next. 

