
library("pacman")
# p_load(argparse, sva,ComICS )
# p_load_gh("omnideconv/immunedeconv", 'dviraran/xCell', "cansysbio/ConsensusTME")
library(argparse)
library(immunedeconv)

#### Parse arguments from bash ####
parser <- ArgumentParser(description="Compute cell fractions with immunedeconv using TPM from bulkRNAseq data")
parser$add_argument("-i", "--tpm_dir", type="character", default=NULL, help="Full path of input file incl. extension", metavar="tpm_dir")
parser$add_argument("-o", "--output_dir", type="character", default=NULL, help="Directory of Seurat objects", metavar="output_dir")
args <- parser$parse_args()

# immunedeconv offers the following methods:
# - quantiseq
# - mcp_counter
# - xcell
# - epic
input_dir <- args$tpm_dir
output_dir <- args$output_dir

input_dir <- "C:/Users/20182460/Desktop/Master_thesis/Code/Data/TCGA_SKCM/TCGA_tpm_SKCM.txt"
output_dir <- "C:/Users/20182460/Desktop/Master_thesis/Code/Outputs/SKCM"

print(input_dir)
print(output_dir)

# Give TPM-normalized
TPM <- read.csv(input_dir, header = T, sep = "\t")

full_output_dir <- paste0(output_dir, "2_train_multitask_models/immunedeconv/")
if (dir.exists(full_output_dir) == FALSE) {
  dir.create(full_output_dir, recursive = TRUE)
}

## 1) quanTIseq
print("Running quanTIseq")
cell_fractions <- deconvolute_quantiseq(TPM, tumor=TRUE, arrays=FALSE, scale_mrna=TRUE)
write.csv(cell_fractions,paste0(full_output_dir, "quantiseq.csv"), row.names = TRUE)

## 2) MCP Counter
print("Running MCP Counter")
cell_fractions <- deconvolute(TPM, "mcp_counter")
write.csv(cell_fractions,paste0(full_output_dir, "mcp_counter.csv"), row.names = TRUE)

## 3) XCell
print("Running XCell")
cell_fractions <- deconvolute(TPM, "xcell")
write.csv(cell_fractions,paste0(full_output_dir, "xcell.csv"), row.names = TRUE)

## 4) EPIC
print("Running EPIC")
cell_fractions <- deconvolute_epic(TPM, tumor=TRUE, scale_mrna=T)
write.csv(cell_fractions,paste0(full_output_dir, "epic.csv"), row.names = TRUE)

# Session Info
print(sessionInfo())
