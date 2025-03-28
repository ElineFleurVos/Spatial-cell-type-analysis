# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:45:25 2025

@author: 20182460
"""
import pickle
    
#Load MIL data 
MIL_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\MIL\MIL_final_pcchip"
with open(f"{MIL_dir}\AUC_results_tumor_MIL.pkl", "rb") as file:
    AUC_tumor_MIL = pickle.load(file)
with open(f"{MIL_dir}\AUC_results_T_MIL.pkl", "rb") as file:
    AUC_T_MIL = pickle.load(file)
with open(f"{MIL_dir}\cadpan_99.pkl", "rb") as file:
    IF_cadpan_MIL = pickle.load(file)
with open(f"{MIL_dir}\E-cadherin_99.pkl", "rb") as file:
    IF_cadherin_MIL = pickle.load(file)
with open(f"{MIL_dir}\Pan-CK_99.pkl", "rb") as file:
    IF_panck_MIL = pickle.load(file)
with open(f"{MIL_dir}\CD3e_99.pkl", "rb") as file:
    IF_CD3e_MIL = pickle.load(file)
with open(f"{MIL_dir}\CD31_99.pkl", "rb") as file:
    IF_CD31_MIL = pickle.load(file)

#Load RMTLR data 
RMTLR_dir = r"C:\Users\20182460\Desktop\Master_thesis\Code\Outputs\FINAL RESULTS\CRC\multitask_lasso"
with open(f"{RMTLR_dir}\AUC_results_tumor_RMTLR.pkl", "rb") as file:
    AUC_tumor_RMTLR = pickle.load(file)
with open(f"{RMTLR_dir}\AUC_results_T_RMTLR.pkl", "rb") as file:
    AUC_T_RMTLR = pickle.load(file)
with open(f"{RMTLR_dir}\cadpan_99.pkl", "rb") as file:
    IF_cadpan_RMTLR = pickle.load(file)
with open(f"{RMTLR_dir}\E-cadherin_99.pkl", "rb") as file:
    IF_cadherin_RMTLR = pickle.load(file)
with open(f"{RMTLR_dir}\Pan-CK_99.pkl", "rb") as file:
    IF_panck_RMTLR = pickle.load(file)
with open(f"{RMTLR_dir}\CD3e_99.pkl", "rb") as file:
    IF_CD3e_RMTLR = pickle.load(file)
with open(f"{RMTLR_dir}\CD31_99.pkl", "rb") as file:
    IF_CD31_RMTLR = pickle.load(file)

#%%
slide_id_list = list(IF_cadpan_MIL['separate_corrs'].keys())
slide_id_nr = 13
slide_id = slide_id_list[slide_id_nr]
print(slide_id)

# IF_MIL_cadpan = IF_cadpan_MIL['separate_corrs'][slide_id][0]
# print(f"IF MIL cadpan: {IF_MIL_cadpan:.3f}")
# IF_RMTLR_cadpan = IF_cadpan_RMTLR['separate_corrs'][slide_id][0]
# print(f"IF RMTLR cadpan: {IF_RMTLR_cadpan:.3f}")
IF_MIL_cadherin = IF_cadherin_MIL['separate_corrs'][slide_id][0]
print(f"IF MIL cadherin: {IF_MIL_cadherin:.3f}")
IF_RMTLR_cadherin = IF_cadherin_RMTLR['separate_corrs'][slide_id][0]
print(f"IF RMTLR cadherin: {IF_RMTLR_cadherin:.3f}")
IF_MIL_panck = IF_panck_MIL['separate_corrs'][slide_id][0]
print(f"IF MIL Pan-CK: {IF_MIL_panck:.3f}")
IF_RMTLR_panck = IF_panck_RMTLR['separate_corrs'][slide_id][0]
print(f"IF RMTLR Pan-CK: {IF_RMTLR_panck:.3f}")
AUC_MIL_tumor = AUC_tumor_MIL['AUC_values'][slide_id]
print(f"AUC MIL tumor: {AUC_MIL_tumor:.3f}")
AUC_RMTLR_tumor = AUC_tumor_RMTLR['AUC_values'][slide_id]
print(f"AUC RMTLR tumor: {AUC_RMTLR_tumor:.3f}")

print("\n")

IF_MIL_CD3e = IF_CD3e_MIL['separate_corrs'][slide_id][0]
print(f"IF MIL CD3e: {IF_MIL_CD3e:.3f}")
IF_RMTLR_CD3e = IF_CD3e_RMTLR['separate_corrs'][slide_id][0]
print(f"IF RMTLR CD3e: {IF_RMTLR_CD3e:.3f}")
AUC_MIL_T = AUC_T_MIL['AUC_values'][slide_id]
print(f"AUC MIL T: {AUC_MIL_T:.3f}")
AUC_RMTLR_T = AUC_T_RMTLR['AUC_values'][slide_id]
print(f"AUC RMTLR T: {AUC_RMTLR_T:.3f}")

print("\n")

IF_MIL_CD31 = IF_CD31_MIL['separate_corrs'][slide_id][0]
print(f"IF MIL CD31: {IF_MIL_CD31:.3f}")
IF_RMTLR_CD31 = IF_CD31_RMTLR['separate_corrs'][slide_id][0]
print(f"IF RMTLR CD31: {IF_RMTLR_CD31:.3f}")

#%%
#average_first_values = sum(value[0] for value in IF_panck_MIL['separate_corrs'].values()) / len(IF_panck_MIL['separate_corrs'])
#print("Average of first values:", average_first_values)

