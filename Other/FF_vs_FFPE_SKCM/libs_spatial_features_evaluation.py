# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 15:39:30 2024

@author: 20182460
"""

import pandas as pd 
from scipy.stats import pearsonr, spearmanr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel, wilcoxon, shapiro

def clean_graph_feature_df(df):
    IDs = ['TCGA patient ID', 'slide_submitter_id', 'sample submitter id'] #remove IDs from dataframe 
    df_cleaned = df.drop(columns=IDs)
   
    #substitute all nan's with zeros
    df_cleaned.fillna(0, inplace=True)
    
        
    #check for duplicates in index. If all duplicates have the exact same row, keep one, if they are different, 
    #remove all. 
    duplicate_indices = df_cleaned[df_cleaned.index.duplicated()].index.unique().tolist()
    for duplicate_index in duplicate_indices:
        duplicate_rows = df_cleaned.loc[duplicate_index]
        for i in range(len(duplicate_rows)):
            are_all_same = duplicate_rows.iloc[i].equals(duplicate_rows.iloc[0])
            if are_all_same == False:
                break
        if are_all_same == True:
            row_to_keep = duplicate_rows.iloc[0:1]
            df_cleaned = df_cleaned.drop(index=duplicate_index)
            df_cleaned = pd.concat([df_cleaned, row_to_keep])
        else:
            df_cleaned = df_cleaned.drop(index=duplicate_index)
  
    return df_cleaned

def calc_correlations_slow(merged_df, min_pairs=2, corr_method='pearson'):
 
    #merge dataframe to allign slides + add suffixes 
    #merged_df = pd.merge(df1, df2, left_index=True, right_index=True, suffixes=(' 1', ' 2'))
    
    #easier way for correlation matrix but without p-values 
    #correlation_matrix = merged_df.corr(method=corr_method, min_periods=2)
    
    #more dificult way with p-values. P-values are calculated by a permutation test 
    correlation_matrix = pd.DataFrame(index=merged_df.columns, columns=merged_df.columns)
    p_values_matrix = pd.DataFrame(index=merged_df.columns, columns=merged_df.columns)

    for col1 in merged_df.columns:
        for col2 in merged_df.columns:
            # Select the columns to analyze
            if col1 == col2:
                clean_df = merged_df[[col1]].copy()
                clean_df = clean_df.dropna()
            else:
                clean_df = merged_df[[col1, col2]].copy()
                clean_df = clean_df.dropna()
            
            # Remove rows with non-finite values for the selected columns
            valid_indices = np.isfinite(clean_df[col1]) & np.isfinite(clean_df[col2])
            clean_df = clean_df.loc[valid_indices]
    
            # Calculate correlation coefficient and p-value if there are enough remaining samples
            if len(clean_df) >= min_pairs:
                if corr_method == 'pearson':
                    corr_coef, p_value = pearsonr(clean_df[col1].values, clean_df[col2].values)
                elif corr_method == 'spearman':
                    corr_coef, p_value = spearmanr(clean_df[col1].values, clean_df[col2].values)
                else:
                    raise ValueError(f"Unsupported correlation method: {corr_method}")
                
                # Store the results
                correlation_matrix.loc[col1, col2] = corr_coef
                p_values_matrix.loc[col1, col2] = p_value


    #now we need to find the interesting correlations, so only the correlations between the same 
    #features
    interesting_correlations = {}
    interesting_p_values = {}
    #Loop over row names and column names and save the correct correlations (-2 because of suffix)
    for i in range(len(correlation_matrix.index)):
        for c in range(len(correlation_matrix.columns)):
            if (correlation_matrix.index[i] != correlation_matrix.columns[c]) and correlation_matrix.index[i][:-2] == correlation_matrix.columns[c][:-2]:
                interesting_correlations[correlation_matrix.index[i][:-2]] = correlation_matrix.iloc[i,c]   
                interesting_p_values[p_values_matrix.index[i][:-2]] = p_values_matrix.iloc[i,c]
                #dictionary values are overwritten, so each correlation is there once
    corr_coefs = list(interesting_correlations.values())
    p_values = list(interesting_p_values.values())
    alpha = 0.05
    percentage_significant_corr = (sum(1 for value in p_values if value < alpha) / len(p_values)) * 100
    print(f"percentage significant correlation: {percentage_significant_corr:.2f}%")
    #calculate the mean, standard deviation and medianof all the correlations
    mean_corr_coef = np.nanmean(corr_coefs)
    std_corr_coef = np.nanstd(corr_coefs)
    median_corr_coef = np.nanmedian(corr_coefs)
    
    all_results = {'corrs': interesting_correlations, 'mean': mean_corr_coef, 'std': std_corr_coef, 'median': median_corr_coef, 'percentage significant corr': percentage_significant_corr}
    
    return all_results

def calc_correlations(merged_df, min_pairs=2, corr_method='pearson'):
    
    # Precompute interesting pairs
    column_names = merged_df.columns
    interesting_pairs = [
        (col1, col2)
        for col1 in column_names
        for col2 in column_names
        if col1 != col2 and col1[:-2] == col2[:-2]
    ]

    # Preprocess merged_df to drop non-finite rows
    clean_merged_df = merged_df.dropna(how='any').applymap(lambda x: x if np.isfinite(x) else np.nan).dropna()

    # Preallocate results
    interesting_correlations = {}
    interesting_p_values = {}

    for col1, col2 in interesting_pairs:
        # Drop rows with NaN in the pair of columns
        clean_pair = clean_merged_df[[col1, col2]].dropna()

        # Only compute if there are enough samples
        if len(clean_pair) >= min_pairs:
            # Calculate correlation coefficient and p-value
            if corr_method == 'pearson':
                corr_coef, p_value = pearsonr(clean_pair[col1].values, clean_pair[col2].values)
            elif corr_method == 'spearman':
                corr_coef, p_value = spearmanr(clean_pair[col1].values, clean_pair[col2].values)
            else:
                raise ValueError(f"Unsupported correlation method: {corr_method}")

            # Store results
            feature_name = col1[:-2]
            interesting_correlations[feature_name] = corr_coef
            interesting_p_values[feature_name] = p_value

    # Compute statistics
    corr_coefs = list(interesting_correlations.values())
    p_values = list(interesting_p_values.values())
    alpha = 0.05
    percentage_significant_corr = (sum(1 for p in p_values if p < alpha) / len(p_values)) * 100
    print(f"percentage significant correlation: {percentage_significant_corr:.1f}%")
    mean_corr_coef = np.nanmean(corr_coefs)
    std_corr_coef = np.nanstd(corr_coefs)
    median_corr_coef = np.nanmedian(corr_coefs)

    all_results = {
        'corrs': interesting_correlations,
        'mean': mean_corr_coef,
        'std': std_corr_coef,
        'median': median_corr_coef,
        'percentage significant corr': percentage_significant_corr
    }
    
    return all_results


def statistical_tests(df1, df2):
    """
    Shapiro-Wilk: TRUE if p-value smaller than statistic --> no normality
    T-test and Wilcoxon: TRUE if p-value smaller than alpha --> mean value significantly different

    Parameters
    ----------
    df1 : cleaned dataframe 1
    df2 : cleaned dataframe 2

    Returns
    -------
    percentage_significant: percentage for which the mean is significantly different 

    """
    #merge dataframe to allign slides 
    merged_df = pd.merge(df1, df2, left_index=True, right_index=True, suffixes=(' 1', ' 2'))
    
    #add all features to a list. The cleaned dataframes do not necesarily have the same features, 
    #because columns (features) with nan's are removed in the cleaning process. So we only take the ones that 
    #have a prefix. 
    prefixes = []
    for i in range(len(merged_df.columns)):
        name = merged_df.columns[i]
        if (name.endswith("1") or name.endswith("2")) and (name[:-2] not in prefixes):
            prefixes.append(name[:-2])
    
    alpha = 0.05
    shapiro_wilk_results = {}
    t_test_results = {}
    wilcoxon_signed_rank_results = {}
    
    for prefix in prefixes: #loop over prefix
        # Find columns with the current prefix
        matching_columns = [col for col in merged_df.columns if col[:-2] == prefix]
        col1 = matching_columns[0]
        col2 = matching_columns[1]
        
        shapiro_stat1, pvalue1 = shapiro(merged_df[col1])
        shapiro_stat2, pvalue2 = shapiro(merged_df[col2])
        t_stat, t_test_pvalue = ttest_rel(merged_df[col1], merged_df[col2])
        
        #for Wilcoxon check if there is a difference in columns, otherwise it won't work. 
        if not (merged_df[col1] - merged_df[col2]).any():
            print("The difference between", col1, "and", col2, "is zero for all elements.")
            wilcoxon_stat = 100 #number not used in further analysis
            wilcoxon_pvalue = 1 #set a high p-value, because exact same columns means no significant difference
        else:
            wilcoxon_stat, wilcoxon_pvalue = wilcoxon(merged_df[col1], merged_df[col2])
        
        #smaller than alpha means sifnificantly different or not normal
        shapiro_wilk_results[prefix] = [shapiro_stat1, pvalue1, pvalue1 < alpha, shapiro_stat2, pvalue2, pvalue2 < alpha]
        t_test_results[prefix] = [t_stat, t_test_pvalue, t_test_pvalue < alpha]
        wilcoxon_signed_rank_results[prefix] = [wilcoxon_stat, wilcoxon_pvalue, wilcoxon_pvalue < alpha]
    
    significant_count = 0
    for feature in prefixes:
        shapiro_result = shapiro_wilk_results[feature]
        #check if both columns of a feature are normally distributed. True means not normally distributed.
        #If both columns not normally distributed, do Wilcoxon signed rank test, otherwise paired sample T-test
        if shapiro_result[2]==True and shapiro_result[5]==True: 
            if wilcoxon_signed_rank_results[feature][-1]==True:
                significant_count += 1
        else:
            if t_test_results[feature][-1]==True:
                significant_count += 1
    #percentage of features for which the mean is significantly different
    percentage_significant = (significant_count/len(prefixes))*100
    
    return percentage_significant


def statistical_tests_v2(df1, df2):
    """
    Shapiro-Wilk: TRUE if p-value smaller than statistic --> no normality
    T-test and Wilcoxon: TRUE if p-value smaller than alpha --> mean value significantly different

    Parameters
    ----------
    df1 : cleaned dataframe 1
    df2 : cleaned dataframe 2

    Returns
    -------
    percentage_significant: percentage for which the mean is significantly different 

    """
    #merge dataframe to allign slides 
    merged_df = pd.merge(df1, df2, left_index=True, right_index=True, suffixes=(' 1', ' 2'))
    
    #add all features to a list. The cleaned dataframes do not necesarily have the same features, 
    #because columns (features) with nan's are removed in the cleaning process. So we only take the ones that 
    #have a prefix. 
    prefixes = []
    for i in range(len(merged_df.columns)):
        name = merged_df.columns[i]
        if (name.endswith("1") or name.endswith("2")) and (name[:-2] not in prefixes):
            prefixes.append(name[:-2])
    
    alpha = 0.05
    shapiro_wilk_results = {}
    t_test_results = {}
    wilcoxon_signed_rank_results = {}
    
    for prefix in prefixes: #loop over prefix
        # Find columns with the current prefix
        matching_columns = [col for col in merged_df.columns if col[:-2] == prefix]
        col1 = matching_columns[0]
        col2 = matching_columns[1]
        
        shapiro_stat1, pvalue1 = shapiro(merged_df[col1])
        shapiro_stat2, pvalue2 = shapiro(merged_df[col2])
        t_stat, t_test_pvalue = ttest_rel(merged_df[col1], merged_df[col2])
        wilcoxon_stat, wilcoxon_pvalue = wilcoxon(merged_df[col1], merged_df[col2])
        
        #smaller than alpha means sifnificantly different or not normal
        shapiro_wilk_results[prefix] = [shapiro_stat1, pvalue1, pvalue1 < alpha, shapiro_stat2, pvalue2, pvalue2 < alpha]
        t_test_results[prefix] = [t_stat, t_test_pvalue, t_test_pvalue < alpha]
        wilcoxon_signed_rank_results[prefix] = [wilcoxon_stat, wilcoxon_pvalue, wilcoxon_pvalue < alpha]
    
    significant_count = 0
    for feature in prefixes:
        shapiro_result = shapiro_wilk_results[feature]
        #check if both columns of a feature are normally distributed. True means not normally distributed.
        #If both columns not normally distributed, do Wilcoxon signed rank test, otherwise paired sample T-test
        if shapiro_result[2]==True and shapiro_result[5]==True: 
            if wilcoxon_signed_rank_results[feature][-1]==True:
                significant_count += 1
        else:
            if t_test_results[feature][-1]==True:
                significant_count += 1
    #percentage of features for which the mean is significantly different
    percentage_significant = (significant_count/len(prefixes))*100
    
    return percentage_significant

def plot_corr_boxplot(all_results, titles, corr_method='Pearson'):
    data = [all_results[0]['corrs'].values(), all_results[1]['corrs'].values(), all_results[2]['corrs'].values()]
    clean_data = []
    for l in data:
        clean_list = [x for x in l if not np.isnan(x)]
        clean_data.append(clean_list)
        
    plt.figure(figsize=(6,4))
    plt.boxplot(clean_data, positions=[1, 2, 3])
    plt.ylim(-1.1,1.15)
    plt.xticks([1, 2, 3], titles)
    plt.ylabel(f'{corr_method} correlation coefficient')
    plt.title(f'Boxplots with {corr_method} correlations between different types of tissue slides')
    for i in range(len(all_results)):
        x_pos = i+1
        y_pos = max(all_results[i]['corrs'].values())+0.03
        plt.text(x_pos, y_pos, f'{all_results[i]["median"]:.2f}', ha='center', va='bottom', color='black', fontsize=10)

def plot_corr_histogram(all_results, titles, corr_method='Pearson'):
    data = [all_results[0]['corrs'].values(), all_results[1]['corrs'].values(), all_results[2]['corrs'].values()]
    clean_data = []
    for l in data:
        clean_list = [x for x in l if not np.isnan(x)]
        clean_data.append(clean_list)
        
    #bin_edges = [i / 10.0 for i in range(11)]
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharey=True)
    #fig.suptitle(f'Histograms with {corr_method} correlations between different types of tissue slides', fontsize=20)
    
    # Set labels and title
    for i in range(len(all_results)):  
        sns.histplot(data[i], kde=True, ax=axes[i], legend=False)
        y_axis_top_limit = axes[i].get_ylim()[1]
        y_position = y_axis_top_limit - 3
    
        axes[i].set_title(titles[i], fontsize=20)
        axes[i].set_xlim(-1,1)
        axes[i].tick_params(axis='x', labelsize=14)
        axes[i].tick_params(axis='y', labelsize=14)
        axes[i].text(0.5, y_position, f'{all_results[i]["mean"]:.2f} $\pm$ {all_results[i]["std"]:.2f}', ha='center', va='bottom', color='black', fontsize=18)
        axes[i].text(-0.75,10, f'{all_results[i]["percentage significant corr"]:.0f}% significant', fontsize=18)
        if i == (len(all_results)-1):
            axes[i].set_xlabel(f'{corr_method} correlation coefficient', fontsize=18)
        axes[i].set_ylabel('Frequency', fontsize=18)
    plt.tight_layout()
