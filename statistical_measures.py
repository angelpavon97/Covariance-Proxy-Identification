
from scipy.stats import ttest_ind, chi2_contingency, entropy, kruskal, mannwhitneyu
from sklearn import metrics
import pandas as pd
import numpy as np

def get_entropy(s):
    return entropy(s.value_counts(normalize=True))

def get_mi(df, class_name = 'Class'):
    
    mis = {}
    
    for col_name in df.select_dtypes(object).columns:
        mis[col_name] = metrics.mutual_info_score(df[class_name], df[col_name])

    return {k: v for k, v in sorted(mis.items(), key=lambda item: item[1], reverse=True)}

def get_gr(df, class_name = 'Class'):
    
    grs = {}
    
    for col_name in df.select_dtypes(object).columns:
        mi = metrics.mutual_info_score(df[class_name], df[col_name])
        grs[col_name] = mi/get_entropy(df[col_name])

    return {k: v for k, v in sorted(grs.items(), key=lambda item: item[1], reverse=True)}

def get_suc(df, class_name = 'Class'):
    
    sucs = {}
    
    for col_name in df.select_dtypes(object).columns:
        mi = metrics.mutual_info_score(df[class_name], df[col_name])
        sucs[col_name] = 2 * (mi/(get_entropy(df[col_name]) + get_entropy(df[class_name])))

    return {k: v for k, v in sorted(sucs.items(), key=lambda item: item[1], reverse=True)}

def check_chi2_assumptions(contigency, col_name):
    # No more than 20% of the expected counts are less than 5
    n_true = contigency[contigency >= 5].count().sum()
    n_false = contigency[contigency < 5].count().sum()
    
    if n_false/(n_true + n_false) > 0.2:
        print(f'WARNING: Assumption is not met in {col_name} contigency table as more than 20% of expected counts are less than 5.')
        
    # all individual expected counts are 1 or greater
    if contigency.equals(contigency[contigency >= 1]) == False:
        print(f'WARNING: Assumption is not met in {col_name} contigency table as some expected counts are less than 1.')

def get_chi2(df, class_name = 'Class', alpha = 0.05):
    dependent_attributes = {}
    independent_attributes = {}
    
    for col_name in df.select_dtypes(object).columns:
        contigency = pd.crosstab(df[class_name], df[col_name])

        check_chi2_assumptions(contigency, col_name)

        chi2, p_value, dof, expected = chi2_contingency(contigency)

        if p_value < alpha:
            dependent_attributes[col_name] = p_value
        else:
            independent_attributes[col_name] = p_value

    d_sorted = {k: v for k, v in sorted(dependent_attributes.items(), key=lambda item: item[1], reverse=False)}
    i_sorted = {k: v for k, v in sorted(independent_attributes.items(), key=lambda item: item[1], reverse=False)}
    
    return d_sorted, i_sorted

def get_mannwhitneyu(df, class_name = 'Class', alpha = 0.05):
    
    diff_dis_attributes = {}
    same_dis_attributes = {}
    
    # Separate both populations (ex: if class_name = gender, df1 will be for females and df2 for males)
    class_values = list(set(df[class_name].values))
    df1 = df[df[class_name] == class_values[0]]
    df2 = df[df[class_name] != class_values[0]]
    
    for col_name in df.select_dtypes([np.number]).columns:
        t_statistic, p_value = mannwhitneyu(df1[col_name], df2[col_name])

        if p_value < alpha:
            diff_dis_attributes[col_name] = p_value
        else:
            same_dis_attributes[col_name] = p_value
            
    d_sorted = {k: v for k, v in sorted(diff_dis_attributes.items(), key=lambda item: item[1], reverse=False)}
    s_sorted = {k: v for k, v in sorted(same_dis_attributes.items(), key=lambda item: item[1], reverse=False)}
    
    return d_sorted, s_sorted