import numpy as np
from scipy.stats import levene, ttest_ind, shapiro, mannwhitneyu, f_oneway
from statsmodels.sandbox.stats.multicomp import multipletests
import itertools
import pandas as pd
from config.settings import STATISTICAL_CONFIG

from pyspark.sql.functions import collect_list




def convert_spark_data_to_dict(df, group_col, value_col):
    """
    Convert Spark DataFrame to dictionary for statistical testing.
    
    Args:
        df (DataFrame): Spark DataFrame
        group_col (str): Column to group by
        value_col (str): Column with values to analyze
        
    Returns:
        dict: Dictionary with group as key and list of values as value
    """
    condensed = df.groupBy(group_col).agg(collect_list(value_col).alias("values"))
    data_for_tests = condensed.collect()
    data_dict = {row[group_col]: row['values'] for row in data_for_tests}
    return data_dict

def check_normality(data_dict, test_type="shapiro"):
    """
    Check normality for multiple groups using Shapiro-Wilk test.
    
    Args:
        data_dict (dict): Dictionary with group names as keys and data as values
        test_type (str): Type of normality test ("shapiro" or "kstest")
        
    Returns:
        dict: Normality test results for each group
    """
    print("\n--- Normality Check (Shapiro-Wilk Test) ---")
    results = {}
    
    for group_name, data in data_dict.items():
            
        try:
            stat, p_value = shapiro(data)
                
            is_normal = p_value > STATISTICAL_CONFIG["significance_level"]
            print(f"Group {group_name}: P-value={p_value:.3f} -> {'Normal' if is_normal else 'Not Normal'}")
            
            results[group_name] = {
                "statistic": stat,
                "p_value": p_value,
                "is_normal": is_normal
            }
        except Exception as e:
            print(f"Group {group_name}: Error in normality test - {e}")
            results[group_name] = {"error": str(e)}
    
    return results

def check_homogeneity_of_variances(data_dict):
    """
    Check homogeneity of variances using Levene's test.
    
    Args:
        data_dict (dict): Dictionary with group names as keys and data as values
        
    Returns:
        dict: Variance homogeneity test results
    """
    print("\n--- Homogeneity of Variances Check (Levene's Test) ---")
    
    # Filter out groups with less than 2 samples
    samples_for_levene = [np.array(data) for data in data_dict.values() if len(data) >= 2]
    
    if len(samples_for_levene) < 2:
        print("Not enough groups with sufficient data to perform Levene's test (need at least 2 groups with N >= 2).")
        return {"error": "Insufficient groups"}
    
    try:
        stat, p_value = levene(*samples_for_levene)
        variances_equal = p_value > STATISTICAL_CONFIG["significance_level"]
        print(f"Levene's Test: P-value={p_value:.3f} -> {'Variances Equal' if variances_equal else 'Variances Not Equal'}")
        
        return {
            "statistic": stat,
            "p_value": p_value,
            "variances_equal": variances_equal
        }
    except Exception as e:
        print(f"Error in Levene's test: {e}")
        return {"error": str(e)}

def perform_anova_analysis(data_dict):
    """
    Perform one-way ANOVA test.
    
    Args:
        data_dict (dict): Dictionary with group names as keys and data as values
        
    Returns:
        dict: ANOVA test results
    """
    print("\n--- ANOVA Analysis ---")
    
    data_list = [data for data in data_dict.values() if len(data) >= 2]
    
    if len(data_list) < 2:
        print("Need at least 2 groups with sufficient data for ANOVA.")
        return {"error": "Insufficient groups"}
    
    try:
        f_statistic, p_value = f_oneway(*data_list)
        significant = p_value < STATISTICAL_CONFIG["significance_level"]
        
        print(f"ANOVA: F-statistic={f_statistic:.4f}, P-value={p_value:.3f}")
        if significant:
            print("There is a statistically significant difference between the groups")
        else:
            print("There is no statistically significant difference between the groups")
        
        return {
            "f_statistic": f_statistic,
            "p_value": p_value,
            "significant": significant
        }
    except Exception as e:
        print(f"Error in ANOVA: {e}")
        return {"error": str(e)}

def pairwise_mannwhitneyu_tests(data_dict, apply_correction=True):
    """
    Perform pairwise Mann-Whitney U tests with optional Bonferroni correction.
    
    Args:
        data_dict (dict): Dictionary with group names as keys and data as values
        apply_correction (bool): Whether to apply Bonferroni correction
        
    Returns:
        DataFrame: Results of pairwise comparisons
    """
    print(f"\n--- Pairwise Mann-Whitney U Tests (Bonferroni correction: {apply_correction}) ---")
    
    group_names = list(data_dict.keys())
    pairs = list(itertools.combinations(group_names, 2))
    
    results = []
    raw_p_values = []
    
    for group1, group2 in pairs:
        data1 = data_dict[group1]
        data2 = data_dict[group2]
        
        if len(data1) < 2 or len(data2) < 2:
            continue
            
        try:
            stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
            
            results.append({
                'group1': group1,
                'group2': group2,
                'median1': np.median(data1),
                'median2': np.median(data2),
                'statistic': stat,
                'p_value_raw': p_value
            })
            raw_p_values.append(p_value)
            
        except Exception as e:
            print(f"Error comparing {group1} vs {group2}: {e}")
    
    if not results:
        print("No valid pairwise comparisons could be performed.")
        return pd.DataFrame()
    
    results_df = pd.DataFrame(results)
    
    # Apply Bonferroni correction if requested
    if apply_correction and len(raw_p_values) > 0:
        reject, corrected_p_values, _, _ = multipletests(raw_p_values, method='bonferroni')
        results_df['p_value_corrected'] = corrected_p_values
        results_df['significant_after_correction'] = reject
    
    return results_df

def perform_two_sample_ttest(group1_data, group2_data, group1_name="Group 1", group2_name="Group 2", 
                           alternative='two-sided'):
    """
    Perform two-sample t-test.
    
    Args:
        group1_data (list): Data for first group
        group2_data (list): Data for second group
        group1_name (str): Name of first group
        group2_name (str): Name of second group
        alternative (str): Alternative hypothesis ('two-sided', 'less', 'greater')
        
    Returns:
        dict: T-test results
    """
    print(f"\n--- Two-Sample T-Test: {group1_name} vs {group2_name} ---")
    
    # Basic statistics
    mean1, mean2 = np.mean(group1_data), np.mean(group2_data)
    n1, n2 = len(group1_data), len(group2_data)
    
    print(f"{group1_name}: Mean={mean1:.2f}, N={n1}")
    print(f"{group2_name}: Mean={mean2:.2f}, N={n2}")
    
    # Check for equal variances first
    try:
        levene_stat, levene_p = levene(group1_data, group2_data)
        equal_var = levene_p > 0.05
        print(f"Levene's test for equal variances: p={levene_p:.3f} ({'Equal' if equal_var else 'Unequal'} variances)")
    except:
        equal_var = True
        print("Could not perform Levene's test, assuming equal variances")
    
    # Perform t-test
    try:
        t_statistic, p_value = ttest_ind(group1_data, group2_data, equal_var=equal_var, alternative=alternative)
        
        significant = p_value < STATISTICAL_CONFIG["significance_level"]
        
        print(f"T-test: t-statistic={t_statistic:.4f}, p-value={p_value:.4f}")
        if significant:
            print("Reject the null hypothesis - statistically significant difference")
        else:
            print("Fail to reject the null hypothesis - no significant difference")
        
        return {
            "group1_name": group1_name,
            "group2_name": group2_name,
            "mean1": mean1,
            "mean2": mean2,
            "n1": n1,
            "n2": n2,
            "t_statistic": t_statistic,
            "p_value": p_value,
            "significant": significant,
            "equal_variances": equal_var
        }
        
    except Exception as e:
        print(f"Error in t-test: {e}")
        return {"error": str(e)}