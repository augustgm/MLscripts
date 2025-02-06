##############################################
# Helper Functions for Olink Proteomics data #
##############################################
from scipy.stats import mannwhitneyu, fisher_exact, chi2_contingency, anderson, shapiro, levene, ttest_ind
from statsmodels.stats.multitest import multipletests

import pandas as pd
from pandas import CategoricalDtype
from pandas.api.types import is_float, is_float_dtype, is_categorical_dtype, is_int64_dtype, \
    is_integer_dtype, is_string_dtype, is_numeric_dtype

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler



###############################
# Missing data and imputation #
###############################
def compute_missingness(df):    
    print(fr"Smoking status: {100 * (len(df.loc[df['Smoking.status.0.0'].isna()]) / len(df))}")  # 0.489%
    print(fr"Age: {100 * (len(df.loc[df['Age'].isna()])/ len(df))}")  # 0
    print(fr"Sex: {100 * (len(df.loc[df['Sex.0.0'].isna()]) / len(df))}")  # 0
    print(fr"BMI: {100 * (len(df.loc[df['BMI.0.0'].isna()]) / len(df))}")  # 0.476%
    print(fr"Packyears: {100 * (len(df.loc[df['Packyears.0.0'].isna()]) / len(df))}")  # 15.4%
    print(fr"Passive smoking: {100 * (len(df.loc[df['Tobacco.smoke.exposure.home.0.0'].isna()]) / len(df))}")  # 10.3%
    print(fr"Household income: {100 * (len(df.loc[df['Total.household.income.before.tax.0.0'].isna()]) / len(df))}")  # 15.4%
    print(fr"Education: {100 * (len(df.loc[df['Educational.attainment'].isna()]) / len(df))}")  # 1.72%
    print(fr"PM2.5: {100 * (len(df.loc[df['PM2.5.0.0'].isna()]) / len(df))}")  # 8.07%
    print(fr"gPCs: {100 * (len(df.loc[df['Genetic.principal.components.0.1'].isna()]) / len(df))}")  # 0.713%
    return None



###############################
# Statistics between 2 groups #
###############################
def anderson_darling_normality(vec, distribution="norm"):
    """
    Use Scipy.stats to perform anderson test. Determines if data is non-normal (True) or normal (False)
    """
    and_obj = anderson(vec, dist=distribution)
    sig_levels = getattr(and_obj, "significance_level")
    crit_vals = getattr(and_obj, "critical_values")
    non_normal = getattr(and_obj, "statistic") > crit_vals[list(sig_levels).index(5.0)]
    return non_normal


def levene_then_2sample_ttest(series1, series2, na_pol):
    """
    Compute the levene test of equal variance and then compute the independent (2-sample) t-test accordingly.
    :param series1: pandas series containing the values for group 1
    :param series2: pandas series containing the values for group 2
    :param na_pol: how ttest_ind()function should handle NAs in the data. In the levene test of equal variances, I
     have elected to drop NA values.
    :return: t-test object
    """
    # H0 = samples are from proportions with equal variances: p<0.05 = reject H0 = samples have unequal variances
    eq_var = getattr(levene(series1.dropna(), series2.dropna()), "pvalue") >= 0.05
    test_obj = ttest_ind(a=series1, b=series2, alternative="two-sided", nan_policy=na_pol, equal_var=eq_var)
    return test_obj


def compute_similarity_stats_from_2_dfs(df1, df2, categ_cols, na_pol):
    """
    Iterate through all columns of input data frames. If categorical, compute the Fisher's exact test. If continuous,
    assess normality and then compute Mann-Whitney U or T-test as necessary. My T-test function also assess for equal
    variance and then performs Welch's correction in the case of unequal variance.
    :param df1: first data frame object. Should be the entries where the binary grouping variable (e.g. LC status) is
    equal to the first of the two options (e.g. LC=diagnosed LC)
    :param df2: second data frame object. Should be the entries where the binary grouping variable (e.g. LC status) is
    equal to the second of the two options (e.g. LC=non-LC)
    :param categ_cols: list of columns that are categorical (must be the same between the two data frames). If data are
    numerical but should be interpreted as categorical (e.g. blood groups encoded as 0, 1, 2), then put that
    column name here
    :param na_pol: string, na-policy for tests that allow this
    :return: dataframe with results of similarity stats and FDR-adjusted p-values
    """
    # print("currently not checking for normality - applying non-parametric methods")
    out = {"variable": [], "test": [], "pval": []}
    for counter, des_col in enumerate(df1.columns):
        # print(f"iteration from compute_similarity_stats_from_2_dfs: {counter}")
        if des_col in categ_cols:
            print("dropping any NA values before performing fisher's exact test")
            conting_tab = pd.concat([df1[des_col].value_counts(), df2[des_col].value_counts()], axis=1)
            if conting_tab.shape[0] > 2:
                chi2_contingency(observed=conting_tab)
                out[f"test"].append("chi_square")
            else:
                test_obj = fisher_exact(conting_tab, alternative="two-sided")
                out[f"test"].append("fishers_exact")
            out["variable"].append(des_col)
            out[f"pval"].append(getattr(test_obj, "pvalue"))

        elif is_numeric_dtype(df1[des_col]) and is_numeric_dtype(df2[des_col]):
            vec1 = df1[des_col].dropna().values
            vec2 = df2[des_col].dropna().values

            if (len(vec1) < 3) or (len(vec2) < 3):
                print(f"skipping current column: {counter}={des_col} as not enough samples: vec1={len(vec1)}; vec2={len(vec2)}")
                continue

            non_norm_1 = (getattr(shapiro(vec1), "pvalue") < 0.05) if (len(vec1) < 5e3) else anderson_darling_normality(vec=vec1)
            non_norm_2 = (getattr(shapiro(vec2), "pvalue") < 0.05) if len(vec2) < 5e3 else anderson_darling_normality(vec=vec2)

            if non_norm_1 or non_norm_2:  # if either are non-parametric, perform mann-whitney U
                test_obj = mannwhitneyu(x=df1[des_col], y=df2[des_col], alternative="two-sided", nan_policy=na_pol)
                out[f"test"].append("mannwhitney_u")
            else:  # otherwise, perform t-test
                test_obj = levene_then_2sample_ttest(series1=df1[des_col], series2=df2[des_col], na_pol=na_pol)
                out[f"test"].append("t-test")

            out["variable"].append(des_col)
            out[f"pval"].append(getattr(test_obj, "pvalue"))
        else:
            raise Warning("No tests performed!!! - column in input dfs not \nresolved to numerical or categorical")

    out["padj"] = multipletests(pvals=out["pval"], alpha=0.05, method="fdr_bh")[1]
    return pd.DataFrame(out)


def olink_diff_npx(df, proteomics, na_pol, phenotype_oi_col):
    """

    :param df: UKBB data frame containing patient IDs only
    :param proteomics: Olink proteomics data in wide format containing only Patient_ID and protein markers
    :param na_pol: how to handle NaN (raise, omit)
    :param phenotype_oi_col: the column to use to split the dataset into 2 groups (e.g. LC and non-LC)
    :return: dataframe with results of similarity stats and FDR-adjusted p-values
    """
    # merge to gain access to the proteomics data
    prot_df = df.reset_index().merge(right=proteomics, on="Patient_ID", how="inner")

    # Split into 2 data frames by phenotype status
    prot_lc = prot_df.loc[prot_df[phenotype_oi_col] == 1].drop(columns=[phenotype_oi_col, "Patient_ID"])
    prot_nonlc = prot_df.loc[prot_df[phenotype_oi_col] == 0].drop(columns=[phenotype_oi_col, "Patient_ID"])

    # Compute statistics between the two groups
    return compute_similarity_stats_from_2_dfs(df1=prot_lc, df2=prot_nonlc, categ_cols=[], na_pol=na_pol)
