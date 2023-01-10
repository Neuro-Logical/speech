import numpy as np
import statsmodels.api
from scipy.stats import spearmanr
import pandas as pd

def normalize(dataframe, columns):

    # select only columns containing the values of the features.
    feats = dataframe.iloc[:, -2:-1]
    # select only columns containing the info about the task/subject (i.e., speaker ID, UPDRS, task)
    info_subject = new.iloc[:, :3]
    df_z_scaled = feats.copy()

    # apply normalization techniques
    for column in df_z_scaled.columns:
        df_z_scaled[column] = (df_z_scaled[column] -
                               df_z_scaled[column].mean()) / df_z_scaled[column].std()

    normalized = df_z_scaled
    norm_data = pd.concat([normalized, info_subject], axis=1)

    return norm_data


def compute_correlation(data_frame, num_cols, clinical_score):

    """ Compute correlation between biomarker values and UPDRSIII/UPDRSIII.I scores/H&Y scale.
    After computing spearman correlation with corresponding p-value, FDR correction is applied.

    dataframe: pandas dataframe where the columns represent the normalized features,
    each row corresponds to a different subject and a single column contains the clinical score
    for each of the subject.
    num_cols (int): number of columns containing (starting from the beginning) containing the values of the features only.
    clinical_score (string): "updrs", "hoen" "updrs_speech" """

    feats_col = data_frame.iloc[:, :num_cols]  # select only columns containing features
    print(feats_col)
    updrs_pd = data_frame[clinical_score].tolist()
    feats = feats_col.columns.values.tolist()
    file = []
    p_values = []

    for fea in feats:
        data = feats_col[fea].tolist()
        corr, _ = spearmanr(data, updrs_pd)
        p_values.append(_)
        file.append((f'Spearm correlation for feats {fea}: p_value {_} and correlation coeff is {corr}'))
    print(file)
    # Apply correction
    res = statsmodels.stats.multitest.fdrcorrection(p_values, alpha=0.05, method='indep', is_sorted=False)
    ows = np.where(res[1][:, ] < 0.05)
    l = list(ows[0])
    values = res[1][l]
    for m in zip(l, values):
        print(m, feats[m[0]])

