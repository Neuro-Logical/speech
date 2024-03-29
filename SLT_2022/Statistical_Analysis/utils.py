from scipy import stats
import statsmodels
import numpy as np
from sklearn import metrics

# Functions used to perform:
#    '1 - Pair-wise Kruskal-Wallis H-tests.
#    '2 - FDR corrections.'
#    '3 - Compute the AUROC associated to each biomarker.
#    '4 - Compute the eta squared effect size.


def delete_multiple_element(list_object, indices):

    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)

    return list_object


def kruskal(output_path, biomarkers_name, c, p, c_name, p_name):

    """
     Function that performs pair-wise Kruskal-Wallis H-test from each pair of biomarkers/features.
    f: output path where to save the statistics.
    biomarkers_name: list of biomarkers' name.
    c: matrix of features from the control group.
    p: matrix of features from the parkinson's group.
    c_name: control group name (e.g., "CN")
    p_name: Parkinson's group name (e.g., "PD")

    Output: text file saving the p-value in correspondence to each pair-wise comparison between two distributions of a given biomarker.
    An example of text file generated using this function can be found in Statistical_Analysis/results_statistical_analysis.txt.

    """

    for i, title in enumerate(biomarkers_name):
        nome = title
        output_path.write(('\n' + f'kruskal results for {title} {c_name} {p_name} {stats.kruskal(c[i], p[i]).pvalue} \n\n'))


def compute_auc(array_1, array_2):

    """ Function that computes AUROC in each pair-wise comparison.
    The function takes as input the two arrays of biomarkers from the two experimental group under analysis (e.g., controls vs Parkinson's."""

    xs = np.concatenate([array_1, array_2], axis=1)
    y = np.concatenate([array_1.shape[1] * [2], array_2.shape[1] * [1]])

    for i, x in enumerate(xs):
        fpr, tpr, thresholds = metrics.roc_curve(y, x, pos_label=2)
        m = metrics.roc_auc_score(y, x)
        print(round(max(m, 1 - m), 2))


def compute_eta_squared(H, n_of_grp, n_of_observations):

    """ Function that computes the eta squared effect size.
    H: is the value of the Kruskal Wallis H-test.
    n_of_grp: is the number of experimental group considered.
    n_of_observations: is the total number of samples considered."""

    return (H - n_of_grp + 1) / (n_of_observations - n_of_grp)


def holm_correction(kruskal):

    """ Holm correction to apply after Kruskal wallis test.
    This function takes as input a text containing the results of the Kruskal-Wallis test (see function kruskal). """

    line_to_remove = []
    values = []
    corrected = []
    final = []
    for l in kruskal:
        if "nan" in l:
            line_to_remove.append(kruskal.index(l))

    new_krusk = delete_multiple_element(kruskal, line_to_remove)

    for line in new_krusk:
        ok = line.split('vs.')[1]
        num = ok.split(" ")[2]
        values.append(float(num))
    # values = [x for x in values if isnan(x) == False]
    result = statsmodels.stats.multitest.fdrcorrection(values, alpha=0.05, method='indep', is_sorted=False)
    num = np.where(result[0] == True)
    list_index = ((num)[0]).tolist()

    for i in list_index:
        corrected.append(result[1][i])
    for i in list_index:
        final.append(kruskal[i])

    return final, corrected



