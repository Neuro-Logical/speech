
from scipy import stats
from sklearn import metrics
import numpy as np
import statsmodels


def delete_multiple_element(list_object, indices):

    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)

    return list_object


def kruskal(f, biomarker, c, p, c_name, p_name):
    '''Function perform Kruskal-Wallis t-tests. '''
    for i, title in enumerate(biomarker):
        nome = title
        f.write(('\n' + f'kruskal results for {title} {c_name} {p_name} {stats.kruskal(c[i], p[i]).pvalue} \n\n'))


def compute_auc(array_1, array_2):

    ''' Compute AUROC for each pair of biomarkes.'''

    xs = np.concatenate([array_1, array_2], axis=1)
    y = np.concatenate([array_1.shape[1] * [2], array_2.shape[1] * [1]])

    for i, x in enumerate(xs):
        fpr, tpr, thresholds = metrics.roc_curve(y, x, pos_label=2)
        # print(i, metrics.auc(fpr, tpr))
        m = metrics.roc_auc_score(y, x)
        print(round(max(m, 1 - m), 2))


def compute_eta_squared(H, n_of_grp, n_of_observ):

    '''Compute eta squared effect size.'''

    return (H - n_of_grp + 1) / (n_of_observ - n_of_grp)


def holm_correction(kruskal):

    ''''Holm correction to apply after Kruskal-Wallis t-test. '''

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


def read_stats_test(file):

    ''' Read the statistics from .txt files. '''

    with open(file, 'r') as f:
        lista = []
        testo = f.readlines()
        testo = [line.strip("\n") for line in testo]

        for line in testo:
            if line == "":
                pass
            else:
                lista.append(line)

    return lista



def compute_best_scores(lista):

    ''' Extract only p-values < 0.0.5 from saved statistics. '''

    values = []
    critical = []
    final = []

    for l in lista:
        ok = l.split('vs.')[1]
        num = ok.split(" ")[2]
        values.append(num)

    for value in values:
        if float(value) < 0.05:
            critical.append(value)

    for li in lista:
        for cri in critical:
            if cri in li:
                final.append(li)

    return final