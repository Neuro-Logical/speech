import pandas as pd


def gita_prep(path_to_dataframe):

    """GITA data preprocessing.
       path_to_dataframe: path csv dataframe containing the features for classification, speaker ID and labels (i.e., HC vs PD).
       This function returns a pre-processed pandas data frame and the name of the columns in the dataframe. """

    spain = pd.read_csv(path_to_dataframe)
    spain = spain.dropna()
    spain = spain.iloc[:, 13:]
    spain['task'] = [elem.split("_")[2] for elem in spain['AudioFile'].tolist()]
    task = ['TDU']
    spain = spain[spain['task'].isin(task)]
    # spain = spain.drop(columns=['Unnamed: 0'])
    spain['labels'] = [elem.split("_")[0] for elem in spain['AudioFile'].tolist()]
    new_lab = []
    for lab in spain['labels'].tolist():
        if lab == "PD":
            new_lab.append(1)
        if lab == "CN":
            new_lab.append(0)
        if lab == "HC":
            new_lab.append(0)
    spain['labels'] = new_lab
    spain['names'] = [elem.split("_")[1] for elem in spain['AudioFile'].tolist()]

    return spain

def neurovoz_prep(path_to_dataframe):

    """Neurovoz data preprocessing.
    path_to_dataframe: path csv dataframe containing the features for classification, speaker ID and labels (i.e., HC vs PD).
    This function returns a pre-processed pandas data frame and the name of the columns in the dataframe. """

    spain = pd.read_csv(path_to_dataframe)
    spain = spain.drop(columns=['Unnamed: 0'])
    spain = spain.iloc[:, 13:]
    spain = spain.dropna()
    spain['labels'] = [elem.split("_")[0] for elem in spain['AudioFile'].tolist()]
    spain['task'] = [elem.split("_")[1] for elem in spain['AudioFile'].tolist()]
    spain['id'] = [elem.split("_")[2].split("-")[0] for elem in spain['AudioFile'].tolist()]
    task = ['concatenateread']
    spain = spain[spain['task'].isin(task)]
    spain = spain.drop(columns=['AudioFile', 'task'])
    lab = []
    for m in spain['labels'].tolist():
        if m == 'PD':
            lab.append(1)
        if m == 'HC':
            lab.append(0)
    spain['labels'] = lab

    return spain

def german_prep(path_to_dataframe):

    """GermanPD data preprocessing.
        path_to_dataframe: path csv dataframe containing the features for classification, speaker ID and labels (i.e., HC vs PD).
        This function returns a pre-processed pandas data frame and the name of the columns in the dataframe. """

    spain = pd.read_csv(path_to_dataframe)
    spain = spain.drop(columns=['Unnamed: 0'])
    spain = spain.iloc[:, 14:]
    spain['names'] = [elem.split("_")[1] for elem in spain['AudioFile'].tolist()]
    spain['labels'] = [elem.split("_")[0] for elem in spain['AudioFile'].tolist()]
    spain['task'] = [elem.split("_")[-2] for elem in spain['AudioFile'].tolist()]
    task = ['concatenateread']
    spain = spain[spain['task'].isin(task)]
    spain = spain.drop(columns=['AudioFile', 'task'])

    lab = []
    for m in spain['labels'].tolist():
        if m == "PD":
            lab.append(1)
        if m == 'CN':
            lab.append(0)
    spain['labels'] = lab
    spain = spain.dropna()

    return spain

def italian_prep(path_to_dataframe):

    """ItalianPVS data preprocessing.
        path_to_dataframe: path csv dataframe containing the features for classification, speaker ID and labels (i.e., HC vs PD).
        This function returns a pre-processed pandas data frame and the name of the columns in the dataframe. """

    italian = pd.read_csv(path_to_dataframe)
    italian['labels'] = [m.split("_")[0] for m in italian.AudioFile.tolist()]
    italian['task'] = [elem.split("_")[3][:2] for elem in italian['AudioFile'].tolist()]
    task = ['FB']
    italian = italian[italian['task'].isin(task)]
    lab = []
    for m in italian['labels']:
        if m == 'PD':
            lab.append(1)
        if m == 'CN':
            lab.append(0)
    italian['labels'] = lab
    names = [m.split("_", -1)[1] for m in italian.AudioFile.tolist()]
    surname = [m.split("_", -1)[2] for m in italian.AudioFile.tolist()]
    totale_names = []
    for i in zip(names, surname):
        totale_names.append(i[0] + i[1])
    italian['id'] = totale_names
    italian = italian.drop(columns=['Unnamed: 0', 'AudioFile', 'task'])

    return italian