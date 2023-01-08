import pandas as pd

def gita_prep(path_to_dataframe):

    """ GITA data preprocessing.
    path_to_dataframe: path csv dataframe containing the features for classification, speaker ID and labels (i.e., HC vs PD).
    This function returns a pre-processed pandas data frame and the name of the columns in the dataframe. """

    #path_to_dataframe = "/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/GITA/total_data_frame_novel_task_combined_ling_tot.csv"
    colombian = pd.read_csv(path_to_dataframe)
    colombian = colombian.dropna()
    colombian['names'] = [elem.split("_")[1] for elem in colombian.AudioFile.tolist()]
    colombian['labels'] = [elem.split("_")[0] for elem in colombian.AudioFile.tolist()]
    colombian['task'] = [elem.split("_")[2] for elem in colombian['AudioFile'].tolist()]
    task = ['TDU']
    colombian = colombian[colombian['task'].isin(task)]
    colombian = colombian.drop(columns=['Unnamed: 0'])
    new_lab = []
    for lab in colombian['labels'].tolist():
        if lab == "PD":
            new_lab.append(1)
        if lab == "CN":
            new_lab.append(0)
        if lab == "HC":
            new_lab.append(0)
    colombian['labels'] = new_lab
    colombian = colombian.rename(columns={'names': 'id'})
    # colombian['id'] = [elem.split("_")[1] for elem in colombian['AudioFile'].tolist()]
    colombian_cols = colombian.columns.tolist()

    return colombian, colombian_cols

def neurovoz_prep(path_to_dataframe):

    """ Neurovoz data preprocessing.
      path_to_dataframe: path csv dataframe containing the features for classification, speaker ID and labels (i.e., HC vs PD).
      This function returns a pre-processed pandas data frame and the name of the columns in the dataframe. """

    #path_to_dataframe = "/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/NEUROVOZ/tot_data_experiments.csv"
    spain = pd.read_csv(path_to_dataframe)
    spain = spain.dropna()
    spain['labels'] = [elem.split("_")[0] for elem in spain['AudioFile'].tolist()]
    spain['task'] = [elem.split("_")[1] for elem in spain['AudioFile'].tolist()]
    spain['id'] = [elem.split("_")[2].split("-")[0] for elem in spain['AudioFile'].tolist()]
    task = ['concatenateread']
    spain = spain[spain['task'].isin(task)]
    spain = spain.drop(columns=['Unnamed: 0', 'AudioFile', 'task'])
    lab = []
    for m in spain['labels'].tolist():
        if m == 'PD':
            lab.append(1)
        if m == 'HC':
            lab.append(0)
    spain['labels'] = lab
    spain_cols = spain.columns.tolist()

    return spain, spain_cols


def german_prep(path_to_dataframe):

    """ GermanPD data preprocessing.
    path_to_dataframe: path csv dataframe containing the features for classification, speaker ID and labels (i.e., HC vs PD).
    This function returns a pre-processed pandas data frame and the name of the columns in the dataframe. """

    #path_to_dataframe = "/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/GERMAN/final_data_frame_with_intensity.csv"
    german = pd.read_csv(path_to_dataframe)
    german = german.drop(columns=['Unnamed: 0'])
    german['names'] = [elem.split("_")[1] for elem in german['AudioFile'].tolist()]
    german['labels'] = [elem.split("_")[0] for elem in german['AudioFile'].tolist()]
    german['task'] = [elem.split("_")[-2] for elem in german['AudioFile'].tolist()]
    task = ['concatenateread']
    german = german[german['task'].isin(task)]
    german = german.drop(columns=['AudioFile', 'task'])
    lab = []
    for m in german['labels'].tolist():
        if m == "PD":
            lab.append(1)
        if m == 'CN':
            lab.append(0)
    german['labels'] = lab
    german = german.dropna()
    german = german.rename(columns={"names": "id"})
    german_cols = german.columns.tolist()
    return german, german_cols


def italian_prep(path_to_dataframe):

    """ ItalianPVS data preprocessing.
     path_to_dataframe: path csv dataframe containing the features for classification, speaker ID and labels (i.e., HC vs PD).
     This function returns a pre-processed pandas data frame and the name of the columns in the dataframe. """

    #path_to_dataframe = "/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/ITALIAN_PD/tot_experiments_ling_fin.csv"
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
    italian_cols = italian.columns.tolist()

    return italian, italian_cols