import os
import sys
sys.path.append("/export/b15/afavaro/stable-ts")
from stable_whisper import modify_model
from stable_whisper import stabilize_timestamps
import pandas as pd
from stable_whisper import load_model
import torch

def uncertainty(text):
    """ Function design to capture the level of certainty of patients of participants when delivering  the description
    of the image"""

    cont_con = 0
    if "?" in text:
        cont_con = cont_con + 1
    if "why" in text:
        cont_con = cont_con + 1
    if "might" in text:
        cont_con = cont_con + 1
    if "could" in text:
        cont_con = cont_con + 1
    if "may" in text:
        cont_con = cont_con + 1
    if "uhm" in text:
        cont_con = cont_con + 1
    if "ah" in text:
        cont_con = cont_con + 1
    if "perhaps" in text:
        cont_con = cont_con + 1
    if "looks like" in text:
        cont_con = cont_con + 1

    return cont_con


def repetitions(text):
    """ Count the number of repetitions in each recording, after stop word removal """

    stopwords = list(stopwords.words('english'))
    repetition = 0
    text = text.split()
    d = dict()

    for line in text:
        line = line.strip()
        line = line.lower()
        words = line.split(" ")
        for word in words:

            if word in d:
                d[word] = d[word] + 1
            else:
                d[word] = 1

    for key in list(d.keys()):
        if key not in stopwords:
            if d[key] > 1:
                repetition += 1

    return repetition


def informational_verb(text):
    """ Compute the informativeness of the narratives by
    counting how many (if any) salient events (verbs) are mentioned. """

    cont_con = 0

    if "washing" in text:
        cont_con = cont_con + 1
    if "overflowing" in text:
        cont_con = cont_con + 1
    if "hanging" in text:
        cont_con = cont_con + 1
    if "trying to help" in text:
        cont_con = cont_con + 1
    if "falling" in text:
        cont_con = cont_con + 1
    if "wobbling" in text:
        cont_con = cont_con + 1
    if "drying" in text:
        cont_con = cont_con + 1
    if "ignoring" in text:
        cont_con = cont_con + 1
    if "reaching" in text:
        cont_con = cont_con + 1
    if "reaching up" in text:
        cont_con = cont_con + 1
    if "asking for cookie" in text:
        cont_con = cont_con + 1
    if "laughing" in text:
        cont_con = cont_con + 1
    if "standing" in text:
        cont_con = cont_con + 1

    return cont_con


def informational_content(text):

    """ Count how many (if any) salient object (nouns) are mentioned in the description"""

    cont_con = 0

    if "mother" in text:
        cont_con = cont_con + 1
    if "sister" in text:
        cont_con = cont_con + 1
    if "cookie" in text:
        cont_con = cont_con + 1
    if "cookie jar" in text:
        cont_con = cont_con + 1
    if "curtains" in text:
        cont_con = cont_con + 1
    if "cabinet" in text:
        cont_con = cont_con + 1
    if "brother" in text:
        cont_con = cont_con + 1
    if "chair" in text:
        cont_con = cont_con + 1
    if "kitchen" in text:
        cont_con = cont_con + 1
    if "sink" in text:
        cont_con = cont_con + 1
    if "garden" in text:
        cont_con = cont_con + 1
    if "fall" in text:
        cont_con = cont_con + 1
    if "dishes" in text:
        cont_con = cont_con + 1
    if "stool" in text:
        cont_con = cont_con + 1
    if "poddle" in text:
        cont_con = cont_con + 1
    if "shoes" in text:
        cont_con = cont_con + 1
    if "apron" in text:
        cont_con = cont_con + 1

    return cont_con


def ratio_info_rep_plus_uncert(df):

    """ Compute ratio between informativeness and uncertanty,
    where uncertainty is represented as repetition + uncertainty"""

    summation = df['repetition'] + df["uncertanty"]
    ratio = df['informational'] / summation  # info / rep + uncertanty
    df["ratio_info_rep_plus_uncert"] = ratio

    return df


def ratio_rep_certanty(df):

    """ Compute the ratio between repetitions and uncertainty """

    division = df['repetition'] / df["uncertanty"]  # repetition / uncertainty
    df["ratio_rep_certanty"] = division

    return df


def extract_word_timestamp(path_recordings, output_folder):

    """  Code to extract word starting timestamps using Whisper.
    This code uses an existing script that modifies methods of Whisper's model to gain
    access to the predicted timestamp tokens of each word (token) without needing additional inference.
     It also stabilizes the timestamps down to the word (token) level to ensure chronology.
     Additionally, it can suppress gaps in speech for more accurate timestamps.
     Original code can be found at: https://github.com/jianfch/stable-ts. """

    paths = [os.path.join(path_recordings, base) for base in os.listdir(path_recordings)]

    # remove empty recordings
    files = []
    for m in paths:
        size = os.stat(m).st_size / 1000
        if size > 56:
            files.append(m)

    model = load_model('medium')
    modify_model(model)
    for recording in files:
        whole_tokens = []
        whole_time_stamps = []
        base_name = recording.split('/')[-1].split(".wav")[0]
        print(base_name)
        with torch.no_grad():
            results = model.transcribe(recording)
        stab_segments = stabilize_timestamps(results, top_focus=True)
        for i in range(len(stab_segments)):
            chunk = (stab_segments[i]['whole_word_timestamps'])
            for index in range(len(chunk)):
                whole_tokens.append(chunk[index]['word'])
                whole_time_stamps.append(chunk[index]['timestamp'])
                dict = {'token': whole_tokens, 'time_stamp': whole_time_stamps}
                df = pd.DataFrame(dict)
                df.to_csv(f"{output_folder}/{base_name}.csv")
    #





