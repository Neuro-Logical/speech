import os
import sys
sys.path.append("/export/b15/afavaro/stable-ts")
from stable_whisper import modify_model
from stable_whisper import stabilize_timestamps
import pandas as pd
from stable_whisper import load_model
import torch


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

#########################################################################################################

def extract_word_timestamp(path_recordings, output_folder):
    """
    Code to extract word starting timestamps using Whisper.
    This code uses an existing script that modifies methods of Whisper's model to gain
    access to the predicted timestamp tokens of each word (token) without needing additional inference.
     It also stabilizes the timestamps down to the word (token) level to ensure chronology.
     Additionally, it can suppress gaps in speech for more accurate timestamps.
     Original code can be found at: https://github.com/jianfch/stable-ts.

    """
    paths = [os.path.join(path_recordings, base) for base in os.listdir(path_recordings)]

    #remove empty recordings
    files = []
    for m in paths:
        size = os.stat(m).st_size/1000
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

