import sys
sys.path.append("/export/c07/afavaro/whisperX")
import whisperx
import os
import pandas as pd


def informational_verb(text):

    """ Compute the informativeness of the narratives by counting how many (if any) salient events (verbs) are mentioned.
    text: string containing speech transcripts."""

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


def extract_word_starting_timestamps(BASE, OUT_PATH):

    """  Code to extract word starting timestamps using whisperx.
     Whisperx is a whisper-Based Automatic Speech Recognition (ASR) with improved timestamp accuracy using forced alignment.
     Code can be found at:  https://github.com/m-bain/whisperX.
     BASE: path to the folder where recordings are stored.
     OUTPATH: path to the folder where the file containing word starting time stamps will be stored.
     This function outputs for each recording a csv file with two columns: the first containing the list of words and the second the corresponding starting point. """

    device = "cpu"
    model = whisperx.load_model("medium", device)
    audios = [os.path.join(BASE, elem) for elem in os.listdir(BASE)]

    for audio in audios:
        text =[]
        time_stamps = []
        base_name = os.path.basename(audio).split("wav")[0]
        result = model.transcribe(audio)
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result_aligned = whisperx.align(result["segments"], model_a, metadata, audio, device)
        out = (result_aligned["word_segments"]) # after alignment
        for element in out:
            text.append(element['text'])
            time_stamps.append(element['start'])
        data = pd.DataFrame({'word': text, 'time_stamps': time_stamps})
        data.to_csv(os.path.join(OUT_PATH, base_name + ".csv"))

