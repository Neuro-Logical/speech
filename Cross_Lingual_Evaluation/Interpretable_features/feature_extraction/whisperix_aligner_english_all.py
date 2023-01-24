#BASE = "/export/c12/afavaro/New_NLS/NLS_Speech_Data_All_16k/"
BASE = "/export/b15/afavaro/Trevor_paper/speech_16/"
OUT_PATH = '/export/b15/afavaro/Trevor_paper/Alignment/'
#OUT_PATH = "/export/c12/afavaro/New_NLS/NLS_Speech_Data_Word_Alignment_whisperx"

import sys
sys.path.append("/export/c07/afavaro/whisperX")
import whisperx
import os
import pandas as pd

device = "cpu"
model = whisperx.load_model("medium", device)
audios = [os.path.join(BASE, elem) for elem in os.listdir(BASE)]

files = []
for m in audios:
    size = os.stat(m).st_size / 1000
    if size > 56:
            files.append(m)

indx = files.index("/export/b15/afavaro/Trevor_paper/speech_16/AD_002_ses03_SecuencestroopPrevious2.wav")
for audio in files[indx:]:
    print(audio)
    text =[]
    time_stamps = []
    base_name = os.path.basename(audio).split(".wav")[0]
    print(base_name)
    result = model.transcribe(audio)
    model_a, metadata = whisperx.load_align_model(language_code='en', device=device)
    result_aligned = whisperx.align(result["segments"], model_a, metadata, audio, device)
    out = (result_aligned["word_segments"])
    for element in out:
        text.append(element['text'])
        time_stamps.append(element['start'])
    data = pd.DataFrame({'word': text, 'time_stamps': time_stamps})
    data.to_csv(os.path.join(OUT_PATH, base_name + ".csv"))