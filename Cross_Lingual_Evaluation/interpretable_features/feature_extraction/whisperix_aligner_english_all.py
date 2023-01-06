BASE = "/export/c12/afavaro/New_NLS/NLS_Speech_Data_All_16k/"
OUT_PATH = "/export/c12/afavaro/New_NLS/NLS_Speech_Data_Word_Alignment_whisperx"

import sys
sys.path.append("/export/c07/afavaro/whisperX")
import whisperx
import os
import pandas as pd


device = "cpu"
model = whisperx.load_model("medium", device)

audios = [os.path.join(BASE, elem) for elem in os.listdir(BASE)]

for audio in audios:
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