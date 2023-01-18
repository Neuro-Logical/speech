OUT_PATH = "/export/b16/afavaro/Alignment/"

import sys
sys.path.append("/export/c07/afavaro/whisperX")
import whisperx
import os
import pandas as pd

tot = "/export/b16/afavaro/AD_speech/"
tots =[elem.split(".wav")[0] for elem in os.listdir(tot)]
tot_1 = "/export/b16/afavaro/Alignment/"
tots_1 =[elem.split(".csv")[0] for elem in os.listdir(tot_1)]
files = (set(tots)^set(tots_1))
files = [os.path.join(tot, file + ".wav") for file in files]

device = "cpu"
model = whisperx.load_model("small", device)


files_new = []
for m in files:
    size = os.stat(m).st_size / 1000
    if size > 56:
            files_new.append(m)

for audio in files_new:
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