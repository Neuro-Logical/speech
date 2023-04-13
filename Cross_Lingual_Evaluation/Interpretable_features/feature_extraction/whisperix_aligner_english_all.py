OUT_PATH = "/export/c12/afavaro/New_NLS/NLS_Speech_Data_Word_Alignment_whisperx"

import sys
sys.path.append("/export/c07/afavaro/whisperX")
import whisperx
import os
import pandas as pd

all_paths = '/export/c12/afavaro/New_NLS/NLS_Speech_Data_All_16k/'
transc_path = '/export/c12/afavaro/New_NLS/NLS_Speech_Data_Word_Alignment_whisperx/'

path_audiods = [os.path.join(all_paths, elem) for elem in os.listdir(all_paths)]
path_tr = [os.path.join(transc_path, elem) for elem in os.listdir(transc_path)]

names_tr = [os.path.basename(elem).split(".csv")[0] for elem in path_tr]
names_audio = [os.path.basename(elem).split(".wav")[0] for elem in path_audiods]

all_names = list(set(names_tr)^set(names_audio))
all_names_complete = [os.path.join(all_paths, elem + ".wav") for elem in all_names]
print("done")
device = "cpu"
model = whisperx.load_model("small", device)
#audios = [os.path.join(BASE, elem) for elem in os.listdir(BASE)]

files = []
for m in all_names_complete:
    size = os.stat(m).st_size / 1000
    if size > 56:

      #  if "Poem" in m:
       #     files.append(m)
       # if "Cookie" in m:
            files.append(m)
        if "Rainbow" in m:
            files.append(m)
        if "Word" in m:
            files.append(m)
        if "Sequence" in m:
            files.append(m)
        if "Joke" in m:
            files.append(m)

#indx = files.index("/export/c12/afavaro/New_NLS/NLS_Speech_Data_All_16k/PEC_019_ses01_SmoothSustained5.wav")

for audio in files:
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