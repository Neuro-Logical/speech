BASE = '/export/c07/afavaro/KCL_PD_DATA/KCL_PD_Dataset/ReadText/all_audio_resampled/'
OUT_PATH = '/export/c07/afavaro/KCL_PD_DATA/KCL_PD_Dataset/ReadText/all_audios_aligner/'
#OUT_2 = '/export/c06/afavaro/aligner_3'

import sys
sys.path.append("/export/c07/afavaro/whisperX")
import whisperx
import os
import pandas as pd

device = "cpu"
model = whisperx.load_model("medium", device)

path_audiods = [os.path.join(BASE, elem) for elem in os.listdir(BASE)]

files = []
for m in path_audiods:
    size = os.stat(m).st_size / 1000
    if size > 56:
       # if "PEC_011_ses01_SecuencestroopPrevious2.wav" in m:
            files.append(m)
print(files)

     #   if "Poem" in m:
     #       files.append(m)
     #   if "Cookie" in m:
     #       files.append(m)
     #   if "Rainbow" in m:
     #       files.append(m)
     #   if "Word" in m:
     #       files.append(m)
     #   if "Secuences" in m:
     #       files.append(m)
     #   if "Joke" in m:
     #       files.append(m)

for audio in files:
    print(audio)
    text =[]
    time_stamps = []
    base_name = os.path.basename(audio).split(".wav")[0]
    print(base_name)
    result = model.transcribe(audio)
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result_aligned = whisperx.align(result["segments"], model_a, metadata, audio, device)
    out = (result_aligned["word_segments"])
    for element in out:
        text.append(element['text'])
        time_stamps.append(element['start'])
    data = pd.DataFrame({'word': text, 'time_stamps': time_stamps})
    data.to_csv(os.path.join(OUT_PATH, base_name + ".csv"))