OUT_PATH = '/data/lmorove1/afavaro/data/others/trevor_alignment/'


import sys
sys.path.append("/export/c07/afavaro/whisperX")
import whisperx
import os
import pandas as pd
device = "cuda"
model = whisperx.load_model("small", device)


one = '/data/lmorove1/afavaro/data/others/trevor_audios/'
files = [os.path.join(one, elem) for elem in os.listdir(one)]

files_new = []
for m in files:
    size = os.stat(m).st_size / 1000
    if size > 70:
        if os.path.exists(m) == True:
            files_new.append(m)
#print(len(files_new))

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