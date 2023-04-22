BASE = '/export/c12/afavaro/New_NLS/audio_fusion_new/all_audios'
OUT_PATH = '/export/c12/afavaro/New_NLS/audio_fusion_new/all_audio_aligner/'

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

#indx = files.index("/export/c12/afavaro/New_NLS/audio_fusion/joke_clean/NLS_089_ses01_Joke.wav")

for audio in files[-300:]:
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