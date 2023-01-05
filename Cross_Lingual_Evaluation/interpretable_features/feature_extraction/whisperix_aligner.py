BASE = "/export/b15/afavaro/Frontiers/Italian_PD/Audio_Whole_Italian/"
OUT_PATH = "/export/b15/afavaro/Frontiers/Italian_PD/alignment_whisperix"
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
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result_aligned = whisperx.align(result["segments"], model_a, metadata, audio_file, device)
    out = (result_aligned["word_segments"]) # after alignment
    for element in out:
        text.append(element['text'])
        time_stamps.append(element['start'])
    data = pd.DataFrame({'word': text, 'time_stamps': time_stamps})
    data.to_csv(os.path.join(OUT_PATH, base_name + ".csv"))