OUT_PATH = '/export/b01/afavaro/tmeyer_new_output/tmeyer_alignment_all/'

import sys
sys.path.append("/export/c07/afavaro/whisperX")
import whisperx
import os
import pandas as pd
device = "cpu"
model = whisperx.load_model("small", device)

one = '/export/b01/afavaro/tmeyer_alignment/audio_all/'
files = [os.path.join(one, elem) for elem in os.listdir(one)]
#tots = [elem.split(".wav")[0] for elem in os.listdir(one)]
#two = '/export/c12/afavaro/New_NLS/NLS_Speech_Data_Word_Alignment_whisperx'
##tots_1 = [elem.split(".csv")[0] for elem in os.listdir(two)]

#tot_wc = []
#for m in tots:
#    if "SecuencestroopPrevious" in m:
#        tot_wc.append(m)
#    if "WordColor" in m:
#        tot_wc.append(m)
#
#files = (set(tots) ^ set(tot_wc))
#files = [os.path.join(one, file + ".wav") for file in files]
#

files_new = []
for m in files:
    size = os.stat(m).st_size / 1000
    if size > 56:
        if os.path.exists(m) == True:
            files_new.append(m)
print(len(files_new))

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