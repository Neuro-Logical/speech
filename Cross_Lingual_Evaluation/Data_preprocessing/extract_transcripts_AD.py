
#output_folder = '/export/b16/afavaro/Transcripts/'

import os
import whisper

tot = "/export/b16/afavaro/AD_speech/"
tots =[elem.split(".wav")[0] for elem in os.listdir(tot)]
tot_1 = "/export/b16/afavaro/Transcripts/"
tots_1 =[elem.split(".txt")[0] for elem in os.listdir(tot_1)]
files = (set(tots)^set(tots_1))
files = [os.path.join(tot, file + ".wav") for file in files]

files_new = []
for m in files:
    size = os.stat(m).st_size / 1000
    if size > 56:
        if "Cookie" in m:
            files_new.append(m)
print(len(files_new))
# extract and save transcripts in text files.
for i in files_new:
    print(i)
    model = whisper.load_model("medium")
    result = model.transcribe(i)
    test = result['text']
    base = os.path.basename(i).split(".wav")[0]
    total = os.path.join(output_folder, base + ".txt")
    text_file = open(total, "wt")
    n = text_file.write(test)
    text_file.close()
