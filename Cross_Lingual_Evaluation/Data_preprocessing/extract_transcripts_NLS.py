# base input directory containing recordings to transcribe.
base = '/export/c12/afavaro/New_NLS/NLS_Speech_Data/AD_014/speech/'
# output directory where to save speech transcripts.
output_folder = '/export/c12/afavaro/New_NLS/NLS_Speech_Data_Transcripts/'

import os
import whisper
print("CIAOAOO")

paths = [os.path.join(base, elem) for elem in os.listdir(base)]
files = []
for m in paths:
    size = os.stat(m).st_size / 1000
    if size > 56:
        if "CookieThief" in m:
            files.append(m)

      #  if "NLS_082_ses01_CookieThief" in m:


print("done")
#indx = files.index("/export/c12/afavaro/New_NLS/NLS_Speech_Data_All_16k/PEC_063_ses01_CookieThief.wav")

# extract and save transcripts in text files.
for i in files:
    print(i)
    model = whisper.load_model("medium")
    result = model.transcribe(i)
    test = result['text']
    base = os.path.basename(i).split(".wav")[0]
    total = os.path.join(output_folder, base + ".txt")
    text_file = open(total, "wt")
    n = text_file.write(test)
    text_file.close()
