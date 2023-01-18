# base input directory containing recordings to transcribe.
base = '/export/c12/afavaro/New_NLS/NLS_Speech_Data_All_16k'
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
        if "NLS_126" in m:
            files.append(m)
        if "NLS_125" in m:
            files.append(m)

indx = files.index("/export/c12/afavaro/New_NLS/NLS_Speech_Data_All_16k/NLS_125_ses01_Namingaction9.wav")
print("CIAOAOO")
# extract and save transcripts in text files.
for i in files[indx+2:]:
    print(i)
    model = whisper.load_model("medium")
    result = model.transcribe(i)
    test = result['text']
    base = os.path.basename(i).split(".wav")[0]
    total = os.path.join(output_folder, base + ".txt")
    text_file = open(total, "wt")
    n = text_file.write(test)
    text_file.close()
