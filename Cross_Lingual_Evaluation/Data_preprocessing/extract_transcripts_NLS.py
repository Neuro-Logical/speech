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
        if "AD_022" in m:
            files.append(m)
        if "AD_023" in m:
            files.append(m)
        if "AD_024" in m:
            files.append(m)
        if "AD_025" in m:
            files.append(m)

indx = files.index("/export/c12/afavaro/New_NLS/NLS_Speech_Data_All_16k/AD_022_ses01_SmoothSustained5.wav")
#print(indx)
# extract and save transcripts in text files.
#for i in files[indx+2:]:
for i in files[indx:]:
    print(i)
    model = whisper.load_model("medium")
    result = model.transcribe(i)
    test = result['text']
    base = os.path.basename(i).split(".wav")[0]
    total = os.path.join(output_folder, base + ".txt")
    text_file = open(total, "wt")
    n = text_file.write(test)
    text_file.close()
