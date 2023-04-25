# base input directory containing recordings to transcribe.
base = '/export/c07/afavaro/KCL_PD_DATA/KCL_PD_Dataset/ReadText/all_audio_resampled/'
# output directory where to save speech transcripts.
output_folder = '/export/c07/afavaro/KCL_PD_DATA/KCL_PD_Dataset/ReadText/all_audios_transcripts/'

import os
import whisper
print("CIAOAOO")


path_audiods = [os.path.join(base, elem) for elem in os.listdir(base)]
path_tr = [os.path.join(output_folder, elem) for elem in os.listdir(output_folder)]
#
names_tr = [os.path.basename(elem).split(".txt")[0] for elem in path_tr]
names_audio = [os.path.basename(elem).split(".wav")[0] for elem in path_audiods]
#
all_names = list(set(names_tr) ^ set(names_audio))
all_names_complete = [os.path.join(base, elem + ".wav") for elem in all_names]


files = []
for m in all_names_complete:
    size = os.stat(m).st_size / 1000
    if size > 56:
       # if "CookieThief" in m:
            files.append(m)

      #  if "NLS_082_ses01_CookieThief" in m:

print(len(files))
print("done")
#indx = files.index("/export/c12/afavaro/New_NLS/audio_fusion_new/all_audios/PEC_028_ses01_CookieThief.wav")

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
