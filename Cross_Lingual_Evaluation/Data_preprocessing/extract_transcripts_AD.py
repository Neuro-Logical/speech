
#output_folder = '/export/b16/afavaro/Transcripts/'

import os
import whisper
base = '/export/c07/afavaro/KCL_PD_DATA/KCL_PD_Dataset/ReadText/all_audio_resampled/'
# output directory where to save speech transcripts.
output_folder = '/export/c07/afavaro/KCL_PD_DATA/KCL_PD_Dataset/ReadText/all_audios_transcripts/'
path_audiods = [os.path.join(base, elem) for elem in os.listdir(base)]

files_new = []
for m in path_audiods:
    size = os.stat(m).st_size / 1000
    if size > 56:
        files_new.append(m)
print(len(files_new))

# extract and save transcripts in text files.
for i in files_new[-20]:
    print(i)
    model = whisper.load_model("medium")
    result = model.transcribe(i)
    test = result['text']
    base = os.path.basename(i).split(".wav")[0]
    total = os.path.join(output_folder, base + ".txt")
    text_file = open(total, "wt")
    n = text_file.write(test)
    text_file.close()
