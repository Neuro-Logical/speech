# base input directory containing recordings to transcribe.
base = '/export/b16/afavaro/PARKCELEB/experiments_english3/longitudinal_study/data/before/10_years/'
# output directory where to save speech transcripts.
output_folder = '/export/b16/afavaro/PARKCELEB/experiments_english3/longitudinal_study/transcripts/10_before/'

import os
import whisper

path_audios = [os.path.join(base, elem) for elem in os.listdir(base)]

print(len(path_audios))
# extract and save transcripts in text files.
for num, i in enumerate(path_audios):
    print(num)
    print(i)
    model = whisper.load_model("base")
    result = model.transcribe(i)
    test = result['text']
    base = os.path.basename(i).split(".wav")[0]
    total = os.path.join(output_folder, base + ".txt")
    text_file = open(total, "wt")
    n = text_file.write(test)
    text_file.close()
