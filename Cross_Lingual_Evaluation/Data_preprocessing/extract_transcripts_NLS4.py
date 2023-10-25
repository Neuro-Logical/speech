# base input directory containing recordings to transcribe.
base = '/export/b16/afavaro/PARKCELEB/experiments_english3/longitudinal_study/data/NEW_PD/before_10/'
# output directory where to save speech transcripts.
output_folder = '/export/b16/afavaro/PARKCELEB/experiments_english3/longitudinal_study/transcripts/10_before/'

import os
import whisper

path_audios = [os.path.basename(elem).split('.wav')[0] for elem in os.listdir(base)]
existent_audio = [os.path.basename(elem).split('.txt')[0] for elem in os.listdir(output_folder)]

non_existent = list(set(existent_audio)^set(path_audios))
missing_audios = [os.path.join(path_audios, elem) for elem in non_existent]

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
