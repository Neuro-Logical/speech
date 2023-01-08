import os
import whisper


base = '/export/c12/afavaro/New_NLS/NLS_Speech_Data_All_16k'
output_folder = '/export/c12/afavaro/New_NLS/NLS_Speech_Data_Transcripts/'

paths = [os.path.join(base, elem) for elem in os.listdir(base)]

# keep only non-empty recordings (> 56 Bytes)
files = []
for m in paths:
    size = os.stat(m).st_size / 1000
    if size > 56:
        files.append(m)

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
