import os
import pandas as pd
import whisper
pd.options.display.max_rows = 100

repo = '/export/c12/afavaro/New_NLS/NLS_Speech_Data_All_16k/'
output_folder = '/export/c12/afavaro/New_NLS/NLS_Speech_Data_Transcripts_2'
paths = [os.path.join(repo, base) for base in os.listdir(repo)]

files = []
for m in paths:
    size = os.stat(m).st_size/1000
    if size > 56:
        files.append(m)

for i in files:
    #print(i)
    model = whisper.load_model("medium")
    result = model.transcribe(i)
    test = result['text']
    base = os.path.basename(i).split(".wav")[0]
    total = os.path.join(output_folder, base +  ".txt")
    text_file = open(total, "wt")
    n = text_file.write(test)
    text_file.close()
                                                                    
    

