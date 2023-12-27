OUT_PATH = '/data/lmorove1/afavaro/data/others/redo_all/alignment/'

import sys
sys.path.append("/export/c07/afavaro/whisperX")
import whisperx
import os
import json

device = "cuda"
batch_size = 16  # reduce if low on GPU mem
compute_type = "float16"

one = '/data/lmorove1/afavaro/data/others/redo_all/redo_all/'
files = [os.path.join(one, elem) for elem in os.listdir(one)]

files_new = []
for m in files:
    size = os.stat(m).st_size / 1000
    if size > 56:
        if os.path.exists(m) == True:
            files_new.append(m)
#print(len(files_new))

for audio_file in files_new:
    base_name = os.path.basename(audio_file).split(".wav")[0]

    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)

    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)
    print(result["segments"])  # before alignment

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code='en', device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    json_path = os.path.join(OUT_PATH, base_name + ".json")
    #result["segments"].to_csv(os.path.join(OUT_PATH, base_name + ".csv"))
    with open(json_path, "w") as outfile:
        json.dump(result, outfile)