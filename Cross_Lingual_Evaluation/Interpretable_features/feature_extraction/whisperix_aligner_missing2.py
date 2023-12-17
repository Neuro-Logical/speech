OUT_PATH = '/data/lmorove1/afavaro/data/others/trevor_alignment/'


import sys
sys.path.append("/export/c07/afavaro/whisperX")
import whisperx
import os
import pandas as pd
device = "cuda"
batch_size = 16  # reduce if low on GPU mem
compute_type = "float16"

one = '/data/lmorove1/afavaro/data/others/trevor_audios/'
files = [os.path.join(one, elem) for elem in os.listdir(one)]

files_new = []
for m in files:
    size = os.stat(m).st_size / 1000
    if size > 70:
        if os.path.exists(m) == True:
            files_new.append(m)
#print(len(files_new))

for audio_file in files_new:

    import gc
    batch_size = 16  # reduce if low on GPU mem

    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model("medium", device, compute_type=compute_type)

    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)
    print(result["segments"])  # before alignment

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    print(result["segments"])  # after alignment




   # data = pd.DataFrame({'word': text, 'time_stamps': time_stamps})
   # data.to_csv(os.path.join(OUT_PATH, base_name + ".csv"))