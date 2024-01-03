OUT_PATH = '/scratch4/lmorove1/afavaro/nls_experiments/Action/out_diarization_all/'

import json
import os
import whisperx

YOUR_HF_TOKEN = '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'
device = "cuda"
batch_size = 16  # reduce if low on GPU mem
compute_type = "float16"
model = whisperx.load_model("medium.en", device, compute_type=compute_type)

one = '/scratch4/lmorove1/afavaro/nls_experiments/Action/audio/'
files = [os.path.join(one, elem) for elem in os.listdir(one)]

files_new = []
for m in files:
    size = os.stat(m).st_size / 1000
    if size > 56:
        if os.path.exists(m) == True:
            files_new.append(m)

path = os.path.join(one, 'NLS_047_ses01_Namingaction1.wav')
ind = files_new.index(path)

for audio_file in files_new[ind+1:]:
    #all_sents_list = sorted([os.path.join(transcripts_path, elem) for elem in os.listdir(transcripts_path)])
    # path = os.path.join(transcripts_path, 'pd113_ses01_CookieThief.txt')
    # ind = all_sents_list.index(path)
    base_name = os.path.basename(audio_file).split(".wav")[0]
    print(base_name)

    csv_path = os.path.join(OUT_PATH, base_name + ".csv")
    print(csv_path)
    json_path = os.path.join(OUT_PATH, base_name + ".json")
    print(json_path)

    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)
    # print(result["segments"]) # before alignment

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code='en', device=device)
    #model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)


    # 3. Assign speaker labels
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=YOUR_HF_TOKEN, device=device)

    # add min/max number of speakers if known
    diarize_segments = diarize_model(audio_file)
    result = whisperx.assign_word_speakers(diarize_segments, result)
    ##print(diarize_segments)
    # print(result["segments"]) #
    diarize_segments.to_csv(csv_path)
    with open(json_path, "w") as outfile:
        json.dump(result, outfile)