wav2vec = '/export/b11/ytsai25/feats/wav2vec/GITA/'
audio_dir = '/export/b15/afavaro/Frontiers/GITA_NEW_TASKS/All_Recordings_Correct_Naming/'



import os
import numpy as np
import torchaudio

tot = [os.path.join(audio_dir, elem) for elem in os.listdir(audio_dir)]
ind = tot.index("/export/b15/afavaro/Frontiers/GITA_NEW_TASKS/All_Recordings_Correct_Naming/PD_AVPEPUDEA0013_TDU.wav")
bundle = torchaudio.pipelines.WAV2VEC2_XLSR53
model = bundle.get_model()

for audio in tot[ind:]:
    base = os.path.basename(audio).split(".wav")[0]
    waveform, sample_rate = torchaudio.load(audio)
    waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
    features, _ = model.extract_features(waveform)
    numpy_array = features[-1].cpu().detach().numpy()
    output_file = os.path.join(wav2vec, base + ".npy")
    print(output_file)
    with open(output_file, 'wb') as f:
        np.save(f, numpy_array)

