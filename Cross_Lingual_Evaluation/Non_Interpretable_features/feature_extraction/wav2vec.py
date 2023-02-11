#wav2vec = '/export/b11/ytsai25/feats/wav2vec/GITA/'
#audio_dir = '/export/b15/afavaro/Frontiers/GITA_NEW_TASKS/All_Recordings_Correct_Naming/'

#wav2vec = '/export/b11/ytsai25/feats/wav2vec/Neurovoz/'
#audio_dir = '/export/b15/afavaro/Frontiers/Neurovoz_data/audio_used_frontiers/'

# GermanPD
#wav2vec = "/export/b11/ytsai25/feats/wav2vec/German_PD/"
#audio_dir = '/export/b15/afavaro/Frontiers/German_PD/All/'

#wav2vec = "/export/b11/ytsai25/feats/wav2vec/Czech_PD/"
#audio_dir = '/export/b15/afavaro/Frontiers/Czech_PD/All_16k/'

audio_dir =  '/export/b15/afavaro/Frontiers/NLS/RP_conc_resampled/'
wav2vec = "/export/b11/ytsai25/feats/wav2vec/NLS/"

import os
import numpy as np
import torchaudio

tot = [os.path.join(audio_dir, elem) for elem in os.listdir(audio_dir)]
#ind = tot.index("/export/b15/afavaro/Frontiers/German_PD/All/PD_043_monologue_German.wav")
bundle = torchaudio.pipelines.WAV2VEC2_XLSR53
model = bundle.get_model()

for audio in tot:
    base = os.path.basename(audio).split(".wav")[0]
    waveform, sample_rate = torchaudio.load(audio)
    waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
    features, _ = model.extract_features(waveform)
    numpy_array = features[-1].cpu().detach().numpy()
    output_file = os.path.join(wav2vec, base + ".npy")
    print(output_file)
    with open(output_file, 'wb') as f:
        np.save(f, numpy_array)

