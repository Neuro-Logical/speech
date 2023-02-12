# GITA

#hubert_base = '/export/b11/ytsai25/feats/hubert/GITA/'
#audio_dir = '/export/b15/afavaro/Frontiers/GITA_NEW_TASKS/All_Recordings_Correct_Naming/'

# Neurovoz
#hubert_base = "/export/b11/ytsai25/feats/hubert/Neurovoz"
#audio_dir = '/export/b15/afavaro/Frontiers/Neurovoz_data/audio_used_frontiers/'

# GermanPD
#hubert_base = "/export/b11/ytsai25/feats/hubert/German_PD/"
#audio_dir = '/export/b15/afavaro/Frontiers/German_PD/All/'

#hubert_base = "/export/b11/ytsai25/feats/hubert/Czech_PD/"
#audio_dir = '/export/b15/afavaro/Frontiers/Czech_PD/All_16k/'


#audio_dir =  '/export/b15/afavaro/Frontiers/NLS/NLS_RESAMPLED/'
#hubert_base = "/export/b11/ytsai25/feats/hubert/NLS/"

audio_dir = '/export/b15/afavaro/Frontiers/Italian_PD/Audio_Whole_Ita_16/'
hubert_base = "/export/b11/ytsai25/feats/hubert/Italian/"

import os
import numpy as np
import torchaudio

tot = [os.path.join(audio_dir, elem) for elem in os.listdir(audio_dir)]
ind = tot.index(os.path.join(audio_dir, "PD_Michele_C_FB1MCIICLHL46M240120171837.wav"))
bundle = torchaudio.pipelines.HUBERT_BASE
model = bundle.get_model()

for audio in tot[ind:]:
    base = os.path.basename(audio).split(".wav")[0]
    waveform, sample_rate = torchaudio.load(audio)
    waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
    features, _ = model.extract_features(waveform)

    #save feat from last layer
    numpy_array = features[-1].cpu().detach().numpy()
    output_file = os.path.join(hubert_base, base + ".npy")
    print(output_file)
    with open(output_file, 'wb') as f:
        np.save(f, numpy_array)

