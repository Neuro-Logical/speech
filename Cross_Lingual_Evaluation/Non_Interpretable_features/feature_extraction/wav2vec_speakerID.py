audio_dir = '/export/b15/afavaro/Frontiers/Italian_PD/Audio_Whole_Ita_16/'
wav2vec_base = "/export/b11/ytsai25/feats/hubert/Italian/"

import os
import torchaudio
import torch
import numpy as np
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

tot = [os.path.join(audio_dir, elem) for elem in os.listdir(audio_dir)]
model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-sid")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-sid")

for audio in tot:
    base = os.path.basename(audio).split(".wav")[0]
    array, fs = torchaudio.load(audio)
    inputs = feature_extractor(array.squeeze(), sampling_rate=16000, padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # choose specific layer
    last_hidden_states = outputs.hidden_states[-2].squeeze().mean(axis=0).numpy()
    output_file = os.path.join(wav2vec_base, base + ".npy")
    print(output_file)
    with open(output_file, 'wb') as f:
        np.save(f, last_hidden_states)

