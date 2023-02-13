audio_dir = '/export/b15/afavaro/Frontiers/Italian_PD/Audio_Whole_Ita_16/'
hubert_base = "/export/b11/ytsai25/feats/hubert/new/"

import os
import torchaudio
import numpy as np
import torch
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor

tot = [os.path.join(audio_dir, elem) for elem in os.listdir(audio_dir)]
model = HubertForSequenceClassification.from_pretrained("superb/hubert-large-superb-sid")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-large-superb-sid")

for audio in tot:
    print(audio)
    base = os.path.basename(audio).split(".wav")[0]
    array, fs = torchaudio.load(audio)
    inputs = feature_extractor(array.squeeze(), sampling_rate=16000, padding=True, return_tensors="pt")
    print("ok")
    with torch.no_grad():
        outputs = model(**inputs)

    # choose specific layer
    last_hidden_states = outputs.hidden_states[-2].squeeze().mean(axis=0).numpy()
    output_file = os.path.join(hubert_base, base + ".npy")
    print(output_file)
    with open(output_file, 'wb') as f:
        np.save(f, last_hidden_states)

