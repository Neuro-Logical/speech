
input_dir = '/export/b01/afavaro/data_exp/'

import shutil
shutil._USE_CP_SENDFILE = False
import os
import librosa
import pandas as pd
import mutagen
from mutagen.wave import WAVE


def get_duration_librosa(file_path):
   audio_data, sample_rate = librosa.load(file_path)
   duration = librosa.get_duration(y=audio_data, sr=sample_rate)
   return duration

files = [os.path.join(input_dir, elem) for elem in os.listdir(input_dir)]

for file in files:
   print(file, get_duration_librosa(file))