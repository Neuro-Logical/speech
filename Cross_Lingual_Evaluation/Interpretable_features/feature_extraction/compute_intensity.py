#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import os
import parselmouth
from feature_extraction_utils import *


name = 'poem'
gita = '/export/c12/afavaro/New_NLS/audio_fusion/audio_intensity_normalized/poem/'
files= [os.path.join(gita, elem) for elem in sorted(os.listdir(gita))]

out_path = '/export/c12/afavaro/New_NLS/audio_fusion/features/'
out_file = os.path.join(out_path, name + ".csv")
print(out_file)

def compute_intensity_attribute(files):

    df_tot = []

    #for sound in zip(files, file_transcr):
    for sound in files:
        print(sound)

        df = pd.DataFrame()
        attributes = {}
        sound_filepath = os.path.basename(sound)
        #print(sound_filepath)

        sound_file = parselmouth.Sound(sound)

        intensity_attributes = get_intensity_attributes(sound_file)[0]
        #speak_rate = get_speaking_rate(sound_file, sound[1])
        pitch_attributes = get_pitch_attributes(sound_file)[0]
        attributes.update(intensity_attributes)
        attributes.update(pitch_attributes)
        #attributes.update({'speaking_rate': speak_rate})

        hnr_attributes = get_harmonics_to_noise_ratio_attributes(sound_file)[0]
        attributes.update(hnr_attributes)

        for attribute in attributes:
            df.at[0, attribute] = attributes[attribute]

        df.at[0, 'sound_filepath' ] = sound_filepath
        rearranged_columns = df.columns.tolist()[-1:] + df.columns.tolist()[:-1]
        df = df[rearranged_columns]
      #  print(df)
        df_tot.append(df)

    new_df = pd.concat(df_tot)
    return new_df

new_df = compute_intensity_attribute(files)

new_df.to_csv(out_file)
