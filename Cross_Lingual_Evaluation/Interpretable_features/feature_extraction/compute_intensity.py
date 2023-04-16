#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import os
import parselmouth
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from feature_extraction_utils import *



gita = "/export/b15/afavaro/Trevor_paper/speech_16_norm_2/"
files= [os.path.join(gita, elem) for elem in sorted(os.listdir(gita))]


for sound in zip(files, file_transcr):
    print(sound)
    


# In[20]:


gita = '/export/b15/afavaro/Frontiers/NLS/L_Normalized/'
#gita = '/export/b15/afavaro/Frontiers/NLS/RPs_concatenated/'
#f = [elem.split(".wav")[0] for elem in sorted(os.listdir(gita))]
files= [os.path.join(gita, elem) for elem in sorted(os.listdir(gita))]


# In[3]:


gita


# In[7]:


## Monologue

gita = '/export/b15/afavaro/Frontiers/Italian_PD/RPs_concatenated/'
files= [os.path.join(gita, elem) for elem in sorted(os.listdir(gita))]

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



new_df.to_csv("/export/b15/afavaro/Trevor_paper/Results/intensity_2.csv")
