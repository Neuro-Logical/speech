OUT_PATH = '/export/b01/afavaro/tmeyer_new_output/tmeyer_alignment_all2/'


import sys
sys.path.append("/export/c07/afavaro/whisperX")
import whisperx
import os
import pandas as pd
device = "cpu"
model = whisperx.load_model("small", device)

files = ['/export/b01/afavaro/tmeyer_new_output/tmeyer_ressampled_norm_audios/PEC_063_ses01_SecuenceStroopPrevious1.wav'
'/export/b01/afavaro/tmeyer_new_output/tmeyer_ressampled_norm_audios/NLS_058_ses01_SecuenceStroopPrevious2.wav'
'/export/b01/afavaro/tmeyer_new_output/tmeyer_ressampled_norm_audios/NLS_167_ses01_SecuenceStroopPrevious1.wav'
'/export/b01/afavaro/tmeyer_new_output/tmeyer_ressampled_norm_audios/NLS_101_ses01_SecuenceStroopPrevious1.wav'
'/export/b01/afavaro/tmeyer_new_output/tmeyer_ressampled_norm_audios/AD_020_ses02_SecuenceStroopPrevious2.wav'
'/export/b01/afavaro/tmeyer_new_output/tmeyer_ressampled_norm_audios/NLS_033_ses01_SecuenceStroopPrevious2.wav'
'/export/b01/afavaro/tmeyer_new_output/tmeyer_ressampled_norm_audios/NLS_113_ses01_SecuenceStroopPrevious1.wav'
'/export/b01/afavaro/tmeyer_new_output/tmeyer_ressampled_norm_audios/NLS_062_ses01_SecuenceStroopPrevious1.wav'
'/export/b01/afavaro/tmeyer_new_output/tmeyer_ressampled_norm_audios/NLS_169_ses01_Wordcolor.wav'
'/export/b01/afavaro/tmeyer_new_output/tmeyer_ressampled_norm_audios/NLS_108_ses01_SecuenceStroopPrevious2.wav'
'/export/b01/afavaro/tmeyer_new_output/tmeyer_ressampled_norm_audios/NLS_094_ses01_Wordcolor.wav'
'/export/b01/afavaro/tmeyer_new_output/tmeyer_ressampled_norm_audios/NLS_094_ses01_SecuenceStroopPrevious2.wav'
'/export/b01/afavaro/tmeyer_new_output/tmeyer_ressampled_norm_audios/NLS_022_ses01_SecuenceStroopPrevious1.wav'
'/export/b01/afavaro/tmeyer_new_output/tmeyer_ressampled_norm_audios/NLS_007_ses02_SecuenceStroopPrevious1.wav'
'/export/b01/afavaro/tmeyer_new_output/tmeyer_ressampled_norm_audios/PEC_011_ses01_SecuenceStroopPrevious1.wav'
'/export/b01/afavaro/tmeyer_new_output/tmeyer_ressampled_norm_audios/AD_026_ses01_SecuenceStroopPrevious2.wav'
'/export/b01/afavaro/tmeyer_new_output/tmeyer_ressampled_norm_audios/AD_024_ses01_SecuenceStroopPrevious1.wav'
'/export/b01/afavaro/tmeyer_new_output/tmeyer_ressampled_norm_audios/NLS_083_ses01_SecuenceStroopPrevious1.wav'
'/export/b01/afavaro/tmeyer_new_output/tmeyer_ressampled_norm_audios/NLS_081_ses01_SecuenceStroopPrevious1.wav'
'/export/b01/afavaro/tmeyer_new_output/tmeyer_ressampled_norm_audios/NLS_054_ses01_SecuenceStroopPrevious1.wav'
'/export/b01/afavaro/tmeyer_new_output/tmeyer_ressampled_norm_audios/NLS_121_ses02_SecuenceStroopPrevious1.wav'
'/export/b01/afavaro/tmeyer_new_output/tmeyer_ressampled_norm_audios/NLS_035_ses01_SecuenceStroopPrevious2.wav'
'/export/b01/afavaro/tmeyer_new_output/tmeyer_ressampled_norm_audios/PEC_041_ses01_SecuenceStroopPrevious2.wav'
'/export/b01/afavaro/tmeyer_new_output/tmeyer_ressampled_norm_audios/NLS_126_ses01_SecuenceStroopPrevious1.wav'
'/export/b01/afavaro/tmeyer_new_output/tmeyer_ressampled_norm_audios/NLS_100_ses01_SecuenceStroopPrevious1.wav'
'/export/b01/afavaro/tmeyer_new_output/tmeyer_ressampled_norm_audios/NLS_094_ses01_SecuenceStroopPrevious1.wav'
'/export/b01/afavaro/tmeyer_new_output/tmeyer_ressampled_norm_audios/NLS_080_ses02_SecuenceStroopPrevious1.wav'
'/export/b01/afavaro/tmeyer_new_output/tmeyer_ressampled_norm_audios/NLS_079_ses01_SecuenceStroopPrevious1.wav'
'/export/b01/afavaro/tmeyer_new_output/tmeyer_ressampled_norm_audios/NLS_035_ses01_SecuenceStroopPrevious1.wav'
'/export/b01/afavaro/tmeyer_new_output/tmeyer_ressampled_norm_audios/NLS_031_ses01_SecuenceStroopPrevious2.wav']




#one = '/export/b01/afavaro/tmeyer_new_output/tmeyer_ressampled_norm_audios/'
#files = [os.path.join(one, elem) for elem in os.listdir(one)]
#tots = [elem.split(".wav")[0] for elem in os.listdir(one)]
#two = '/export/c12/afavaro/New_NLS/NLS_Speech_Data_Word_Alignment_whisperx'
##tots_1 = [elem.split(".csv")[0] for elem in os.listdir(two)]

#tot_wc = []
#for m in tots:
#    if "SecuencestroopPrevious" in m:
#        tot_wc.append(m)
#    if "WordColor" in m:
#        tot_wc.append(m)
#
#files = (set(tots) ^ set(tot_wc))
#files = [os.path.join(one, file + ".wav") for file in files]
#

files_new = []
for m in files:
    size = os.stat(m).st_size / 1000
    if size > 56:
        if os.path.exists(m) == True:
            files_new.append(m)
print(len(files_new))

for audio in files_new:
    print(audio)
    text =[]
    time_stamps = []
    base_name = os.path.basename(audio).split(".wav")[0]
    print(base_name)
    result = model.transcribe(audio)
    model_a, metadata = whisperx.load_align_model(language_code='en', device=device)
    result_aligned = whisperx.align(result["segments"], model_a, metadata, audio, device)
    out = (result_aligned["word_segments"])
    for element in out:
        text.append(element['text'])
        time_stamps.append(element['start'])
    data = pd.DataFrame({'word': text, 'time_stamps': time_stamps})
    data.to_csv(os.path.join(OUT_PATH, base_name + ".csv"))