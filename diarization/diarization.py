import wave
import os
import pandas as pd
from pydub import AudioSegment

root2 = '/export/c12/afavaro/New_NLS/NLS_Speech_Data/'
total_files = []
for path, subdir, files in os.walk(root2):
    print(subdir)
    for name in files:
        if name.endswith(".csv"):
            if "general" not in name:
                if "speaker" not in name and "gecko" not in name:
                    total_files.append(os.path.join(path, name))

for file in total_files:
    print(file)
    cd = os.path.split(file)[0]
    base = cd.split("diarization")[0]
    base_name = os.path.basename(file).split(".csv")[0]
    df = pd.read_csv(file)
    audio_ = os.path.join(base, 'speech', base_name + ".wav")
    read_audio = AudioSegment.from_wav(audio_)
    sp = 'SPEAKER_00'

    path_s = os.path.join(base, "diarized_speech")
    if os.path.isdir(path_s) == False:
        print("--> creating a directory")
        os.mkdir(path_s)
    path_sp = os.path.join(base, "diarized_speech", base_name)
    print(path_sp)
    if os.path.isdir(path_sp) == False:
        print("--> creating a directory")
        os.mkdir(path_sp)
        sp_val = df.loc[df['speaker'] == sp]
        start = sp_val['start'].tolist()
        starting_time_stamps = sp_val['start'].tolist()
        ending_time_stamps = sp_val['end'].tolist()
        list_of_timestamps = list(zip(starting_time_stamps, ending_time_stamps))
        for idx, t in enumerate(list_of_timestamps):
            # break loop if at last element of list
            if idx == len(list_of_timestamps):
                break
            start = t[0] * 1000
            end = t[1] * 1000
            print("split at [ {}:{}] ms".format(start, end))
            audio_chunk = read_audio[start:end]
            # audio_chunks.append(audio_chunk)
            exp = os.path.join(path_sp, str(end) + ".wav")
            audio_chunk.export(exp, format="wav")
        tot_chuncks = os.listdir(path_sp)
        list_ordered = [float(x.split(".wav")[0]) for x in tot_chuncks]
        list_ordered.sort()
        wav_ordered = [os.path.join(str(x) + ".wav") for x in list_ordered]
        print(wav_ordered)
        files = []
        for elem in wav_ordered:
            files.append(os.path.join(path_sp, elem))
        outfile = os.path.join(path_sp, base_name + ".wav")
        data = []
        for infile in files:
            w = wave.open(infile, 'rb')
            data.append([w.getparams(), w.readframes(w.getnframes())])
            w.close()
        output = wave.open(outfile, 'wb')
        output.setparams(data[0][0])
        for i in range(len(data)):
            output.writeframes(data[i][1])
        output.close()



