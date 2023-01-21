#out_dir=/export/b15/afavaro/Frontiers/GITA_NEW_TASKS/L_Normalized/
#in_dir=/export/b15/afavaro/Frontiers/Neurovoz_data/audio_used_frontiers/
#in_dir=/export/b15/afavaro/Frontiers/NLS/NLS_RESAMPLED/
#in_dir=/export/b15/afavaro/Frontiers/Czech_PD/All_16k/
#in_dir=/export/b15/afavaro/Frontiers/German_PD/All/
#in_dir=/export/b15/afavaro/Frontiers/Italian_PD/Audio_Whole_Ita_16/
#in_dir=/export/b16/afavaro/AD_speech/
#out_dir=/export/b16/afavaro/AD_speech_16/
#out_dir=/export/b15/afavaro/Frontiers/Italian_PD/L_Normalized/
#out_dir=/export/b15/afavaro/Frontiers/German_PD/L_Normalized/
#out_dir=/export/b15/afavaro/Frontiers/Czech_PD/L_Normalized/
#out_dir=/export/b15/afavaro/Frontiers/NLS/L_Normalized/
#in_dir=/export/b15/afavaro/Frontiers/GITA_NEW_TASKS/All_Recordings_Correct_Naming/
in_dir=/export/b15/afavaro/Frontiers/GITA_PD/Gita_monologue/
out_dir=/export/b15/afavaro/Frontiers/GITA_PD/L_norm_mon/


for audio_File in $in_dir/*.wav
do
	uttID=$(basename "$audio_File")
       	out_path=$out_dir$uttID
       	echo $out_path
	ffmpeg-normalize $audio_File -o $out_path 
done	
