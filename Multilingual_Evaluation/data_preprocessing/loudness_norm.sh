in_dir=/export/b15/afavaro/Frontiers/GITA_PD/Gita_monologue/
out_dir=/export/b15/afavaro/Frontiers/GITA_PD/L_norm_mon/

#This script normalizes audio files to a certain loudness level using the EBU R128 loudness normalization procedure.
# This scfipt is using ffmpeg-normalize 1.26.1
for audio_File in $in_dir/*.wav
do
	uttID=$(basename "$audio_File")
       	out_path=$out_dir$uttID
       	echo $out_path
	ffmpeg-normalize $audio_File -o $out_path 
done	
