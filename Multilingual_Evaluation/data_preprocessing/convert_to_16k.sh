# Resample directory containing audio files to 16 kHZ using sox.
# in_dir: directory containing recordings to be converted
# out_dir: output directory containing recordings converted to 16 kHz.

in_dir=/export/c12/afavaro/NLS_Data_Set
out_dir=/export/b15/afavaro/Frontiers/NLS/NLS_RESAMPLED/

for audio_File in $in_dir/*.wav
do
	uttID=$(basename "$audio_File")
	out_path=$out_dir$uttID
	echo $out_path
	sox $audio_File -b 16 -r 16k $out_path remix -
done
