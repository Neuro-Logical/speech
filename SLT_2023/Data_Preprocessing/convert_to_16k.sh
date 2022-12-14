in_dir=/export/c12/afavaro/NLS_Data_Set
out_dir=/export/c12/afavaro/NLS_Data_Set_16k

for audio_File in $in_dir/*.wav
do
	uttID=$(basename "$audio_File")
	#echo $out_dir$uttID
	out_path=$out_dir$uttID
	echo $out_path
	#out_path="$(echo $audio_File | sed 's|audio_INV_PAR|audio_preprocessed_16kHz_INV_PAR|g' | sed 's|audio_INV_PAR|audio_preprocessed_16kHz_INV_PAR|g')" 
	#echo $out_path
	sox $audio_File -b 16 -r 16k $out_path remix -
done
