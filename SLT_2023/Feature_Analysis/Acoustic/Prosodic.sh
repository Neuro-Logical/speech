KALDI_ROOT=/home/afavaro/kaldi
export PATH=$PATH:$KALDI_ROOT/src/featbin/



# prosodic features extraction using Disvoice: https://github.com/jcvasquezc/DisVoice/tree/master/disvoice/prosody


python prosody.py "/export/c12/afavaro/NLS_Data_Set_16k" "/export/b14/afavaro/Acoustic_Features/prosody.csv" "true" "false" "csv"

#python prosody.py "/export/b15/afavaro/Frontiers/ADR_2021/audio_16k/" "/export/b15/afavaro/Frontiers/ADR_2021/prosody.csv" "true" "false" "csv"




# Pause-related features extraction using DigiPsych Prosody: https://github.com/NeuroLexDiagnostics/DigiPsych_Prosody



python featurize.py -a  /export/c12/afavaro/NLS_Data_Set_16k -f 20

python featurize.py -a  /export/b15/afavaro/Frontiers/ADR_2021/audio_16k/ -f 20