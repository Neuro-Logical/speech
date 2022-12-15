KALDI_ROOT=/home/afavaro/kaldi
export PATH=$PATH:$KALDI_ROOT/src/featbin/


# prosodic features extraction using Disvoice: https://github.com/jcvasquezc/DisVoice/tree/master/disvoice/prosody
# The script saves a feature matrix in csv format.

python prosody.py "/export/c12/afavaro/NLS_Data_Set_16k" "/export/b14/afavaro/Acoustic_Features/prosody.csv" "true" "false" "csv"

python prosody.py "/export/b15/afavaro/Frontiers/ADR_2021/audio_16k/" "/export/b15/afavaro/Frontiers/ADR_2021/prosody.csv" "true" "false" "csv"


# Pause related features extracted with: https://github.com/NeuroLexDiagnostics/DigiPsych_Prosody

# f = frame length, namely the length of the frame used by the VAD adopted for the feature extraction.
# f can be 10, 20, 30 ms.
# -a is the path to the folder containing the recordings.

python featurize.py -a  /export/c12/afavaro/NLS_Data_Set_16k -f 20

python featurize.py -a  /export/b15/afavaro/Frontiers/ADR_2021/audio_16k/ -f 20