KALDI_ROOT=/home/afavaro/kaldi
export PATH=$PATH:$KALDI_ROOT/src/featbin/

# prosodic features extraction using Disvoice: https://github.com/jcvasquezc/DisVoice/tree/master/disvoice/prosody
# The script saves a feature matrix in csv format.
# This script should be run inside the folder named: disvoice/prosody.
# This script is called as follows:
## python prosody.py <file_or_folder_audio> <file_features> <static (true or false)> <plots (true or false)> <format (csv, txt, npy, kaldi, torch)>

python prosody.py "/export/c12/afavaro/NLS_Data_Set_16k" "/export/b14/afavaro/Acoustic_Features/prosody.csv" "true" "false" "csv"
python prosody.py "/export/b15/afavaro/Frontiers/ADR_2021/audio_16k/" "/export/b15/afavaro/Frontiers/ADR_2021/prosody.csv" "true" "false" "csv"

##############################################################################################################################################

# Pause related features extracted with: https://github.com/NeuroLexDiagnostics/DigiPsych_Prosody
# This script should be run inside the folder named: DigiPsych_Prosody.
# This script is called as follows:
## python featurize.py -a <file_or_folder_audio> -f f <frame length (10, 20, 30 ms}>

python featurize.py -a  /export/c12/afavaro/NLS_Data_Set_16k -f 20
python featurize.py -a  /export/b15/afavaro/Frontiers/ADR_2021/audio_16k/ -f 20